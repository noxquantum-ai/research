"""Exact linear Value/readout construction with a fixed causal attention map.

Q and K stay fixed. Both V -> O A(Q,K) V X and O -> O context are linear,
so their Schmidt cores can be constructed by normal-equation solves without
softmax derivatives or gradient-based model healing.
"""

from __future__ import annotations

import math

import torch

from nox.block_compiler import apply_llama2_rope, compute_llama2_rotary_emb
from nox.functional_schmidt import SchmidtCoreMap, _prior_anchored_pcg
from nox.packed_schmidt import (
    packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)


class FixedCausalAttention:
    """Causal probabilities with an exact linear forward and transpose map."""

    @torch.no_grad()
    def __init__(self, queries, keys, n_heads, n_kv_heads, sequence_length):
        width, tokens = queries.shape
        if (tokens % sequence_length or width % n_heads or n_heads % n_kv_heads
                or keys.shape != (n_kv_heads * (width // n_heads), tokens)):
            raise ValueError("invalid whole-window MHA/GQA dimensions")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim, self.sequence_length = width // n_heads, sequence_length
        self.windows = tokens // sequence_length
        q = self._heads(queries, n_heads)
        k = self._heads(keys, n_kv_heads)
        cos, sin = compute_llama2_rotary_emb(
            self.head_dim, sequence_length, device=queries.device, dtype=queries.dtype,
        )
        q, k = apply_llama2_rope(q, k, cos, sin)
        k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)
        self.probabilities = torch.empty(
            self.windows, n_heads, sequence_length, sequence_length,
            device=queries.device, dtype=queries.dtype,
        )
        mask = torch.ones(sequence_length, sequence_length, device=queries.device,
                          dtype=torch.bool).triu(1)
        # One window at a time avoids a second full attention-score tensor.
        for index in range(self.windows):
            logits = (q[index] @ k[index].transpose(-1, -2)) / math.sqrt(self.head_dim)
            logits.masked_fill_(mask, -torch.inf)
            self.probabilities[index].copy_(logits.softmax(dim=-1))

    def _heads(self, values, heads):
        return values.reshape(heads, self.head_dim, self.windows, self.sequence_length).permute(2, 0, 3, 1)

    def _flat(self, heads):
        return heads.permute(1, 3, 0, 2).reshape(-1, self.windows * self.sequence_length)

    def window_slice(self, start, end):
        result = object.__new__(FixedCausalAttention)
        result.n_heads, result.n_kv_heads = self.n_heads, self.n_kv_heads
        result.head_dim, result.sequence_length = self.head_dim, self.sequence_length
        result.windows = end - start
        result.probabilities = self.probabilities[start:end]
        return result

    def forward(self, values):
        v = self._heads(values, self.n_kv_heads).repeat_interleave(
            self.n_heads // self.n_kv_heads, dim=1,
        )
        return self._flat(self.probabilities @ v)

    def adjoint(self, output):
        response = self.probabilities.transpose(-1, -2) @ self._heads(output, self.n_heads)
        response = response.reshape(
            self.windows, self.n_kv_heads, self.n_heads // self.n_kv_heads,
            self.sequence_length, self.head_dim,
        ).sum(dim=2)
        return self._flat(response)


@torch.no_grad()
def construct_functional_schmidt_attention(
    artifacts, inputs, targets, queries, keys, *, n_heads, n_kv_heads,
    sequence_length, sweeps=1, cg_iterations=8, ridge_ratio=0.05,
    guard_fraction=0.25,
):
    """Construct V/O cores jointly, preserving all stored shapes and budgets."""
    if not 0 < guard_fraction < 1 or min(sweeps, cg_iterations) < 1 or ridge_ratio <= 0:
        raise ValueError("invalid functional attention construction controls")
    windows = inputs.shape[1] // sequence_length
    if windows < 2 or inputs.shape[1] % sequence_length or targets.shape != inputs.shape:
        raise ValueError("attention construction needs aligned whole-window inputs and targets")
    guard_windows = max(1, math.floor(windows * guard_fraction))
    fit_windows = windows - guard_windows
    split = fit_windows * sequence_length
    operator = FixedCausalAttention(queries, keys, n_heads, n_kv_heads, sequence_length)
    fit_map = operator.window_slice(0, fit_windows)
    guard_map = operator.window_slice(fit_windows, windows)
    result = {role: dict(artifacts[role]) for role in ("v_proj", "o_proj")}
    weights = {role: reconstruct_packed_schmidt(value, device=inputs.device)
               for role, value in result.items()}
    fit_x, guard_x = inputs[:, :split].float(), inputs[:, split:].float()
    fit_y, guard_y = targets[:, :split].float(), targets[:, split:].float()

    def errors(candidate):
        return tuple(float((candidate["o_proj"] @ op.forward(candidate["v_proj"] @ x) - y)
                           .square().sum() / y.square().sum().clamp_min(1e-30))
                     for op, x, y in ((fit_map, fit_x, fit_y), (guard_map, guard_x, guard_y)))

    initial = errors(weights)
    current_fit, current_guard = initial
    records = []
    for sweep in range(sweeps):
        for role in ("o_proj", "v_proj", "o_proj"):
            sides = ("left", "right", "amplitudes") if result[role].get("codec_type") == "nec_sparse_excitation_schmidt_v1" else ("left", "right")
            for side in sides:
                mapper = SchmidtCoreMap(result[role], inputs.device)
                core = getattr(mapper, side) if side == "amplitudes" else (mapper.left if side == "left" else mapper.right)
                context = fit_map.forward(weights["v_proj"] @ fit_x)
                core_inputs = context if role == "o_proj" else fit_x
                scale = 1 / math.sqrt(split)

                def forward(increment):
                    response = mapper.weight(increment, side) @ core_inputs
                    if role == "v_proj":
                        response = weights["o_proj"] @ fit_map.forward(response)
                    return response * scale

                def adjoint(output):
                    if role == "v_proj":
                        output = fit_map.adjoint(weights["o_proj"].T @ output)
                    return mapper.adjoint((output @ core_inputs.T) * scale, side)

                sensitivity = (torch.ones(targets.shape[0], device=inputs.device)
                               if role == "o_proj" else weights["o_proj"].square().sum(dim=0))
                if role == "v_proj" and n_kv_heads != n_heads:
                    sensitivity = sensitivity.reshape(n_kv_heads, n_heads // n_kv_heads, -1).sum(1).flatten()
                diagonal = mapper.diagonal(core_inputs.square().mean(1), sensitivity, side)
                ridge = ridge_ratio * diagonal.mean().clamp_min(1e-20)
                rhs = adjoint((fit_y - weights["o_proj"] @ context) * scale)
                delta, history = _prior_anchored_pcg(
                    lambda value: adjoint(forward(value)) + ridge * value,
                    rhs, diagonal + ridge, iterations=cg_iterations, tolerance=1e-3,
                )
                best_fit, best_guard = current_fit, current_guard
                chosen, chosen_weight, fraction_chosen = None, None, None
                trials = []
                for fraction in (1., .5, .25):
                    candidate = mapper.packed_candidate(core + fraction * delta, side)
                    decoded = reconstruct_packed_schmidt(candidate, device=inputs.device)
                    fit_error, guard_error = errors({**weights, role: decoded})
                    trials.append({"fraction": fraction, "fit_relative_mse": fit_error,
                                   "guard_relative_mse": guard_error})
                    if fit_error < current_fit and guard_error < best_guard:
                        chosen, chosen_weight, fraction_chosen = candidate, decoded, fraction
                        best_fit, best_guard = fit_error, guard_error
                if chosen is not None:
                    for counter in (packed_schmidt_coefficient_count_from_artifact,
                                    packed_schmidt_payload_bytes_from_artifact):
                        if counter(chosen) != counter(result[role]):
                            raise RuntimeError("attention construction changed the representation budget")
                    result[role], weights[role] = chosen, chosen_weight
                    current_fit, current_guard = best_fit, best_guard
                records.append({"sweep": sweep, "role": role, "core": side,
                                "accepted_fraction": fraction_chosen, "trials": trials,
                                "cg_history": history})
    return result, weights, {
        "objective": "exact_fixed_attention_value_readout_core_construction",
        "uses_autograd": False, "uses_nonlinear_derivatives": False,
        "fit_tokens": split, "guard_tokens": inputs.shape[1] - split,
        "initial_fit_relative_mse": initial[0], "initial_guard_relative_mse": initial[1],
        "final_fit_relative_mse": current_fit, "final_guard_relative_mse": current_guard,
        "attention_probability_scratch_bytes": operator.probabilities.numel() * operator.probabilities.element_size(),
        "stored_coefficients_added": 0, "tensor_payload_bytes_added": 0, "records": records,
    }
