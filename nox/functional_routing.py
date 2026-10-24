"""Paired Query/Key construction by linear attention-score reconstruction.

With one partner fixed, pre-softmax scores are linear in the other's stored
coefficients. Teacher probabilities provide a fixed measurement metric, not
a differentiated softmax. Packed proposals are accepted only if actual
attention-distribution KL improves on construction and guard windows.
"""

from __future__ import annotations

import math

import torch

from nox.block_compiler import compute_llama2_rotary_emb, rotate_half
from nox.functional_schmidt import SchmidtCoreMap, _prior_anchored_pcg, curvature_anchor
from nox.packed_schmidt import (
    packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)


class PairedRoutingProbe:
    def __init__(self, inputs, teacher_queries, teacher_keys, *, n_heads, n_kv_heads,
                 sequence_length, queries_per_window):
        width, tokens = teacher_queries.shape
        if (tokens % sequence_length or width % n_heads or n_heads % n_kv_heads
                or teacher_keys.shape != (width // n_heads * n_kv_heads, tokens)):
            raise ValueError("invalid paired attention dimensions")
        self.windows, self.length = tokens // sequence_length, sequence_length
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.dim = width // n_heads
        self.positions = torch.linspace(0, sequence_length - 1,
                                        min(queries_per_window, sequence_length),
                                        device=inputs.device).round().long().unique()
        self.query_count = self.positions.numel()
        self.query_indices = (torch.arange(self.windows, device=inputs.device)[:, None]
                              * sequence_length + self.positions[None, :]).flatten()
        self.inputs = inputs.float()
        self.query_inputs = inputs[:, self.query_indices].float()
        self.cos, self.sin = compute_llama2_rotary_emb(
            self.dim, sequence_length, device=inputs.device, dtype=inputs.dtype,
        )
        self.cos = self.cos.reshape(sequence_length, self.dim)
        self.sin = self.sin.reshape(sequence_length, self.dim)
        self.causal_mask = torch.arange(sequence_length, device=inputs.device)[None, :] > self.positions[:, None]
        tq = self.query_heads(teacher_queries[:, self.query_indices])
        tk = self.key_heads(teacher_keys)
        self.teacher_scores = tq @ tk.transpose(-1, -2) / math.sqrt(self.dim)
        masked_scores = self.teacher_scores.masked_fill(self.causal_mask, -1e30)
        self.teacher_log_probs = masked_scores.log_softmax(dim=-1)
        self.probabilities = self.teacher_log_probs.exp()
        self.root_probability = self.probabilities.sqrt()

    def rotate(self, values, *, queries, inverse=False):
        cos, sin = self.cos, self.sin
        if queries:
            cos, sin = cos[self.positions], sin[self.positions]
        return values * cos[None, None] + (-1 if inverse else 1) * rotate_half(values) * sin[None, None]

    def query_heads(self, values):
        heads = values.reshape(self.n_heads, self.dim, self.windows, self.query_count).permute(2, 0, 3, 1)
        return self.rotate(heads, queries=True)

    def key_heads(self, values):
        heads = values.reshape(self.n_kv_heads, self.dim, self.windows, self.length).permute(2, 0, 3, 1)
        return self.rotate(heads, queries=False).repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

    def query_adjoint(self, values):
        return self.rotate(values, queries=True, inverse=True).permute(1, 3, 0, 2).reshape(-1, self.windows * self.query_count)

    def key_adjoint(self, values):
        grouped = values.reshape(self.windows, self.n_kv_heads, self.n_heads // self.n_kv_heads,
                                 self.length, self.dim).sum(dim=2)
        return self.rotate(grouped, queries=False, inverse=True).permute(1, 3, 0, 2).reshape(-1, self.windows * self.length)

    def scores(self, weights):
        q = self.query_heads(weights['q_proj'] @ self.query_inputs)
        k = self.key_heads(weights['k_proj'] @ self.inputs)
        return q @ k.transpose(-1, -2) / math.sqrt(self.dim)

    def measure(self, scores):
        centered = scores - (self.probabilities * scores).sum(dim=-1, keepdim=True)
        return self.root_probability * centered

    def measure_adjoint(self, values):
        weighted = self.root_probability * values
        return weighted - self.probabilities * weighted.sum(dim=-1, keepdim=True)

    def errors(self, weights, fit_windows):
        log_probs = self.scores(weights).masked_fill(self.causal_mask, -1e30).log_softmax(dim=-1)
        kl = (self.probabilities * (self.teacher_log_probs - log_probs)).sum(dim=-1)
        return float(kl[:fit_windows].mean()), float(kl[fit_windows:].mean())


@torch.no_grad()
def construct_functional_schmidt_routing(
    artifacts, inputs, teacher_queries, teacher_keys, *, n_heads, n_kv_heads,
    sequence_length, queries_per_window=64, sweeps=1, cg_iterations=8,
    ridge_ratio=.05, guard_fraction=.25,
    ridge_metric='mean',
):
    if min(queries_per_window, sweeps, cg_iterations) < 1 or ridge_ratio <= 0 or not 0 < guard_fraction < 1:
        raise ValueError("invalid routing construction controls")
    probe = PairedRoutingProbe(inputs, teacher_queries, teacher_keys, n_heads=n_heads,
                               n_kv_heads=n_kv_heads, sequence_length=sequence_length,
                               queries_per_window=queries_per_window)
    if probe.windows < 2:
        raise ValueError("paired routing construction requires at least two windows")
    fit_windows = probe.windows - max(1, math.floor(probe.windows * guard_fraction))
    fit_tokens, fit_queries = fit_windows * sequence_length, fit_windows * probe.query_count
    result = {role: dict(artifacts[role]) for role in ('q_proj', 'k_proj')}
    weights = {role: reconstruct_packed_schmidt(a, device=inputs.device) for role, a in result.items()}
    initial = probe.errors(weights, fit_windows)
    current_fit, current_guard = initial
    records = []
    scale = 1 / math.sqrt(fit_queries)
    for sweep in range(sweeps):
        for role in ('q_proj', 'k_proj'):
            sides = ('left', 'right', 'amplitudes') if result[role].get('codec_type') == 'nec_sparse_excitation_schmidt_v1' else ('left', 'right')
            for side in sides:
                mapper = SchmidtCoreMap(result[role], inputs.device)
                core = mapper.amplitudes if side == 'amplitudes' else getattr(mapper, side)
                q = probe.query_heads(weights['q_proj'] @ probe.query_inputs)
                k = probe.key_heads(weights['k_proj'] @ probe.inputs)
                core_inputs = probe.query_inputs if role == 'q_proj' else probe.inputs

                def forward(increment):
                    values = mapper.weight(increment, side) @ core_inputs
                    response = (probe.query_heads(values) @ k.transpose(-1, -2)
                                if role == 'q_proj' else q @ probe.key_heads(values).transpose(-1, -2))
                    return probe.measure(response / math.sqrt(probe.dim))[:fit_windows] * scale

                def adjoint(values):
                    all_values = torch.zeros_like(probe.teacher_scores)
                    all_values[:fit_windows] = values
                    scores = probe.measure_adjoint(all_values) / math.sqrt(probe.dim)
                    response = (probe.query_adjoint(scores @ k) if role == 'q_proj'
                                else probe.key_adjoint(scores.transpose(-1, -2) @ q))
                    return mapper.adjoint((response @ core_inputs.T) * scale, side)

                other = k if role == 'q_proj' else q
                sensitivity = other[:fit_windows].square().mean(dim=(0, 2)) / probe.dim
                if role == 'k_proj' and n_kv_heads != n_heads:
                    sensitivity = sensitivity.reshape(n_kv_heads, n_heads // n_kv_heads, probe.dim).sum(1)
                fit_count = fit_queries if role == 'q_proj' else fit_tokens
                diagonal = mapper.diagonal(core_inputs[:, :fit_count].square().mean(dim=1), sensitivity.flatten(), side)
                ridge = curvature_anchor(diagonal, ridge_ratio, ridge_metric)
                rhs = adjoint(probe.measure(probe.teacher_scores - probe.scores(weights))[:fit_windows] * scale)
                delta, history = _prior_anchored_pcg(
                    lambda values: adjoint(forward(values)) + ridge * values,
                    rhs, diagonal + ridge, iterations=cg_iterations, tolerance=1e-3,
                )
                best_fit, best_guard = current_fit, current_guard
                chosen, chosen_weight, chosen_fraction = None, None, None
                trials = []
                for fraction in (1., .5, .25):
                    candidate = mapper.packed_candidate(core + fraction * delta, side)
                    decoded = reconstruct_packed_schmidt(candidate, device=inputs.device)
                    fit_error, guard_error = probe.errors({**weights, role: decoded}, fit_windows)
                    trials.append({'fraction': fraction, 'fit_attention_kl': fit_error, 'guard_attention_kl': guard_error})
                    if fit_error < current_fit and guard_error < best_guard:
                        chosen, chosen_weight, chosen_fraction = candidate, decoded, fraction
                        best_fit, best_guard = fit_error, guard_error
                if chosen is not None:
                    for count in (packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact):
                        if count(chosen) != count(result[role]):
                            raise RuntimeError("routing construction changed the representation budget")
                    result[role], weights[role] = chosen, chosen_weight
                    current_fit, current_guard = best_fit, best_guard
                records.append({'sweep': sweep, 'role': role, 'core': side,
                                'accepted_fraction': chosen_fraction, 'trials': trials, 'cg_history': history})
    return result, weights, {
        'objective': 'paired_teacher_probability_weighted_centered_logit_reconstruction',
        'ridge_metric': ridge_metric,
        'uses_autograd': False, 'uses_nonlinear_derivatives': False,
        'queries_per_window': probe.query_count, 'fit_windows': fit_windows,
        'guard_windows': probe.windows - fit_windows,
        'initial_fit_attention_kl': initial[0], 'initial_guard_attention_kl': initial[1],
        'final_fit_attention_kl': current_fit, 'final_guard_attention_kl': current_guard,
        'stored_coefficients_added': 0, 'tensor_payload_bytes_added': 0, 'records': records,
    }
