"""Construct packed Schmidt MLP cores against the coupled SwiGLU function.

For one fixed MPO core, Up and Down give linear least-squares systems.
Gate candidates solve linear reconstruction of clean teacher preactivations.
Matrix-free, prior-anchored conjugate gradients solve those systems without
autograd. Candidate cores are packed before scoring the exact nonlinear
function on separate construction and guard samples. No coefficients, ranks,
or inference branches are added.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nox.operator_schmidt import _layout_from_value, _matrix_to_unfolding, _unfolding_to_matrix
from nox.packed_schmidt import (
    packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact,
    reconstruct_packed_schmidt,
)
from nox.residual_codec import decode_groupwise, encode_groupwise


class SchmidtCoreMap:
    """Exact linear core-to-weight map and its Frobenius adjoint."""

    def __init__(self, artifact: dict, device):
        self.artifact = artifact
        self.sector = artifact.get("codec_type") == "nec_packed_sector_schmidt_v1"
        self.base = artifact["base_artifact"] if self.sector else artifact
        self.bulk_indices = (
            artifact["bulk_indices"].to(device=device, dtype=torch.long) if self.sector else None
        )
        self.layout = _layout_from_value(self.base["layout"])
        self.left = decode_groupwise(self.base["left_codec"], device=device)
        self.right = decode_groupwise(self.base["right_codec"], device=device)
        self.input_scale = self.base["input_scale"].to(device=device, dtype=torch.float32)
        output_scale = self.base.get("output_scale")
        self.output_scale = (
            torch.ones(self.base["orig_shape"][0], device=device)
            if output_scale is None else output_scale.to(device=device, dtype=torch.float32)
        )

    def weight(self, core: torch.Tensor, side: str) -> torch.Tensor:
        unfolding = core.T @ self.right if side == "left" else self.left.T @ core
        dense = (
            _unfolding_to_matrix(unfolding, self.layout)
            * self.output_scale[:, None] / self.input_scale[None, :]
        )
        if self.sector:
            full = torch.zeros(self.artifact["orig_shape"], device=dense.device, dtype=dense.dtype)
            return full.index_copy_(self.artifact["neuron_axis"], self.bulk_indices, dense)
        return dense

    def adjoint(self, dense: torch.Tensor, side: str) -> torch.Tensor:
        if self.sector:
            dense = dense.index_select(self.artifact["neuron_axis"], self.bulk_indices)
        unfolding = _matrix_to_unfolding(
            dense * self.output_scale[:, None] / self.input_scale[None, :], self.layout,
        )
        return self.right @ unfolding.T if side == "left" else self.left @ unfolding

    def diagonal(self, input_variance, output_sensitivity, side):
        if self.sector:
            if self.artifact["neuron_axis"] == 0:
                output_sensitivity = output_sensitivity.index_select(0, self.bulk_indices)
            else:
                input_variance = input_variance.index_select(0, self.bulk_indices)
        entry_metric = (
            (output_sensitivity * self.output_scale.square())[:, None]
            * (input_variance / self.input_scale.square())[None, :]
        )
        unfolded = _matrix_to_unfolding(entry_metric, self.layout)
        return (
            self.right.square() @ unfolded.T
            if side == "left" else self.left.square() @ unfolded
        ).clamp_min(1e-20)

    def packed_candidate(self, core, side):
        base = dict(self.base)
        base[side + "_codec"] = encode_groupwise(
            core, bits=int(self.base["factor_bits"]), group_size=int(self.base["group_size"]),
        )
        if self.sector:
            return {**self.artifact, "base_artifact": base}
        return base


def _prior_anchored_pcg(normal, rhs, diagonal, *, iterations, tolerance):
    """Solve a core increment; callers include the anchoring term in normal."""

    delta = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    dot = lambda a, b: (a * b).sum(dtype=torch.float64)
    rz = dot(residual, preconditioned)
    initial_norm = residual.norm().clamp_min(1e-30)
    history = []
    for iteration in range(iterations):
        product = normal(direction)
        denominator = dot(direction, product)
        if not torch.isfinite(denominator) or denominator <= 0:
            history.append({"iteration": iteration, "stopped": "nonpositive_curvature"})
            break
        alpha = (rz / denominator).to(rhs.dtype)
        delta.add_(direction * alpha)
        residual.sub_(product * alpha)
        relative = float(residual.norm() / initial_norm)
        history.append({"iteration": iteration + 1, "relative_normal_residual": relative})
        if relative <= tolerance:
            break
        preconditioned = residual / diagonal
        rz_new = dot(residual, preconditioned)
        beta = (rz_new / rz.clamp_min(1e-30)).to(rhs.dtype)
        direction = preconditioned + beta * direction
        rz = rz_new
    return delta, history


def _mlp(weights, inputs):
    return weights["down_proj"] @ (
        F.silu(weights["gate_proj"] @ inputs) * (weights["up_proj"] @ inputs)
    )


@torch.no_grad()
def construct_functional_schmidt_mlp(
    artifacts: dict,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    gate_targets: torch.Tensor | None = None,
    sweeps: int = 1,
    cg_iterations: int = 8,
    ridge_ratio: float = 0.05,
    guard_fraction: float = 0.25,
    sequence_length: int = 128,
) -> tuple[dict, dict, dict]:
    """Joint construction with a disjoint guard from calibration data only.

    Targets may include the residual-stream correction needed for the whole
    block. Guard columns never enter normal equations. All trial acceptance
    uses the exact packed nonlinear MLP and requires improvement on both
    construction and guard objectives.
    """

    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[1] != targets.shape[1]:
        raise ValueError("inputs and targets must share their sample axis")
    if gate_targets is not None and (
        gate_targets.ndim != 2 or gate_targets.shape[1] != inputs.shape[1]
        or gate_targets.shape[0] != artifacts["gate_proj"]["orig_shape"][0]
    ):
        raise ValueError("gate targets must match the gate output and sample dimensions")
    if not 0 < guard_fraction < 1 or ridge_ratio <= 0 or cg_iterations < 1 or sweeps < 1:
        raise ValueError("invalid functional construction controls")
    if sequence_length < 1 or inputs.shape[1] % sequence_length:
        raise ValueError("calibration samples must contain complete sequence windows")
    windows = inputs.shape[1] // sequence_length
    guard_windows = max(1, math.floor(windows * guard_fraction))
    split = (windows - guard_windows) * sequence_length
    if split < sequence_length:
        raise ValueError("functional construction needs at least two sequence windows")
    result = {name: dict(artifacts[name]) for name in ("gate_proj", "up_proj", "down_proj")}
    weights = {
        name: reconstruct_packed_schmidt(artifact, device=inputs.device)
        for name, artifact in result.items()
    }
    fit_x, guard_x = inputs.float()[:, :split], inputs.float()[:, split:]
    fit_y, guard_y = targets.float()[:, :split], targets.float()[:, split:]
    fit_total = fit_y.square().sum().clamp_min(1e-30)
    guard_total = guard_y.square().sum().clamp_min(1e-30)

    def errors(candidate_weights):
        return (
            float((_mlp(candidate_weights, fit_x) - fit_y).square().sum() / fit_total),
            float((_mlp(candidate_weights, guard_x) - guard_y).square().sum() / guard_total),
        )

    initial_errors = errors(weights)
    current_fit, current_guard = initial_errors
    records = []
    for sweep in range(sweeps):
        # Down first exposes the best readout of the current nonlinear
        # features; Up and Gate then improve those features jointly through
        # that readout; a final Down solve closes the module construction.
        for role in ("down_proj", "up_proj", "gate_proj", "down_proj"):
            if role == "gate_proj" and gate_targets is None:
                continue
            for side in ("left", "right"):
                artifact = result[role]
                mapper = SchmidtCoreMap(artifact, inputs.device)
                core = mapper.left if side == "left" else mapper.right
                gate = weights["gate_proj"] @ fit_x
                up = weights["up_proj"] @ fit_x
                active = F.silu(gate) * up
                if role == "down_proj":
                    core_inputs = active
                    channel_multiplier = None
                    output_sensitivity = torch.ones(targets.shape[0], device=inputs.device)
                elif role == "up_proj":
                    core_inputs = fit_x
                    channel_multiplier = F.silu(gate)
                    output_sensitivity = (
                        weights["down_proj"].square().sum(dim=0)
                        * channel_multiplier.square().mean(dim=1)
                    )
                else:
                    core_inputs = fit_x
                    channel_multiplier = None
                    output_sensitivity = torch.ones(gate.shape[0], device=inputs.device)
                scale = 1 / math.sqrt(split)

                def forward(increment):
                    response = mapper.weight(increment, side) @ core_inputs
                    if channel_multiplier is not None:
                        response = weights["down_proj"] @ (channel_multiplier * response)
                    return response * scale

                def adjoint(output):
                    if channel_multiplier is not None:
                        output = channel_multiplier * (weights["down_proj"].T @ output)
                    dense = (output @ core_inputs.T) * scale
                    return mapper.adjoint(dense, side)

                diagonal = mapper.diagonal(
                    core_inputs.square().mean(dim=1), output_sensitivity, side,
                )
                ridge = ridge_ratio * diagonal.mean().clamp_min(1e-20)
                normal = lambda increment: adjoint(forward(increment)) + ridge * increment
                residual = (
                    (gate_targets[:, :split].float() - gate) * scale
                    if role == "gate_proj" else
                    (fit_y - weights["down_proj"] @ active) * scale
                )
                rhs = adjoint(residual)
                delta, cg_history = _prior_anchored_pcg(
                    normal, rhs, diagonal + ridge, iterations=cg_iterations, tolerance=1e-3,
                )
                trial_scores = []
                accepted_fraction = None
                accepted_artifact = None
                accepted_weight = None
                best_fit, best_guard = current_fit, current_guard
                for fraction in (1.0, 0.5, 0.25):
                    trial = mapper.packed_candidate(core + fraction * delta, side)
                    decoded = reconstruct_packed_schmidt(trial, device=inputs.device)
                    candidate_weights = {**weights, role: decoded}
                    fit_error, guard_error = errors(candidate_weights)
                    trial_scores.append({
                        "fraction": fraction, "fit_relative_mse": fit_error,
                        "guard_relative_mse": guard_error,
                    })
                    if fit_error < current_fit and guard_error < best_guard:
                        accepted_fraction = fraction
                        accepted_artifact, accepted_weight = trial, decoded
                        best_fit, best_guard = fit_error, guard_error
                if accepted_artifact is not None:
                    before_coefficients = packed_schmidt_coefficient_count_from_artifact(artifact)
                    before_bytes = packed_schmidt_payload_bytes_from_artifact(artifact)
                    after_coefficients = packed_schmidt_coefficient_count_from_artifact(accepted_artifact)
                    after_bytes = packed_schmidt_payload_bytes_from_artifact(accepted_artifact)
                    if before_coefficients != after_coefficients or before_bytes != after_bytes:
                        raise RuntimeError("functional construction changed the representation budget")
                    result[role], weights[role] = accepted_artifact, accepted_weight
                    current_fit, current_guard = best_fit, best_guard
                records.append({
                    "sweep": sweep, "role": role, "core": side,
                    "solve_objective": (
                        "clean_gate_preactivations" if role == "gate_proj"
                        else "coupled_mlp_output"
                    ),
                    "accepted_fraction": accepted_fraction,
                    "fit_relative_mse": current_fit, "guard_relative_mse": current_guard,
                    "cg_history": cg_history, "trials": trial_scores,
                })
    return result, weights, {
        "objective": "exact_coupled_swiglu_output_with_prior_anchored_core_solves",
        "uses_autograd": False,
        "uses_nonlinear_derivatives": False,
        "fit_tokens": split, "guard_tokens": targets.shape[1] - split,
        "initial_fit_relative_mse": initial_errors[0],
        "initial_guard_relative_mse": initial_errors[1],
        "final_fit_relative_mse": current_fit, "final_guard_relative_mse": current_guard,
        "stored_coefficients_added": 0,
        "tensor_payload_bytes_added": 0,
        "records": records,
    }
