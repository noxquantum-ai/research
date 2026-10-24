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


def curvature_anchor(diagonal, ridge_ratio, metric='mean'):
    """Prior precision in global or coordinate-relative construction units."""
    if metric == 'mean':
        return ridge_ratio * diagonal.mean().clamp_min(1e-20)
    if metric == 'diagonal':
        return ridge_ratio * diagonal.clamp_min(1e-20)
    raise ValueError('unknown curvature anchoring metric')


class SchmidtCoreMap:
    """Exact linear core-to-weight map and its Frobenius adjoint."""

    def __init__(self, artifact: dict, device):
        self.artifact = artifact
        self.sector = artifact.get("codec_type") == "nec_packed_sector_schmidt_v1"
        self.sparse = artifact.get("codec_type") == "nec_sparse_excitation_schmidt_v1"
        self.base = artifact["base_artifact"] if self.sector or self.sparse else artifact
        if self.sparse:
            from nox.sparse_schmidt import excitation_indices, excitation_values
            self.excitation_indices = excitation_indices(artifact, device)
            self.amplitudes = excitation_values(artifact, device)
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
        if side == "amplitudes":
            return torch.zeros(self.artifact["orig_shape"], device=core.device, dtype=core.dtype).scatter_(
                1, self.excitation_indices, core,
            )
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
        if side == "amplitudes":
            return dense.gather(1, self.excitation_indices)
        if self.sector:
            dense = dense.index_select(self.artifact["neuron_axis"], self.bulk_indices)
        unfolding = _matrix_to_unfolding(
            dense * self.output_scale[:, None] / self.input_scale[None, :], self.layout,
        )
        return self.right @ unfolding.T if side == "left" else self.left @ unfolding

    def diagonal(self, input_variance, output_sensitivity, side):
        if side == "amplitudes":
            return output_sensitivity[:, None] * input_variance[self.excitation_indices]
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

    def packed_candidate(self, core, side, *, entry_weights=None):
        if side == "amplitudes":
            codec = self.artifact["amplitude_codec"]
            return {**self.artifact, "amplitude_codec": encode_groupwise(
                core, bits=int(codec["bits"]), group_size=int(codec["group_size"]),
                scale_policy=codec.get("scale_policy", "maxabs"),
                quant_grid=codec.get("quant_grid", "symmetric"),
                entry_weights=entry_weights,
            )}
        base = dict(self.base)
        base[side + "_codec"] = encode_groupwise(
            core, bits=int(self.base["factor_bits"]), group_size=int(self.base["group_size"]),
        )
        if self.sector or self.sparse:
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


def inverse_silu_nearest_branch(values, reference, *, iterations=32):
    """Invert SiLU without derivatives, choosing the nearest real branch.

    Values below SiLU's mathematical minimum project onto that minimum.
    Both negative-output roots are considered; positive outputs have one root.
    This does not clip large genuine activations.
    """
    minimum_point = -1.2784645427610738
    minimum_value = float(F.silu(torch.tensor(minimum_point, dtype=torch.float64)))
    targets = values.clamp_min(minimum_value)
    lo = torch.full_like(targets, minimum_point)
    hi = targets.clamp_min(0) + 1
    for _ in range(iterations):
        mid = (lo + hi) * .5
        below = F.silu(mid) < targets
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    right = (lo + hi) * .5
    negative_targets = targets.clamp_max(-1e-7)
    lo, hi = torch.full_like(targets, -40), torch.full_like(targets, minimum_point)
    for _ in range(iterations):
        mid = (lo + hi) * .5
        above = F.silu(mid) > negative_targets
        lo, hi = torch.where(above, mid, lo), torch.where(above, hi, mid)
    left = (lo + hi) * .5
    use_left = (targets < -1e-7) & ((left - reference).abs() < (right - reference).abs())
    return torch.where(use_left, left, right)


def readout_projected_gate_targets(gate, up, down, residual, *, ridge_ratio):
    """Alternating projection onto the linear readout and exact SiLU range.

    The minimum-norm hidden-feature correction solves a linear readout system.
    A regularized product division and branch-aware inverse SiLU then produce
    Gate preactivation targets for an ordinary linear core solve. No nonlinear
    derivative or optimizer is involved; exact nonlinear guard acceptance is
    still required by the caller.
    """
    covariance = down @ down.T
    ridge = ridge_ratio * covariance.diagonal().mean().clamp_min(1e-20)
    covariance.diagonal().add_(ridge)
    correction = down.T @ torch.cholesky_solve(residual, torch.linalg.cholesky(covariance))
    product_ridge = ridge_ratio * up.square().mean(dim=1, keepdim=True).clamp_min(1e-20)
    activation_target = F.silu(gate) + correction * up / (up.square() + product_ridge)
    return inverse_silu_nearest_branch(activation_target, gate)


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
    gate_target_policy: str = "teacher_preactivation",
    output_metric: str = "energy",
    ridge_metric: str = "mean",
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
    if gate_target_policy not in ("teacher_preactivation", "readout_projection"):
        raise ValueError("unknown Gate construction target policy")
    if output_metric not in ("energy", "channel_relative"):
        raise ValueError("unknown coupled MLP output metric")
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
    def channel_weights(values):
        energy = values.square().mean(dim=1)
        floor = energy.median().clamp_min(1e-20) * .01
        scale = energy.clamp_min(floor).rsqrt()
        return scale / scale.mean().clamp_min(1e-20)

    # A training-only diagonal output metric prevents a few large residual
    # channels from defining the entire reconstruction objective. It is a
    # construction statistic and is not an additional inference parameter.
    output_weights = channel_weights(fit_y) if output_metric == 'channel_relative' else None
    gate_weights = None
    if output_weights is not None:
        gate_weights = channel_weights(gate_targets[:, :split].float() if gate_targets is not None
                                       else weights['gate_proj'] @ fit_x)
    weighted = lambda value: value if output_weights is None else value * output_weights[:, None]
    fit_total = weighted(fit_y).square().sum().clamp_min(1e-30)
    guard_total = weighted(guard_y).square().sum().clamp_min(1e-30)

    def errors(candidate_weights):
        return (
            float(weighted(_mlp(candidate_weights, fit_x) - fit_y).square().sum() / fit_total),
            float(weighted(_mlp(candidate_weights, guard_x) - guard_y).square().sum() / guard_total),
        )

    initial_errors = errors(weights)
    current_fit, current_guard = initial_errors
    records = []
    for sweep in range(sweeps):
        # Down first exposes the best readout of the current nonlinear
        # features; Up and Gate then improve those features jointly through
        # that readout; a final Down solve closes the module construction.
        for role in ("down_proj", "up_proj", "gate_proj", "down_proj"):
            if role == "gate_proj" and gate_targets is None and gate_target_policy == "teacher_preactivation":
                continue
            sides = ("left", "right", "amplitudes") if result[role].get("codec_type") == "nec_sparse_excitation_schmidt_v1" else ("left", "right")
            for side in sides:
                artifact = result[role]
                mapper = SchmidtCoreMap(artifact, inputs.device)
                core = getattr(mapper, side) if side == "amplitudes" else (mapper.left if side == "left" else mapper.right)
                gate = weights["gate_proj"] @ fit_x
                up = weights["up_proj"] @ fit_x
                active = F.silu(gate) * up
                if role == "down_proj":
                    core_inputs = active
                    channel_multiplier = None
                    output_sensitivity = (torch.ones(targets.shape[0], device=inputs.device)
                                          if output_weights is None else output_weights.square())
                elif role == "up_proj":
                    core_inputs = fit_x
                    channel_multiplier = F.silu(gate)
                    output_sensitivity = (
                        (weights["down_proj"].square().sum(dim=0) if output_weights is None else
                         (weights["down_proj"].square() * output_weights.square()[:, None]).sum(dim=0))
                        * channel_multiplier.square().mean(dim=1)
                    )
                else:
                    core_inputs = fit_x
                    channel_multiplier = None
                    output_sensitivity = (torch.ones(gate.shape[0], device=inputs.device)
                                          if gate_weights is None else gate_weights.square())
                solve_weights = gate_weights if role == 'gate_proj' else output_weights
                gate_solve_target = None
                if role == "gate_proj":
                    gate_solve_target = (
                        gate_targets[:, :split].float() if gate_target_policy == "teacher_preactivation"
                        else readout_projected_gate_targets(
                            gate, up, weights["down_proj"], fit_y - weights["down_proj"] @ active,
                            ridge_ratio=ridge_ratio,
                        )
                    )
                scale = 1 / math.sqrt(split)

                def forward(increment):
                    response = mapper.weight(increment, side) @ core_inputs
                    if channel_multiplier is not None:
                        response = weights["down_proj"] @ (channel_multiplier * response)
                    if solve_weights is not None:
                        response = response * solve_weights[:, None]
                    return response * scale

                def adjoint(output):
                    if solve_weights is not None:
                        output = output * solve_weights[:, None]
                    if channel_multiplier is not None:
                        output = channel_multiplier * (weights["down_proj"].T @ output)
                    dense = (output @ core_inputs.T) * scale
                    return mapper.adjoint(dense, side)

                diagonal = mapper.diagonal(
                    core_inputs.square().mean(dim=1), output_sensitivity, side,
                )
                ridge = curvature_anchor(diagonal, ridge_ratio, ridge_metric)
                normal = lambda increment: adjoint(forward(increment)) + ridge * increment
                residual = (
                    (gate_solve_target - gate) * scale
                    if role == "gate_proj" else
                    (fit_y - weights["down_proj"] @ active) * scale
                )
                if solve_weights is not None:
                    residual = residual * solve_weights[:, None]
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
                    trial = mapper.packed_candidate(
                        core + fraction * delta, side,
                        entry_weights=diagonal if output_metric == 'channel_relative' and side == 'amplitudes' else None,
                    )
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
                        gate_target_policy if role == "gate_proj"
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
        "gate_target_policy": gate_target_policy,
        "output_metric": output_metric,
        "ridge_metric": ridge_metric,
        "output_metric_uses_training_only": True,
        "fit_tokens": split, "guard_tokens": targets.shape[1] - split,
        "initial_fit_relative_mse": initial_errors[0],
        "initial_guard_relative_mse": initial_errors[1],
        "final_fit_relative_mse": current_fit, "final_guard_relative_mse": current_guard,
        "stored_coefficients_added": 0,
        "tensor_payload_bytes_added": 0,
        "records": records,
    }
