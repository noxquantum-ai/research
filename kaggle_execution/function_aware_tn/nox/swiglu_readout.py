"""Closed-form readout diagnostics for the shared-neuron SwiGLU TTN.

This module does not introduce a new representation.  It keeps the existing
shared basis and Gate/Up factors fixed, exposes their rank-r nonlinear feature
matrix, and solves only the artifact's existing Down readout.  The oracle
solver is diagnostic; the calibration-only ridge solver is a valid algebraic
initial construction with no gradient or iterative weight recovery.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def shared_swiglu_features(artifact: dict, inputs: torch.Tensor) -> torch.Tensor:
    """Return T = Z^T[SiLU(Z G_r X) * (Z U_r X)]."""
    if inputs.ndim != 2:
        raise ValueError("inputs must be a matrix")
    basis = artifact["neuron_basis"].to(device=inputs.device, dtype=torch.float32)
    gate_factor = artifact["gate_factor"].to(device=inputs.device, dtype=torch.float32)
    up_factor = artifact["up_factor"].to(device=inputs.device, dtype=torch.float32)
    x = inputs.float()
    if gate_factor.shape != up_factor.shape:
        raise ValueError("Gate and Up factor shapes differ")
    if basis.shape[1] != gate_factor.shape[0] or x.shape[0] != gate_factor.shape[1]:
        raise ValueError("artifact factors and inputs have incompatible dimensions")
    gate = basis @ (gate_factor @ x)
    up = basis @ (up_factor @ x)
    return basis.T @ (F.silu(gate) * up)


def _normalized_design(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("features must have shape [rank, samples]")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    scale = features.float().square().mean(dim=1).sqrt().clamp_min(1e-8)
    return (features.float() / scale[:, None]).T.contiguous(), scale


@torch.no_grad()
def fit_linear_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_ratio: float = 0.0,
) -> torch.Tensor:
    """Solve min_B ||B T-Y||, optionally with scale-normalized ridge.

    ``ridge_ratio`` multiplies the mean diagonal of the normalized feature
    Gram matrix, making the grid dimensionless. Feature-row normalization is
    undone in the returned B, whose shape is [output_dim, rank].
    """
    if targets.ndim != 2 or targets.shape[1] != features.shape[1]:
        raise ValueError("targets must have shape [output_dim, samples]")
    if features.device != targets.device:
        raise ValueError("features and targets must share a device")
    if ridge_ratio < 0:
        raise ValueError("ridge_ratio must be nonnegative")
    design, scale = _normalized_design(features)
    response = targets.float().T.contiguous()
    if ridge_ratio == 0:
        driver = "gels" if design.is_cuda else "gelsd"
        normalized = torch.linalg.lstsq(design, response, driver=driver).solution.T
    else:
        gram = design.T @ design
        alpha = float(ridge_ratio) * float(gram.diagonal().mean())
        regularized = gram + alpha * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype,
        )
        rhs = design.T @ response
        normalized = torch.linalg.solve(regularized, rhs).T
    return (normalized / scale[None, :]).contiguous()


def relative_output_error(
    readout: torch.Tensor,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    estimate = readout.float() @ features.float()
    denominator = torch.linalg.vector_norm(targets.float()).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(estimate - targets.float()) / denominator)


def sequence_block_split(
    sample_count: int,
    sequence_length: int,
    *,
    validation_stride: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split calibration by whole sequences; every fourth is validation."""
    sample_count = int(sample_count)
    sequence_length = int(sequence_length)
    validation_stride = int(validation_stride)
    if sample_count <= 0 or sequence_length <= 0 or sample_count % sequence_length:
        raise ValueError("samples must be a positive multiple of sequence_length")
    if validation_stride < 2:
        raise ValueError("validation_stride must be at least two")
    blocks = torch.arange(sample_count) // sequence_length
    validation = torch.nonzero(blocks.remainder(validation_stride) == validation_stride - 1).flatten()
    training = torch.nonzero(blocks.remainder(validation_stride) != validation_stride - 1).flatten()
    if training.numel() == 0 or validation.numel() == 0:
        raise ValueError("split must contain both fitting and validation samples")
    return training, validation


@torch.no_grad()
def select_calibration_ridge_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    sequence_length: int,
    ridge_ratios: Iterable[float],
    validation_stride: int = 4,
) -> tuple[torch.Tensor, dict]:
    """Select ridge on an internal calibration split, then refit all data."""
    ratios = tuple(float(value) for value in ridge_ratios)
    if not ratios or any(value < 0 for value in ratios):
        raise ValueError("ridge_ratios must be a nonempty nonnegative sequence")
    fit_cpu, select_cpu = sequence_block_split(
        features.shape[1], sequence_length,
        validation_stride=validation_stride,
    )
    fit_idx = fit_cpu.to(features.device)
    select_idx = select_cpu.to(features.device)
    records = []
    best_ratio = None
    best_error = float("inf")
    for ratio in ratios:
        readout = fit_linear_readout(
            features[:, fit_idx], targets[:, fit_idx], ridge_ratio=ratio,
        )
        error = relative_output_error(
            readout, features[:, select_idx], targets[:, select_idx],
        )
        records.append({"ridge_ratio": ratio, "selection_relative_error": error})
        if error < best_error:
            best_error = error
            best_ratio = ratio
    assert best_ratio is not None
    final = fit_linear_readout(features, targets, ridge_ratio=best_ratio)
    return final, {
        "selection": "whole_sequence_internal_calibration_split",
        "validation_stride": validation_stride,
        "fit_samples": int(fit_idx.numel()),
        "selection_samples": int(select_idx.numel()),
        "grid": records,
        "selected_ridge_ratio": best_ratio,
        "selected_internal_relative_error": best_error,
        "heldout_used_for_selection": False,
    }
