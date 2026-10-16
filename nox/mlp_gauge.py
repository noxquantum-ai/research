"""Exact function-preserving gauges for SwiGLU MLP projections."""

import torch


def _normalize_feature_block(block):
    block = block.float()
    block = block - block.mean(dim=0, keepdim=True)
    scale = torch.sqrt(block.square().mean(dim=0, keepdim=True)).clamp_min(1e-6)
    return block / scale


def _rademacher(rows, columns, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (rows, columns), device=device, generator=generator)
    return signs.float().mul_(2.0).sub_(1.0).div_(rows ** 0.5)


def _balanced_pca_tensor_order(features, mode_sizes):
    """Map channels onto a tensor grid by balanced PCA quantiles."""
    mode_sizes = tuple(int(size) for size in mode_sizes)
    if any(size <= 0 for size in mode_sizes) or torch.tensor(mode_sizes).prod().item() != features.shape[0]:
        raise ValueError(f"mode_sizes={mode_sizes} do not factor {features.shape[0]} channels")
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -len(mode_sizes):].flip(1)
    scores = centered @ components

    groups = [torch.arange(features.shape[0], device=features.device)]
    for level, factor in enumerate(mode_sizes):
        next_groups = []
        for group in groups:
            if group.numel() % factor:
                raise ValueError("A balanced tensor partition requires divisible parent groups")
            values = scores.index_select(0, group)[:, level]
            tie_scale = values.abs().max().clamp_min(1.0) * 1e-7
            order = torch.argsort(values + group.float() * tie_scale / features.shape[0])
            ordered = group.index_select(0, order)
            next_groups.extend(ordered.chunk(factor))
        groups = next_groups
    permutation = torch.cat(groups)
    if torch.unique(permutation).numel() != features.shape[0]:
        raise RuntimeError("PCA tensor ordering did not produce a channel permutation")
    return permutation


def _tensor_neighbor_energy(features, permutation, mode_sizes):
    arranged = features.index_select(0, permutation).reshape(*mode_sizes, features.shape[1])
    energy = torch.zeros((), device=features.device, dtype=torch.float32)
    edges = 0
    for axis, size in enumerate(mode_sizes):
        if size <= 1:
            continue
        left = arranged.narrow(axis, 0, size - 1)
        right = arranged.narrow(axis, 1, size - 1)
        energy += (right - left).square().sum()
        edges += right.numel() // features.shape[1]
    return float(energy / max(1, edges * features.shape[1]))


@torch.no_grad()
def balanced_swiglu_gauge(weights, min_scale=0.125, max_scale=8.0):
    """Balance paired up rows and down columns under an exact SwiGLU gauge.

    For positive diagonal S, the transformation

        U' = S U,  D' = D S^-1

    leaves D(silu(Gx) * Ux) unchanged. Minimizing the paired Frobenius
    objective for each channel gives

        s_j = sqrt(||D[:, j]||_2 / ||U[j, :]||_2).

    Clipping avoids extreme finite-precision scales while preserving exactness.
    The transformed pair can remain in gauge coordinates in the checkpoint, so
    no scale vector or additional runtime parameter is stored.
    """
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError("Require 0 < min_scale <= max_scale")
    up = weights["up_proj"]
    down = weights["down_proj"]
    if up.ndim != 2 or down.ndim != 2 or up.shape[0] != down.shape[1]:
        raise ValueError(
            f"Incompatible SwiGLU up/down shapes: up={tuple(up.shape)}, down={tuple(down.shape)}"
        )

    up_norm = torch.linalg.vector_norm(up.float(), dim=1).clamp_min(1e-12)
    down_norm = torch.linalg.vector_norm(down.float(), dim=0).clamp_min(1e-12)
    unconstrained = torch.sqrt(down_norm / up_norm)
    scale = unconstrained.clamp(min=min_scale, max=max_scale).to(device=up.device, dtype=up.dtype)

    transformed = dict(weights)
    transformed["up_proj"] = up * scale.unsqueeze(1)
    transformed["down_proj"] = down / scale.unsqueeze(0).to(device=down.device, dtype=down.dtype)

    before = up_norm.square().sum() + down_norm.square().sum()
    after = (up_norm * scale.float()).square().sum() + (down_norm / scale.float()).square().sum()
    metadata = {
        "method": "exact_balanced_swiglu_up_down_gauge",
        "formula": "up'=diag(s)@up; down'=down@diag(1/s)",
        "min_scale": float(scale.float().min()),
        "max_scale": float(scale.float().max()),
        "mean_scale": float(scale.float().mean()),
        "clipped_low_channels": int((unconstrained < min_scale).sum()),
        "clipped_high_channels": int((unconstrained > max_scale).sum()),
        "paired_frobenius_objective_ratio": float(after / before.clamp_min(1e-30)),
        "stored_scale_parameters": 0,
    }
    return transformed, metadata


@torch.no_grad()
def entanglement_ordered_swiglu_gauge(
    weights,
    calibration_input,
    mode_sizes=(43, 4, 8, 8),
    weight_sketch_size=8,
    activation_sketch_size=16,
    min_scale=0.125,
    max_scale=8.0,
):
    """Choose an exact SwiGLU gauge aligned with the fixed MPO tensor grid.

    A common intermediate-channel permutation is an exact symmetry:

        G' = P G,  U' = S P U,  D' = D P^T S^-1.

    Deterministic sketches describe gate rows, up rows, down columns, and
    calibrated nonlinear activations. A balanced PCA quantile map places
    correlated channels into the existing four physical MPO modes. The
    transformed matrices remain in gauge coordinates, so P and S are not
    stored and the compressed parameter budget is unchanged.
    """
    gate = weights["gate_proj"]
    up = weights["up_proj"]
    down = weights["down_proj"]
    if gate.shape != up.shape or gate.shape[0] != down.shape[1]:
        raise ValueError(
            f"Incompatible SwiGLU shapes: gate={tuple(gate.shape)}, "
            f"up={tuple(up.shape)}, down={tuple(down.shape)}"
        )
    if calibration_input.ndim != 2 or calibration_input.shape[0] != gate.shape[1]:
        raise ValueError(
            f"Expected calibration_input [{gate.shape[1]}, N], got {tuple(calibration_input.shape)}"
        )

    input_sketch = _rademacher(gate.shape[1], weight_sketch_size, gate.device, seed=7101)
    output_sketch = _rademacher(down.shape[0], weight_sketch_size, gate.device, seed=7102)
    token_sketch = _rademacher(
        calibration_input.shape[1], activation_sketch_size, gate.device, seed=7103,
    )
    gate_activation = gate.float() @ calibration_input.float()
    up_activation = up.float() @ calibration_input.float()
    swiglu_activation = torch.nn.functional.silu(gate_activation) * up_activation
    feature_blocks = (
        gate.float() @ input_sketch,
        up.float() @ input_sketch,
        down.float().T @ output_sketch,
        swiglu_activation @ token_sketch,
    )
    features = torch.cat([_normalize_feature_block(block) for block in feature_blocks], dim=1)
    permutation = _balanced_pca_tensor_order(features, mode_sizes)
    identity = torch.arange(gate.shape[0], device=gate.device)

    permuted = dict(weights)
    permuted["gate_proj"] = gate.index_select(0, permutation)
    permuted["up_proj"] = up.index_select(0, permutation)
    permuted["down_proj"] = down.index_select(1, permutation)
    transformed, balance_metadata = balanced_swiglu_gauge(
        permuted, min_scale=min_scale, max_scale=max_scale,
    )
    identity_energy = _tensor_neighbor_energy(features, identity, mode_sizes)
    selected_energy = _tensor_neighbor_energy(features, permutation, mode_sizes)
    metadata = {
        "method": "exact_entanglement_ordered_swiglu_gauge",
        "formula": "gate'=P@gate; up'=diag(s)@P@up; down'=down@P.T@diag(1/s)",
        "selection_data": "training calibration activations and dense weights only",
        "mode_sizes": tuple(mode_sizes),
        "weight_sketch_size": int(weight_sketch_size),
        "activation_sketch_size": int(activation_sketch_size),
        "feature_dimensions": int(features.shape[1]),
        "identity_neighbor_energy": identity_energy,
        "selected_neighbor_energy": selected_energy,
        "relative_neighbor_energy_reduction": (
            0.0 if identity_energy == 0.0 else 1.0 - selected_energy / identity_energy
        ),
        "stored_permutation_parameters": 0,
        **{key: value for key, value in balance_metadata.items() if key != "method"},
    }
    return transformed, metadata
