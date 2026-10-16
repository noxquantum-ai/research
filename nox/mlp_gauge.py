"""Exact function-preserving gauges for SwiGLU MLP projections."""

import torch


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
