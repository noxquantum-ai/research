"""Packed operator-Schmidt codec for direct NEC construction.

The stored operator is a two-site matrix product operator.  Its canonical
Schmidt factors are packed at low bit width, while the bond dimension is
chosen from an explicit serialized byte allowance.  This is a direct weight
representation: it has no dense base matrix, sparse shield, correction
sidecar, or post-compression healing parameters.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

from nox.operator_schmidt import (
    OperatorSchmidtLayout,
    _matrix_to_unfolding,
    _truncated_svd,
    _unfolding_to_matrix,
    candidate_layouts,
    operator_schmidt_to_dense,
)
from nox.residual_codec import codec_payload_bytes, decode_groupwise, encode_groupwise


CODEC_TYPE = "nec_packed_operator_schmidt_v1"


def packed_matrix_payload_bytes(rows: int, columns: int, bits: int, group_size: int) -> int:
    """Exact tensor payload emitted by ``encode_groupwise`` for one matrix."""

    if min(rows, columns, bits, group_size) <= 0:
        raise ValueError("matrix dimensions, bits, and group size must be positive")
    groups_per_row = (columns + group_size - 1) // group_size
    padded_values = rows * groups_per_row * group_size
    code_bytes = (padded_values * bits + 7) // 8
    scale_bytes = rows * groups_per_row * 2
    return code_bytes + scale_bytes


def packed_schmidt_payload_bytes(
    layout: OperatorSchmidtLayout,
    rank: int,
    bits: int,
    group_size: int,
    *,
    input_scale_count: int = 0,
) -> int:
    rows, columns = layout.unfolding_shape
    return (
        packed_matrix_payload_bytes(rank, rows, bits, group_size)
        + packed_matrix_payload_bytes(rank, columns, bits, group_size)
        + 2 * int(input_scale_count)
    )


def rank_for_payload_budget(
    layout: OperatorSchmidtLayout,
    byte_budget: int,
    bits: int,
    group_size: int,
    *,
    input_scale_count: int = 0,
) -> int:
    """Return the greatest Schmidt rank whose packed tensors fit the budget."""

    maximum = min(layout.unfolding_shape)
    low, high = 0, maximum
    while low < high:
        middle = (low + high + 1) // 2
        payload = packed_schmidt_payload_bytes(
            layout, middle, bits, group_size,
            input_scale_count=input_scale_count,
        )
        if payload <= byte_budget:
            low = middle
        else:
            high = middle - 1
    return low


def _semantic_layout(
    d_out: int,
    d_in: int,
    projection_name: str | None,
    *,
    n_heads: int,
    head_dim: int,
) -> OperatorSchmidtLayout:
    """Choose the least redundant nontrivial cut from the declared layouts."""

    layouts = candidate_layouts(
        d_out, d_in, projection_name, n_heads=n_heads, head_dim=head_dim,
    )
    if not layouts:
        raise ValueError(f"no exact operator-Schmidt layout for {(d_out, d_in)}")
    return min(
        layouts,
        key=lambda item: min(item.unfolding_shape) * sum(item.unfolding_shape),
    )


def _quantize_dequantize(matrix: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    """Simulate the row-local packed codec without transferring through CPU."""

    rows, columns = matrix.shape
    groups = (columns + group_size - 1) // group_size
    padded = groups * group_size
    source = torch.nn.functional.pad(matrix, (0, padded - columns))
    blocks = source.reshape(rows, groups, group_size)
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    scale = blocks.abs().amax(dim=-1) / float(qmax)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(blocks / safe.unsqueeze(-1)).clamp(qmin, qmax)
    return (codes * scale.unsqueeze(-1)).reshape(rows, padded)[:, :columns]


def _relative_frobenius(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference - candidate).norm() / reference.norm().clamp_min(1e-30))


def _quantized_als(
    unfolding: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sweeps: int,
    ridge_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Fit packed Schmidt cores by gradient-free alternating least squares."""

    left = _quantize_dequantize(left, bits, group_size)
    right = _quantize_dequantize(right, bits, group_size)
    history = [{"sweep": 0, "relative_unfolding_error": _relative_frobenius(
        unfolding, left.T @ right,
    )}]
    for sweep in range(1, sweeps + 1):
        previous_left, previous_right = left, right
        previous_error = history[-1]["relative_unfolding_error"]
        gram_right = right @ right.T
        ridge_right = ridge_ratio * gram_right.diagonal().mean().clamp_min(1e-12)
        rhs_left = unfolding @ right.T
        left_t = torch.linalg.solve(
            gram_right + ridge_right * torch.eye(
                gram_right.shape[0], device=gram_right.device, dtype=gram_right.dtype,
            ),
            rhs_left.T,
        ).T
        left = _quantize_dequantize(left_t.T.contiguous(), bits, group_size)

        left_t = left.T
        gram_left = left @ left_t
        ridge_left = ridge_ratio * gram_left.diagonal().mean().clamp_min(1e-12)
        right = torch.linalg.solve(
            gram_left + ridge_left * torch.eye(
                gram_left.shape[0], device=gram_left.device, dtype=gram_left.dtype,
            ),
            left @ unfolding,
        )
        right = _quantize_dequantize(right, bits, group_size)
        error = _relative_frobenius(unfolding, left.T @ right)
        accepted = error <= previous_error
        if not accepted:
            left, right, error = previous_left, previous_right, previous_error
        history.append({
            "sweep": sweep,
            "relative_unfolding_error": error,
            "accepted": accepted,
        })
    return left, right, history


def packed_schmidt_payload_bytes_from_artifact(artifact: dict) -> int:
    return int(
        codec_payload_bytes(artifact["left_codec"])
        + codec_payload_bytes(artifact["right_codec"])
        + artifact["input_scale"].numel() * artifact["input_scale"].element_size()
    )


def reconstruct_packed_schmidt(
    artifact: dict,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported packed Schmidt codec: {artifact.get('codec_type')!r}")
    left = decode_groupwise(artifact["left_codec"], device=device, dtype=torch.float32)
    right = decode_groupwise(artifact["right_codec"], device=device, dtype=torch.float32)
    dense = operator_schmidt_to_dense(left, right, artifact["layout"])
    scale = artifact["input_scale"].to(device=device, dtype=torch.float32)
    return (dense / scale[None, :]).to(dtype=dtype)


@torch.no_grad()
def compile_packed_schmidt(
    weight: torch.Tensor,
    *,
    byte_budget: int,
    projection_name: str | None = None,
    calibration_inputs: torch.Tensor | None = None,
    bits_candidates: Iterable[int] = (2, 3, 4),
    group_size: int = 64,
    activation_alpha: float = 0.5,
    als_sweeps: int = 1,
    ridge_ratio: float = 1e-5,
    n_heads: int = 32,
    head_dim: int = 128,
    oversample: int = 8,
    power_iterations: int = 1,
    seed: int = 0,
) -> tuple[dict, torch.Tensor, dict]:
    """Compile one matrix directly into a byte-bounded packed Schmidt MPO."""

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if byte_budget <= 0:
        raise ValueError("byte_budget must be positive")
    if group_size <= 0 or als_sweeps < 0 or ridge_ratio < 0:
        raise ValueError("invalid group size or ALS controls")
    bits_values = tuple(dict.fromkeys(int(value) for value in bits_candidates))
    if not bits_values or any(value not in (2, 3, 4) for value in bits_values):
        raise ValueError("bits_candidates must contain only 2, 3, or 4")

    dense_source_bytes = int(weight.numel() * weight.element_size())
    source = weight.detach().float()
    d_out, d_in = map(int, source.shape)
    layout = _semantic_layout(
        d_out, d_in, projection_name, n_heads=n_heads, head_dim=head_dim,
    )
    if calibration_inputs is None:
        input_scale = torch.ones(d_in, device=source.device, dtype=torch.float32)
    else:
        if calibration_inputs.ndim != 2 or calibration_inputs.shape[0] != d_in:
            raise ValueError("calibration_inputs must have shape [in_features, tokens]")
        rms = calibration_inputs.detach().float().square().mean(dim=1).add(1e-12).sqrt()
        normalized = rms / rms.mean().clamp_min(1e-12)
        input_scale = normalized.pow(float(activation_alpha)).clamp(0.25, 4.0)
    conditioned = source * input_scale[None, :]
    unfolding = _matrix_to_unfolding(conditioned, layout)

    ranks = {
        bits: rank_for_payload_budget(
            layout, byte_budget, bits, group_size, input_scale_count=d_in,
        )
        for bits in bits_values
    }
    if max(ranks.values(), default=0) <= 0:
        raise ValueError("byte budget cannot store rank one")
    maximum_rank = max(ranks.values())
    u, values, vh = _truncated_svd(
        unfolding, maximum_rank, oversample=oversample,
        power_iterations=power_iterations, seed=seed,
    )

    candidates = []
    best = None
    for bits in bits_values:
        rank = ranks[bits]
        if rank <= 0:
            continue
        root = values[:rank].clamp_min(0).sqrt()
        left = (u[:, :rank] * root[None, :]).T.contiguous()
        right = (root[:, None] * vh[:rank]).contiguous()
        left, right, history = _quantized_als(
            unfolding, left, right, bits=bits, group_size=group_size,
            sweeps=als_sweeps, ridge_ratio=ridge_ratio,
        )
        reconstructed_unfolding = left.T @ right
        objective = _relative_frobenius(unfolding, reconstructed_unfolding)
        record = {
            "bits": bits,
            "rank": rank,
            "payload_bytes_planned": packed_schmidt_payload_bytes(
                layout, rank, bits, group_size, input_scale_count=d_in,
            ),
            "relative_conditioned_weight_error": objective,
            "als_history": history,
        }
        candidates.append(record)
        if best is None or objective < best[0]:
            best = (objective, bits, rank, left, right, record)
    assert best is not None
    _, bits, rank, left, right, selected = best
    left_codec = encode_groupwise(left, bits=bits, group_size=group_size)
    right_codec = encode_groupwise(right, bits=bits, group_size=group_size)
    artifact = {
        "codec_type": CODEC_TYPE,
        "representation": "two_site_operator_schmidt_mpo",
        "orig_shape": (d_out, d_in),
        "layout": {
            "name": layout.name,
            "output_factors": layout.output_factors,
            "input_factors": layout.input_factors,
            "partition_axes": layout.partition_axes,
        },
        "rank": rank,
        "factor_bits": bits,
        "group_size": group_size,
        "left_codec": left_codec,
        "right_codec": right_codec,
        "input_scale": input_scale.to(device="cpu", dtype=torch.float16),
        "byte_budget": int(byte_budget),
        "selection": selected,
        "candidate_frontier": candidates,
        "construction": "canonical_schmidt_plus_gradient_free_quantized_als",
        "post_compression_healing": False,
        "dense_base_matrix": False,
        "sparse_sidecar": False,
    }
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    if artifact["payload_bytes"] > byte_budget:
        raise RuntimeError("serialized tensor payload exceeded the assigned byte budget")
    reconstructed = reconstruct_packed_schmidt(
        artifact, device=source.device, dtype=torch.float32,
    )
    diagnostics = {
        "layout": artifact["layout"],
        "selected_bits": bits,
        "selected_rank": rank,
        "payload_bytes": artifact["payload_bytes"],
        "byte_budget": int(byte_budget),
        "dense_bytes": dense_source_bytes,
        "payload_compression_ratio": dense_source_bytes / artifact["payload_bytes"],
        "relative_weight_error": _relative_frobenius(source, reconstructed),
        "candidate_frontier": candidates,
    }
    if calibration_inputs is not None:
        target = source @ calibration_inputs.float()
        prediction = reconstructed @ calibration_inputs.float()
        diagnostics["relative_calibration_output_error"] = _relative_frobenius(target, prediction)
    return artifact, reconstructed, diagnostics
