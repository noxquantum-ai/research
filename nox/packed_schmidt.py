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


LEGACY_CODEC_TYPES = (
    "nec_packed_operator_schmidt_v1",
    "nec_packed_operator_schmidt_v2",
)
CODEC_TYPE = "nec_packed_operator_schmidt_v3"


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
    output_scale_count: int = 0,
) -> int:
    rows, columns = layout.unfolding_shape
    return (
        packed_matrix_payload_bytes(rank, rows, bits, group_size)
        + packed_matrix_payload_bytes(rank, columns, bits, group_size)
        + 2 * int(input_scale_count)
        + 2 * int(output_scale_count)
    )


def rank_for_payload_budget(
    layout: OperatorSchmidtLayout,
    byte_budget: int,
    bits: int,
    group_size: int,
    *,
    input_scale_count: int = 0,
    output_scale_count: int = 0,
) -> int:
    """Return the greatest Schmidt rank whose packed tensors fit the budget."""

    maximum = min(layout.unfolding_shape)
    low, high = 0, maximum
    while low < high:
        middle = (low + high + 1) // 2
        payload = packed_schmidt_payload_bytes(
            layout, middle, bits, group_size,
            input_scale_count=input_scale_count,
            output_scale_count=output_scale_count,
        )
        if payload <= byte_budget:
            low = middle
        else:
            high = middle - 1
    return low


def direct_schmidt_payload_bytes(
    d_out: int, d_in: int, rank: int, bits: int, group_size: int,
) -> int:
    return (
        packed_matrix_payload_bytes(rank, d_out, bits, group_size)
        + packed_matrix_payload_bytes(rank, d_in, bits, group_size)
    )


def direct_rank_for_payload_budget(
    d_out: int, d_in: int, byte_budget: int, bits: int, group_size: int,
    *, maximum_rank: int,
) -> int:
    low, high = 0, min(d_out, d_in, maximum_rank)
    while low < high:
        middle = (low + high + 1) // 2
        if direct_schmidt_payload_bytes(
            d_out, d_in, middle, bits, group_size,
        ) <= byte_budget:
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


def _fit_calibration_schmidt_branch(
    residual_outputs: torch.Tensor,
    calibration_inputs: torch.Tensor,
    *,
    byte_budget: int,
    bits: int,
    group_size: int,
    fit_tokens: int,
    ridge_ratio: float,
    seed: int,
) -> tuple[dict | None, torch.Tensor | None, dict]:
    """Fit a byte-bounded Schmidt branch to residual behavior by direct ALS."""

    d_out, tokens = residual_outputs.shape
    d_in = int(calibration_inputs.shape[0])
    sample_count = min(tokens, int(fit_tokens))
    if byte_budget <= 0 or sample_count <= 0:
        return None, None, {"rank": 0, "payload_bytes": 0}
    rank = direct_rank_for_payload_budget(
        d_out, d_in, byte_budget, bits, group_size,
        maximum_rank=sample_count,
    )
    if rank <= 0:
        return None, None, {"rank": 0, "payload_bytes": 0}

    if sample_count == tokens:
        indices = torch.arange(tokens, device=calibration_inputs.device)
    else:
        indices = torch.linspace(
            0, tokens - 1, sample_count, device=calibration_inputs.device,
        ).round().long()
    inputs = calibration_inputs.float().index_select(1, indices)
    residual = residual_outputs.float().index_select(1, indices)
    u, values, vh = _truncated_svd(
        residual, rank, oversample=8, power_iterations=1, seed=seed,
    )
    root = values[:rank].clamp_min(0).sqrt()
    left = (u[:, :rank] * root[None, :]).T.contiguous()
    temporal = (root[:, None] * vh[:rank]).contiguous()

    input_gram = inputs.T @ inputs
    input_ridge = ridge_ratio * input_gram.diagonal().mean().clamp_min(1e-12)
    input_dual = torch.linalg.solve(
        input_gram + input_ridge * torch.eye(
            sample_count, device=input_gram.device, dtype=input_gram.dtype,
        ),
        inputs.T,
    )
    right = temporal @ input_dual
    left = _quantize_dequantize(left, bits, group_size)
    right = _quantize_dequantize(right, bits, group_size)

    # One gradient-free alternating sweep uses every calibration token for the
    # output core, then the sampled activation system for the input core.
    all_inputs = calibration_inputs.float()
    all_residual = residual_outputs.float()
    coefficients = right @ all_inputs
    coefficient_gram = coefficients @ coefficients.T
    coefficient_ridge = ridge_ratio * coefficient_gram.diagonal().mean().clamp_min(1e-12)
    left_t = torch.linalg.solve(
        coefficient_gram + coefficient_ridge * torch.eye(
            rank, device=coefficient_gram.device, dtype=coefficient_gram.dtype,
        ),
        (all_residual @ coefficients.T).T,
    ).T
    left = _quantize_dequantize(left_t.T.contiguous(), bits, group_size)

    left_gram = left @ left.T
    left_ridge = ridge_ratio * left_gram.diagonal().mean().clamp_min(1e-12)
    desired_temporal = torch.linalg.solve(
        left_gram + left_ridge * torch.eye(
            rank, device=left_gram.device, dtype=left_gram.dtype,
        ),
        left @ residual,
    )
    right = _quantize_dequantize(desired_temporal @ input_dual, bits, group_size)

    coefficients = right @ all_inputs
    coefficient_gram = coefficients @ coefficients.T
    coefficient_ridge = ridge_ratio * coefficient_gram.diagonal().mean().clamp_min(1e-12)
    left_t = torch.linalg.solve(
        coefficient_gram + coefficient_ridge * torch.eye(
            rank, device=coefficient_gram.device, dtype=coefficient_gram.dtype,
        ),
        (all_residual @ coefficients.T).T,
    ).T
    left = _quantize_dequantize(left_t.T.contiguous(), bits, group_size)

    correction = left.T @ right
    predicted_residual = correction @ all_inputs
    branch = {
        "representation": "activation_residual_schmidt_branch",
        "rank": rank,
        "factor_bits": bits,
        "group_size": group_size,
        "left_codec": encode_groupwise(left, bits=bits, group_size=group_size),
        "right_codec": encode_groupwise(right, bits=bits, group_size=group_size),
        "fit_tokens": sample_count,
        "gradient_free_als_sweeps": 1,
    }
    payload = codec_payload_bytes(branch["left_codec"]) + codec_payload_bytes(
        branch["right_codec"],
    )
    branch["payload_bytes"] = int(payload)
    diagnostics = {
        "rank": rank,
        "factor_bits": bits,
        "payload_bytes": int(payload),
        "residual_error_after_branch": _relative_frobenius(
            all_residual, predicted_residual,
        ),
    }
    return branch, correction, diagnostics


def packed_schmidt_payload_bytes_from_artifact(artifact: dict) -> int:
    output_scale = artifact.get("output_scale")
    payload = int(
        codec_payload_bytes(artifact["left_codec"])
        + codec_payload_bytes(artifact["right_codec"])
        + artifact["input_scale"].numel() * artifact["input_scale"].element_size()
        + (0 if output_scale is None else output_scale.numel() * output_scale.element_size())
    )
    residual = artifact.get("calibration_schmidt_branch")
    if residual is not None:
        payload += codec_payload_bytes(residual["left_codec"])
        payload += codec_payload_bytes(residual["right_codec"])
    return int(payload)


def reconstruct_packed_schmidt(
    artifact: dict,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if artifact.get("codec_type") not in (*LEGACY_CODEC_TYPES, CODEC_TYPE):
        raise ValueError(f"unsupported packed Schmidt codec: {artifact.get('codec_type')!r}")
    left = decode_groupwise(artifact["left_codec"], device=device, dtype=torch.float32)
    right = decode_groupwise(artifact["right_codec"], device=device, dtype=torch.float32)
    dense = operator_schmidt_to_dense(left, right, artifact["layout"])
    scale = artifact["input_scale"].to(device=device, dtype=torch.float32)
    dense = dense / scale[None, :]
    if artifact.get("output_scale") is not None:
        output_scale = artifact["output_scale"].to(device=device, dtype=torch.float32)
        dense = dense * output_scale[:, None]
    residual = artifact.get("calibration_schmidt_branch")
    if residual is not None:
        residual_left = decode_groupwise(
            residual["left_codec"], device=device, dtype=torch.float32,
        )
        residual_right = decode_groupwise(
            residual["right_codec"], device=device, dtype=torch.float32,
        )
        dense = dense + residual_left.T @ residual_right
    return dense.to(dtype=dtype)


@torch.no_grad()
def compile_packed_schmidt(
    weight: torch.Tensor,
    *,
    byte_budget: int,
    projection_name: str | None = None,
    calibration_inputs: torch.Tensor | None = None,
    calibration_targets: torch.Tensor | None = None,
    bits_candidates: Iterable[int] = (2, 3, 4),
    group_size: int = 64,
    activation_alpha: float = 0.5,
    calibration_branch_fraction: float = 0.0,
    calibration_branch_bits: int = 3,
    calibration_branch_fit_tokens: int = 512,
    calibration_branch_ridge_ratio: float = 1e-4,
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
    if not 0 <= calibration_branch_fraction < 1:
        raise ValueError("calibration_branch_fraction must be in [0, 1)")
    if calibration_branch_bits not in (2, 3, 4):
        raise ValueError("calibration_branch_bits must be 2, 3, or 4")
    if calibration_branch_fit_tokens <= 0:
        raise ValueError("calibration_branch_fit_tokens must be positive")
    if calibration_branch_ridge_ratio < 0:
        raise ValueError("calibration_branch_ridge_ratio must be nonnegative")
    if calibration_branch_fraction and calibration_inputs is None:
        raise ValueError("a calibration Schmidt branch requires calibration inputs")
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
        if calibration_targets is not None:
            raise ValueError("calibration_targets require calibration_inputs")
        input_scale = torch.ones(d_in, device=source.device, dtype=torch.float32)
        target_outputs = None
    else:
        if calibration_inputs.ndim != 2 or calibration_inputs.shape[0] != d_in:
            raise ValueError("calibration_inputs must have shape [in_features, tokens]")
        if calibration_targets is not None and (
            calibration_targets.ndim != 2
            or calibration_targets.shape != (d_out, calibration_inputs.shape[1])
        ):
            raise ValueError("calibration_targets must have shape [out_features, tokens]")
        rms = calibration_inputs.detach().float().square().mean(dim=1).add(1e-12).sqrt()
        normalized = rms / rms.mean().clamp_min(1e-12)
        input_scale = normalized.pow(float(activation_alpha)).clamp(0.25, 4.0)
        target_outputs = (
            source @ calibration_inputs.float()
            if calibration_targets is None
            else calibration_targets.detach().float()
        )
    conditioned = source * input_scale[None, :]
    unfolding = _matrix_to_unfolding(conditioned, layout)

    branch_budget = math.floor(byte_budget * calibration_branch_fraction)
    base_budget = byte_budget - branch_budget

    ranks = {
        bits: rank_for_payload_budget(
            layout, base_budget, bits, group_size,
            input_scale_count=d_in, output_scale_count=d_out,
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
        conditioned_objective = _relative_frobenius(unfolding, reconstructed_unfolding)
        output_scale = torch.ones(d_out, device=source.device, dtype=torch.float32)
        calibration_objective = None
        if calibration_inputs is not None:
            candidate_matrix = (
                _unfolding_to_matrix(reconstructed_unfolding, layout)
                / input_scale[None, :]
            )
            prediction = candidate_matrix @ calibration_inputs.float()
            numerator = (target_outputs * prediction).sum(dim=1)
            denominator = prediction.square().sum(dim=1).clamp_min(1e-20)
            output_scale = (numerator / denominator).nan_to_num(1.0).clamp(0.25, 4.0)
            prediction.mul_(output_scale[:, None])
            calibration_objective = _relative_frobenius(target_outputs, prediction)
            objective = calibration_objective
            del candidate_matrix, prediction
        else:
            objective = conditioned_objective
        record = {
            "bits": bits,
            "rank": rank,
            "payload_bytes_planned": packed_schmidt_payload_bytes(
                layout, rank, bits, group_size, input_scale_count=d_in,
                output_scale_count=d_out,
            ),
            "relative_conditioned_weight_error": conditioned_objective,
            "relative_calibration_output_error_after_output_leg_scale": calibration_objective,
            "als_history": history,
        }
        candidates.append(record)
        if best is None or objective < best[0]:
            best = (objective, bits, rank, left, right, output_scale, record)
    assert best is not None
    _, bits, rank, left, right, output_scale, selected = best
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
        "output_scale": output_scale.to(device="cpu", dtype=torch.float16),
        "byte_budget": int(byte_budget),
        "base_byte_budget": int(base_budget),
        "calibration_branch_byte_budget": int(branch_budget),
        "selection": selected,
        "candidate_frontier": candidates,
        "construction": "byte_bounded_block_sum_schmidt_mpo_with_gradient_free_als",
        "post_compression_healing": False,
        "dense_base_matrix": False,
        "sparse_sidecar": False,
    }
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    if artifact["payload_bytes"] > base_budget:
        raise RuntimeError("serialized tensor payload exceeded the assigned byte budget")
    reconstructed = reconstruct_packed_schmidt(
        artifact, device=source.device, dtype=torch.float32,
    )
    branch_diagnostics = None
    if branch_budget:
        residual_outputs = target_outputs - reconstructed @ calibration_inputs.float()
        branch, correction, branch_diagnostics = _fit_calibration_schmidt_branch(
            residual_outputs,
            calibration_inputs,
            byte_budget=branch_budget,
            bits=calibration_branch_bits,
            group_size=group_size,
            fit_tokens=calibration_branch_fit_tokens,
            ridge_ratio=calibration_branch_ridge_ratio,
            seed=seed + 10_000,
        )
        if branch is not None:
            artifact["calibration_schmidt_branch"] = branch
            reconstructed = reconstructed + correction
        artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
        if artifact["payload_bytes"] > byte_budget:
            raise RuntimeError("Schmidt branches exceeded the assigned byte budget")
    diagnostics = {
        "layout": artifact["layout"],
        "selected_bits": bits,
        "selected_rank": rank,
        "payload_bytes": artifact["payload_bytes"],
        "byte_budget": int(byte_budget),
        "dense_bytes": dense_source_bytes,
        "payload_compression_ratio": dense_source_bytes / artifact["payload_bytes"],
        "relative_weight_error": _relative_frobenius(source, reconstructed),
        "output_leg_scale_min": float(output_scale.min()),
        "output_leg_scale_max": float(output_scale.max()),
        "output_leg_scale_mean": float(output_scale.mean()),
        "candidate_frontier": candidates,
        "calibration_schmidt_branch": branch_diagnostics,
    }
    if calibration_inputs is not None:
        prediction = reconstructed @ calibration_inputs.float()
        diagnostics["relative_calibration_output_error"] = _relative_frobenius(
            target_outputs, prediction,
        )
        diagnostics["calibration_target"] = (
            "dense_operator_on_same_inputs"
            if calibration_targets is None
            else "external_clean_teacher_outputs"
        )
    return artifact, reconstructed, diagnostics
