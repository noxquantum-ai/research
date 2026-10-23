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
from nox.residual_codec import SUPPORTED_BITS, codec_payload_bytes, decode_groupwise, encode_groupwise


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


def packed_matrix_coefficient_count(rows: int, columns: int, group_size: int) -> int:
    """Logical stored coefficients for a packed matrix and its group scales.

    Bit packing changes storage, not structural parameter count. Padding codes
    are deterministic container overhead and are excluded; every logical factor
    value and every stored group scale is included.
    """

    if min(rows, columns, group_size) <= 0:
        raise ValueError("matrix dimensions and group size must be positive")
    groups_per_row = (columns + group_size - 1) // group_size
    return rows * columns + rows * groups_per_row


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


def packed_schmidt_coefficient_count(
    layout: OperatorSchmidtLayout,
    rank: int,
    group_size: int,
    *,
    input_scale_count: int = 0,
    output_scale_count: int = 0,
) -> int:
    """Logical factor and auxiliary coefficient count for a Schmidt operator."""

    rows, columns = layout.unfolding_shape
    return (
        packed_matrix_coefficient_count(rank, rows, group_size)
        + packed_matrix_coefficient_count(rank, columns, group_size)
        + int(input_scale_count)
        + int(output_scale_count)
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


def rank_for_coefficient_budget(
    layout: OperatorSchmidtLayout,
    coefficient_budget: int,
    group_size: int,
    *,
    input_scale_count: int = 0,
    output_scale_count: int = 0,
) -> int:
    """Greatest Schmidt rank fitting a logical structural budget."""

    maximum = min(layout.unfolding_shape)
    low, high = 0, maximum
    while low < high:
        middle = (low + high + 1) // 2
        count = packed_schmidt_coefficient_count(
            layout, middle, group_size,
            input_scale_count=input_scale_count,
            output_scale_count=output_scale_count,
        )
        if count <= coefficient_budget:
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


def direct_schmidt_coefficient_count(
    d_out: int, d_in: int, rank: int, group_size: int,
) -> int:
    return (
        packed_matrix_coefficient_count(rank, d_out, group_size)
        + packed_matrix_coefficient_count(rank, d_in, group_size)
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


def direct_rank_for_coefficient_budget(
    d_out: int, d_in: int, coefficient_budget: int, group_size: int,
    *, maximum_rank: int,
) -> int:
    low, high = 0, min(d_out, d_in, maximum_rank)
    while low < high:
        middle = (low + high + 1) // 2
        if direct_schmidt_coefficient_count(
            d_out, d_in, middle, group_size,
        ) <= coefficient_budget:
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
    policy: str,
) -> OperatorSchmidtLayout:
    """Choose an exact tensor cut for semantic structure or packed bond capacity."""

    if policy == "native_low_rank":
        # Matched low-rank control: the native output/input bipartition.
        # With product_covariance, its single nontrivial input mode uses
        # full input covariance. This is an established SVD baseline.
        return OperatorSchmidtLayout(
            "native_output_input_low_rank_control",
            (d_out, 1), (1, d_in), (0, 2, 1, 3),
        )
    layouts = candidate_layouts(
        d_out, d_in, projection_name, n_heads=n_heads, head_dim=head_dim,
    )
    if not layouts:
        raise ValueError(f"no exact operator-Schmidt layout for {(d_out, d_in)}")
    if policy == "semantic":
        key = lambda item: min(item.unfolding_shape) * sum(item.unfolding_shape)
    elif policy == "packed_efficiency":
        key = lambda item: sum(item.unfolding_shape)
    else:
        raise ValueError(f"unsupported layout policy: {policy!r}")
    return min(layouts, key=key)


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


def _input_mode_cholesky_factors(
    inputs: torch.Tensor,
    input_factors: tuple[int, int],
    *,
    ridge_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial traces of input covariance define a separable input metric.

    Each marginal is normalized before ridge regularization. Their Kronecker
    product is an approximation to full covariance; it is not a full whitening
    certificate. Only these small mode systems are formed, never X X^T.
    """

    i0, i1 = input_factors
    x = inputs.float().reshape(i0, i1, -1)
    marginals = (
        torch.einsum("ijn,kjn->ik", x, x),
        torch.einsum("ijn,ikn->jk", x, x),
    )
    factors = []
    for marginal in marginals:
        normalized = marginal / marginal.diagonal().mean().clamp_min(1e-20)
        normalized = (normalized + normalized.T) * 0.5
        identity = torch.eye(
            normalized.shape[0], device=normalized.device, dtype=normalized.dtype,
        )
        factors.append(torch.linalg.cholesky(normalized + ridge_ratio * identity))
    return tuple(factors)


def _matrix_in_input_metric(
    matrix: torch.Tensor, factors: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    s0, s1 = factors
    tensor = matrix.reshape(matrix.shape[0], s0.shape[0], s1.shape[0])
    return torch.einsum("oij,ia->oaj", tensor @ s1, s0).reshape_as(matrix)


def _core_from_input_metric(core: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    shape = core.shape
    rows = core.reshape(-1, chol.shape[0])
    return torch.linalg.solve_triangular(
        chol.T, rows.T.contiguous(), upper=True,
    ).T.contiguous().reshape(shape)


def _core_in_input_metric(core: torch.Tensor, chol: torch.Tensor) -> torch.Tensor:
    return (core.reshape(-1, chol.shape[0]) @ chol).reshape_as(core)


def _quantized_als(
    unfolding: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sweeps: int,
    ridge_ratio: float,
    input_metric_factors: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Fit packed Schmidt cores by gradient-free alternating least squares."""

    def quantize(core, side):
        if input_metric_factors is None:
            return _quantize_dequantize(core, bits, group_size)
        chol = input_metric_factors[side]
        physical = _core_from_input_metric(core, chol)
        physical = _quantize_dequantize(physical, bits, group_size)
        return _core_in_input_metric(physical, chol)

    left = quantize(left, 0)
    right = quantize(right, 1)
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
        left = quantize(left_t.T.contiguous(), 0)

        left_t = left.T
        gram_left = left @ left_t
        ridge_left = ridge_ratio * gram_left.diagonal().mean().clamp_min(1e-12)
        right = torch.linalg.solve(
            gram_left + ridge_left * torch.eye(
                gram_left.shape[0], device=gram_left.device, dtype=gram_left.dtype,
            ),
            left @ unfolding,
        )
        right = quantize(right, 1)
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
    coefficient_budget: int | None,
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
    if coefficient_budget is not None:
        rank = min(
            rank,
            direct_rank_for_coefficient_budget(
                d_out, d_in, coefficient_budget, group_size,
                maximum_rank=sample_count,
            ),
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
    # Construction must propagate the serialized FP16 scales as inference
    # does, rather than the higher-precision quantization simulation.
    stored_left = decode_groupwise(
        branch["left_codec"], device=all_inputs.device, dtype=torch.float32,
    )
    stored_right = decode_groupwise(
        branch["right_codec"], device=all_inputs.device, dtype=torch.float32,
    )
    correction = stored_left.T @ stored_right
    predicted_residual = correction @ all_inputs
    payload = codec_payload_bytes(branch["left_codec"]) + codec_payload_bytes(
        branch["right_codec"],
    )
    branch["payload_bytes"] = int(payload)
    diagnostics = {
        "rank": rank,
        "factor_bits": bits,
        "payload_bytes": int(payload),
        "logical_coefficient_count": direct_schmidt_coefficient_count(
            d_out, d_in, rank, group_size,
        ),
        "residual_error_after_branch": _relative_frobenius(
            all_residual, predicted_residual,
        ),
    }
    return branch, correction, diagnostics


def _column_branch_payload_bytes(branch: dict) -> int:
    return int(
        codec_payload_bytes(branch["value_codec"])
        + branch["column_indices"].numel() * branch["column_indices"].element_size()
    )


def _column_branch_coefficient_count(branch: dict) -> int:
    return packed_codec_coefficient_count(branch["value_codec"])


def _column_count_for_budgets(
    d_out: int,
    d_in: int,
    *,
    byte_budget: int,
    coefficient_budget: int | None,
    bits: int,
    group_size: int,
) -> int:
    low, high = 0, d_in
    while low < high:
        middle = (low + high + 1) // 2
        payload = packed_matrix_payload_bytes(d_out, middle, bits, group_size) + 4 * middle
        coefficients = packed_matrix_coefficient_count(d_out, middle, group_size)
        if payload <= byte_budget and (
            coefficient_budget is None or coefficients <= coefficient_budget
        ):
            low = middle
        else:
            high = middle - 1
    return low


def _fit_calibration_column_branch(
    source_residual: torch.Tensor,
    residual_outputs: torch.Tensor,
    calibration_inputs: torch.Tensor,
    *,
    byte_budget: int,
    coefficient_budget: int | None,
    bits: int,
    group_size: int,
    ridge_ratio: float,
) -> tuple[dict | None, torch.Tensor | None, dict]:
    """Fit an activation-sensitive coordinate defect outside the Schmidt cut."""

    d_out, d_in = map(int, source_residual.shape)
    column_count = _column_count_for_budgets(
        d_out, d_in, byte_budget=byte_budget,
        coefficient_budget=coefficient_budget, bits=bits, group_size=group_size,
    )
    if column_count <= 0:
        return None, None, {"column_count": 0, "payload_bytes": 0}

    input_rms = calibration_inputs.float().square().mean(dim=1).add(1e-20).sqrt()
    attribution = source_residual.float().square().sum(dim=0).sqrt() * input_rms
    indices = torch.topk(attribution, column_count, sorted=True).indices
    selected_inputs = calibration_inputs.float().index_select(0, indices)
    prior = source_residual.float().index_select(1, indices)
    gram = selected_inputs @ selected_inputs.T
    ridge = ridge_ratio * gram.diagonal().mean().clamp_min(1e-12)
    rhs = residual_outputs.float() @ selected_inputs.T + ridge * prior
    values = torch.linalg.solve(
        gram + ridge * torch.eye(
            column_count, device=gram.device, dtype=gram.dtype,
        ),
        rhs.T,
    ).T
    value_codec = encode_groupwise(values, bits=bits, group_size=group_size)
    decoded_values = decode_groupwise(
        value_codec, device=source_residual.device, dtype=torch.float32,
    )
    correction = torch.zeros_like(source_residual, dtype=torch.float32)
    correction[:, indices] = decoded_values
    predicted = decoded_values @ selected_inputs
    branch = {
        "representation": "activation_sensitive_column_defect",
        "column_indices": indices.to(device="cpu", dtype=torch.int32),
        "value_codec": value_codec,
        "factor_bits": bits,
        "group_size": group_size,
        "ridge_ratio": float(ridge_ratio),
    }
    branch["payload_bytes"] = _column_branch_payload_bytes(branch)
    branch["logical_coefficient_count"] = _column_branch_coefficient_count(branch)
    diagnostics = {
        "column_count": column_count,
        "factor_bits": bits,
        "payload_bytes": branch["payload_bytes"],
        "logical_coefficient_count": branch["logical_coefficient_count"],
        "attribution_fraction": float(
            attribution.index_select(0, indices).sum() / attribution.sum().clamp_min(1e-30)
        ),
        "residual_error_after_branch": _relative_frobenius(
            residual_outputs.float(), predicted,
        ),
    }
    return branch, correction, diagnostics


def packed_schmidt_payload_bytes_from_artifact(artifact: dict) -> int:
    if artifact.get("codec_type") == "nec_packed_sector_schmidt_v1":
        return (
            packed_schmidt_payload_bytes_from_artifact(artifact["base_artifact"])
            + codec_payload_bytes(artifact["protected_codec"])
            + artifact["protected_indices"].numel() * artifact["protected_indices"].element_size()
            + artifact["bulk_indices"].numel() * artifact["bulk_indices"].element_size()
        )
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
    column_branch = artifact.get("calibration_column_branch")
    if column_branch is not None:
        payload += _column_branch_payload_bytes(column_branch)
    complementary_branch = artifact.get("complementary_schmidt_branch")
    if complementary_branch is not None:
        payload += packed_schmidt_payload_bytes_from_artifact(complementary_branch)
    return int(payload)


def packed_codec_coefficient_count(codec: dict) -> int:
    """Count logical values represented by a groupwise codec artifact."""

    rows, columns = map(int, codec["shape"])
    count = rows * columns
    count += int(codec["group_scales"].numel())
    column_scales = codec.get("column_scales")
    if column_scales is not None:
        count += int(column_scales.numel())
    return count


def packed_schmidt_coefficient_count_from_artifact(artifact: dict) -> int:
    """Exact logical coefficient count, including every learned auxiliary."""

    if artifact.get("codec_type") == "nec_packed_sector_schmidt_v1":
        return (
            packed_schmidt_coefficient_count_from_artifact(artifact["base_artifact"])
            + packed_codec_coefficient_count(artifact["protected_codec"])
        )
    output_scale = artifact.get("output_scale")
    count = (
        packed_codec_coefficient_count(artifact["left_codec"])
        + packed_codec_coefficient_count(artifact["right_codec"])
        + int(artifact["input_scale"].numel())
        + (0 if output_scale is None else int(output_scale.numel()))
    )
    residual = artifact.get("calibration_schmidt_branch")
    if residual is not None:
        count += packed_codec_coefficient_count(residual["left_codec"])
        count += packed_codec_coefficient_count(residual["right_codec"])
    column_branch = artifact.get("calibration_column_branch")
    if column_branch is not None:
        count += _column_branch_coefficient_count(column_branch)
    complementary_branch = artifact.get("complementary_schmidt_branch")
    if complementary_branch is not None:
        count += packed_schmidt_coefficient_count_from_artifact(complementary_branch)
    return int(count)


def reconstruct_packed_schmidt(
    artifact: dict,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if artifact.get("codec_type") == "nec_packed_sector_schmidt_v1":
        from nox.sector_schmidt import reconstruct_sector_schmidt
        return reconstruct_sector_schmidt(artifact, device=device, dtype=dtype)
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
    complementary_branch = artifact.get("complementary_schmidt_branch")
    if complementary_branch is not None:
        dense += reconstruct_packed_schmidt(
            complementary_branch, device=device, dtype=torch.float32,
        )
    column_branch = artifact.get("calibration_column_branch")
    if column_branch is not None:
        indices = column_branch["column_indices"].to(device=device, dtype=torch.long)
        values = decode_groupwise(
            column_branch["value_codec"], device=device, dtype=torch.float32,
        )
        dense[:, indices] += values
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
    coefficient_budget: int | None = None,
    projection_name: str | None = None,
    layout_policy: str = "semantic",
    calibration_inputs: torch.Tensor | None = None,
    calibration_targets: torch.Tensor | None = None,
    bits_candidates: Iterable[int] = (2, 3, 4),
    group_size: int = 64,
    activation_alpha: float = 0.5,
    calibration_branch_fraction: float = 0.0,
    calibration_branch_bits: int = 3,
    calibration_branch_fit_tokens: int = 512,
    calibration_branch_ridge_ratio: float = 1e-4,
    column_branch_fraction: float = 0.0,
    column_branch_bits: int = 4,
    column_branch_ridge_ratio: float = 1e-3,
    complementary_branch_fraction: float = 0.0,
    input_metric: str = "diagonal",
    input_metric_ridge_ratio: float = 0.01,
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
    if coefficient_budget is not None and coefficient_budget <= 0:
        raise ValueError("coefficient_budget must be positive")
    if input_metric not in ("diagonal", "product_covariance"):
        raise ValueError("unsupported input metric")
    if not math.isfinite(input_metric_ridge_ratio) or input_metric_ridge_ratio <= 0:
        raise ValueError("input metric ridge must be finite and positive")
    if input_metric == "product_covariance" and calibration_inputs is None:
        raise ValueError("product covariance requires calibration inputs")
    if group_size <= 0 or als_sweeps < 0 or ridge_ratio < 0:
        raise ValueError("invalid group size or ALS controls")
    if not 0 <= calibration_branch_fraction < 1 or not 0 <= column_branch_fraction < 1:
        raise ValueError("calibration branch fractions must be in [0, 1)")
    if not 0 <= complementary_branch_fraction < 1:
        raise ValueError("complementary_branch_fraction must be in [0, 1)")
    if (
        calibration_branch_fraction + column_branch_fraction
        + complementary_branch_fraction >= 1
    ):
        raise ValueError("combined calibration branch fractions must be below one")
    if calibration_branch_bits not in SUPPORTED_BITS:
        raise ValueError(f"calibration_branch_bits must be one of {SUPPORTED_BITS}")
    if calibration_branch_fit_tokens <= 0:
        raise ValueError("calibration_branch_fit_tokens must be positive")
    if calibration_branch_ridge_ratio < 0:
        raise ValueError("calibration_branch_ridge_ratio must be nonnegative")
    if column_branch_bits not in SUPPORTED_BITS or column_branch_ridge_ratio < 0:
        raise ValueError("invalid column branch codec or ridge controls")
    if (calibration_branch_fraction or column_branch_fraction) and calibration_inputs is None:
        raise ValueError("calibration branches require calibration inputs")
    bits_values = tuple(dict.fromkeys(int(value) for value in bits_candidates))
    if not bits_values or any(value not in SUPPORTED_BITS for value in bits_values):
        raise ValueError(f"bits_candidates must be drawn from {SUPPORTED_BITS}")

    dense_source_bytes = int(weight.numel() * weight.element_size())
    source = weight.detach().float()
    d_out, d_in = map(int, source.shape)
    layout = _semantic_layout(
        d_out, d_in, projection_name, n_heads=n_heads, head_dim=head_dim,
        policy=layout_policy,
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
    mode_metric_factors = None
    core_metric_factors = None
    if input_metric == "product_covariance":
        mode_metric_factors = _input_mode_cholesky_factors(
            calibration_inputs.float() / input_scale[:, None],
            layout.input_factors, ridge_ratio=input_metric_ridge_ratio,
        )
        conditioned = _matrix_in_input_metric(conditioned, mode_metric_factors)
        if layout.partition_axes == (0, 2, 1, 3):
            core_metric_factors = mode_metric_factors
        elif layout.partition_axes == (0, 3, 1, 2):
            core_metric_factors = mode_metric_factors[::-1]
        else:
            raise ValueError("input metric requires a paired operator-Schmidt cut")
    unfolding = _matrix_to_unfolding(conditioned, layout)

    branch_budget = math.floor(byte_budget * calibration_branch_fraction)
    column_branch_budget = math.floor(byte_budget * column_branch_fraction)
    complementary_branch_budget = math.floor(byte_budget * complementary_branch_fraction)
    base_budget = (
        byte_budget - branch_budget - column_branch_budget - complementary_branch_budget
    )
    branch_coefficient_budget = (
        None
        if coefficient_budget is None
        else math.floor(coefficient_budget * calibration_branch_fraction)
    )
    column_branch_coefficient_budget = (
        None
        if coefficient_budget is None
        else math.floor(coefficient_budget * column_branch_fraction)
    )
    complementary_branch_coefficient_budget = (
        None if coefficient_budget is None
        else math.floor(coefficient_budget * complementary_branch_fraction)
    )
    base_coefficient_budget = (
        None
        if coefficient_budget is None
        else coefficient_budget
        - branch_coefficient_budget
        - column_branch_coefficient_budget
        - complementary_branch_coefficient_budget
    )

    ranks_by_bytes = {
        bits: rank_for_payload_budget(
            layout, base_budget, bits, group_size,
            input_scale_count=d_in, output_scale_count=d_out,
        )
        for bits in bits_values
    }
    rank_by_coefficients = (
        None
        if base_coefficient_budget is None
        else rank_for_coefficient_budget(
            layout, base_coefficient_budget, group_size,
            input_scale_count=d_in, output_scale_count=d_out,
        )
    )
    ranks = {
        bits: min(rank, rank_by_coefficients)
        if rank_by_coefficients is not None else rank
        for bits, rank in ranks_by_bytes.items()
    }
    if max(ranks.values(), default=0) <= 0:
        raise ValueError("byte and coefficient budgets cannot store rank one")
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
            input_metric_factors=core_metric_factors,
        )
        reconstructed_unfolding = left.T @ right
        conditioned_objective = _relative_frobenius(unfolding, reconstructed_unfolding)
        if core_metric_factors is not None:
            left = _core_from_input_metric(left, core_metric_factors[0])
            right = _core_from_input_metric(right, core_metric_factors[1])
            reconstructed_unfolding = left.T @ right
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
            "logical_coefficient_count_planned": packed_schmidt_coefficient_count(
                layout, rank, group_size, input_scale_count=d_in,
                output_scale_count=d_out,
            ),
            "rank_limit_from_bytes": ranks_by_bytes[bits],
            "rank_limit_from_coefficients": rank_by_coefficients,
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
        "construction_input_metric": input_metric,
        "construction_input_metric_ridge_ratio": input_metric_ridge_ratio,
        "left_codec": left_codec,
        "right_codec": right_codec,
        "input_scale": input_scale.to(device="cpu", dtype=torch.float16),
        "output_scale": output_scale.to(device="cpu", dtype=torch.float16),
        "byte_budget": int(byte_budget),
        "coefficient_budget": (
            None if coefficient_budget is None else int(coefficient_budget)
        ),
        "base_byte_budget": int(base_budget),
        "base_coefficient_budget": (
            None if base_coefficient_budget is None else int(base_coefficient_budget)
        ),
        "calibration_branch_byte_budget": int(branch_budget),
        "calibration_column_branch_byte_budget": int(column_branch_budget),
        "calibration_branch_coefficient_budget": (
            None
            if branch_coefficient_budget is None
            else int(branch_coefficient_budget)
        ),
        "calibration_column_branch_coefficient_budget": (
            None
            if column_branch_coefficient_budget is None
            else int(column_branch_coefficient_budget)
        ),
        "selection": selected,
        "candidate_frontier": candidates,
        "construction": "byte_bounded_block_sum_schmidt_mpo_with_gradient_free_als",
        "post_compression_healing": False,
        "dense_base_matrix": False,
        "sparse_sidecar": False,
    }
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    artifact["logical_coefficient_count"] = (
        packed_schmidt_coefficient_count_from_artifact(artifact)
    )
    if artifact["payload_bytes"] > base_budget:
        raise RuntimeError("serialized tensor payload exceeded the assigned byte budget")
    if (
        base_coefficient_budget is not None
        and artifact["logical_coefficient_count"] > base_coefficient_budget
    ):
        raise RuntimeError("artifact exceeded the assigned structural coefficient budget")
    reconstructed = reconstruct_packed_schmidt(
        artifact, device=source.device, dtype=torch.float32,
    )
    complementary_branch_diagnostics = None
    if complementary_branch_budget:
        complementary_policy = (
            "semantic" if layout_policy == "packed_efficiency" else "packed_efficiency"
        )
        complementary_targets = (
            None if target_outputs is None
            else target_outputs - reconstructed @ calibration_inputs.float()
        )
        complementary_branch, correction, complementary_branch_diagnostics = (
            compile_packed_schmidt(
                source - reconstructed,
                byte_budget=complementary_branch_budget,
                coefficient_budget=complementary_branch_coefficient_budget,
                projection_name=projection_name,
                layout_policy=complementary_policy,
                calibration_inputs=calibration_inputs,
                calibration_targets=complementary_targets,
                bits_candidates=bits_values,
                group_size=group_size,
                activation_alpha=activation_alpha,
                als_sweeps=als_sweeps,
                ridge_ratio=ridge_ratio,
                n_heads=n_heads,
                head_dim=head_dim,
                oversample=oversample,
                power_iterations=power_iterations,
                input_metric=input_metric,
                input_metric_ridge_ratio=input_metric_ridge_ratio,
                seed=seed + 20_000,
            )
        )
        artifact["complementary_schmidt_branch"] = complementary_branch
        artifact["representation"] = "sum_of_complementary_schmidt_mpos"
        artifact["complementary_branch_byte_budget"] = complementary_branch_budget
        artifact["complementary_branch_coefficient_budget"] = (
            complementary_branch_coefficient_budget
        )
        reconstructed = reconstructed + correction
    column_branch_diagnostics = None
    if column_branch_budget:
        residual_outputs = target_outputs - reconstructed @ calibration_inputs.float()
        column_branch, correction, column_branch_diagnostics = (
            _fit_calibration_column_branch(
                source - reconstructed,
                residual_outputs,
                calibration_inputs,
                byte_budget=column_branch_budget,
                coefficient_budget=column_branch_coefficient_budget,
                bits=column_branch_bits,
                group_size=group_size,
                ridge_ratio=column_branch_ridge_ratio,
            )
        )
        if column_branch is not None:
            artifact["calibration_column_branch"] = column_branch
            artifact["sparse_sidecar"] = True
            reconstructed = reconstructed + correction
    branch_diagnostics = None
    if branch_budget:
        residual_outputs = target_outputs - reconstructed @ calibration_inputs.float()
        branch, correction, branch_diagnostics = _fit_calibration_schmidt_branch(
            residual_outputs,
            calibration_inputs,
            byte_budget=branch_budget,
            coefficient_budget=branch_coefficient_budget,
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
        artifact["logical_coefficient_count"] = (
            packed_schmidt_coefficient_count_from_artifact(artifact)
        )
        if artifact["payload_bytes"] > byte_budget:
            raise RuntimeError("Schmidt branches exceeded the assigned byte budget")
        if (
            coefficient_budget is not None
            and artifact["logical_coefficient_count"] > coefficient_budget
        ):
            raise RuntimeError("Schmidt branches exceeded the structural coefficient budget")
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    artifact["logical_coefficient_count"] = (
        packed_schmidt_coefficient_count_from_artifact(artifact)
    )
    if artifact["payload_bytes"] > byte_budget:
        raise RuntimeError("combined NEC branches exceeded the assigned byte budget")
    if (
        coefficient_budget is not None
        and artifact["logical_coefficient_count"] > coefficient_budget
    ):
        raise RuntimeError("combined NEC branches exceeded the coefficient budget")
    diagnostics = {
        "layout": artifact["layout"],
        "layout_policy": layout_policy,
        "input_metric": input_metric,
        "selected_bits": bits,
        "selected_rank": rank,
        "payload_bytes": artifact["payload_bytes"],
        "byte_budget": int(byte_budget),
        "logical_coefficient_count": artifact["logical_coefficient_count"],
        "coefficient_budget": coefficient_budget,
        "dense_bytes": dense_source_bytes,
        "payload_compression_ratio": dense_source_bytes / artifact["payload_bytes"],
        "relative_weight_error": _relative_frobenius(source, reconstructed),
        "output_leg_scale_min": float(output_scale.min()),
        "output_leg_scale_max": float(output_scale.max()),
        "output_leg_scale_mean": float(output_scale.mean()),
        "candidate_frontier": candidates,
        "calibration_column_branch": column_branch_diagnostics,
        "complementary_schmidt_branch": complementary_branch_diagnostics,
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
