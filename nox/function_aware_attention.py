"""Function-aware low-rank controls for a complete Llama attention module."""

from __future__ import annotations

import torch

from nox.function_aware_tn import (
    activation_weighted_svd_factors,
    activation_weighted_svd_factors_from_cholesky,
    fit_functional_down_readout,
    ordinary_svd_factors,
    regularized_covariance_cholesky,
)
from nox.block_compiler import scaled_dot_product_attention_2d


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def attention_factor_coefficient_count(weights, ranks):
    if len(ranks) != 4:
        raise ValueError("attention needs Q, K, V, and O ranks")
    return sum(
        weights[name].shape[0] * int(rank) + int(rank) * weights[name].shape[1]
        for name, rank in zip(ATTENTION_PROJECTIONS, ranks)
    )


def attention_dense_coefficient_count(weights):
    return sum(weights[name].numel() for name in ATTENTION_PROJECTIONS)


def _factor(weight, inputs, rank, mode, seed, input_cholesky=None):
    if mode == "ordinary_svd":
        return ordinary_svd_factors(weight, rank, seed=seed)
    if mode == "activation_weighted_svd":
        if input_cholesky is not None:
            return activation_weighted_svd_factors_from_cholesky(
                weight, input_cholesky, rank, seed=seed,
            )
        return activation_weighted_svd_factors(weight, inputs, rank, seed=seed)
    raise ValueError(f"unknown projection factorization mode: {mode}")


def _projected_context(weights, inputs, n_heads, n_kv_heads, seq_len, factors=None):
    projected = {}
    for name in ("q_proj", "k_proj", "v_proj"):
        if factors is None:
            projected[name] = weights[name] @ inputs
        else:
            left, right = factors[name]
            projected[name] = left @ (right @ inputs)
    return scaled_dot_product_attention_2d(
        projected["q_proj"], projected["k_proj"], projected["v_proj"],
        n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len,
    )


def apply_factorized_attention(artifact, inputs, *, n_heads, n_kv_heads, seq_len):
    factors = artifact["factors"]
    context = _projected_context(
        {}, inputs, n_heads, n_kv_heads, seq_len,
        factors={name: factors[name] for name in ("q_proj", "k_proj", "v_proj")},
    )
    left, right = factors["o_proj"]
    return left @ (right @ context)


def _streamed_covariance_cholesky(inputs, *, chunk_size=512, ridge_ratio=0.01):
    covariance = torch.zeros(
        inputs.shape[0], inputs.shape[0], device=inputs.device, dtype=torch.float32,
    )
    source = inputs.detach().float()
    for start in range(0, source.shape[1], int(chunk_size)):
        chunk = source[:, start:start + int(chunk_size)]
        covariance.addmm_(chunk, chunk.T, alpha=1.0 / source.shape[1])
    return regularized_covariance_cholesky(covariance, ridge_ratio)


@torch.no_grad()
def build_factorized_attention(
    weights,
    fit_inputs,
    dense_attention_targets,
    ranks,
    *,
    n_heads: int,
    n_kv_heads: int,
    seq_len: int,
    construction: str = "function_aware",
    ridge_ratio: float = 0.01,
    seed: int = 0,
    storage_dtype: torch.dtype | None = None,
    fit_input_cholesky: torch.Tensor | None = None,
):
    """Build a compact complete-attention approximation.

    Q/K/V use ordinary or activation-weighted rank factors. The output
    projection is either an independent weight/activation SVD baseline or a
    rank-constrained readout fit to the dense *complete attention output* for
    the approximate context. The output readout is a closed-form initial fit.
    """
    if any(name not in weights for name in ATTENTION_PROJECTIONS):
        raise KeyError("attention weights must include q_proj, k_proj, v_proj, o_proj")
    if fit_inputs.ndim != 2 or dense_attention_targets.ndim != 2:
        raise ValueError("attention fitting tensors must be [features, samples]")
    if fit_inputs.shape[1] != dense_attention_targets.shape[1]:
        raise ValueError("attention inputs and targets must share sample columns")
    ranks = tuple(map(int, ranks))
    if len(ranks) != 4 or any(rank <= 0 for rank in ranks):
        raise ValueError("ranks must contain positive Q/K/V/O ranks")
    if construction not in {
        "function_aware", "module_output_readout", "ordinary_svd", "activation_weighted_svd",
    }:
        raise ValueError("unsupported attention construction")

    mode = "ordinary_svd" if construction in {"ordinary_svd", "module_output_readout"} else "activation_weighted_svd"
    if mode == "activation_weighted_svd" and fit_input_cholesky is None:
        fit_input_cholesky, _ = _streamed_covariance_cholesky(fit_inputs)
    factors = {}
    for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
        factors[name] = _factor(
            weights[name], fit_inputs, ranks[i], mode, seed + i,
            input_cholesky=fit_input_cholesky,
        )
    approximate_context = _projected_context(
        weights, fit_inputs, n_heads, n_kv_heads, seq_len, factors=factors,
    )
    readout_metadata = None
    if construction in {"function_aware", "module_output_readout"}:
        o_left, o_right, readout_metadata = fit_functional_down_readout(
            approximate_context, dense_attention_targets, ranks[3],
            ridge_ratio=ridge_ratio, seed=seed + 3,
        )
        factors["o_proj"] = (o_left, o_right)
    else:
        output_features = (
            _projected_context(weights, fit_inputs, n_heads, n_kv_heads, seq_len)
            if construction == "activation_weighted_svd"
            else fit_inputs
        )
        output_cholesky = (
            _streamed_covariance_cholesky(output_features)[0]
            if construction == "activation_weighted_svd" else None
        )
        factors["o_proj"] = _factor(
            weights["o_proj"], output_features, ranks[3], mode, seed + 3,
            input_cholesky=output_cholesky,
        )

    target_dtype = storage_dtype or weights["q_proj"].dtype
    factors = {
        name: tuple(value.to(target_dtype) for value in pair)
        for name, pair in factors.items()
    }
    coefficient_count = sum(left.numel() + right.numel() for left, right in factors.values())
    expected_count = attention_factor_coefficient_count(weights, ranks)
    if coefficient_count != expected_count:
        raise RuntimeError("attention factor coefficient accounting mismatch")
    return {
        "codec_type": "function_aware_factorized_attention_v1",
        "construction": construction,
        "ranks": ranks,
        "factors": factors,
        "coefficient_count": coefficient_count,
        "dense_residual": False,
        "external_teacher_at_inference": False,
        "post_compression_optimization": False,
        "fit_samples": int(fit_inputs.shape[1]),
        "module_output_readout": readout_metadata,
    }
