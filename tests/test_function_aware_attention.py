import torch

from nox.block_compiler import scaled_dot_product_attention_2d
from nox.function_aware_attention import (
    apply_factorized_attention,
    attention_factor_coefficient_count,
    build_factorized_attention,
)


def test_function_aware_attention_uses_complete_output_readout_under_budget():
    generator = torch.Generator().manual_seed(601)
    d_model = 16
    weights = {
        name: torch.randn(d_model, d_model, generator=generator) / 4
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    fit = torch.randn(d_model, 64, generator=generator)
    heldout = torch.randn(d_model, 32, generator=generator)

    def dense(x):
        q, k, v = (weights[name] @ x for name in ("q_proj", "k_proj", "v_proj"))
        context = scaled_dot_product_attention_2d(q, k, v, 4, 4, 4)
        return weights["o_proj"] @ context

    artifact = build_factorized_attention(
        weights, fit, dense(fit), (3, 3, 3, 3), n_heads=4,
        n_kv_heads=4, seq_len=4, construction="function_aware",
        storage_dtype=torch.float32, seed=602,
    )
    candidate = apply_factorized_attention(
        artifact, heldout, n_heads=4, n_kv_heads=4, seq_len=4,
    )
    assert artifact["coefficient_count"] == attention_factor_coefficient_count(weights, (3, 3, 3, 3))
    assert artifact["dense_residual"] is False
    assert candidate.shape == dense(heldout).shape
    assert torch.isfinite(candidate).all()


def test_attention_module_output_readout_is_distinct_from_weight_svd():
    generator = torch.Generator().manual_seed(603)
    d_model = 16
    weights = {
        name: torch.randn(d_model, d_model, generator=generator) / 4
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    fit = torch.randn(d_model, 48, generator=generator)
    q, k, v = (weights[name] @ fit for name in ("q_proj", "k_proj", "v_proj"))
    target = weights["o_proj"] @ scaled_dot_product_attention_2d(q, k, v, 4, 4, 4)
    artifact = build_factorized_attention(
        weights, fit, target, (2, 2, 2, 2), n_heads=4,
        n_kv_heads=4, seq_len=4, construction="function_aware",
        storage_dtype=torch.float32,
    )
    assert artifact["module_output_readout"]["objective"].startswith("mean_module_output")
