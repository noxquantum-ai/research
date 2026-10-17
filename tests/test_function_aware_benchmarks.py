import torch

from nox.block_compiler import scaled_dot_product_attention_2d
from nox.function_aware_benchmarks import (
    benchmark_attention_module,
    benchmark_swiglu_module,
)


def test_matched_budget_functional_mlp_benchmark_has_selection_and_all_controls():
    generator = torch.Generator().manual_seed(701)
    d_model, intermediate = 16, 24
    weights = {
        "gate_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "up_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "down_proj": torch.randn(d_model, intermediate, generator=generator) / 4,
    }
    fit = torch.randn(d_model, 64, generator=generator)
    selection = torch.randn(d_model, 32, generator=generator)
    heldout = torch.randn(d_model, 32, generator=generator)
    result = benchmark_swiglu_module(
        weights, fit, selection, heldout,
        budgets=(0.5,), include_grouped=True, seed=702,
    )
    candidates = result["budgets"]["0.5"]["candidates"]
    assert {
        "ordinary_svd", "activation_weighted_svd", "function_aware",
        "modegpt_inspired_shared_neuron_selection", "previous_grouped_tn",
    } <= set(candidates)
    for candidate in candidates.values():
        assert candidate["coefficient_count"] <= result["budgets"]["0.5"]["coefficient_budget"]
        assert candidate["metrics"]["heldout"]["mlp_output"]["relative_error"] >= 0
    assert candidates["function_aware"]["internal_selection"]["selection_trials"]


def test_complete_attention_benchmark_reports_module_and_residual_outputs():
    generator = torch.Generator().manual_seed(703)
    d_model = 16
    weights = {
        name: torch.randn(d_model, d_model, generator=generator) / 4
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }

    def attention_output(inputs):
        q, k, v = (weights[name] @ inputs for name in ("q_proj", "k_proj", "v_proj"))
        return weights["o_proj"] @ scaled_dot_product_attention_2d(
            q, k, v, n_heads=4, n_kv_heads=4, seq_len=4,
        )

    fit = torch.randn(d_model, 64, generator=generator)
    selection = torch.randn(d_model, 32, generator=generator)
    heldout = torch.randn(d_model, 32, generator=generator)
    result = benchmark_attention_module(
        weights, fit, attention_output(fit), selection, attention_output(selection),
        heldout, attention_output(heldout), n_heads=4, n_kv_heads=4,
        seq_len=4, fit_residual=fit, selection_residual=selection,
        heldout_residual=heldout,
        downstream_fn=lambda x: 2 * x,
        budgets=(0.5,), seed=704,
    )
    candidates = result["budgets"]["0.5"]["candidates"]
    assert "function_aware" in candidates
    metrics = candidates["function_aware"]["metrics"]["heldout"]
    assert "attention_output" in metrics
    assert "full_residual_block_output" in metrics
