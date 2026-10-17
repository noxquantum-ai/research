import torch
import torch.nn.functional as F

from nox.function_aware_tn import (
    activation_weighted_svd_factors,
    apply_factorized_swiglu,
    apply_selected_swiglu,
    budgeted_swiglu_rank,
    build_factorized_swiglu,
    build_selected_swiglu,
    fit_functional_down_readout,
    ordinary_svd_factors,
    swiglu_dense_coefficient_count,
    swiglu_factor_coefficient_count,
)


def _rel(a, b):
    return torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(a.float()).clamp_min(1e-30)


def test_factorized_parameter_accounting_and_requested_budgets():
    result = budgeted_swiglu_rank(8, 12, 0.5)
    assert result["dense_coefficients"] == swiglu_dense_coefficient_count(8, 12)
    assert result["ranks"] == (3, 2, 2)
    assert result["coefficient_count"] == swiglu_factor_coefficient_count(8, 12, (3, 2, 2))
    assert result["actual_fraction"] <= result["fraction"]


def test_activation_weighted_factor_shapes_and_finite_reconstruction():
    generator = torch.Generator().manual_seed(411)
    weight = torch.randn(12, 8, generator=generator)
    inputs = torch.randn(8, 32, generator=generator)
    left, right = activation_weighted_svd_factors(weight, inputs, 3)
    assert left.shape == (12, 3)
    assert right.shape == (3, 8)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()


def test_functional_down_readout_beats_weight_svd_on_its_fit_function():
    generator = torch.Generator().manual_seed(412)
    d_model, intermediate, samples = 8, 12, 48
    gate = torch.randn(intermediate, d_model, generator=generator) / 3
    up = torch.randn(intermediate, d_model, generator=generator) / 3
    down = torch.randn(d_model, intermediate, generator=generator) / 3
    x = torch.randn(d_model, samples, generator=generator)
    features = F.silu(gate @ x) * (up @ x)
    targets = down @ features
    left, right, metadata = fit_functional_down_readout(features, targets, 3, ridge_ratio=0.0)
    fitted = left @ (right @ features)
    svd_left, svd_right = ordinary_svd_factors(down, 3)
    weight_only = svd_left @ (svd_right @ features)
    assert metadata["uses_gradients"] is False
    assert torch.mean((targets - fitted).square()) <= torch.mean((targets - weight_only).square()) + 1e-6


def test_candidate_is_compact_factor_only_and_applies_full_swiglu():
    generator = torch.Generator().manual_seed(413)
    d_model, intermediate = 8, 12
    gate = torch.randn(intermediate, d_model, generator=generator) / 3
    up = torch.randn(intermediate, d_model, generator=generator) / 3
    down = torch.randn(d_model, intermediate, generator=generator) / 3
    fit = torch.randn(d_model, 64, generator=generator)
    heldout = torch.randn(d_model, 19, generator=generator)
    artifact = build_factorized_swiglu(
        gate, up, down, fit, (3, 3, 3), construction="function_aware",
        storage_dtype=torch.float32,
    )
    reference = down @ (F.silu(gate @ heldout) * (up @ heldout))
    estimate = apply_factorized_swiglu(artifact, heldout)
    assert artifact["coefficient_count"] == 3 * (intermediate + d_model) * 3
    assert artifact["dense_residual"] is False
    assert artifact["external_teacher_at_inference"] is False
    assert estimate.shape == reference.shape
    assert torch.isfinite(estimate).all()
    assert _rel(reference, estimate) >= 0


def test_modegpt_inspired_shared_neuron_selection_respects_budget():
    generator = torch.Generator().manual_seed(414)
    d_model, intermediate = 8, 12
    weights = [torch.randn(intermediate, d_model, generator=generator) / 4 for _ in range(2)]
    down = torch.randn(d_model, intermediate, generator=generator) / 4
    fit = torch.randn(d_model, 48, generator=generator)
    budget = int(3 * d_model * intermediate * 0.5)
    artifact = build_selected_swiglu(
        weights[0], weights[1], down, fit, budget, ridge_ratio=0.01,
        storage_dtype=torch.float32,
    )
    estimate = apply_selected_swiglu(artifact, fit)
    assert artifact["coefficient_count"] <= budget
    assert artifact["coefficient_count"] == sum(
        artifact[name].numel() for name in ("gate_proj", "up_proj", "down_proj")
    )
    assert artifact["dense_residual"] is False
    assert artifact["selection_method"].startswith("top diagonal")
    assert estimate.shape == fit.shape
    assert torch.isfinite(estimate).all()
