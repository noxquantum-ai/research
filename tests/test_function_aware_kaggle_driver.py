import torch
import torch.nn.functional as F

from kaggle_execution.function_aware_tn.driver import (
    dense_module_data,
    evaluate_stage_a_gate,
    exact_gauge_scales,
    sample_windows,
    stream_forward_block,
)
from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d
from nox.sequential_compiler import forward_transformer_block


def test_kaggle_driver_streamed_forward_and_module_capture_match_dense_block():
    generator = torch.Generator().manual_seed(801)
    d_model, intermediate = 32, 48
    weights = {
        "q_proj": torch.randn(d_model, d_model, generator=generator) / 6,
        "k_proj": torch.randn(16, d_model, generator=generator) / 6,
        "v_proj": torch.randn(16, d_model, generator=generator) / 6,
        "o_proj": torch.randn(d_model, d_model, generator=generator) / 6,
        "gate_proj": torch.randn(intermediate, d_model, generator=generator) / 6,
        "up_proj": torch.randn(intermediate, d_model, generator=generator) / 6,
        "down_proj": torch.randn(d_model, intermediate, generator=generator) / 6,
        "input_layernorm": torch.rand(d_model, generator=generator) + 0.5,
        "post_attention_layernorm": torch.rand(d_model, generator=generator) + 0.5,
    }
    inputs = torch.randn(d_model, 256, generator=generator) / 3
    dense_output = forward_transformer_block(
        weights, inputs, n_heads=4, n_kv_heads=2, seq_len=128,
    )[0]
    streamed_output = stream_forward_block(
        weights, inputs, forward_transformer_block,
        n_heads=4, n_kv_heads=2, seq_len=128, chunk_tokens=128,
    )
    captured = dense_module_data(
        weights, inputs, 4, 2, 128, RMSNorm,
        scaled_dot_product_attention_2d, F, chunk_tokens=128,
    )
    attn_input, attn_output, after_attention, mlp_input = captured
    mlp_output = weights["down_proj"] @ (
        F.silu(weights["gate_proj"] @ mlp_input)
        * (weights["up_proj"] @ mlp_input)
    )
    assert torch.allclose(streamed_output, dense_output, rtol=1e-5, atol=1e-5)
    assert torch.allclose(after_attention + mlp_output, dense_output, rtol=1e-5, atol=1e-5)
    assert attn_input.shape == attn_output.shape == mlp_input.shape == inputs.shape


def test_kaggle_driver_legal_gauge_scales_have_expected_head_shapes():
    generator = torch.Generator().manual_seed(802)
    weights = {
        "q_proj": torch.randn(32, 32, generator=generator),
        "k_proj": torch.randn(16, 32, generator=generator),
        "v_proj": torch.randn(16, 32, generator=generator),
        "o_proj": torch.randn(32, 32, generator=generator),
        "up_proj": torch.randn(48, 32, generator=generator),
        "down_proj": torch.randn(32, 48, generator=generator),
    }
    scales = exact_gauge_scales(weights, 4, 2)
    assert scales["qk_scales"].shape == (2, 4)
    assert scales["vo_scales"].shape == (2, 4)
    assert scales["updown_scales"].shape == (48,)
    assert torch.isfinite(scales["qk_angles"]).all()
    assert torch.isfinite(scales["vo_angles"]).all()


def test_seeded_window_sampler_is_reproducible_and_keeps_sequence_boundaries():
    tokens = list(range(10000))
    first = sample_windows(tokens, 1024, 29)
    second = sample_windows(tokens, 1024, 29)
    assert first == second
    assert len(first) == 1024
    assert len(set(first)) == len(first)
    assert all(window[0] % 128 == 0 for window in [first[index:index + 128] for index in range(0, 1024, 128)])


def test_stage_a_does_not_mix_incompatible_gauges_operator_by_operator():
    def record(count, orders=("0123",)):
        return {"tensorizations": {
            order: {"coefficient_distortion_targets": {"10": {
                "estimated_target_met": True,
                "spectrum_complete_for_target": True,
                "tt_independent_coefficients": count,
            }}}
            for order in orders
        }}

    spectra = {
        f"original_nec_identity/{name}": record(100, ("0123", "3210", "0213"))
        for name in ("q_proj", "k_proj")
    }
    spectra["rope_commuting_qk_gauge/q_proj"] = record(50)
    mixed = evaluate_stage_a_gate(spectra, 15)
    assert not mixed["provisional_stage_a_pass_for_this_layer"]
    spectra["rope_commuting_qk_gauge/k_proj"] = record(50)
    coherent = evaluate_stage_a_gate(spectra, 15)
    assert coherent["provisional_stage_a_pass_for_this_layer"]
    assert "qk_rope_commutant" in coherent["passing_groups"]
