import math

import torch
import torch.nn.functional as F

from nox.representation_symmetries import (
    llama_attention_output,
    rope_commuting_qk_gauge,
    signed_permutation_residual_gauge,
    swiglu_neuron_permutation_gauge,
    swiglu_up_down_scaling_gauge,
    value_output_gauge,
)


def _attention_weights(dtype=torch.float32):
    generator = torch.Generator().manual_seed(311)
    return {
        "q_proj": torch.randn(32, 32, generator=generator, dtype=dtype) / 8,
        "k_proj": torch.randn(16, 32, generator=generator, dtype=dtype) / 8,
        "v_proj": torch.randn(16, 32, generator=generator, dtype=dtype) / 8,
        "o_proj": torch.randn(32, 32, generator=generator, dtype=dtype) / 8,
    }


def _relative(reference, candidate):
    return torch.linalg.vector_norm(candidate.float() - reference.float()) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)


def test_rope_commuting_qk_gauge_preserves_gqa_attention_fp32():
    weights = _attention_weights()
    generator = torch.Generator().manual_seed(312)
    inputs = torch.randn(2, 11, 32, generator=generator) / 3
    scales = torch.exp(torch.randn(2, 4, generator=generator) * 0.25)
    angles = torch.randn(2, 4, generator=generator) * 0.4
    transformed = rope_commuting_qk_gauge(
        weights, 4, 2, pair_scales=scales, pair_angles=angles,
    )
    before = llama_attention_output(weights, inputs, n_heads=4, n_kv_heads=2)
    after = llama_attention_output(transformed, inputs, n_heads=4, n_kv_heads=2)
    assert _relative(before, after) < 1e-5


def test_value_output_gauge_preserves_gqa_attention_fp32():
    weights = _attention_weights()
    generator = torch.Generator().manual_seed(313)
    inputs = torch.randn(2, 13, 32, generator=generator) / 3
    scales = torch.exp(torch.randn(2, 4, generator=generator) * 0.2)
    angles = torch.randn(2, 4, generator=generator) * 0.5
    transformed = value_output_gauge(
        weights, 4, 2, pair_scales=scales, pair_angles=angles,
    )
    before = llama_attention_output(weights, inputs, n_heads=4, n_kv_heads=2)
    after = llama_attention_output(transformed, inputs, n_heads=4, n_kv_heads=2)
    assert _relative(before, after) < 1e-5


def test_rope_commuting_qk_and_value_output_gauges_preserve_fp16_path():
    weights = {name: value.half() for name, value in _attention_weights().items()}
    generator = torch.Generator().manual_seed(314)
    inputs = (torch.randn(2, 9, 32, generator=generator) / 3).half()
    scales = torch.exp(torch.randn(2, 4, generator=generator) * 0.1)
    angles = torch.randn(2, 4, generator=generator) * 0.2
    transformed = rope_commuting_qk_gauge(
        weights, 4, 2, pair_scales=scales, pair_angles=angles,
    )
    transformed = value_output_gauge(
        transformed, 4, 2, pair_scales=scales, pair_angles=angles,
    )
    before = llama_attention_output(weights, inputs, n_heads=4, n_kv_heads=2)
    after = llama_attention_output(transformed, inputs, n_heads=4, n_kv_heads=2)
    assert _relative(before, after) < 5e-3


def test_signed_permutation_preserves_rmsnorm_residual_swiglu_block():
    generator = torch.Generator().manual_seed(315)
    d_model, intermediate = 16, 24
    weights = {
        "gate_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "up_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "down_proj": torch.randn(d_model, intermediate, generator=generator) / 4,
        "input_layernorm": torch.rand(d_model, generator=generator) + 0.5,
        "post_attention_layernorm": torch.rand(d_model, generator=generator) + 0.5,
    }
    x = torch.randn(d_model, 7, generator=generator)
    permutation = torch.randperm(d_model, generator=generator)
    signs = torch.where(torch.rand(d_model, generator=generator) > 0.5, 1.0, -1.0)
    transformed = signed_permutation_residual_gauge(weights, permutation, signs)

    def block(w, value, gamma):
        normalized = value * torch.rsqrt(value.square().mean(0, keepdim=True) + 1e-5)
        normalized = normalized * gamma[:, None]
        hidden = F.silu(w["gate_proj"] @ normalized) * (w["up_proj"] @ normalized)
        return value + w["down_proj"] @ hidden

    before = block(weights, x, weights["post_attention_layernorm"])
    x_new = signs[:, None] * x.index_select(0, permutation)
    after = block(
        transformed, x_new, transformed["post_attention_layernorm"],
    )
    expected = signs[:, None] * before.index_select(0, permutation)
    assert _relative(expected, after) < 1e-5


def test_signed_permutation_folds_consistently_into_embeddings_final_norm_and_lm_head():
    generator = torch.Generator().manual_seed(319)
    d_model, vocab = 16, 23
    weights = {
        "embed_tokens": torch.randn(vocab, d_model, generator=generator),
        "lm_head": torch.randn(vocab, d_model, generator=generator),
        "final_norm": torch.rand(d_model, generator=generator) + 0.5,
    }
    permutation = torch.randperm(d_model, generator=generator)
    signs = torch.where(torch.rand(d_model, generator=generator) > 0.5, 1.0, -1.0)
    transformed = signed_permutation_residual_gauge(weights, permutation, signs)
    token_ids = torch.tensor([0, 3, 9, 22])
    original_embedding = weights["embed_tokens"][token_ids]
    changed_embedding = transformed["embed_tokens"][token_ids]
    expected_embedding = signs[None, :] * original_embedding.index_select(1, permutation)
    assert torch.allclose(changed_embedding, expected_embedding, rtol=0, atol=0)

    hidden = torch.randn(5, d_model, generator=generator)
    hidden_new = signs * hidden.index_select(-1, permutation)
    normed = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-5)
    normed = normed * weights["final_norm"]
    normed_new = hidden_new * torch.rsqrt(hidden_new.square().mean(-1, keepdim=True) + 1e-5)
    normed_new = normed_new * transformed["final_norm"]
    logits = F.linear(normed, weights["lm_head"])
    logits_new = F.linear(normed_new, transformed["lm_head"])
    assert _relative(logits, logits_new) < 1e-6


def test_swiglu_permutation_and_reciprocal_scaling_preserve_outputs_fp32():
    generator = torch.Generator().manual_seed(316)
    d_model, intermediate = 16, 24
    weights = {
        "gate_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "up_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "down_proj": torch.randn(d_model, intermediate, generator=generator) / 4,
    }
    x = torch.randn(d_model, 37, generator=generator)

    def mlp(w):
        return w["down_proj"] @ (
            F.silu(w["gate_proj"] @ x) * (w["up_proj"] @ x)
        )

    scales = torch.exp(torch.randn(intermediate, generator=generator) * 0.3)
    transformed = swiglu_up_down_scaling_gauge(weights, scales)
    permutation = torch.randperm(intermediate, generator=generator)
    transformed = swiglu_neuron_permutation_gauge(transformed, permutation)
    assert _relative(mlp(weights), mlp(transformed)) < 1e-5


def test_gate_scaling_is_not_misclassified_as_a_swiglu_exact_gauge():
    generator = torch.Generator().manual_seed(320)
    d_model, intermediate = 16, 24
    weights = {
        "gate_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "up_proj": torch.randn(intermediate, d_model, generator=generator) / 4,
        "down_proj": torch.randn(d_model, intermediate, generator=generator) / 4,
    }
    inputs = torch.randn(d_model, 64, generator=generator)

    def output(candidate):
        return candidate["down_proj"] @ (
            F.silu(candidate["gate_proj"] @ inputs)
            * (candidate["up_proj"] @ inputs)
        )

    candidate = dict(weights)
    candidate["gate_proj"] = weights["gate_proj"] * torch.linspace(0.5, 1.5, intermediate)[:, None]
    assert _relative(output(weights), output(candidate)) > 1e-3


def test_swiglu_permutation_and_reciprocal_scaling_preserve_fp16_path():
    generator = torch.Generator().manual_seed(317)
    d_model, intermediate = 16, 24
    weights = {
        "gate_proj": (torch.randn(intermediate, d_model, generator=generator) / 4).half(),
        "up_proj": (torch.randn(intermediate, d_model, generator=generator) / 4).half(),
        "down_proj": (torch.randn(d_model, intermediate, generator=generator) / 4).half(),
    }
    x = (torch.randn(d_model, 37, generator=generator) / 3).half()

    def mlp(w):
        return w["down_proj"] @ (
            F.silu(w["gate_proj"] @ x) * (w["up_proj"] @ x)
        )

    scales = torch.exp(torch.randn(intermediate, generator=generator) * 0.1)
    transformed = swiglu_up_down_scaling_gauge(weights, scales)
    transformed = swiglu_neuron_permutation_gauge(
        transformed, torch.randperm(intermediate, generator=generator),
    )
    assert _relative(mlp(weights), mlp(transformed)) < 5e-3


def test_signed_permutation_preserves_full_gqa_attention_and_swiglu_residual_block():
    generator = torch.Generator().manual_seed(318)
    d_model, intermediate = 32, 48
    weights = _attention_weights()
    weights.update({
        "gate_proj": torch.randn(intermediate, d_model, generator=generator) / 6,
        "up_proj": torch.randn(intermediate, d_model, generator=generator) / 6,
        "down_proj": torch.randn(d_model, intermediate, generator=generator) / 6,
        "input_layernorm": torch.rand(d_model, generator=generator) + 0.5,
        "post_attention_layernorm": torch.rand(d_model, generator=generator) + 0.5,
    })
    x = torch.randn(2, 7, d_model, generator=generator) / 3
    permutation = torch.randperm(d_model, generator=generator)
    signs = torch.where(torch.rand(d_model, generator=generator) > 0.5, 1.0, -1.0)
    transformed = signed_permutation_residual_gauge(weights, permutation, signs)
    x_new = signs[None, None, :] * x.index_select(-1, permutation)

    def block(w, value):
        normed = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        normed = normed * w["input_layernorm"]
        attention = llama_attention_output(
            {name: w[name] for name in ("q_proj", "k_proj", "v_proj", "o_proj")},
            normed, n_heads=4, n_kv_heads=2,
        )
        after_attention = value + attention
        mlp_input = after_attention * torch.rsqrt(
            after_attention.square().mean(-1, keepdim=True) + 1e-5,
        )
        mlp_input = mlp_input * w["post_attention_layernorm"]
        mlp = F.linear(
            F.silu(F.linear(mlp_input, w["gate_proj"]))
            * F.linear(mlp_input, w["up_proj"]),
            w["down_proj"],
        )
        return after_attention + mlp

    before = block(weights, x)
    after = block(transformed, x_new)
    expected = signs[None, None, :] * before.index_select(-1, permutation)
    assert _relative(expected, after) < 1e-5


def test_attention_gauge_rejects_malformed_pair_coordinates():
    weights = _attention_weights()
    try:
        rope_commuting_qk_gauge(weights, 4, 2, pair_scales=torch.ones(2, 3))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("malformed RoPE pair map was accepted")
