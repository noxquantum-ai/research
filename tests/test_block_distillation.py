import unittest

import torch

from nox.block_distillation import (
    fake_quantize_ste,
    forward_block_with_grad,
    make_trainable_components,
    reconstruct_trainable_artifact,
)
from nox.compiler import quantize_4bit_grouped, reconstruct_from_artifact
from nox.sequential_compiler import forward_transformer_block


class BlockDistillationTests(unittest.TestCase):
    def test_fake_quantization_preserves_int4_forward_and_straight_through_gradient(self):
        value = torch.linspace(-2.0, 2.0, 130, requires_grad=True)
        forward = fake_quantize_ste(value)

        self.assertTrue(torch.equal(forward.detach(), fake_quantize_ste(value.detach())))
        forward.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_differentiable_block_matches_existing_forward_and_trains_weights(self):
        torch.manual_seed(7)
        d_model, tokens = 8, 4
        weights = {
            name: torch.randn(d_model, d_model)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        }
        weights["q_proj"].requires_grad_()
        weights["input_layernorm"] = torch.ones(d_model)
        weights["post_attention_layernorm"] = torch.ones(d_model)
        x = torch.randn(d_model, tokens)

        expected, _, _ = forward_transformer_block(
            weights, x, n_heads=2, n_kv_heads=2, seq_len=tokens, rms_norm_eps=1e-5
        )
        actual = forward_block_with_grad(
            weights, x, n_heads=2, n_kv_heads=2, seq_len=tokens, rms_norm_eps=1e-5
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

        actual.square().mean().backward()
        self.assertIsNotNone(weights["q_proj"].grad)
        self.assertGreater(weights["q_proj"].grad.abs().sum().item(), 0.0)

    def test_trainable_artifact_matches_saved_int4_factors_and_has_gradients(self):
        shapes = ((1, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 1))
        artifact = {
            "quantization_bits": 4,
            "cores": [quantize_4bit_grouped(torch.randn(shape)).half() for shape in shapes],
            "A_bridge": quantize_4bit_grouped(torch.randn(16, 2)).half(),
            "B_bridge": quantize_4bit_grouped(torch.randn(2, 16)).half(),
            "d_awt": torch.ones(16, dtype=torch.float16),
            "sparse_indices": torch.tensor([[0, 0], [15, 15]], dtype=torch.int32),
            "sparse_values": torch.tensor([0.25, -0.5], dtype=torch.float16),
            "orig_shape": (16, 16),
        }
        components = make_trainable_components(artifact, "cpu")

        trainable = reconstruct_trainable_artifact(artifact, components, "cpu")
        saved = reconstruct_from_artifact(artifact, device="cpu", dtype=torch.float32)
        relative_error = (trainable - saved).norm().item() / saved.norm().item()
        self.assertTrue(
            relative_error < 1e-3,
            f"Initial trainable reconstruction differs: rel={relative_error:.6g}",
        )
        trainable.square().sum().backward()
        self.assertTrue(all(core.grad is not None for core in components["cores"]))
        self.assertGreater(components["A_bridge"].grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
