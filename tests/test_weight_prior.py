import unittest
from unittest.mock import patch

import torch

from nox.compiler import augment_diagonal_weight_prior
from nox.block_compiler import compile_transformer_block
from scripts.audit_projection_fit import projection_inputs


class WeightPriorTests(unittest.TestCase):
    def test_full_block_passes_prior_and_keeps_dense_no_healing_inputs(self):
        generator = torch.Generator().manual_seed(17)
        weights = {name: torch.randn(16, 16, generator=generator) * 0.1
                   for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
        weights.update(input_layernorm=torch.ones(16), post_attention_layernorm=torch.ones(16))
        x = torch.randn(16, 8, generator=generator)
        expected = projection_inputs(weights, x, 1e-5, 4, 2, 2)
        for artifacts in (False, True):
            calls = []

            def fake_fit(**kwargs):
                name = next(name for name in expected if weights[name] is kwargs["W"])
                self.assertEqual(kwargs["diagonal_prior_strength"], 1.0)
                torch.testing.assert_close(kwargs["X_calib"], expected[name])
                torch.testing.assert_close(kwargs["Y_dense"], weights[name] @ expected[name])
                calls.append(name)
                result = (kwargs["W"] * 0.5, 0.0, [], [])
                return result + ({},) if kwargs.get("return_artifacts") else result

            with patch("nox.block_compiler.nox_hybrid_dmrg_decompose", side_effect=fake_fit):
                compile_transformer_block(weights, x, n_heads=2, n_kv_heads=2, seq_len=4,
                                          causal_healing=False, return_artifacts=artifacts,
                                          verbose=False, diagonal_prior_strength=1.0)
            self.assertEqual(set(calls), set(expected))
            self.assertEqual(len(calls), 7)

    def test_augmented_loss_equals_fixed_dense_weight_penalty(self):
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(5, 3, generator=generator, dtype=torch.float64)
        w = torch.randn(4, 5, generator=generator, dtype=torch.float64)
        fitted = torch.randn(4, 5, generator=generator, dtype=torch.float64)
        y = w @ x
        scale = x.square().mean(dim=1).sqrt()
        strength = 1.0
        xa, ya = augment_diagonal_weight_prior(x, y, w, scale, strength)
        expected = (fitted @ x - y).square().sum()
        expected += x.shape[1] * strength * ((fitted - w) * scale).square().sum()
        torch.testing.assert_close((fitted @ xa - ya).square().sum(), expected)
        self.assertEqual(torch.linalg.matrix_rank(xa), x.shape[0])

    def test_disabled_prior_preserves_original_tensors(self):
        x, y, w, scale = torch.ones(2, 1), torch.ones(3, 1), torch.ones(3, 2), torch.ones(2)
        xa, ya = augment_diagonal_weight_prior(x, y, w, scale, 0)
        self.assertIs(xa, x)
        self.assertIs(ya, y)
        with self.assertRaises(ValueError):
            augment_diagonal_weight_prior(x, y, w, scale, -1)
