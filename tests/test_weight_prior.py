import unittest

import torch

from nox.compiler import augment_diagonal_weight_prior


class WeightPriorTests(unittest.TestCase):
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
