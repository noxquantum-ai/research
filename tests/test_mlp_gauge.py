import unittest

import torch
import torch.nn.functional as F

from nox.mlp_gauge import balanced_swiglu_gauge, entanglement_ordered_swiglu_gauge


class MlpGaugeTests(unittest.TestCase):
    def test_balanced_gauge_preserves_swiglu_function(self):
        torch.manual_seed(29)
        weights = {
            "gate_proj": torch.randn(12, 8),
            "up_proj": torch.randn(12, 8) * torch.logspace(-2, 2, 12).unsqueeze(1),
            "down_proj": torch.randn(8, 12),
        }
        inputs = torch.randn(8, 17)
        expected = weights["down_proj"] @ (
            F.silu(weights["gate_proj"] @ inputs) * (weights["up_proj"] @ inputs)
        )
        gauged, metadata = balanced_swiglu_gauge(weights)
        actual = gauged["down_proj"] @ (
            F.silu(gauged["gate_proj"] @ inputs) * (gauged["up_proj"] @ inputs)
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-5)
        self.assertLess(metadata["paired_frobenius_objective_ratio"], 1.0)
        self.assertEqual(metadata["stored_scale_parameters"], 0)

    def test_rejects_incompatible_projection_shapes(self):
        with self.assertRaises(ValueError):
            balanced_swiglu_gauge({"up_proj": torch.randn(3, 2), "down_proj": torch.randn(2, 4)})

    def test_entanglement_ordered_gauge_is_exact_and_budget_free(self):
        torch.manual_seed(11)
        hidden, intermediate, tokens = 12, 24, 17
        weights = {
            "gate_proj": torch.randn(intermediate, hidden),
            "up_proj": torch.randn(intermediate, hidden),
            "down_proj": torch.randn(hidden, intermediate),
        }
        inputs = torch.randn(hidden, tokens)
        expected = weights["down_proj"] @ (
            F.silu(weights["gate_proj"] @ inputs) * (weights["up_proj"] @ inputs)
        )

        transformed, metadata = entanglement_ordered_swiglu_gauge(
            weights, inputs, mode_sizes=(2, 2, 2, 3),
            weight_sketch_size=4, activation_sketch_size=5,
        )
        actual = transformed["down_proj"] @ (
            F.silu(transformed["gate_proj"] @ inputs) * (transformed["up_proj"] @ inputs)
        )

        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(metadata["stored_permutation_parameters"], 0)
        self.assertEqual(metadata["stored_scale_parameters"], 0)
        self.assertEqual(metadata["mode_sizes"], (2, 2, 2, 3))

    def test_entanglement_ordered_gauge_is_deterministic(self):
        torch.manual_seed(12)
        weights = {
            "gate_proj": torch.randn(24, 12),
            "up_proj": torch.randn(24, 12),
            "down_proj": torch.randn(12, 24),
        }
        inputs = torch.randn(12, 16)
        first, first_metadata = entanglement_ordered_swiglu_gauge(
            weights, inputs, mode_sizes=(2, 2, 2, 3),
            weight_sketch_size=3, activation_sketch_size=4,
        )
        second, second_metadata = entanglement_ordered_swiglu_gauge(
            weights, inputs, mode_sizes=(2, 2, 2, 3),
            weight_sketch_size=3, activation_sketch_size=4,
        )

        for name in ("gate_proj", "up_proj", "down_proj"):
            torch.testing.assert_close(first[name], second[name])
        self.assertEqual(first_metadata, second_metadata)


if __name__ == "__main__":
    unittest.main()
