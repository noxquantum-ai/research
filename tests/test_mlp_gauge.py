import unittest

import torch
import torch.nn.functional as F

from nox.mlp_gauge import balanced_swiglu_gauge


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


if __name__ == "__main__":
    unittest.main()
