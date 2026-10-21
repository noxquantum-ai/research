import unittest

import torch

from benchmarks.analyze_projection_waterfall import output_metrics
from benchmarks.diagnose_mpo_gauge_quantization import (
    apply_bond_gauges,
    left_canonicalize,
    quantize_cores,
    right_canonicalize,
    run_diagnostic,
)
from nox.utils import mpo_to_dense


class MPOQuantizationDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(700)
        ranks = (1, 3, 4, 3, 1)
        self.cores = [
            torch.randn(ranks[i], 2, 2, ranks[i + 1])
            for i in range(4)
        ]

    def test_invertible_bond_gauges_preserve_operator(self):
        gauges = []
        for core in self.cores[:-1]:
            size = core.shape[-1]
            matrix = torch.randn(size, size)
            gauges.append(matrix + 2.0 * torch.eye(size))
        transformed = apply_bond_gauges(self.cores, gauges)
        torch.testing.assert_close(
            mpo_to_dense(transformed), mpo_to_dense(self.cores),
            atol=2e-4, rtol=2e-4,
        )

    def test_canonicalizations_preserve_operator(self):
        reference = mpo_to_dense(self.cores)
        torch.testing.assert_close(
            mpo_to_dense(left_canonicalize(self.cores)), reference,
            atol=2e-4, rtol=2e-4,
        )
        torch.testing.assert_close(
            mpo_to_dense(right_canonicalize(self.cores)), reference,
            atol=2e-4, rtol=2e-4,
        )

    def test_selected_gauge_uses_calibration_objective(self):
        report = run_diagnostic(seed=701)
        selected = report["selected_candidate"]
        minimum = min(
            item["post_q4"]["calibration_relative_error"]
            for item in report["gauge_candidates"]
        )
        self.assertAlmostEqual(
            selected["post_q4"]["calibration_relative_error"], minimum,
        )
        self.assertEqual(set(report["per_core_q4"]), {
            "core_0", "core_1", "core_2", "core_3",
        })

    def test_output_metric_matches_direct_definition(self):
        reference = torch.eye(4)
        candidate = 1.5 * reference
        inputs = torch.randn(4, 17)
        metrics = output_metrics(reference, candidate, inputs, token_chunk=5)
        self.assertAlmostEqual(metrics["relative_error"], 0.5)
        self.assertAlmostEqual(metrics["norm_ratio"], 1.5)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)

    def test_quantize_selected_core_leaves_shapes_unchanged(self):
        result = quantize_cores(self.cores, {2})
        self.assertEqual([core.shape for core in result], [core.shape for core in self.cores])
        torch.testing.assert_close(result[0], self.cores[0])
        self.assertFalse(torch.equal(result[2], self.cores[2]))


if __name__ == "__main__":
    unittest.main()
