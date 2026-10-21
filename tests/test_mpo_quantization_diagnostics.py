import unittest

import torch

from benchmarks.analyze_projection_waterfall import (
    core_quantization_waterfall,
    output_metrics,
)
from benchmarks.diagnose_mpo_gauge_quantization import (
    apply_bond_gauges,
    left_canonicalize,
    quantize_cores,
    right_canonicalize,
    run_diagnostic,
)
from nox.compiler import quantize_4bit_grouped
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

    def test_core_waterfall_holds_non_mpo_operator_fixed(self):
        ranks = (1, 2, 3, 2, 1)
        cores = [
            torch.randn(ranks[index], 2, 2, ranks[index + 1])
            for index in range(4)
        ]
        d_awt = torch.rand(16).add_(0.5)
        fixed = torch.randn(16, 16)
        pre_q4 = fixed + mpo_to_dense(cores) / d_awt.unsqueeze(0)
        dense = torch.randn(16, 16)
        calibration = torch.randn(16, 19)
        validation = torch.randn(16, 23)
        quantized = [
            quantize_4bit_grouped(core, group_size=64) for core in cores
        ]
        final = fixed + mpo_to_dense(quantized) / d_awt.unsqueeze(0)

        result = core_quantization_waterfall(
            dense, pre_q4, final, cores, d_awt, calibration, validation,
        )

        final_delta = result["all_q4_to_serialized_artifact"]
        self.assertLess(final_delta["relative_weight_difference"], 1e-6)
        self.assertLess(final_delta["calibration"]["relative_error"], 1e-6)
        self.assertLess(final_delta["validation"]["relative_error"], 1e-6)


if __name__ == "__main__":
    unittest.main()
