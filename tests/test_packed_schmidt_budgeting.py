import unittest

import torch

from nox.operator_schmidt import (
    OperatorSchmidtLayout, _matrix_to_unfolding, operator_schmidt_to_dense,
)
from nox.packed_schmidt import (
    _core_from_input_metric,
    _input_mode_cholesky_factors,
    _matrix_in_input_metric,
    compile_packed_schmidt,
    packed_matrix_coefficient_count,
    packed_schmidt_coefficient_count,
    packed_schmidt_coefficient_count_from_artifact,
    rank_for_coefficient_budget,
    reconstruct_packed_schmidt,
)
from nox.residual_codec import encode_groupwise


class PackedSchmidtStructuralBudgetTests(unittest.TestCase):
    def test_mode_metric_is_absorbed_into_either_tensor_cut(self):
        generator = torch.Generator().manual_seed(53)
        weight = torch.randn(12, 24, generator=generator)
        inputs = torch.randn(24, 80, generator=generator)
        factors = _input_mode_cholesky_factors(
            inputs, (4, 6), ridge_ratio=0.01,
        )
        weighted = _matrix_in_input_metric(weight, factors)
        torch.testing.assert_close(weighted, weight @ torch.kron(*factors))
        for axes in ((0, 2, 1, 3), (0, 3, 1, 2)):
            layout = OperatorSchmidtLayout("metric_test", (3, 4), (4, 6), axes)
            u, values, vh = torch.linalg.svd(
                _matrix_to_unfolding(weighted, layout), full_matrices=False,
            )
            left, right = (u * values.sqrt()).T, values.sqrt()[:, None] * vh
            core_factors = factors if axes == (0, 2, 1, 3) else factors[::-1]
            left = _core_from_input_metric(left, core_factors[0])
            right = _core_from_input_metric(right, core_factors[1])
            torch.testing.assert_close(
                operator_schmidt_to_dense(left, right, layout), weight,
                atol=8e-6, rtol=8e-6,
            )

    def test_compiled_defects_roundtrip_and_stay_inside_both_budgets(self):
        generator = torch.Generator().manual_seed(41)
        weight = torch.randn(32, 32, generator=generator)
        inputs = torch.randn(32, 64, generator=generator)
        artifact, decoded, _ = compile_packed_schmidt(
            weight, byte_budget=2200, coefficient_budget=1600,
            calibration_inputs=inputs, bits_candidates=(3, 4),
            group_size=4, als_sweeps=0,
            calibration_branch_fraction=0.1,
            calibration_branch_fit_tokens=32,
            column_branch_fraction=0.2,
            complementary_branch_fraction=0.2,
            input_metric="product_covariance",
        )
        self.assertIn("complementary_schmidt_branch", artifact)
        self.assertIn("calibration_column_branch", artifact)
        self.assertIn("calibration_schmidt_branch", artifact)
        self.assertTrue(artifact["sparse_sidecar"])
        torch.testing.assert_close(reconstruct_packed_schmidt(artifact), decoded)
        self.assertLessEqual(artifact["payload_bytes"], 2200)
        self.assertLessEqual(artifact["logical_coefficient_count"], 1600)

    def setUp(self):
        self.layout = OperatorSchmidtLayout(
            "test", (2, 3), (2, 4), (0, 2, 1, 3),
        )

    def test_packed_bits_do_not_change_structural_count(self):
        # Three rows, five logical values and two scale groups per row.
        self.assertEqual(packed_matrix_coefficient_count(3, 5, 4), 21)
        for bits in (2, 3, 4):
            codec = encode_groupwise(torch.ones(3, 5), bits=bits, group_size=4)
            self.assertEqual(
                codec["shape"][0] * codec["shape"][1]
                + codec["group_scales"].numel(),
                21,
            )

    def test_rank_respects_exact_coefficient_boundary(self):
        rank_two = packed_schmidt_coefficient_count(
            self.layout, 2, 4, input_scale_count=8, output_scale_count=6,
        )
        self.assertEqual(
            rank_for_coefficient_budget(
                self.layout, rank_two, 4,
                input_scale_count=8, output_scale_count=6,
            ),
            2,
        )
        self.assertEqual(
            rank_for_coefficient_budget(
                self.layout, rank_two - 1, 4,
                input_scale_count=8, output_scale_count=6,
            ),
            1,
        )

    def test_artifact_count_includes_scales_and_residual_branch(self):
        left = encode_groupwise(torch.ones(2, 6), bits=3, group_size=4)
        right = encode_groupwise(torch.ones(2, 8), bits=3, group_size=4)
        residual_left = encode_groupwise(torch.ones(1, 6), bits=3, group_size=4)
        residual_right = encode_groupwise(torch.ones(1, 8), bits=3, group_size=4)
        column_values = encode_groupwise(torch.ones(6, 2), bits=4, group_size=4)
        artifact = {
            "left_codec": left,
            "right_codec": right,
            "input_scale": torch.ones(8),
            "output_scale": torch.ones(6),
            "calibration_schmidt_branch": {
                "left_codec": residual_left,
                "right_codec": residual_right,
            },
            "calibration_column_branch": {
                "column_indices": torch.tensor([1, 5], dtype=torch.int32),
                "value_codec": column_values,
            },
        }
        expected = sum(
            codec["shape"][0] * codec["shape"][1]
            + codec["group_scales"].numel()
            for codec in (left, right, residual_left, residual_right, column_values)
        ) + 8 + 6
        self.assertEqual(
            packed_schmidt_coefficient_count_from_artifact(artifact), expected,
        )


if __name__ == "__main__":
    unittest.main()
