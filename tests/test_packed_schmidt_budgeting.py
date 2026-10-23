import unittest

import torch

from nox.operator_schmidt import OperatorSchmidtLayout
from nox.packed_schmidt import (
    packed_matrix_coefficient_count,
    packed_schmidt_coefficient_count,
    packed_schmidt_coefficient_count_from_artifact,
    rank_for_coefficient_budget,
)
from nox.residual_codec import encode_groupwise


class PackedSchmidtStructuralBudgetTests(unittest.TestCase):
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
