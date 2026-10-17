import unittest

import torch

from nox.swiglu_rank_bound import (
    evaluate_down_crossovers,
    output_rank_curve,
    relative_rank_tail_errors,
    swiglu_output,
)


class SwigluRankBoundTests(unittest.TestCase):
    def test_eckart_young_curve_matches_diagonal_tail(self):
        matrix = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        curve = output_rank_curve(matrix, [0, 1, 2, 3, 4])
        self.assertAlmostEqual(curve["relative_tail_error_by_rank"]["0"], 1.0)
        self.assertAlmostEqual(
            curve["relative_tail_error_by_rank"]["2"], (5.0 / 30.0) ** 0.5,
        )
        self.assertAlmostEqual(curve["relative_tail_error_by_rank"]["4"], 0.0)

    def test_tail_curve_is_monotone_and_clips_large_rank(self):
        errors = relative_rank_tail_errors(torch.tensor([2.0, 1.0]), [0, 1, 2, 9])
        self.assertEqual(errors[2], 0.0)
        self.assertEqual(errors[9], 0.0)
        self.assertGreater(errors[0], errors[1])
        self.assertGreaterEqual(errors[1], errors[2])

    def test_rank426_bound_cannot_be_applied_to_general_full_rank_outputs(self):
        # A compact shared Down factor constrains every batch's output columns
        # to span at most r output directions, independent of Gate/Up.
        torch.manual_seed(3)
        d_model, intermediate, rank, samples = 12, 19, 4, 21
        down_factor = torch.randn(d_model, rank)
        basis = torch.randn(intermediate, rank)
        product = torch.randn(intermediate, samples)
        represented = down_factor @ (basis.T @ product)
        self.assertLessEqual(int(torch.linalg.matrix_rank(represented)), rank)
        # The diagnostic itself makes no claim that all matrix/TN outputs have
        # this rank: this deliberately full-rank matrix has positive tail.
        arbitrary = torch.randn(d_model, samples)
        arbitrary_curve = output_rank_curve(arbitrary, [rank, d_model])
        self.assertGreater(arbitrary_curve["relative_tail_error_by_rank"][str(rank)], 0.0)

    def test_general_neuron_projection_does_not_commute_with_swiglu(self):
        angle = torch.tensor(0.7)
        projection = torch.tensor([
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ])
        gate = torch.tensor([[1.0], [-0.4]])
        up = torch.tensor([[0.2], [1.7]])
        lhs = torch.nn.functional.silu(projection @ gate) * (projection @ up)
        rhs = projection @ (torch.nn.functional.silu(gate) * up)
        self.assertGreater(float(torch.linalg.vector_norm(lhs - rhs)), 1e-2)

    def test_swiglu_output_shapes_and_linear_conventions(self):
        torch.manual_seed(8)
        d_model, intermediate, samples = 5, 7, 3
        gate = torch.randn(intermediate, d_model)
        up = torch.randn(intermediate, d_model)
        down = torch.randn(d_model, intermediate)
        inputs = torch.randn(d_model, samples)
        expected = down @ (
            torch.nn.functional.silu(gate @ inputs) * (up @ inputs)
        )
        actual = swiglu_output(gate, up, down, inputs)
        self.assertEqual(tuple(actual.shape), (d_model, samples))
        torch.testing.assert_close(actual, expected)

    def test_crossovers_hold_gate_up_product_fixed_and_decompose_exactly(self):
        torch.manual_seed(11)
        d_model, intermediate, samples = 6, 9, 5
        dense_gate = torch.randn(intermediate, d_model)
        dense_up = torch.randn(intermediate, d_model)
        dense_down = torch.randn(d_model, intermediate)
        candidate_gate = dense_gate + 0.1 * torch.randn_like(dense_gate)
        candidate_up = dense_up + 0.1 * torch.randn_like(dense_up)
        candidate_down = dense_down + 0.1 * torch.randn_like(dense_down)
        saved_gate = dense_gate + 0.2 * torch.randn_like(dense_gate)
        saved_up = dense_up + 0.2 * torch.randn_like(dense_up)
        saved_down = dense_down + 0.2 * torch.randn_like(dense_down)
        inputs = torch.randn(d_model, samples)
        rows = evaluate_down_crossovers(
            dense_gate=dense_gate,
            dense_up=dense_up,
            dense_down=dense_down,
            candidate_gate=candidate_gate,
            candidate_up=candidate_up,
            candidate_down=candidate_down,
            saved_nec_gate=saved_gate,
            saved_nec_up=saved_up,
            saved_nec_down=saved_down,
            inputs=inputs,
        )
        self.assertEqual(
            set(rows),
            {"dense_down_oracle", "saved_nec_down", "candidate_down", "saved_nec_baseline"},
        )
        for name in ("dense_down_oracle", "saved_nec_down", "candidate_down"):
            self.assertLess(rows[name]["decomposition_identity_relative_residual"], 1e-5)
        # Because the oracle branch is exactly dense D, its Down error is zero.
        self.assertLess(rows["dense_down_oracle"]["down_term_relative_to_dense_output"], 1e-7)
        # All crossover branches share the same G/U product and therefore the
        # same product error, isolating the effect of the selected Down matrix.
        product_errors = [
            rows[name]["product_relative_error"]
            for name in ("dense_down_oracle", "saved_nec_down", "candidate_down")
        ]
        self.assertAlmostEqual(max(product_errors) - min(product_errors), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
