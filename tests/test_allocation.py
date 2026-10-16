import unittest

from nox.allocation import PROJECTION_NAMES, projection_budgets


class ProjectionAllocationTests(unittest.TestCase):
    def test_interaction_aware_allocation_preserves_total_budget(self):
        base = 524288
        budgets = projection_budgets("interaction_aware", r_bridge=128, n_outliers=base)
        self.assertEqual(set(budgets), set(PROJECTION_NAMES))
        self.assertEqual(sum(item["n_outliers"] for item in budgets.values()), 7 * base)
        self.assertTrue(all(item["r_bridge"] == 128 for item in budgets.values()))
        self.assertEqual(budgets["q_proj"]["n_outliers"], base // 2)
        self.assertEqual(budgets["o_proj"]["n_outliers"], base // 2)
        self.assertEqual(budgets["gate_proj"]["n_outliers"], 3 * base // 2)
        self.assertEqual(budgets["down_proj"]["n_outliers"], 3 * base // 2)

    def test_uniform_allocation_matches_legacy_budget(self):
        budgets = projection_budgets("uniform", r_bridge=64, n_outliers=101)
        self.assertTrue(all(item == {"r_bridge": 64, "n_outliers": 101}
                            for item in budgets.values()))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            projection_budgets("mystery", r_bridge=1, n_outliers=1)


if __name__ == "__main__":
    unittest.main()
