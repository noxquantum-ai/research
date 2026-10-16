import unittest

import torch

from nox.compiler import solve_accumulated_system


class SolverStabilityTests(unittest.TestCase):
    def test_singular_small_normal_equation_uses_stable_on_device_fallback(self):
        H = torch.zeros(4, 4)
        rhs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        prior = torch.zeros_like(rhs)
        solution = solve_accumulated_system(H, rhs, prior, gamma=0.0, D=4)
        self.assertEqual(solution.shape, (2, 4))
        self.assertTrue(torch.isfinite(solution).all())


if __name__ == "__main__":
    unittest.main()
