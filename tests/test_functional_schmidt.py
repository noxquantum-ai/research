import unittest

import torch
import torch.nn.functional as F

from nox.functional_schmidt import (
    SchmidtCoreMap, _prior_anchored_pcg, construct_functional_schmidt_mlp,
)
from nox.packed_schmidt import (
    compile_packed_schmidt, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)


class FunctionalSchmidtTests(unittest.TestCase):
    def test_core_adjoint_is_exact_with_input_and_output_scales(self):
        torch.manual_seed(79)
        weight = torch.randn(12, 8)
        inputs = torch.randn(8, 64)
        artifact, _, _ = compile_packed_schmidt(
            weight, byte_budget=4000, calibration_inputs=inputs,
            bits_candidates=(4,), group_size=4, als_sweeps=0,
        )
        artifact["output_scale"] = torch.linspace(0.5, 1.5, 12)
        mapper = SchmidtCoreMap(artifact, "cpu")
        for side in ("left", "right"):
            core = torch.randn_like(mapper.left if side == "left" else mapper.right)
            dense = torch.randn_like(weight)
            lhs = (mapper.weight(core, side) * dense).sum()
            rhs = (core * mapper.adjoint(dense, side)).sum()
            torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)

    def test_pcg_solves_known_positive_definite_normal_system(self):
        generator = torch.Generator().manual_seed(83)
        a = torch.randn(4, 4, generator=generator, dtype=torch.float64)
        h = a.T @ a + 0.2 * torch.eye(4, dtype=torch.float64)
        rhs = torch.randn(4, generator=generator, dtype=torch.float64)
        delta, history = _prior_anchored_pcg(
            lambda value: h @ value, rhs, h.diagonal(), iterations=6, tolerance=1e-9,
        )
        torch.testing.assert_close(delta, torch.linalg.solve(h, rhs))
        self.assertLess(history[-1]["relative_normal_residual"], 1e-9)

    def test_joint_construction_improves_guard_without_adding_coefficients(self):
        torch.manual_seed(89)
        shapes = {"gate_proj": (12, 8), "up_proj": (12, 8), "down_proj": (8, 12)}
        dense = {name: torch.randn(shape) * 0.3 for name, shape in shapes.items()}
        inputs = torch.randn(8, 64)
        target = dense["down_proj"] @ (
            F.silu(dense["gate_proj"] @ inputs) * (dense["up_proj"] @ inputs)
        )
        artifacts = {}
        for name in shapes:
            calibration = (
                inputs if name != "down_proj" else
                F.silu(dense["gate_proj"] @ inputs) * (dense["up_proj"] @ inputs)
            )
            artifacts[name], _, _ = compile_packed_schmidt(
                dense[name], byte_budget=4000, calibration_inputs=calibration,
                bits_candidates=(4,), group_size=4, als_sweeps=0,
            )
        result, weights, report = construct_functional_schmidt_mlp(
            artifacts, inputs, target, cg_iterations=6, sequence_length=8,
        )
        self.assertLess(report["final_guard_relative_mse"], report["initial_guard_relative_mse"])
        self.assertLessEqual(report["final_fit_relative_mse"], report["initial_fit_relative_mse"])
        self.assertFalse(report["uses_autograd"])
        for name in shapes:
            self.assertEqual(
                packed_schmidt_coefficient_count_from_artifact(artifacts[name]),
                packed_schmidt_coefficient_count_from_artifact(result[name]),
            )
            self.assertEqual(
                packed_schmidt_payload_bytes_from_artifact(artifacts[name]),
                packed_schmidt_payload_bytes_from_artifact(result[name]),
            )
            torch.testing.assert_close(reconstruct_packed_schmidt(result[name]), weights[name])


if __name__ == "__main__":
    unittest.main()
