import unittest
import tempfile
from pathlib import Path

import torch

from nox.compiler import reconstruct_from_artifact
from nox.pees import (
    append_shield_to_projection_artifact,
    fit_propagated_eigenspace_shield,
)


@unittest.skipUnless(torch.cuda.is_available(), "PEES regression tests run on the authorized CUDA host")
class PropagatedEigenspaceShieldTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(913)

    def test_rank_limited_shield_recovers_propagated_low_rank_defect(self):
        d_out, d_in, samples, rank = 12, 16, 96, 3
        x = torch.randn(d_in, samples, device=self.device)
        dense_weight = torch.randn(d_out, d_in, device=self.device)
        left = torch.randn(d_out, rank, device=self.device)
        right = torch.randn(rank, d_in, device=self.device)
        true_correction = left @ right
        compressed_weight = dense_weight - true_correction

        shield_left, shield_right, metrics = fit_propagated_eigenspace_shield(
            x, x, dense_weight, compressed_weight, rank=rank, ridge_ratio=1e-8,
        )
        reconstructed = shield_left.to(self.device).float() @ shield_right.to(self.device).float()
        relative_error = torch.linalg.vector_norm((dense_weight - compressed_weight - reconstructed) @ x)
        relative_error /= torch.linalg.vector_norm(true_correction @ x)
        self.assertLess(float(relative_error), 2e-3)
        self.assertEqual(metrics["effective_rank"], rank)
        self.assertLess(metrics["calibration_relative_error_after"], metrics["calibration_relative_error_before"])

    def test_target_includes_corrupted_input_trajectory(self):
        d_out, d_in, samples = 10, 14, 128
        x_dense = torch.randn(d_in, samples, device=self.device)
        x_compressed = x_dense + 0.3 * torch.randn_like(x_dense)
        dense_weight = torch.randn(d_out, d_in, device=self.device)
        compressed_weight = dense_weight + 0.1 * torch.randn_like(dense_weight)

        _, _, metrics = fit_propagated_eigenspace_shield(
            x_dense,
            x_compressed,
            dense_weight,
            compressed_weight,
            rank=8,
            ridge_ratio=0.01,
        )
        self.assertGreater(metrics["calibration_relative_error_before"], 0)
        self.assertLess(metrics["calibration_relative_error_after"], metrics["calibration_relative_error_before"])
        self.assertGreaterEqual(metrics["attainable_defect_energy_fraction"], 0)
        self.assertLessEqual(metrics["attainable_defect_energy_fraction"], 1)

    def test_appended_factors_reconstruct_as_a_sum(self):
        d_out, d_in = 16, 16
        old_left = torch.randn(d_out, 2, dtype=torch.float16)
        old_right = torch.randn(2, d_in, dtype=torch.float16)
        new_left = torch.randn(d_out, 3, dtype=torch.float16)
        new_right = torch.randn(3, d_in, dtype=torch.float16)
        artifact = {
            "orig_shape": (d_out, d_in),
            "cores": [torch.zeros(1, 2, 2, 1, dtype=torch.float16) for _ in range(4)],
            "d_awt": torch.ones(d_out, dtype=torch.float16),
            "sparse_indices": None,
            "sparse_values": None,
            "A_bridge": old_left,
            "B_bridge": old_right,
        }
        added_bytes = append_shield_to_projection_artifact(artifact, new_left, new_right)
        expected = old_left.float() @ old_right.float() + new_left.float() @ new_right.float()
        actual = artifact["A_bridge"].float() @ artifact["B_bridge"].float()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(artifact["A_bridge"].shape, (d_out, 5))
        self.assertEqual(artifact["B_bridge"].shape, (5, d_in))
        self.assertEqual(added_bytes, (d_out * 3 + 3 * d_in) * 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pees_layer.pt"
            torch.save({"compressed_artifacts": {"v_proj": artifact}}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            reconstructed = reconstruct_from_artifact(
                loaded["compressed_artifacts"]["v_proj"], device="cpu",
            )
        self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
