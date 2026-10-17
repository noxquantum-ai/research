import io
import unittest

import torch
import torch.nn.functional as F

from nox.joint_swiglu import (
    CODEC_TYPE,
    apply_joint_swiglu,
    build_joint_swiglu_artifact,
    joint_neuron_coefficient_count,
    joint_neuron_rank_for_budget,
    mlp_error_decomposition,
    reconstruct_joint_swiglu_weights,
    shared_swiglu_payload_bytes,
    shared_swiglu_serialized_bytes,
)


class JointSwiGLUTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(1701)
        self.d_model = 5
        self.intermediate_size = 8
        self.gate = torch.randn(
            self.intermediate_size, self.d_model, generator=generator,
        )
        self.up = torch.randn(
            self.intermediate_size, self.d_model, generator=generator,
        )
        self.down = torch.randn(
            self.d_model, self.intermediate_size, generator=generator,
        )
        self.train = torch.randn(self.d_model, 16, generator=generator)
        self.valid = torch.randn(self.d_model, 11, generator=generator)

    def test_full_shared_rank_reconstructs_the_complete_mlp(self):
        artifact = build_joint_swiglu_artifact(
            self.gate, self.up, self.down, self.train, rank=self.intermediate_size,
        )
        reconstructed = reconstruct_joint_swiglu_weights(artifact)
        for name, expected in (
            ("gate_proj", self.gate),
            ("up_proj", self.up),
            ("down_proj", self.down),
        ):
            torch.testing.assert_close(reconstructed[name], expected, atol=2e-5, rtol=2e-5)
        expected_output = self.down @ (
            F.silu(self.gate @ self.valid) * (self.up @ self.valid)
        )
        actual_output = apply_joint_swiglu(artifact, self.valid)
        torch.testing.assert_close(actual_output, expected_output, atol=3e-5, rtol=3e-5)

    def test_budget_rank_and_coefficient_accounting(self):
        one_rank = self.intermediate_size + 3 * self.d_model
        budget = 3 * one_rank + one_rank - 1
        rank = joint_neuron_rank_for_budget(
            budget, self.d_model, self.intermediate_size,
            calibration_tokens=self.train.shape[1],
        )
        self.assertEqual(rank, 3)
        self.assertLessEqual(
            joint_neuron_coefficient_count(self.d_model, self.intermediate_size, rank),
            budget,
        )
        with self.assertRaisesRegex(ValueError, "too small"):
            joint_neuron_rank_for_budget(
                one_rank - 1, self.d_model, self.intermediate_size,
            )

    def test_shared_activation_basis_is_best_joint_rank_one_projection(self):
        artifact = build_joint_swiglu_artifact(
            self.gate, self.up, self.down, self.train, rank=1,
        )
        basis = artifact["neuron_basis"].float()
        gate_state = self.gate @ self.train
        up_state = self.up @ self.train
        product_state = F.silu(gate_state) * up_state
        joint_states = torch.cat((gate_state, up_state, product_state), dim=1)
        actual_residual = torch.linalg.vector_norm(
            joint_states - basis @ (basis.T @ joint_states),
        )
        generator = torch.Generator().manual_seed(91)
        random_basis = torch.randn(self.intermediate_size, 1, generator=generator)
        random_basis = F.normalize(random_basis, dim=0)
        random_residual = torch.linalg.vector_norm(
            joint_states - random_basis @ (random_basis.T @ joint_states),
        )
        self.assertLessEqual(float(actual_residual), float(random_residual) + 1e-5)

    def test_factorized_apply_serialization_and_actual_coefficient_count(self):
        artifact = build_joint_swiglu_artifact(
            self.gate, self.up, self.down, self.train, rank=3,
            storage_dtype=torch.float16,
        )
        self.assertEqual(artifact["codec_type"], CODEC_TYPE)
        self.assertFalse(artifact["construction_uses_validation"])
        self.assertFalse(artifact["post_compression_optimization"])
        self.assertEqual(
            artifact["coefficient_count"],
            sum(artifact[key].numel() for key in (
                "neuron_basis", "gate_factor", "up_factor", "down_factor",
            )),
        )
        self.assertEqual(
            shared_swiglu_payload_bytes(artifact),
            sum(artifact[key].numel() * artifact[key].element_size() for key in (
                "neuron_basis", "gate_factor", "up_factor", "down_factor",
            )),
        )
        stream = io.BytesIO()
        torch.save(artifact, stream)
        self.assertEqual(stream.tell(), shared_swiglu_serialized_bytes(artifact))
        stream.seek(0)
        loaded = torch.load(stream, map_location="cpu", weights_only=True)
        direct = apply_joint_swiglu(loaded, self.valid)
        dense = reconstruct_joint_swiglu_weights(loaded)
        expected = dense["down_proj"] @ (
            F.silu(dense["gate_proj"] @ self.valid)
            * (dense["up_proj"] @ self.valid)
        )
        torch.testing.assert_close(direct, expected, atol=2e-5, rtol=2e-5)

    def test_eight_controlled_dense_compressed_combinations_are_reported(self):
        compressed_gate = self.gate + 0.08
        compressed_up = self.up - 0.06
        compressed_down = self.down + 0.04
        result = mlp_error_decomposition(
            self.gate, self.up, self.down,
            compressed_gate, compressed_up, compressed_down,
            self.valid,
        )
        self.assertEqual(len(result["combinations"]), 8)
        dense_key = "gate_dense__up_dense__down_dense"
        self.assertLess(result["combinations"][dense_key]["mlp_output_relative_error"], 1e-7)
        self.assertLess(result["gate_up_product_identity_relative_residual"], 1e-6)
        for record in result["combinations"].values():
            self.assertLess(record["decomposition_identity_relative_residual"], 1e-5)

    def test_down_error_can_cancel_gate_up_product_error(self):
        gate = torch.tensor([[1.0]])
        up = torch.tensor([[2.0]])
        down = torch.tensor([[3.0]])
        inputs = torch.tensor([[1.0]])
        compressed_gate = torch.tensor([[2.0]])
        compressed_up = up.clone()
        dense_product = F.silu(gate @ inputs) * (up @ inputs)
        compressed_product = F.silu(compressed_gate @ inputs) * (up @ inputs)
        product_term = down @ (compressed_product - dense_product)
        canceling_down = down - product_term / compressed_product
        result = mlp_error_decomposition(
            gate, up, down, compressed_gate, compressed_up, canceling_down, inputs,
        )
        record = result["combinations"]["gate_compressed__up_compressed__down_compressed"]
        self.assertLess(record["mlp_output_relative_error"], 1e-6)
        self.assertLess(record["product_down_error_cosine"], -0.99999)
        self.assertLess(record["product_down_cross_energy_fraction"], -0.99999)
        self.assertLess(record["decomposition_identity_relative_residual"], 1e-6)

    def test_gate_up_cross_term_completes_product_error(self):
        compressed_gate = self.gate + 0.2 * torch.randn_like(self.gate)
        compressed_up = self.up + 0.2 * torch.randn_like(self.up)
        result = mlp_error_decomposition(
            self.gate, self.up, self.down,
            compressed_gate, compressed_up, self.down,
            self.valid,
        )
        terms = result["gate_up_product_terms"]
        self.assertEqual(set(terms), {"gate_only", "up_only", "gate_up_interaction"})
        self.assertLess(result["gate_up_product_identity_relative_residual"], 1e-6)

    def test_zero_activations_remain_finite(self):
        artifact = build_joint_swiglu_artifact(
            torch.zeros_like(self.gate), torch.zeros_like(self.up),
            torch.zeros_like(self.down), torch.zeros_like(self.train), rank=1,
        )
        output = apply_joint_swiglu(artifact, torch.zeros_like(self.valid))
        self.assertTrue(torch.isfinite(artifact["neuron_basis"]).all())
        self.assertTrue(torch.isfinite(output).all())
        result = mlp_error_decomposition(
            torch.zeros_like(self.gate), torch.zeros_like(self.up),
            torch.zeros_like(self.down), torch.zeros_like(self.gate),
            torch.zeros_like(self.up), torch.zeros_like(self.down),
            torch.zeros_like(self.valid),
        )
        self.assertTrue(all(
            torch.isfinite(torch.tensor(item["mlp_output_relative_error"]))
            for item in result["combinations"].values()
        ))

    def test_coefficient_budget_too_small_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            build_joint_swiglu_artifact(
                self.gate, self.up, self.down, self.train,
                coefficient_budget=self.intermediate_size + 3 * self.d_model - 1,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test")
    def test_cuda_factor_construction_serialization_and_direct_apply(self):
        device = torch.device("cuda")
        gate = self.gate.to(device)
        up = self.up.to(device)
        down = self.down.to(device)
        train = self.train.to(device)
        valid = self.valid.to(device)
        artifact = build_joint_swiglu_artifact(
            gate, up, down, train, rank=4, storage_dtype=torch.float16,
        )
        stream = io.BytesIO()
        torch.save(artifact, stream)
        stream.seek(0)
        loaded = torch.load(stream, map_location=device, weights_only=True)
        direct = apply_joint_swiglu(loaded, valid)
        reconstructed = reconstruct_joint_swiglu_weights(
            loaded, device=device, dtype=torch.float32,
        )
        reconstructed_output = reconstructed["down_proj"] @ (
            F.silu(reconstructed["gate_proj"] @ valid)
            * (reconstructed["up_proj"] @ valid)
        )
        torch.testing.assert_close(direct, reconstructed_output, atol=2e-5, rtol=2e-5)
        self.assertTrue(torch.isfinite(direct).all())
        self.assertEqual(artifact["coefficient_count"], 4 * (8 + 3 * 5))


if __name__ == "__main__":
    unittest.main()
