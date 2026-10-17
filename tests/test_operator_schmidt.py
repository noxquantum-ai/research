import io
import unittest

import torch

from nox.compiler import reconstruct_from_artifact
from nox.block_compiler import compile_transformer_block
from nox.operator_schmidt import (
    OperatorSchmidtLayout,
    _normalized_objective,
    apply_operator_schmidt_artifact,
    apply_operator_schmidt_factors,
    four_site_parameter_count,
    nox_operator_schmidt_decompose,
    operator_schmidt_to_dense,
)


class OperatorSchmidtTests(unittest.TestCase):
    def test_dense_and_factorized_application_agree_for_both_partitions(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(12, 9, generator=generator)
        for axes in ((0, 2, 1, 3), (0, 3, 1, 2)):
            layout = OperatorSchmidtLayout("test", (2, 3), (3, 4), axes)
            rows, columns = layout.unfolding_shape
            left = torch.randn(5, rows, generator=generator)
            right = torch.randn(5, columns, generator=generator)
            dense = operator_schmidt_to_dense(left, right, layout)
            actual = apply_operator_schmidt_factors(left, right, layout, x)
            torch.testing.assert_close(actual, dense @ x, atol=2e-5, rtol=2e-5)

    def test_full_rank_two_site_factorization_reconstructs_matrix(self):
        generator = torch.Generator().manual_seed(11)
        weight = torch.randn(16, 16, generator=generator)
        inputs = torch.randn(16, 20, generator=generator)
        result = nox_operator_schmidt_decompose(
            weight, inputs, weight @ inputs,
            o_modes=(2, 2, 2, 2), i_modes=(2, 2, 2, 2),
            ranks=(1, 16, 16, 16, 1), r_bridge=0, n_outliers=0,
            quantize_components=False, return_artifacts=True,
            projection_name="q_proj", n_heads=4, weight_mix=1.0,
            activation_mix=0.0, verbose=False,
        )
        torch.testing.assert_close(result[0], weight, atol=2e-5, rtol=2e-5)
        self.assertEqual(result[4]["codec_type"], "nec_operator_schmidt_v1")
        self.assertFalse(result[4]["gradient_healing"])

    def test_screening_finds_crossed_structure(self):
        generator = torch.Generator().manual_seed(19)
        layout = OperatorSchmidtLayout("cross", (4, 4), (4, 4), (0, 3, 1, 2))
        left = torch.randn(1, 16, generator=generator)
        right = torch.randn(1, 16, generator=generator)
        weight = operator_schmidt_to_dense(left, right, layout)
        inputs = torch.randn(16, 32, generator=generator)
        result = nox_operator_schmidt_decompose(
            weight, inputs, weight @ inputs,
            o_modes=(2, 2, 2, 2), i_modes=(2, 2, 2, 2),
            ranks=(1, 2, 2, 2, 1), r_bridge=0, n_outliers=0,
            quantize_components=False, return_artifacts=True,
            projection_name="q_proj", n_heads=4, weight_mix=1.0,
            activation_mix=0.0, verbose=False,
        )
        self.assertIn("crossed", result[4]["layout"]["name"])
        self.assertLess(torch.linalg.vector_norm(result[0] - weight), 2e-4)

    def test_joint_rescale_does_not_worsen_mixed_objective(self):
        generator = torch.Generator().manual_seed(23)
        weight = torch.randn(16, 16, generator=generator)
        inputs = torch.randn(16, 12, generator=generator)
        target = weight @ inputs
        result = nox_operator_schmidt_decompose(
            weight, inputs, target,
            o_modes=(2, 2, 2, 2), i_modes=(2, 2, 2, 2),
            ranks=(1, 2, 3, 2, 1), r_bridge=2, n_outliers=8,
            quantize_components=True, return_artifacts=True,
            projection_name="q_proj", n_heads=4, verbose=False,
        )
        selected = result[4]["selected_layout"]
        final = float(_normalized_objective(weight, target, result[0], inputs, 1.0, 1.0))
        self.assertLessEqual(final, selected["mixed_objective_before_joint_rescale"] + 1e-5)
        self.assertTrue(torch.isfinite(result[0]).all())

    def test_artifact_roundtrip_factorized_apply_and_accounting(self):
        generator = torch.Generator().manual_seed(29)
        weight = torch.randn(12, 12, generator=generator)
        inputs = torch.randn(12, 17, generator=generator)
        o_modes, i_modes = (2, 3, 2, 1), (3, 2, 2, 1)
        ranks = (1, 3, 5, 3, 1)
        result = nox_operator_schmidt_decompose(
            weight, inputs, weight @ inputs, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=2, n_outliers=6, return_artifacts=True,
            quantize_components=True, projection_name="gate_proj", verbose=False,
        )
        artifact = result[4]
        buffer = io.BytesIO()
        torch.save(artifact, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, map_location="cpu", weights_only=True)
        decoded = reconstruct_from_artifact(loaded)
        direct = apply_operator_schmidt_artifact(loaded, inputs)
        torch.testing.assert_close(direct, decoded @ inputs, atol=2e-4, rtol=2e-4)
        self.assertLessEqual(
            artifact["core_coefficient_count"],
            four_site_parameter_count(o_modes, i_modes, ranks),
        )
        expected_payload = sum(
            value.numel() * value.element_size()
            for key, value in artifact.items()
            if key in {
                "left_factor", "right_factor", "input_scale", "A_bridge", "B_bridge",
                "sparse_indices", "sparse_values",
            } and value is not None
        )
        self.assertEqual(artifact["payload_bytes"], expected_payload)

    def test_zero_activation_context_is_finite(self):
        # Tied outlier scores and an activation-only degenerate objective
        # exercise exact sparse-budget selection and the FP16 input scale.
        weight = torch.ones(8, 8)
        inputs = torch.zeros(8, 4)
        result = nox_operator_schmidt_decompose(
            weight, inputs, torch.zeros(8, 4),
            o_modes=(2, 2, 2, 1), i_modes=(2, 2, 2, 1),
            ranks=(1, 2, 3, 2, 1), r_bridge=2, n_outliers=2,
            quantize_components=False, return_artifacts=True,
            projection_name="q_proj", n_heads=2, weight_mix=0.0,
            activation_mix=1.0, verbose=False,
        )
        self.assertTrue(torch.isfinite(result[0]).all())
        self.assertTrue(torch.isfinite(result[4]["input_scale"]).all())
        self.assertGreaterEqual(
            float(result[4]["input_scale"].min()), torch.finfo(torch.float16).tiny,
        )
        self.assertEqual(result[4]["sparse_indices"].shape[0], 2)

    def test_full_block_compiler_routes_every_projection_to_new_representation(self):
        generator = torch.Generator().manual_seed(37)
        weights = {
            name: torch.randn(shape, generator=generator) * 0.05
            for name, shape in {
                "q_proj": (16, 16), "k_proj": (16, 16), "v_proj": (16, 16),
                "o_proj": (16, 16), "gate_proj": (24, 16),
                "up_proj": (24, 16), "down_proj": (16, 24),
            }.items()
        }
        weights["input_layernorm"] = torch.ones(16)
        weights["post_attention_layernorm"] = torch.ones(16)
        inputs = torch.randn(16, 8, generator=generator)
        compiled = compile_transformer_block(
            weights, inputs, ranks=(1, 3, 4, 3, 1), r_bridge=1, n_outliers=4,
            n_heads=2, n_kv_heads=2, seq_len=4, causal_healing=False,
            return_artifacts=True, verbose=False, representation="operator_schmidt",
        )
        artifacts = compiled["compressed_artifacts"]
        self.assertEqual(set(artifacts), set(weights) - {"input_layernorm", "post_attention_layernorm"})
        for name, artifact in artifacts.items():
            self.assertEqual(artifact["codec_type"], "nec_operator_schmidt_v1")
            decoded = reconstruct_from_artifact(artifact)
            torch.testing.assert_close(decoded, compiled["compressed_weights"][name], atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    unittest.main()
