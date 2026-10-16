import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nox.block_compiler import (
    compile_attention_block,
    compile_mlp_block,
    compile_transformer_block,
    scaled_dot_product_attention_2d,
)


def exact_mock(calls):
    def decompose(**kwargs):
        calls.append(kwargs)
        return kwargs["W"].clone(), 1.0, [], {}

    return decompose


class TrajectoryAwareCompilerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.comp = torch.randn(4, 4)
        self.clean = self.comp + 0.2 * torch.randn(4, 4)

    def test_mlp_conditions_down_on_serialized_factor_trajectory_at_fixed_budget(self):
        weights = {
            "gate_proj": torch.randn(4, 4),
            "up_proj": torch.randn(4, 4),
            "down_proj": torch.randn(4, 4),
        }
        calls = []
        with patch("nox.block_compiler.nox_hybrid_dmrg_decompose", side_effect=exact_mock(calls)):
            compile_mlp_block(
                weights, self.comp, M_in_dense=self.clean, causal_healing=False,
                conditional_context=True, ranks=(1, 2, 2, 2, 1),
                r_bridge=7, n_outliers=11,
            )

        expected_context = F.silu(weights["gate_proj"] @ self.comp) * (weights["up_proj"] @ self.comp)
        torch.testing.assert_close(calls[2]["X_calib"], expected_context)
        torch.testing.assert_close(
            calls[2]["Y_dense"],
            weights["down_proj"] @ (
                F.silu(weights["gate_proj"] @ self.clean) * (weights["up_proj"] @ self.clean)
            ),
        )
        self.assertEqual(calls[2]["r_bridge"], 7)
        self.assertEqual(calls[2]["n_outliers"], 11)
        self.assertEqual(calls[2]["prior_gamma"], 0.01)

    def test_attention_conditions_output_on_compressed_qkv_context_at_fixed_budget(self):
        weights = {name: torch.randn(4, 4) for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        calls = []
        with patch("nox.block_compiler.nox_hybrid_dmrg_decompose", side_effect=exact_mock(calls)):
            compile_attention_block(
                weights, self.comp, X_in_dense=self.clean, causal_healing=False,
                conditional_context=True, ranks=(1, 2, 2, 2, 1),
                r_bridge=7, n_outliers=11, n_heads=1, n_kv_heads=1, seq_len=2,
            )

        expected_context = scaled_dot_product_attention_2d(
            weights["q_proj"] @ self.comp,
            weights["k_proj"] @ self.comp,
            weights["v_proj"] @ self.comp,
            n_heads=1, n_kv_heads=1, seq_len=2,
        )
        torch.testing.assert_close(calls[3]["X_calib"], expected_context)
        self.assertEqual(calls[3]["r_bridge"], 7)
        self.assertEqual(calls[3]["n_outliers"], 11)
        self.assertEqual(calls[3]["prior_gamma"], 0.01)

    def test_trajectory_and_legacy_healing_modes_are_mutually_exclusive(self):
        weights = {name: torch.eye(4) for name in (
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        )}
        weights.update({"input_layernorm": torch.ones(4), "post_attention_layernorm": torch.ones(4)})
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            compile_transformer_block(
                weights, self.comp, causal_healing=True, trajectory_mode="mlp",
                n_heads=1, n_kv_heads=1, seq_len=2,
            )


if __name__ == "__main__":
    unittest.main()
