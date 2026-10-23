import unittest

import torch

from nox.block_compiler import scaled_dot_product_attention_2d
from nox.functional_attention import FixedCausalAttention, construct_functional_schmidt_attention
from nox.packed_schmidt import (
    compile_packed_schmidt, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact,
)


class FunctionalAttentionTests(unittest.TestCase):
    def test_causal_map_matches_sdpa_and_has_exact_gqa_adjoint(self):
        torch.manual_seed(113)
        for kv_heads in (1, 2):
            q, k, v = torch.randn(8, 24), torch.randn(4 * kv_heads, 24), torch.randn(4 * kv_heads, 24)
            operator = FixedCausalAttention(q, k, 2, kv_heads, 8)
            response = operator.forward(v)
            torch.testing.assert_close(response, scaled_dot_product_attention_2d(q, k, v, 2, kv_heads, 8))
            test_output = torch.randn_like(response)
            torch.testing.assert_close((response * test_output).sum(), (v * operator.adjoint(test_output)).sum())
            torch.testing.assert_close(operator.window_slice(1, 3).forward(v[:, 8:]), response[:, 8:])

    def test_coupled_construction_preserves_budget_and_improves_guard(self):
        torch.manual_seed(127)
        x = torch.randn(8, 64)
        q, k = torch.randn(8, 8) @ x * .2, torch.randn(8, 8) @ x * .2
        v, o = torch.randn(8, 8) * .3, torch.randn(8, 8) * .3
        context = scaled_dot_product_attention_2d(q, k, v @ x, 2, 2, 8)
        target = o @ context
        artifacts = {}
        for role, weight, cal in (("v_proj", v, x), ("o_proj", o, context)):
            artifacts[role], _, _ = compile_packed_schmidt(
                weight, byte_budget=4000, calibration_inputs=cal,
                bits_candidates=(4,), group_size=4, als_sweeps=0,
            )
        result, _, report = construct_functional_schmidt_attention(
            artifacts, x, target, q, k, n_heads=2, n_kv_heads=2, sequence_length=8,
            cg_iterations=6,
        )
        self.assertLess(report['final_guard_relative_mse'], report['initial_guard_relative_mse'])
        self.assertLessEqual(report['final_fit_relative_mse'], report['initial_fit_relative_mse'])
        self.assertFalse(report['uses_nonlinear_derivatives'])
        for role in result:
            for count in (packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact):
                self.assertEqual(count(result[role]), count(artifacts[role]))


if __name__ == '__main__':
    unittest.main()
