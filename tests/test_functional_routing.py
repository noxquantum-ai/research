import unittest

import torch

from nox.functional_routing import PairedRoutingProbe, construct_functional_schmidt_routing
from nox.packed_schmidt import compile_packed_schmidt, packed_schmidt_coefficient_count_from_artifact


class FunctionalRoutingTests(unittest.TestCase):
    def test_rope_gqa_and_centered_measurement_adjoints(self):
        torch.manual_seed(137)
        for kv_heads in (1, 2):
            inputs, q, k = torch.randn(8, 32), torch.randn(8, 32), torch.randn(4 * kv_heads, 32)
            p = PairedRoutingProbe(inputs, q, k, n_heads=2, n_kv_heads=kv_heads,
                                   sequence_length=8, queries_per_window=4)
            q_sample = q[:, p.query_indices]
            q_dual = torch.randn_like(p.query_heads(q_sample))
            k_dual = torch.randn_like(p.key_heads(k))
            torch.testing.assert_close((p.query_heads(q_sample) * q_dual).sum(), (q_sample * p.query_adjoint(q_dual)).sum())
            torch.testing.assert_close((p.key_heads(k) * k_dual).sum(), (k * p.key_adjoint(k_dual)).sum())
            a, b = torch.randn_like(p.teacher_scores), torch.randn_like(p.teacher_scores)
            torch.testing.assert_close((p.measure(a) * b).sum(), (a * p.measure_adjoint(b)).sum(), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(p.measure(torch.ones_like(a)), torch.zeros_like(a), atol=1e-6, rtol=0)

    def test_packed_routing_acceptance_improves_held_out_window_kl(self):
        torch.manual_seed(139)
        x = torch.randn(8, 64)
        teacher = {role: torch.randn(8, 8) * .2 for role in ('q_proj', 'k_proj')}
        artifacts = {}
        for role, weight in teacher.items():
            artifacts[role], _, _ = compile_packed_schmidt(
                weight, byte_budget=4000, calibration_inputs=x,
                bits_candidates=(4,), group_size=4, als_sweeps=0,
            )
        result, _, report = construct_functional_schmidt_routing(
            artifacts, x, teacher['q_proj'] @ x, teacher['k_proj'] @ x,
            n_heads=2, n_kv_heads=2, sequence_length=8, queries_per_window=4,
            cg_iterations=6,
        )
        self.assertLess(report['final_guard_attention_kl'], report['initial_guard_attention_kl'])
        self.assertLessEqual(report['final_fit_attention_kl'], report['initial_fit_attention_kl'])
        for role in artifacts:
            self.assertEqual(packed_schmidt_coefficient_count_from_artifact(result[role]),
                             packed_schmidt_coefficient_count_from_artifact(artifacts[role]))
        self.assertFalse(report['uses_nonlinear_derivatives'])


if __name__ == '__main__':
    unittest.main()
