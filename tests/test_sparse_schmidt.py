import unittest

import torch

from nox.functional_schmidt import SchmidtCoreMap, construct_functional_schmidt_mlp
from nox.packed_schmidt import (
    packed_codec_coefficient_count, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)
from nox.residual_codec import codec_payload_bytes, decode_groupwise
from nox.sparse_schmidt import (compile_sparse_schmidt, covariance_feedback_excitations,
                               reconstruct_excitation_operator)


class SparseSchmidtTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(131)
        self.weight = torch.randn(64, 64) * .1
        self.inputs = torch.randn(64, 256)
        self.artifact, self.decoded, self.report = compile_sparse_schmidt(
            self.weight, byte_budget=1800, coefficient_budget=2048,
            calibration_inputs=self.inputs, backbone_fraction=.2,
            bits_candidates=(4,), amplitude_bits=3, group_size=16,
            als_sweeps=0, excitation_sequence_length=16,
        )

    def test_exact_payload_coefficient_accounting_and_roundtrip(self):
        a = self.artifact
        self.assertEqual(a['support_bits'].numel(), 64 * 64 // 8)
        self.assertEqual(packed_schmidt_payload_bytes_from_artifact(a),
                         packed_schmidt_payload_bytes_from_artifact(a['base_artifact'])
                         + codec_payload_bytes(a['amplitude_codec']) + 512)
        self.assertEqual(packed_schmidt_coefficient_count_from_artifact(a),
                         packed_schmidt_coefficient_count_from_artifact(a['base_artifact'])
                         + packed_codec_coefficient_count(a['amplitude_codec']))
        self.assertLessEqual(a['payload_bytes'], 1800)
        self.assertLessEqual(a['logical_coefficient_count'], 2048)
        torch.testing.assert_close(reconstruct_packed_schmidt(a), self.decoded)
        self.assertLessEqual(self.report['final_guard_relative_mse'], self.report['initial_guard_relative_mse'])

    def test_amplitude_and_core_adjoint_and_packed_updates(self):
        mapper = SchmidtCoreMap(self.artifact, 'cpu')
        for side, parameters in (('left', mapper.left), ('right', mapper.right), ('amplitudes', mapper.amplitudes)):
            core = torch.randn_like(parameters)
            matrix = torch.randn_like(self.weight)
            torch.testing.assert_close((mapper.weight(core, side) * matrix).sum(),
                                       (core * mapper.adjoint(matrix, side)).sum(), atol=2e-5, rtol=2e-5)
            candidate = mapper.packed_candidate(parameters, side)
            self.assertEqual(packed_schmidt_coefficient_count_from_artifact(candidate),
                             packed_schmidt_coefficient_count_from_artifact(self.artifact))

    def test_coupled_solver_can_construct_sparse_amplitudes_without_adding_weights(self):
        artifacts = {name: self.artifact for name in ('gate_proj', 'up_proj', 'down_proj')}
        target = self.weight @ (torch.nn.functional.silu(self.weight @ self.inputs) * (self.weight @ self.inputs))
        result, _, report = construct_functional_schmidt_mlp(
            artifacts, self.inputs, target, gate_targets=self.weight @ self.inputs,
            cg_iterations=3, sequence_length=16,
        )
        self.assertTrue(any(row['core'] == 'amplitudes' for row in report['records']))
        self.assertLessEqual(report['final_guard_relative_mse'], report['initial_guard_relative_mse'])
        for name in artifacts:
            self.assertEqual(packed_schmidt_coefficient_count_from_artifact(result[name]),
                             packed_schmidt_coefficient_count_from_artifact(artifacts[name]))

    def test_constructor_selection_preserves_budget_and_improves_both_training_partitions(self):
        candidate, decoded, report = reconstruct_excitation_operator(
            self.weight, self.artifact, self.inputs, self.weight @ self.inputs, sequence_length=16)
        self.assertLessEqual(report['final_fit_relative_mse'], report['initial_fit_relative_mse'])
        self.assertLessEqual(report['final_guard_relative_mse'], report['initial_guard_relative_mse'])
        self.assertEqual(packed_schmidt_payload_bytes_from_artifact(candidate), self.artifact['payload_bytes'])
        self.assertEqual(packed_schmidt_coefficient_count_from_artifact(candidate), self.artifact['logical_coefficient_count'])
        torch.testing.assert_close(decoded, reconstruct_packed_schmidt(candidate))

    def test_channel_relative_coupled_solve_uses_same_representation(self):
        artifacts = {name: self.artifact for name in ('gate_proj', 'up_proj', 'down_proj')}
        target = self.weight @ (torch.nn.functional.silu(self.weight @ self.inputs) * (self.weight @ self.inputs))
        result, _, report = construct_functional_schmidt_mlp(
            artifacts, self.inputs, target, gate_targets=self.weight @ self.inputs,
            cg_iterations=3, sequence_length=16, output_metric='channel_relative')
        self.assertEqual(report['output_metric'], 'channel_relative')
        self.assertLessEqual(report['final_guard_relative_mse'], report['initial_guard_relative_mse'])
        for name in artifacts:
            self.assertEqual(packed_schmidt_coefficient_count_from_artifact(result[name]),
                             packed_schmidt_coefficient_count_from_artifact(artifacts[name]))


class CovarianceFeedbackTests(unittest.TestCase):
    def test_exact_grid_recovers_all_coordinates_and_undoes_construction_gauge(self):
        torch.manual_seed(77)
        weight = torch.tensor([[-3., -1., 0., 3., -3., 1., 3.]]).repeat(3, 1)
        inputs = torch.randn(7, 128) * torch.tensor([.01, .2, 1., 3., 10., 50., 100.])[:, None]
        indices, codec, _ = covariance_feedback_excitations(
            weight, inputs, weight @ inputs, kept=7, bits=3, group_size=4,
            block_size=3, scale_policy='maxabs', fixed_indices=torch.arange(7).repeat(3, 1),
        )
        actual = torch.zeros_like(weight).scatter_(1, indices, decode_groupwise(codec))
        torch.testing.assert_close(actual, weight, atol=1e-6, rtol=1e-6)

    def test_inverse_covariance_support_keeps_exactly_the_requested_count(self):
        torch.manual_seed(78)
        weight, inputs = torch.randn(4, 12), torch.randn(12, 80)
        indices, codec, report = covariance_feedback_excitations(
            weight, inputs, weight @ inputs, kept=5, bits=3, group_size=4, block_size=3,
            quant_grid='nonzero',
        )
        self.assertEqual(tuple(indices.shape), (4, 5))
        self.assertTrue(torch.all(indices[:, 1:] > indices[:, :-1]))
        self.assertTrue(torch.isfinite(decode_groupwise(codec)).all())
        self.assertEqual(report['stored_covariance_coefficients'], 0)


if __name__ == '__main__':
    unittest.main()
