import unittest

import torch

from nox.functional_schmidt import SchmidtCoreMap, construct_functional_schmidt_mlp
from nox.packed_schmidt import (
    packed_codec_coefficient_count, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)
from nox.residual_codec import codec_payload_bytes
from nox.sparse_schmidt import compile_sparse_schmidt


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


if __name__ == '__main__':
    unittest.main()
