import unittest
import copy

import torch
from nox.compiler import reconstruct_from_artifact

from nox.functional_schmidt import SchmidtCoreMap, construct_functional_schmidt_mlp
from nox.packed_schmidt import (
    packed_codec_coefficient_count, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)
from nox.residual_codec import codec_payload_bytes, decode_groupwise, encode_groupwise, _pack_codes
from nox.sparse_schmidt import (compile_sparse_schmidt, covariance_feedback_excitations,
                               reconstruct_excitation_operator, construction_matmul_precision, regularized_teacher_target,
                               reconstruct_global_excitations)
from nox.sparse_schmidt import block_shrunk_output_precision, refit_sparse_excitations
from nox.sparse_schmidt import reorder_excitation_support


class SparseSchmidtTests(unittest.TestCase):
    def test_output_covariance_uses_fit_only_and_stays_positive_definite(self):
        torch.manual_seed(251)
        targets=torch.randn(8,64);targets[1]+=targets[0]*3
        first,report=block_shrunk_output_precision(targets,fit_tokens=48,sequence_length=8)
        targets[:,48:]*=100000
        second,_=block_shrunk_output_precision(targets,fit_tokens=48,sequence_length=8)
        torch.testing.assert_close(first,second,rtol=0,atol=0)
        self.assertGreater(float(torch.linalg.eigvalsh(first).min()),0)
        self.assertFalse(report['guard_used'])
        torch.testing.assert_close(first.diagonal().mean(),torch.tensor(1.))

    def test_identity_output_geometry_matches_existing_sparse_solve(self):
        args=(self.artifact,self.inputs,self.weight@self.inputs)
        first=refit_sparse_excitations(*args,split=192,iterations=5)
        second=refit_sparse_excitations(*args,split=192,iterations=5,output_precision=torch.eye(64))
        torch.testing.assert_close(first[1],second[1],rtol=1e-5,atol=1e-6)

    def test_global_excitation_decode_with_unequal_row_support(self):
        mask=torch.zeros(64,64,dtype=torch.bool)
        mask[0,:9]=True;mask[1,:3]=True;mask[2,:4]=True
        values=torch.linspace(-.2,.2,int(mask.sum())).reshape(1,-1)
        codec=encode_groupwise(values,bits=3,group_size=64)
        artifact={'codec_type':'nec_global_excitation_schmidt_v1','orig_shape':(64,64),
                  'base_artifact':self.artifact['base_artifact'],'support_bits':_pack_codes(mask,1),'amplitude_codec':codec}
        expected=reconstruct_packed_schmidt(artifact['base_artifact']);expected[mask]+=decode_groupwise(codec).reshape(-1)
        torch.testing.assert_close(reconstruct_from_artifact(artifact),expected,rtol=0,atol=0)
        changed=reorder_excitation_support(artifact,'column_major')
        torch.testing.assert_close(reconstruct_from_artifact(changed),expected,rtol=0,atol=0)
        self.assertEqual(packed_schmidt_coefficient_count_from_artifact(artifact),
            packed_schmidt_coefficient_count_from_artifact(artifact['base_artifact'])+packed_codec_coefficient_count(codec))

    def test_support_transpose_preserves_native_weights_and_both_budgets(self):
        for codec_type in ('nec_sparse_excitation_schmidt_v1','nec_sparse_excitation_schmidt_v2'):
            artifact=copy.deepcopy(self.artifact);artifact['codec_type']=codec_type
            if codec_type.endswith('v2'):artifact['amplitude_basis']='backbone_input_scale'
            changed=reorder_excitation_support(artifact,'column_major')
            torch.testing.assert_close(reconstruct_from_artifact(changed),reconstruct_from_artifact(artifact),rtol=0,atol=0)
            for count in (packed_schmidt_payload_bytes_from_artifact,packed_schmidt_coefficient_count_from_artifact):
                self.assertEqual(count(changed),count(artifact))

    def test_global_allocation_obeys_source_coefficient_cap_and_guards(self):
        artifact,decoded,report=reconstruct_global_excitations(self.weight,self.artifact,self.inputs,
            self.weight@self.inputs,torch.linspace(.1,3,64),sequence_length=16,minimum_per_row=4)
        self.assertLessEqual(packed_schmidt_coefficient_count_from_artifact(artifact),self.artifact['logical_coefficient_count'])
        self.assertFalse(report['guard_used_for_support'])
        torch.testing.assert_close(reconstruct_from_artifact(artifact),decoded)

    def test_target_regularization_is_closed_form_and_ignores_guard(self):
        symmetric=torch.zeros(3,16);teacher=torch.ones(3,16);prior=torch.full((3,16),.25)
        target,report=regularized_teacher_target(symmetric,teacher,prior,fit_tokens=12)
        self.assertEqual(report['alpha'],.25)
        teacher[:,12:]=10000
        _,second=regularized_teacher_target(symmetric,teacher,prior,fit_tokens=12)
        self.assertEqual(second['alpha'],.25)
        torch.testing.assert_close(target,prior)

    def test_construction_precision_restores_reference_even_after_failure(self):
        previous=torch.get_float32_matmul_precision()
        with self.assertRaises(RuntimeError):
            with construction_matmul_precision('high'):
                self.assertEqual(torch.get_float32_matmul_precision(),'high')
                raise RuntimeError('test failure')
        self.assertEqual(torch.get_float32_matmul_precision(),previous)

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

    def test_shared_input_basis_decode_adjoint_and_budget(self):
        artifact=copy.deepcopy(self.artifact)
        artifact.update(codec_type='nec_sparse_excitation_schmidt_v2',amplitude_basis='backbone_input_scale')
        artifact['base_artifact']['input_scale']=torch.logspace(-1,1,64).half()
        mapper=SchmidtCoreMap(artifact,'cpu')
        native=mapper.weight(mapper.amplitudes,'amplitudes')
        base=reconstruct_packed_schmidt(artifact['base_artifact'])
        torch.testing.assert_close(reconstruct_packed_schmidt(artifact),base+native)
        torch.testing.assert_close(reconstruct_from_artifact(artifact),base+native)
        core=torch.randn_like(mapper.amplitudes);matrix=torch.randn_like(self.weight)
        torch.testing.assert_close((mapper.weight(core,'amplitudes')*matrix).sum(),
                                   (core*mapper.adjoint(matrix,'amplitudes')).sum(),atol=2e-5,rtol=2e-5)
        candidate=mapper.packed_candidate(mapper.amplitudes,'amplitudes')
        self.assertEqual(packed_schmidt_coefficient_count_from_artifact(candidate),self.artifact['logical_coefficient_count'])
        self.assertEqual(packed_schmidt_payload_bytes_from_artifact(candidate),self.artifact['payload_bytes'])

    def test_canonical_constructors_keep_guard_and_tensor_budgets(self):
        candidate,_,report=reconstruct_excitation_operator(self.weight,self.artifact,self.inputs,
            self.weight@self.inputs,sequence_length=16,methods=('pcg_canonical','feedback_canonical'))
        self.assertLessEqual(report['final_fit_relative_mse'],report['initial_fit_relative_mse'])
        self.assertLessEqual(report['final_guard_relative_mse'],report['initial_guard_relative_mse'])
        self.assertEqual(packed_schmidt_coefficient_count_from_artifact(candidate),self.artifact['logical_coefficient_count'])
        self.assertEqual(packed_schmidt_payload_bytes_from_artifact(candidate),self.artifact['payload_bytes'])

    def test_explicit_precision_reallocation_never_adds_coefficients(self):
        candidate,_,report=reconstruct_excitation_operator(self.weight,self.artifact,self.inputs,
            self.weight@self.inputs,sequence_length=16,methods=('pcg_mse',),output_bits=4)
        self.assertEqual(packed_schmidt_coefficient_count_from_artifact(candidate),self.artifact['logical_coefficient_count'])
        self.assertEqual(report['tensor_payload_bytes_added'],
            packed_schmidt_payload_bytes_from_artifact(candidate)-self.artifact['payload_bytes'])
        self.assertTrue(report['explicit_precision_reallocation'])
        self.assertEqual(candidate['payload_bytes'],packed_schmidt_payload_bytes_from_artifact(candidate))
        self.assertLessEqual(report['final_guard_relative_mse'],report['initial_guard_relative_mse'])

    def test_fresh_compilation_can_encode_in_the_existing_backbone_basis(self):
        artifact,decoded,report=compile_sparse_schmidt(self.weight,byte_budget=1800,
            coefficient_budget=2048,calibration_inputs=self.inputs,backbone_fraction=.2,
            bits_candidates=(4,),amplitude_bits=3,group_size=16,als_sweeps=0,
            excitation_sequence_length=16,amplitude_basis='backbone_input_scale')
        self.assertEqual(artifact['codec_type'],'nec_sparse_excitation_schmidt_v2')
        self.assertLessEqual(artifact['payload_bytes'],1800)
        self.assertLessEqual(artifact['logical_coefficient_count'],2048)
        torch.testing.assert_close(decoded,reconstruct_from_artifact(artifact))
        self.assertLessEqual(report['final_guard_relative_mse'],report['initial_guard_relative_mse'])

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
            cg_iterations=3, sequence_length=16, output_metric='channel_relative', ridge_metric='diagonal')
        self.assertEqual(report['output_metric'], 'channel_relative')
        self.assertEqual(report['ridge_metric'], 'diagonal')
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
