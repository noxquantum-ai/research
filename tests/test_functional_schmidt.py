import unittest

import torch
import torch.nn.functional as F

from nox.functional_schmidt import (
    SchmidtCoreMap, _prior_anchored_pcg, construct_functional_schmidt_mlp,
    inverse_silu_nearest_branch, readout_projected_gate_targets,
    curvature_anchor,
    native_up_preconditioner,
)
from nox.packed_schmidt import (
    compile_packed_schmidt, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)
from nox.residual_codec import encode_groupwise


class FunctionalSchmidtTests(unittest.TestCase):
    def test_exact_preconditioner_solves_normal_system_in_one_step(self):
        torch.manual_seed(263)
        a=torch.randn(6,6,dtype=torch.float64);normal=a.T@a+.1*torch.eye(6,dtype=torch.float64)
        rhs=torch.randn(6,dtype=torch.float64)
        solution,history=_prior_anchored_pcg(lambda x:normal@x,rhs,normal.diagonal(),iterations=1,
            tolerance=1e-10,preconditioner=lambda x:torch.linalg.solve(normal,x))
        torch.testing.assert_close(solution,torch.linalg.solve(normal,rhs))
        self.assertLess(history[-1]['relative_normal_residual'],1e-10)

    def test_native_preconditioners_are_positive_and_self_adjoint(self):
        torch.manual_seed(269)
        artifact,_,_=compile_packed_schmidt(torch.randn(12,8),byte_budget=4000,
                                           bits_candidates=(4,),group_size=4,als_sweeps=0)
        artifact['calibration_schmidt_branch']={
            'left_codec':encode_groupwise(torch.randn(3,12),bits=4,group_size=4),
            'right_codec':encode_groupwise(torch.randn(3,8),bits=4,group_size=4)}
        mapper=SchmidtCoreMap(artifact,'cpu');x=torch.randn(8,64);phi=F.silu(torch.randn(12,64));down=torch.randn(8,12)
        sensitivity=down.square().sum(0)*phi.square().mean(1)
        for side in ('calibration_left','calibration_right'):
            diagonal=mapper.diagonal(x.square().mean(1),sensitivity,side);ridge=.05*diagonal
            apply,_=native_up_preconditioner(mapper,side,x,phi,down,ridge,diagonal)
            self.assertIsNotNone(apply)
            a=torch.randn_like(diagonal);b=torch.randn_like(diagonal)
            self.assertGreater(float((a*apply(a)).sum()),0)
            torch.testing.assert_close((a*apply(b)).sum(),(b*apply(a)).sum(),atol=1e-4,rtol=1e-4)

    def test_native_calibration_branch_maps_and_budget_are_exact(self):
        torch.manual_seed(257)
        artifact,_,_=compile_packed_schmidt(torch.randn(12,8),byte_budget=4000,
                                           bits_candidates=(4,),group_size=4,als_sweeps=0)
        artifact['calibration_schmidt_branch']={
            'left_codec':encode_groupwise(torch.randn(3,12),bits=4,group_size=4),
            'right_codec':encode_groupwise(torch.randn(3,8),bits=4,group_size=4)}
        mapper=SchmidtCoreMap(artifact,'cpu');inputs=torch.randn(8,17);output=torch.randn(12,17)
        for side in ('calibration_left','calibration_right'):
            core=torch.randn_like(getattr(mapper,side));dense=torch.randn(12,8)
            torch.testing.assert_close((mapper.weight(core,side)*dense).sum(),(core*mapper.adjoint(dense,side)).sum(),atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(mapper.apply(core,inputs,side),mapper.weight(core,side)@inputs,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close((mapper.apply(core,inputs,side)*output).sum(),
                                       (core*mapper.apply_adjoint(output,inputs,side)).sum(),atol=2e-5,rtol=2e-5)
            candidate=mapper.packed_candidate(core,side)
            self.assertEqual(packed_schmidt_coefficient_count_from_artifact(candidate),packed_schmidt_coefficient_count_from_artifact(artifact))
            self.assertEqual(packed_schmidt_payload_bytes_from_artifact(candidate),packed_schmidt_payload_bytes_from_artifact(artifact))

    def test_diagonal_anchoring_is_invariant_to_parameter_units(self):
        torch.manual_seed(85)
        a=torch.randn(11,4,dtype=torch.float64)
        gram=a.T @ a
        rhs=torch.randn(4,dtype=torch.float64)
        units=torch.tensor([.01,.3,7.,100.],dtype=torch.float64)
        solve=lambda g,b:torch.linalg.solve(g+torch.diag(curvature_anchor(g.diagonal(),.05,'diagonal')),b)
        original=solve(gram,rhs)
        transformed=solve(units[:,None]*gram*units[None,:],rhs*units)
        torch.testing.assert_close(transformed*units,original)

    def test_inverse_silu_preserves_both_real_branches(self):
        reference = torch.tensor([-10., -3., -2., -1.27846454, -1., -.1, 0., 1., 10.])
        values = F.silu(reference)
        inverse = inverse_silu_nearest_branch(values, reference)
        torch.testing.assert_close(F.silu(inverse), values, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(inverse, reference, atol=1e-3, rtol=1e-4)
        projected = inverse_silu_nearest_branch(torch.tensor([-2.]), torch.tensor([0.]))
        torch.testing.assert_close(F.silu(projected), torch.tensor([-.27846454]), atol=1e-6, rtol=1e-6)

    def test_readout_projection_constructs_finite_linear_gate_targets(self):
        torch.manual_seed(149)
        gate, up, down = torch.randn(12, 64), torch.randn(12, 64), torch.randn(8, 12)
        residual = torch.randn(8, 64) * .01
        target = readout_projected_gate_targets(gate, up, down, residual, ridge_ratio=.05)
        self.assertTrue(torch.isfinite(target).all())
        before = residual.square().sum()
        correction = down @ ((F.silu(target) - F.silu(gate)) * up)
        self.assertLess((residual - correction).square().sum(), before)

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
            artifacts, inputs, target, gate_targets=dense["gate_proj"] @ inputs,
            cg_iterations=6, sequence_length=8,
        )
        self.assertLess(report["final_guard_relative_mse"], report["initial_guard_relative_mse"])
        self.assertLessEqual(report["final_fit_relative_mse"], report["initial_fit_relative_mse"])
        self.assertFalse(report["uses_autograd"])
        self.assertFalse(report["uses_nonlinear_derivatives"])
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
