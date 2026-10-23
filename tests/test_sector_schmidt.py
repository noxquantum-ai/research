import unittest

import torch

from nox.compiler import reconstruct_from_artifact
from nox.functional_schmidt import SchmidtCoreMap, construct_functional_schmidt_mlp
from nox.packed_schmidt import (
    packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact,
)
from nox.residual_codec import decode_groupwise
from nox.sector_schmidt import compile_sector_schmidt, select_swiglu_sectors


class SectorSchmidtTests(unittest.TestCase):
    def test_joint_construction_operates_on_bulk_without_changing_sector_budgets(self):
        torch.manual_seed(103)
        dense = {
            "gate_proj": torch.randn(12, 8) * 0.2,
            "up_proj": torch.randn(12, 8) * 0.2,
            "down_proj": torch.randn(8, 12) * 0.2,
        }
        x = torch.randn(8, 32)
        hidden = torch.nn.functional.silu(dense["gate_proj"] @ x) * (dense["up_proj"] @ x)
        artifacts = {}
        for role, weight in dense.items():
            artifacts[role], _, _ = compile_sector_schmidt(
                weight, protected_indices=torch.tensor([1, 4, 7]),
                neuron_axis=1 if role == "down_proj" else 0,
                protected_bits=5, byte_budget=2000, coefficient_budget=500,
                calibration_inputs=hidden if role == "down_proj" else x,
                bits_candidates=(5,), group_size=4, als_sweeps=0,
            )
        updated, _, report = construct_functional_schmidt_mlp(
            artifacts, x, dense["down_proj"] @ hidden,
            gate_targets=dense["gate_proj"] @ x,
            cg_iterations=3, sequence_length=8,
        )
        self.assertLessEqual(report["final_guard_relative_mse"], report["initial_guard_relative_mse"])
        for role in artifacts:
            self.assertEqual(
                packed_schmidt_coefficient_count_from_artifact(artifacts[role]),
                packed_schmidt_coefficient_count_from_artifact(updated[role]),
            )
            torch.testing.assert_close(
                decode_groupwise(artifacts[role]["protected_codec"]),
                decode_groupwise(updated[role]["protected_codec"]), atol=0, rtol=0,
            )

    def test_sector_roundtrip_and_adjoint_for_rows_and_columns(self):
        torch.manual_seed(101)
        protected = torch.tensor([1, 4, 7])
        for axis, shape in ((0, (12, 8)), (1, (8, 12))):
            weight = torch.randn(shape)
            inputs = torch.randn(shape[1], 32)
            artifact, decoded, _ = compile_sector_schmidt(
                weight, protected_indices=protected, neuron_axis=axis,
                protected_bits=5, byte_budget=2000, coefficient_budget=500,
                calibration_inputs=inputs, bits_candidates=(5,),
                group_size=4, als_sweeps=0,
            )
            torch.testing.assert_close(reconstruct_from_artifact(artifact), decoded)
            torch.testing.assert_close(
                decoded.index_select(axis, protected),
                decode_groupwise(artifact["protected_codec"]),
            )
            mapper = SchmidtCoreMap(artifact, "cpu")
            for side in ("left", "right"):
                core = torch.randn_like(mapper.left if side == "left" else mapper.right)
                dense = torch.randn_like(weight)
                torch.testing.assert_close(
                    (mapper.weight(core, side) * dense).sum(),
                    (core * mapper.adjoint(dense, side)).sum(),
                    atol=2e-5, rtol=2e-5,
                )
                self.assertEqual(float(mapper.weight(core, side).index_select(axis, protected).abs().max()), 0)
            self.assertEqual(artifact["logical_coefficient_count"],
                             packed_schmidt_coefficient_count_from_artifact(artifact))
            self.assertLessEqual(artifact["logical_coefficient_count"], 500)
            self.assertEqual(artifact["payload_bytes"], packed_schmidt_payload_bytes_from_artifact(artifact))
            self.assertLessEqual(artifact["payload_bytes"], 2000)

    def test_intervention_score_finds_the_damaged_nonlinear_feature(self):
        gate = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        up = gate.clone()
        down = torch.eye(3)
        inputs = torch.ones(2, 8)
        damaged_up = up.clone()
        damaged_up[1] = 0
        selected, metadata = select_swiglu_sectors(
            gate, up, down, gate, damaged_up, inputs, 1,
        )
        self.assertEqual(selected.tolist(), [1])
        self.assertFalse(metadata["score_sum_is_joint_error_reduction"])


if __name__ == "__main__":
    unittest.main()
