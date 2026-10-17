import io
import unittest

import torch

from nox.grouped_swiglu_tn import (
    GroupedSwiGLUModule,
    apply_grouped_swiglu,
    build_grouped_swiglu_artifact,
    grouped_rank_for_fraction,
    grouped_swiglu_coefficient_count,
    grouped_swiglu_features,
    grouped_swiglu_payload_bytes,
    grouped_swiglu_serialized_bytes,
    replace_grouped_down_readout,
)


class GroupedSwiGLUTNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.d = 8
        self.m = 12
        self.x = torch.randn(self.d, 20)
        self.gate = torch.randn(self.m, self.d)
        self.up = torch.randn(self.m, self.d)
        self.down = torch.randn(self.d, self.m)

    def test_full_group_rank_reconstructs_function(self):
        artifact, _, _ = build_grouped_swiglu_artifact(
            self.gate, self.up, self.down, self.x, groups=3, rank=4,
            storage_dtype=torch.float32,
        )
        expected = self.down @ (
            torch.nn.functional.silu(self.gate @ self.x) * (self.up @ self.x)
        )
        torch.testing.assert_close(
            apply_grouped_swiglu(artifact, self.x), expected, rtol=1e-4, atol=1e-4,
        )

    def test_factorized_module_matches_matrix_apply(self):
        artifact, _, _ = build_grouped_swiglu_artifact(
            self.gate, self.up, self.down, self.x, groups=3, rank=2,
            storage_dtype=torch.float32,
        )
        hidden = torch.randn(2, 5, self.d)
        expected = apply_grouped_swiglu(
            artifact, hidden.reshape(-1, self.d).T,
        ).T.reshape(2, 5, self.d)
        torch.testing.assert_close(GroupedSwiGLUModule(artifact)(hidden), expected)

    def test_readout_replacement_preserves_layout(self):
        artifact, features, _ = build_grouped_swiglu_artifact(
            self.gate, self.up, self.down, self.x, groups=3, rank=2,
            storage_dtype=torch.float32,
        )
        readout = torch.randn(self.d, 6)
        replace_grouped_down_readout(artifact, readout)
        torch.testing.assert_close(
            apply_grouped_swiglu(artifact, self.x),
            readout @ grouped_swiglu_features(artifact, self.x),
        )
        self.assertEqual(features.shape, (6, 20))

    def test_budget_and_storage_accounting(self):
        rank = grouped_rank_for_fraction(self.d, self.m, 3, 0.5)
        artifact, _, _ = build_grouped_swiglu_artifact(
            self.gate, self.up, self.down, self.x, groups=3, rank=rank,
        )
        expected = grouped_swiglu_coefficient_count(self.d, self.m, 3, rank)
        self.assertEqual(artifact["coefficient_count"], expected)
        self.assertEqual(grouped_swiglu_payload_bytes(artifact), 2 * expected)
        stream = io.BytesIO()
        torch.save(artifact, stream)
        self.assertEqual(stream.tell(), grouped_swiglu_serialized_bytes(artifact))

    def test_asymmetric_rank_profile_keeps_full_down_rank(self):
        artifact, features, _ = build_grouped_swiglu_artifact(
            self.gate, self.up, self.down, self.x, groups=1,
            rank_gate=2, rank_up=3, rank_down=self.d,
            storage_dtype=torch.float32,
        )
        self.assertEqual((artifact["rank_gate"], artifact["rank_up"]), (2, 3))
        self.assertEqual(artifact["rank_down"], self.d)
        self.assertEqual(tuple(artifact["down_right"].shape), (1, self.d, self.m))
        expected_count = (2 + 3 + self.d) * (self.d + self.m)
        self.assertEqual(artifact["coefficient_count"], expected_count)
        self.assertEqual(features.shape, (self.d, self.x.shape[1]))
        hidden = torch.randn(2, 5, self.d)
        flat = hidden.reshape(-1, self.d).T
        expected = apply_grouped_swiglu(artifact, flat).T.reshape_as(hidden)
        torch.testing.assert_close(GroupedSwiGLUModule(artifact)(hidden), expected)

        readout = torch.randn(self.d, self.d)
        replace_grouped_down_readout(artifact, readout)
        torch.testing.assert_close(
            apply_grouped_swiglu(artifact, self.x),
            readout @ grouped_swiglu_features(artifact, self.x),
        )


if __name__ == "__main__":
    unittest.main()
