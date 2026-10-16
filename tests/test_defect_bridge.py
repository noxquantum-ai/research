import tempfile
import unittest
from pathlib import Path

import torch

from nox.defect_bridge import (
    apply_defect_bridge,
    apply_defect_bridge_sequence,
    bridge_tensor_bytes,
    fit_hybrid_defect_bridge,
    fit_ridge_low_rank_bridge,
)


class DefectBridgeTests(unittest.TestCase):
    def test_ridge_bridge_recovers_low_rank_defect(self):
        generator = torch.Generator().manual_seed(41)
        context = torch.randn(24, 20, generator=generator)
        left = torch.randn(24, 3, generator=generator)
        right = torch.randn(24, 3, generator=generator)
        defect = left @ (right.T @ context)

        bridge_left, bridge_right, metadata = fit_ridge_low_rank_bridge(
            context, defect, rank=3, ridge_ratio=0.0,
        )
        predicted = bridge_left @ (bridge_right.T @ context)
        relative_error = torch.linalg.vector_norm(defect - predicted) / torch.linalg.vector_norm(defect)

        self.assertLess(float(relative_error), 2e-5)
        self.assertEqual(metadata["effective_rank"], 3)

    def test_hybrid_bridge_round_trip_preserves_reconstruction(self):
        generator = torch.Generator().manual_seed(42)
        context = torch.randn(32, 48, generator=generator)
        defect = torch.tanh(torch.randn(32, 48, generator=generator))
        bridge = fit_hybrid_defect_bridge(
            context,
            defect,
            linear_rank=8,
            memory_size=12,
            memory_top_k=1,
            temperature=0.02,
            ridge_ratio=0.01,
            storage_dtype=torch.float32,
        )
        before = apply_defect_bridge(context[:, :9], bridge)
        self.assertEqual(bridge_tensor_bytes(bridge), bridge["metadata"]["serialized_tensor_bytes"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.pt"
            torch.save(bridge, path)
            restored = torch.load(path, map_location="cpu", weights_only=False)
        after = apply_defect_bridge(context[:, :9], restored)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertEqual(restored["metadata"]["memory_count"], 12)

        sequence_states = context[:, :9].T.reshape(1, 3, 3, 32)
        sequence_output = apply_defect_bridge_sequence(sequence_states, restored)
        torch.testing.assert_close(sequence_output.reshape(-1, 32), before.T, rtol=0, atol=0)

    def test_memory_budget_matches_two_factor_payload(self):
        generator = torch.Generator().manual_seed(43)
        context = torch.randn(16, 10, generator=generator)
        defect = torch.randn(16, 10, generator=generator)
        bridge = fit_hybrid_defect_bridge(
            context,
            defect,
            linear_rank=4,
            memory_size=4,
            storage_dtype=torch.float16,
        )
        expected = 2 * (16 * 4 + 16 * 4) * 2
        self.assertEqual(bridge_tensor_bytes(bridge), expected)


if __name__ == "__main__":
    unittest.main()
