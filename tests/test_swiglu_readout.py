import unittest

import torch

from nox.joint_swiglu import apply_joint_swiglu, build_joint_swiglu_artifact
from nox.swiglu_readout import (
    fit_linear_readout,
    select_calibration_ridge_readout,
    sequence_block_split,
    shared_swiglu_features,
)


class SwiGLUReadoutTests(unittest.TestCase):
    def test_features_match_existing_factorized_execution(self):
        torch.manual_seed(1)
        d_model, intermediate, samples = 5, 9, 16
        gate = torch.randn(intermediate, d_model)
        up = torch.randn(intermediate, d_model)
        down = torch.randn(d_model, intermediate)
        inputs = torch.randn(d_model, samples)
        artifact = build_joint_swiglu_artifact(
            gate, up, down, inputs, rank=4, storage_dtype=torch.float32,
        )
        features = shared_swiglu_features(artifact, inputs)
        expected = artifact["down_factor"] @ features
        torch.testing.assert_close(apply_joint_swiglu(artifact, inputs), expected)

    def test_unregularized_readout_recovers_exact_linear_target(self):
        torch.manual_seed(2)
        features = torch.randn(5, 30)
        truth = torch.randn(7, 5)
        target = truth @ features
        fitted = fit_linear_readout(features, target)
        torch.testing.assert_close(fitted @ features, target, rtol=2e-5, atol=2e-5)

    def test_sequence_split_keeps_whole_sequences_disjoint(self):
        fit, select = sequence_block_split(32, 4, validation_stride=4)
        self.assertEqual(set(fit.tolist()).intersection(select.tolist()), set())
        self.assertEqual(select.tolist(), list(range(12, 16)) + list(range(28, 32)))

    def test_ridge_selection_uses_only_supplied_calibration(self):
        torch.manual_seed(3)
        features = torch.randn(4, 32)
        target = torch.randn(6, 4) @ features + 0.01 * torch.randn(6, 32)
        readout, metadata = select_calibration_ridge_readout(
            features, target, sequence_length=4,
            ridge_ratios=(0.0, 1e-4, 1e-2),
        )
        self.assertEqual(tuple(readout.shape), (6, 4))
        self.assertFalse(metadata["heldout_used_for_selection"])
        self.assertEqual(metadata["fit_samples"], 24)
        self.assertEqual(metadata["selection_samples"], 8)
        self.assertIn(metadata["selected_ridge_ratio"], (0.0, 1e-4, 1e-2))


if __name__ == "__main__":
    unittest.main()
