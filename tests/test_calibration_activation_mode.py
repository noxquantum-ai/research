import unittest

import torch

from scripts.compile_high_fidelity import (
    resume_identity_compatible,
    select_calibration_activations,
    select_reference_activations,
)


class CalibrationActivationModeTests(unittest.TestCase):
    def test_dense_mode_always_uses_dense_stream(self):
        dense = torch.tensor([[1.0]])
        compressed = torch.tensor([[2.0]])
        self.assertIs(select_calibration_activations("dense", dense, compressed), dense)

    def test_compressed_prefix_uses_frozen_compressed_stream(self):
        dense = torch.tensor([[1.0]])
        compressed = torch.tensor([[2.0]])
        self.assertIs(select_calibration_activations("compressed_prefix", dense, compressed), compressed)

    def test_first_selected_layer_starts_from_dense_prefix_input(self):
        dense = torch.tensor([[1.0]])
        self.assertIs(select_calibration_activations("compressed_prefix", dense, None), dense)

    def test_unknown_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            select_calibration_activations("healed", torch.tensor([[1.0]]), None)

    def test_trajectory_aware_mode_keeps_clean_reference_separate(self):
        dense = torch.tensor([[1.0]])
        compressed = torch.tensor([[2.0]])
        self.assertIs(select_reference_activations("trajectory_aware_mlp", dense, compressed), dense)
        self.assertIs(select_reference_activations("trajectory_aware_all", dense, compressed), dense)

    def test_independent_mode_preserves_legacy_reference(self):
        dense = torch.tensor([[1.0]])
        calibration = torch.tensor([[2.0]])
        self.assertIs(select_reference_activations("independent", dense, calibration), calibration)

    def test_unknown_construction_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            select_reference_activations("healing", torch.tensor([[1.0]]), torch.tensor([[2.0]]))

    def test_resume_compatibility_ignores_only_source_fingerprint(self):
        previous = {"model": "pinned", "layers": [0, 1], "order": [0, 1], "source_fingerprint": "old"}
        repaired = {"model": "pinned", "layers": [0, 1], "order": (0, 1), "source_fingerprint": "new"}
        changed_science = {"model": "other", "layers": [0, 1], "source_fingerprint": "new"}
        self.assertTrue(resume_identity_compatible(previous, repaired))
        self.assertFalse(resume_identity_compatible(previous, changed_science))


if __name__ == "__main__":
    unittest.main()
