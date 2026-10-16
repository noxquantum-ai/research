import unittest

import torch

from scripts.compile_high_fidelity import select_calibration_activations


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


if __name__ == "__main__":
    unittest.main()
