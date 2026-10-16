import unittest

import torch

from nox.packing import pack_artifact, pack_factor, unpack_artifact, unpack_factor


class PackingTests(unittest.TestCase):
    def test_signed_codes_padding_and_zero_groups(self):
        values = torch.tensor([-7, -6, -1, 0, 1, 3, 7], dtype=torch.float16)
        packed = pack_factor(values, group_size=8)
        self.assertTrue(torch.equal(unpack_factor(packed), values))
        zero = torch.zeros(3, 7, dtype=torch.float16)
        self.assertTrue(torch.equal(unpack_factor(pack_factor(zero)), zero))
        self.assertEqual(unpack_factor(packed).shape, values.shape)

    def test_known_quantized_factor_roundtrip(self):
        values = torch.arange(-7, 8).repeat(4).half() * 0.125
        self.assertTrue(torch.equal(unpack_factor(pack_factor(values)), values))

    def test_sparse_coordinates_and_values_are_lossless(self):
        artifact = {"cores": [torch.zeros(2, 2, 2, 2).half()], "d_awt": torch.ones(4).half(),
                    "A_bridge": None, "B_bridge": None, "orig_shape": (4, 4),
                    "sparse_indices": torch.tensor([[0, 1], [0, 3], [3, 0]], dtype=torch.int32),
                    "sparse_values": torch.tensor([1, -2, 3]).half()}
        packed = pack_artifact(artifact)
        self.assertEqual(packed["sparse_columns"].dtype, torch.int16)
        unpacked = unpack_artifact(packed)
        for key in ("sparse_indices", "sparse_values", "d_awt"):
            self.assertTrue(torch.equal(artifact[key], unpacked[key]))
        with self.assertRaises(ValueError):
            pack_artifact(dict(artifact, quantization_bits=16))
