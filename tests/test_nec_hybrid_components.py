import unittest

import torch

from nox.operator_schmidt import operator_schmidt_to_dense
from nox.packed_schmidt import reconstruct_packed_schmidt
from nox.residual_codec import decode_groupwise
from nox.sparse_schmidt import compile_sparse_schmidt
from scripts.inspect_nec_hybrid import component_counts, zero_main_mpo_cores


class HybridComponentTests(unittest.TestCase):
    def test_ablation_removes_only_the_main_mpo_and_preserves_counted_payload(self):
        torch.manual_seed(157)
        weight, inputs = torch.randn(64, 64), torch.randn(64, 128)
        artifact, original, _ = compile_sparse_schmidt(
            weight, byte_budget=1800, coefficient_budget=2048,
            calibration_inputs=inputs, backbone_fraction=.2,
            bits_candidates=(4,), amplitude_bits=3, group_size=16,
            als_sweeps=0, excitation_sequence_length=16,
        )
        base = artifact['base_artifact']
        main = operator_schmidt_to_dense(decode_groupwise(base['left_codec']),
                                         decode_groupwise(base['right_codec']), base['layout'])
        main /= base['input_scale'][None, :]
        if base.get('output_scale') is not None:
            main *= base['output_scale'][:, None]
        ablated = zero_main_mpo_cores(artifact)
        torch.testing.assert_close(reconstruct_packed_schmidt(ablated), original - main)
        torch.testing.assert_close(reconstruct_packed_schmidt(artifact), original)
        self.assertEqual(component_counts(artifact), component_counts(ablated))
        self.assertGreater(component_counts(artifact)['main_mpo_core_values'], 0)


if __name__ == '__main__':
    unittest.main()
