import unittest
from itertools import permutations
from tempfile import TemporaryDirectory
from pathlib import Path

import torch

from nox.compiler import nox_hybrid_dmrg_decompose, reconstruct_from_artifact
from nox.tensor_order import (
    mode_index_permutation,
    permute_projection,
    tt_svd_relative_error,
    unpermute_projection,
    validate_site_order,
)


class TensorSiteOrderTests(unittest.TestCase):
    def test_all_orders_reindex_rectangular_projection_and_calibration_exactly(self):
        torch.manual_seed(17)
        o_modes = (2, 3, 2, 1)
        i_modes = (3, 2, 1, 2)
        W = torch.randn(12, 12)
        X = torch.randn(12, 7)
        Y = W @ X

        for order in permutations(range(4)):
            Wo, Xo, Yo, oo, io, po, pi = permute_projection(
                W, X, Y, o_modes, i_modes, order,
            )
            self.assertEqual(tuple(oo), tuple(o_modes[i] for i in order))
            self.assertEqual(tuple(io), tuple(i_modes[i] for i in order))
            torch.testing.assert_close(Wo @ Xo, Yo)
            torch.testing.assert_close(unpermute_projection(Wo, po, pi), W)

    def test_artifact_reconstruction_inverts_site_order_after_mpo_decode(self):
        # A Kronecker MPO with nontrivial values makes the metadata path exercise
        # the actual reconstruction helper rather than only its index utility.
        A = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        B = torch.tensor([[2.0, -1.0], [4.0, 3.0]])
        one = torch.ones(1, 1, 1, 1)
        cores = [
            A.reshape(1, 2, 2, 1),
            B.reshape(1, 2, 2, 1),
            one,
            one,
        ]
        ordered = torch.kron(A, B)
        order = (1, 0, 2, 3)
        modes = (2, 2, 1, 1)
        out_perm = mode_index_permutation(modes, order)
        in_perm = mode_index_permutation(modes, order)
        original = unpermute_projection(ordered, out_perm, in_perm)
        artifact = {
            "cores": [core.half() for core in cores],
            "d_awt": torch.ones(4, dtype=torch.float16),
            "A_bridge": None,
            "B_bridge": None,
            "sparse_indices": None,
            "sparse_values": None,
            "site_order": order,
            "original_o_modes": modes,
            "original_i_modes": modes,
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "projection.pt"
            torch.save(artifact, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
        torch.testing.assert_close(
            reconstruct_from_artifact(loaded, dtype=torch.float32),
            original,
        )

    def test_rejects_non_permutations(self):
        for invalid in ((0, 1, 1, 3), (0, 1, 2), (0, 1, 2, 4)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_site_order(invalid)

    def test_full_rank_tt_svd_is_order_invariant(self):
        torch.manual_seed(31)
        W = torch.randn(16, 16)
        modes = (2, 2, 2, 2)
        ranks = (1, 4, 16, 4, 1)
        for order in permutations(range(4)):
            error = tt_svd_relative_error(W, modes, modes, ranks, order)
            self.assertLess(error, 1e-6)

    def test_compiler_returns_original_coordinate_matrix_and_serializable_artifact(self):
        torch.manual_seed(9)
        W = torch.randn(16, 16)
        X = torch.randn(16, 24)
        Y = W @ X
        options = dict(
            W=W,
            X_calib=X,
            Y_dense=Y,
            o_modes=(2, 2, 2, 2),
            i_modes=(2, 2, 2, 2),
            ranks=(1, 4, 16, 4, 1),
            r_bridge=0,
            n_outliers=0,
            n_sweeps=1,
            prior_gamma=0.01,
            alpha_awt=0.0,
            return_artifacts=True,
            verbose=False,
            quantize_components=False,
        )
        identity = nox_hybrid_dmrg_decompose(**options)
        reordered = nox_hybrid_dmrg_decompose(**options, site_order=(2, 0, 3, 1))
        torch.testing.assert_close(reordered[0], identity[0], atol=2e-4, rtol=2e-4)
        decoded = reconstruct_from_artifact(reordered[4], dtype=torch.float32)
        torch.testing.assert_close(decoded, reordered[0], atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
