import unittest

import torch

from nox.quant import (HessianAccumulator, RandomizedHadamard, gptq_quantize,
                       incoherent_gptq_quantize, rtn_quantize)


def _problem(d_out=64, d_in=256, n=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(d_in, d_in, generator=g) / d_in ** 0.5
    scales = torch.logspace(0, -1.5, d_in)          # correlated, anisotropic activations
    x = (torch.randn(n, d_in, generator=g) * scales) @ mix
    w = torch.randn(d_out, d_in, generator=g)
    w[:, :4] *= 8                                    # outlier input channels
    return w, x


def _err(w, w_hat, x):
    return float(((w - w_hat) @ x.T).norm() / (w @ x.T).norm())


class QuantTests(unittest.TestCase):
    def test_hadamard_is_orthogonal_and_invertible(self):
        for n in (64, 96, 192):
            q = RandomizedHadamard(n, seed=3)
            x = torch.randn(5, n)
            self.assertTrue(torch.allclose(q.apply_t(q.apply(x)), x, atol=1e-5))
            self.assertTrue(torch.allclose(q.apply(x).norm(dim=1), x.norm(dim=1), atol=1e-4))

    def test_streaming_hessian_matches_batch(self):
        _, x = _problem()
        acc = HessianAccumulator(x.shape[1])
        for chunk in x.split(1000):
            acc.update(chunk)
        self.assertTrue(torch.allclose(acc.hessian(), x.T @ x / x.shape[0], rtol=1e-4, atol=1e-5))

    def test_gptq_beats_rtn_on_output_error(self):
        w, x = _problem()
        h = x.T @ x / x.shape[0]
        for bits in (4, 3):
            e_rtn = _err(w, rtn_quantize(w, bits, 64), x)
            e_gptq = _err(w, gptq_quantize(w, h, bits, 64, block_size=64), x)
            self.assertLess(e_gptq, e_rtn)

    def test_gptq_output_lies_on_grid(self):
        w, x = _problem()
        q = gptq_quantize(w, x.T @ x / x.shape[0], bits=2, group_size=64, block_size=64)
        groups = q.reshape(w.shape[0], -1, 64)
        levels = [len(torch.unique(row.round(decimals=5))) for row in groups.reshape(-1, 64)[:20]]
        self.assertTrue(all(n <= 4 for n in levels))

    def test_incoherence_helps_on_heavy_tailed_weights(self):
        _, x = _problem()
        h = x.T @ x / x.shape[0]
        torch.manual_seed(1)
        w = torch.distributions.StudentT(2.5).sample((64, 256))
        for bits in (4, 3):
            plain = _err(w, gptq_quantize(w, h, bits, 64, block_size=64), x)
            incoh = _err(w, incoherent_gptq_quantize(w, h, bits, 64), x)
            self.assertLess(incoh, plain)

    def test_rotated_objective_equals_original_objective(self):
        w, x = _problem()
        h = x.T @ x / x.shape[0]
        q = RandomizedHadamard(w.shape[1], seed=0)
        h_rot = q.apply(q.apply(h).T).T
        d = torch.randn_like(w) * 0.1
        d_rot = q.apply(d)
        self.assertAlmostEqual(float(torch.trace(d @ h @ d.T)),
                               float(torch.trace(d_rot @ h_rot @ d_rot.T)), places=2)

    def test_act_order_runs_and_restores_layout(self):
        w, x = _problem()
        q = gptq_quantize(w, x.T @ x / x.shape[0], 4, 64, act_order=True, block_size=64)
        self.assertEqual(q.shape, w.shape)
        self.assertLess(_err(w, q, x), 0.5)


if __name__ == "__main__":
    unittest.main()
