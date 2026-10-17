"""Incoherence processing with randomized Hadamard transforms.

W' = U W V^T with U, V = (Hadamard blocks x random signs). The transform is an
orthogonal product of small structured gates, applied exactly (no loss) before
quantization; H' = V H V^T keeps the objective unchanged. The inverse is applied
at load time, or folded into neighbours' weights where an exact symmetry exists.
"""

import math

import torch

from .gptq import gptq_quantize


def _hadamard(n, device, dtype):
    assert n & (n - 1) == 0, "power of two required"
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class RandomizedHadamard:
    """Orthogonal map on R^n: block-diagonal Hadamard (largest power-of-two tiles) after signs.

    For n = 2^a * m, uses m independent 2^a tiles; a second pass with a fixed
    permutation mixes tiles so no coordinate group stays isolated.
    """

    def __init__(self, n, seed=0, device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        tile = n & -n
        self.n, self.tile = n, tile
        self.signs1 = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(device, dtype)
        self.signs2 = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).to(device, dtype)
        self.perm = torch.randperm(n, generator=g).to(device)
        self.inv_perm = torch.argsort(self.perm)
        self.h = _hadamard(tile, device, dtype)

    def _tiles(self, x, signs):
        shape = x.shape
        x = (x * signs).reshape(-1, self.n // self.tile, self.tile)
        return (x @ self.h).reshape(shape)

    def _tiles_t(self, x, signs):
        shape = x.shape
        x = (x.reshape(-1, self.n // self.tile, self.tile) @ self.h.T).reshape(shape)
        return x * signs

    def apply(self, x):
        """x -> Q x along the last dimension."""
        x = self._tiles(x, self.signs1)[..., self.perm]
        return self._tiles(x, self.signs2)

    def apply_t(self, x):
        """x -> Q^T x along the last dimension (the exact inverse)."""
        x = self._tiles_t(x, self.signs2)[..., self.inv_perm]
        return self._tiles_t(x, self.signs1)


@torch.no_grad()
def incoherent_gptq_quantize(weight, hessian, bits=2, group_size=128, damp=0.01, seed=0,
                             rotate_output=True):
    """Rotate in/out spaces, GPTQ in the rotated basis, return the dense dequantized W_hat."""
    w = weight.float()
    d_out, d_in = w.shape
    v = RandomizedHadamard(d_in, seed=seed, device=w.device)
    w_rot = v.apply(w)                       # W V^T  (rows rotated along d_in)
    h_rot = v.apply(v.apply(hessian.float().to(w.device)).T).T   # V H V^T
    if rotate_output:
        u = RandomizedHadamard(d_out, seed=seed + 1, device=w.device)
        w_rot = u.apply(w_rot.T).T           # U W V^T
    q = gptq_quantize(w_rot, h_rot, bits=bits, group_size=group_size, damp=damp)
    if rotate_output:
        q = u.apply_t(q.T).T
    return v.apply_t(q)
