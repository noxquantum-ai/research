"""Streaming second-moment accumulation H = sum_t x_t x_t^T for a linear layer's inputs."""

import torch


class HessianAccumulator:
    """Accumulates the input Gram matrix over arbitrarily many tokens.

    Devices without float64 (MPS) accumulate float32 partial sums that are flushed
    into a float64 CPU total every ``flush_every`` updates, bounding rounding error.
    """

    def __init__(self, dim, device="cpu", flush_every=8):
        self.dim = dim
        self.count = 0
        self.device = torch.device(device)
        self.flush_every = flush_every
        self.total = torch.zeros(dim, dim, dtype=torch.float64)
        self._partial = None
        self._pending = 0

    @torch.no_grad()
    def update(self, x):
        """x has shape (..., dim); every leading index is one token."""
        x = x.reshape(-1, self.dim)
        if self.device.type == "cpu":
            x = x.to(torch.float64)
        else:
            x = x.to(self.device, torch.float32)
        gram = x.T @ x
        self._partial = gram if self._partial is None else self._partial + gram
        self.count += x.shape[0]
        self._pending += 1
        if self._pending >= self.flush_every:
            self._flush()

    def _flush(self):
        if self._partial is not None:
            self.total += self._partial.cpu().to(torch.float64)
            self._partial, self._pending = None, 0

    def hessian(self):
        """Mean Gram matrix (float32 CPU); mean scaling keeps damping dimensionless."""
        if self.count == 0:
            raise ValueError("no tokens accumulated")
        self._flush()
        return (self.total / self.count).float()
