"""Streaming sufficient statistics for NEC projection calibration.

For a linear projection with calibration columns ``X`` and targets ``Y``, the
active NEC MPO fitter only needs the second-order products ``X X^T`` and
``Y X^T`` once the sparse support has been selected.  Accumulating those
products over token chunks avoids retaining the token axis on the accelerator.

This module deliberately does not claim that arbitrary Transformer operations
can be reduced to second-order statistics.  Attention and SwiGLU activations
must first be produced by sequence-preserving forward passes; their resulting
projection pairs can then be accumulated here.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


@dataclass
class InputCovariance:
    """Streaming covariance for a projection input before targets are formed."""

    xx: torch.Tensor
    sample_count: int

    def __post_init__(self):
        if self.xx.ndim != 2 or self.xx.shape[0] != self.xx.shape[1]:
            raise ValueError("xx must be square")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")

    def projection_statistics(self, weight):
        """Create statistics for the exact dense target ``Y = weight @ X``."""
        xx = self.xx.to(device=weight.device)
        yx = weight.to(dtype=xx.dtype) @ xx
        return CalibrationStatistics(xx=xx, yx=yx, sample_count=self.sample_count)


class InputCovarianceAccumulator:
    """Accumulate only ``X X^T`` when dense targets are known linear maps."""

    def __init__(self, dimension, *, device="cpu", accumulation_dtype=torch.float32):
        self.xx = torch.zeros((dimension, dimension), device=device, dtype=accumulation_dtype)
        self.sample_count = 0

    @torch.no_grad()
    def update(self, x):
        if x.ndim != 2 or x.shape[0] != self.xx.shape[0] or x.shape[1] == 0:
            raise ValueError("Input chunk has the wrong shape")
        x_acc = x.to(device=self.xx.device, dtype=self.xx.dtype)
        self.xx.addmm_(x_acc, x_acc.T)
        self.sample_count += int(x.shape[1])
        return self

    def finalize(self):
        if self.sample_count == 0:
            raise ValueError("Cannot finalize an empty covariance")
        return InputCovariance(xx=self.xx, sample_count=self.sample_count)


@dataclass
class CalibrationStatistics:
    """Second-order statistics for one globally fitted linear projection."""

    xx: torch.Tensor
    yx: torch.Tensor
    sample_count: int
    yy: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.xx.ndim != 2 or self.xx.shape[0] != self.xx.shape[1]:
            raise ValueError("xx must be a square [d_in, d_in] matrix")
        if self.yx.ndim != 2 or self.yx.shape[1] != self.xx.shape[0]:
            raise ValueError("yx must have shape [d_out, d_in]")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.yy is not None:
            if self.yy.ndim != 2 or self.yy.shape != (self.yx.shape[0], self.yx.shape[0]):
                raise ValueError("yy must have shape [d_out, d_out]")
        tensors = (self.xx, self.yx) if self.yy is None else (self.xx, self.yx, self.yy)
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("Calibration statistics contain non-finite values")

    @property
    def input_dimension(self):
        return int(self.xx.shape[0])

    @property
    def output_dimension(self):
        return int(self.yx.shape[0])

    def to(self, *, device=None, dtype=None):
        return CalibrationStatistics(
            xx=self.xx.to(device=device, dtype=dtype),
            yx=self.yx.to(device=device, dtype=dtype),
            yy=None if self.yy is None else self.yy.to(device=device, dtype=dtype),
            sample_count=self.sample_count,
        )

    def permuted(self, output_indices, input_indices):
        """Apply the same coordinate permutation as ``permute_projection``."""
        input_indices = input_indices.to(device=self.xx.device)
        output_indices = output_indices.to(device=self.yx.device)
        xx = self.xx.index_select(0, input_indices).index_select(1, input_indices)
        yx = self.yx.index_select(0, output_indices).index_select(1, input_indices)
        yy = None
        if self.yy is not None:
            yy = self.yy.index_select(0, output_indices).index_select(1, output_indices)
        return CalibrationStatistics(xx=xx, yx=yx, yy=yy, sample_count=self.sample_count)


class CalibrationStatisticsAccumulator:
    """Incrementally accumulate projection statistics without retaining tokens.

    ``accumulation_dtype=torch.float64`` is the safest CPU reference mode.
    Float32 is normally preferable on GPUs where FP64 throughput is low.  The
    selected dtype is recorded by callers and equivalence must be measured for
    the target hardware; accumulation order is not bitwise invariant.
    """

    def __init__(
        self,
        input_dimension,
        output_dimension,
        *,
        device="cpu",
        accumulation_dtype=torch.float64,
        include_yy=False,
    ):
        if input_dimension <= 0 or output_dimension <= 0:
            raise ValueError("Projection dimensions must be positive")
        self.xx = torch.zeros(
            (input_dimension, input_dimension), device=device, dtype=accumulation_dtype,
        )
        self.yx = torch.zeros(
            (output_dimension, input_dimension), device=device, dtype=accumulation_dtype,
        )
        self.yy = (
            torch.zeros((output_dimension, output_dimension), device=device, dtype=accumulation_dtype)
            if include_yy else None
        )
        self.sample_count = 0

    @torch.no_grad()
    def update(self, x, y):
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be [features, tokens] matrices")
        if x.shape[0] != self.xx.shape[0] or y.shape[0] != self.yx.shape[0]:
            raise ValueError("Chunk feature dimensions do not match the accumulator")
        if x.shape[1] != y.shape[1] or x.shape[1] == 0:
            raise ValueError("x and y must contain the same positive number of tokens")
        x_acc = x.to(device=self.xx.device, dtype=self.xx.dtype)
        y_acc = y.to(device=self.yx.device, dtype=self.yx.dtype)
        self.xx.addmm_(x_acc, x_acc.T)
        self.yx.addmm_(y_acc, x_acc.T)
        if self.yy is not None:
            self.yy.addmm_(y_acc, y_acc.T)
        self.sample_count += int(x.shape[1])
        return self

    def finalize(self):
        if self.sample_count == 0:
            raise ValueError("Cannot finalize empty calibration statistics")
        return CalibrationStatistics(
            xx=self.xx,
            yx=self.yx,
            yy=self.yy,
            sample_count=self.sample_count,
        )


@torch.no_grad()
def accumulate_calibration_statistics(
    x,
    y,
    *,
    chunk_size,
    device=None,
    accumulation_dtype=torch.float64,
    include_yy=False,
):
    """Accumulate exact full-dataset normal-equation statistics by columns."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected X[d_in,N] and Y[d_out,N] with matching N")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    target_device = x.device if device is None else device
    accumulator = CalibrationStatisticsAccumulator(
        x.shape[0], y.shape[0], device=target_device,
        accumulation_dtype=accumulation_dtype, include_yy=include_yy,
    )
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        accumulator.update(x[:, start:stop], y[:, start:stop])
    return accumulator.finalize()


@torch.no_grad()
def accumulate_statistics_from_chunks(
    chunks: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    input_dimension,
    output_dimension,
    device="cpu",
    accumulation_dtype=torch.float64,
    include_yy=False,
):
    """Accumulate statistics from a forward-pass chunk iterator."""
    accumulator = CalibrationStatisticsAccumulator(
        input_dimension,
        output_dimension,
        device=device,
        accumulation_dtype=accumulation_dtype,
        include_yy=include_yy,
    )
    for x_chunk, y_chunk in chunks:
        accumulator.update(x_chunk, y_chunk)
    return accumulator.finalize()
