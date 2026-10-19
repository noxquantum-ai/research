#!/usr/bin/env python3
"""Measure full-matrix versus sufficient-statistics NEC on identical data.

The default fixture is deliberately small enough for CPU regression profiling.
Use ``--device cuda --dimension ...`` on a rented smoke-test GPU for capacity
measurements.  This script never extrapolates a 32-layer runtime on its own.
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.calibration import CalibrationStatisticsAccumulator
from nox.compiler import (
    nox_hybrid_dmrg_decompose,
    nox_hybrid_dmrg_decompose_from_statistics,
)


class PeakMonitor:
    def __init__(self, device):
        self.process = psutil.Process(os.getpid())
        self.device = torch.device(device)
        self.peak_rss = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.wait(0.005):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.start_rss = self.process.memory_info().rss
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.started
        self.stop_event.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        if self.device.type == "cuda":
            self.cuda_allocated = torch.cuda.max_memory_allocated(self.device)
            self.cuda_reserved = torch.cuda.max_memory_reserved(self.device)
        else:
            self.cuda_allocated = self.cuda_reserved = 0

    def record(self):
        return {
            "elapsed_seconds": self.elapsed,
            "rss_start_bytes": self.start_rss,
            "rss_peak_bytes": self.peak_rss,
            "rss_peak_delta_bytes": max(0, self.peak_rss - self.start_rss),
            "cuda_peak_allocated_bytes": self.cuda_allocated,
            "cuda_peak_reserved_bytes": self.cuda_reserved,
        }


def parse_positive_tuple(value):
    values = tuple(int(item) for item in value.split(","))
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def relative_error(actual, expected):
    return float(
        (torch.linalg.vector_norm(actual.float() - expected.float())
         / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)).item()
    )


def input_chunks(input_dimension, tokens, chunk_size, device, seed):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for start in range(0, tokens, chunk_size):
        width = min(chunk_size, tokens - start)
        yield torch.randn(
            input_dimension, width, device=device, generator=generator,
        )


def chunked_output_error(actual_weight, target_weight, tokens, chunk_size, seed):
    numerator = torch.zeros((), device=actual_weight.device, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for inputs in input_chunks(
        target_weight.shape[1], tokens, chunk_size, target_weight.device, seed,
    ):
        target = target_weight @ inputs
        difference = actual_weight @ inputs - target
        numerator += difference.double().square().sum()
        denominator += target.double().square().sum()
    return float(torch.sqrt(numerator / denominator.clamp_min(1e-30)).item())


@torch.no_grad()
def run_case(tokens, args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if args.weight_file is None:
        weight = torch.randn(
            args.output_dimension, args.input_dimension, device=device,
        )
    else:
        loaded = torch.load(args.weight_file, map_location="cpu", weights_only=False)
        if not isinstance(loaded, torch.Tensor) or loaded.ndim != 2:
            raise ValueError("--weight-file must contain one matrix tensor")
        weight = loaded.to(device=device, dtype=torch.float32)
    if weight.shape != (
        int(torch.tensor(args.output_modes).prod().item()),
        int(torch.tensor(args.input_modes).prod().item()),
    ):
        raise ValueError("Weight shape does not match output/input modes")
    input_seed = args.seed + 1_000_003 + tokens
    common = dict(
        o_modes=args.output_modes,
        i_modes=args.input_modes,
        ranks=args.ranks,
        r_bridge=args.bridge_rank,
        n_outliers=args.outliers,
        n_sweeps=args.sweeps,
        prior_gamma=0.01,
        alpha_awt=0.5,
        chunk_size=args.dmrg_chunk_size,
        quantize_components=args.quantize,
        return_artifacts=True,
        verbose=False,
    )

    with PeakMonitor(device) as full_monitor:
        inputs = torch.cat(list(input_chunks(
            weight.shape[1], tokens, args.statistics_chunk_size, device, input_seed,
        )), dim=1)
        targets = weight @ inputs
        full = nox_hybrid_dmrg_decompose(
            W=weight, X_calib=inputs, Y_dense=targets, **common,
        )
    full_calibration_error = relative_error(full[0] @ inputs, targets)
    del inputs, targets
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with PeakMonitor(device) as statistics_monitor:
        accumulator = CalibrationStatisticsAccumulator(
            weight.shape[1],
            weight.shape[0],
            device=device,
            accumulation_dtype=args.accumulation_dtype,
        )
        for inputs in input_chunks(
            weight.shape[1], tokens, args.statistics_chunk_size, device, input_seed,
        ):
            accumulator.update(inputs, weight @ inputs)
        statistics = accumulator.finalize()
        streamed = nox_hybrid_dmrg_decompose_from_statistics(
            W=weight, calibration_statistics=statistics, **common,
        )

    heldout = torch.randn(weight.shape[1], min(tokens, 1024), device=device)
    record = {
        "tokens": tokens,
        "weight_shape": list(weight.shape),
        "full_matrix": full_monitor.record(),
        "streaming_statistics": statistics_monitor.record(),
        "relative_reconstructed_weight_difference": relative_error(streamed[0], full[0]),
        "full_weight_relative_error": relative_error(full[0], weight),
        "streamed_weight_relative_error": relative_error(streamed[0], weight),
        "full_calibration_relative_error": full_calibration_error,
        "streamed_calibration_relative_error": chunked_output_error(
            streamed[0], weight, tokens, args.statistics_chunk_size, input_seed,
        ),
        "full_heldout_relative_error": relative_error(full[0] @ heldout, weight @ heldout),
        "streamed_heldout_relative_error": relative_error(streamed[0] @ heldout, weight @ heldout),
        "full_ranks": full[3],
        "streamed_ranks": streamed[3],
        "full_artifact_payload_kib": full[1],
        "streamed_artifact_payload_kib": streamed[1],
        "streamed_numerical_rank": streamed[4]["calibration_statistics_diagnostics"]["numerical_rank"],
    }
    del full, streamed, statistics, heldout, weight
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="256,1024,4096")
    parser.add_argument("--input-dimension", type=int, default=64)
    parser.add_argument("--output-dimension", type=int, default=64)
    parser.add_argument("--input-modes", type=parse_positive_tuple, default=(2, 4, 4, 2))
    parser.add_argument("--output-modes", type=parse_positive_tuple, default=(2, 4, 4, 2))
    parser.add_argument("--ranks", type=parse_positive_tuple, default=(1, 4, 8, 4, 1))
    parser.add_argument("--weight-file", type=Path)
    parser.add_argument("--bridge-rank", type=int, default=8)
    parser.add_argument("--outliers", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--dmrg-chunk-size", type=int, default=16)
    parser.add_argument("--statistics-chunk-size", type=int, default=128)
    parser.add_argument("--accumulation-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if len(args.input_modes) != 4 or len(args.output_modes) != 4:
        raise ValueError("input and output modes must each contain four values")
    if len(args.ranks) != 5 or args.ranks[0] != 1 or args.ranks[-1] != 1:
        raise ValueError("ranks must contain five values with unit boundary ranks")
    if args.weight_file is None:
        if args.input_dimension != int(torch.tensor(args.input_modes).prod().item()):
            raise ValueError("input-dimension must equal the product of input-modes")
        if args.output_dimension != int(torch.tensor(args.output_modes).prod().item()):
            raise ValueError("output-dimension must equal the product of output-modes")
    args.accumulation_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.accumulation_dtype]
    records = [run_case(int(tokens), args) for tokens in args.tokens.split(",")]
    print(json.dumps({
        "scope": "synthetic CPU/GPU projection parity benchmark; not a Llama quality result",
        "torch_version": torch.__version__,
        "device": str(torch.device(args.device)),
        "records": records,
    }, indent=2))


if __name__ == "__main__":
    main()
