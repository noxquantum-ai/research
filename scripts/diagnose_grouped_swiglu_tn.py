#!/usr/bin/env python3
"""Compare grouped native-coordinate SwiGLU TN against saved NEC on layer 0."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact  # noqa: E402
from nox.grouped_swiglu_tn import (  # noqa: E402
    apply_grouped_swiglu,
    build_grouped_swiglu_artifact,
    grouped_swiglu_features,
    grouped_swiglu_payload_bytes,
    grouped_swiglu_serialized_bytes,
    replace_grouped_down_readout,
)
from nox.native_swiglu import fit_anchored_down_readout  # noqa: E402
from nox.utils import compute_metrics  # noqa: E402
from scripts.diagnose_cross_layer_interactions import forward_details  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _metrics(reference, candidate):
    return {key: float(value) for key, value in compute_metrics(reference, candidate).items()}


def _dense_mlp(weights, inputs):
    return weights["down_proj"] @ (
        F.silu(weights["gate_proj"] @ inputs) * (weights["up_proj"] @ inputs)
    )


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.55, 0.625))
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-runtime-seconds", type=float, default=1200.0)
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-model grouped diagnostic requires CUDA")
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint_dir / f"compressed_layer_{args.layer}.pt",
        map_location="cpu", weights_only=True,
    )
    dense = {
        name: torch.load(
            args.dense_weights_dir / f"layer_{args.layer}_{name}.pt",
            map_location="cpu", weights_only=True,
        ).to(device=device, dtype=torch.float32)
        for name in PROJECTIONS
    }
    for name in ("input_layernorm", "post_attention_layernorm"):
        dense[name] = checkpoint[name].to(device=device, dtype=torch.float32)
    compressed = dict(dense)
    compressed.update({
        name: reconstruct_from_artifact(
            checkpoint["compressed_artifacts"][name], device=device, dtype=torch.float32,
        )
        for name in PROJECTIONS
    })
    calibration = torch.load(
        args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt",
        map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    heldout = torch.load(
        args.checkpoint_dir / f"layer_input_validation_layer_{args.layer}.pt",
        map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    options = {
        "n_heads": 32, "n_kv_heads": 32, "seq_len": args.seq_len,
        "rms_norm_eps": float(checkpoint.get("rms_norm_eps", 1e-5)),
    }
    dense_cal = forward_details(dense, calibration, **options)
    dense_val = forward_details(dense, heldout, **options)
    compressed_val = forward_details(compressed, heldout, **options)
    target_cal = _dense_mlp(dense, dense_cal["mlp_input"])
    target_val = _dense_mlp(dense, dense_val["mlp_input"])
    result = {
        "diagnostic": "grouped_native_coordinate_swiglu_tensor_network",
        "groups": args.groups,
        "fractions": args.fractions,
        "baseline": {
            "saved_nec_mlp_compressed_trajectory": _metrics(
                _dense_mlp(dense, compressed_val["mlp_input"]),
                compressed_val["mlp_output"],
            ),
            "saved_nec_full_block": _metrics(
                dense_val["output"], compressed_val["output"],
            ),
        },
        "candidates": {},
        "status": "INCOMPLETE",
        "uses_gradients": False,
        "uses_dense_residual": False,
        "post_compression_optimization": False,
    }
    _save(args.output, result)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for fraction in args.fractions:
        trial_started = time.perf_counter()
        artifact, calibration_features, construction_target = (
            build_grouped_swiglu_artifact(
                dense["gate_proj"], dense["up_proj"], dense["down_proj"],
                dense_cal["mlp_input"], groups=args.groups,
                coefficient_fraction=fraction, storage_dtype=torch.float16,
            )
        )
        construction_identity = _metrics(target_cal, construction_target)
        prior = artifact["down_left"].permute(1, 0, 2).reshape(
            dense["down_proj"].shape[0], -1,
        ).to(device=device, dtype=torch.float32)
        readout, selection = fit_anchored_down_readout(
            calibration_features, target_cal, prior,
            sequence_length=args.seq_len,
        )
        replace_grouped_down_readout(artifact, readout)
        output = apply_grouped_swiglu(artifact, dense_val["mlp_input"])
        block = dense_val["residual1"] + output
        active = dense_val["attention_output"] + output
        record = {
            "rank": int(artifact["rank"]),
            "float_coefficients": int(artifact["coefficient_count"]),
            "coefficient_fraction_of_dense_mlp": artifact[
                "coefficient_fraction_of_dense_mlp"
            ],
            "payload_bytes": grouped_swiglu_payload_bytes(artifact),
            "serialized_bytes": grouped_swiglu_serialized_bytes(artifact),
            "construction_target_identity": construction_identity,
            "ridge_selection": selection,
            "heldout_mlp": _metrics(target_val, output),
            "heldout_active_transform": _metrics(
                dense_val["active_transform"], active,
            ),
            "heldout_full_block": _metrics(dense_val["output"], block),
            "runtime_seconds": float(time.perf_counter() - trial_started),
        }
        result["candidates"][f"{fraction:.6f}"] = record
        _save(args.output, result)
        print(json.dumps({"fraction": fraction, **record}, sort_keys=True), flush=True)
        del artifact, calibration_features, construction_target, prior, readout, output
        torch.cuda.empty_cache()
        if time.perf_counter() - started > args.max_runtime_seconds:
            raise TimeoutError("grouped diagnostic runtime cap reached")
    torch.cuda.synchronize(device)
    result["status"] = "COMPLETE"
    result["runtime_seconds"] = float(time.perf_counter() - started)
    result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    _save(args.output, result)
    print(json.dumps({
        "status": result["status"], "runtime_seconds": result["runtime_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
    }), flush=True)


if __name__ == "__main__":
    main()
