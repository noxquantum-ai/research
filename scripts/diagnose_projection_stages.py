#!/usr/bin/env python3
"""Controlled projection fits: stage errors with and without component quantization."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_projection_fit import projection_inputs
from nox.compiler import nox_hybrid_dmrg_decompose, reconstruct_from_artifact
from nox.sequential_compiler import get_tier_config
from nox.utils import compute_metrics, get_modes_for_projection


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--projection", default="v_proj")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-input", type=Path)
    parser.add_argument("--only-quantized", action="store_true")
    parser.add_argument("--diagonal-prior-strength", type=float, default=0.0)
    parser.add_argument("--bridge-rank-control", type=int,
                        help="Diagnostic budget control; does not change the engine default")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint_dir / f"compressed_layer_{args.layer}.pt", map_location="cpu", weights_only=True)
    names = list(checkpoint["compressed_artifacts"])
    weights = {name: torch.load(args.dense_weights_dir / f"layer_{args.layer}_{name}.pt", map_location="cpu", weights_only=True).cuda().float()
               for name in names + ["input_layernorm", "post_attention_layernorm"]}
    calibration_path = args.calibration_input or args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt"
    train = torch.load(calibration_path, map_location="cpu", weights_only=True)
    valid = torch.load(args.checkpoint_dir / f"dense_input_layer_{args.layer}.pt", map_location="cpu", weights_only=True)["X_test"]
    inputs = {split: projection_inputs(weights, x.cuda(), checkpoint["rms_norm_eps"], 128, 32, 32)[args.projection]
              for split, x in (("calibration", train), ("validation", valid))}
    weight = weights[args.projection]
    targets = {split: weight @ x for split, x in inputs.items()}
    previous = reconstruct_from_artifact(checkpoint["compressed_artifacts"][args.projection], device="cuda")
    config = get_tier_config("high_fidelity_10x")
    if args.bridge_rank_control is not None:
        if args.bridge_rank_control <= 0:
            raise ValueError("Bridge rank must be positive")
        config = dict(config, r_bridge=args.bridge_rank_control)
    output_modes, input_modes = get_modes_for_projection(*weight.shape)
    results = {"projection": args.projection, "layer": args.layer, "config": config,
               "diagonal_prior_strength": args.diagonal_prior_strength,
               "calibration_input": str(calibration_path),
               "calibration_tokens": train.shape[1], "validation_tokens": valid.shape[1], "controls": {}}
    for quantized in ((True,) if args.only_quantized else (True, False)):
        control = "int4_simulation" if quantized else "no_quantization_control"
        artifact_path = args.output.with_name(args.output.stem + "_" + control + ".pt")
        if artifact_path.exists():
            raise FileExistsError(artifact_path)
        records = []

        def measure(stage, approximation):
            record = {"stage": stage,
                      **{split: compute_metrics(targets[split], approximation @ x)
                         for split, x in inputs.items()}}
            records.append(record)
            print(control, json.dumps(record), flush=True)

        started = time.monotonic()
        fitted, _, _, _, artifact = nox_hybrid_dmrg_decompose(
            weight, inputs["calibration"], targets["calibration"],
            o_modes=output_modes, i_modes=input_modes,
            ranks=config["ranks"], r_bridge=config["r_bridge"], n_outliers=config["n_outliers"],
            quantize_components=quantized, diagnostic_callback=measure,
            diagonal_prior_strength=args.diagonal_prior_strength,
            return_artifacts=True, verbose=False,
        )
        torch.save(artifact, artifact_path)
        saved_artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        reconstructed = reconstruct_from_artifact(saved_artifact, device="cuda")
        measure("serialized_fp16", reconstructed)
        results["controls"][control] = {"stages": records, "seconds": time.monotonic() - started,
                                        "artifact_path": str(artifact_path),
                                        "artifact_bytes": artifact_path.stat().st_size,
                                        "serialization_relative_error": ((reconstructed - fitted).norm() / fitted.norm()).item()}
        if quantized and args.calibration_input is None and args.diagonal_prior_strength == 0:
            relative_difference = ((reconstructed - previous).norm() / previous.norm()).item()
            results["controls"][control]["previous_artifact_relative_difference"] = relative_difference
            if relative_difference > 1e-3:
                raise AssertionError(f"Diagnostic baseline did not reproduce the existing projection: {relative_difference}")
        del fitted, reconstructed, artifact
        gc.collect()
        torch.cuda.empty_cache()
    with args.output.open("x") as output:
        json.dump(results, output, indent=2)


if __name__ == "__main__":
    main()
