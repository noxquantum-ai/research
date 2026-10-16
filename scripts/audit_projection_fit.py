#!/usr/bin/env python3
"""Measure each saved projection on dense calibration and held-out activations."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d
from nox.compiler import quantize_4bit_grouped, reconstruct_from_artifact
from nox.utils import compute_metrics, mpo_to_dense


def projection_inputs(weights, x, eps, seq_len, n_heads, n_kv_heads):
    norm1 = RMSNorm(x.shape[0], eps=eps, weight=weights["input_layernorm"])(x)
    q, k, v = (weights[name] @ norm1 for name in ("q_proj", "k_proj", "v_proj"))
    context = scaled_dot_product_attention_2d(q, k, v, n_heads, n_kv_heads, seq_len)
    residual = x + weights["o_proj"] @ context
    norm2 = RMSNorm(x.shape[0], eps=eps, weight=weights["post_attention_layernorm"])(residual)
    swiglu = F.silu(weights["gate_proj"] @ norm2) * (weights["up_proj"] @ norm2)
    return {"q_proj": norm1, "k_proj": norm1, "v_proj": norm1, "o_proj": context,
            "gate_proj": norm2, "up_proj": norm2, "down_proj": swiglu}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint_dir / f"compressed_layer_{args.layer}.pt", map_location="cpu", weights_only=True)
    eps = checkpoint["rms_norm_eps"]
    names = list(checkpoint["compressed_artifacts"])
    weights = {name: torch.load(args.dense_weights_dir / f"layer_{args.layer}_{name}.pt", map_location="cpu", weights_only=True).to(args.device).float()
               for name in names + ["input_layernorm", "post_attention_layernorm"]}
    train = torch.load(args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt", map_location="cpu", weights_only=True)
    valid = torch.load(args.checkpoint_dir / f"dense_input_layer_{args.layer}.pt", map_location="cpu", weights_only=True)["X_test"]
    inputs = {split: projection_inputs(weights, x.to(args.device), eps, args.seq_len, args.n_heads, args.n_kv_heads)
              for split, x in (("calibration", train), ("validation", valid))}
    results = {"layer": args.layer, "rms_norm_eps": eps, "calibration_tokens": train.shape[1],
               "validation_tokens": valid.shape[1], "projections": {}}
    for name in names:
        artifact = checkpoint["compressed_artifacts"][name]
        reconstructed = reconstruct_from_artifact(artifact, device=args.device)
        core = mpo_to_dense([t.to(args.device).float() for t in artifact["cores"]]) / artifact["d_awt"].to(args.device).float().unsqueeze(0)
        bridge = artifact["A_bridge"].to(args.device).float() @ artifact["B_bridge"].to(args.device).float()
        dense_fp16 = weights[name].half().float()
        dense_int4 = quantize_4bit_grouped(weights[name])
        record = {}
        for split in inputs:
            x = inputs[split][name]
            target = weights[name] @ x
            record[split] = {
                "full": compute_metrics(target, reconstructed @ x),
                "without_core": compute_metrics(target, (reconstructed - core) @ x),
                "without_bridge": compute_metrics(target, (reconstructed - bridge) @ x),
                "dense_fp16_control": compute_metrics(target, dense_fp16 @ x),
                "dense_int4_simulation_control": compute_metrics(target, dense_int4 @ x),
                "target_norm": target.norm().item(),
            }
        results["projections"][name] = record
        print(name, json.dumps(record), flush=True)
        del core, bridge, reconstructed, dense_fp16, dense_int4
    with args.output.open("x") as output:
        json.dump(results, output, indent=2)


if __name__ == "__main__":
    main()
