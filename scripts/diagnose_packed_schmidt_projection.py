#!/usr/bin/env python3
"""Compile one Llama projection into the direct packed NEC Schmidt codec.

This is a representation gate, not a model quality claim.  It uses disjoint
saved calibration and validation activations, charges the artifact against an
explicit per-operator byte budget, and reports held-out function error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.block_compiler import scaled_dot_product_attention_2d
from nox.packed_schmidt import compile_packed_schmidt


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=0, keepdim=True) + eps) * weight[:, None]


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference - candidate).norm() / reference.norm().clamp_min(1e-30))


def load_named_tensors(model_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for name in names:
        by_shard.setdefault(weight_map[name], []).append(name)
    result = {}
    for shard, shard_names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as source:
            for name in shard_names:
                result[name] = source.get_tensor(name)
    return result


@torch.no_grad()
def projection_input(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    projection: str,
    *,
    layer: int,
    sequence_length: int,
    n_heads: int,
    n_kv_heads: int,
    eps: float,
) -> torch.Tensor:
    prefix = f"model.layers.{layer}."
    norm1 = rms_norm(hidden, weights[prefix + "input_layernorm.weight"].float(), eps)
    if projection in {"q_proj", "k_proj", "v_proj"}:
        return norm1
    q = weights[prefix + "self_attn.q_proj.weight"].float() @ norm1
    k = weights[prefix + "self_attn.k_proj.weight"].float() @ norm1
    v = weights[prefix + "self_attn.v_proj.weight"].float() @ norm1
    context = scaled_dot_product_attention_2d(
        q, k, v, n_heads, n_kv_heads, sequence_length,
    )
    if projection == "o_proj":
        return context
    attention = weights[prefix + "self_attn.o_proj.weight"].float() @ context
    residual = hidden + attention
    norm2 = rms_norm(residual, weights[prefix + "post_attention_layernorm.weight"].float(), eps)
    if projection in {"gate_proj", "up_proj"}:
        return norm2
    gate = weights[prefix + "mlp.gate_proj.weight"].float() @ norm2
    up = weights[prefix + "mlp.up_proj.weight"].float() @ norm2
    return F.silu(gate) * up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--projection", choices=PROJECTIONS, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--bits-per-original-weight", type=float, default=2.6)
    parser.add_argument("--factor-bits", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--als-sweeps", type=int, default=1)
    parser.add_argument("--calibration-tokens", type=int, default=1024)
    parser.add_argument("--validation-tokens", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-artifact", type=Path)
    args = parser.parse_args()
    if args.bits_per_original_weight <= 0:
        parser.error("--bits-per-original-weight must be positive")
    if args.calibration_tokens % args.sequence_length or args.validation_tokens % args.sequence_length:
        parser.error("token counts must be multiples of sequence length")

    started = time.monotonic()
    device = torch.device(args.device)
    prefix = f"model.layers.{args.layer}."
    tensor_names = [
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        *(prefix + f"self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")),
        *(prefix + f"mlp.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")),
    ]
    weights = {
        name: value.to(device=device)
        for name, value in load_named_tensors(args.model_dir, tensor_names).items()
    }
    calibration = torch.load(
        args.activation_dir / f"calibration_input_layer_{args.layer}.pt",
        map_location=device, weights_only=True,
    ).float()[:, :args.calibration_tokens].contiguous()
    validation = torch.load(
        args.activation_dir / f"layer_input_validation_layer_{args.layer}.pt",
        map_location=device, weights_only=True,
    ).float()[:, :args.validation_tokens].contiguous()
    config = json.loads((args.model_dir / "config.json").read_text())
    common = {
        "layer": args.layer,
        "sequence_length": args.sequence_length,
        "n_heads": int(config["num_attention_heads"]),
        "n_kv_heads": int(config.get("num_key_value_heads", config["num_attention_heads"])),
        "eps": float(config.get("rms_norm_eps", 1e-6)),
    }
    train_input = projection_input(calibration, weights, args.projection, **common)
    heldout_input = projection_input(validation, weights, args.projection, **common)
    family = "self_attn" if args.projection in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    weight_name = prefix + f"{family}.{args.projection}.weight"
    weight = weights[weight_name]
    byte_budget = math.floor(weight.numel() * args.bits_per_original_weight / 8)
    artifact, reconstructed, diagnostics = compile_packed_schmidt(
        weight,
        byte_budget=byte_budget,
        projection_name=args.projection,
        calibration_inputs=train_input,
        bits_candidates=args.factor_bits,
        group_size=args.group_size,
        als_sweeps=args.als_sweeps,
        n_heads=common["n_heads"],
        head_dim=int(config["hidden_size"]) // common["n_heads"],
    )
    dense_heldout = weight.float() @ heldout_input
    compressed_heldout = reconstructed @ heldout_input
    report = {
        "schema": "nec_packed_schmidt_projection_diagnostic_v1",
        "model_dir": str(args.model_dir),
        "layer": args.layer,
        "projection": args.projection,
        "representation": artifact["representation"],
        "post_compression_healing": False,
        "dense_base_matrix": False,
        "bits_per_original_weight_budget": args.bits_per_original_weight,
        "byte_budget": byte_budget,
        "diagnostics": diagnostics,
        "heldout_relative_output_error": relative_error(dense_heldout, compressed_heldout),
        "calibration_tokens": args.calibration_tokens,
        "validation_tokens": args.validation_tokens,
        "runtime_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    encoded = json.dumps(report, indent=2) + "\n"
    print(encoded, end="", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    if args.save_artifact:
        args.save_artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.save_artifact)
        report["serialized_artifact_bytes"] = args.save_artifact.stat().st_size
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
