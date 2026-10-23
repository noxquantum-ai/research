#!/usr/bin/env python3
"""Compile transformer operators into direct, byte-bounded NEC Schmidt MPOs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
ATTENTION = frozenset(("q_proj", "k_proj", "v_proj", "o_proj"))


def parse_layers(spec: str, count: int) -> tuple[int, ...]:
    result = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            begin, end = map(int, item.split("-", 1))
            result.update(range(begin, end + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"layers must be within 0..{count - 1}")
    return tuple(sorted(result))


def resolved_directory_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().mean(dim=0, keepdim=True) + eps) * weight[:, None]


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference - candidate).norm() / reference.norm().clamp_min(1e-30))


class SafetensorReader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.weight_map = json.loads(
            (model_dir / "model.safetensors.index.json").read_text()
        )["weight_map"]

    def load(self, names: list[str]) -> dict[str, torch.Tensor]:
        by_shard: dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)
        result = {}
        for shard, shard_names in by_shard.items():
            with safe_open(self.model_dir / shard, framework="pt", device="cpu") as source:
                for name in shard_names:
                    result[name] = source.get_tensor(name)
        return result


def layer_tensor_names(layer: int) -> list[str]:
    prefix = f"model.layers.{layer}."
    return [
        prefix + "input_layernorm.weight",
        prefix + "post_attention_layernorm.weight",
        *(prefix + f"self_attn.{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")),
        *(prefix + f"mlp.{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")),
    ]


@torch.no_grad()
def dense_projection_inputs(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    *,
    layer: int,
    sequence_length: int,
    n_heads: int,
    n_kv_heads: int,
    eps: float,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer}."
    norm1 = rms_norm(hidden, weights[prefix + "input_layernorm.weight"].float(), eps)
    q = weights[prefix + "self_attn.q_proj.weight"].float() @ norm1
    k = weights[prefix + "self_attn.k_proj.weight"].float() @ norm1
    v = weights[prefix + "self_attn.v_proj.weight"].float() @ norm1
    context = scaled_dot_product_attention_2d(
        q, k, v, n_heads, n_kv_heads, sequence_length,
    )
    attention = weights[prefix + "self_attn.o_proj.weight"].float() @ context
    norm2 = rms_norm(
        hidden + attention,
        weights[prefix + "post_attention_layernorm.weight"].float(),
        eps,
    )
    gate = weights[prefix + "mlp.gate_proj.weight"].float() @ norm2
    up = weights[prefix + "mlp.up_proj.weight"].float() @ norm2
    return {
        "q_proj": norm1,
        "k_proj": norm1,
        "v_proj": norm1,
        "o_proj": context,
        "gate_proj": norm2,
        "up_proj": norm2,
        "down_proj": F.silu(gate) * up,
    }


def role_multipliers(value: str | None) -> dict[str, float]:
    result = {name: 1.0 for name in PROJECTIONS}
    if value:
        supplied = json.loads(value)
        unknown = sorted(set(supplied) - set(PROJECTIONS))
        if unknown:
            raise ValueError(f"unknown projection multipliers: {unknown}")
        for name, multiplier in supplied.items():
            if not math.isfinite(float(multiplier)) or float(multiplier) <= 0:
                raise ValueError("role multipliers must be finite and positive")
            result[name] = float(multiplier)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--target-whole-model-ratio", type=float, default=5.0)
    parser.add_argument("--metadata-reserve-mib", type=float, default=16.0)
    parser.add_argument("--role-multipliers")
    parser.add_argument("--factor-bits", type=int, nargs="+", default=(2, 3))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--als-sweeps", type=int, default=1)
    parser.add_argument("--activation-alpha", type=float, default=0.5)
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument(
        "--calibration-trajectory",
        choices=("compiled_prefix", "dense"),
        default="dense",
        help="Use the actual compiled prefix or independent dense layer inputs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.target_whole_model_ratio < 5:
        parser.error("NEC whole-model target must be at least 5x")
    if args.calibration_tokens % args.sequence_length:
        parser.error("calibration tokens must be a multiple of sequence length")
    if args.metadata_reserve_mib <= 0:
        parser.error("metadata reserve must be positive")
    if not 0 <= args.activation_alpha <= 1:
        parser.error("--activation-alpha must be between zero and one")

    config = json.loads((args.model_dir / "config.json").read_text())
    layer_count = int(config["num_hidden_layers"])
    layers = parse_layers(args.layers, layer_count)
    multipliers = role_multipliers(args.role_multipliers)
    device = torch.device(args.device)
    reader = SafetensorReader(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.iterdir()):
        raise FileExistsError("output directory must be empty unless --resume is used")

    # Derive the full 32-layer plan from source tensor sizes.  A partial pilot
    # receives the same per-operator allowance as the eventual full compile.
    shape_probe = reader.load(layer_tensor_names(0))
    scalar_by_projection = {}
    for name in PROJECTIONS:
        family = "self_attn" if name in ATTENTION else "mlp"
        tensor = shape_probe[f"model.layers.0.{family}.{name}.weight"]
        scalar_by_projection[name] = int(tensor.numel())
    del shape_probe
    full_projection_scalars = layer_count * sum(scalar_by_projection.values())
    full_projection_dense_bytes = full_projection_scalars * 2
    dense_package_bytes = resolved_directory_bytes(args.model_dir)
    fixed_package_bytes = dense_package_bytes - full_projection_dense_bytes
    package_cap = math.floor(dense_package_bytes / args.target_whole_model_ratio)
    metadata_reserve_bytes = round(args.metadata_reserve_mib * 1024 * 1024)
    artifact_budget = package_cap - fixed_package_bytes - metadata_reserve_bytes
    if artifact_budget <= 0:
        raise RuntimeError("unchanged model bytes consume the requested package budget")
    weighted_scalars = layer_count * sum(
        scalar_by_projection[name] * multipliers[name] for name in PROJECTIONS
    )
    per_operator_budgets = {
        name: math.floor(
            artifact_budget * scalar_by_projection[name] * multipliers[name] / weighted_scalars
        )
        for name in PROJECTIONS
    }
    planned_artifact_budget = layer_count * sum(per_operator_budgets.values())
    preflight = {
        "dense_package_bytes": dense_package_bytes,
        "target_whole_model_ratio": args.target_whole_model_ratio,
        "package_cap_bytes": package_cap,
        "full_projection_dense_bytes": full_projection_dense_bytes,
        "fixed_package_bytes": fixed_package_bytes,
        "metadata_reserve_bytes": metadata_reserve_bytes,
        "artifact_budget_bytes": artifact_budget,
        "planned_artifact_budget_bytes": planned_artifact_budget,
        "per_operator_byte_budgets": per_operator_budgets,
        "role_multipliers": multipliers,
        "mathematically_storage_feasible": planned_artifact_budget <= artifact_budget,
    }
    (args.output / "storage_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print("PREFLIGHT " + json.dumps(preflight), flush=True)

    started = time.monotonic()
    layer_records = []
    compressed_hidden = None
    previous_layer = None
    for layer in layers:
        output_path = args.output / f"compressed_layer_{layer}.pt"
        if args.resume and output_path.is_file():
            checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
            payload = sum(int(item["payload_bytes"]) for item in checkpoint["compressed_artifacts"].values())
            layer_records.append({
                "layer": layer, "resumed": True, "payload_bytes": payload,
                "serialized_checkpoint_bytes": output_path.stat().st_size,
            })
            print(f"LAYER {layer} resumed", flush=True)
            continue
        layer_started = time.monotonic()
        prefix = f"model.layers.{layer}."
        weights = {
            name: tensor.to(device=device)
            for name, tensor in reader.load(layer_tensor_names(layer)).items()
        }
        dense_hidden = torch.load(
            args.activation_dir / f"calibration_input_layer_{layer}.pt",
            map_location=device, weights_only=True,
        ).float()[:, :args.calibration_tokens].contiguous()
        uses_compiled_prefix = (
            args.calibration_trajectory == "compiled_prefix"
            and compressed_hidden is not None
            and previous_layer is not None
            and layer == previous_layer + 1
        )
        hidden = compressed_hidden if uses_compiled_prefix else dense_hidden
        n_heads = int(config["num_attention_heads"])
        n_kv_heads = int(config.get("num_key_value_heads", n_heads))
        eps = float(config.get("rms_norm_eps", 1e-6))
        norm1 = rms_norm(hidden, weights[prefix + "input_layernorm.weight"].float(), eps)
        dense_norm1 = rms_norm(
            dense_hidden, weights[prefix + "input_layernorm.weight"].float(), eps,
        )
        dense_qkv = {
            name: weights[prefix + f"self_attn.{name}.weight"].float() @ dense_norm1
            for name in ("q_proj", "k_proj", "v_proj")
        }
        dense_context = scaled_dot_product_attention_2d(
            dense_qkv["q_proj"], dense_qkv["k_proj"], dense_qkv["v_proj"],
            n_heads, n_kv_heads, args.sequence_length,
        )
        artifacts = {}
        projection_records = {}

        reconstructed = {}
        for offset, name in enumerate(("q_proj", "k_proj", "v_proj")):
            family = "self_attn" if name in ATTENTION else "mlp"
            weight = weights[prefix + f"{family}.{name}.weight"]
            artifact, decoded, diagnostics = compile_packed_schmidt(
                weight,
                byte_budget=per_operator_budgets[name],
                projection_name=name,
                calibration_inputs=norm1,
                calibration_targets=dense_qkv[name],
                bits_candidates=args.factor_bits,
                group_size=args.group_size,
                activation_alpha=args.activation_alpha,
                als_sweeps=args.als_sweeps,
                n_heads=n_heads,
                head_dim=int(config["hidden_size"]) // n_heads,
                seed=layer * len(PROJECTIONS) + offset,
            )
            artifacts[name] = artifact
            reconstructed[name] = decoded
            projection_records[name] = diagnostics
        compressed_context = scaled_dot_product_attention_2d(
            reconstructed["q_proj"] @ norm1,
            reconstructed["k_proj"] @ norm1,
            reconstructed["v_proj"] @ norm1,
            n_heads, n_kv_heads, args.sequence_length,
        )
        name = "o_proj"
        o_weight = weights[prefix + "self_attn.o_proj.weight"]
        dense_attention = o_weight.float() @ dense_context
        artifact, decoded_o, diagnostics = compile_packed_schmidt(
            o_weight,
            byte_budget=per_operator_budgets[name],
            projection_name=name,
            calibration_inputs=compressed_context,
            calibration_targets=dense_attention,
            bits_candidates=args.factor_bits,
            group_size=args.group_size,
            activation_alpha=args.activation_alpha,
            als_sweeps=args.als_sweeps,
            n_heads=n_heads,
            head_dim=int(config["hidden_size"]) // n_heads,
            seed=layer * len(PROJECTIONS) + 3,
        )
        artifacts[name] = artifact
        projection_records[name] = diagnostics
        residual_hidden = hidden + decoded_o @ compressed_context
        dense_residual_hidden = dense_hidden + dense_attention
        norm2 = rms_norm(
            residual_hidden,
            weights[prefix + "post_attention_layernorm.weight"].float(),
            eps,
        )
        dense_norm2 = rms_norm(
            dense_residual_hidden,
            weights[prefix + "post_attention_layernorm.weight"].float(),
            eps,
        )
        dense_gate_up = {
            name: weights[prefix + f"mlp.{name}.weight"].float() @ dense_norm2
            for name in ("gate_proj", "up_proj")
        }
        for offset, name in enumerate(("gate_proj", "up_proj"), start=4):
            weight = weights[prefix + f"mlp.{name}.weight"]
            artifact, decoded, diagnostics = compile_packed_schmidt(
                weight,
                byte_budget=per_operator_budgets[name],
                projection_name=name,
                calibration_inputs=norm2,
                calibration_targets=dense_gate_up[name],
                bits_candidates=args.factor_bits,
                group_size=args.group_size,
                activation_alpha=args.activation_alpha,
                als_sweeps=args.als_sweeps,
                n_heads=n_heads,
                head_dim=int(config["hidden_size"]) // n_heads,
                seed=layer * len(PROJECTIONS) + offset,
            )
            artifacts[name] = artifact
            reconstructed[name] = decoded
            projection_records[name] = diagnostics
        compressed_active = (
            F.silu(reconstructed["gate_proj"] @ norm2)
            * (reconstructed["up_proj"] @ norm2)
        )
        name = "down_proj"
        down_weight = weights[prefix + "mlp.down_proj.weight"]
        dense_active = F.silu(dense_gate_up["gate_proj"]) * dense_gate_up["up_proj"]
        dense_mlp_output = down_weight.float() @ dense_active
        artifact, decoded_down, diagnostics = compile_packed_schmidt(
            down_weight,
            byte_budget=per_operator_budgets[name],
            projection_name=name,
            calibration_inputs=compressed_active,
            calibration_targets=dense_mlp_output,
            bits_candidates=args.factor_bits,
            group_size=args.group_size,
            activation_alpha=args.activation_alpha,
            als_sweeps=args.als_sweeps,
            n_heads=n_heads,
            head_dim=int(config["hidden_size"]) // n_heads,
            seed=layer * len(PROJECTIONS) + 6,
        )
        artifacts[name] = artifact
        projection_records[name] = diagnostics
        compressed_hidden = residual_hidden + decoded_down @ compressed_active
        previous_layer = layer
        checkpoint = {
            "layer_idx": layer,
            "compressed_artifacts": artifacts,
            "compression_path": "nec_packed_operator_schmidt_v2",
            "post_compression_healing": False,
        }
        torch.save(checkpoint, output_path)
        payload = sum(int(item["payload_bytes"]) for item in artifacts.values())
        record = {
            "layer": layer,
            "resumed": False,
            "payload_bytes": payload,
            "serialized_checkpoint_bytes": output_path.stat().st_size,
            "projection_diagnostics": projection_records,
            "calibration_input": (
                "compiled_prefix" if uses_compiled_prefix else "dense_trajectory"
            ),
            "calibration_input_relative_drift": relative_error(dense_hidden, hidden),
            "seconds": time.monotonic() - layer_started,
        }
        layer_records.append(record)
        print("LAYER " + json.dumps(record), flush=True)
        del norm1, norm2, dense_norm1, dense_norm2
        del compressed_context, compressed_active, dense_context, dense_active
        del dense_qkv, dense_gate_up, dense_attention, dense_mlp_output, dense_residual_hidden
        del residual_hidden, dense_hidden, hidden, reconstructed
        del decoded_o, decoded_down, weights, artifacts, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    checkpoint_bytes = sum(
        path.stat().st_size for path in args.output.glob("compressed_layer_*.pt")
    )
    present_layer_count = len(list(args.output.glob("compressed_layer_*.pt")))
    missing_layer_count = layer_count - present_layer_count
    projected_package_upper_bound = (
        fixed_package_bytes
        + checkpoint_bytes
        + missing_layer_count * sum(per_operator_budgets.values())
        + metadata_reserve_bytes
    )
    manifest = {
        "schema": "nec_packed_schmidt_compile_v1",
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": hashlib.sha256(
            (args.model_dir / "config.json").read_bytes()
        ).hexdigest(),
        "compiled_layers": list(layers),
        "compressed_projections": list(PROJECTIONS),
        "dense_base_matrix": False,
        "post_compression_healing": False,
        "storage_preflight": preflight,
        "calibration": {
            "tokens_per_layer": args.calibration_tokens,
            "sequence_length": args.sequence_length,
            "activation_directory": str(args.activation_dir.resolve()),
            "trajectory": args.calibration_trajectory,
        },
        "factor_bits_candidates": list(args.factor_bits),
        "group_size": args.group_size,
        "activation_alpha": args.activation_alpha,
        "gradient_free_als_sweeps": args.als_sweeps,
        "layers": layer_records,
        "serialized_checkpoint_bytes_present": checkpoint_bytes,
        "projected_complete_package_bytes_upper_bound": projected_package_upper_bound,
        "projected_complete_package_ratio_lower_bound": (
            dense_package_bytes / projected_package_upper_bound
        ),
        "complete_package_is_materialized": present_layer_count == layer_count,
        "seconds": time.monotonic() - started,
    }
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
