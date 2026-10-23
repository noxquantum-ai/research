#!/usr/bin/env python3
"""Compile transformer operators into direct, byte-bounded NEC Schmidt MPOs."""

from __future__ import annotations

import argparse
from functools import partial
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

from nox.block_compiler import ridge_compensated_up_target, scaled_dot_product_attention_2d
from nox.functional_schmidt import construct_functional_schmidt_mlp
from nox.functional_attention import construct_functional_schmidt_attention
from nox.packed_schmidt import (
    compile_packed_schmidt,
    packed_matrix_coefficient_count,
    packed_schmidt_coefficient_count_from_artifact,
    reconstruct_packed_schmidt,
)
from nox.sector_schmidt import compile_sector_schmidt, select_swiglu_sectors


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

    def coefficient_count(self) -> int:
        """Count source tensor scalars without materializing model weights."""

        total = 0
        by_shard: dict[str, list[str]] = {}
        for name, shard in self.weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        for shard, shard_names in by_shard.items():
            with safe_open(self.model_dir / shard, framework="pt", device="cpu") as source:
                for name in shard_names:
                    total += math.prod(source.get_slice(name).get_shape())
        return int(total)


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


def layer_multipliers(value: str | None, layer_count: int) -> dict[int, float]:
    result = {layer: 1.0 for layer in range(layer_count)}
    if value:
        supplied = json.loads(value)
        for raw_layer, raw_multiplier in supplied.items():
            layer = int(raw_layer)
            multiplier = float(raw_multiplier)
            if layer < 0 or layer >= layer_count:
                raise ValueError(f"layer multiplier index {layer} is outside the model")
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("layer multipliers must be finite and positive")
            result[layer] = multiplier
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--target-whole-model-ratio", type=float, default=5.0)
    parser.add_argument(
        "--min-structural-reduction", type=float, default=2.0,
        help=(
            "Minimum reduction in logical stored coefficients, independent of packing; "
            "factor values, learned scales, and residual branches all count."
        ),
    )
    parser.add_argument("--metadata-reserve-mib", type=float, default=20.0)
    parser.add_argument("--role-multipliers")
    parser.add_argument("--layer-multipliers")
    parser.add_argument("--factor-bits", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--layout-policy", choices=("semantic", "packed_efficiency"), default="semantic",
    )
    parser.add_argument(
        "--mlp-layout-policy",
        choices=("semantic", "packed_efficiency", "native_low_rank"),
        help="Optional MLP-only tensor layout; native_low_rank is a matched SVD control.",
    )
    parser.add_argument("--als-sweeps", type=int, default=1)
    parser.add_argument("--activation-alpha", type=float, default=0.5)
    parser.add_argument(
        "--input-metric", choices=("diagonal", "product_covariance"), default="diagonal",
    )
    parser.add_argument("--input-metric-ridge-ratio", type=float, default=0.01)
    parser.add_argument(
        "--mlp-input-metric", choices=("diagonal", "product_covariance"),
    )
    parser.add_argument("--calibration-branch-fraction", type=float, default=0.0)
    parser.add_argument("--calibration-branch-bits", type=int, choices=(2, 3, 4, 5, 6, 8), default=3)
    parser.add_argument("--calibration-branch-fit-tokens", type=int, default=512)
    parser.add_argument("--calibration-branch-ridge-ratio", type=float, default=1e-4)
    parser.add_argument("--column-branch-fraction", type=float, default=0.0)
    parser.add_argument("--column-branch-bits", type=int, choices=(2, 3, 4, 5, 6, 8), default=4)
    parser.add_argument("--column-branch-ridge-ratio", type=float, default=1e-3)
    parser.add_argument("--complementary-branch-fraction", type=float, default=0.0)
    parser.add_argument("--swiglu-compensation", action="store_true")
    parser.add_argument("--joint-mlp-sweeps", type=int, default=0)
    parser.add_argument("--joint-mlp-cg-iterations", type=int, default=8)
    parser.add_argument("--joint-mlp-ridge-ratio", type=float, default=0.05)
    parser.add_argument("--joint-attention-sweeps", type=int, default=0)
    parser.add_argument("--protected-mlp-coefficient-fraction", type=float, default=0.0)
    parser.add_argument("--sector-probe-dir", type=Path)
    parser.add_argument("--sector-factor-bits", type=int, choices=(2, 3, 4, 5, 6, 8), default=5)
    parser.add_argument(
        "--initialization-output", type=Path,
        help="Optional separate checkpoint directory for paired construction controls.",
    )
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
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Write and print mathematical byte/coefficient feasibility, then exit.",
    )
    args = parser.parse_args()
    if args.target_whole_model_ratio < 5:
        parser.error("NEC whole-model target must be at least 5x")
    if args.min_structural_reduction < 2:
        parser.error("NEC structural reduction target must be at least 2x")
    if args.calibration_tokens % args.sequence_length:
        parser.error("calibration tokens must be a multiple of sequence length")
    if args.metadata_reserve_mib <= 0:
        parser.error("metadata reserve must be positive")
    if not 0 <= args.activation_alpha <= 1:
        parser.error("--activation-alpha must be between zero and one")
    if not 0 <= args.calibration_branch_fraction < 1:
        parser.error("--calibration-branch-fraction must be in [0, 1)")
    if args.calibration_branch_ridge_ratio < 0:
        parser.error("--calibration-branch-ridge-ratio must be nonnegative")
    if not 0 <= args.column_branch_fraction < 1:
        parser.error("--column-branch-fraction must be in [0, 1)")
    if not 0 <= args.complementary_branch_fraction < 1:
        parser.error("--complementary-branch-fraction must be in [0, 1)")
    if (
        args.calibration_branch_fraction + args.column_branch_fraction
        + args.complementary_branch_fraction >= 1
    ):
        parser.error("combined calibration branch fractions must be below one")
    if args.column_branch_ridge_ratio < 0:
        parser.error("--column-branch-ridge-ratio must be nonnegative")
    if args.joint_mlp_sweeps < 0 or args.joint_mlp_cg_iterations < 1:
        parser.error("joint MLP sweeps must be nonnegative and CG iterations positive")
    if args.joint_attention_sweeps < 0:
        parser.error("joint attention sweeps must be nonnegative")
    if not math.isfinite(args.joint_mlp_ridge_ratio) or args.joint_mlp_ridge_ratio <= 0:
        parser.error("joint MLP ridge must be finite and positive")
    if not 0 <= args.protected_mlp_coefficient_fraction < 1:
        parser.error("protected MLP coefficient fraction must lie in [0,1)")
    if args.protected_mlp_coefficient_fraction and args.sector_probe_dir is None:
        parser.error("protected sectors require --sector-probe-dir")

    config = json.loads((args.model_dir / "config.json").read_text())
    capture_manifest_path = args.activation_dir / "model_manifest.json"
    capture_manifest = (
        json.loads(capture_manifest_path.read_text()) if capture_manifest_path.is_file() else {}
    )
    captured_sequence_length = capture_manifest.get("sequence_length")
    if captured_sequence_length is not None and captured_sequence_length != args.sequence_length:
        raise ValueError("compiler sequence length differs from activation capture")
    layer_count = int(config["num_hidden_layers"])
    layers = parse_layers(args.layers, layer_count)
    multipliers = role_multipliers(args.role_multipliers)
    per_layer_multipliers = layer_multipliers(args.layer_multipliers, layer_count)
    device = torch.device(args.device)
    reader = SafetensorReader(args.model_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.iterdir()):
        raise FileExistsError("output directory must be empty unless --resume is used")
    if args.initialization_output is not None:
        if args.initialization_output.resolve() == args.output.resolve():
            raise ValueError("initialization controls must use a separate directory")
        args.initialization_output.mkdir(parents=True, exist_ok=True)

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
    dense_model_coefficient_count = reader.coefficient_count()
    fixed_model_coefficient_count = dense_model_coefficient_count - full_projection_scalars
    structural_coefficient_cap = math.floor(
        dense_model_coefficient_count / args.min_structural_reduction
    )
    artifact_coefficient_budget = (
        structural_coefficient_cap - fixed_model_coefficient_count
    )
    if artifact_coefficient_budget <= 0:
        raise RuntimeError(
            "unchanged model coefficients consume the requested structural budget"
        )
    weighted_scalars = sum(
        per_layer_multipliers[layer]
        * sum(scalar_by_projection[name] * multipliers[name] for name in PROJECTIONS)
        for layer in range(layer_count)
    )
    per_layer_operator_budgets = {
        layer: {
            name: math.floor(
                artifact_budget
                * scalar_by_projection[name]
                * multipliers[name]
                * per_layer_multipliers[layer]
                / weighted_scalars
            )
            for name in PROJECTIONS
        }
        for layer in range(layer_count)
    }
    per_layer_operator_coefficient_budgets = {
        layer: {
            name: math.floor(
                artifact_coefficient_budget
                * scalar_by_projection[name]
                * multipliers[name]
                * per_layer_multipliers[layer]
                / weighted_scalars
            )
            for name in PROJECTIONS
        }
        for layer in range(layer_count)
    }
    planned_artifact_budget = sum(
        sum(layer_budgets.values())
        for layer_budgets in per_layer_operator_budgets.values()
    )
    planned_artifact_coefficient_budget = sum(
        sum(layer_budgets.values())
        for layer_budgets in per_layer_operator_coefficient_budgets.values()
    )
    preflight = {
        "dense_package_bytes": dense_package_bytes,
        "target_whole_model_ratio": args.target_whole_model_ratio,
        "package_cap_bytes": package_cap,
        "full_projection_dense_bytes": full_projection_dense_bytes,
        "fixed_package_bytes": fixed_package_bytes,
        "metadata_reserve_bytes": metadata_reserve_bytes,
        "artifact_budget_bytes": artifact_budget,
        "planned_artifact_budget_bytes": planned_artifact_budget,
        "per_layer_operator_byte_budgets": per_layer_operator_budgets,
        "dense_model_coefficient_count": dense_model_coefficient_count,
        "minimum_structural_reduction": args.min_structural_reduction,
        "structural_coefficient_cap": structural_coefficient_cap,
        "structural_caps": {
            "2x": math.floor(dense_model_coefficient_count / 2),
            "3x": math.floor(dense_model_coefficient_count / 3),
        },
        "fixed_model_coefficient_count": fixed_model_coefficient_count,
        "artifact_coefficient_budget": artifact_coefficient_budget,
        "planned_artifact_coefficient_budget": planned_artifact_coefficient_budget,
        "per_layer_operator_coefficient_budgets": (
            per_layer_operator_coefficient_budgets
        ),
        "role_multipliers": multipliers,
        "layer_multipliers": per_layer_multipliers,
        "mathematically_storage_feasible": planned_artifact_budget <= artifact_budget,
        "mathematically_structurally_feasible": (
            planned_artifact_coefficient_budget <= artifact_coefficient_budget
        ),
    }
    (args.output / "storage_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print("PREFLIGHT " + json.dumps(preflight), flush=True)
    if args.preflight_only:
        return

    started = time.monotonic()
    layer_records = []
    compressed_hidden = None
    previous_layer = None
    for layer in layers:
        output_path = args.output / f"compressed_layer_{layer}.pt"
        if args.resume and output_path.is_file():
            checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
            for name, item in checkpoint["compressed_artifacts"].items():
                if item["payload_bytes"] > per_layer_operator_budgets[layer][name]:
                    raise ValueError(f"resumed layer {layer} {name} exceeds its byte budget")
                if (
                    packed_schmidt_coefficient_count_from_artifact(item)
                    > per_layer_operator_coefficient_budgets[layer][name]
                ):
                    raise ValueError(
                        f"resumed layer {layer} {name} exceeds its coefficient budget"
                    )
            payload = sum(int(item["payload_bytes"]) for item in checkpoint["compressed_artifacts"].values())
            coefficient_count = sum(
                packed_schmidt_coefficient_count_from_artifact(item)
                for item in checkpoint["compressed_artifacts"].values()
            )
            layer_records.append({
                "layer": layer, "resumed": True, "payload_bytes": payload,
                "logical_coefficient_count": coefficient_count,
                "serialized_checkpoint_bytes": output_path.stat().st_size,
            })
            print(f"LAYER {layer} resumed", flush=True)
            continue
        layer_started = time.monotonic()
        per_operator_budgets = per_layer_operator_budgets[layer]
        per_operator_coefficient_budgets = (
            per_layer_operator_coefficient_budgets[layer]
        )
        prefix = f"model.layers.{layer}."
        weights = {
            name: tensor.to(device=device)
            for name, tensor in reader.load(layer_tensor_names(layer)).items()
        }
        dense_hidden = torch.load(
            args.activation_dir / f"calibration_input_layer_{layer}.pt",
            map_location=device, weights_only=True,
        ).float()[:, :args.calibration_tokens].contiguous()
        if dense_hidden.shape[1] != args.calibration_tokens:
            raise ValueError("activation cache contains fewer tokens than requested")
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
                coefficient_budget=per_operator_coefficient_budgets[name],
                projection_name=name,
                layout_policy=args.layout_policy,
                calibration_inputs=norm1,
                calibration_targets=dense_qkv[name],
                bits_candidates=args.factor_bits,
                group_size=args.group_size,
                activation_alpha=args.activation_alpha,
                input_metric=args.input_metric,
                input_metric_ridge_ratio=args.input_metric_ridge_ratio,
                calibration_branch_fraction=args.calibration_branch_fraction,
                calibration_branch_bits=args.calibration_branch_bits,
                calibration_branch_fit_tokens=args.calibration_branch_fit_tokens,
                calibration_branch_ridge_ratio=args.calibration_branch_ridge_ratio,
                column_branch_fraction=args.column_branch_fraction,
                column_branch_bits=args.column_branch_bits,
                column_branch_ridge_ratio=args.column_branch_ridge_ratio,
                complementary_branch_fraction=args.complementary_branch_fraction,
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
            coefficient_budget=per_operator_coefficient_budgets[name],
            projection_name=name,
            layout_policy=args.layout_policy,
            calibration_inputs=compressed_context,
            calibration_targets=dense_attention,
            bits_candidates=args.factor_bits,
            group_size=args.group_size,
            activation_alpha=args.activation_alpha,
            input_metric=args.input_metric,
            input_metric_ridge_ratio=args.input_metric_ridge_ratio,
            calibration_branch_fraction=args.calibration_branch_fraction,
            calibration_branch_bits=args.calibration_branch_bits,
            calibration_branch_fit_tokens=args.calibration_branch_fit_tokens,
            calibration_branch_ridge_ratio=args.calibration_branch_ridge_ratio,
            column_branch_fraction=args.column_branch_fraction,
            column_branch_bits=args.column_branch_bits,
            column_branch_ridge_ratio=args.column_branch_ridge_ratio,
            complementary_branch_fraction=args.complementary_branch_fraction,
            als_sweeps=args.als_sweeps,
            n_heads=n_heads,
            head_dim=int(config["hidden_size"]) // n_heads,
            seed=layer * len(PROJECTIONS) + 3,
        )
        artifacts[name] = artifact
        projection_records[name] = diagnostics
        joint_attention_diagnostics = None
        if args.joint_attention_sweeps:
            joint_artifacts, joint_weights, joint_attention_diagnostics = (
                construct_functional_schmidt_attention(
                    artifacts, norm1, dense_hidden + dense_attention - hidden,
                    reconstructed["q_proj"] @ norm1, reconstructed["k_proj"] @ norm1,
                    n_heads=n_heads, n_kv_heads=n_kv_heads,
                    sequence_length=args.sequence_length,
                    sweeps=args.joint_attention_sweeps,
                    cg_iterations=args.joint_mlp_cg_iterations,
                    ridge_ratio=args.joint_mlp_ridge_ratio,
                )
            )
            artifacts.update(joint_artifacts)
            reconstructed.update(joint_weights)
            decoded_o = joint_weights["o_proj"]
            compressed_context = scaled_dot_product_attention_2d(
                reconstructed["q_proj"] @ norm1,
                reconstructed["k_proj"] @ norm1,
                reconstructed["v_proj"] @ norm1,
                n_heads, n_kv_heads, args.sequence_length,
            )
            del joint_artifacts, joint_weights
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
        protected_indices = None
        sector_diagnostics = None
        if args.protected_mlp_coefficient_fraction:
            d_model = int(config["hidden_size"])
            width = int(config["intermediate_size"])
            share = math.floor(
                args.protected_mlp_coefficient_fraction
                * sum(per_operator_coefficient_budgets[name]
                      for name in ("gate_proj", "up_proj", "down_proj"))
            )
            protected_count = min(width - 256, share // (3 * d_model)) // 256 * 256
            while protected_count > 0 and (
                2 * packed_matrix_coefficient_count(protected_count, d_model, args.group_size)
                + packed_matrix_coefficient_count(d_model, protected_count, args.group_size)
                > share
            ):
                protected_count -= 256
            if protected_count <= 0:
                raise ValueError("sector allowance is too small for a 256-neuron sector")
            probe = torch.load(
                args.sector_probe_dir / f"compressed_layer_{layer}.pt",
                map_location="cpu", weights_only=True,
            )["compressed_artifacts"]
            probe_gate = reconstruct_packed_schmidt(probe["gate_proj"], device=device)
            probe_up = reconstruct_packed_schmidt(probe["up_proj"], device=device)
            protected_indices, sector_diagnostics = select_swiglu_sectors(
                weights[prefix + "mlp.gate_proj.weight"],
                weights[prefix + "mlp.up_proj.weight"],
                weights[prefix + "mlp.down_proj.weight"],
                probe_gate, probe_up, norm2, protected_count,
            )
            del probe, probe_gate, probe_up
        for offset, name in enumerate(("gate_proj", "up_proj"), start=4):
            weight = weights[prefix + f"mlp.{name}.weight"]
            target_outputs = dense_gate_up[name]
            compensation = None
            if name == "up_proj" and args.swiglu_compensation:
                compressed_gate = reconstructed["gate_proj"] @ norm2
                target_outputs, compensation_ridge, compensation_importance = (
                    ridge_compensated_up_target(
                        dense_gate_up["gate_proj"],
                        dense_gate_up["up_proj"],
                        compressed_gate,
                        down_weight=weights[prefix + "mlp.down_proj.weight"],
                        ridge_ratio=0.05,
                    )
                )
                compensation = {
                    "target": "down_weighted_swiglu_product",
                    "ridge": float(compensation_ridge),
                    "importance_min": float(compensation_importance.min()),
                    "importance_max": float(compensation_importance.max()),
                }
            mlp_constructor = (
                compile_packed_schmidt if protected_indices is None else
                partial(compile_sector_schmidt, protected_indices=protected_indices,
                        neuron_axis=0, protected_bits=args.sector_factor_bits)
            )
            artifact, decoded, diagnostics = mlp_constructor(
                weight,
                byte_budget=per_operator_budgets[name],
                coefficient_budget=per_operator_coefficient_budgets[name],
                projection_name=name,
                layout_policy=args.mlp_layout_policy or args.layout_policy,
                calibration_inputs=norm2,
                calibration_targets=target_outputs,
                bits_candidates=args.factor_bits,
                group_size=args.group_size,
                activation_alpha=args.activation_alpha,
                input_metric=args.mlp_input_metric or args.input_metric,
                input_metric_ridge_ratio=args.input_metric_ridge_ratio,
                calibration_branch_fraction=args.calibration_branch_fraction,
                calibration_branch_bits=args.calibration_branch_bits,
                calibration_branch_fit_tokens=args.calibration_branch_fit_tokens,
                calibration_branch_ridge_ratio=args.calibration_branch_ridge_ratio,
                column_branch_fraction=args.column_branch_fraction,
                column_branch_bits=args.column_branch_bits,
                column_branch_ridge_ratio=args.column_branch_ridge_ratio,
                complementary_branch_fraction=args.complementary_branch_fraction,
                als_sweeps=args.als_sweeps,
                n_heads=n_heads,
                head_dim=int(config["hidden_size"]) // n_heads,
                seed=layer * len(PROJECTIONS) + offset,
            )
            artifacts[name] = artifact
            reconstructed[name] = decoded
            if compensation is not None:
                diagnostics["swiglu_compensation"] = compensation
            projection_records[name] = diagnostics
        del target_outputs
        if args.swiglu_compensation:
            del compressed_gate, compensation_importance
        compressed_active = (
            F.silu(reconstructed["gate_proj"] @ norm2)
            * (reconstructed["up_proj"] @ norm2)
        )
        name = "down_proj"
        down_weight = weights[prefix + "mlp.down_proj.weight"]
        dense_active = F.silu(dense_gate_up["gate_proj"]) * dense_gate_up["up_proj"]
        dense_mlp_output = down_weight.float() @ dense_active
        down_constructor = (
            compile_packed_schmidt if protected_indices is None else
            partial(compile_sector_schmidt, protected_indices=protected_indices,
                    neuron_axis=1, protected_bits=args.sector_factor_bits)
        )
        artifact, decoded_down, diagnostics = down_constructor(
            down_weight,
            byte_budget=per_operator_budgets[name],
            coefficient_budget=per_operator_coefficient_budgets[name],
            projection_name=name,
            layout_policy=args.mlp_layout_policy or args.layout_policy,
            calibration_inputs=compressed_active,
            calibration_targets=dense_mlp_output,
            bits_candidates=args.factor_bits,
            group_size=args.group_size,
            activation_alpha=args.activation_alpha,
            input_metric=args.mlp_input_metric or args.input_metric,
            input_metric_ridge_ratio=args.input_metric_ridge_ratio,
            calibration_branch_fraction=args.calibration_branch_fraction,
            calibration_branch_bits=args.calibration_branch_bits,
            calibration_branch_fit_tokens=args.calibration_branch_fit_tokens,
            calibration_branch_ridge_ratio=args.calibration_branch_ridge_ratio,
            column_branch_fraction=args.column_branch_fraction,
            column_branch_bits=args.column_branch_bits,
            column_branch_ridge_ratio=args.column_branch_ridge_ratio,
            complementary_branch_fraction=args.complementary_branch_fraction,
            als_sweeps=args.als_sweeps,
            n_heads=n_heads,
            head_dim=int(config["hidden_size"]) // n_heads,
            seed=layer * len(PROJECTIONS) + 6,
        )
        artifacts[name] = artifact
        projection_records[name] = diagnostics
        joint_mlp_diagnostics = None
        if args.initialization_output is not None:
            initial_path = args.initialization_output / f"compressed_layer_{layer}.pt"
            if initial_path.exists():
                raise FileExistsError(initial_path)
            torch.save({
                "layer_idx": layer,
                "compressed_artifacts": artifacts,
                "compression_path": "nec_packed_operator_schmidt_v3",
                "construction_stage": "before_joint_mlp_construction",
                "post_compression_healing": False,
            }, initial_path)
        if args.joint_mlp_sweeps:
            joint_artifacts, joint_weights, joint_mlp_diagnostics = (
                construct_functional_schmidt_mlp(
                    artifacts, norm2,
                    dense_residual_hidden + dense_mlp_output - residual_hidden,
                    gate_targets=dense_gate_up["gate_proj"],
                    sweeps=args.joint_mlp_sweeps,
                    cg_iterations=args.joint_mlp_cg_iterations,
                    ridge_ratio=args.joint_mlp_ridge_ratio,
                    sequence_length=args.sequence_length,
                )
            )
            artifacts.update(joint_artifacts)
            reconstructed.update(joint_weights)
            compressed_active = (
                F.silu(joint_weights["gate_proj"] @ norm2)
                * (joint_weights["up_proj"] @ norm2)
            )
            decoded_down = joint_weights["down_proj"]
            del joint_artifacts, joint_weights
        compressed_hidden = residual_hidden + decoded_down @ compressed_active
        previous_layer = layer
        checkpoint = {
            "layer_idx": layer,
            "compressed_artifacts": artifacts,
            "compression_path": "nec_packed_operator_schmidt_v3",
            "post_compression_healing": False,
        }
        torch.save(checkpoint, output_path)
        payload = sum(int(item["payload_bytes"]) for item in artifacts.values())
        coefficient_count = sum(
            packed_schmidt_coefficient_count_from_artifact(item)
            for item in artifacts.values()
        )
        record = {
            "layer": layer,
            "resumed": False,
            "payload_bytes": payload,
            "logical_coefficient_count": coefficient_count,
            "serialized_checkpoint_bytes": output_path.stat().st_size,
            "projection_diagnostics": projection_records,
            "projection_diagnostics_stage": (
                "before_joint_mlp_construction" if args.joint_mlp_sweeps else "final"
            ),
            "joint_mlp_construction": joint_mlp_diagnostics,
            "joint_attention_construction": joint_attention_diagnostics,
            "nonlinear_feature_sectors": sector_diagnostics,
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
    present_layers = {
        int(path.stem.removeprefix("compressed_layer_"))
        for path in args.output.glob("compressed_layer_*.pt")
    }
    missing_artifact_budget = sum(
        sum(per_layer_operator_budgets[layer].values())
        for layer in range(layer_count)
        if layer not in present_layers
    )
    missing_artifact_coefficient_budget = sum(
        sum(per_layer_operator_coefficient_budgets[layer].values())
        for layer in range(layer_count)
        if layer not in present_layers
    )
    recorded_coefficient_counts = {
        int(record["layer"]): int(record["logical_coefficient_count"])
        for record in layer_records
    }
    present_artifact_coefficient_count = sum(
        recorded_coefficient_counts[layer]
        for layer in present_layers
        if layer in recorded_coefficient_counts
    )
    for layer in present_layers - recorded_coefficient_counts.keys():
        checkpoint = torch.load(
            args.output / f"compressed_layer_{layer}.pt",
            map_location="cpu", weights_only=True,
        )
        present_artifact_coefficient_count += sum(
            packed_schmidt_coefficient_count_from_artifact(item)
            for item in checkpoint["compressed_artifacts"].values()
        )
        del checkpoint
    projected_package_upper_bound = (
        fixed_package_bytes
        + checkpoint_bytes
        + missing_artifact_budget
        + metadata_reserve_bytes
    )
    projected_model_coefficient_upper_bound = (
        fixed_model_coefficient_count
        + present_artifact_coefficient_count
        + missing_artifact_coefficient_budget
    )
    # The model actually evaluated keeps uncompiled layers dense. A future
    # full compile projection must never be presented as its achieved ratio.
    missing_dense_coefficients = (
        layer_count - len(present_layers)
    ) * sum(scalar_by_projection.values())
    evaluated_assembly_coefficients = (
        fixed_model_coefficient_count
        + present_artifact_coefficient_count
        + missing_dense_coefficients
    )
    evaluated_assembly_package_bytes_upper_bound = (
        fixed_package_bytes + checkpoint_bytes
        + 2 * missing_dense_coefficients + metadata_reserve_bytes
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
            "activation_manifest_sha256": (
                hashlib.sha256(capture_manifest_path.read_bytes()).hexdigest()
                if capture_manifest_path.is_file() else None
            ),
            "trajectory": args.calibration_trajectory,
        },
        "factor_bits_candidates": list(args.factor_bits),
        "group_size": args.group_size,
        "layout_policy": args.layout_policy,
        "mlp_layout_policy": args.mlp_layout_policy or args.layout_policy,
        "activation_alpha": args.activation_alpha,
        "input_metric": args.input_metric,
        "mlp_input_metric": args.mlp_input_metric or args.input_metric,
        "input_metric_ridge_ratio": args.input_metric_ridge_ratio,
        "calibration_schmidt_branch": {
            "byte_fraction_per_operator": args.calibration_branch_fraction,
            "factor_bits": args.calibration_branch_bits,
            "fit_tokens": args.calibration_branch_fit_tokens,
            "ridge_ratio": args.calibration_branch_ridge_ratio,
        },
        "calibration_column_branch": {
            "coefficient_and_byte_fraction_per_operator": args.column_branch_fraction,
            "factor_bits": args.column_branch_bits,
            "ridge_ratio": args.column_branch_ridge_ratio,
        },
        "swiglu_compensation": args.swiglu_compensation,
        "joint_mlp_construction": {
            "sweeps": args.joint_mlp_sweeps,
            "cg_iterations": args.joint_mlp_cg_iterations,
            "ridge_ratio": args.joint_mlp_ridge_ratio,
            "target": "clean_whole_block_output_including_residual_stream",
        },
        "joint_attention_construction": {
            "sweeps": args.joint_attention_sweeps,
            "uses_nonlinear_derivatives": False,
        },
        "protected_mlp_sectors": {
            "coefficient_fraction": args.protected_mlp_coefficient_fraction,
            "probe_directory": str(args.sector_probe_dir) if args.sector_probe_dir else None,
            "factor_bits": args.sector_factor_bits,
        },
        "complementary_branch_fraction": args.complementary_branch_fraction,
        "gradient_free_als_sweeps": args.als_sweeps,
        "layers": layer_records,
        "serialized_checkpoint_bytes_present": checkpoint_bytes,
        "logical_artifact_coefficient_count_present": (
            present_artifact_coefficient_count
        ),
        "evaluated_assembly_coefficient_count": evaluated_assembly_coefficients,
        "evaluated_assembly_structural_reduction": (
            dense_model_coefficient_count / evaluated_assembly_coefficients
        ),
        "evaluated_assembly_package_bytes_upper_bound": (
            evaluated_assembly_package_bytes_upper_bound
        ),
        "evaluated_assembly_package_ratio_lower_bound": (
            dense_package_bytes / evaluated_assembly_package_bytes_upper_bound
        ),
        "projected_storage_budget_satisfied": (
            projected_package_upper_bound <= package_cap
        ),
        "projected_coefficient_budget_satisfied": (
            projected_model_coefficient_upper_bound <= structural_coefficient_cap
        ),
        "projected_complete_model_coefficient_count_upper_bound": (
            projected_model_coefficient_upper_bound
        ),
        "projected_complete_structural_reduction_lower_bound": (
            dense_model_coefficient_count / projected_model_coefficient_upper_bound
        ),
        "projected_complete_package_bytes_upper_bound": projected_package_upper_bound,
        "projected_complete_package_ratio_lower_bound": (
            dense_package_bytes / projected_package_upper_bound
        ),
        "all_layer_artifacts_are_materialized": present_layer_count == layer_count,
        "complete_package_is_materialized": False,
        "layer_multipliers": per_layer_multipliers,
        "seconds": time.monotonic() - started,
    }
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
