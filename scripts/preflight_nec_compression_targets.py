#!/usr/bin/env python3
"""CPU-only storage feasibility preflight for a scoped NEC layer experiment.

This combines exact source tensor-payload sizes with available serialized NEC
checkpoint sizes. It does not build a deployable mixed-model bundle, and its
scoped estimates must not be reported as measured package compression.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


PROJECTION_NORMS = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)


def parse_layers(spec: str) -> tuple[int, ...]:
    """Parse comma-separated layer IDs and inclusive ranges (for example 0-15)."""
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            begin, end = int(left), int(right)
            if begin < 0 or end < begin:
                raise ValueError(f"Invalid layer range: {item}")
            layers.update(range(begin, end + 1))
        else:
            layer = int(item)
            if layer < 0:
                raise ValueError(f"Layer IDs must be nonnegative: {item}")
            layers.add(layer)
    if not layers:
        raise ValueError("At least one candidate layer is required")
    return tuple(sorted(layers))


def list_model_files(model_dir: Path) -> list[Path]:
    return sorted(path for path in model_dir.rglob("*") if path.is_file())


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Not a safetensors file (short header): {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"Truncated safetensors header: {path}")
    return json.loads(header)


def inspect_dense_model(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"No weight_map in {index_path}")

    shard_names = sorted(set(weight_map.values()))
    tensor_payload: dict[str, int] = {}
    for shard_name in shard_names:
        header = read_safetensors_header(model_dir / shard_name)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            offsets = metadata.get("data_offsets")
            if not offsets or len(offsets) != 2:
                raise ValueError(f"Missing data_offsets for {name} in {shard_name}")
            tensor_payload[name] = int(offsets[1]) - int(offsets[0])

    indexed_keys = set(weight_map)
    header_keys = set(tensor_payload)
    if indexed_keys != header_keys:
        missing = sorted(indexed_keys - header_keys)[:5]
        unexpected = sorted(header_keys - indexed_keys)[:5]
        raise ValueError(
            "Safetensors index/header mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    files = list_model_files(model_dir)
    file_bytes = {str(path.relative_to(model_dir)): path.stat().st_size for path in files}
    model_bytes = sum(file_bytes.values())
    shard_bytes = sum(file_bytes[name] for name in shard_names)
    metadata_bytes = sum(
        size for name, size in file_bytes.items()
        if name not in shard_names and name != "model.safetensors.index.json"
    )
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    total_layers = config.get("num_hidden_layers")
    if total_layers is None:
        layer_ids = [
            int(name.split("model.layers.", 1)[1].split(".", 1)[0])
            for name in tensor_payload if name.startswith("model.layers.")
        ]
        total_layers = max(layer_ids) + 1 if layer_ids else None
    return {
        "model_bytes": model_bytes,
        "file_count": len(files),
        "weight_shards": shard_names,
        "weight_shard_bytes": shard_bytes,
        "metadata_bytes_excluding_weight_shards_and_index": metadata_bytes,
        "source_index_bytes": file_bytes["model.safetensors.index.json"],
        "weight_payload_bytes": sum(tensor_payload.values()),
        "tensor_payload_bytes": tensor_payload,
        "total_layers": int(total_layers) if total_layers is not None else None,
        "files": file_bytes,
    }


def layer_compressible_payload(tensor_payload: dict[str, int], layer: int) -> int:
    prefix = f"model.layers.{layer}."
    suffixes = set(PROJECTION_NORMS)
    return sum(
        size for name, size in tensor_payload.items()
        if name.startswith(prefix) and name[len(prefix):] in suffixes
    )


def checkpoint_scenarios(checkpoint_dir: Path, candidate_layers: tuple[int, ...]) -> dict:
    observed = {}
    for layer in candidate_layers:
        path = checkpoint_dir / f"compressed_layer_{layer}.pt"
        if path.is_file():
            observed[layer] = path.stat().st_size
    if not observed:
        raise FileNotFoundError(
            f"No compressed_layer_N.pt artifacts found for candidates in {checkpoint_dir}"
        )
    sizes = list(observed.values())
    missing = [layer for layer in candidate_layers if layer not in observed]
    mean_size = sum(sizes) / len(sizes)
    return {
        "observed_by_layer": {str(layer): size for layer, size in observed.items()},
        "missing_layers": missing,
        "observed_min_bytes": min(sizes),
        "observed_mean_bytes": mean_size,
        "observed_max_bytes": max(sizes),
        "all_candidates_sum_actual_and_missing_estimated": {
            "optimistic": sum(sizes) + len(missing) * min(sizes),
            "nominal": sum(sizes) + len(missing) * mean_size,
            "conservative": sum(sizes) + len(missing) * max(sizes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-layers", required=True,
        help="Comma-separated IDs/ranges, e.g. '0-15' or '0-3,8-11'",
    )
    parser.add_argument(
        "--scope-layers",
        help="Layer-weight scope used for this experiment; defaults to candidate layers",
    )
    parser.add_argument("--max-dense-weight-fraction", type=float, default=0.25)
    parser.add_argument(
        "--max-dense-layers", type=int,
        help="Maximum dense layer count within the candidate layer scope",
    )
    parser.add_argument(
        "--whole-model-storage", action="store_true",
        help="Also assess the ratio against all serialized source model files",
    )
    parser.add_argument(
        "--compression-targets", type=float, nargs="+", default=(3.0, 10.0),
        help="Scoped layer-storage ratios to assess; default: 3x 10x",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not 0 <= args.max_dense_weight_fraction <= 1:
        parser.error("--max-dense-weight-fraction must be between 0 and 1")
    if args.max_dense_layers is not None and args.max_dense_layers < 0:
        parser.error("--max-dense-layers must be nonnegative")
    if any(target <= 1 for target in args.compression_targets):
        parser.error("Compression-ratio targets must be greater than 1")
    candidate_layers = parse_layers(args.candidate_layers)
    scope_layers = parse_layers(args.scope_layers or args.candidate_layers)
    model = inspect_dense_model(args.model_dir)
    if model["total_layers"] is None:
        raise ValueError("Could not infer transformer layer count from model config")
    outside_range = [layer for layer in candidate_layers if layer >= model["total_layers"]]
    if outside_range:
        raise ValueError(f"Candidate IDs outside model range: {outside_range}")
    outside_scope = sorted(set(candidate_layers) - set(scope_layers))
    if outside_scope:
        raise ValueError(f"Candidate layers must be inside the experiment scope: {outside_scope}")
    invalid_scope = [layer for layer in scope_layers if layer >= model["total_layers"]]
    if invalid_scope:
        raise ValueError(f"Scope layer IDs outside model range: {invalid_scope}")

    per_layer_payload = {
        layer: layer_compressible_payload(model["tensor_payload_bytes"], layer)
        for layer in candidate_layers
    }
    zero_payload = [layer for layer, size in per_layer_payload.items() if size == 0]
    if zero_payload:
        raise ValueError(f"No expected NEC projection/norm tensors found in layers {zero_payload}")
    suffixes = set(PROJECTION_NORMS)
    incomplete_layers = {
        layer: sorted(
            suffixes - {
                name[len(f"model.layers.{layer}."):]
                for name in model["tensor_payload_bytes"]
                if name.startswith(f"model.layers.{layer}.")
            }
        )
        for layer in candidate_layers
    }
    incomplete_layers = {layer: missing for layer, missing in incomplete_layers.items() if missing}
    if incomplete_layers:
        raise ValueError(f"Missing expected NEC tensors in candidate layers: {incomplete_layers}")
    scope_layer_payload = {
        layer: sum(
            size for name, size in model["tensor_payload_bytes"].items()
            if name.startswith(f"model.layers.{layer}.")
        )
        for layer in scope_layers
    }
    empty_scope = [layer for layer, size in scope_layer_payload.items() if size == 0]
    if empty_scope:
        raise ValueError(f"No model tensors found in scope layers {empty_scope}")
    scope_weight_payload = sum(scope_layer_payload.values())
    candidate_dense_payload = sum(per_layer_payload.values())
    candidate_artifacts = checkpoint_scenarios(args.checkpoint_dir, candidate_layers)

    observed_min = candidate_artifacts["observed_min_bytes"]
    observed_mean = candidate_artifacts["observed_mean_bytes"]
    observed_max = candidate_artifacts["observed_max_bytes"]
    candidate_count_frontier = []
    sorted_layer_payloads = sorted(per_layer_payload.values(), reverse=True)
    for compressed_count in range(len(candidate_layers) + 1):
        compressed_dense_payload = sum(sorted_layer_payloads[:compressed_count])
        dense_payload = scope_weight_payload - compressed_dense_payload
        checkpoint_estimates = {
            "optimistic": compressed_count * observed_min,
            "nominal": round(compressed_count * observed_mean),
            "conservative": compressed_count * observed_max,
        }
        estimates = {}
        for label, checkpoint_bytes in checkpoint_estimates.items():
            total_lower_bound = dense_payload + checkpoint_bytes
            estimates[label] = {
                "checkpoint_bytes_estimated": checkpoint_bytes,
                "scope_package_bytes_lower_bound": total_lower_bound,
                "scope_compression_ratio_upper_bound": (
                    scope_weight_payload / total_lower_bound if total_lower_bound else None
                ),
                "dense_weight_payload_fraction": dense_payload / scope_weight_payload,
                "meets_max_dense_weight_fraction": (
                    dense_payload / scope_weight_payload <= args.max_dense_weight_fraction
                ),
                "meets_max_dense_layer_count": (
                    args.max_dense_layers is None
                    or len(scope_layers) - compressed_count <= args.max_dense_layers
                ),
                "ratio_targets": {
                    str(target): {
                        "target_bytes_at_most": math.floor(scope_weight_payload / target),
                        "meets_even_at_scoped_lower_bound": total_lower_bound <= scope_weight_payload / target,
                    }
                    for target in args.compression_targets
                },
            }
            if args.whole_model_storage:
                whole_model_package_lower_bound = (
                    model["model_bytes"] - compressed_dense_payload + checkpoint_bytes
                )
                estimates[label]["whole_model_package_bytes_lower_bound"] = (
                    whole_model_package_lower_bound
                )
                estimates[label]["whole_model_compression_ratio_upper_bound"] = (
                    model["model_bytes"] / whole_model_package_lower_bound
                    if whole_model_package_lower_bound else None
                )
                estimates[label]["whole_model_ratio_targets"] = {
                    str(target): {
                        "target_bytes_at_most": math.floor(model["model_bytes"] / target),
                        "meets_even_at_whole_model_lower_bound": (
                            whole_model_package_lower_bound <= model["model_bytes"] / target
                        ),
                    }
                    for target in args.compression_targets
                }
        candidate_count_frontier.append({
            "compressed_candidate_count": compressed_count,
            "dense_layers_retained_in_scope": len(scope_layers) - compressed_count,
            "estimate_basis": (
                "checkpoint sizes use observed min/mean/max; dense savings use the "
                "largest per-layer projection/norm payloads first"
            ),
            "scenarios": estimates,
        })

    all_candidates_compressed_payload = candidate_dense_payload
    dense_payload_after_all = scope_weight_payload - all_candidates_compressed_payload
    ten_x_budget = scope_weight_payload / 10
    max_checkpoint_bytes_for_10x = ten_x_budget - dense_payload_after_all

    report = {
        "report_type": "CPU-only NEC 16-layer-scope storage feasibility preflight",
        "model": {
            "source_snapshot": str(args.model_dir),
            "source_serialized_model_bytes": model["model_bytes"],
            "source_weight_payload_bytes": model["weight_payload_bytes"],
            "source_metadata_bytes_excluding_shards_and_index": model[
                "metadata_bytes_excluding_weight_shards_and_index"
            ],
            "experiment_scope_layers": list(scope_layers),
            "experiment_scope_layer_count": len(scope_layers),
            "dense_scope_tensor_payload_bytes": scope_weight_payload,
            "scope_layer_tensor_payload_bytes": {
                str(layer): size for layer, size in scope_layer_payload.items()
            },
        },
        "allocation_scope": {
            "candidate_layers": list(candidate_layers),
            "candidate_count": len(candidate_layers),
            "candidate_compressible_dense_payload_bytes": candidate_dense_payload,
            "checkpoint_artifacts": candidate_artifacts,
            "storage_frontier_by_compressed_count": candidate_count_frontier,
            "10x_artifact_budget_if_all_candidates_compressed": {
                "remaining_dense_scope_payload_bytes": dense_payload_after_all,
                "max_checkpoint_bytes_before_omitted_overhead": max(0, max_checkpoint_bytes_for_10x),
                "max_average_checkpoint_bytes_per_compressed_layer": (
                    max(0, max_checkpoint_bytes_for_10x) / len(candidate_layers)
                ),
                "dense_scope_payload_alone_exceeds_10x_budget": max_checkpoint_bytes_for_10x < 0,
            },
        },
        "objective_requirements": {
            "primary_goal": "maximize measured serialized compression within the 16-layer experiment scope subject to quality constraints",
            "aspirational_ratio_within_scope": 10.0,
            "3x_is_insufficient": True,
            "max_dense_weight_fraction_is_a_ceiling": args.max_dense_weight_fraction,
            "max_dense_layer_count_in_scope": args.max_dense_layers,
            "compression_ratio_definition": "dense tensor payload bytes for the scoped layers divided by scoped dense tensor payload plus serialized NEC checkpoint bytes",
            "ratio_targets_assessed": args.compression_targets,
            "whole_model_storage_assessed": args.whole_model_storage,
            "whole_model_ratio_definition": (
                "serialized bytes of the original complete model divided by a lower bound consisting of original serialized bytes minus replaced tensor payloads plus actual or estimated NEC checkpoint bytes"
                if args.whole_model_storage else None
            ),
        },
        "limitations": [
            "Scoped package sizes and ratios are lower bounds / upper bounds: dense tensor serialization headers and other package overhead are omitted.",
            "Missing checkpoint sizes use the observed candidate artifacts as estimates.",
            "This report is scoped to the selected 16 layers and makes no storage claim about layers outside that scope.",
            "Quality feasibility is determined only by the measured PPL/KL evaluation, not this storage preflight.",
            "Any later deployable-model compression claim requires materializing and byte-counting its complete package separately.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
