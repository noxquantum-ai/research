#!/usr/bin/env python3
"""CPU-only storage feasibility preflight for NEC layer allocations.

This computes optimistic whole-model lower bounds from the serialized dense
model and available NEC checkpoint files. It does not build a deployable mixed
model bundle, and its estimates must not be reported as measured compression.
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


def projection_lower_bound(
    *, baseline_bytes: int, metadata_bytes: int, weight_payload_bytes: int,
    dense_payload_bytes: int, checkpoint_bytes: int,
) -> dict:
    # Output safetensors headers/index, shard padding, runtime/loader files and
    # packaging overhead are intentionally excluded, making this a lower bound.
    lower_bound_bytes = metadata_bytes + dense_payload_bytes + checkpoint_bytes
    return {
        "projected_bundle_lower_bound_bytes": lower_bound_bytes,
        "whole_model_ratio_upper_bound": baseline_bytes / lower_bound_bytes,
        "dense_weight_payload_fraction": dense_payload_bytes / weight_payload_bytes,
    }


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
    parser.add_argument("--max-dense-weight-fraction", type=float, default=0.25)
    parser.add_argument(
        "--compression-targets", type=float, nargs="+", default=(3.0, 10.0),
        help="Whole-model ratio targets to assess; default: 3x 10x",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not 0 <= args.max_dense_weight_fraction <= 1:
        parser.error("--max-dense-weight-fraction must be between 0 and 1")
    if any(target <= 1 for target in args.compression_targets):
        parser.error("Compression-ratio targets must be greater than 1")
    candidate_layers = parse_layers(args.candidate_layers)
    model = inspect_dense_model(args.model_dir)
    if model["total_layers"] is None:
        raise ValueError("Could not infer transformer layer count from model config")
    outside_range = [layer for layer in candidate_layers if layer >= model["total_layers"]]
    if outside_range:
        raise ValueError(f"Candidate IDs outside model range: {outside_range}")

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
    candidate_dense_payload = sum(per_layer_payload.values())
    dense_after_candidate_compression = model["weight_payload_bytes"] - candidate_dense_payload
    candidate_artifacts = checkpoint_scenarios(args.checkpoint_dir, candidate_layers)

    scenarios = {}
    for scenario, checkpoint_bytes in candidate_artifacts[
        "all_candidates_sum_actual_and_missing_estimated"
    ].items():
        scenarios[scenario] = projection_lower_bound(
            baseline_bytes=model["model_bytes"],
            metadata_bytes=model["metadata_bytes_excluding_weight_shards_and_index"],
            weight_payload_bytes=model["weight_payload_bytes"],
            dense_payload_bytes=dense_after_candidate_compression,
            checkpoint_bytes=int(checkpoint_bytes),
        )
        scenarios[scenario]["meets_max_dense_weight_fraction"] = (
            scenarios[scenario]["dense_weight_payload_fraction"]
            <= args.max_dense_weight_fraction
        )
        scenarios[scenario]["ratio_targets"] = {
            str(target): {
                "target_bytes_at_most": math.floor(model["model_bytes"] / target),
                "meets_even_at_projected_lower_bound": (
                    scenarios[scenario]["projected_bundle_lower_bound_bytes"]
                    <= model["model_bytes"] / target
                ),
            }
            for target in args.compression_targets
        }

    all_layer_payload = sum(
        layer_compressible_payload(model["tensor_payload_bytes"], layer)
        for layer in range(model["total_layers"])
    )
    fixed_dense_payload = model["weight_payload_bytes"] - all_layer_payload
    observed_mean = candidate_artifacts["observed_mean_bytes"]
    observed_min = candidate_artifacts["observed_min_bytes"]
    all_layer_projection = {
        "scope": "hypothetical all-transformer-layer extrapolation; not measured artifacts",
        "all_layers_compressible_weight_payload_bytes": all_layer_payload,
        "fixed_dense_weight_payload_bytes": fixed_dense_payload,
        "dense_weight_payload_fraction": fixed_dense_payload / model["weight_payload_bytes"],
        "artifact_size_basis": {
            "mean_observed_checkpoint_bytes": observed_mean,
            "minimum_observed_checkpoint_bytes": observed_min,
            "observed_checkpoint_count": len(candidate_artifacts["observed_by_layer"]),
            "candidate_count": len(candidate_layers),
        },
        "projected_bundle_lower_bound": {},
    }
    for scenario, estimate in (("mean", observed_mean), ("minimum", observed_min)):
        projected = projection_lower_bound(
            baseline_bytes=model["model_bytes"],
            metadata_bytes=model["metadata_bytes_excluding_weight_shards_and_index"],
            weight_payload_bytes=model["weight_payload_bytes"],
            dense_payload_bytes=fixed_dense_payload,
            checkpoint_bytes=round(estimate * model["total_layers"]),
        )
        projected["ratio_targets"] = {
            str(target): projected["projected_bundle_lower_bound_bytes"]
            <= model["model_bytes"] / target
            for target in args.compression_targets
        }
        all_layer_projection["projected_bundle_lower_bound"][scenario] = projected

    ten_x = model["model_bytes"] / 10
    fixed_floor = (
        model["metadata_bytes_excluding_weight_shards_and_index"] + fixed_dense_payload
    )
    max_artifact_bytes_for_10x = ten_x - fixed_floor
    all_layer_projection["10x_artifact_budget_if_every_layer_compressed"] = {
        "total_artifact_bytes_at_most_before_any_omitted_overhead": max(0, max_artifact_bytes_for_10x),
        "average_artifact_bytes_per_layer_at_most_before_omitted_overhead": (
            max_artifact_bytes_for_10x / model["total_layers"]
        ),
        "fixed_bytes_already_exceed_10x_budget": max_artifact_bytes_for_10x < 0,
    }

    report = {
        "report_type": "CPU-only NEC storage feasibility preflight",
        "model": {
            "directory": str(args.model_dir),
            "serialized_dense_snapshot_bytes": model["model_bytes"],
            "file_count": model["file_count"],
            "transformer_layers": model["total_layers"],
            "safetensors_weight_payload_bytes": model["weight_payload_bytes"],
            "weight_shards": model["weight_shards"],
            "weight_shard_bytes": model["weight_shard_bytes"],
            "required_nonweight_metadata_bytes_excluding_index": model[
                "metadata_bytes_excluding_weight_shards_and_index"
            ],
            "source_index_bytes": model["source_index_bytes"],
        },
        "allocation_scope": {
            "candidate_layers": list(candidate_layers),
            "candidate_count": len(candidate_layers),
            "candidate_compressible_dense_payload_bytes": candidate_dense_payload,
            "dense_weight_payload_bytes_after_compressing_all_candidates": dense_after_candidate_compression,
            "dense_weight_payload_fraction_after_compressing_all_candidates": (
                dense_after_candidate_compression / model["weight_payload_bytes"]
            ),
            "checkpoint_artifacts": candidate_artifacts,
            "all_candidates_compressed_scenarios": scenarios,
            "research_test_allowed_even_if_deployment_targets_fail": True,
        },
        "whole_model_future_extrapolation": all_layer_projection,
        "objective_requirements": {
            "primary_goal": "maximize measured complete-model serialized compression ratio subject to quality constraints",
            "aspirational_whole_model_ratio": 10.0,
            "3x_is_insufficient": True,
            "max_dense_weight_fraction_is_a_ceiling": args.max_dense_weight_fraction,
            "compression_ratio_definition": "dense snapshot directory bytes divided by complete deployable mixed-model package bytes",
            "ratio_targets_assessed": args.compression_targets,
        },
        "limitations": [
            "Projected sizes are lower bounds, not measured deployable bundle sizes.",
            "Missing checkpoint sizes use the observed candidate artifacts as estimates.",
            "Output safetensors headers/index, shard padding, runtime/loader files, and packaging overhead are omitted.",
            "A 16-layer quality/allocation experiment remains valid research even when its candidate scope cannot meet a whole-model deployment ratio target.",
            "Final compression claims require materializing the complete deployable package and summing every serialized file byte.",
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
