#!/usr/bin/env python3
"""Export packed NEC weights with retained tensors and measure both budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.search_nec_layer_allocation import materialize_deployable_allocation_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--max-ppl", type=float, default=10.0)
    parser.add_argument("--checkpoint-container", choices=("plain", "gzip"), default="plain")
    args = parser.parse_args()

    config = json.loads((args.model_dir / "config.json").read_text())
    compile_manifest = json.loads((args.checkpoint_dir / "model_manifest.json").read_text())
    source_config_hash = hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest()
    if source_config_hash != compile_manifest["model_config_sha256"]:
        raise ValueError("source model configuration differs from the compiled model")
    layer_count = int(config["num_hidden_layers"])
    layers = tuple(sorted(
        int(path.stem.removeprefix("compressed_layer_"))
        for path in args.checkpoint_dir.glob("compressed_layer_*.pt")
    ))
    if not layers or any(layer < 0 or layer >= layer_count for layer in layers):
        raise ValueError("checkpoint layer IDs are missing or outside the model")
    expected_projection_names = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }
    for layer in layers:
        checkpoint = torch.load(
            args.checkpoint_dir / f"compressed_layer_{layer}.pt",
            map_location="cpu", weights_only=True,
        )
        if int(checkpoint["layer_idx"]) != layer:
            raise ValueError(f"layer index mismatch in checkpoint {layer}")
        if set(checkpoint["compressed_artifacts"]) != expected_projection_names:
            raise ValueError(f"checkpoint {layer} must replace all seven projections")
        if any(not (item.get("codec_type", "").startswith("nec_packed_operator_schmidt_")
                    or item.get("codec_type") in ("nec_packed_sector_schmidt_v1", "nec_sparse_excitation_schmidt_v1", "nec_sparse_excitation_schmidt_v2"))
               for item in checkpoint["compressed_artifacts"].values()):
            raise ValueError("this exporter requires packed Schmidt artifacts")
        del checkpoint

    quality = None
    if args.quality_report:
        quality = json.loads(args.quality_report.read_text())
        if tuple(quality["compressed_layers"]) != layers:
            raise ValueError("quality report layer IDs differ from exported checkpoints")
        if set(quality["compressed_projections"]) != expected_projection_names:
            raise ValueError("quality report must evaluate all exported projections")
        if Path(quality["checkpoint_dir"]).resolve() != args.checkpoint_dir.resolve():
            raise ValueError("quality report refers to a different checkpoint directory")

    measurement = materialize_deployable_allocation_package(
        args.model_dir, args.checkpoint_dir, args.output, tuple(range(layer_count)),
        layers, (), args.output / "unused_int8", config.get("_commit_hash", "local_source"),
        checkpoint_container=args.checkpoint_container,
    )
    expected_coefficients = compile_manifest.get("evaluated_assembly_coefficient_count")
    if (
        expected_coefficients is not None
        and measurement["complete_model_coefficient_count"] != expected_coefficients
    ):
        raise ValueError("exported coefficient count disagrees with compile accounting")
    package = Path(measurement["path"])
    if quality is not None:
        (package / "nec_quality_evaluation.json").write_text(
            json.dumps(quality, indent=2) + "\n"
        )
        quality_note = (
            f"\nMeasured PPL: {quality['student']['perplexity']:.6g}; "
            f"required PPL: <= {args.max_ppl:g}. "
            f"Quality gate: {'PASS' if quality['student']['perplexity'] <= args.max_ppl else 'FAIL'}.\n"
        )
        readme = package / "NEC_PACKAGE_README.md"
        readme.write_text(readme.read_text() + quality_note)
    sizes = {str(path.relative_to(package)): path.stat().st_size
             for path in package.rglob("*") if path.is_file()}
    measurement.update({
        "serialized_bytes": sum(sizes.values()),
        "file_sizes": sizes,
        "file_count": len(sizes),
        "compression_ratio": measurement["source_model_serialized_bytes"] / sum(sizes.values()),
        "whole_deployable_package_measured": True,
        "runtime_reconstructs_dense_weights": True,
    })
    preflight = compile_manifest["storage_preflight"]
    storage_pass = measurement["compression_ratio"] >= preflight["target_whole_model_ratio"]
    structure_pass = (
        measurement["structural_reduction"] >= preflight["minimum_structural_reduction"]
    )
    quality_pass = quality is not None and quality["student"]["perplexity"] <= args.max_ppl
    measurement["acceptance"] = {
        "minimum_storage_ratio": preflight["target_whole_model_ratio"],
        "minimum_structural_reduction": preflight["minimum_structural_reduction"],
        "maximum_ppl": args.max_ppl,
        "storage_passed": storage_pass,
        "structure_passed": structure_pass,
        "quality_passed": quality_pass,
        "quality_evaluation_available": quality is not None,
        "status": (
            "UNVERIFIED" if quality is None
            else "FEASIBLE" if storage_pass and structure_pass and quality_pass
            else "INFEASIBLE"
        ),
    }
    destination = args.output / "package_measurement.json"
    destination.write_text(json.dumps(measurement, indent=2) + "\n")
    print(json.dumps({key: value for key, value in measurement.items()
                      if key != "file_sizes"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
