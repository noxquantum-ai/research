#!/usr/bin/env python3
"""Compare matched full-SVD, FP32-statistics, and FP64-statistics NEC runs."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def load_run(path, expected_backend, expected_statistics_dtype):
    path = Path(path)
    manifest = json.loads((path / "model_manifest.json").read_text())
    checkpoint = torch.load(
        path / "compressed_layer_0.pt", map_location="cpu", weights_only=False,
    )
    if checkpoint.get("layer_idx") != 0:
        raise ValueError(f"{path} does not contain a layer-0 checkpoint")
    identity = manifest.get("run_identity", {})
    if identity.get("calibration_backend") != expected_backend:
        raise ValueError(f"{path} backend does not match {expected_backend}")
    if identity.get("statistics_dtype") != expected_statistics_dtype:
        raise ValueError(f"{path} statistics dtype does not match {expected_statistics_dtype}")
    input_data = identity.get("input_data", {})
    if (
        input_data.get("calibration_token_count") != 4096
        or input_data.get("heldout_token_count") != 1024
        or input_data.get("activation_sequence_length") != 128
    ):
        raise ValueError(f"{path} does not use the pinned qualification samples")
    artifacts = checkpoint["compressed_artifacts"]
    if set(artifacts) != set(PROJECTIONS):
        raise ValueError(f"{path} does not contain exactly seven projections")
    return manifest, checkpoint


def artifact_coefficient_count(artifact):
    total = sum(int(core.numel()) for core in artifact["cores"])
    total += int(artifact["d_awt"].numel())
    for name in ("A_bridge", "B_bridge", "sparse_values"):
        tensor = artifact.get(name)
        if tensor is not None:
            total += int(tensor.numel())
    return total


def relative_error(actual, expected):
    return float(
        (torch.linalg.vector_norm(actual.float() - expected.float())
         / torch.linalg.vector_norm(expected.float()).clamp_min(1e-30)).item()
    )


def compare_variant(reference_checkpoint, variant_manifest, variant_checkpoint, name):
    projections = {}
    maximum = 0.0
    structure_matches = True
    q4_complete = True
    expected_components = {
        "sparse_shield", "bridge_A", "bridge_B",
        "mpo_site_0", "mpo_site_1", "mpo_site_2", "mpo_site_3",
    }
    for projection in PROJECTIONS:
        reference_artifact = reference_checkpoint["compressed_artifacts"][projection]
        variant_artifact = variant_checkpoint["compressed_artifacts"][projection]
        reference_weight = reconstruct_from_artifact(reference_artifact, dtype=torch.float32)
        variant_weight = reconstruct_from_artifact(variant_artifact, dtype=torch.float32)
        difference = relative_error(variant_weight, reference_weight)
        maximum = max(maximum, difference)
        same_structure = (
            reference_artifact["ranks"] == variant_artifact["ranks"]
            and artifact_coefficient_count(reference_artifact)
            == artifact_coefficient_count(variant_artifact)
        )
        structure_matches = structure_matches and same_structure
        projection_q4_complete = (
            variant_artifact.get("core_quantization_bits") == 4
            and variant_artifact.get("sparse_bridge_quantization_bits") == 4
            and set(variant_artifact.get("precision_diagnostics", {})) == expected_components
            and variant_artifact.get("compiler_hyperparameters", {}).get("n_sweeps") == 2
        )
        q4_complete = q4_complete and projection_q4_complete
        diagnostics = variant_artifact.get("calibration_statistics_diagnostics", {})
        projections[projection] = {
            "relative_reconstructed_weight_difference": difference,
            "reference_ranks": reference_artifact["ranks"],
            "variant_ranks": variant_artifact["ranks"],
            "reference_coefficients": artifact_coefficient_count(reference_artifact),
            "variant_coefficients": artifact_coefficient_count(variant_artifact),
            "structure_matches": same_structure,
            "two_sweep_q4_diagnostics_complete": projection_q4_complete,
            "calibration_statistics_diagnostics": diagnostics,
            "precision_diagnostics": variant_artifact.get("precision_diagnostics"),
        }
        del reference_weight, variant_weight
    history = variant_manifest.get("history", [])
    if len(history) != 1 or history[0].get("layer") != 0:
        raise ValueError(f"{name} manifest must contain exactly the layer-0 record")
    return {
        "maximum_relative_reconstructed_weight_difference": maximum,
        "all_projection_structures_match": structure_matches,
        "all_two_sweep_q4_diagnostics_complete": q4_complete,
        "layer_record": history[0],
        "projections": projections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--fp64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-weight-difference", type=float, default=0.01)
    args = parser.parse_args()
    reference_manifest, reference_checkpoint = load_run(args.reference, "full_matrix", "float64")
    fp32_manifest, fp32_checkpoint = load_run(args.fp32, "streaming_statistics", "float32")
    fp64_manifest, fp64_checkpoint = load_run(args.fp64, "streaming_statistics", "float64")
    reference_inputs = reference_manifest["run_identity"]["input_data"]
    for name, manifest in (("fp32", fp32_manifest), ("fp64", fp64_manifest)):
        if manifest["run_identity"]["input_data"] != reference_inputs:
            raise ValueError(f"{name} qualification run did not use identical sampled inputs")
    report = {
        "scope": "real layer-0, seven-projection streaming qualification",
        "reference_backend": reference_manifest["run_identity"]["calibration_backend"],
        "identical_input_data_metadata": True,
        "reference_layer_record": reference_manifest["history"][0],
        "fp32": compare_variant(reference_checkpoint, fp32_manifest, fp32_checkpoint, "fp32"),
        "fp64": compare_variant(reference_checkpoint, fp64_manifest, fp64_checkpoint, "fp64"),
        "acceptance": {
            "maximum_relative_weight_difference": args.max_relative_weight_difference,
        },
    }
    for name in ("fp32", "fp64"):
        maximum = report[name]["maximum_relative_reconstructed_weight_difference"]
        report[name]["passes_weight_parity_threshold"] = bool(
            math.isfinite(maximum) and maximum <= args.max_relative_weight_difference
        )
        report[name]["passes_complete_production_structure"] = bool(
            report[name]["all_projection_structures_match"]
            and report[name]["all_two_sweep_q4_diagnostics_complete"]
        )
    report["all_variants_pass"] = bool(
        report["fp32"]["passes_weight_parity_threshold"]
        and report["fp64"]["passes_weight_parity_threshold"]
        and report["fp32"]["passes_complete_production_structure"]
        and report["fp64"]["passes_complete_production_structure"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_variants_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
