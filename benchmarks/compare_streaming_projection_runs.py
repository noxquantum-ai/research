#!/usr/bin/env python3
"""Compare staged real-projection NEC qualification artifacts."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact  # noqa: E402
from scripts.compile_high_fidelity import sha256_file  # noqa: E402
from benchmarks.qualify_streaming_projection import (  # noqa: E402
    DIAGNOSTIC_STAGES,
    algorithm_identity,
    source_identity,
    validate_completed,
)


EXPECTED_COMPONENTS = {
    "sparse_shield", "bridge_A", "bridge_B",
    "mpo_site_0", "mpo_site_1", "mpo_site_2", "mpo_site_3",
}


def atomic_json_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact_coefficient_count(artifact):
    total = sum(int(core.numel()) for core in artifact["cores"])
    total += int(artifact["d_awt"].numel())
    for name in ("A_bridge", "B_bridge", "sparse_values"):
        value = artifact.get(name)
        if value is not None:
            total += int(value.numel())
    return total


def load_run(path, backend, statistics_dtype, current_algorithm):
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    artifact_path = path / "artifact.pt"
    if report.get("status") != "PASS":
        raise ValueError(f"{path} is not a completed qualification stage")
    if report.get("backend") != backend or report.get("statistics_dtype") != statistics_dtype:
        raise ValueError(f"{path} backend or statistics dtype is not the requested variant")
    if report.get("layer") != 0 or report.get("projection") != "down_proj":
        raise ValueError(f"{path} is not the pinned layer-0 down_proj stage")
    if report.get("algorithm_identity") != current_algorithm:
        raise ValueError(f"{path} was produced by different qualification code")
    if report.get("artifact", {}).get("sha256") != sha256_file(artifact_path):
        raise ValueError(f"{path} artifact checksum does not match its report")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if payload.get("input_data") != report.get("input_data"):
        raise ValueError(f"{path} artifact and report input metadata differ")
    identity = report.get("qualification_identity")
    required_identity = {
        "schema_version": 2,
        "model_id": "NousResearch/Llama-2-7b-hf",
        "model_revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
        "layer": 0,
        "projection": "down_proj",
        "backend": backend,
        "statistics_dtype": statistics_dtype,
        "calibration_tokens": 4096,
        "validation_tokens": 1024,
        "sequence_length": 128,
        "calibration_sampling": "uniform",
        "validation_sampling": "uniform",
        "calibration_seed": 42,
        "validation_seed": 43,
        "ranks": [1, 64, 128, 64, 1],
        "bridge_rank": 128,
        "sparse_budget": 524288,
        "sweeps": 2,
        "prior_gamma": 0.01,
        "activation_scale_exponent": 0.5,
        "dmrg_chunk_size": 32,
        "diagonal_prior_strength": 0.0,
        "site_order": [0, 1, 2, 3],
        "quantize_components": True,
        "stage_diagnostics": list(DIAGNOSTIC_STAGES),
    }
    if not isinstance(identity, dict) or any(
        identity.get(key) != value for key, value in required_identity.items()
    ):
        raise ValueError(f"{path} qualification settings do not match the current protocol")
    if (
        payload.get("algorithm_identity") != current_algorithm
        or payload.get("qualification_identity") != identity
        or not validate_completed(path, identity, current_algorithm)
    ):
        raise ValueError(f"{path} artifact identity does not match its report")
    return report, payload["artifact"]


def relative_difference(actual, reference):
    return float(
        torch.linalg.vector_norm(actual.float() - reference.float())
        .div(torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
        .item()
    )


def compare(reference_report, reference_artifact, variant_report, variant_artifact, hourly_usd):
    if variant_report["input_data"] != reference_report["input_data"]:
        raise ValueError("Qualification variants did not use identical input data")
    if variant_report["hidden_state_sha256"] != reference_report["hidden_state_sha256"]:
        raise ValueError("Qualification variants did not reproduce identical hidden states")
    if variant_report.get("algorithm_identity") != reference_report.get("algorithm_identity"):
        raise ValueError("Qualification variants used different numerical source fingerprints")
    reference_weight = reconstruct_from_artifact(reference_artifact, dtype=torch.float32)
    variant_weight = reconstruct_from_artifact(variant_artifact, dtype=torch.float32)
    difference = relative_difference(variant_weight, reference_weight)
    same_structure = (
        reference_artifact.get("ranks") == variant_artifact.get("ranks")
        and artifact_coefficient_count(reference_artifact) == artifact_coefficient_count(variant_artifact)
    )
    q4_complete = (
        variant_artifact.get("core_quantization_bits") == 4
        and variant_artifact.get("sparse_bridge_quantization_bits") == 4
        and set(variant_artifact.get("precision_diagnostics", {})) == EXPECTED_COMPONENTS
        and variant_artifact.get("compiler_hyperparameters", {}).get("n_sweeps") == 2
    )
    combined_runtime = reference_report["runtime_seconds"] + variant_report["runtime_seconds"]
    metric_differences = {}
    for split in ("calibration", "validation"):
        reference_metrics = reference_report[f"{split}_metrics"]
        variant_metrics = variant_report[f"{split}_metrics"]
        metric_differences[split] = {
            "relative_error_absolute_difference": abs(
                variant_metrics["relative_error"] - reference_metrics["relative_error"]
            ),
            "cosine_similarity_absolute_difference": abs(
                variant_metrics["cosine_similarity"] - reference_metrics["cosine_similarity"]
            ),
        }
    direct_disagreement = variant_report.get("direct_functional_disagreement")
    first_stage_over_one_percent = None
    if direct_disagreement:
        for stage in DIAGNOSTIC_STAGES:
            stage_result = direct_disagreement.get("stages", {}).get(stage, {})
            if (
                stage_result.get("relative_reconstructed_weight_difference", 0.0) > 0.01
                or stage_result.get("validation", {}).get(
                    "relative_functional_disagreement", 0.0,
                ) > 0.01
            ):
                first_stage_over_one_percent = stage
                break
    return {
        "relative_reconstructed_weight_difference": difference,
        "structure_matches": same_structure,
        "two_sweep_q4_diagnostics_complete": q4_complete,
        "reference_coefficients": artifact_coefficient_count(reference_artifact),
        "variant_coefficients": artifact_coefficient_count(variant_artifact),
        "reference_calibration_metrics": reference_report["calibration_metrics"],
        "variant_calibration_metrics": variant_report["calibration_metrics"],
        "reference_validation_metrics": reference_report["validation_metrics"],
        "variant_validation_metrics": variant_report["validation_metrics"],
        "reference_runtime_seconds": reference_report["runtime_seconds"],
        "variant_runtime_seconds": variant_report["runtime_seconds"],
        "combined_runtime_seconds": combined_runtime,
        "combined_observed_gpu_cost_usd": combined_runtime / 3600.0 * hourly_usd,
        "planning_upper_bound_for_two_full_layer_variants": {
            "method": "seven times the observed two-variant down_proj runtime; conservative planning bound, not a runtime model",
            "seconds": 7.0 * combined_runtime,
            "gpu_cost_usd": 7.0 * combined_runtime / 3600.0 * hourly_usd,
        },
        "metric_differences": metric_differences,
        "reference_memory": reference_report["memory"],
        "variant_memory": variant_report["memory"],
        "calibration_statistics_diagnostics": variant_report.get("calibration_statistics_diagnostics"),
        "bridge_statistics_diagnostics": variant_report.get("bridge_statistics_diagnostics"),
        "direct_functional_disagreement": direct_disagreement,
        "first_stage_over_one_percent_disagreement": first_stage_over_one_percent,
        "same_algorithm_fingerprint": True,
        "source_commits": {
            "reference": reference_report.get("source", {}).get("commit"),
            "variant": variant_report.get("source", {}).get("commit"),
        },
        "clean_source_trees": bool(
            reference_report.get("source", {}).get("clean")
            and variant_report.get("source", {}).get("clean")
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--fp64", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-weight-difference", type=float, default=0.01)
    parser.add_argument("--max-metric-absolute-difference", type=float, default=0.01)
    parser.add_argument("--max-functional-disagreement", type=float, default=0.01)
    parser.add_argument("--gpu-hourly-usd", type=float, default=0.59)
    args = parser.parse_args()
    if args.gpu_hourly_usd < 0:
        raise ValueError("gpu-hourly-usd must be nonnegative")

    current_source = source_identity()
    if not current_source["clean"]:
        raise RuntimeError("Comparison requires a clean source tree")
    current_algorithm = algorithm_identity()
    reference_report, reference_artifact = load_run(
        args.reference, "full_matrix", "float64", current_algorithm,
    )
    fp32_report, fp32_artifact = load_run(
        args.fp32, "streaming_statistics", "float32", current_algorithm,
    )
    fp32 = compare(
        reference_report, reference_artifact, fp32_report, fp32_artifact,
        args.gpu_hourly_usd,
    )
    fp32["passes_weight_parity_threshold"] = bool(
        math.isfinite(fp32["relative_reconstructed_weight_difference"])
        and fp32["relative_reconstructed_weight_difference"] <= args.max_relative_weight_difference
    )
    fp32["passes_complete_production_structure"] = bool(
        fp32["structure_matches"] and fp32["two_sweep_q4_diagnostics_complete"]
    )
    fp32["passes_activation_metric_parity"] = all(
        item["relative_error_absolute_difference"] <= args.max_metric_absolute_difference
        and item["cosine_similarity_absolute_difference"] <= args.max_metric_absolute_difference
        for item in fp32["metric_differences"].values()
    )
    fp32["passes_direct_functional_parity"] = bool(
        fp32.get("direct_functional_disagreement")
        and all(
            fp32["direct_functional_disagreement"]["final_artifact"][split][
                "relative_functional_disagreement"
            ] <= args.max_functional_disagreement
            for split in ("calibration", "validation")
        )
    )
    fp32["passes_algorithm_identity"] = bool(
        fp32["same_algorithm_fingerprint"] and fp32["clean_source_trees"]
    )
    report = {
        "scope": "real Llama-2-7B layer-0 down_proj staged streaming qualification",
        "source": current_source,
        "algorithm_identity": current_algorithm,
        "identical_input_data_and_hidden_states": True,
        "gpu_hourly_usd": args.gpu_hourly_usd,
        "acceptance": {
            "maximum_relative_weight_difference": args.max_relative_weight_difference,
            "maximum_activation_metric_absolute_difference": args.max_metric_absolute_difference,
            "maximum_direct_functional_disagreement": args.max_functional_disagreement,
        },
        "fp32": fp32,
    }
    if args.fp64 is not None:
        fp64_report, fp64_artifact = load_run(
            args.fp64, "streaming_statistics", "float64", current_algorithm,
        )
        fp64 = compare(
            reference_report, reference_artifact, fp64_report, fp64_artifact,
            args.gpu_hourly_usd,
        )
        fp64["passes_weight_parity_threshold"] = bool(
            math.isfinite(fp64["relative_reconstructed_weight_difference"])
            and fp64["relative_reconstructed_weight_difference"] <= args.max_relative_weight_difference
        )
        fp64["passes_complete_production_structure"] = bool(
            fp64["structure_matches"] and fp64["two_sweep_q4_diagnostics_complete"]
        )
        fp64["passes_activation_metric_parity"] = all(
            item["relative_error_absolute_difference"] <= args.max_metric_absolute_difference
            and item["cosine_similarity_absolute_difference"] <= args.max_metric_absolute_difference
            for item in fp64["metric_differences"].values()
        )
        fp64["passes_direct_functional_parity"] = bool(
            fp64.get("direct_functional_disagreement")
            and all(
                fp64["direct_functional_disagreement"]["final_artifact"][split][
                    "relative_functional_disagreement"
                ] <= args.max_functional_disagreement
                for split in ("calibration", "validation")
            )
        )
        fp64["passes_algorithm_identity"] = bool(
            fp64["same_algorithm_fingerprint"] and fp64["clean_source_trees"]
        )
        report["fp64"] = fp64

    report["fp32_passes"] = bool(
        fp32["passes_weight_parity_threshold"]
        and fp32["passes_complete_production_structure"]
        and fp32["passes_activation_metric_parity"]
        and fp32["passes_algorithm_identity"]
    )
    if "fp64" in report:
        report["fp64_passes"] = bool(
            report["fp64"]["passes_weight_parity_threshold"]
            and report["fp64"]["passes_complete_production_structure"]
            and report["fp64"]["passes_activation_metric_parity"]
            and report["fp64"]["passes_algorithm_identity"]
        )
    report["recommended_next_stage"] = (
        "one_layer_fp32" if report["fp32_passes"] else (
            "one_layer_fp64" if report.get("fp64_passes") else
            "implement_numerically_stable_factorization" if "fp64" in report else
            "down_projection_fp64"
        )
    )
    report["scientific_scope_warning"] = (
        "Projection parity and memory savings do not establish full-model quality; "
        "PPL and MMLU still require a complete 32-layer evaluation."
    )
    atomic_json_save(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["fp32_passes"] and args.fp64 is None:
        # A failed FP32 comparison is a valid diagnostic result. The nonzero
        # status prevents an unattended workflow from progressing to a layer.
        raise SystemExit(3)
    if args.fp64 is not None and not report["fp64_passes"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
