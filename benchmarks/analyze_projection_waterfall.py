#!/usr/bin/env python3
"""Measure absolute NEC fidelity at every saved projection stage.

This analysis is read-only with respect to a completed qualification run.  It
regenerates the pinned calibration and validation activations, verifies their
identity, and evaluates each saved dense stage against the original projection.
It does not fit parameters or use validation data for selection.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import build_real_token_calibration_data  # noqa: E402
from benchmarks.qualify_streaming_projection import (  # noqa: E402
    DIAGNOSTIC_STAGES,
    LAYER,
    PROJECTION,
    materialize_down_inputs,
    relative_weight_difference,
    source_identity,
    tensor_sha256,
)
from nox.evaluation import validate_llama2_config  # noqa: E402
from scripts.compile_high_fidelity import (  # noqa: E402
    PinnedWeights,
    atomic_json_save,
    sha256_file,
)


def output_metrics(reference_weight, candidate_weight, inputs, token_chunk=256):
    """Return candidate error normalized by the reference output norm."""
    squared_error = 0.0
    reference_squared = 0.0
    candidate_squared = 0.0
    dot = 0.0
    for start in range(0, inputs.shape[1], token_chunk):
        chunk = inputs[:, start : start + token_chunk]
        reference = reference_weight @ chunk
        candidate = candidate_weight @ chunk
        squared_error += float((candidate - reference).double().square().sum().item())
        reference_squared += float(reference.double().square().sum().item())
        candidate_squared += float(candidate.double().square().sum().item())
        dot += float((candidate.double() * reference.double()).sum().item())
    reference_norm = max(math.sqrt(reference_squared), 1e-30)
    candidate_norm = math.sqrt(candidate_squared)
    cosine_denominator = max(reference_norm * candidate_norm, 1e-30)
    cosine = dot / cosine_denominator
    return {
        "sample_count": int(inputs.shape[1]),
        "relative_error": math.sqrt(squared_error) / reference_norm,
        "cosine_similarity": cosine,
        "angular_drift": 1.0 - cosine,
        "norm_ratio": candidate_norm / reference_norm,
    }


def load_stage(run_output, report, stage, device, expected_shape):
    metadata = report["stage_snapshots"][stage]
    path = run_output / "stage_weights" / f"{stage}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != metadata["bytes"] or sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"Stage checksum or size mismatch: {stage}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("stage") != stage
        or payload.get("algorithm_identity") != report.get("algorithm_identity")
        or payload.get("qualification_identity") != report.get("qualification_identity")
    ):
        raise ValueError(f"Stage provenance mismatch: {stage}")
    weight = payload.get("weight")
    if (
        not torch.is_tensor(weight)
        or tuple(weight.shape) != tuple(expected_shape)
        or not torch.isfinite(weight).all()
    ):
        raise ValueError(f"Invalid stage weight: {stage}")
    return weight.to(device=device, dtype=torch.float32)


def artifact_inventory(run_output, report):
    artifact_path = run_output / "artifact.pt"
    metadata = report.get("artifact", {})
    if not artifact_path.is_file() or sha256_file(artifact_path) != metadata.get("sha256"):
        raise ValueError("Artifact checksum mismatch")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    artifact = payload["artifact"]
    return {
        "artifact_file_bytes": int(artifact_path.stat().st_size),
        "artifact_sha256": metadata["sha256"],
        "artifact_keys": sorted(artifact),
        "core_shapes": [list(core.shape) for core in artifact["cores"]],
        "core_storage_dtypes": [str(core.dtype) for core in artifact["cores"]],
        "core_quantization_bits": artifact.get("core_quantization_bits"),
        "bridge_quantization_bits": artifact.get("bridge_quantization_bits"),
        "sparse_quantization_bits": artifact.get("sparse_quantization_bits"),
        "individual_pre_q4_cores_available": False,
        "inventory_conclusion": (
            "The artifact stores only final quantized/dequantized cores; saved "
            "pre-Q4 stages are dense operators. Per-core historical Q4 ablations "
            "cannot be reconstructed without new instrumentation."
        ),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--windows-per-chunk", type=int)
    args = parser.parse_args()

    report_path = args.run_output / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("status") != "PASS":
        raise ValueError("Waterfall requires a completed qualification report")
    identity = report["qualification_identity"]
    if report.get("layer") != LAYER or report.get("projection") != PROJECTION:
        raise ValueError("Only the pinned layer-0 down_proj qualification is supported")
    if set(report.get("stage_snapshots", {})) != set(DIAGNOSTIC_STAGES):
        raise ValueError("Qualification does not contain the complete stage set")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    windows_per_chunk = args.windows_per_chunk or identity["windows_per_chunk"]
    if windows_per_chunk <= 0:
        raise ValueError("windows-per-chunk must be positive")

    model_id = identity["model_id"]
    revision = identity["model_revision"]
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    validate_llama2_config(config)
    pinned = PinnedWeights(model_id, revision)
    embedding = pinned.tensor("model.embed_tokens.weight")
    train_hidden, validation_hidden, input_data = build_real_token_calibration_data(
        embedding,
        model_id,
        identity["calibration_tokens"],
        identity["validation_tokens"],
        identity["sequence_length"],
        revision=revision,
        calibration_sampling=identity["calibration_sampling"],
        validation_sampling=identity["validation_sampling"],
    )
    del embedding
    if input_data != report["input_data"]:
        raise ValueError("Regenerated token data does not match the qualification")
    hidden_hashes = {
        "calibration": tensor_sha256(train_hidden),
        "validation": tensor_sha256(validation_hidden),
    }
    if hidden_hashes != report["hidden_state_sha256"]:
        raise ValueError("Regenerated hidden states do not match the qualification")

    weights = pinned.layer(LAYER, device)
    del pinned
    dense_weight = weights[PROJECTION]
    forward_options = {
        "seq_len": identity["sequence_length"],
        "n_heads": config.num_attention_heads,
        "n_kv_heads": config.num_key_value_heads,
        "rms_norm_eps": config.rms_norm_eps,
    }
    calibration_inputs = materialize_down_inputs(
        weights,
        train_hidden,
        windows_per_chunk=windows_per_chunk,
        forward_options=forward_options,
    )
    validation_inputs = materialize_down_inputs(
        weights,
        validation_hidden,
        windows_per_chunk=windows_per_chunk,
        forward_options=forward_options,
    )

    stages = {}
    transitions = {}
    previous_name = None
    previous_weight = None
    for stage in DIAGNOSTIC_STAGES:
        candidate = load_stage(
            args.run_output, report, stage, device, dense_weight.shape,
        )
        stages[stage] = {
            "relative_weight_error": relative_weight_difference(candidate, dense_weight),
            "calibration": output_metrics(dense_weight, candidate, calibration_inputs),
            "validation": output_metrics(dense_weight, candidate, validation_inputs),
        }
        if previous_weight is not None:
            transitions[f"{previous_name}_to_{stage}"] = {
                "calibration": output_metrics(
                    previous_weight, candidate, calibration_inputs,
                ),
                "validation": output_metrics(
                    previous_weight, candidate, validation_inputs,
                ),
            }
            del previous_weight
        previous_name = stage
        previous_weight = candidate
    del previous_weight

    q4_transition = transitions["sweep_2_to_serialized_artifact"]
    result = {
        "schema_version": 1,
        "scope": "absolute-fidelity waterfall for saved NEC projection stages",
        "analysis_source": source_identity(),
        "qualification_source": report["source"],
        "model_id": model_id,
        "model_revision": revision,
        "layer": LAYER,
        "projection": PROJECTION,
        "weight_shape": list(dense_weight.shape),
        "input_data": input_data,
        "hidden_state_sha256": hidden_hashes,
        "metric_definitions": {
            "absolute_relative_error": "||(W_stage-W_dense)X||_F / ||W_dense X||_F",
            "transition_relative_error": "||(W_next-W_previous)X||_F / ||W_previous X||_F",
            "q4_relative_error": "||(W_Q4-W_float)X||_F / ||W_float X||_F",
        },
        "stages": stages,
        "transitions": transitions,
        "q4_boundary": {
            "from": "sweep_2",
            "to": "serialized_artifact",
            "calibration": q4_transition["calibration"],
            "validation": q4_transition["validation"],
        },
        "artifact_inventory": artifact_inventory(args.run_output, report),
        "validation_use": (
            "Validation activations are used only for measurement, never for "
            "fitting, gauge selection, or parameter selection."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_save(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
