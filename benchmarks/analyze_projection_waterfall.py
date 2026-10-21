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
    CORE_DIAGNOSTIC_STAGE,
    DIAGNOSTIC_STAGES,
    LAYER,
    PROJECTION,
    materialize_down_inputs,
    relative_weight_difference,
    source_identity,
    tensor_sha256,
)
from nox.compiler import quantize_4bit_grouped  # noqa: E402
from nox.evaluation import validate_llama2_config  # noqa: E402
from nox.utils import mpo_to_dense  # noqa: E402
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


def load_core_stage(run_output, report, stage, device):
    metadata = report.get("core_stage_snapshots", {}).get(stage)
    if metadata is None:
        return None
    path = run_output / "core_stages" / f"{stage}.pt"
    if (
        not path.is_file()
        or path.stat().st_size != metadata["bytes"]
        or sha256_file(path) != metadata["sha256"]
    ):
        raise ValueError(f"Core-stage checksum or size mismatch: {stage}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("stage") != stage
        or payload.get("algorithm_identity") != report.get("algorithm_identity")
        or payload.get("qualification_identity") != report.get("qualification_identity")
    ):
        raise ValueError(f"Core-stage provenance mismatch: {stage}")
    cores = payload.get("cores")
    d_awt = payload.get("d_awt")
    if (
        not isinstance(cores, list)
        or len(cores) != 4
        or not all(torch.is_tensor(core) and torch.isfinite(core).all() for core in cores)
        or not torch.is_tensor(d_awt)
        or not torch.isfinite(d_awt).all()
    ):
        raise ValueError(f"Invalid core-stage snapshot: {stage}")
    return (
        [core.to(device=device, dtype=torch.float32) for core in cores],
        d_awt.to(device=device, dtype=torch.float32),
    )


def core_quantization_waterfall(
    dense_weight,
    pre_q4_weight,
    final_weight,
    cores,
    d_awt,
    calibration_inputs,
    validation_inputs,
):
    """Attribute Q4 damage while holding the saved non-MPO operator fixed."""
    pre_mpo = mpo_to_dense(cores) / d_awt.unsqueeze(0)

    def measure(selected):
        quantized = [
            quantize_4bit_grouped(core, group_size=64)
            if index in selected else core
            for index, core in enumerate(cores)
        ]
        quantized_mpo = mpo_to_dense(quantized) / d_awt.unsqueeze(0)
        # The dense pre-Q4 stage contains the sparse shield and bridge at their
        # exact compilation precision.  Adding only the MPO delta prevents
        # FP16 serialization of those components from confounding attribution.
        candidate = pre_q4_weight + (quantized_mpo - pre_mpo)
        return {
            "absolute_relative_weight_error": relative_weight_difference(
                candidate, dense_weight,
            ),
            "absolute_calibration": output_metrics(
                dense_weight, candidate, calibration_inputs,
            ),
            "absolute_validation": output_metrics(
                dense_weight, candidate, validation_inputs,
            ),
            "q4_transition_calibration": output_metrics(
                pre_q4_weight, candidate, calibration_inputs,
            ),
            "q4_transition_validation": output_metrics(
                pre_q4_weight, candidate, validation_inputs,
            ),
        }, candidate

    individual = {}
    for index in range(4):
        individual[f"core_{index}"] = measure({index})[0]
    cumulative = {}
    all_q4_candidate = None
    for stop in range(1, 5):
        metrics, candidate = measure(set(range(stop)))
        cumulative[f"cores_0_through_{stop - 1}"] = metrics
        if stop == 4:
            all_q4_candidate = candidate
        else:
            del candidate
    return {
        "definition": (
            "Quantize selected saved FP32 pre-Q4 MPO cores and add only the "
            "resulting MPO delta to the exact saved sweep-2 dense operator."
        ),
        "individual_cores": individual,
        "cumulative_cores": cumulative,
        "all_q4_to_serialized_artifact": {
            "relative_weight_difference": relative_weight_difference(
                all_q4_candidate, final_weight,
            ),
            "calibration": output_metrics(
                all_q4_candidate, final_weight, calibration_inputs,
            ),
            "validation": output_metrics(
                all_q4_candidate, final_weight, validation_inputs,
            ),
        },
    }


def artifact_inventory(run_output, report):
    artifact_path = run_output / "artifact.pt"
    metadata = report.get("artifact", {})
    if not artifact_path.is_file() or sha256_file(artifact_path) != metadata.get("sha256"):
        raise ValueError("Artifact checksum mismatch")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    artifact = payload["artifact"]
    core_metadata = report.get("core_stage_snapshots", {})
    core_available = CORE_DIAGNOSTIC_STAGE in core_metadata
    return {
        "artifact_file_bytes": int(artifact_path.stat().st_size),
        "artifact_sha256": metadata["sha256"],
        "artifact_keys": sorted(artifact),
        "core_shapes": [list(core.shape) for core in artifact["cores"]],
        "core_storage_dtypes": [str(core.dtype) for core in artifact["cores"]],
        "core_quantization_bits": artifact.get("core_quantization_bits"),
        "bridge_quantization_bits": artifact.get("bridge_quantization_bits"),
        "sparse_quantization_bits": artifact.get("sparse_quantization_bits"),
        "individual_pre_q4_cores_available": core_available,
        "pre_q4_core_snapshot": core_metadata.get(CORE_DIAGNOSTIC_STAGE),
        "inventory_conclusion": (
            "A verified FP32 pre-Q4 core snapshot is available for per-core Q4 attribution."
            if core_available else
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
    core_q4 = None
    core_snapshot = load_core_stage(
        args.run_output, report, CORE_DIAGNOSTIC_STAGE, device,
    )
    if core_snapshot is not None:
        pre_q4_cores, pre_q4_d_awt = core_snapshot
        pre_q4_weight = load_stage(
            args.run_output, report, "sweep_2", device, dense_weight.shape,
        )
        final_weight = load_stage(
            args.run_output, report, "serialized_artifact", device, dense_weight.shape,
        )
        core_q4 = core_quantization_waterfall(
            dense_weight,
            pre_q4_weight,
            final_weight,
            pre_q4_cores,
            pre_q4_d_awt,
            calibration_inputs,
            validation_inputs,
        )
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
        "core_q4_attribution": core_q4,
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
