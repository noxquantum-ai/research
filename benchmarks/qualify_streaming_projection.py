#!/usr/bin/env python3
"""Qualify one real Llama-2 projection against NEC's streaming backend.

This intentionally produces a qualification artifact, not a model checkpoint.
The first paid experiment is restricted to layer-0 ``down_proj`` with real
WikiText-2 activations and the production High-Fidelity NEC hyperparameters.
"""

import argparse
import gc
import hashlib
import json
import math
import resource
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import build_real_token_calibration_data  # noqa: E402
from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d  # noqa: E402
from nox.calibration import InputCovarianceAccumulator  # noqa: E402
from nox.artifacts import fingerprint  # noqa: E402
from nox.compiler import (  # noqa: E402
    nox_hybrid_dmrg_decompose,
    nox_hybrid_dmrg_decompose_from_statistics,
    reconstruct_from_artifact,
)
from nox.evaluation import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    validate_llama2_config,
)
from nox.sequential_compiler import get_tier_config  # noqa: E402
from nox.utils import get_modes_for_projection  # noqa: E402
from scripts.compile_high_fidelity import PinnedWeights, atomic_json_save, atomic_torch_save, sha256_file  # noqa: E402


PROJECTION = "down_proj"
LAYER = 0
EXPECTED_Q4_COMPONENTS = {
    "sparse_shield", "bridge_A", "bridge_B",
    "mpo_site_0", "mpo_site_1", "mpo_site_2", "mpo_site_3",
}
DIAGNOSTIC_STAGES = (
    "sparse",
    "sparse_plus_bridge",
    "tt_warmstart",
    "sweep_1",
    "sweep_2",
    "serialized_artifact",
)
QUALIFICATION_SOURCE_PATHS = (
    "benchmarks/benchmark_day3_cascade.py",
    "benchmarks/qualify_streaming_projection.py",
    "nox/artifacts.py",
    "nox/block_compiler.py",
    "nox/calibration.py",
    "nox/compiler.py",
    "nox/data.py",
    "nox/evaluation.py",
    "nox/sequential_compiler.py",
    "nox/tensor_order.py",
    "nox/utils.py",
    "scripts/compile_high_fidelity.py",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_identity():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"commit": commit, "dirty_paths": dirty, "clean": not dirty}


def algorithm_identity():
    """Fingerprint only files that determine this qualification's numerics."""
    files = {}
    for relative in QUALIFICATION_SOURCE_PATHS:
        path = ROOT / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "files": files,
        "sha256": fingerprint(files),
    }


def qualification_identity(args, device):
    configuration = get_tier_config("high_fidelity_10x")
    hardware = {"requested_device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        hardware.update({
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        })
    return {
        "schema_version": 2,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layer": LAYER,
        "projection": PROJECTION,
        "backend": args.backend,
        "statistics_dtype": args.statistics_dtype,
        "calibration_tokens": int(args.n_calib),
        "validation_tokens": int(args.n_validation),
        "sequence_length": int(args.seq_len),
        "calibration_sampling": "uniform",
        "validation_sampling": "uniform",
        "calibration_seed": 42,
        "validation_seed": 43,
        "windows_per_chunk": int(args.windows_per_chunk),
        "ranks": list(configuration["ranks"]),
        "bridge_rank": int(configuration["r_bridge"]),
        "sparse_budget": int(configuration["n_outliers"]),
        "sweeps": 2,
        "prior_gamma": 0.01,
        "activation_scale_exponent": 0.5,
        "dmrg_chunk_size": 32,
        "diagonal_prior_strength": 0.0,
        "site_order": [0, 1, 2, 3],
        "quantize_components": True,
        "stage_diagnostics": list(DIAGNOSTIC_STAGES),
        "hardware": hardware,
    }


def tensor_sha256(tensor):
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def artifact_coefficient_count(artifact):
    total = sum(int(core.numel()) for core in artifact["cores"])
    total += int(artifact["d_awt"].numel())
    for name in ("A_bridge", "B_bridge", "sparse_values"):
        value = artifact.get(name)
        if value is not None:
            total += int(value.numel())
    return total


class Telemetry:
    """Sample process and CUDA memory without changing compiler behavior."""

    def __init__(self, device, interval=0.25):
        self.device = torch.device(device)
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_process_rss_bytes = 0
        self.peak_cuda_device_used_bytes = 0
        self.samples = 0

    def _sample(self):
        while not self.stop_event.is_set():
            self.samples += 1
            self.peak_process_rss_bytes = max(
                self.peak_process_rss_bytes,
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            )
            if self.device.type == "cuda":
                free, total = torch.cuda.mem_get_info(self.device)
                self.peak_cuda_device_used_bytes = max(
                    self.peak_cuda_device_used_bytes, int(total - free),
                )
            self.stop_event.wait(self.interval)

    def start(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._sample_once()

    def _sample_once(self):
        self.peak_process_rss_bytes = max(
            self.peak_process_rss_bytes,
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        )
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            self.peak_cuda_device_used_bytes = max(
                self.peak_cuda_device_used_bytes, int(total - free),
            )

    def record(self):
        result = {
            "process_peak_rss_bytes": self.peak_process_rss_bytes,
            "sampler_peak_cuda_device_used_bytes": self.peak_cuda_device_used_bytes,
            "sampler_interval_seconds": self.interval,
            "sampler_count": self.samples,
        }
        if self.device.type == "cuda":
            result.update({
                "torch_peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                "torch_peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                "cuda_device_name": torch.cuda.get_device_name(self.device),
                "cuda_total_bytes": int(torch.cuda.get_device_properties(self.device).total_memory),
            })
        return result


def down_projection_input(weights, x, *, seq_len, n_heads, n_kv_heads, rms_norm_eps):
    """Return the exact dense SwiGLU activation consumed by ``down_proj``."""
    norm1 = RMSNorm(
        x.shape[0], eps=rms_norm_eps, weight=weights["input_layernorm"],
    )(x)
    q = weights["q_proj"] @ norm1
    k = weights["k_proj"] @ norm1
    v = weights["v_proj"] @ norm1
    context = scaled_dot_product_attention_2d(q, k, v, n_heads, n_kv_heads, seq_len)
    residual = x + weights["o_proj"] @ context
    norm2 = RMSNorm(
        x.shape[0], eps=rms_norm_eps, weight=weights["post_attention_layernorm"],
    )(residual)
    gate = weights["gate_proj"] @ norm2
    up = weights["up_proj"] @ norm2
    return F.silu(gate) * up


def iter_down_inputs(weights, hidden, *, windows_per_chunk, forward_options):
    seq_len = int(forward_options["seq_len"])
    if hidden.ndim != 2 or hidden.shape[1] % seq_len:
        raise ValueError("Hidden states must contain complete calibration windows")
    token_chunk = int(windows_per_chunk) * seq_len
    device = weights[PROJECTION].device
    for start in range(0, hidden.shape[1], token_chunk):
        stop = min(start + token_chunk, hidden.shape[1])
        x = hidden[:, start:stop].to(device=device, dtype=weights["q_proj"].dtype)
        projection_input = down_projection_input(weights, x, **forward_options)
        yield start, stop, projection_input
        del x


def materialize_down_inputs(weights, hidden, *, windows_per_chunk, forward_options):
    chunks = []
    for _, _, chunk in iter_down_inputs(
        weights, hidden, windows_per_chunk=windows_per_chunk,
        forward_options=forward_options,
    ):
        chunks.append(chunk)
    return torch.cat(chunks, dim=1)


def accumulate_down_covariance(
    weights, hidden, *, windows_per_chunk, forward_options, accumulation_dtype,
):
    dimension = int(weights[PROJECTION].shape[1])
    accumulator = InputCovarianceAccumulator(
        dimension,
        device=weights[PROJECTION].device,
        accumulation_dtype=accumulation_dtype,
    )
    for _, _, chunk in iter_down_inputs(
        weights, hidden, windows_per_chunk=windows_per_chunk,
        forward_options=forward_options,
    ):
        accumulator.update(chunk)
        del chunk
    return accumulator.finalize()


def projection_metrics_chunked(
    dense_weight, reconstructed_weight, weights, hidden, *, windows_per_chunk,
    forward_options,
):
    squared_error = 0.0
    target_squared = 0.0
    prediction_squared = 0.0
    dot = 0.0
    samples = 0
    for _, _, projection_input in iter_down_inputs(
        weights, hidden, windows_per_chunk=windows_per_chunk,
        forward_options=forward_options,
    ):
        target = dense_weight @ projection_input
        prediction = reconstructed_weight @ projection_input
        squared_error += float((target - prediction).double().square().sum().item())
        target_squared += float(target.double().square().sum().item())
        prediction_squared += float(prediction.double().square().sum().item())
        dot += float((target.double() * prediction.double()).sum().item())
        samples += int(projection_input.shape[1])
        del projection_input, target, prediction
    denominator = max(math.sqrt(target_squared * prediction_squared), 1e-30)
    cosine = dot / denominator
    return {
        "sample_count": samples,
        "relative_error": math.sqrt(squared_error) / max(math.sqrt(target_squared), 1e-30),
        "cosine_similarity": cosine,
        "angular_drift": 1.0 - cosine,
        "norm_ratio": math.sqrt(prediction_squared) / max(math.sqrt(target_squared), 1e-30),
    }


def functional_disagreement_from_inputs(
    reference_weight, variant_weight, inputs, *, token_chunk=512,
):
    """Measure D_X directly between two operators on identical activations."""
    squared_difference = 0.0
    reference_squared = 0.0
    variant_squared = 0.0
    dot = 0.0
    for start in range(0, inputs.shape[1], token_chunk):
        stop = min(start + token_chunk, inputs.shape[1])
        chunk = inputs[:, start:stop]
        reference_output = reference_weight @ chunk
        variant_output = variant_weight @ chunk
        squared_difference += float(
            (variant_output - reference_output).double().square().sum().item()
        )
        reference_squared += float(reference_output.double().square().sum().item())
        variant_squared += float(variant_output.double().square().sum().item())
        dot += float(
            (reference_output.double() * variant_output.double()).sum().item()
        )
        del chunk, reference_output, variant_output
    denominator = max(math.sqrt(reference_squared * variant_squared), 1e-30)
    cosine = dot / denominator
    return {
        "sample_count": int(inputs.shape[1]),
        "relative_functional_disagreement": (
            math.sqrt(squared_difference) / max(math.sqrt(reference_squared), 1e-30)
        ),
        "cosine_similarity": cosine,
        "angular_drift": 1.0 - cosine,
        "norm_ratio": (
            math.sqrt(variant_squared) / max(math.sqrt(reference_squared), 1e-30)
        ),
    }


def relative_weight_difference(actual, reference):
    return float(
        torch.linalg.vector_norm(actual.float() - reference.float())
        .div(torch.linalg.vector_norm(reference.float()).clamp_min(1e-30))
        .item()
    )


def validate_stage_snapshots(output, report, expected_algorithm):
    snapshots = report.get("stage_snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(DIAGNOSTIC_STAGES):
        return False
    for stage, metadata in snapshots.items():
        path = output / "stage_weights" / f"{stage}.pt"
        if (
            not path.is_file()
            or metadata.get("sha256") != sha256_file(path)
            or metadata.get("bytes") != path.stat().st_size
        ):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("stage") != stage
            or payload.get("algorithm_identity") != expected_algorithm
            or not torch.is_tensor(payload.get("weight"))
        ):
            return False
    return True


def validate_completed(output, expected_identity, expected_algorithm):
    report_path = output / "report.json"
    artifact_path = output / "artifact.pt"
    if not report_path.exists() or not artifact_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
        if not (
            report.get("status") == "PASS"
            and report.get("artifact", {}).get("sha256") == sha256_file(artifact_path)
            and report.get("algorithm_identity") == expected_algorithm
            and report.get("qualification_identity") == expected_identity
            and validate_stage_snapshots(output, report, expected_algorithm)
        ):
            return False
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        return (
            payload.get("algorithm_identity") == expected_algorithm
            and payload.get("qualification_identity") == expected_identity
        )
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError, EOFError):
        return False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("full_matrix", "streaming_statistics"), required=True)
    parser.add_argument("--statistics-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--n-calib", type=int, default=4096)
    parser.add_argument("--n-validation", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--windows-per-chunk", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()

    if args.model_id != MODEL_ID or args.model_revision != MODEL_REVISION:
        raise ValueError("Qualification requires the pinned production model snapshot")
    if args.n_calib != 4096 or args.n_validation != 1024 or args.seq_len != 128:
        raise ValueError("Initial qualification is pinned to 4096/1024 tokens and 128-token windows")
    if args.windows_per_chunk <= 0:
        raise ValueError("windows-per-chunk must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA qualification requested but CUDA is unavailable")
    current_source = source_identity()
    if not current_source["clean"]:
        raise RuntimeError(
            "Qualification requires a clean source tree; commit or remove local changes first"
        )
    expected_identity = qualification_identity(args, device)
    current_algorithm = algorithm_identity()
    if args.backend == "full_matrix" and args.reference_output is not None:
        raise ValueError("The full-matrix reference cannot reference itself")
    if args.backend == "streaming_statistics" and args.reference_output is None:
        raise ValueError("Streaming qualification requires --reference-output")
    if validate_completed(args.output, expected_identity, current_algorithm):
        print(f"Reusing verified completed stage: {args.output}", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / "artifact.pt"
    if artifact_path.exists():
        raise RuntimeError(f"Unverified artifact already exists: {artifact_path}")

    progress = {
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "phase": "initializing",
        "backend": args.backend,
        "statistics_dtype": args.statistics_dtype,
        "scope": "real Llama-2-7B layer-0 down_proj qualification",
        "source": current_source,
        "algorithm_identity": current_algorithm,
        "qualification_identity": expected_identity,
    }
    atomic_json_save(progress, args.output / "progress.json")

    def record_phase(phase):
        progress["phase"] = phase
        progress["updated_at_utc"] = utc_now()
        atomic_json_save(progress, args.output / "progress.json")
        print(f"QUALIFICATION_PHASE={phase}", flush=True)

    def terminate(signum, _frame):
        progress.update({
            "status": "INTERRUPTED",
            "signal": int(signum),
            "interrupted_at_utc": utc_now(),
        })
        atomic_json_save(progress, args.output / "progress.json")
        raise TimeoutError(f"Qualification interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    telemetry = Telemetry(device)
    telemetry.start()
    started = time.monotonic()
    try:
        record_phase("loading_pinned_model_and_data")
        config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
        validate_llama2_config(config)
        pinned = PinnedWeights(args.model_id, args.model_revision)
        embedding = pinned.tensor("model.embed_tokens.weight")
        train_hidden, validation_hidden, input_data = build_real_token_calibration_data(
            embedding,
            args.model_id,
            args.n_calib,
            args.n_validation,
            args.seq_len,
            revision=args.model_revision,
            calibration_sampling="uniform",
            validation_sampling="uniform",
        )
        del embedding
        weights = pinned.layer(LAYER, device)
        del pinned
        dense_weight = weights[PROJECTION]
        forward_options = {
            "seq_len": args.seq_len,
            "n_heads": config.num_attention_heads,
            "n_kv_heads": config.num_key_value_heads,
            "rms_norm_eps": config.rms_norm_eps,
        }
        configuration = get_tier_config("high_fidelity_10x")
        stage_directory = args.output / "stage_weights"
        stage_directory.mkdir(parents=True, exist_ok=True)
        stage_snapshots = {}

        def record_stage(stage, weight):
            if stage not in DIAGNOSTIC_STAGES:
                raise ValueError(f"Unexpected diagnostic stage: {stage}")
            path = stage_directory / f"{stage}.pt"
            payload = {
                "stage": stage,
                "storage_dtype": "float16",
                "source": current_source,
                "algorithm_identity": current_algorithm,
                "qualification_identity": expected_identity,
                "weight": weight.detach().to(device="cpu", dtype=torch.float16),
            }
            atomic_torch_save(payload, path)
            stage_snapshots[stage] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "storage_dtype": "float16",
            }
            print(f"QUALIFICATION_STAGE_SNAPSHOT={stage}", flush=True)

        common = {
            "W": dense_weight,
            "o_modes": get_modes_for_projection(*dense_weight.shape)[0],
            "i_modes": get_modes_for_projection(*dense_weight.shape)[1],
            "ranks": configuration["ranks"],
            "r_bridge": configuration["r_bridge"],
            "n_outliers": configuration["n_outliers"],
            "n_sweeps": 2,
            "prior_gamma": 0.01,
            "alpha_awt": 0.5,
            "chunk_size": 32,
            "return_artifacts": True,
            "verbose": True,
            "quantize_components": True,
            "diagonal_prior_strength": 0.0,
            "site_order": (0, 1, 2, 3),
            "diagnostic_callback": record_stage,
        }

        record_phase("collecting_calibration_inputs")
        if args.backend == "full_matrix":
            calibration_input = materialize_down_inputs(
                weights,
                train_hidden,
                windows_per_chunk=args.windows_per_chunk,
                forward_options=forward_options,
            )
            target = dense_weight @ calibration_input
            record_phase("compiling_projection")
            result = nox_hybrid_dmrg_decompose(
                X_calib=calibration_input,
                Y_dense=target,
                **common,
            )
            del calibration_input, target
        else:
            accumulation_dtype = {
                "float32": torch.float32,
                "float64": torch.float64,
            }[args.statistics_dtype]
            covariance = accumulate_down_covariance(
                weights,
                train_hidden,
                windows_per_chunk=args.windows_per_chunk,
                forward_options=forward_options,
                accumulation_dtype=accumulation_dtype,
            )
            statistics = covariance.projection_statistics(dense_weight)
            del covariance
            record_phase("compiling_projection")
            result = nox_hybrid_dmrg_decompose_from_statistics(
                calibration_statistics=statistics,
                **common,
            )
            del statistics

        compiler_weight, _, _, realized_ranks, artifact = result
        record_phase("validating_artifact")
        reconstructed = reconstruct_from_artifact(artifact, device=device, dtype=torch.float32)
        if reconstructed.shape != dense_weight.shape or not torch.isfinite(reconstructed).all():
            raise FloatingPointError("Decoded qualification artifact is invalid")
        serialization_difference = float(
            torch.linalg.vector_norm(reconstructed - compiler_weight.float())
            .div(torch.linalg.vector_norm(compiler_weight.float()).clamp_min(1e-30))
            .item()
        )
        if not math.isfinite(serialization_difference) or serialization_difference > 1e-2:
            raise RuntimeError(f"Artifact reconstruction mismatch: {serialization_difference}")
        q4_complete = (
            artifact.get("core_quantization_bits") == 4
            and artifact.get("sparse_bridge_quantization_bits") == 4
            and set(artifact.get("precision_diagnostics", {})) == EXPECTED_Q4_COMPONENTS
            and artifact.get("compiler_hyperparameters", {}).get("n_sweeps") == 2
        )
        if not q4_complete:
            raise RuntimeError("Qualification artifact does not contain complete two-sweep Q4 diagnostics")
        record_stage("serialized_artifact", reconstructed)
        if set(stage_snapshots) != set(DIAGNOSTIC_STAGES):
            missing = sorted(set(DIAGNOSTIC_STAGES) - set(stage_snapshots))
            raise RuntimeError(f"Missing compiler stage snapshots: {missing}")

        record_phase("measuring_calibration_and_validation")
        calibration_metrics = projection_metrics_chunked(
            dense_weight,
            reconstructed,
            weights,
            train_hidden,
            windows_per_chunk=args.windows_per_chunk,
            forward_options=forward_options,
        )
        validation_metrics = projection_metrics_chunked(
            dense_weight,
            reconstructed,
            weights,
            validation_hidden,
            windows_per_chunk=args.windows_per_chunk,
            forward_options=forward_options,
        )
        artifact_payload = {
            "layer": LAYER,
            "projection": PROJECTION,
            "backend": args.backend,
            "statistics_dtype": args.statistics_dtype,
            "input_data": input_data,
            "source": current_source,
            "algorithm_identity": current_algorithm,
            "qualification_identity": expected_identity,
            "artifact": artifact,
        }
        atomic_torch_save(artifact_payload, artifact_path)
        qualification_runtime_seconds = time.monotonic() - started
        telemetry.finish()
        qualification_memory = telemetry.record()

        direct_functional_disagreement = None
        functional_diagnostic_runtime_seconds = 0.0
        diagnostic_memory = None
        if args.reference_output is not None:
            record_phase("measuring_direct_functional_disagreement")
            diagnostic_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            reference_identity = dict(expected_identity)
            reference_identity.update({
                "backend": "full_matrix",
                "statistics_dtype": "float64",
            })
            if not validate_completed(
                args.reference_output, reference_identity, current_algorithm,
            ):
                raise RuntimeError("Reference output failed artifact, stage, or algorithm validation")
            reference_report = json.loads(
                (args.reference_output / "report.json").read_text()
            )
            if (
                reference_report.get("input_data") != input_data
                or reference_report.get("hidden_state_sha256") != {
                    "calibration": tensor_sha256(train_hidden),
                    "validation": tensor_sha256(validation_hidden),
                }
            ):
                raise RuntimeError("Reference and variant activation inputs differ")
            reference_payload = torch.load(
                args.reference_output / "artifact.pt",
                map_location="cpu",
                weights_only=False,
            )
            reference_reconstructed = reconstruct_from_artifact(
                reference_payload["artifact"], device=device, dtype=torch.float32,
            )
            calibration_inputs = materialize_down_inputs(
                weights,
                train_hidden,
                windows_per_chunk=args.windows_per_chunk,
                forward_options=forward_options,
            )
            validation_inputs = materialize_down_inputs(
                weights,
                validation_hidden,
                windows_per_chunk=args.windows_per_chunk,
                forward_options=forward_options,
            )

            def compare_pair(reference_weight, variant_weight):
                return {
                    "relative_reconstructed_weight_difference": (
                        relative_weight_difference(variant_weight, reference_weight)
                    ),
                    "calibration": functional_disagreement_from_inputs(
                        reference_weight, variant_weight, calibration_inputs,
                    ),
                    "validation": functional_disagreement_from_inputs(
                        reference_weight, variant_weight, validation_inputs,
                    ),
                }

            direct_functional_disagreement = {
                "definition": "||((W_variant-W_reference)X)||_F / ||W_reference X||_F",
                "final_artifact": compare_pair(
                    reference_reconstructed, reconstructed,
                ),
                "stages": {},
            }
            for stage in DIAGNOSTIC_STAGES:
                reference_stage = torch.load(
                    args.reference_output / "stage_weights" / f"{stage}.pt",
                    map_location="cpu",
                    weights_only=False,
                )["weight"].to(device=device, dtype=torch.float32)
                variant_stage = torch.load(
                    stage_directory / f"{stage}.pt",
                    map_location="cpu",
                    weights_only=False,
                )["weight"].to(device=device, dtype=torch.float32)
                direct_functional_disagreement["stages"][stage] = compare_pair(
                    reference_stage, variant_stage,
                )
                del reference_stage, variant_stage
            del (
                reference_payload,
                reference_reconstructed,
                calibration_inputs,
                validation_inputs,
            )
            functional_diagnostic_runtime_seconds = time.monotonic() - diagnostic_started
            diagnostic_memory = {
                "torch_peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda" else None
                ),
                "torch_peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(device))
                    if device.type == "cuda" else None
                ),
            }
        completed_source = source_identity()
        completed_algorithm = algorithm_identity()
        if completed_algorithm != current_algorithm:
            raise RuntimeError("Relevant compiler source changed during qualification")
        report = {
            "status": "PASS",
            "completed_at_utc": utc_now(),
            "scope": "real Llama-2-7B layer-0 down_proj qualification; not a model-quality benchmark",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "source": completed_source,
            "algorithm_identity": completed_algorithm,
            "qualification_identity": expected_identity,
            "runtime": {
                "torch_version": torch.__version__,
                "cuda_runtime_version": torch.version.cuda,
            },
            "layer": LAYER,
            "projection": PROJECTION,
            "weight_shape": list(dense_weight.shape),
            "backend": args.backend,
            "statistics_dtype": args.statistics_dtype,
            "input_data": input_data,
            "hidden_state_sha256": {
                "calibration": tensor_sha256(train_hidden),
                "validation": tensor_sha256(validation_hidden),
            },
            "configuration": {
                "ranks": list(configuration["ranks"]),
                "bridge_rank": configuration["r_bridge"],
                "sparse_budget": configuration["n_outliers"],
                "sweeps": 2,
                "activation_scale_exponent": 0.5,
                "quantization": "Q4 simulation for shield, bridge, and final-sweep MPO cores",
                "windows_per_chunk": args.windows_per_chunk,
            },
            "realized_ranks": list(realized_ranks),
            "independent_numerical_coefficients": artifact_coefficient_count(artifact),
            "two_sweep_q4_diagnostics_complete": q4_complete,
            "serialization_relative_difference": serialization_difference,
            "calibration_metrics": calibration_metrics,
            "validation_metrics": validation_metrics,
            "calibration_statistics_diagnostics": artifact.get("calibration_statistics_diagnostics"),
            "bridge_statistics_diagnostics": artifact.get("bridge_statistics_diagnostics"),
            "direct_functional_disagreement": direct_functional_disagreement,
            "stage_snapshots": stage_snapshots,
            "runtime_seconds": time.monotonic() - started,
            "core_qualification_runtime_seconds": qualification_runtime_seconds,
            "functional_diagnostic_runtime_seconds": functional_diagnostic_runtime_seconds,
            "memory": qualification_memory,
            "functional_diagnostic_memory": diagnostic_memory,
            "artifact": {
                "path": str(artifact_path),
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            },
        }
        atomic_json_save(report, args.output / "report.json")
        progress.update({
            "status": "PASS",
            "phase": "complete",
            "completed_at_utc": report["completed_at_utc"],
            "runtime_seconds": report["runtime_seconds"],
        })
        atomic_json_save(progress, args.output / "progress.json")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        del result, compiler_weight, reconstructed, artifact
    except BaseException as error:
        telemetry.finish()
        progress.update({
            "status": "FAIL",
            "failed_at_utc": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "runtime_seconds": time.monotonic() - started,
            "memory": telemetry.record(),
        })
        atomic_json_save(progress, args.output / "progress.json")
        raise
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
