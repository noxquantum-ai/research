#!/usr/bin/env python3
"""Quality-first native-coordinate SwiGLU structural compression diagnostic.

The experiment reuses the saved layer-0 NEC attention trajectory.  It removes
whole native SwiGLU neurons, constructs a calibration-only prior-anchored Down
readout, freezes the artifact, and evaluates untouched held-out inputs.  No
gradient optimization, dense residual, checkpoint healing, PPL, or prefix
compilation is performed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact  # noqa: E402
from nox.native_swiglu import (  # noqa: E402
    apply_native_swiglu,
    build_native_swiglu_artifact,
    fit_anchored_down_readout,
    native_swiglu_features,
    native_swiglu_payload_bytes,
    native_swiglu_serialized_bytes,
)
from nox.utils import compute_metrics  # noqa: E402
from scripts.diagnose_cross_layer_interactions import forward_details  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RIDGE_RATIOS = (1e-6, 1e-4, 1e-2, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_coefficients(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel()) if value.is_floating_point() else 0
    if isinstance(value, dict):
        return sum(_float_coefficients(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_float_coefficients(item) for item in value)
    return 0


def _metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    return {name: float(value) for name, value in compute_metrics(reference, estimate).items()}


def _dense_mlp(weights: dict, inputs: torch.Tensor) -> torch.Tensor:
    return weights["down_proj"] @ (
        F.silu(weights["gate_proj"] @ inputs) * (weights["up_proj"] @ inputs)
    )


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@torch.no_grad()
def run(args) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real-model diagnostic requires CUDA")
    checkpoint_path = args.checkpoint_dir / f"compressed_layer_{args.layer}.pt"
    calibration_path = args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt"
    heldout_path = args.checkpoint_dir / f"layer_input_validation_layer_{args.layer}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epsilon = args.rms_norm_eps or float(checkpoint.get("rms_norm_eps", 1e-5))
    dense = {
        name: torch.load(
            args.dense_weights_dir / f"layer_{args.layer}_{name}.pt",
            map_location="cpu", weights_only=True,
        ).to(device=device, dtype=torch.float32)
        for name in PROJECTIONS
    }
    for name in ("input_layernorm", "post_attention_layernorm"):
        dense[name] = checkpoint[name].to(device=device, dtype=torch.float32)
    compressed = dict(dense)
    compressed.update({
        name: reconstruct_from_artifact(
            checkpoint["compressed_artifacts"][name], device=device, dtype=torch.float32,
        )
        for name in PROJECTIONS
    })
    calibration_input = torch.load(
        calibration_path, map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    heldout_input = torch.load(
        heldout_path, map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    options = {
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "seq_len": args.seq_len,
        "rms_norm_eps": epsilon,
    }

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    paths = {}
    for split, raw in (("calibration", calibration_input), ("heldout", heldout_input)):
        dense_path = forward_details(dense, raw, **options)
        compressed_path = forward_details(compressed, raw, **options)
        paths[split] = {
            "dense_mlp_input": dense_path["mlp_input"],
            "dense_residual1": dense_path["residual1"],
            "dense_attention_output": dense_path["attention_output"],
            "compressed_mlp_input": compressed_path["mlp_input"],
            "compressed_residual1": compressed_path["residual1"],
            "compressed_attention_output": compressed_path["attention_output"],
            "compressed_mlp_output": compressed_path["mlp_output"],
            "compressed_block_output": compressed_path["output"],
            "dense_active_transform": dense_path["active_transform"],
            "dense_block_output": dense_path["output"],
        }
        del dense_path, compressed_path

    heldout = paths["heldout"]
    trajectory = args.trajectory
    mlp_input_key = f"{trajectory}_mlp_input"
    residual_key = f"{trajectory}_residual1"
    attention_key = f"{trajectory}_attention_output"
    heldout_target = _dense_mlp(dense, heldout[mlp_input_key])
    compressed_trajectory_target = _dense_mlp(
        dense, heldout["compressed_mlp_input"],
    )
    baseline = {
        "saved_nec_mlp_on_compressed_trajectory": _metrics(
            compressed_trajectory_target, heldout["compressed_mlp_output"],
        ),
        "saved_nec_active_transform": _metrics(
            heldout["dense_active_transform"],
            heldout["compressed_attention_output"] + heldout["compressed_mlp_output"],
        ),
        "saved_nec_full_block": _metrics(
            heldout["dense_block_output"], heldout["compressed_block_output"],
        ),
        "dense_mlp_oracle_with_saved_attention_full_block": _metrics(
            heldout["dense_block_output"],
            heldout["compressed_residual1"] + compressed_trajectory_target,
        ),
        "dense_mlp_oracle_on_candidate_trajectory_full_block": _metrics(
            heldout["dense_block_output"],
            heldout[residual_key] + heldout_target,
        ),
    }
    old_artifacts = {
        name: checkpoint["compressed_artifacts"][name] for name in MLP_PROJECTIONS
    }
    intermediate_size, d_model = dense["gate_proj"].shape
    dense_mlp_coefficients = 3 * d_model * intermediate_size
    result = {
        "diagnostic": "quality_first_native_coordinate_swiglu",
        "status": "INCOMPLETE",
        "layer": args.layer,
        "device": str(device),
        "retain_fractions": list(args.retain_fractions),
        "candidate_trajectory": trajectory,
        "ridge_ratios": list(RIDGE_RATIOS),
        "calibration_tokens": int(calibration_input.shape[1]),
        "heldout_tokens": int(heldout_input.shape[1]),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "calibration_input_sha256": _sha256(calibration_path),
        "heldout_input_sha256": _sha256(heldout_path),
        "dense_weight_sha256": {
            name: _sha256(args.dense_weights_dir / f"layer_{args.layer}_{name}.pt")
            for name in PROJECTIONS
        },
        "budgets": {
            "dense_mlp_float_coefficients": dense_mlp_coefficients,
            "dense_mlp_fp16_payload_bytes": 2 * dense_mlp_coefficients,
            "saved_nec_mlp_float_coefficients": _float_coefficients(old_artifacts),
        },
        "baseline": baseline,
        "uses_gradients": False,
        "uses_dense_residual": False,
        "post_compression_optimization": False,
        "candidates": {},
    }
    _save(args.output, result)

    calibration_target = _dense_mlp(
        dense, paths["calibration"][mlp_input_key],
    )
    for fraction in args.retain_fractions:
        if time.perf_counter() - started > args.max_runtime_seconds:
            raise TimeoutError("runtime cap reached between candidates")
        candidate_started = time.perf_counter()
        artifact = build_native_swiglu_artifact(
            dense["gate_proj"], dense["up_proj"], dense["down_proj"],
            paths["calibration"][mlp_input_key],
            retain_fraction=fraction, storage_dtype=torch.float16,
        )
        calibration_features = native_swiglu_features(
            artifact, paths["calibration"][mlp_input_key],
        )
        heldout_features = native_swiglu_features(
            artifact, heldout[mlp_input_key],
        )
        prior_output = artifact["down_columns"].to(
            device=device, dtype=torch.float32,
        ) @ heldout_features
        prior_block = heldout[residual_key] + prior_output
        readout, selection = fit_anchored_down_readout(
            calibration_features, calibration_target,
            artifact["down_columns"].to(device=device, dtype=torch.float32),
            sequence_length=args.seq_len, ridge_ratios=RIDGE_RATIOS,
        )
        artifact["down_columns"] = readout.to(device="cpu", dtype=torch.float16)
        artifact["construction_objective"] = (
            "native_neuron_selection_plus_calibration_only_prior_anchored_output_ridge"
        )
        artifact["ridge_selection"] = selection
        artifact_path = None
        artifact_file_bytes = None
        if args.artifact_dir is not None:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = args.artifact_dir / (
                f"native_mlp_layer_{args.layer}_retain_{fraction:.6f}.pt"
            )
            if artifact_path.exists():
                raise FileExistsError(f"refusing to overwrite {artifact_path}")
            torch.save({
                "format": "nox_quality_first_native_swiglu_checkpoint_v1",
                "layer_idx": int(args.layer),
                "native_mlp_artifact": artifact,
                "attention_representation": "dense_reference",
                "construction_trajectory": trajectory,
                "post_compression_optimization": False,
            }, artifact_path)
            artifact_file_bytes = artifact_path.stat().st_size
        candidate_output = apply_native_swiglu(
            artifact, heldout[mlp_input_key],
        )
        candidate_block = heldout[residual_key] + candidate_output
        candidate_active = heldout[attention_key] + candidate_output
        record = {
            "retain_fraction_requested": float(fraction),
            "retained_neurons": int(artifact["retained_neurons"]),
            "retained_fraction_actual": float(
                artifact["retained_neurons"] / intermediate_size
            ),
            "float_coefficients": int(artifact["coefficient_count"]),
            "coefficient_fraction_of_dense_mlp": float(
                artifact["coefficient_count"] / dense_mlp_coefficients
            ),
            "payload_bytes": native_swiglu_payload_bytes(artifact),
            "serialized_bytes": native_swiglu_serialized_bytes(artifact),
            "artifact_path": None if artifact_path is None else str(artifact_path),
            "artifact_file_bytes": artifact_file_bytes,
            "ridge_selection": selection,
            "heldout_uncompensated": {
                "mlp": _metrics(heldout_target, prior_output),
                "full_block": _metrics(heldout["dense_block_output"], prior_block),
            },
            "heldout_compensated": {
                "mlp": _metrics(heldout_target, candidate_output),
                "active_transform": _metrics(
                    heldout["dense_active_transform"], candidate_active,
                ),
                "full_block": _metrics(
                    heldout["dense_block_output"], candidate_block,
                ),
            },
            "runtime_seconds": float(time.perf_counter() - candidate_started),
        }
        result["candidates"][f"{fraction:.6f}"] = record
        _save(args.output, result)
        del artifact, calibration_features, heldout_features, readout
        torch.cuda.empty_cache()

    result["status"] = "COMPLETE"
    torch.cuda.synchronize(device)
    result["runtime_seconds"] = float(time.perf_counter() - started)
    result["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    _save(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--rms-norm-eps", type=float)
    parser.add_argument("--retain-fractions", type=float, nargs="+", default=(0.75,))
    parser.add_argument("--trajectory", choices=("compressed", "dense"), default="compressed")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--max-runtime-seconds", type=float, default=900.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "status": result["status"],
        "baseline": result["baseline"],
        "candidates": result["candidates"],
        "runtime_seconds": result["runtime_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
