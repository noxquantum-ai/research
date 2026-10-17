#!/usr/bin/env python3
"""Final bounded feasibility test for the shared-neuron SwiGLU representation.

Exactly ranks 426 and 651 are tested. Gate/Up factors are constructed from
calibration data only. For each rank, an oracle Down readout is solved on held-
out data solely as a feasibility bound. If that oracle passes, a deployable
Down readout is selected and fitted algebraically using calibration data only,
then frozen and evaluated on held-out MLP, active-transform, and block outputs.
No gradients, iterative repair, dense residual, PPL, or prefix compile occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact  # noqa: E402
from nox.joint_swiglu import (  # noqa: E402
    apply_joint_swiglu,
    build_joint_swiglu_artifact,
    shared_swiglu_payload_bytes,
    shared_swiglu_serialized_bytes,
)
from nox.swiglu_readout import (  # noqa: E402
    fit_linear_readout,
    relative_output_error,
    select_calibration_ridge_readout,
    shared_swiglu_features,
)
from nox.utils import compute_metrics  # noqa: E402
from scripts.diagnose_cross_layer_interactions import forward_details  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
RANKS = (426, 651)
RIDGE_RATIOS = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0)
MLP_THRESHOLD = 0.470
BLOCK_THRESHOLD = 0.273


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _serialized_bytes(value) -> int:
    stream = io.BytesIO()
    torch.save(value, stream)
    return int(stream.tell())


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


def _save(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)


@torch.no_grad()
def run(args) -> dict:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this authorized feasibility run requires CUDA")
    checkpoint_path = args.checkpoint_dir / f"compressed_layer_{args.layer}.pt"
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
    calibration_path = args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt"
    heldout_path = args.checkpoint_dir / f"layer_input_validation_layer_{args.layer}.pt"
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

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    paths = {}
    for split, raw in (("calibration", calibration_input), ("heldout", heldout_input)):
        dense_path = forward_details(dense, raw, **options)
        compressed_path = forward_details(compressed, raw, **options)
        paths[split] = {
            "compressed_mlp_input": compressed_path["mlp_input"],
            "compressed_residual1": compressed_path["residual1"],
            "compressed_attention_output": compressed_path["attention_output"],
            "compressed_mlp_output": compressed_path["mlp_output"],
            "compressed_block_output": compressed_path["output"],
            "dense_block_output": dense_path["output"],
            "dense_active_transform": dense_path["active_transform"],
        }

    old_artifacts = {
        name: checkpoint["compressed_artifacts"][name] for name in MLP_PROJECTIONS
    }
    old_coefficients = sum(_float_coefficients(value) for value in old_artifacts.values())
    old_bytes = sum(_serialized_bytes(value) for value in old_artifacts.values())
    heldout = paths["heldout"]
    baseline = {
        "mlp_on_compressed_trajectory": _metrics(
            _dense_mlp(dense, heldout["compressed_mlp_input"]),
            heldout["compressed_mlp_output"],
        ),
        "active_transform": _metrics(
            heldout["dense_active_transform"],
            heldout["compressed_attention_output"] + heldout["compressed_mlp_output"],
        ),
        "full_block": _metrics(
            heldout["dense_block_output"], heldout["compressed_block_output"],
        ),
    }
    result = {
        "diagnostic": "final_shared_swiglu_fixed_feature_readout_feasibility",
        "decision": "INCOMPLETE",
        "layer": args.layer,
        "device": str(device),
        "ranks": list(RANKS),
        "thresholds": {
            "heldout_compressed_trajectory_mlp_relative_error_max": MLP_THRESHOLD,
            "heldout_full_block_relative_error_max": BLOCK_THRESHOLD,
            "active_transform": "no regression versus saved NEC baseline",
        },
        "ridge_ratios": list(RIDGE_RATIOS),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "calibration_input_sha256": _sha256(calibration_path),
        "heldout_input_sha256": _sha256(heldout_path),
        "dense_weight_sha256": {
            name: _sha256(args.dense_weights_dir / f"layer_{args.layer}_{name}.pt")
            for name in PROJECTIONS
        },
        "calibration_tokens": int(calibration_input.shape[1]),
        "heldout_tokens": int(heldout_input.shape[1]),
        "saved_nec_budget": {
            "mlp_float_coefficients": old_coefficients,
            "mlp_serialized_artifact_bytes": old_bytes,
        },
        "saved_nec_heldout": baseline,
        "post_compression_optimization": False,
        "uses_gradients": False,
        "uses_dense_residual": False,
        "rank_results": {},
    }
    _save(args.output, result)

    for rank in RANKS:
        if time.perf_counter() - started > args.max_runtime_seconds:
            raise TimeoutError("strict diagnostic runtime cap reached between ranks")
        rank_started = time.perf_counter()
        artifact = build_joint_swiglu_artifact(
            dense["gate_proj"], dense["up_proj"], dense["down_proj"],
            paths["calibration"]["compressed_mlp_input"],
            rank=rank, storage_dtype=torch.float16, seed=0,
        )
        initial_bytes = shared_swiglu_serialized_bytes(artifact)
        budget_kind = "floating_coefficients" if rank == 426 else "serialized_bytes"
        budget_pass = (
            artifact["coefficient_count"] <= old_coefficients
            if rank == 426 else initial_bytes <= old_bytes
        )
        calibration_features = shared_swiglu_features(
            artifact, paths["calibration"]["compressed_mlp_input"],
        )
        heldout_features = shared_swiglu_features(
            artifact, heldout["compressed_mlp_input"],
        )
        calibration_target = _dense_mlp(
            dense, paths["calibration"]["compressed_mlp_input"],
        )
        heldout_target = _dense_mlp(dense, heldout["compressed_mlp_input"])

        oracle_readout = fit_linear_readout(heldout_features, heldout_target)
        oracle_output = oracle_readout @ heldout_features
        oracle_mlp_error = relative_output_error(
            oracle_readout, heldout_features, heldout_target,
        )
        oracle_block = heldout["compressed_residual1"] + oracle_output
        oracle_active = heldout["compressed_attention_output"] + oracle_output
        record = {
            "rank": rank,
            "budget_kind": budget_kind,
            "float_coefficients": int(artifact["coefficient_count"]),
            "initial_serialized_bytes": initial_bytes,
            "budget_pass": bool(budget_pass),
            "oracle": {
                "diagnostic_only": True,
                "uses_heldout_targets": True,
                "mlp_relative_error": oracle_mlp_error,
                "active_transform": _metrics(
                    heldout["dense_active_transform"], oracle_active,
                ),
                "full_block": _metrics(heldout["dense_block_output"], oracle_block),
                "mlp_threshold_pass": oracle_mlp_error <= MLP_THRESHOLD,
            },
        }
        if oracle_mlp_error <= MLP_THRESHOLD:
            readout, selection = select_calibration_ridge_readout(
                calibration_features, calibration_target,
                sequence_length=args.seq_len, ridge_ratios=RIDGE_RATIOS,
            )
            prequantized_error = relative_output_error(
                readout, heldout_features, heldout_target,
            )
            artifact["down_factor"] = readout.to(dtype=torch.float16)
            artifact["construction_objective"] = (
                "calibration_only_closed_form_ridge_output_readout"
            )
            artifact["ridge_selection"] = selection
            artifact["construction_uses_validation"] = False
            artifact["post_compression_optimization"] = False
            candidate_output = apply_joint_swiglu(
                artifact, heldout["compressed_mlp_input"],
            )
            candidate_block = heldout["compressed_residual1"] + candidate_output
            candidate_active = heldout["compressed_attention_output"] + candidate_output
            mlp_metrics = _metrics(heldout_target, candidate_output)
            active_metrics = _metrics(heldout["dense_active_transform"], candidate_active)
            block_metrics = _metrics(heldout["dense_block_output"], candidate_block)
            final_bytes = shared_swiglu_serialized_bytes(artifact)
            budget_pass = (
                artifact["coefficient_count"] <= old_coefficients
                if rank == 426 else final_bytes <= old_bytes
            )
            passes = {
                "mlp": mlp_metrics["relative_error"] <= MLP_THRESHOLD,
                "full_block": block_metrics["relative_error"] <= BLOCK_THRESHOLD,
                "active_transform_no_regression": (
                    active_metrics["relative_error"]
                    <= baseline["active_transform"]["relative_error"]
                ),
                "budget": bool(budget_pass),
            }
            record["calibration_only_readout"] = {
                "selection": selection,
                "heldout_pre_fp16_readout_mlp_relative_error": prequantized_error,
                "heldout_mlp": mlp_metrics,
                "heldout_active_transform": active_metrics,
                "heldout_full_block": block_metrics,
                "serialized_bytes": final_bytes,
                "payload_bytes": shared_swiglu_payload_bytes(artifact),
                "passes": passes,
                "overall_pass": all(passes.values()),
            }
            record["budget_pass"] = bool(budget_pass)
        else:
            record["calibration_only_readout"] = {
                "skipped": True,
                "reason": "feature-specific heldout oracle exceeds MLP threshold",
            }
        torch.cuda.synchronize(device)
        record["runtime_seconds"] = float(time.perf_counter() - rank_started)
        result["rank_results"][str(rank)] = record
        _save(args.output, result)

    deployable_pass = any(
        value.get("calibration_only_readout", {}).get("overall_pass", False)
        for value in result["rank_results"].values()
    )
    result["decision"] = "GO" if deployable_pass else "NO-GO"
    result["decision_reason"] = (
        "at least one calibration-only candidate meets all declared gates"
        if deployable_pass else
        "no calibration-only candidate meets all declared gates"
    )
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
    parser.add_argument("--max-runtime-seconds", type=float, default=600.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "decision": result["decision"],
        "runtime_seconds": result["runtime_seconds"],
        "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
        "rank_results": {
            rank: {
                "oracle_mlp": value["oracle"]["mlp_relative_error"],
                "calibration_only": value.get("calibration_only_readout", {}),
            }
            for rank, value in result["rank_results"].items()
        },
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
