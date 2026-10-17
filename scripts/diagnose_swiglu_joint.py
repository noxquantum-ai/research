#!/usr/bin/env python3
"""Layer-local joint SwiGLU error decomposition and shared-neuron TTN screen.

This is a diagnostic, not a checkpoint compiler: it reads a saved NEC layer,
measures all dense/compressed Gate-Up-Down combinations, and builds one
calibration-only joint shared-neuron candidate at the saved MLP coefficient
budget.  It performs no training, healing, layer-prefix compilation, or full
model inference.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact  # noqa: E402
from nox.joint_swiglu import (  # noqa: E402
    apply_joint_swiglu,
    build_joint_swiglu_artifact,
    joint_neuron_rank_for_budget,
    mlp_error_decomposition,
    reconstruct_joint_swiglu_weights,
    shared_swiglu_payload_bytes,
    shared_swiglu_serialized_bytes,
)
from nox.utils import compute_metrics  # noqa: E402
from scripts.diagnose_cross_layer_interactions import forward_details  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _float_coefficients(value):
    if torch.is_tensor(value):
        return int(value.numel()) if value.is_floating_point() else 0
    if isinstance(value, dict):
        return sum(_float_coefficients(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_float_coefficients(item) for item in value)
    return 0


def _tensor_scalars(value, *, floating=None):
    if torch.is_tensor(value):
        if floating is None or value.is_floating_point() == floating:
            return int(value.numel())
        return 0
    if isinstance(value, dict):
        return sum(_tensor_scalars(item, floating=floating) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_scalars(item, floating=floating) for item in value)
    return 0


def _tensor_payload_bytes(value):
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_payload_bytes(item) for item in value)
    return 0


def _serialized_bytes(value):
    import io

    stream = io.BytesIO()
    torch.save(value, stream)
    return int(stream.tell())


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable_metrics(reference, candidate):
    return {name: float(value) for name, value in compute_metrics(reference, candidate).items()}


@torch.no_grad()
def run_diagnostic(
    *, checkpoint_dir, dense_weights_dir, layer, device, seq_len, n_heads,
    n_kv_heads, rms_norm_eps, output,
):
    checkpoint_path = checkpoint_dir / f"compressed_layer_{layer}.pt"
    manifest_path = checkpoint_dir / "diagnostic_data_manifest.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    target_device = torch.device(device)
    epsilon = rms_norm_eps or float(checkpoint.get("rms_norm_eps", 1e-5))

    dense = {
        name: torch.load(
            dense_weights_dir / f"layer_{layer}_{name}.pt",
            map_location="cpu", weights_only=True,
        ).to(device=target_device, dtype=torch.float32)
        for name in PROJECTIONS
    }
    for norm in ("input_layernorm", "post_attention_layernorm"):
        dense[norm] = checkpoint[norm].to(device=target_device, dtype=torch.float32)

    compressed = dict(dense)
    compressed.update({
        name: reconstruct_from_artifact(
            checkpoint["compressed_artifacts"][name],
            device=target_device, dtype=torch.float32,
        )
        for name in PROJECTIONS
    })
    train_input = torch.load(
        checkpoint_dir / f"calibration_input_layer_{layer}.pt",
        map_location="cpu", weights_only=True,
    ).to(device=target_device, dtype=torch.float32)
    validation_input = torch.load(
        checkpoint_dir / f"layer_input_validation_layer_{layer}.pt",
        map_location="cpu", weights_only=True,
    ).to(device=target_device, dtype=torch.float32)
    options = {
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "seq_len": seq_len,
        "rms_norm_eps": epsilon,
    }

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
        initial_cuda_memory_bytes = int(torch.cuda.memory_allocated(target_device))
        gpu_properties = torch.cuda.get_device_properties(target_device)
    else:
        initial_cuda_memory_bytes = None
        gpu_properties = None
    started = time.perf_counter()
    split_inputs = {}
    for split, raw_input in (("calibration", train_input), ("heldout", validation_input)):
        dense_path = forward_details(dense, raw_input, **options)
        compressed_path = forward_details(compressed, raw_input, **options)
        split_inputs[split] = {
            "dense_trajectory": dense_path["mlp_input"],
            "compressed_trajectory": compressed_path["mlp_input"],
            "dense_block_output": dense_path["output"],
            "compressed_block_output": compressed_path["output"],
            "compressed_residual1": compressed_path["residual1"],
            "compressed_attention_output": compressed_path["attention_output"],
            "dense_mlp_output": dense_path["mlp_output"],
            "compressed_mlp_output": compressed_path["mlp_output"],
        }

    result = {
        "diagnostic": "layer_local_joint_swiglu_error_decomposition",
        "layer": int(layer),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "device": str(target_device),
        "rms_norm_eps": epsilon,
        "calibration_tokens": int(train_input.shape[1]),
        "heldout_tokens": int(validation_input.shape[1]),
        "fixed_inputs": "dense and compressed transformer trajectories measured separately",
        "dense_weight_sha256": {
            name: _sha256(dense_weights_dir / f"layer_{layer}_{name}.pt")
            for name in PROJECTIONS
        },
        "calibration_input_sha256": _sha256(
            checkpoint_dir / f"calibration_input_layer_{layer}.pt",
        ),
        "heldout_input_sha256": _sha256(
            checkpoint_dir / f"layer_input_validation_layer_{layer}.pt",
        ),
        "gpu": None if gpu_properties is None else {
            "name": gpu_properties.name,
            "total_memory_bytes": int(gpu_properties.total_memory),
        },
        "initial_cuda_memory_bytes": initial_cuda_memory_bytes,
        "post_compression_optimization": False,
        "projection_combinations": {},
        "candidate": {},
    }
    if manifest_path.exists():
        result["diagnostic_data_manifest_sha256"] = _sha256(manifest_path)
        result["diagnostic_data_manifest"] = json.loads(manifest_path.read_text())

    for split, contexts in split_inputs.items():
        result["projection_combinations"][split] = {}
        for trajectory_name in ("dense_trajectory", "compressed_trajectory"):
            context = contexts[trajectory_name]
            ablation = mlp_error_decomposition(
                dense["gate_proj"], dense["up_proj"], dense["down_proj"],
                compressed["gate_proj"], compressed["up_proj"], compressed["down_proj"],
                context,
            )
            result["projection_combinations"][split][trajectory_name] = ablation

    heldout_contexts = split_inputs["heldout"]
    result["saved_nec_block_validation"] = {
        "dense_block_output": _jsonable_metrics(
            heldout_contexts["dense_block_output"],
            heldout_contexts["compressed_block_output"],
        ),
        "dense_active_transform": _jsonable_metrics(
            heldout_contexts["dense_block_output"]
            - validation_input.to(device=target_device, dtype=torch.float32),
            heldout_contexts["compressed_attention_output"]
            + heldout_contexts["compressed_mlp_output"],
        ),
        "mlp_function_on_compressed_trajectory":
            result["projection_combinations"]["heldout"]["compressed_trajectory"]
            ["combinations"][
                "gate_compressed__up_compressed__down_compressed"
            ],
    }

    old_artifacts = {
        name: checkpoint["compressed_artifacts"][name]
        for name in MLP_PROJECTIONS
    }
    old_coefficient_budget = sum(_float_coefficients(value) for value in old_artifacts.values())
    old_all_tensor_scalars = sum(_tensor_scalars(value) for value in old_artifacts.values())
    old_nonfloating_scalars = sum(
        _tensor_scalars(value, floating=False) for value in old_artifacts.values()
    )
    old_tensor_payload_bytes = sum(
        _tensor_payload_bytes(value) for value in old_artifacts.values()
    )
    old_serialized_bytes = sum(_serialized_bytes(value) for value in old_artifacts.values())
    intermediate_size, d_model = dense["gate_proj"].shape
    candidate_rank = joint_neuron_rank_for_budget(
        old_coefficient_budget,
        d_model,
        intermediate_size,
        calibration_tokens=train_input.shape[1],
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        pre_candidate_peak_cuda_bytes = int(
            torch.cuda.max_memory_allocated(target_device),
        )
        candidate_baseline_cuda_bytes = int(
            torch.cuda.memory_allocated(target_device),
        )
        torch.cuda.reset_peak_memory_stats(target_device)
    candidate_started = time.perf_counter()
    candidate = build_joint_swiglu_artifact(
        dense["gate_proj"], dense["up_proj"], dense["down_proj"],
        split_inputs["calibration"]["compressed_trajectory"],
        coefficient_budget=old_coefficient_budget,
        storage_dtype=torch.float16,
        seed=0,
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    candidate_seconds = time.perf_counter() - candidate_started
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        candidate_peak_cuda_bytes = int(torch.cuda.max_memory_allocated(target_device))
        candidate_incremental_peak_cuda_bytes = max(
            0, candidate_peak_cuda_bytes - candidate_baseline_cuda_bytes,
        )
    else:
        candidate_peak_cuda_bytes = None
        candidate_incremental_peak_cuda_bytes = None
        pre_candidate_peak_cuda_bytes = None
        candidate_baseline_cuda_bytes = None
    reconstructed_candidate = reconstruct_joint_swiglu_weights(
        candidate, device=target_device, dtype=torch.float32,
    )

    for split, contexts in split_inputs.items():
        result["candidate"][split] = {}
        for trajectory_name in ("dense_trajectory", "compressed_trajectory"):
            context = contexts[trajectory_name]
            decomposition = mlp_error_decomposition(
                dense["gate_proj"], dense["up_proj"], dense["down_proj"],
                reconstructed_candidate["gate_proj"],
                reconstructed_candidate["up_proj"],
                reconstructed_candidate["down_proj"],
                context,
            )
            direct_output = apply_joint_swiglu(candidate, context)
            dense_same_input = dense["down_proj"] @ (
                torch.nn.functional.silu(dense["gate_proj"] @ context)
                * (dense["up_proj"] @ context)
            )
            saved_nec_same_input = compressed["down_proj"] @ (
                torch.nn.functional.silu(compressed["gate_proj"] @ context)
                * (compressed["up_proj"] @ context)
            )
            reconstructed_output = (
                reconstructed_candidate["down_proj"]
                @ (torch.nn.functional.silu(
                    reconstructed_candidate["gate_proj"] @ context,
                ) * (reconstructed_candidate["up_proj"] @ context))
            )
            result["candidate"][split][trajectory_name] = {
                "mlp_error_decomposition": decomposition,
                "direct_factorized_vs_reconstructed_relative_error": float(
                    torch.linalg.vector_norm(direct_output - reconstructed_output)
                    / torch.linalg.vector_norm(reconstructed_output).clamp_min(1e-30)
                ),
                "dense_mlp_output_same_input": _jsonable_metrics(
                    dense_same_input, direct_output,
                ),
                "saved_nec_mlp_output_same_input": _jsonable_metrics(
                    dense_same_input, saved_nec_same_input,
                ),
            }

    dense_heldout = forward_details(dense, validation_input, **options)
    candidate_heldout = split_inputs["heldout"]["compressed_trajectory"]
    candidate_mlp_output = apply_joint_swiglu(
        candidate, candidate_heldout,
    )
    candidate_block_output = (
        split_inputs["heldout"]["compressed_residual1"] + candidate_mlp_output
    )
    candidate_active_transform = (
        split_inputs["heldout"]["compressed_attention_output"]
        + candidate_mlp_output
    )
    result["candidate"]["block_validation"] = {
        "dense_block_output": _jsonable_metrics(
            dense_heldout["output"], candidate_block_output,
        ),
        "dense_active_transform": _jsonable_metrics(
            dense_heldout["active_transform"], candidate_active_transform,
        ),
        "dense_mlp_output": _jsonable_metrics(
            dense_heldout["mlp_output"], candidate_mlp_output,
        ),
        "candidate_execution": "factorized_apply_joint_swiglu",
    }
    candidate_payload_bytes = shared_swiglu_payload_bytes(candidate)
    candidate_serialized_bytes = shared_swiglu_serialized_bytes(candidate)
    result["candidate"]["budget"] = {
        "old_mlp_numeric_coefficients": old_coefficient_budget,
        "old_mlp_all_tensor_scalars": old_all_tensor_scalars,
        "old_mlp_nonfloating_tensor_scalars": old_nonfloating_scalars,
        "old_mlp_tensor_payload_bytes": old_tensor_payload_bytes,
        "old_mlp_artifact_serialized_bytes": old_serialized_bytes,
        "shared_rank": int(candidate["rank"]),
        "shared_factor_coefficients": int(candidate["coefficient_count"]),
        "candidate_payload_bytes": candidate_payload_bytes,
        "candidate_serialized_bytes": candidate_serialized_bytes,
        "candidate_to_old_artifact_bytes_ratio": (
            candidate_serialized_bytes / max(1, old_serialized_bytes)
        ),
        "candidate_construction_seconds": candidate_seconds,
        "candidate_baseline_cuda_memory_bytes": candidate_baseline_cuda_bytes,
        "candidate_peak_cuda_memory_bytes": candidate_peak_cuda_bytes,
        "candidate_incremental_peak_cuda_memory_bytes": candidate_incremental_peak_cuda_bytes,
    }
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
        result["overall_peak_cuda_memory_bytes"] = max(
            pre_candidate_peak_cuda_bytes,
            int(torch.cuda.max_memory_allocated(target_device)),
        )
    else:
        result["overall_peak_cuda_memory_bytes"] = None
    result["runtime_seconds"] = float(time.perf_counter() - started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--rms-norm-eps", type=float)
    args = parser.parse_args()
    result = run_diagnostic(
        checkpoint_dir=args.checkpoint_dir,
        dense_weights_dir=args.dense_weights_dir,
        layer=args.layer,
        device=args.device,
        seq_len=args.seq_len,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        rms_norm_eps=args.rms_norm_eps,
        output=args.output,
    )
    print(json.dumps({
        "output": str(args.output),
        "candidate_rank": result["candidate"]["budget"]["shared_rank"],
        "candidate_bytes": result["candidate"]["budget"]["candidate_serialized_bytes"],
        "runtime_seconds": result["runtime_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
