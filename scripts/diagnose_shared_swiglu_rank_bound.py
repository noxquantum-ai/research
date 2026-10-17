#!/usr/bin/env python3
"""Rank-floor and Down-crossover diagnostic for joint SwiGLU NEC.

The script consumes the existing layer checkpoint, pinned dense projection
weights, and held-out layer inputs. It computes the two transformer
trajectories with the existing block forward helper, evaluates the dense MLP
output spectral tail, and tests one shared-neuron candidate with dense-oracle
and saved-NEC Down operators. It performs no training, compression
compilation, model recovery, or post-compression optimization. Forward passes,
candidate construction, and the output SVD can run on CPU or CUDA.

The previous GPU run did not save the candidate factor tensors. Therefore,
unless --candidate-artifact is supplied, the script deterministically rebuilds
the same shared-neuron construction from the saved calibration input. Its
randomized SVD need not be bitwise identical to the previous GPU realization;
the JSON records this distinction explicitly.
"""

from __future__ import annotations

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
    build_joint_swiglu_artifact,
    reconstruct_joint_swiglu_weights,
)
from nox.swiglu_rank_bound import (  # noqa: E402
    evaluate_down_crossovers,
    output_rank_curve,
    swiglu_output,
)
from scripts.diagnose_cross_layer_interactions import forward_details  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_RANKS = (0, 64, 128, 256, 322, 426, 512, 768, 1024, 1536, 2048, 3072, 4096)


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


def _artifact_sha256(artifact: dict) -> str:
    digest = hashlib.sha256()
    for name in ("neuron_basis", "gate_factor", "up_factor", "down_factor"):
        value = artifact[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _serialized_bytes(value) -> int:
    import io

    stream = io.BytesIO()
    torch.save(value, stream)
    return int(stream.tell())


@torch.no_grad()
def run_diagnostic(
    *,
    checkpoint_dir: Path,
    dense_weights_dir: Path,
    output: Path,
    layer: int = 0,
    candidate_artifact_path: Path | None = None,
    seq_len: int = 128,
    n_heads: int = 32,
    n_kv_heads: int = 32,
    rms_norm_eps: float | None = None,
    ranks: tuple[int, ...] = DEFAULT_RANKS,
    cpu_threads: int = 8,
    device: str = "cpu",
) -> dict:
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if target_device.type == "cpu":
        torch.set_num_threads(cpu_threads)
    elif target_device.type != "cuda":
        raise ValueError("device must be cpu or cuda[:index]")

    checkpoint_path = checkpoint_dir / f"compressed_layer_{layer}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    calibration_path = checkpoint_dir / f"calibration_input_layer_{layer}.pt"
    heldout_path = checkpoint_dir / f"layer_input_validation_layer_{layer}.pt"
    calibration_input = torch.load(
        calibration_path, map_location="cpu", weights_only=True,
    ).to(device=target_device, dtype=torch.float32)
    heldout_input = torch.load(
        heldout_path, map_location="cpu", weights_only=True,
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
    baseline_allocated_bytes = (
        torch.cuda.memory_allocated(target_device) if target_device.type == "cuda" else 0
    )
    started = time.perf_counter()
    contexts: dict[str, dict[str, torch.Tensor]] = {}
    for split, raw_input in (("calibration", calibration_input), ("heldout", heldout_input)):
        dense_path = forward_details(dense, raw_input, **options)
        compressed_path = forward_details(compressed, raw_input, **options)
        contexts[split] = {
            "dense_trajectory": dense_path["mlp_input"].detach(),
            "compressed_trajectory": compressed_path["mlp_input"].detach(),
        }
        del dense_path, compressed_path

    d_model, intermediate_size = dense["down_proj"].shape
    requested_ranks = sorted(set(int(value) for value in ranks))
    if any(rank < 0 for rank in requested_ranks):
        raise ValueError("ranks must be nonnegative")
    max_rank = min(d_model, int(heldout_input.shape[1]))
    requested_ranks = [min(rank, max_rank) for rank in requested_ranks]
    requested_ranks = sorted(set(requested_ranks))

    old_budget = sum(
        _float_coefficients(checkpoint["compressed_artifacts"][name])
        for name in MLP_PROJECTIONS
    )
    if candidate_artifact_path is None:
        candidate = build_joint_swiglu_artifact(
            dense["gate_proj"], dense["up_proj"], dense["down_proj"],
            contexts["calibration"]["compressed_trajectory"],
            coefficient_budget=old_budget,
            storage_dtype=torch.float16,
            seed=0,
        )
        candidate_origin = f"rebuilt_on_{target_device}_from_saved_calibration_and_seed_0"
    else:
        candidate = torch.load(
            candidate_artifact_path, map_location=target_device, weights_only=True,
        )
        candidate_origin = "loaded_exact_saved_candidate_artifact"
    candidate_weights = reconstruct_joint_swiglu_weights(
        candidate, device=target_device, dtype=torch.float32,
    )

    result = {
        "diagnostic": "shared_swiglu_rank_floor_and_down_crossovers",
        "layer": int(layer),
        "device": str(target_device),
        "cpu_threads": int(cpu_threads),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dense_weight_sha256": {
            name: _sha256(dense_weights_dir / f"layer_{layer}_{name}.pt")
            for name in PROJECTIONS
        },
        "calibration_input_sha256": _sha256(calibration_path),
        "heldout_input_sha256": _sha256(heldout_path),
        "calibration_tokens": int(calibration_input.shape[1]),
        "heldout_tokens": int(heldout_input.shape[1]),
        "fixed_inputs": "dense and saved-NEC transformer trajectories measured separately",
        "post_compression_optimization": False,
        "candidate_origin": candidate_origin,
        "candidate_sha256": _artifact_sha256(candidate),
        "candidate_rank": int(candidate["rank"]),
        "candidate_down_matrix_rank_upper_bound": int(candidate["rank"]),
        "candidate_float_coefficients": int(candidate["coefficient_count"]),
        "saved_nec_mlp_float_coefficient_budget": int(old_budget),
        "candidate_serialized_bytes": _serialized_bytes(candidate),
        "heldout_rank_floor": {},
        "down_crossovers": {},
    }

    for trajectory_name, context in contexts["heldout"].items():
        dense_y = swiglu_output(
            dense["gate_proj"], dense["up_proj"], dense["down_proj"], context,
        )
        curve = output_rank_curve(dense_y, requested_ranks, device=target_device)
        curve["rank_426_error"] = curve["relative_tail_error_by_rank"].get(str(min(426, max_rank)))
        result["heldout_rank_floor"][trajectory_name] = curve

    for split, trajectory_contexts in contexts.items():
        result["down_crossovers"][split] = {}
        for trajectory_name, context in trajectory_contexts.items():
            result["down_crossovers"][split][trajectory_name] = evaluate_down_crossovers(
                dense_gate=dense["gate_proj"],
                dense_up=dense["up_proj"],
                dense_down=dense["down_proj"],
                candidate_gate=candidate_weights["gate_proj"],
                candidate_up=candidate_weights["up_proj"],
                candidate_down=candidate_weights["down_proj"],
                saved_nec_gate=compressed["gate_proj"],
                saved_nec_up=compressed["up_proj"],
                saved_nec_down=compressed["down_proj"],
                inputs=context,
            )

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    result["runtime_seconds"] = float(time.perf_counter() - started)
    result["baseline_cuda_memory_bytes"] = int(baseline_allocated_bytes)
    result["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(target_device))
        if target_device.type == "cuda" else 0
    )
    result["incremental_peak_cuda_memory_bytes"] = max(
        0, result["peak_cuda_memory_bytes"] - result["baseline_cuda_memory_bytes"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--rms-norm-eps", type=float)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    args = parser.parse_args()
    result = run_diagnostic(
        checkpoint_dir=args.checkpoint_dir,
        dense_weights_dir=args.dense_weights_dir,
        output=args.output,
        candidate_artifact_path=args.candidate_artifact,
        layer=args.layer,
        seq_len=args.seq_len,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        rms_norm_eps=args.rms_norm_eps,
        ranks=tuple(args.ranks),
        cpu_threads=args.cpu_threads,
        device=args.device,
    )
    brief = {
        "output": str(args.output),
        "candidate_rank": result["candidate_rank"],
        "candidate_origin": result["candidate_origin"],
        "rank_426_tail_errors": {
            name: info["rank_426_error"]
            for name, info in result["heldout_rank_floor"].items()
        },
        "runtime_seconds": result["runtime_seconds"],
    }
    print(json.dumps(brief, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
