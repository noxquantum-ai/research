#!/usr/bin/env python3
"""Sequentially compile a quality-first native-coordinate SwiGLU prefix.

Attention and normalization remain dense.  Each MLP is constructed on the
actual activation trajectory produced by previously compiled native SwiGLU
modules.  Neuron selection and the closed-form Down solve use calibration
inputs only; the disjoint held-out stream is evaluated after each artifact is
frozen.  No gradients, dense residuals, or post-compression repair are used.
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

from nox.native_swiglu import (  # noqa: E402
    apply_native_swiglu,
    build_native_swiglu_artifact,
    fit_anchored_down_readout,
    native_swiglu_features,
    native_swiglu_payload_bytes,
    native_swiglu_serialized_bytes,
)
from nox.grouped_swiglu_tn import (  # noqa: E402
    apply_grouped_swiglu,
    build_grouped_swiglu_artifact,
    grouped_swiglu_payload_bytes,
    grouped_swiglu_serialized_bytes,
    replace_grouped_down_readout,
)
from nox.utils import compute_metrics  # noqa: E402
from scripts.diagnose_cross_layer_interactions import forward_details  # noqa: E402


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
RIDGE_RATIOS = (1e-6, 1e-4, 1e-2, 1.0)


def _metrics(reference, estimate):
    return {key: float(value) for key, value in compute_metrics(reference, estimate).items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layer_weights(layer, device):
    result = {}
    for name in PROJECTIONS:
        module = getattr(layer.self_attn if name in PROJECTIONS[:4] else layer.mlp, name)
        result[name] = module.weight.detach().to(device=device, dtype=torch.float32)
    result["input_layernorm"] = layer.input_layernorm.weight.detach().to(
        device=device, dtype=torch.float32,
    )
    result["post_attention_layernorm"] = layer.post_attention_layernorm.weight.detach().to(
        device=device, dtype=torch.float32,
    )
    return result


def _save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--retain-fraction", type=float, default=0.625)
    parser.add_argument(
        "--representation", choices=("native", "grouped_tn"), default="native",
    )
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument(
        "--rank-profile", type=int, nargs=3, metavar=("GATE", "UP", "DOWN"),
        help="explicit per-operator grouped-TN ranks; mutually exclusive with adaptive candidates",
    )
    parser.add_argument("--retain-candidates", type=float, nargs="+")
    parser.add_argument("--selection-error-max", type=float, default=0.15)
    parser.add_argument(
        "--allocation-metric", choices=("module", "prefix"), default="module",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-5)
    parser.add_argument("--max-runtime-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-model prefix compilation requires CUDA")
    if not 1 <= args.layers <= 32:
        raise ValueError("layers must be in [1, 32]")
    if args.rank_profile is not None:
        if args.representation != "grouped_tn":
            raise ValueError("--rank-profile requires --representation grouped_tn")
        if args.retain_candidates is not None:
            raise ValueError("--rank-profile cannot be combined with adaptive candidates")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    device = torch.device(args.device)
    calibration_path = args.initial_input_dir / "calibration_input_layer_0.pt"
    heldout_path = args.initial_input_dir / "layer_input_validation_layer_0.pt"
    calibration_dense = torch.load(
        calibration_path, map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    heldout_dense = torch.load(
        heldout_path, map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    calibration_candidate = calibration_dense.clone()
    heldout_candidate = heldout_dense.clone()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model.eval()
    options = {
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "seq_len": args.seq_len,
        "rms_norm_eps": args.rms_norm_eps,
    }
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    retain_candidates = (
        sorted(set(float(value) for value in args.retain_candidates))
        if args.retain_candidates else [float(args.retain_fraction)]
    )
    if any(not 0.0 < value <= 1.0 for value in retain_candidates):
        raise ValueError("retain candidates must be in (0, 1]")
    adaptive = args.retain_candidates is not None
    manifest = {
        "format": "nox_quality_first_native_swiglu_prefix_v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layers": args.layers,
        "retain_fraction": args.retain_fraction,
        "representation": args.representation,
        "groups": args.groups if args.representation == "grouped_tn" else None,
        "rank_profile": args.rank_profile,
        "retain_candidates": retain_candidates,
        "adaptive_allocation": adaptive,
        "selection_error_max": args.selection_error_max,
        "allocation_metric": args.allocation_metric,
        "calibration_input_sha256": _sha256(calibration_path),
        "heldout_input_sha256": _sha256(heldout_path),
        "calibration_tokens": int(calibration_dense.shape[1]),
        "heldout_tokens": int(heldout_dense.shape[1]),
        "uses_gradients": False,
        "uses_dense_residual": False,
        "post_compression_optimization": False,
        "layer_records": [],
        "status": "INCOMPLETE",
    }
    manifest_path = args.output_dir / "manifest.json"
    _save_json(manifest_path, manifest)

    for layer_idx in range(args.layers):
        if time.perf_counter() - started > args.max_runtime_seconds:
            raise TimeoutError("prefix compilation runtime cap reached")
        layer_started = time.perf_counter()
        weights = _layer_weights(model.model.layers[layer_idx], device)
        dense_calibration_path = forward_details(
            weights, calibration_dense, **options,
        )
        dense_heldout_path = forward_details(weights, heldout_dense, **options)
        candidate_calibration_path = forward_details(
            weights, calibration_candidate, **options,
        )
        candidate_heldout_path = forward_details(
            weights, heldout_candidate, **options,
        )
        allocation_trials = []
        artifact = selection = readout = calibration_features = None
        for fraction in retain_candidates:
            if args.representation == "native":
                trial_artifact = build_native_swiglu_artifact(
                    weights["gate_proj"], weights["up_proj"], weights["down_proj"],
                    candidate_calibration_path["mlp_input"],
                    retain_fraction=fraction, storage_dtype=torch.float16,
                )
                trial_features = None
                trial_prior = trial_artifact["down_columns"].to(
                    device=device, dtype=torch.float32,
                )
            else:
                grouped_kwargs = (
                    {
                        "rank_gate": args.rank_profile[0],
                        "rank_up": args.rank_profile[1],
                        "rank_down": args.rank_profile[2],
                    }
                    if args.rank_profile is not None else
                    {"coefficient_fraction": fraction}
                )
                trial_artifact, trial_features, _ = build_grouped_swiglu_artifact(
                    weights["gate_proj"], weights["up_proj"], weights["down_proj"],
                    candidate_calibration_path["mlp_input"], groups=args.groups,
                    storage_dtype=torch.float16, **grouped_kwargs,
                )
            effective_fraction = (
                float(trial_artifact["coefficient_fraction_of_dense_mlp"])
                if args.representation == "grouped_tn" else fraction
            )
                trial_prior = trial_artifact["down_left"].permute(1, 0, 2).reshape(
                    weights["down_proj"].shape[0], -1,
                ).to(device=device, dtype=torch.float32)
            if fraction == 1.0 and args.representation == "native":
                trial_selection = {
                    "objective": "exact_full_native_mlp",
                    "ridge_ratio": None,
                    "fit_tokens": int(candidate_calibration_path["mlp_input"].shape[1]),
                    "selection_relative_error": 0.0,
                    "construction_uses_validation": False,
                    "trials": [],
                }
                trial_readout = trial_prior
                module_selection_error = 0.0
            else:
                if trial_features is None:
                    trial_features = native_swiglu_features(
                        trial_artifact, candidate_calibration_path["mlp_input"],
                    )
                trial_readout, trial_selection = fit_anchored_down_readout(
                    trial_features, candidate_calibration_path["mlp_output"],
                    trial_prior,
                    sequence_length=args.seq_len, ridge_ratios=RIDGE_RATIOS,
                )
                module_selection_error = min(
                    item["selection_relative_error"]
                    for item in trial_selection["trials"]
                )
                trial_selection["selection_relative_error"] = module_selection_error
                del trial_features
            if args.representation == "native":
                trial_artifact["down_columns"] = trial_readout.to(
                    device="cpu", dtype=torch.float16,
                )
                trial_calibration_mlp = apply_native_swiglu(
                    trial_artifact, candidate_calibration_path["mlp_input"],
                )
            else:
                replace_grouped_down_readout(trial_artifact, trial_readout)
                trial_calibration_mlp = apply_grouped_swiglu(
                    trial_artifact, candidate_calibration_path["mlp_input"],
                )
            trial_calibration_next = (
                candidate_calibration_path["residual1"] + trial_calibration_mlp
            )
            prefix_selection_error = _metrics(
                dense_calibration_path["output"], trial_calibration_next,
            )["relative_error"]
            selection_error = (
                module_selection_error
                if args.allocation_metric == "module" else prefix_selection_error
            )
            allocation_trials.append({
                "retain_fraction": effective_fraction,
                "retained_neurons": trial_artifact.get("retained_neurons"),
                "groups": trial_artifact.get("groups"),
                "rank": trial_artifact.get("rank"),
                "rank_profile": [
                    trial_artifact.get("rank_gate", trial_artifact.get("rank")),
                    trial_artifact.get("rank_up", trial_artifact.get("rank")),
                    trial_artifact.get("rank_down", trial_artifact.get("rank")),
                ],
                "allocation_score": selection_error,
                "module_selection_relative_error": module_selection_error,
                "prefix_selection_relative_error": prefix_selection_error,
                "ridge_ratio": trial_selection["ridge_ratio"],
                "float_coefficients": int(trial_artifact["coefficient_count"]),
            })
            passes = selection_error <= args.selection_error_max
            if passes or not adaptive:
                artifact = trial_artifact
                selection = trial_selection
                readout = trial_readout
                break
            del (
                trial_artifact, trial_readout, trial_calibration_mlp,
                trial_calibration_next,
            )
            torch.cuda.empty_cache()
        if artifact is None:
            raise RuntimeError(
                "no retain candidate met the calibration-only allocation threshold"
            )
        if args.representation == "native":
            artifact["down_columns"] = readout.to(device="cpu", dtype=torch.float16)
        else:
            replace_grouped_down_readout(artifact, readout)
        artifact["construction_objective"] = (
            f"{'adaptive_' if adaptive else ''}sequential_{args.representation}"
            "_plus_calibration_only_anchored_ridge"
        )
        artifact["ridge_selection"] = selection
        artifact["allocation_trials"] = allocation_trials
        del trial_calibration_mlp, trial_calibration_next
        artifact_path = args.output_dir / f"native_mlp_layer_{layer_idx}.pt"
        payload = {
            "format": "nox_quality_first_native_swiglu_checkpoint_v1",
            "layer_idx": layer_idx,
            "mlp_artifact": artifact,
            "attention_representation": "dense_reference",
            "construction_trajectory": "sequential_compressed_prefix",
            "post_compression_optimization": False,
        }
        torch.save(payload, artifact_path)

        apply_mlp = (
            apply_native_swiglu
            if args.representation == "native" else apply_grouped_swiglu
        )
        calibration_mlp = apply_mlp(artifact, candidate_calibration_path["mlp_input"])
        heldout_mlp = apply_mlp(artifact, candidate_heldout_path["mlp_input"])
        calibration_candidate_next = (
            candidate_calibration_path["residual1"] + calibration_mlp
        )
        heldout_candidate_next = candidate_heldout_path["residual1"] + heldout_mlp
        record = {
            "layer": layer_idx,
            "representation": args.representation,
            "retained_neurons": artifact.get("retained_neurons"),
            "groups": artifact.get("groups"),
            "rank": artifact.get("rank"),
            "float_coefficients": int(artifact["coefficient_count"]),
            "payload_bytes": (
                native_swiglu_payload_bytes(artifact)
                if args.representation == "native" else
                grouped_swiglu_payload_bytes(artifact)
            ),
            "artifact_schema_bytes": (
                native_swiglu_serialized_bytes(artifact)
                if args.representation == "native" else
                grouped_swiglu_serialized_bytes(artifact)
            ),
            "checkpoint_bytes": artifact_path.stat().st_size,
            "checkpoint_sha256": _sha256(artifact_path),
            "ridge_selection": selection,
            "allocation_trials": allocation_trials,
            "heldout_local_mlp_on_candidate_trajectory": _metrics(
                candidate_heldout_path["mlp_output"], heldout_mlp,
            ),
            "heldout_prefix_output": _metrics(
                dense_heldout_path["output"], heldout_candidate_next,
            ),
            "heldout_candidate_input_vs_dense_input": _metrics(
                heldout_dense, heldout_candidate,
            ),
            "runtime_seconds": float(time.perf_counter() - layer_started),
        }
        manifest["layer_records"].append(record)
        calibration_dense = dense_calibration_path["output"].detach()
        heldout_dense = dense_heldout_path["output"].detach()
        calibration_candidate = calibration_candidate_next.detach()
        heldout_candidate = heldout_candidate_next.detach()
        _save_json(manifest_path, manifest)
        print(json.dumps(record, sort_keys=True), flush=True)
        del (
            weights, dense_calibration_path, dense_heldout_path,
            candidate_calibration_path, candidate_heldout_path, artifact,
            readout, calibration_mlp, heldout_mlp,
            calibration_candidate_next, heldout_candidate_next,
        )
        torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    manifest["status"] = "COMPLETE"
    manifest["runtime_seconds"] = float(time.perf_counter() - started)
    manifest["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    manifest["total_checkpoint_bytes"] = sum(
        record["checkpoint_bytes"] for record in manifest["layer_records"]
    )
    _save_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "runtime_seconds": manifest["runtime_seconds"],
        "peak_cuda_memory_bytes": manifest["peak_cuda_memory_bytes"],
        "total_checkpoint_bytes": manifest["total_checkpoint_bytes"],
        "manifest": str(manifest_path),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
