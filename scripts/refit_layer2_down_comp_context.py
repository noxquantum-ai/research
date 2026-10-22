#!/usr/bin/env python3
"""Refit only layer-2 NEC down_proj on saved compressed SwiGLU products.

This is a zero-gradient, one-projection diagnostic. It reads but never changes
the original four-layer checkpoint. Calibration and held-out MLP inputs are
reconstructed from the saved dense block inputs and pinned dense weights, so
the baseline compiler's layer-local context and the held-out split are exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d
from nox.compiler import nox_hybrid_dmrg_decompose, reconstruct_from_artifact
from nox.evaluation import MODEL_ID, MODEL_REVISION, validate_mpo_artifact
from nox.utils import compute_metrics, get_modes_for_projection
from scripts.compile_high_fidelity import PinnedWeights


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_save(value: dict, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def atomic_torch_save(value: object, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    with temp.open("wb") as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def dense_post_attention_mlp_input(
    weights: dict[str, torch.Tensor], x: torch.Tensor, *, seq_len: int,
    n_heads: int, n_kv_heads: int, rms_norm_eps: float,
) -> torch.Tensor:
    """Return the dense post-attention RMSNorm output, with windows kept separate."""
    if x.ndim != 2 or x.shape[0] != weights["q_proj"].shape[1]:
        raise ValueError(f"Invalid block-input shape: {tuple(x.shape)}")
    if x.shape[1] % seq_len:
        raise ValueError("Input columns must contain complete calibration windows")
    norm1 = RMSNorm(x.shape[0], eps=rms_norm_eps, weight=weights["input_layernorm"])
    norm2 = RMSNorm(x.shape[0], eps=rms_norm_eps, weight=weights["post_attention_layernorm"])
    normalized = norm1(x)
    q = weights["q_proj"] @ normalized
    k = weights["k_proj"] @ normalized
    v = weights["v_proj"] @ normalized
    context = scaled_dot_product_attention_2d(
        q, k, v, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len,
    )
    return norm2(x + weights["o_proj"] @ context)


def swiglu_products(
    weights: dict[str, torch.Tensor],
    compressed: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_dense = weights["gate_proj"] @ x
    up_dense = weights["up_proj"] @ x
    product_dense = F.silu(gate_dense) * up_dense
    gate_comp = compressed["gate_proj"] @ x
    up_comp = compressed["up_proj"] @ x
    product_comp = F.silu(gate_comp) * up_comp
    return product_dense, product_comp


def validate_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name}: contains non-finite values")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=2)
    args = parser.parse_args()
    if args.layer != 2:
        raise ValueError("This controlled diagnostic is pinned to layer 2")
    if not torch.cuda.is_available():
        raise RuntimeError("The real-model refit requires the A40 CUDA runtime")
    if args.output.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite output: {args.output}")

    started = time.monotonic()
    args.output.mkdir(parents=True)
    log(f"START layer={args.layer} output={args.output}")
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(20261022)
    torch.cuda.reset_peak_memory_stats(device)

    manifest_path = args.checkpoint_dir / "model_manifest.json"
    checkpoint_path = args.checkpoint_dir / f"compressed_layer_{args.layer}.pt"
    calibration_path = args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt"
    validation_path = args.checkpoint_dir / f"layer_input_validation_layer_{args.layer}.pt"
    for path in (manifest_path, checkpoint_path, calibration_path, validation_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text())
    checkpoint_hash_before = sha256_file(checkpoint_path)
    manifest_hash = manifest.get("checkpoint_file_sha256", {}).get(checkpoint_path.name)
    if manifest_hash != checkpoint_hash_before:
        raise ValueError("Layer-2 checkpoint hash does not match its manifest")
    if manifest.get("layers_compressed") != [0, 1, 2, 3]:
        raise ValueError("Expected the immutable four-layer prefix checkpoint")
    identity = manifest["run_identity"]
    expected_settings = {
        "calibration_activation_mode": "dense",
        "construction_mode": "independent",
        "mlp_objective": "independent",
        "mpo_n_sweeps": 2,
        "mpo_quantization_schedule": "post_sweep",
        "mpo_cg_iterations": 48,
        "calibration_backend": "full_matrix",
    }
    for key, expected in expected_settings.items():
        actual = identity.get(key)
        if actual != expected:
            raise ValueError(f"Unexpected source setting {key}={actual!r}; expected {expected!r}")
    if identity["model_id"] != MODEL_ID or identity["model_revision"] != MODEL_REVISION:
        raise ValueError("Checkpoint model or revision is not the pinned Llama-2-7B snapshot")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("layer_idx") != args.layer:
        raise ValueError("Checkpoint layer index mismatch")
    if checkpoint.get("run_fingerprint") != manifest.get("run_fingerprint"):
        raise ValueError("Checkpoint and manifest run fingerprints differ")
    old_artifacts = checkpoint["compressed_artifacts"]
    if not {"gate_proj", "up_proj", "down_proj"}.issubset(old_artifacts):
        raise ValueError("Layer checkpoint lacks required MLP projection artifacts")

    x_train_cpu = torch.load(calibration_path, map_location="cpu", weights_only=True)
    x_val_cpu = torch.load(validation_path, map_location="cpu", weights_only=True)
    if x_train_cpu.ndim != 2 or x_val_cpu.ndim != 2:
        raise ValueError("Saved block inputs must be rank-two [hidden,tokens] tensors")
    if x_train_cpu.shape != (4096, 4096) or x_val_cpu.shape != (4096, 1024):
        raise ValueError(
            f"Unexpected saved train/validation shapes: {tuple(x_train_cpu.shape)}, "
            f"{tuple(x_val_cpu.shape)}"
        )
    if x_train_cpu.shape[1] % 128 or x_val_cpu.shape[1] % 128:
        raise ValueError("Saved activations must contain complete 128-token windows")

    log("LOAD pinned layer weights and decode saved Gate/Up/Down artifacts")
    weights = PinnedWeights(MODEL_ID, MODEL_REVISION).layer(args.layer, device)
    weights = {key: value.to(device=device, dtype=torch.float32) for key, value in weights.items()}
    compressed = {
        name: reconstruct_from_artifact(old_artifacts[name], device=device, dtype=torch.float32)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    expected_shapes = {
        "gate_proj": (11008, 4096), "up_proj": (11008, 4096), "down_proj": (4096, 11008),
    }
    for name, shape in expected_shapes.items():
        validate_tensor(f"dense {name}", weights[name], shape)
        validate_tensor(f"compressed {name}", compressed[name], shape)

    options = {
        "seq_len": 128,
        "n_heads": int(identity["forward_options"]["n_heads"]),
        "n_kv_heads": int(identity["forward_options"]["n_kv_heads"]),
        "rms_norm_eps": float(identity["forward_options"]["rms_norm_eps"]),
    }
    log("BUILD dense post-attention MLP contexts from saved disjoint windows")
    x_train = x_train_cpu.to(device=device, dtype=torch.float32)
    x_val = x_val_cpu.to(device=device, dtype=torch.float32)
    m_train = dense_post_attention_mlp_input(weights, x_train, **options)
    m_val = dense_post_attention_mlp_input(weights, x_val, **options)
    del x_train, x_val
    p_dense_train, p_comp_train = swiglu_products(weights, compressed, m_train)
    p_dense_val, p_comp_val = swiglu_products(weights, compressed, m_val)
    y_dense_train = weights["down_proj"] @ p_dense_train
    y_dense_val = weights["down_proj"] @ p_dense_val

    fit_context = {
        "p_dense_train": p_dense_train,
        "p_comp_train": p_comp_train,
        "y_dense_train": y_dense_train,
        "m_val": m_val,
        "p_dense_val": p_dense_val,
        "p_comp_val": p_comp_val,
        "y_dense_val": y_dense_val,
    }
    for key, tensor in fit_context.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values in {key}")

    original_down_report = validate_mpo_artifact(
        "layer2.down_proj", old_artifacts["down_proj"], (4096, 11008), [1, 64, 128, 64, 1],
    )
    budget = identity["projection_budgets"]["down_proj"]
    if budget != {"r_bridge": 128, "n_outliers": 524288}:
        raise ValueError(f"Unexpected layer-2 Down budget: {budget}")
    old_hyper = old_artifacts["down_proj"]["compiler_hyperparameters"]
    if old_hyper["calibration_sample_count"] != p_comp_train.shape[1]:
        raise ValueError("Saved Down artifact and training activation token counts differ")

    dense_down_on_comp_train = weights["down_proj"] @ p_comp_train
    saved_down_on_dense_train = compressed["down_proj"] @ p_dense_train
    saved_down_on_comp_train = compressed["down_proj"] @ p_comp_train
    dense_down_on_comp_val = weights["down_proj"] @ p_comp_val
    saved_down_on_dense_val = compressed["down_proj"] @ p_dense_val
    saved_down_on_comp_val = compressed["down_proj"] @ p_comp_val
    before_metrics = {
        "train": {
            "gate_up_product_relative_error": compute_metrics(p_dense_train, p_comp_train),
            "dense_down_applied_to_compressed_product": compute_metrics(y_dense_train, dense_down_on_comp_train),
            "saved_down_on_dense_product": compute_metrics(y_dense_train, saved_down_on_dense_train),
            "saved_down_on_compressed_product": compute_metrics(y_dense_train, saved_down_on_comp_train),
        },
        "heldout": {
            "gate_up_product_relative_error": compute_metrics(p_dense_val, p_comp_val),
            "dense_down_applied_to_compressed_product": compute_metrics(y_dense_val, dense_down_on_comp_val),
            "saved_down_on_dense_product": compute_metrics(y_dense_val, saved_down_on_dense_val),
            "saved_down_on_compressed_product": compute_metrics(y_dense_val, saved_down_on_comp_val),
        },
    }
    log(
        "BASELINE "
        f"heldout product={before_metrics['heldout']['gate_up_product_relative_error']['relative_error']:.6f} "
        f"saved_down_on_compressed={before_metrics['heldout']['saved_down_on_compressed_product']['relative_error']:.6f}"
    )

    ranks = tuple(int(value) for value in identity["config"]["ranks"])
    o_modes, i_modes = get_modes_for_projection(4096, 11008)
    if list(ranks) != [1, 64, 128, 64, 1]:
        raise ValueError(f"Unexpected High-Fidelity ranks: {ranks}")
    log("COMPILE only layer2.down_proj on compressed Gate/Up SwiGLU products")
    compile_started = time.monotonic()
    W_fit, total_kb, _, chosen_ranks, refit_artifact = nox_hybrid_dmrg_decompose(
        W=weights["down_proj"],
        X_calib=p_comp_train,
        Y_dense=y_dense_train,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=ranks,
        r_bridge=int(budget["r_bridge"]),
        n_outliers=int(budget["n_outliers"]),
        n_sweeps=int(old_hyper["n_sweeps"]),
        prior_gamma=float(old_hyper["prior_gamma"]),
        alpha_awt=float(old_hyper["alpha_awt"]),
        chunk_size=int(old_hyper["chunk_size"]),
        return_artifacts=True,
        verbose=True,
        quantize_components=bool(old_hyper["quantize_components"]),
        diagonal_prior_strength=float(old_hyper["diagonal_prior_strength"]),
        site_order=tuple(checkpoint["mpo_site_order"]),
        quantization_schedule=str(old_hyper["quantization_schedule"]),
        bridge_quantization_mode=str(old_hyper["bridge_quantization_mode"]),
        bridge_refit_iterations=int(old_hyper["bridge_refit_iterations"]),
        cg_threshold=int(old_hyper["cg_threshold"]),
        cg_iterations=int(old_hyper["cg_iterations"]),
    )
    compile_seconds = time.monotonic() - compile_started
    if list(chosen_ranks) != [1, 64, 128, 64, 1]:
        raise RuntimeError(f"Fitted Down ranks changed: {chosen_ranks}")
    new_down_report = validate_mpo_artifact(
        "refit.layer2.down_proj", refit_artifact, (4096, 11008), [1, 64, 128, 64, 1],
    )

    artifact_path = args.output / "layer2_down_refit.pt"
    output_payload = {
        "layer_idx": args.layer,
        "projection": "down_proj",
        "parent_run_fingerprint": checkpoint["run_fingerprint"],
        "objective": "fit compressed SwiGLU inputs to dense Down(SiLU(Gx) * Ux) targets",
        "artifact": refit_artifact,
    }
    atomic_torch_save(output_payload, artifact_path)
    disk_payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    W_new = reconstruct_from_artifact(disk_payload["artifact"], device=device, dtype=torch.float32)
    if tuple(W_new.shape) != (4096, 11008) or not torch.isfinite(W_new).all():
        raise FloatingPointError("Saved refit Down artifact failed reconstruction checks")
    serialized_matrix_relative_difference = float(
        (W_new - W_fit).norm() / W_fit.norm().clamp_min(1e-30)
    )
    if not math.isfinite(serialized_matrix_relative_difference) or serialized_matrix_relative_difference > 1e-2:
        raise RuntimeError("Serialized refit artifact disagrees with compiler output")

    new_down_on_dense_val = W_new @ p_dense_val
    new_down_on_comp_train = W_new @ p_comp_train
    new_down_on_comp_val = W_new @ p_comp_val
    after_metrics = {
        "train": {
            "refit_down_on_dense_product": compute_metrics(y_dense_train, W_new @ p_dense_train),
            "refit_down_on_compressed_product": compute_metrics(y_dense_train, new_down_on_comp_train),
            "saved_down_on_compressed_product": before_metrics["train"]["saved_down_on_compressed_product"],
        },
        "heldout": {
            "refit_down_on_dense_product": compute_metrics(y_dense_val, new_down_on_dense_val),
            "refit_down_on_compressed_product": compute_metrics(y_dense_val, new_down_on_comp_val),
            "saved_down_on_compressed_product": before_metrics["heldout"]["saved_down_on_compressed_product"],
            "dense_down_on_compressed_product": before_metrics["heldout"]["dense_down_applied_to_compressed_product"],
        },
    }

    if sha256_file(checkpoint_path) != checkpoint_hash_before:
        raise RuntimeError("Original layer-2 checkpoint changed during diagnostic")
    report = {
        "status": "COMPLETE",
        "scope": "single layer-2 Down refit; no other projections or checkpoints changed",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_commit": source_commit(),
        "script_sha256": sha256_file(Path(__file__)),
        "parent_run_fingerprint": checkpoint["run_fingerprint"],
        "parent_checkpoint_sha256": checkpoint_hash_before,
        "parent_manifest_sha256": sha256_file(manifest_path),
        "train_activation_sha256": sha256_file(calibration_path),
        "heldout_activation_sha256": sha256_file(validation_path),
        "train_tokens": int(p_comp_train.shape[1]),
        "heldout_tokens": int(p_comp_val.shape[1]),
        "validation_is_disjoint_by_manifest": True,
        "objective": "min ||D_hat p_hat - D p_dense||_F^2 on training tokens; p_hat=SiLU(G_hat x)*U_hat x",
        "settings": {
            "layer": args.layer,
            "mode": "same dense-context input x for dense teacher and saved compressed Gate/Up; no prefix trajectory compensation",
            "ranks": list(ranks),
            "projection_budget": budget,
            "sweeps": int(old_hyper["n_sweeps"]),
            "prior_gamma": float(old_hyper["prior_gamma"]),
            "diagonal_prior_strength": float(old_hyper["diagonal_prior_strength"]),
            "alpha_awt": float(old_hyper["alpha_awt"]),
            "quantize_components": bool(old_hyper["quantize_components"]),
            "quantization_schedule": str(old_hyper["quantization_schedule"]),
            "bridge_quantization_mode": str(old_hyper["bridge_quantization_mode"]),
            "cg_iterations": int(old_hyper["cg_iterations"]),
            "sequence_length": options["seq_len"],
            "rms_norm_eps": options["rms_norm_eps"],
            "gradient_updates": 0,
        },
        "coefficient_audit": {
            "original_down": original_down_report,
            "refit_down": new_down_report,
            "same_coefficient_count": (
                original_down_report["independent_numerical_coefficients"]
                == new_down_report["independent_numerical_coefficients"]
            ),
            "count_delta": (
                new_down_report["independent_numerical_coefficients"]
                - original_down_report["independent_numerical_coefficients"]
            ),
            "original_artifact_bytes": int(checkpoint_path.stat().st_size),
            "refit_projection_artifact_bytes": int(artifact_path.stat().st_size),
            "compiler_reported_payload_kb": float(total_kb),
        },
        "baseline_metrics": before_metrics,
        "refit_metrics": after_metrics,
        "serialized_matrix_relative_difference": serialized_matrix_relative_difference,
        "compile_seconds": compile_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(device),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "refit_artifact_sha256": sha256_file(artifact_path),
    }
    atomic_json_save(report, args.output / "report.json")
    log(
        "RESULT "
        f"heldout_saved={after_metrics['heldout']['saved_down_on_compressed_product']['relative_error']:.6f} "
        f"heldout_refit={after_metrics['heldout']['refit_down_on_compressed_product']['relative_error']:.6f} "
        f"coeff_delta={report['coefficient_audit']['count_delta']}"
    )
    log(f"COMPLETE compile_seconds={compile_seconds:.1f} output={args.output}")


if __name__ == "__main__":
    main()
