#!/usr/bin/env python3
"""Strict matched WikiText-2 evaluation of dense or High-Fidelity NEC Llama-2-7B."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import (
    compute_perplexity,
    inject_nox_checkpoints,
    load_wikitext2_text,
)
from nox.artifacts import fingerprint
from nox.evaluation import (
    DENSE_LLAMA2_7B_COEFFICIENTS,
    MODEL_ID,
    MODEL_REVISION,
    validate_high_fidelity_checkpoint,
    validate_llama2_model_tokenizer,
)


def source_commit() -> str:
    override = os.environ.get("NEC_SOURCE_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


MODEL_VARIANTS = ("dense", "high_fidelity_nec")


def save_top1_predictions(predictions: torch.Tensor, path: Path) -> dict:
    """Persist exact ordered predictions in a compact, hash-audited sidecar."""
    values = predictions.detach().cpu().numpy().astype("<i4", copy=False)
    payload = values.tobytes(order="C")
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"Existing prediction sidecar differs; refusing overwrite: {path}")
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(existing).hexdigest(),
            "dtype": "int32_little_endian",
            "count": int(values.size),
            "bytes": len(existing),
        }
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dtype": "int32_little_endian",
        "count": int(values.size),
        "bytes": len(payload),
    }


def save_token_nll(values: torch.Tensor, path: Path) -> dict:
    """Write ordered per-target NLLs for paired dense/NEC comparison."""
    array = values.detach().cpu().numpy().astype("<f4", copy=False)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    payload = buffer.getvalue()
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"Existing token-NLL sidecar differs; refusing overwrite: {path}")
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(existing).hexdigest(),
            "dtype": "float32_little_endian_npy",
            "count": int(array.size),
            "bytes": len(existing),
        }
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dtype": "float32_little_endian_npy",
        "count": int(array.size),
        "bytes": int(path.stat().st_size),
    }


def load_model(
    checkpoint_dir: Path | None,
    offload_folder: Path | None = None,
    *,
    variant: str = "high_fidelity_nec",
    allow_partial_prefix: bool = False,
):
    """Load the identical pinned base model, optionally replacing all linear weights with NEC."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    if variant == "high_fidelity_nec" and checkpoint_dir is None:
        raise ValueError("High-Fidelity NEC evaluation requires --checkpoint-dir")
    if variant == "dense" and checkpoint_dir is not None:
        raise ValueError("Dense evaluation must not receive --checkpoint-dir")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("Production NEC evaluation requires CUDA")
    # Keep the conservative historical T4 defaults; the RunPod runner derives
    # larger caps from free VRAM/RAM and sets them in this process environment.
    # Placement changes only; model dtype, logits, and token scoring do not.
    max_memory = evaluation_memory_budget(gpu_count)
    offload = offload_folder or (
        (checkpoint_dir / "evaluation_offload")
        if checkpoint_dir is not None else (ROOT / ".evaluation_offload")
    )
    offload.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload),
        attn_implementation="sdpa",
    ).eval()
    model.config.use_cache = False
    compatibility = validate_llama2_model_tokenizer(model, tokenizer)
    reconstruction_diagnostics: list[dict] = []
    injected: list[int] = []
    if variant == "high_fidelity_nec":
        checkpoint_manifest = json.loads(
            (checkpoint_dir / "model_manifest.json").read_text()
        )
        expected_layers = checkpoint_manifest.get("run_identity", {}).get("layers")
        if not isinstance(expected_layers, list) or not expected_layers:
            raise ValueError("NEC checkpoint manifest does not declare compressed layers")
        if expected_layers != list(range(len(expected_layers))):
            raise ValueError("NEC evaluation accepts only a contiguous compressed layer prefix")
        if len(expected_layers) != 32 and not allow_partial_prefix:
            raise ValueError("Partial NEC prefix requires --allow-partial-prefix")
        injected = inject_nox_checkpoints(
            model, str(checkpoint_dir), dtype=torch.float16,
            diagnostics=reconstruction_diagnostics,
            expected_layers=expected_layers,
        )
        if injected != expected_layers:
            raise RuntimeError(f"Incomplete NEC injection: {injected}, expected {expected_layers}")
    return model, tokenizer, sorted(injected), compatibility, reconstruction_diagnostics


def evaluation_memory_budget(gpu_count: int) -> dict[int | str, str]:
    """Return Accelerate placement caps; defaults preserve historical Kaggle limits."""
    if gpu_count < 1:
        raise ValueError("Evaluation requires at least one CUDA GPU")
    gpu_limit = os.environ.get("NEC_EVAL_GPU_MEMORY_GIB", "10")
    cpu_limit = os.environ.get("NEC_EVAL_CPU_MEMORY_GIB", "24")
    if not gpu_limit.isdigit() or int(gpu_limit) < 1:
        raise ValueError("NEC_EVAL_GPU_MEMORY_GIB must be a positive integer")
    if not cpu_limit.isdigit() or int(cpu_limit) < 1:
        raise ValueError("NEC_EVAL_CPU_MEMORY_GIB must be a positive integer")
    return {
        **{index: f"{gpu_limit}GiB" for index in range(gpu_count)},
        "cpu": f"{cpu_limit}GiB",
    }


def reset_cuda_peak_memory_stats() -> None:
    """Reset allocator peaks per device using the current-device API.

    The RunPod PyTorch 2.11 build rejects every explicit device argument to
    ``reset_peak_memory_stats`` (including ``torch.device('cuda:0')``), while
    the no-argument form works for the current device.
    """
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.reset_peak_memory_stats()


def cuda_peak_memory_allocated() -> dict[str, int]:
    """Read allocator peaks without relying on backend-specific device arguments."""
    result = {}
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            result[str(index)] = int(torch.cuda.max_memory_allocated())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of equal-length sliding windows per forward pass")
    parser.add_argument("--max-eval-tokens", type=int, default=10_000,
                        help="Zero scores the complete test split")
    parser.add_argument("--offload-folder", type=Path, default=None,
                        help="Writable folder for Accelerate disk offload")
    parser.add_argument("--full-test", action="store_true",
                        help="Report both the required 10k-token result and full raw test split")
    parser.add_argument("--dataset-split", choices=("validation", "test"), default="test",
                        help="Use validation for prefix development runs; reserve test for final evaluation")
    parser.add_argument("--allow-partial-prefix", action="store_true",
                        help="Allow a complete contiguous compressed layer prefix, leaving later layers dense")
    parser.add_argument("--expected-quantization-schedule", choices=("in_sweep", "post_sweep"),
                        default="in_sweep")
    parser.add_argument("--expected-cg-threshold", type=int, default=4096)
    parser.add_argument("--expected-cg-iterations", type=int, default=6)
    parser.add_argument("--skip-reconstruct-audit", action="store_true",
                        help=("Skip the redundant dense reconstruction pass in the metadata audit; "
                              "the subsequent model injection still reconstructs and shape/finite-checks "
                              "every projection before evaluation."))
    parser.add_argument("--expected-calibration-tokens", type=int, default=256)
    parser.add_argument("--expected-heldout-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.variant == "high_fidelity_nec" and args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for --variant high_fidelity_nec")
    if args.variant == "dense" and args.checkpoint_dir is not None:
        parser.error("--checkpoint-dir is forbidden for --variant dense")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    audit_path = (
        args.output.with_name(args.output.stem + ".checkpoint_audit.json")
        if args.variant == "high_fidelity_nec" else None
    )
    if audit_path is not None and audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite {audit_path}")
    if args.context_length != 2048 or args.stride != 512:
        raise ValueError("The production protocol is fixed at context=2048, stride=512")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be a positive integer")
    if args.max_eval_tokens != 0 and args.max_eval_tokens < 10_000:
        raise ValueError("Production evaluation requires 10,000 tokens or the full split")
    if args.allow_partial_prefix and args.dataset_split != "validation":
        raise ValueError("Partial-prefix development evaluation must use WikiText-2 validation, not test")
    if args.expected_cg_threshold <= 0 or args.expected_cg_iterations <= 0:
        raise ValueError("Expected MPO CG settings must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    audit = None
    if args.variant == "high_fidelity_nec":
        audit = validate_high_fidelity_checkpoint(
            args.checkpoint_dir,
            reconstruct=not args.skip_reconstruct_audit,
            reconstruction_device="cuda" if torch.cuda.is_available() else "cpu",
            expected_calibration_tokens=args.expected_calibration_tokens,
            expected_heldout_tokens=args.expected_heldout_tokens,
            allow_partial_prefix=args.allow_partial_prefix,
            expected_quantization_schedule=args.expected_quantization_schedule,
            expected_cg_threshold=args.expected_cg_threshold,
            expected_cg_iterations=args.expected_cg_iterations,
        )
        if audit_path.exists():
            if json.loads(audit_path.read_text()) != audit:
                raise FileExistsError(
                    f"Existing checkpoint audit differs from current checkpoint: {audit_path}"
                )
        else:
            audit_temp = audit_path.with_suffix(audit_path.suffix + ".partial")
            audit_temp.write_text(json.dumps(audit, indent=2) + "\n")
            audit_temp.replace(audit_path)

    reset_cuda_peak_memory_stats()
    started = time.monotonic()
    model, tokenizer, injected, compatibility, reconstruction_diagnostics = load_model(
        args.checkpoint_dir, offload_folder=args.offload_folder, variant=args.variant,
        allow_partial_prefix=args.allow_partial_prefix,
    )
    text, dataset = load_wikitext2_text(args.dataset_split, return_metadata=True)
    ppl, scored_tokens, details = compute_perplexity(
        model,
        tokenizer,
        text,
        seq_len=args.context_length,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
        device=str(model.get_input_embeddings().weight.device),
        return_details=True,
        verify_reference_loss=True,
        batch_size=args.batch_size,
        return_token_nll=True,
        progress_interval=32,
    )
    if args.max_eval_tokens > 0 and details["input_tokens"] < args.max_eval_tokens:
        raise ValueError(
            f"Requested {args.max_eval_tokens} input tokens but dataset/tokenizer produced "
            f"only {details['input_tokens']}"
        )
    expected_scored = details["input_tokens"] - 1
    if scored_tokens != expected_scored:
        raise RuntimeError(
            f"Expected {expected_scored} unique next-token targets, got {scored_tokens}"
        )
    if abs(ppl - float(torch.exp(torch.tensor(details["mean_nll"], dtype=torch.float64)))) > 1e-9 * max(1.0, ppl):
        raise RuntimeError("Perplexity is inconsistent with exp(mean NLL)")

    primary_predictions = save_top1_predictions(
        details["top1_predictions"],
        args.output.with_name(
            args.output.stem + f".primary_{args.max_eval_tokens or 'full'}.top1.i32"
        ),
    )
    primary_token_nll = save_token_nll(
        details["token_nll"],
        args.output.with_name(
            args.output.stem + f".primary_{args.max_eval_tokens or 'full'}.token_nll.npy"
        ),
    )
    full_result = None
    if args.full_test:
        if args.max_eval_tokens != 10_000:
            raise ValueError("--full-test requires the primary 10,000-token run")
        full_ppl, full_scored, full_details = compute_perplexity(
            model,
            tokenizer,
            text,
            seq_len=args.context_length,
            stride=args.stride,
            max_eval_tokens=0,
            device=str(model.get_input_embeddings().weight.device),
            return_details=True,
            verify_reference_loss=True,
            batch_size=args.batch_size,
        )
        if full_scored != full_details["input_tokens"] - 1:
            raise RuntimeError("Full-test target count is inconsistent with causal shift")
        if abs(full_ppl - math.exp(full_details["mean_nll"])) > 1e-9 * max(1.0, full_ppl):
            raise RuntimeError("Full-test perplexity is inconsistent with exp(mean NLL)")
        full_result = {
            "perplexity": full_ppl,
            "mean_nll": full_details["mean_nll"],
            "next_token_accuracy": full_details["next_token_accuracy"],
            "maximum_absolute_logit": full_details["maximum_absolute_logit"],
            "input_tokens": full_details["input_tokens"],
            "scored_tokens": full_scored,
            "token_ids_sha256": full_details["token_ids_sha256"],
            "bos_token_id": full_details["bos_token_id"],
            "tokenizer_compatibility": compatibility["tokenizer"],
            "window_count": full_details["window_count"],
            "inference_batch_size": full_details["inference_batch_size"],
            "forward_passes": full_details["forward_passes"],
            "window_specification": full_details["window_specification"],
            "nll_formula_crosscheck": full_details["nll_formula_crosscheck"],
            "top1_prediction_sidecar": save_top1_predictions(
                full_details["top1_predictions"],
                args.output.with_name(args.output.stem + ".full_test.top1.i32"),
            ),
        }

    versions = {}
    for package in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub"):
        module = __import__(package)
        versions[package] = getattr(module, "__version__", "unknown")
    protocol_identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "dataset": dataset,
        "tokenizer": compatibility["tokenizer"],
        "context_length": args.context_length,
        "stride": args.stride,
        "primary_input_tokens": args.max_eval_tokens,
        "primary_evaluation_scope": "full_split" if args.max_eval_tokens == 0 else "token_prefix",
        "full_test": args.full_test,
        "inference_batch_size": args.batch_size,
        "text_policy": "raw WikiText-2 text joined with two newlines; no instruction/chat template",
        "causal_shift": "logits[:, :-1] predict labels[:, 1:]",
        "overlap_masking": "each corpus target after the first is scored exactly once",
        "loss": "sum float32 shifted cross entropy divided by exact scored-token count",
        "use_cache": False,
    }
    is_dense = args.variant == "dense"
    primary_result = {
        "perplexity": ppl,
        "mean_nll": details["mean_nll"],
        "next_token_accuracy": details["next_token_accuracy"],
        "maximum_absolute_logit": details["maximum_absolute_logit"],
        "input_tokens": details["input_tokens"],
        "scored_tokens": scored_tokens,
        "token_ids_sha256": details["token_ids_sha256"],
        "bos_token_id": details["bos_token_id"],
        "window_count": details["window_count"],
        "inference_batch_size": details["inference_batch_size"],
        "forward_passes": details["forward_passes"],
        "window_specification": details["window_specification"],
        "nll_formula_crosscheck": details["nll_formula_crosscheck"],
        "top1_prediction_sidecar": primary_predictions,
        "token_nll_sidecar": primary_token_nll,
    }
    result = {
        "status": "COMPLETE",
        "evaluation": f"{args.variant}_wikitext2_raw",
        "model_variant": args.variant,
        "compressed_layers": injected,
        "partial_prefix": bool(args.allow_partial_prefix),
        "dataset_split": args.dataset_split,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tokenizer_compatibility": compatibility,
        "source_commit": source_commit(),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "checkpoint_audit": str(audit_path.resolve()) if audit_path else None,
        "checkpoint_manifest_sha256": audit["manifest_sha256"] if audit else None,
        "independent_numerical_coefficients": (
            DENSE_LLAMA2_7B_COEFFICIENTS if is_dense
            else audit["independent_numerical_coefficients"]
        ),
        "structural_parameter_reduction": 0.0 if is_dense else audit["structural_parameter_reduction"],
        "parameter_compression_ratio": 1.0 if is_dense else audit["parameter_compression_ratio"],
        "serialized_layer_checkpoint_bytes": None if is_dense else audit["serialized_layer_checkpoint_bytes"],
        "dataset": dataset,
        "protocol_identity": protocol_identity,
        "protocol_fingerprint": fingerprint(protocol_identity),
        "protocol": {
            "context_length": args.context_length,
            "stride": args.stride,
            "requested_max_input_tokens": args.max_eval_tokens,
            "dataset_split": args.dataset_split,
            "input_tokens": details["input_tokens"],
            "scored_tokens": scored_tokens,
            "token_ids_sha256": details["token_ids_sha256"],
            "maximum_absolute_logit": details["maximum_absolute_logit"],
            "bos_token_id": details["bos_token_id"],
            "tokenizer": compatibility["tokenizer"],
            "text_policy": "raw WikiText-2 text joined with two newlines; no instruction/chat template",
            "use_cache": False,
            "window_count": details["window_count"],
            "inference_batch_size": details["inference_batch_size"],
            "forward_passes": details["forward_passes"],
            "window_specification": details["window_specification"],
            "text_join": dataset["text_join"],
            "causal_shift": "logits[:, :-1] predict labels[:, 1:]",
            "overlap_masking": "each corpus target after the first is scored exactly once",
            "aggregation": "sum shifted cross entropy over scored labels, divided by the exact label count",
            "nll_formula_crosscheck": details["nll_formula_crosscheck"],
            "full_test_evaluated": full_result is not None,
            "full_test": full_result,
        },
        "metrics": {
            "perplexity": ppl,
            "mean_nll": details["mean_nll"],
            "next_token_accuracy": details["next_token_accuracy"],
            "primary": primary_result,
            "full_test": full_result,
        },
        "runtime": {
            "seconds": time.monotonic() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": versions,
            "cuda_peak_memory_bytes": cuda_peak_memory_allocated(),
            "hf_device_map": getattr(model, "hf_device_map", None),
        },
        "projection_reconstruction_audit": reconstruction_diagnostics,
        "precision_audit": None if is_dense else {
            "quantizer": "NEC groupwise Q4 simulation, group_size=64, signed codes [-8,7], followed by FP16 serialization",
            "per_component": {
                f"layer_{layer['layer']}.{projection}": artifact["precision_diagnostics"]
                for layer in audit["layers"]
                for projection, artifact in layer["projections"].items()
            },
            "reconstruction_error": "per-projection dense reference versus decoded NEC matrix; separate from language-model metrics",
        },
        "dense_model_evaluated": is_dense,
    }
    if args.max_eval_tokens == 10_000:
        result["metrics"]["primary_10k"] = primary_result
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "ppl": ppl,
        "mean_nll": details["mean_nll"],
        "next_token_accuracy": details["next_token_accuracy"],
        "scored_tokens": scored_tokens,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
