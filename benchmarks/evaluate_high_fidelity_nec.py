#!/usr/bin/env python3
"""Strict WikiText-2 evaluation of a complete High-Fidelity NEC checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import (
    compute_perplexity,
    inject_nox_checkpoints,
    load_wikitext2_test_text,
)
from nox.evaluation import (
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


def load_model(checkpoint_dir: Path, offload_folder: Path | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("Production NEC evaluation requires CUDA")
    # Keep the conservative historical T4 defaults; the RunPod runner derives
    # larger caps from free VRAM/RAM and sets them in this process environment.
    # Placement changes only; model dtype, logits, and token scoring do not.
    max_memory = evaluation_memory_budget(gpu_count)
    offload = offload_folder or (checkpoint_dir / "evaluation_offload")
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
    reconstruction_diagnostics = []
    injected = inject_nox_checkpoints(
        model, str(checkpoint_dir), dtype=torch.float16,
        diagnostics=reconstruction_diagnostics,
    )
    if len(injected) != 32 or set(injected) != set(range(32)):
        raise RuntimeError(f"Incomplete NEC injection: {sorted(injected)}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-eval-tokens", type=int, default=10_000,
                        help="Zero scores the complete test split")
    parser.add_argument("--offload-folder", type=Path, default=None,
                        help="Writable folder for Accelerate disk offload")
    parser.add_argument("--full-test", action="store_true",
                        help="Report both the required 10k-token result and full raw test split")
    parser.add_argument("--skip-reconstruct-audit", action="store_true",
                        help=("Skip the redundant dense reconstruction pass in the metadata audit; "
                              "the subsequent model injection still reconstructs and shape/finite-checks "
                              "every projection before evaluation."))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    audit_path = args.output.with_name(args.output.stem + ".checkpoint_audit.json")
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite {audit_path}")
    if args.context_length != 2048 or args.stride != 512:
        raise ValueError("The production protocol is fixed at context=2048, stride=512")
    if args.max_eval_tokens != 0 and args.max_eval_tokens < 10_000:
        raise ValueError("Production evaluation requires 10,000 tokens or the full split")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    audit = validate_high_fidelity_checkpoint(
        args.checkpoint_dir,
        reconstruct=not args.skip_reconstruct_audit,
        reconstruction_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    audit_temp = audit_path.with_suffix(audit_path.suffix + ".partial")
    audit_temp.write_text(json.dumps(audit, indent=2) + "\n")
    audit_temp.replace(audit_path)

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.monotonic()
    model, tokenizer, injected, compatibility, reconstruction_diagnostics = load_model(
        args.checkpoint_dir, offload_folder=args.offload_folder,
    )
    text, dataset = load_wikitext2_test_text(return_metadata=True)
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
            "window_specification": full_details["window_specification"],
            "nll_formula_crosscheck": full_details["nll_formula_crosscheck"],
        }

    versions = {}
    for package in ("torch", "transformers", "accelerate", "safetensors", "huggingface_hub"):
        module = __import__(package)
        versions[package] = getattr(module, "__version__", "unknown")
    result = {
        "status": "COMPLETE",
        "evaluation": "high_fidelity_nec_wikitext2_raw",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tokenizer_compatibility": compatibility,
        "source_commit": source_commit(),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_audit": str(audit_path.resolve()),
        "checkpoint_manifest_sha256": audit["manifest_sha256"],
        "compressed_layers": injected,
        "independent_numerical_coefficients": audit["independent_numerical_coefficients"],
        "structural_parameter_reduction": audit["structural_parameter_reduction"],
        "parameter_compression_ratio": audit["parameter_compression_ratio"],
        "serialized_layer_checkpoint_bytes": audit["serialized_layer_checkpoint_bytes"],
        "dataset": dataset,
        "protocol": {
            "context_length": args.context_length,
            "stride": args.stride,
            "requested_max_input_tokens": args.max_eval_tokens,
            "input_tokens": details["input_tokens"],
            "scored_tokens": scored_tokens,
            "token_ids_sha256": details["token_ids_sha256"],
            "maximum_absolute_logit": details["maximum_absolute_logit"],
            "bos_token_id": details["bos_token_id"],
            "tokenizer": compatibility["tokenizer"],
            "text_policy": "raw WikiText-2 text joined with two newlines; no instruction/chat template",
            "use_cache": False,
            "window_count": details["window_count"],
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
            "primary_10k": {
                "perplexity": ppl,
                "mean_nll": details["mean_nll"],
                "next_token_accuracy": details["next_token_accuracy"],
                "maximum_absolute_logit": details["maximum_absolute_logit"],
                "input_tokens": details["input_tokens"],
                "scored_tokens": scored_tokens,
            },
            "full_test": full_result,
        },
        "runtime": {
            "seconds": time.monotonic() - started,
            "python": sys.version,
            "platform": platform.platform(),
            "package_versions": versions,
            "cuda_peak_memory_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "hf_device_map": getattr(model, "hf_device_map", None),
        },
        "projection_reconstruction_audit": reconstruction_diagnostics,
        "precision_audit": {
            "quantizer": "NEC groupwise Q4 simulation, group_size=64, signed codes [-8,7], followed by FP16 serialization",
            "per_component": {
                f"layer_{layer['layer']}.{projection}": artifact["precision_diagnostics"]
                for layer in audit["layers"]
                for projection, artifact in layer["projections"].items()
            },
            "reconstruction_error": "per-projection dense reference versus decoded NEC matrix; separate from language-model metrics",
        },
        "dense_model_evaluated": False,
    }
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
