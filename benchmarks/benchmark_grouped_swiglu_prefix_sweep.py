#!/usr/bin/env python3
"""Measure WikiText-2 PPL as progressively more grouped-SwiGLU layers are applied.

The model is loaded once. The dense baseline is measured first; for each prefix,
the original dense MLP modules are restored and artifacts for layers [0, N) are
injected. No fitting or weight updates occur in this benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import compute_perplexity  # noqa: E402
from benchmarks.benchmark_native_swiglu_perplexity import (  # noqa: E402
    inject_native_mlp_artifacts,
)
from nox.data import load_wikitext2_raw_split  # noqa: E402


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-model perplexity evaluation requires CUDA")
    if any(length < 0 or length > 32 for length in args.prefix_lengths):
        raise ValueError("prefix lengths must be between 0 and 32")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True,
    ).to(args.device)
    model.eval()
    text, metadata = load_wikitext2_raw_split("test")
    dense_ppl, labels, dense_details = compute_perplexity(
        model, tokenizer, text, seq_len=args.seq_len, stride=args.stride,
        max_eval_tokens=args.max_eval_tokens, device=args.device,
        return_details=True,
    )

    dense_mlps = [layer.mlp for layer in model.model.layers]
    for mlp in dense_mlps:
        mlp.to("cpu")
    results = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "test_split": metadata,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_eval_tokens": args.max_eval_tokens,
        "labels": labels,
        "dense": {
            "ppl": dense_ppl,
            "mean_nll": dense_details["mean_nll"],
            "next_token_accuracy": dense_details["next_token_accuracy"],
        },
        "prefixes": [],
        "uses_gradients": False,
        "post_compression_optimization": False,
        "uses_dense_residual": False,
    }
    _write_json(args.output, results)

    for prefix_length in args.prefix_lengths:
        torch.cuda.reset_peak_memory_stats()
        for layer_idx, mlp in enumerate(dense_mlps):
            model.model.layers[layer_idx].mlp = mlp.to(args.device)
        artifact_paths = [
            args.artifact_dir / f"native_mlp_layer_{layer_idx}.pt"
            for layer_idx in range(prefix_length)
        ]
        missing = [str(path) for path in artifact_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing prefix artifacts: {missing[:3]}")
        artifacts = inject_native_mlp_artifacts(model, artifact_paths)
        value, prefix_labels, details = compute_perplexity(
            model, tokenizer, text, seq_len=args.seq_len, stride=args.stride,
            max_eval_tokens=args.max_eval_tokens, device=args.device,
            return_details=True,
        )
        torch.cuda.synchronize()
        results["prefixes"].append({
            "length": prefix_length,
            "ppl": value,
            "labels": prefix_labels,
            "mean_nll": details["mean_nll"],
            "next_token_accuracy": details["next_token_accuracy"],
            "artifact_bytes": sum(item["bytes"] for item in artifacts),
            "artifacts": artifacts,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        })
        print(json.dumps(results["prefixes"][-1], sort_keys=True), flush=True)
        for layer_idx, mlp in enumerate(dense_mlps):
            model.model.layers[layer_idx].mlp = mlp.to("cpu")
        _write_json(args.output, results)

    results["runtime_seconds"] = float(time.perf_counter() - started)
    _write_json(args.output, results)
    print(json.dumps({
        "dense_ppl": dense_ppl,
        "prefix_ppl": [(item["length"], item["ppl"]) for item in results["prefixes"]],
        "runtime_seconds": results["runtime_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
