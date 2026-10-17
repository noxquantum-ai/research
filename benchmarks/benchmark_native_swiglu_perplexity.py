#!/usr/bin/env python3
"""Evaluate native-coordinate SwiGLU artifacts in the dense reference model."""

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

from benchmarks.benchmark_day4_perplexity import compute_perplexity  # noqa: E402
from nox.data import load_wikitext2_raw_split  # noqa: E402
from nox.native_swiglu import CODEC_TYPE, NativeSwiGLUModule  # noqa: E402
from nox.grouped_swiglu_tn import (  # noqa: E402
    CODEC_TYPE as GROUPED_CODEC_TYPE,
    GroupedSwiGLUModule,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inject_native_mlp_artifacts(model, artifact_paths: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for path in artifact_paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        layer = int(payload["layer_idx"])
        if layer in seen:
            raise ValueError(f"duplicate layer artifact: {layer}")
        seen.add(layer)
        artifact = payload.get("mlp_artifact", payload.get("native_mlp_artifact"))
        codec = artifact.get("codec_type")
        if codec not in (CODEC_TYPE, GROUPED_CODEC_TYPE):
            raise ValueError(f"unexpected artifact codec in {path}: {codec}")
        target = model.model.layers[layer]
        weight = target.self_attn.q_proj.weight
        module_type = NativeSwiGLUModule if codec == CODEC_TYPE else GroupedSwiGLUModule
        target.mlp = module_type(artifact, device=weight.device, dtype=weight.dtype)
        records.append({
            "layer": layer,
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "codec_type": codec,
            "retained_neurons": artifact.get("retained_neurons"),
            "groups": artifact.get("groups"),
            "rank": artifact.get("rank"),
            "float_coefficients": int(artifact["coefficient_count"]),
        })
    return sorted(records, key=lambda value: value["layer"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--eval-dense", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real-model perplexity evaluation requires CUDA")

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
    text, metadata = load_wikitext2_raw_split("test")
    dense = None
    if args.eval_dense:
        value, labels, details = compute_perplexity(
            model, tokenizer, text, seq_len=args.seq_len, stride=args.stride,
            max_eval_tokens=args.max_eval_tokens, device=args.device,
            return_details=True,
        )
        dense = {
            "ppl": value,
            "labels": labels,
            "mean_nll": details["mean_nll"],
            "next_token_accuracy": details["next_token_accuracy"],
        }
    artifacts = inject_native_mlp_artifacts(model, args.artifact)
    value, labels, details = compute_perplexity(
        model, tokenizer, text, seq_len=args.seq_len, stride=args.stride,
        max_eval_tokens=args.max_eval_tokens, device=args.device,
        return_details=True,
    )
    torch.cuda.synchronize()
    result = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "test_split": metadata,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "max_eval_tokens": args.max_eval_tokens,
        "dense": dense,
        "native_swiglu": {
            "ppl": value,
            "labels": labels,
            "mean_nll": details["mean_nll"],
            "next_token_accuracy": details["next_token_accuracy"],
        },
        "artifacts": artifacts,
        "artifact_bytes": sum(item["bytes"] for item in artifacts),
        "runtime_seconds": float(time.perf_counter() - started),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "uses_gradients": False,
        "post_compression_optimization": False,
        "uses_dense_residual": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
