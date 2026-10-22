#!/usr/bin/env python3
"""Measure cumulative-prefix NEC quality at a full-model compile milestone.

All compiled layers in the prefix are replaced by NEC while the remainder of
the original transformer stays dense. The complete model is used to score the
validation tokens; this diagnostic does not select the final allocation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints
from benchmarks.evaluate_high_fidelity_nec import load_model
from huggingface_hub import hf_hub_download
from nox.evaluation import MODEL_ID, MODEL_REVISION, sha256_file
from preflight_nec_compression_targets import inspect_dense_model
from search_nec_layer_allocation import evaluate_ppl_kl, sample_ids, storage_summary
from diagnose_prefix_failure import NORMS, PROJECTIONS, module as layer_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, required=True, choices=(8, 16, 24, 32))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-tokens", type=int, default=4096)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    expected_layers = list(range(args.candidate_count))
    checkpoint_paths = [
        args.checkpoint_dir / f"compressed_layer_{layer}.pt"
        for layer in expected_layers
    ]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Milestone is missing checkpoints: {missing[:8]}")
    extra = sorted(
        int(path.stem.removeprefix("compressed_layer_"))
        for path in args.checkpoint_dir.glob("compressed_layer_*.pt")
        if int(path.stem.removeprefix("compressed_layer_")) >= args.candidate_count
    )
    if extra:
        raise ValueError(
            f"Checkpoint count is already beyond milestone {args.candidate_count}: {extra[:8]}"
        )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    started = time.monotonic()
    print("PHASE=load_full_dense_model", flush=True)
    model, tokenizer, _, compatibility, _ = load_model(
        None, offload_folder=args.output / "offload", variant="dense",
    )
    if len(model.model.layers) != 32:
        raise ValueError(f"Expected the full 32-layer model, got {len(model.model.layers)} layers")

    dense = {
        layer: {
            name: layer_module(model.model.layers[layer], name).weight.detach().cpu().clone()
            for name in (*PROJECTIONS, *NORMS)
        }
        for layer in expected_layers
    }
    print(f"PHASE=inject_prefix count={args.candidate_count}", flush=True)
    injected = inject_nox_checkpoints(
        model, str(args.checkpoint_dir), expected_layers=expected_layers,
    )
    if sorted(injected) != expected_layers:
        raise RuntimeError(f"Injected layers {injected}; expected {expected_layers}")
    compressed = {
        layer: {
            name: layer_module(model.model.layers[layer], name).weight.detach().cpu().clone()
            for name in (*PROJECTIONS, *NORMS)
        }
        for layer in expected_layers
    }

    ids, split_metadata, token_hash, split_token_count = sample_ids(
        tokenizer, "validation", 0, args.eval_tokens,
    )
    print(
        f"PHASE=full_model_quality split=validation candidate_count={args.candidate_count} "
        f"input_tokens={args.eval_tokens}",
        flush=True,
    )
    metrics = evaluate_ppl_kl(
        model, ids, dense, compressed, set(expected_layers),
        max_eval_tokens=0, seq_len=args.context_length, stride=args.stride,
        label=f"milestone_{args.candidate_count}",
    )
    manifest_path = args.checkpoint_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Resolve only the model index file. snapshot_download validates every
    # repository file, including README/license metadata that this compute
    # node does not cache, even though the tokenizer/config/weight shards used
    # above are present and complete. inspect_dense_model reads the index,
    # shard headers, and config from this same pinned snapshot.
    model_index = Path(hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors.index.json",
        revision=MODEL_REVISION,
        local_files_only=True,
    ))
    source_model_info = inspect_dense_model(model_index.parent)
    report = {
        "status": "COMPLETE",
        "purpose": "intermediate cumulative-prefix diagnostic; not a final adaptive allocation",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "quality_evaluation_model_layer_count": len(model.model.layers),
        "compressed_prefix_layers": expected_layers,
        "dense_suffix_layers": list(range(args.candidate_count, len(model.model.layers))),
        "compressed_candidate_count": args.candidate_count,
        "dense_layer_count_in_model": len(model.model.layers) - args.candidate_count,
        "selection_rule": "compress every currently compiled prefix layer; all other layers remain dense",
        "data": {
            "split": "validation",
            "split_metadata": split_metadata,
            "token_ids_sha256": token_hash,
            "split_token_count": split_token_count,
            "input_tokens": args.eval_tokens,
            "scored_tokens": metrics["scored_tokens"],
            "context_length": args.context_length,
            "stride": args.stride,
        },
        "metrics": metrics,
        "storage_preflight": storage_summary(
            args.checkpoint_dir, dense, tuple(expected_layers),
            set(expected_layers), source_model_info,
        ),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": {
            path.name: sha256_file(path) for path in checkpoint_paths
        },
        "checkpoint_compile_identity": manifest.get("run_identity"),
        "elapsed_seconds": time.monotonic() - started,
        "limitation": (
            "The allocation at each milestone is deliberately the cumulative compiled prefix. "
            "It is a diagnostic for error accumulation, not evidence that the adaptive "
            "allocator has selected the final mask."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"COMPLETE candidate_count={args.candidate_count} "
        f"student_ppl={metrics['perplexity']:.6g} "
        f"teacher_ppl={metrics['teacher_perplexity']:.6g} "
        f"teacher_kl={metrics['teacher_kl_nats_per_token']:.6g} "
        f"report={args.output / 'report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
