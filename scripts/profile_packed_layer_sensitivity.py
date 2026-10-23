#!/usr/bin/env python3
"""Measure end-to-end marginal damage for each packed NEC layer once per model load."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nox.compiler import reconstruct_from_artifact
from nox.data import load_wikitext2_raw_split
from search_nec_layer_allocation import collect_logits, token_logit_metrics


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def modules_for_layer(layer) -> dict[str, torch.nn.Module]:
    return {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj,
        "up_proj": layer.mlp.up_proj,
        "down_proj": layer.mlp.down_proj,
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--start-token", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.tokens < 2:
        parser.error("--tokens must be at least two")
    started = time.monotonic()
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if args.device.startswith("cuda") else None,
    )
    if not args.device.startswith("cuda"):
        model.to(args.device)
    model.eval()

    text, dataset = load_wikitext2_raw_split(args.split)
    all_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    end = args.start_token + args.tokens
    if args.start_token < 0 or end > all_ids.shape[1]:
        raise ValueError("requested token range is outside the selected split")
    ids = all_ids[:, args.start_token:end].contiguous()
    token_sha256 = hashlib.sha256(ids.to(torch.int64).numpy().tobytes()).hexdigest()
    dense_logits, targets = collect_logits(
        model, ids, seq_len=args.seq_len, stride=args.stride,
    )
    dense_metrics = token_logit_metrics(
        dense_logits, dense_logits, targets, device=model.device,
    )

    checkpoint_paths = sorted(
        args.checkpoint_dir.glob("compressed_layer_*.pt"),
        key=lambda path: int(path.stem.removeprefix("compressed_layer_")),
    )
    if not checkpoint_paths:
        raise FileNotFoundError("no compressed layer checkpoints found")

    rows = []
    for path in checkpoint_paths:
        layer_started = time.monotonic()
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        layer_index = int(checkpoint["layer_idx"])
        layer = model.model.layers[layer_index]
        modules = modules_for_layer(layer)
        originals = {
            name: module.weight.detach().clone()
            for name, module in modules.items()
        }
        for name, module in modules.items():
            reconstructed = reconstruct_from_artifact(
                checkpoint["compressed_artifacts"][name],
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            module.weight.copy_(reconstructed)
            del reconstructed

        candidate_logits, candidate_targets = collect_logits(
            model, ids, seq_len=args.seq_len, stride=args.stride,
        )
        if not torch.equal(targets, candidate_targets):
            raise RuntimeError("dense and candidate probes used different targets")
        metrics = token_logit_metrics(
            dense_logits, candidate_logits, targets, device=model.device,
        )
        for name, module in modules.items():
            module.weight.copy_(originals[name])
        row = {
            "layer": layer_index,
            "checkpoint_bytes": path.stat().st_size,
            **metrics,
            "excess_nll": metrics["mean_nll"] - dense_metrics["mean_nll"],
            "seconds": time.monotonic() - layer_started,
        }
        rows.append(row)
        print(
            f"layer={layer_index} ppl={metrics['perplexity']:.6f} "
            f"kl={metrics['mean_kl_nats_per_token']:.6f} "
            f"seconds={row['seconds']:.2f}",
            flush=True,
        )
        del checkpoint, originals, candidate_logits, candidate_targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "schema": "nec_packed_layer_sensitivity_v1",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "model_id": args.model_id,
        "dataset": dataset,
        "selection_range": {
            "split": args.split,
            "start_token": args.start_token,
            "end_token_exclusive": end,
            "token_sha256": token_sha256,
            "scored_tokens": int(targets.numel()),
        },
        "dense": dense_metrics,
        "layers": rows,
        "ranked_least_to_most_damage": [
            row["layer"] for row in sorted(rows, key=lambda item: item["mean_kl_nats_per_token"])
        ],
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
