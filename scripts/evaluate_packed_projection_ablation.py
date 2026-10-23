#!/usr/bin/env python3
"""Measure which packed NEC projection groups cause end-to-end quality loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import compute_perplexity, inject_nox_checkpoints
from nox.data import load_wikitext2_raw_split


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CONFIGURATIONS = {
    "dense": (),
    "q_k": ("q_proj", "k_proj"),
    "v_o": ("v_proj", "o_proj"),
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "gate_up": ("gate_proj", "up_proj"),
    "down": ("down_proj",),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "all": PROJECTIONS,
}


def module(layer, name: str):
    return getattr(layer.self_attn if name in PROJECTIONS[:4] else layer.mlp, name)


@torch.inference_mode()
def install(model, dense, compressed, names) -> None:
    selected = set(names)
    for layer_index in sorted(dense):
        for name in PROJECTIONS:
            source = compressed[layer_index][name] if name in selected else dense[layer_index][name]
            destination = module(model.model.layers[layer_index], name).weight
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if str(args.device).startswith("cuda") else None,
    )
    if not str(args.device).startswith("cuda"):
        model.to(args.device)
    model.eval()
    layers = sorted(
        int(path.stem.removeprefix("compressed_layer_"))
        for path in args.checkpoint_dir.glob("compressed_layer_*.pt")
    )
    if not layers:
        raise FileNotFoundError("checkpoint directory contains no compressed layers")
    dense = {
        index: {
            name: module(model.model.layers[index], name).weight.detach().cpu().clone()
            for name in PROJECTIONS
        }
        for index in layers
    }
    inject_nox_checkpoints(
        model, args.checkpoint_dir, device=args.device, dtype=dtype,
        expected_layers=layers,
    )
    compressed = {
        index: {
            name: module(model.model.layers[index], name).weight.detach().cpu().clone()
            for name in PROJECTIONS
        }
        for index in layers
    }
    text, dataset = load_wikitext2_raw_split("test")
    rows = {}
    for label, names in CONFIGURATIONS.items():
        install(model, dense, compressed, names)
        perplexity, scored_tokens, details = compute_perplexity(
            model, tokenizer, text,
            seq_len=args.seq_len,
            stride=args.stride,
            max_eval_tokens=args.max_eval_tokens,
            device=args.device,
            return_details=True,
        )
        rows[label] = {
            "compressed_projections": list(names),
            "perplexity": perplexity,
            "mean_nll": details["mean_nll"],
            "next_token_accuracy": details["next_token_accuracy"],
            "scored_tokens": scored_tokens,
        }
        print("RESULT " + label + " " + json.dumps(rows[label]), flush=True)
    report = {
        "schema": "nec_packed_projection_ablation_v1",
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "compressed_layers": layers,
        "dataset": dataset,
        "protocol": {
            "max_eval_tokens": args.max_eval_tokens,
            "seq_len": args.seq_len,
            "stride": args.stride,
        },
        "results": rows,
        "seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
