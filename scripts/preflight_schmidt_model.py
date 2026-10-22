#!/usr/bin/env python3
"""Discover an HF causal LM and preflight NEC binary-Schmidt storage targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.model_adapter import inventory_model
from nox.schmidt_budget import plan_uniform_schmidt_budget


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--target-ratio", type=float, action="append", default=[])
    parser.add_argument("--scale-bits", type=int, choices=(16, 32), default=16)
    parser.add_argument("--minimum-rank", type=int, default=8)
    parser.add_argument("--max-paths", type=int, default=8)
    parser.add_argument("--metadata-reserve-mib", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.float16,
        )
    model.tie_weights()
    model.to(dtype=torch.float16)
    inventory = inventory_model(model)

    ratios = args.target_ratio or [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    plans = [
        plan_uniform_schmidt_budget(
            inventory,
            target_ratio=ratio,
            scale_bits=args.scale_bits,
            minimum_rank=args.minimum_rank,
            max_paths=args.max_paths,
            metadata_reserve_bytes=round(args.metadata_reserve_mib * 1024 * 1024),
        )
        for ratio in ratios
    ]
    report = {
        "schema": "nec_schmidt_preflight_v1",
        "model_id": args.model_id,
        "revision": args.revision,
        "dense_baseline_dtype": "float16",
        "inventory": inventory.to_dict(),
        "plans": [plan.to_dict() for plan in plans],
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
