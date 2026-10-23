#!/usr/bin/env python3
"""Compare exported NEC inference with dense-source checkpoint injection."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=128)
    args = parser.parse_args()
    if args.tokens < 2:
        parser.error("--tokens must be at least two")

    # The source interpreter has imported NEC already. Prove the packaged
    # reconstruction source matches it, rather than silently testing different
    # code through Python's module cache.
    source_matches = all(
        (ROOT / "nox" / path.relative_to(args.package_dir / "nox")).read_bytes()
        == path.read_bytes()
        for path in (args.package_dir / "nox").rglob("*.py")
    )
    if not source_matches:
        raise ValueError("bundled NEC source differs from the verification interpreter")
    started = time.monotonic()
    reference = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16, device_map="auto", local_files_only=True,
    ).eval()
    inject_nox_checkpoints(
        reference, args.package_dir / "nec_checkpoints", device="cuda", dtype=torch.float16,
    )
    token_ids = torch.randint(
        0, reference.config.vocab_size, (1, args.tokens),
        generator=torch.Generator().manual_seed(71),
    )
    expected = reference(token_ids.to("cuda"), use_cache=False).logits.cpu()
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    spec = importlib.util.spec_from_file_location(
        "exported_nec_package", args.package_dir / "load_nec_package.py",
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    restored = loader.load_model(device_map="auto", torch_dtype=torch.float16)
    actual = restored(token_ids.to("cuda"), use_cache=False).logits.cpu()
    difference = actual.float() - expected.float()
    passed = torch.allclose(actual, expected, atol=1e-3, rtol=1e-3)
    report = {
        "schema": "nec_exported_package_parity_v1",
        "package_dir": str(args.package_dir.resolve()),
        "tokens": args.tokens,
        "input_seed": 71,
        "maximum_absolute_logit_difference": float(difference.abs().max()),
        "relative_logit_l2_error": float(
            difference.norm() / expected.float().norm().clamp_min(1e-30)
        ),
        "logits_bitwise_equal": torch.equal(actual, expected),
        "bundled_nec_source_matches": source_matches,
        "passed": passed,
        "note": "Serialization parity only; random inputs do not measure capability.",
        "seconds": time.monotonic() - started,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("exported package does not match the injected model")


if __name__ == "__main__":
    main()
