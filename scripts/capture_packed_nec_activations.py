#!/usr/bin/env python3
"""Capture representative dense-model trajectories for NEC construction."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import sample_calibration_offsets
from nox.data import load_wikitext2_raw_split


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=int, default=32768)
    parser.add_argument("--validation-tokens", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.calibration_tokens, args.validation_tokens, args.sequence_length,
           args.batch_windows) <= 0:
        raise ValueError("capture sizes must be positive")
    if any(count % args.sequence_length for count in (
        args.calibration_tokens, args.validation_tokens,
    )):
        raise ValueError("capture token counts must contain complete windows")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16, device_map="auto", local_files_only=True,
    ).eval()
    layers = model.model.layers
    records = {}
    for split, count, seed, prefix in (
        ("train", args.calibration_tokens, args.seed, "calibration_input_layer_"),
        ("validation", args.validation_tokens, args.seed + 1, "layer_input_validation_layer_"),
    ):
        text, metadata = load_wikitext2_raw_split(split)
        corpus = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
        offsets = sample_calibration_offsets(
            int(corpus.numel()), count, args.sequence_length, seed,
        )
        windows = torch.stack([corpus[start:start + args.sequence_length] for start in offsets])
        captures = [torch.empty(model.config.hidden_size, count, dtype=torch.float16)
                    for _ in layers]
        handles = []
        current_begin = 0
        for index, layer in enumerate(layers):
            def capture(_module, values, layer_index=index):
                hidden = values[0].detach()
                flat = hidden.reshape(-1, hidden.shape[-1]).T
                captures[layer_index][:, current_begin:current_begin + flat.shape[1]].copy_(
                    flat.to(device="cpu", dtype=torch.float16),
                )
            handles.append(layer.register_forward_pre_hook(capture))
        try:
            for begin in range(0, len(offsets), args.batch_windows):
                current_begin = begin * args.sequence_length
                batch = windows[begin:begin + args.batch_windows].to(model.device)
                # Decoder-only capture avoids the unnecessary vocabulary head
                # and its large [batch, sequence, vocabulary] output tensor.
                model.model(batch, use_cache=False)
                print(f"CAPTURE {split} windows={min(begin + args.batch_windows, len(offsets))}"
                      f"/{len(offsets)}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
        file_sizes = {}
        for layer_index, hidden in enumerate(captures):
            path = args.output / f"{prefix}{layer_index}.pt"
            torch.save(hidden, path)
            file_sizes[path.name] = path.stat().st_size
        records[split] = {
            "source": metadata, "tokens": count, "window_offsets": offsets,
            "sampling_seed": seed,
            "token_ids_sha256": hashlib.sha256(windows.numpy().tobytes()).hexdigest(),
            "file_sizes": file_sizes,
        }
        del captures, windows, corpus, text
        gc.collect()
    manifest = {
        "schema": "nec_dense_activation_capture_v1",
        "model_dir": str(args.model_dir.resolve()),
        "model_config_sha256": hashlib.sha256(
            (args.model_dir / "config.json").read_bytes(),
        ).hexdigest(),
        "sequence_length": args.sequence_length,
        "capture_backend": "original_huggingface_decoder_pre_hooks",
        "dtype": "float16",
        "records": records,
        "seconds": time.monotonic() - started,
    }
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key != "records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
