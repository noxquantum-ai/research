#!/usr/bin/env python3
"""Prepare matched prefix/uniform calibration activations with canonical dense weights."""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compile_high_fidelity import PinnedWeights
from benchmarks.benchmark_day3_cascade import build_real_token_calibration_data
from nox.sequential_compiler import forward_transformer_block


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument("--n-validation", type=int, default=128)
    parser.add_argument("--validation-sampling", choices=("prefix", "uniform"), default="prefix")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model_id = "NousResearch/Llama-2-7b-hf"
    revision = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    source = PinnedWeights(model_id, revision)
    embedding = source.tensor("model.embed_tokens.weight")
    streams, metadata = {}, {}
    for sampling in ("prefix", "uniform"):
        train, validation, data = build_real_token_calibration_data(
            embedding, model_id, n_calib=args.n_calib, n_test=args.n_validation, seq_len=128,
            revision=revision, calibration_sampling=sampling, calibration_seed=42,
            validation_sampling=args.validation_sampling, validation_seed=43,
        )
        streams[sampling], metadata[sampling] = train.cuda(), data
    streams["validation"] = validation.cuda()
    del embedding
    for index in range(args.layer):
        weights = source.layer(index, "cuda")
        for sampling in streams:
            streams[sampling], _, _ = forward_transformer_block(
                weights, streams[sampling], n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads, seq_len=128, rms_norm_eps=config.rms_norm_eps,
            )
        del weights
        gc.collect()
        print(f"dense_prefix_layer={index}", flush=True)
    for sampling, activations in streams.items():
        torch.save(activations.cpu(), args.output / f"{sampling}.pt")
    metadata["layer"] = args.layer
    metadata["rms_norm_eps"] = config.rms_norm_eps
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
