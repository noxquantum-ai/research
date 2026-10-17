#!/usr/bin/env python3
"""Prepare matched layer-0 WikiText activations and dense weights for NEC diagnosis.

This downloads only the layer tensors and token embedding needed for layer 0,
then creates disjoint train/validation embedding streams. It does not compile,
fit, or modify any NEC checkpoint.
"""

import argparse
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import (
    HF_TOKEN,
    build_real_token_calibration_data,
    fetch_layer_weights,
    fetch_named_tensor,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument(
        "--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--n-calib", type=int, default=4096)
    parser.add_argument("--n-validation", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=43)
    args = parser.parse_args()

    if args.layer != 0:
        raise ValueError("This preparation script is intentionally limited to layer 0.")
    if args.seq_len < 2 or any(
        count < args.seq_len or count % args.seq_len
        for count in (args.n_calib, args.n_validation)
    ):
        raise ValueError("Token counts must be positive multiples of seq-len.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = (
        args.output_dir / "calibration_input_layer_0.pt",
        args.output_dir / "layer_input_validation_layer_0.pt",
        args.output_dir / "diagnostic_data_manifest.json",
    )
    if any(path.exists() for path in output_files):
        raise FileExistsError("Refusing to overwrite existing diagnostic input files.")

    args.dense_weights_dir.mkdir(parents=True, exist_ok=True)
    weights = fetch_layer_weights(
        repo_id=args.model_id,
        layer_idx=0,
        save_dir=str(args.dense_weights_dir),
        token=HF_TOKEN,
        revision=args.model_revision,
    )
    embedding = fetch_named_tensor(
        args.model_id,
        "model.embed_tokens.weight",
        "model_embed_tokens",
        str(args.dense_weights_dir),
        token=HF_TOKEN,
        revision=args.model_revision,
    )
    calibration, validation, data_metadata = build_real_token_calibration_data(
        embedding,
        args.model_id,
        n_calib=args.n_calib,
        n_test=args.n_validation,
        seq_len=args.seq_len,
        revision=args.model_revision,
        calibration_sampling="uniform",
        calibration_seed=args.calibration_seed,
        validation_sampling="uniform",
        validation_seed=args.validation_seed,
    )
    torch.save(calibration.cpu(), output_files[0])
    torch.save(validation.cpu(), output_files[1])
    metadata = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layer": args.layer,
        "dense_projection_files": sorted(
            str(args.dense_weights_dir / f"layer_0_{name}.pt") for name in weights
        ),
        "calibration_tokens": int(calibration.shape[1]),
        "heldout_tokens": int(validation.shape[1]),
        "sequence_length": args.seq_len,
        "calibration_seed": args.calibration_seed,
        "validation_seed": args.validation_seed,
        "calibration_sampling": "uniform",
        "validation_sampling": "uniform",
        "calibration_tensor_shape": list(calibration.shape),
        "validation_tensor_shape": list(validation.shape),
        "huggingface_token_present": bool(HF_TOKEN or os.getenv("HF_TOKEN")),
        "dataset_and_tokenizer": data_metadata,
        "post_compression_optimization": False,
    }
    output_files[2].write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "dense_weights_dir": str(args.dense_weights_dir),
        "calibration_shape": list(calibration.shape),
        "validation_shape": list(validation.shape),
        "model_revision": args.model_revision,
        "dataset_train_sha256": data_metadata["calibration_source"]["source_sha256"],
        "dataset_validation_sha256": data_metadata["heldout_source"]["source_sha256"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
