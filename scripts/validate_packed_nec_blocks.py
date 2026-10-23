#!/usr/bin/env python3
"""Compare paired NEC constructions on separate dense validation trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.packed_schmidt import reconstruct_packed_schmidt
from nox.sequential_compiler import forward_transformer_block
from scripts.compile_packed_schmidt_model import (
    ATTENTION, PROJECTIONS, SafetensorReader, layer_tensor_names,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--activation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())
    capture = json.loads((args.activation_dir / "model_manifest.json").read_text())
    sequence_length = capture.get("sequence_length", 128)
    reader = SafetensorReader(args.model_dir)
    records = []
    for path in sorted(args.checkpoint_dir.glob("compressed_layer_*.pt")):
        layer = int(path.stem.removeprefix("compressed_layer_"))
        prefix = f"model.layers.{layer}."
        tensors = reader.load(layer_tensor_names(layer))
        dense = {
            name: tensors[prefix + ("self_attn." if name in ATTENTION else "mlp.")
                          + name + ".weight"].cuda().float()
            for name in PROJECTIONS
        }
        for norm in ("input_layernorm", "post_attention_layernorm"):
            dense[norm] = tensors[prefix + norm + ".weight"].cuda().float()
        inputs = torch.load(
            args.activation_dir / f"layer_input_validation_layer_{layer}.pt",
            map_location="cuda", weights_only=True,
        ).float()
        options = dict(
            n_heads=config["num_attention_heads"],
            n_kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            seq_len=sequence_length, rms_norm_eps=config["rms_norm_eps"],
        )
        reference, reference_attention, reference_mlp = forward_transformer_block(
            dense, inputs, **options,
        )
        row = {"layer": layer, "heldout_tokens": inputs.shape[1]}
        for label, checkpoint_path in (("baseline", args.baseline_dir / path.name),
                                       ("candidate", path)):
            data = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            weights = {**dense, **{
                name: reconstruct_packed_schmidt(artifact, device="cuda")
                for name, artifact in data["compressed_artifacts"].items()
            }}
            output, attention, mlp = forward_transformer_block(weights, inputs, **options)
            relative = lambda value, teacher: float(
                (value - teacher).norm() / teacher.norm().clamp_min(1e-30)
            )
            row[label] = {
                "attention_relative_error": relative(attention, reference_attention),
                "mlp_relative_error": relative(mlp, reference_mlp),
                "block_update_relative_error": relative(output - inputs, reference - inputs),
                "residual_stream_relative_error": relative(output, reference),
            }
            del data, weights, output, attention, mlp
        records.append(row)
        print(json.dumps(row), flush=True)
        del dense, tensors, inputs, reference, reference_attention, reference_mlp
        torch.cuda.empty_cache()
    report = {
        "schema": "nec_paired_heldout_block_v1",
        "activation_dir": str(args.activation_dir.resolve()),
        "sequence_length": sequence_length,
        "selection_uses_these_activations": False,
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
