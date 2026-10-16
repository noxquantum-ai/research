#!/usr/bin/env python3
"""Compare original and packed checkpoints directly, including held-out outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nox.compiler import reconstruct_from_artifact
from nox.packing import unpack_artifact
from nox.sequential_compiler import forward_transformer_block
from scripts.audit_checkpoint_reconstruction import compare_weights


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--activation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    original = torch.load(args.original, map_location="cpu", weights_only=True)
    packed = torch.load(args.packed, map_location="cpu", weights_only=True)
    if original["layer_idx"] != packed["layer_idx"] or original["rms_norm_eps"] != packed["rms_norm_eps"]:
        raise ValueError("Checkpoint layer/config mismatch")
    x = torch.load(args.activation_file, map_location="cpu", weights_only=True)
    if isinstance(x, dict):
        x = x["X_test"]
    first, second, scores = {}, {}, {}
    for name, artifact in original["compressed_artifacts"].items():
        first[name] = reconstruct_from_artifact(artifact)
        second[name] = reconstruct_from_artifact(unpack_artifact(packed["compressed_artifacts"][name]))
        scores[name] = compare_weights(first[name], second[name])
        print(name, json.dumps(scores[name]), flush=True)
    for name in ("input_layernorm", "post_attention_layernorm"):
        if not torch.equal(original[name], packed[name]):
            raise ValueError("Normalization weights changed")
        first[name], second[name] = original[name].float(), packed[name].float()
    options = dict(n_heads=32, n_kv_heads=32, seq_len=128, rms_norm_eps=original["rms_norm_eps"])
    output_first, _, _ = forward_transformer_block(first, x.float(), **options)
    output_second, _, _ = forward_transformer_block(second, x.float(), **options)
    result = {"layer": original["layer_idx"], "device": "cpu", "forward_options": options,
              "heldout_tokens": x.shape[1], "projections": scores,
              "heldout_block_difference": compare_weights(output_first, output_second),
              "input_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in (("original", args.original), ("packed", args.packed), ("activations", args.activation_file))},
              "full_model_quality_verified": False}
    result["passes_roundtrip_gate"] = (all(score["relative_frobenius_error"] < 0.001 for score in scores.values())
                                        and result["heldout_block_difference"]["relative_frobenius_error"] < 0.001)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result["heldout_block_difference"]), flush=True)
    if not result["passes_roundtrip_gate"]:
        raise SystemExit("Packed checkpoint failed the diagnostic round-trip gate")


if __name__ == "__main__":
    main()
