#!/usr/bin/env python3
"""Measure a packed storage prototype and export its decoded quality control.

The decoded checkpoint can use existing inference tools. Its bytes are not the
packed size, and this script does not establish inference memory savings.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nox.artifacts import source_files_fingerprint
from nox.packing import pack_artifact, tensor_payload_bytes, unpack_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    packed = dict(checkpoint)
    packed["compressed_artifacts"] = {name: pack_artifact(artifact)
                                      for name, artifact in checkpoint["compressed_artifacts"].items()}
    packed["total_kb"] = tensor_payload_bytes(packed) / 1024
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "packed_checkpoint.pt"
    torch.save(packed, path)
    decoded = torch.load(path, map_location="cpu", weights_only=True)
    decoded["compressed_artifacts"] = {name: unpack_artifact(artifact)
                                       for name, artifact in decoded["compressed_artifacts"].items()}
    errors = {}
    for name, original in checkpoint["compressed_artifacts"].items():
        restored = decoded["compressed_artifacts"][name]
        for key in ("sparse_indices", "sparse_values", "d_awt"):
            if original.get(key) is not None and not torch.equal(original[key], restored[key]):
                raise AssertionError(f"Lossless component changed: {name}/{key}")
        pairs = list(zip(original["cores"], restored["cores"]))
        pairs += [(original[key], restored[key]) for key in ("A_bridge", "B_bridge") if original.get(key) is not None]
        relative = [float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)) for a, b in pairs]
        if max(relative) > 0.001:
            raise AssertionError(f"Factor packing error exceeds diagnostic gate: {name}: {relative}")
        errors[name] = relative
    decoded["total_kb"] = tensor_payload_bytes(decoded) / 1024
    roundtrip = args.output_dir / "roundtrip"
    roundtrip.mkdir()
    torch.save(decoded, roundtrip / f"compressed_layer_{checkpoint['layer_idx']}.pt")
    report = {"status": "COMPONENT_ROUNDTRIP_ONLY", "full_model_quality_verified": False,
              "input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "source_fingerprint": source_files_fingerprint([Path(__file__), ROOT / "nox/packing.py"]),
              "original_checkpoint_bytes": args.input.stat().st_size,
              "packed_checkpoint_bytes": path.stat().st_size,
              "factor_relative_errors": errors,
              "packed_vs_original_ratio": args.input.stat().st_size / path.stat().st_size}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
