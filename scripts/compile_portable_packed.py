#!/usr/bin/env python3
"""Compile an architecture-neutral activation-aware packed NEC package."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.data import load_wikitext2_raw_split
from nox.model_adapter import inventory_model
from nox.portable_quant import ActivationRMSCollector, artifact_filename, encode_target
from nox.residual_codec import codec_payload_bytes


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--activation-alpha", type=float, default=0.5)
    parser.add_argument("--calibration-tokens", type=int, default=1024)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--calibration-sampling", choices=("prefix", "uniform"), default="uniform")
    parser.add_argument("--include-embeddings", action="store_true")
    parser.add_argument("--include-output-head", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.calibration_tokens < args.sequence_length or (
        args.calibration_tokens % args.sequence_length
    ):
        raise ValueError("calibration tokens must be a positive sequence-length multiple")
    if not 0 <= args.activation_alpha <= 1:
        raise ValueError("activation alpha must lie in [0, 1]")

    staging = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    staging.mkdir(parents=True)
    operators_dir = staging / "operators"
    operators_dir.mkdir()
    started = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if str(args.device).startswith("cuda") else None,
            trust_remote_code=args.trust_remote_code,
        )
        if not str(args.device).startswith("cuda"):
            model.to(args.device)
        model.eval()
        inventory = inventory_model(model)
        roles = {"block_linear"}
        if args.include_embeddings:
            roles.update({"embedding", "input_embedding", "tied_input_output_embedding"})
        if args.include_output_head:
            roles.update({"output_head", "tied_input_output_embedding"})
        candidates = []
        selected_groups: set[int] = set()
        for target in inventory.targets:
            if target.role not in roles or target.tied_group in selected_groups:
                continue
            selected_groups.add(target.tied_group)
            candidates.append(target)
        candidates = tuple(candidates)
        if not candidates:
            raise RuntimeError("model discovery produced no selected compression targets")

        text, calibration_source = load_wikitext2_raw_split("train")
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if ids.shape[1] < args.calibration_tokens:
            raise ValueError("calibration corpus is shorter than requested token count")
        window_count = args.calibration_tokens // args.sequence_length
        if args.calibration_sampling == "prefix":
            starts = [index * args.sequence_length for index in range(window_count)]
        else:
            maximum_start = int(ids.shape[1] - args.sequence_length)
            starts = [
                round(index * maximum_start / max(window_count - 1, 1))
                for index in range(window_count)
            ]
        with ActivationRMSCollector(model, candidates) as collector:
            for begin in starts:
                batch = ids[:, begin:begin + args.sequence_length].to(model.device)
                model(batch, use_cache=False)
        del ids

        # Remove every state-dict alias of a selected shared parameter.  The
        # operator itself is encoded once and PyTorch's tied Parameter object
        # propagates the injected value to all aliases.
        compressed_names = {
            f"{target.name}.weight"
            for target in inventory.targets
            if target.tied_group in selected_groups
        }
        compressed_records = []
        for index, target in enumerate(candidates, start=1):
            rms = collector.rms(target.name)
            artifact = encode_target(
                model,
                target,
                bits=args.bits,
                group_size=args.group_size,
                activation_rms=rms,
                activation_alpha=args.activation_alpha,
            )
            filename = artifact_filename(target.name)
            destination = operators_dir / filename
            torch.save(artifact, destination)
            compressed_records.append({
                "name": target.name,
                "role": target.role,
                "shape": list(target.shape),
                "artifact_file": filename,
                "payload_bytes": codec_payload_bytes(artifact),
                "serialized_bytes": destination.stat().st_size,
                "activation_samples": collector.counts.get(target.name, 0),
                "aliases": [
                    alias.name for alias in inventory.targets
                    if alias.tied_group == target.tied_group and alias.name != target.name
                ],
            })
            if index == 1 or index % 16 == 0 or index == len(candidates):
                print(f"encoded modules={index}/{len(candidates)}", flush=True)

        base_state = {
            name: value.detach().to("cpu")
            for name, value in model.state_dict().items()
            if name not in compressed_names
        }
        torch.save(base_state, staging / "base_state.pt")
        del base_state, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metadata_dir = staging / "hf_metadata"
        metadata_dir.mkdir()
        tokenizer.save_pretrained(metadata_dir)
        # AutoConfig is already represented by model config in the source;
        # reload it without retaining the released model.
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            trust_remote_code=args.trust_remote_code,
        )
        config.save_pretrained(metadata_dir)

        manifest = {
            "schema": "nec_portable_packed_v1",
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "model_class": inventory.model_class,
            "model_type": inventory.model_type,
            "block_container": inventory.block_container,
            "block_count": inventory.block_count,
            "codec": {
                "type": "groupwise_packed_v1",
                "bits": args.bits,
                "group_size": args.group_size,
                "activation_alpha": args.activation_alpha,
            },
            "calibration": {
                **calibration_source,
                "tokens": args.calibration_tokens,
                "sequence_length": args.sequence_length,
                "sampling": args.calibration_sampling,
                "window_starts": starts,
            },
            "dense_parameter_payload_bytes": inventory.total_unique_parameter_bytes,
            "compressed_modules": compressed_records,
            "dense_base_file_bytes": (staging / "base_state.pt").stat().st_size,
            "construction_seconds_before_manifest": time.monotonic() - started,
            "storage_note": (
                "The package contains packed selected operators, every uncompressed state tensor, "
                "configuration, tokenizer, and this manifest. Runtime kernels are accounted separately."
            ),
        }
        (staging / "nec_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["serialized_package_bytes"] = tree_bytes(staging)
        manifest["payload_compression_ratio"] = (
            inventory.total_unique_parameter_bytes / manifest["serialized_package_bytes"]
        )
        (staging / "nec_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(staging, args.output)
        print(json.dumps({
            "output": str(args.output),
            "compressed_modules": len(compressed_records),
            "serialized_package_bytes": tree_bytes(args.output),
            "dense_parameter_payload_bytes": inventory.total_unique_parameter_bytes,
            "payload_compression_ratio": manifest["payload_compression_ratio"],
            "seconds": time.monotonic() - started,
        }, indent=2), flush=True)
    except BaseException:
        print(f"partial package preserved at {staging}", flush=True)
        raise


if __name__ == "__main__":
    main()
