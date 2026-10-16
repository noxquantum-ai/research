#!/usr/bin/env python3
"""Compile one high-fidelity configuration from dense activations with zero healing."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import build_real_token_calibration_data
from nox.artifacts import fingerprint, source_files_fingerprint
from nox.block_compiler import compile_transformer_block
from nox.compiler import reconstruct_from_artifact
from nox.sequential_compiler import forward_transformer_block, get_tier_config
from nox.utils import compute_metrics


class PinnedWeights:
    """Read tensors from the same immutable model snapshot used by HF evaluation."""
    def __init__(self, model_id, revision):
        self.model_id, self.revision = model_id, revision
        self.paths = {}
        index_path = self.download("model.safetensors.index.json")
        self.index = json.loads(Path(index_path).read_text())["weight_map"]

    def download(self, filename):
        if filename not in self.paths:
            self.paths[filename] = hf_hub_download(self.model_id, filename=filename, revision=self.revision)
        return self.paths[filename]

    def tensor(self, name):
        with safe_open(self.download(self.index[name]), framework="pt", device="cpu") as shard:
            return shard.get_tensor(name).float()

    def layer(self, index, device):
        prefix = f"model.layers.{index}."
        names = {name: "self_attn." + name + ".weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
        names.update({name: "mlp." + name + ".weight" for name in ("gate_proj", "up_proj", "down_proj")})
        names.update({name: name + ".weight" for name in ("input_layernorm", "post_attention_layernorm")})
        return {name: self.tensor(prefix + suffix).to(device) for name, suffix in names.items()}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-layer", type=int, default=15)
    parser.add_argument("--end-layer", type=int, default=15)
    parser.add_argument("--n-calib", type=int, default=256)
    parser.add_argument("--n-validation", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--calibration-sampling", choices=("prefix", "uniform"), default="prefix")
    parser.add_argument("--validation-sampling", choices=("prefix", "uniform"), default="prefix")
    parser.add_argument("--diagonal-prior-strength", type=float, default=0.0)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
    if not 0 <= args.start_layer <= args.end_layer < config.num_hidden_layers:
        raise ValueError("Invalid layer interval")
    if args.seq_len < 2 or any(n < args.seq_len or n % args.seq_len for n in (args.n_calib, args.n_validation)):
        raise ValueError("Calibration/validation counts must be positive multiples of seq-len")
    if list(args.output.glob("compressed_layer_*.pt")) or (args.output / "model_manifest.json").exists():
        raise FileExistsError("Use a fresh output directory; checkpoints will not be overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    weights = PinnedWeights(args.model_id, args.model_revision)
    embedding = weights.tensor("model.embed_tokens.weight")
    train, validation, data = build_real_token_calibration_data(
        embedding, args.model_id, args.n_calib, args.n_validation,
        args.seq_len, revision=args.model_revision,
        calibration_sampling=args.calibration_sampling, validation_sampling=args.validation_sampling,
    )
    del embedding
    train, validation = train.to(args.device), validation.to(args.device)
    forward_options = dict(n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                           seq_len=args.seq_len, rms_norm_eps=config.rms_norm_eps)
    compression = get_tier_config("high_fidelity_10x")
    compression["diagonal_prior_strength"] = args.diagonal_prior_strength
    identity = {
        "model_id": args.model_id, "model_revision": args.model_revision,
        "compression_path": "no_healing_independent", "config": compression,
        "forward_options": forward_options, "input_data": data,
        "layers": list(range(args.start_layer, args.end_layer + 1)),
        "source_fingerprint": source_files_fingerprint([
            Path(__file__), ROOT / "nox/compiler.py", ROOT / "nox/block_compiler.py",
            ROOT / "nox/sequential_compiler.py", ROOT / "nox/data.py", ROOT / "nox/utils.py",
        ]),
    }
    manifest = {"run_identity": identity, "run_fingerprint": fingerprint(identity),
                "tier": "high_fidelity_10x", "compression_path": "no_healing_independent",
                "rms_norm_eps": config.rms_norm_eps, "history": [], "layers_compressed": [],
                "serialized_checkpoint_bytes": 0, "checkpoint_file_sizes": {}}
    compressed_validation = None
    for index in range(args.end_layer + 1):
        started = time.monotonic()
        dense = weights.layer(index, args.device)
        train_next, _, _ = forward_transformer_block(dense, train, **forward_options)
        validation_next, _, _ = forward_transformer_block(dense, validation, **forward_options)
        if index >= args.start_layer:
            if compressed_validation is None:
                compressed_validation = validation.clone()
            torch.save({"X_test": validation.cpu(), "rms_norm_eps": config.rms_norm_eps}, args.output / f"dense_input_layer_{index}.pt")
            torch.save(train.cpu(), args.output / f"calibration_input_layer_{index}.pt")
            compiled = compile_transformer_block(
                dense, train, X_calib_dense=train, causal_healing=False,
                ranks=compression["ranks"], r_bridge=compression["r_bridge"],
                n_outliers=compression["n_outliers"], return_artifacts=True,
                diagonal_prior_strength=args.diagonal_prior_strength,
                verbose=True, **forward_options,
            )
            artifact = {
                "layer_idx": index, "compression_path": "no_healing_independent",
                "rms_norm_eps": config.rms_norm_eps, "run_fingerprint": manifest["run_fingerprint"],
                "compressed_artifacts": compiled["compressed_artifacts"],
                "input_layernorm": dense["input_layernorm"].cpu(),
                "post_attention_layernorm": dense["post_attention_layernorm"].cpu(),
                "total_kb": compiled["total_kb"],
            }
            path = args.output / f"compressed_layer_{index}.pt"
            torch.save(artifact, path)
            reconstructed = dict(dense)
            reconstructed.update({name: reconstruct_from_artifact(value, device=args.device)
                                  for name, value in artifact["compressed_artifacts"].items()})
            same_input, _, _ = forward_transformer_block(reconstructed, validation, **forward_options)
            compressed_validation, _, _ = forward_transformer_block(reconstructed, compressed_validation, **forward_options)
            train_output, _, _ = forward_transformer_block(reconstructed, train, **forward_options)
            record = {
                "layer": index, "same_input_validation": compute_metrics(validation_next, same_input),
                "cascade_validation": compute_metrics(validation_next, compressed_validation),
                "calibration": compute_metrics(train_next, train_output),
                "serialized_bytes": path.stat().st_size,
                "dense_fp16_tensor_bytes": sum(t.numel() * 2 for t in dense.values()),
                "seconds": time.monotonic() - started,
            }
            manifest["history"].append(record)
            manifest["layers_compressed"].append(index)
            manifest["checkpoint_file_sizes"][path.name] = path.stat().st_size
            manifest["serialized_checkpoint_bytes"] += path.stat().st_size
            (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
            print(json.dumps(record), flush=True)
            del compiled, reconstructed, same_input, train_output, artifact
        train, validation = train_next, validation_next
        del dense
        gc.collect()
        torch.cuda.empty_cache()
        print(f"layer={index} elapsed={time.monotonic() - started:.2f}s", flush=True)
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
