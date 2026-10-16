#!/usr/bin/env python3
"""Compile one high-fidelity configuration with zero healing."""

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
from nox.gauge import balanced_attention_gauge
from nox.mlp_gauge import balanced_swiglu_gauge, entanglement_ordered_swiglu_gauge
from nox.sequential_compiler import forward_transformer_block, get_tier_config
from nox.utils import compute_metrics
from nox.tensor_order import IDENTITY_SITE_ORDER, validate_site_order


def select_calibration_activations(mode, dense_activations, compressed_activations):
    """Select the inputs used to fit this layer's codec, without changing targets."""
    if mode == "dense":
        return dense_activations
    if mode == "compressed_prefix":
        return dense_activations if compressed_activations is None else compressed_activations
    raise ValueError(f"Unknown calibration activation mode: {mode}")


def select_reference_activations(construction_mode, dense_activations, calibration_activations):
    """Select clean targets for trajectory-aware initial construction."""
    if construction_mode == "independent":
        return calibration_activations
    if construction_mode in ("trajectory_aware_mlp", "trajectory_aware_all"):
        return dense_activations
    raise ValueError(f"Unknown construction mode: {construction_mode}")


def resume_identity_compatible(previous, current):
    """Require identical science inputs while allowing a recorded source repair."""
    previous = dict(previous)
    current = dict(current)
    previous.pop("source_fingerprint", None)
    current.pop("source_fingerprint", None)
    return json.dumps(previous, sort_keys=True) == json.dumps(current, sort_keys=True)


def parse_site_order(value):
    try:
        return validate_site_order(tuple(int(part.strip()) for part in value.split(",")))
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("site order must be four comma-separated digits, e.g. 2,0,3,1") from exc


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
    parser.add_argument(
        "--calibration-activation-mode", choices=("dense", "compressed_prefix"), default="dense",
        help=("Use dense layer inputs (independent control) or propagate calibration inputs "
              "through already-written compressed layers. Both modes keep causal_healing=False."),
    )
    parser.add_argument(
        "--construction-mode",
        choices=("independent", "trajectory_aware_mlp", "trajectory_aware_all"),
        default="independent",
        help=("Initial zero-healing construction objective. Trajectory-aware modes regress from "
              "the actual compressed prefix toward its corresponding clean dense trajectory."),
    )
    parser.add_argument("--diagonal-prior-strength", type=float, default=0.0)
    parser.add_argument("--attention-gauge", choices=("none", "balanced"), default="none",
                        help="Apply an exact paired attention reparameterization before compression")
    parser.add_argument(
        "--mlp-gauge", choices=("none", "balanced_up_down", "entanglement_ordered"), default="none",
        help=("Apply an exact joint SwiGLU gauge before initial compression. The entanglement-ordered "
              "variant maps calibration-correlated neurons onto the fixed MPO tensor grid."),
    )
    parser.add_argument(
        "--site-order", type=parse_site_order, default=IDENTITY_SITE_ORDER,
        help=("Shared four-site MPO order for every projection in the layer stack; "
              "the reindexing is inverted in the saved dense-weight reconstruction"),
    )
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse fully written layer artifacts after validating all non-source run identity fields",
    )
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
    if not 0 <= args.start_layer <= args.end_layer < config.num_hidden_layers:
        raise ValueError("Invalid layer interval")
    if args.seq_len < 2 or any(n < args.seq_len or n % args.seq_len for n in (args.n_calib, args.n_validation)):
        raise ValueError("Calibration/validation counts must be positive multiples of seq-len")
    if args.construction_mode != "independent" and args.calibration_activation_mode != "compressed_prefix":
        raise ValueError("Trajectory-aware construction requires --calibration-activation-mode compressed_prefix")
    has_existing = bool(list(args.output.glob("compressed_layer_*.pt"))) or (args.output / "model_manifest.json").exists()
    if has_existing and not args.resume:
        raise FileExistsError("Use a fresh output directory; checkpoints will not be overwritten")
    if args.resume and not (args.output / "model_manifest.json").exists():
        raise FileNotFoundError("--resume requires an existing model_manifest.json")
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
    compression_path = (
        "no_healing_independent"
        if args.calibration_activation_mode == "dense"
        else (
            "no_healing_prefix_calibrated"
            if args.construction_mode == "independent"
            else f"zero_healing_{args.construction_mode}"
        )
    )
    identity = {
        "model_id": args.model_id, "model_revision": args.model_revision,
        "compression_path": compression_path,
        "calibration_activation_mode": args.calibration_activation_mode,
        "construction_mode": args.construction_mode,
        "config": compression,
        "attention_gauge": args.attention_gauge,
        "mlp_gauge": args.mlp_gauge,
        "mpo_site_order": args.site_order,
        "forward_options": forward_options, "input_data": data,
        "layers": list(range(args.start_layer, args.end_layer + 1)),
        "source_fingerprint": source_files_fingerprint([
            Path(__file__), ROOT / "nox/compiler.py", ROOT / "nox/block_compiler.py", ROOT / "nox/gauge.py",
            ROOT / "nox/mlp_gauge.py", ROOT / "nox/sequential_compiler.py", ROOT / "nox/data.py", ROOT / "nox/utils.py",
            ROOT / "nox/tensor_order.py",
        ]),
    }
    if args.resume:
        manifest = json.loads((args.output / "model_manifest.json").read_text())
        if not resume_identity_compatible(manifest["run_identity"], identity):
            raise ValueError("Existing checkpoint identity does not match requested resumed run")
        manifest.setdefault("resume_events", []).append({
            "previous_source_fingerprint": manifest["run_identity"].get("source_fingerprint"),
            "resumed_source_fingerprint": identity["source_fingerprint"],
            "reason": "resume after completed layer artifact; source change is recorded per layer",
        })
        (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    else:
        manifest = {"run_identity": identity, "run_fingerprint": fingerprint(identity),
                    "tier": "high_fidelity_10x", "compression_path": compression_path,
                    "calibration_activation_mode": args.calibration_activation_mode,
                    "construction_mode": args.construction_mode,
                    "mpo_site_order": args.site_order,
                    "rms_norm_eps": config.rms_norm_eps, "history": [], "layers_compressed": [],
                    "serialized_checkpoint_bytes": 0, "checkpoint_file_sizes": {}}
    compressed_validation = None
    compressed_calibration = None
    for index in range(args.end_layer + 1):
        started = time.monotonic()
        dense = weights.layer(index, args.device)
        gauge_metadata = None
        mlp_gauge_metadata = None
        if index == args.start_layer and args.attention_gauge == "balanced":
            dense, gauge_metadata = balanced_attention_gauge(
                dense, config.num_attention_heads, config.num_key_value_heads,
            )
            train_next, _, _ = forward_transformer_block(dense, train, **forward_options)
            validation_next, _, _ = forward_transformer_block(dense, validation, **forward_options)
            # Verify the exact symmetry against the original pinned dense block
            # before any lossy compression is attempted.
            original = weights.layer(index, args.device)
            original_train, _, _ = forward_transformer_block(original, train, **forward_options)
            original_validation, _, _ = forward_transformer_block(original, validation, **forward_options)
            gauge_metadata["train_parity_relative_error"] = compute_metrics(original_train, train_next)["relative_error"]
            gauge_metadata["validation_parity_relative_error"] = compute_metrics(original_validation, validation_next)["relative_error"]
            if max(gauge_metadata["train_parity_relative_error"],
                   gauge_metadata["validation_parity_relative_error"]) > 2e-5:
                raise RuntimeError(f"Exact attention gauge parity failed: {gauge_metadata}")
            del original, original_train, original_validation
        else:
            train_next, _, _ = forward_transformer_block(dense, train, **forward_options)
            validation_next, _, _ = forward_transformer_block(dense, validation, **forward_options)
        if index >= args.start_layer and args.mlp_gauge != "none":
            reference_train_next = train_next
            reference_validation_next = validation_next
            if args.mlp_gauge == "balanced_up_down":
                dense, mlp_gauge_metadata = balanced_swiglu_gauge(dense)
            else:
                gauge_input = select_calibration_activations(
                    args.calibration_activation_mode, train, compressed_calibration,
                )
                dense, mlp_gauge_metadata = entanglement_ordered_swiglu_gauge(
                    dense, gauge_input,
                )
            train_next, _, _ = forward_transformer_block(dense, train, **forward_options)
            validation_next, _, _ = forward_transformer_block(dense, validation, **forward_options)
            mlp_gauge_metadata["train_parity_relative_error"] = compute_metrics(
                reference_train_next, train_next,
            )["relative_error"]
            mlp_gauge_metadata["validation_parity_relative_error"] = compute_metrics(
                reference_validation_next, validation_next,
            )["relative_error"]
            if max(
                mlp_gauge_metadata["train_parity_relative_error"],
                mlp_gauge_metadata["validation_parity_relative_error"],
            ) > 2e-5:
                raise RuntimeError(f"Exact MLP gauge parity failed: {mlp_gauge_metadata}")
            del reference_train_next, reference_validation_next
        if index >= args.start_layer:
            if compressed_validation is None:
                compressed_validation = validation.clone()
            if args.calibration_activation_mode == "compressed_prefix" and compressed_calibration is None:
                compressed_calibration = train.clone()
            calibration_input = select_calibration_activations(
                args.calibration_activation_mode, train, compressed_calibration,
            )
            validation_calibration_input = select_calibration_activations(
                args.calibration_activation_mode, validation, compressed_validation,
            )
            reference_input = select_reference_activations(
                args.construction_mode, train, calibration_input,
            )
            path = args.output / f"compressed_layer_{index}.pt"
            if args.resume and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                if artifact.get("layer_idx") != index or artifact.get("run_fingerprint") != manifest["run_fingerprint"]:
                    raise ValueError(f"Incompatible resumed artifact: {path}")
                reconstructed = dict(dense)
                reconstructed.update({
                    name: reconstruct_from_artifact(value, device=args.device)
                    for name, value in artifact["compressed_artifacts"].items()
                })
                compressed_validation, _, _ = forward_transformer_block(
                    reconstructed, compressed_validation, **forward_options,
                )
                if args.calibration_activation_mode == "compressed_prefix":
                    compressed_calibration, _, _ = forward_transformer_block(
                        reconstructed, compressed_calibration, **forward_options,
                    )
                train, validation = train_next, validation_next
                del dense, reconstructed, artifact
                gc.collect()
                torch.cuda.empty_cache()
                print(f"layer={index} resumed_from={path}", flush=True)
                continue
            torch.save({"X_test": validation.cpu(), "rms_norm_eps": config.rms_norm_eps}, args.output / f"dense_input_layer_{index}.pt")
            torch.save(calibration_input.cpu(), args.output / f"calibration_input_layer_{index}.pt")
            torch.save(validation_calibration_input.cpu(), args.output / f"layer_input_validation_layer_{index}.pt")
            compiled = compile_transformer_block(
                dense, calibration_input, X_calib_dense=reference_input, causal_healing=False,
                ranks=compression["ranks"], r_bridge=compression["r_bridge"],
                n_outliers=compression["n_outliers"], return_artifacts=True,
                diagonal_prior_strength=args.diagonal_prior_strength,
                trajectory_mode=("none" if args.construction_mode == "independent"
                                 else args.construction_mode.removeprefix("trajectory_aware_")),
                verbose=True, site_order=args.site_order, **forward_options,
            )
            artifact = {
                "layer_idx": index, "compression_path": compression_path,
                "construction_mode": args.construction_mode,
                "source_fingerprint": identity["source_fingerprint"],
                "mpo_site_order": args.site_order,
                "attention_gauge": gauge_metadata,
                "mlp_gauge": mlp_gauge_metadata,
                "rms_norm_eps": config.rms_norm_eps, "run_fingerprint": manifest["run_fingerprint"],
                "compressed_artifacts": compiled["compressed_artifacts"],
                "input_layernorm": dense["input_layernorm"].cpu(),
                "post_attention_layernorm": dense["post_attention_layernorm"].cpu(),
                "total_kb": compiled["total_kb"],
            }
            torch.save(artifact, path)
            reconstructed = dict(dense)
            reconstructed.update({name: reconstruct_from_artifact(value, device=args.device)
                                  for name, value in artifact["compressed_artifacts"].items()})
            dense_same_input, _, _ = forward_transformer_block(
                dense, validation_calibration_input, **forward_options,
            )
            same_input, _, _ = forward_transformer_block(
                reconstructed, validation_calibration_input, **forward_options,
            )
            compressed_validation, _, _ = forward_transformer_block(reconstructed, compressed_validation, **forward_options)
            calibration_dense_output, _, _ = forward_transformer_block(
                dense, reference_input, **forward_options,
            )
            train_output, _, _ = forward_transformer_block(
                reconstructed, calibration_input, **forward_options,
            )
            record = {
                "layer": index, "calibration_activation_mode": args.calibration_activation_mode,
                "construction_mode": args.construction_mode,
                "mpo_site_order": args.site_order,
                "same_input_validation": compute_metrics(dense_same_input, same_input),
                "cascade_validation": compute_metrics(validation_next, compressed_validation),
                "calibration": compute_metrics(calibration_dense_output, train_output),
                "attention_gauge": gauge_metadata,
                "mlp_gauge": mlp_gauge_metadata,
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
            if args.calibration_activation_mode == "compressed_prefix":
                compressed_calibration, _, _ = forward_transformer_block(
                    reconstructed, compressed_calibration, **forward_options,
                )
            del compiled, reconstructed, same_input, train_output, artifact
        train, validation = train_next, validation_next
        del dense
        gc.collect()
        torch.cuda.empty_cache()
        print(f"layer={index} elapsed={time.monotonic() - started:.2f}s", flush=True)
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
