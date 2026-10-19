#!/usr/bin/env python3
"""Compile one high-fidelity configuration with zero healing."""

import argparse
import gc
import hashlib
import io
import json
import math
import os
import resource
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
from nox.allocation import projection_budgets
from nox.block_compiler import compile_transformer_block
from nox.compiler import reconstruct_from_artifact
from nox.gauge import balanced_attention_gauge
from nox.mlp_gauge import balanced_swiglu_gauge, entanglement_ordered_swiglu_gauge
from nox.sequential_compiler import (
    compile_independent_transformer_block_streaming,
    forward_transformer_block,
    forward_transformer_block_chunked,
    get_tier_config,
)
from nox.utils import compute_metrics
from nox.tensor_order import IDENTITY_SITE_ORDER, validate_site_order
from nox.defect_bridge import fit_hybrid_defect_bridge, bridge_tensor_bytes
from nox.evaluation import (
    MODEL_ID as NEC_MODEL_ID,
    MODEL_REVISION as NEC_MODEL_REVISION,
    validate_llama2_config,
)
from nox.pees import (
    DEFAULT_PROJECTIONS as PEES_DEFAULT_PROJECTIONS,
    append_shield_to_projection_artifact,
    fit_propagated_eigenspace_shield,
    layer_projection_trajectories,
)


def atomic_torch_save(value, path):
    """Write a checkpoint atomically so interrupted writes are never resumable."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    if construction_mode in (
        "trajectory_aware_mlp", "trajectory_aware_all", "trajectory_aware_product",
    ):
        return dense_activations
    raise ValueError(f"Unknown construction mode: {construction_mode}")


def resume_identity_compatible(previous, current):
    """Allow source repairs and extension of an otherwise identical layer prefix."""
    previous = dict(previous)
    current = dict(current)
    previous.pop("source_fingerprint", None)
    current.pop("source_fingerprint", None)
    previous_layers = previous.pop("layers", None)
    current_layers = current.pop("layers", None)
    if previous_layers is None or current_layers is None:
        return False
    previous_layers = list(previous_layers)
    current_layers = list(current_layers)
    if current_layers[:len(previous_layers)] != previous_layers:
        return False
    # These fields were added after older run records. Missing fields map only
    # to their historical defaults; they must not silently adopt a new method.
    legacy_defaults = {
        "mlp_objective": "independent",
        "mlp_product_ridge_ratio": 0.05,
        "budget_allocation": "uniform",
        "defect_bridge_mode": "none",
        "defect_bridge_linear_rank": 512,
        "defect_bridge_memory_size": 512,
        "defect_bridge_memory_top_k": 8,
        "defect_bridge_ridge_ratio": 0.01,
        "max_checkpoint_multiplier": 1.25,
        "pees_rank": 0,
        "pees_projections": list(PEES_DEFAULT_PROJECTIONS),
        "pees_tokens": 4096,
        "pees_ridge_ratio": 0.01,
        "representation": "mpo4",
        "operator_schmidt_weight_mix": 1.0,
        "operator_schmidt_activation_mix": 1.0,
    }
    for identity in (previous, current):
        for key, value in legacy_defaults.items():
            identity.setdefault(key, value)
    previous.setdefault("projection_budgets", current.get("projection_budgets"))
    current.setdefault("projection_budgets", previous.get("projection_budgets"))
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
        "--calibration-backend",
        choices=("full_matrix", "streaming_statistics"),
        default="full_matrix",
        help=("Retain the reference full-matrix SVD path or accumulate exact projection "
              "normal-equation statistics over complete sequence windows."),
    )
    parser.add_argument(
        "--calibration-windows-per-chunk", type=int, default=4,
        help="Complete seq-len windows per forward/statistics chunk in streaming mode.",
    )
    parser.add_argument(
        "--statistics-dtype", choices=("float32", "float64"), default="float32",
        help="Accumulation dtype for streamed covariance and cross-covariance matrices.",
    )
    parser.add_argument(
        "--calibration-activation-mode", choices=("dense", "compressed_prefix"), default="dense",
        help=("Use dense layer inputs (independent control) or propagate calibration inputs "
              "through already-written compressed layers. Both modes keep causal_healing=False."),
    )
    parser.add_argument(
        "--construction-mode",
        choices=(
            "independent", "trajectory_aware_mlp", "trajectory_aware_all",
            "trajectory_aware_product",
        ),
        default="independent",
        help=("Initial zero-healing construction objective. Trajectory-aware modes regress from "
              "the actual compressed prefix toward its corresponding clean dense trajectory."),
    )
    parser.add_argument("--diagonal-prior-strength", type=float, default=0.0)
    parser.add_argument(
        "--mpo-sweeps", type=int, default=2,
        help="Number of activation-space local MPO sweeps; 0 evaluates the TT-SVD initialization.",
    )
    parser.add_argument(
        "--mpo-alpha-awt", type=float, default=0.5,
        help="Diagonal activation-scaling exponent used before MPO TT-SVD.",
    )
    parser.add_argument(
        "--representation", choices=("mpo4", "operator_schmidt"), default="mpo4",
        help=("Core tensor-network representation. operator_schmidt screens projection-aware "
              "two-site MPO cuts at the legacy four-site core coefficient budget."),
    )
    parser.add_argument("--operator-schmidt-weight-mix", type=float, default=1.0)
    parser.add_argument("--operator-schmidt-activation-mix", type=float, default=1.0)
    parser.add_argument("--attention-gauge", choices=("none", "balanced"), default="none",
                        help="Apply an exact paired attention reparameterization before compression")
    parser.add_argument(
        "--mlp-gauge", choices=("none", "balanced_up_down", "entanglement_ordered"), default="none",
        help=("Apply an exact joint SwiGLU gauge before initial compression. The entanglement-ordered "
              "variant maps calibration-correlated neurons onto the fixed MPO tensor grid."),
    )
    parser.add_argument(
        "--mlp-objective", choices=("independent", "product_aware"), default="independent",
        help=("Fit gate/up independently or fit Up to the closed-form ridge-compensated target "
              "induced by the initially compressed gate."),
    )
    parser.add_argument("--mlp-product-ridge-ratio", type=float, default=0.05)
    parser.add_argument(
        "--budget-allocation", choices=("uniform", "interaction_aware"), default="uniform",
        help="Keep uniform projection budgets or exchange Q/O sparse entries for Gate/Down at fixed total bytes",
    )
    parser.add_argument(
        "--defect-bridge-mode", choices=("none", "hybrid"), default="none",
        help=("Fit a rank-limited global correction plus calibration-state residual memory "
              "during initial artifact construction; no checkpoint is modified afterward."),
    )
    parser.add_argument("--defect-bridge-linear-rank", type=int, default=512)
    parser.add_argument("--defect-bridge-memory-size", type=int, default=512)
    parser.add_argument("--defect-bridge-memory-top-k", type=int, default=8)
    parser.add_argument("--defect-bridge-ridge-ratio", type=float, default=0.01)
    parser.add_argument(
        "--pees-rank", type=int, default=0,
        help="Add an initial-construction propagated eigenspace sidecar at this per-projection rank (0 disables PEES)",
    )
    parser.add_argument(
        "--pees-projections", default=",".join(PEES_DEFAULT_PROJECTIONS),
        help="Comma-separated subset of v_proj,o_proj,gate_proj,down_proj",
    )
    parser.add_argument("--pees-tokens", type=int, default=4096)
    parser.add_argument("--pees-ridge-ratio", type=float, default=0.01)
    parser.add_argument("--max-checkpoint-multiplier", type=float, default=1.25)
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
    if args.model_id != NEC_MODEL_ID or args.model_revision != NEC_MODEL_REVISION:
        raise ValueError("This High-Fidelity production compiler run requires the pinned Llama-2-7B snapshot")
    validate_llama2_config(config)
    if not 0 <= args.start_layer <= args.end_layer < config.num_hidden_layers:
        raise ValueError("Invalid layer interval")
    if args.seq_len < 2 or any(n < args.seq_len or n % args.seq_len for n in (args.n_calib, args.n_validation)):
        raise ValueError("Calibration/validation counts must be positive multiples of seq-len")
    if args.calibration_windows_per_chunk <= 0:
        raise ValueError("--calibration-windows-per-chunk must be positive")
    if args.construction_mode != "independent" and args.calibration_activation_mode != "compressed_prefix":
        raise ValueError("Trajectory-aware construction requires --calibration-activation-mode compressed_prefix")
    if args.mlp_product_ridge_ratio <= 0:
        raise ValueError("--mlp-product-ridge-ratio must be positive")
    if args.mpo_sweeps < 0:
        raise ValueError("--mpo-sweeps must be nonnegative")
    if not math.isfinite(args.mpo_alpha_awt):
        raise ValueError("--mpo-alpha-awt must be finite")
    if (args.operator_schmidt_weight_mix < 0 or args.operator_schmidt_activation_mix < 0
            or args.operator_schmidt_weight_mix + args.operator_schmidt_activation_mix <= 0):
        raise ValueError("Operator-Schmidt objective weights must be nonnegative and not both zero")
    if args.representation == "operator_schmidt" and args.site_order != IDENTITY_SITE_ORDER:
        raise ValueError("--site-order applies only to the legacy four-site MPO representation")
    if args.representation != "mpo4" and (
        args.mpo_sweeps != 2 or args.mpo_alpha_awt != 0.5
    ):
        raise ValueError("MPO sweep and AWT options apply only to --representation mpo4")
    if args.defect_bridge_linear_rank < 0 or args.defect_bridge_memory_size < 0:
        raise ValueError("Defect bridge rank and memory size must be nonnegative")
    if args.defect_bridge_mode == "hybrid" and args.defect_bridge_linear_rank == 0:
        raise ValueError("Hybrid defect bridge requires a positive linear rank")
    if args.defect_bridge_memory_top_k <= 0 or args.defect_bridge_ridge_ratio < 0:
        raise ValueError("Invalid defect bridge kernel or ridge configuration")
    pees_projections = tuple(
        name.strip() for name in args.pees_projections.split(",") if name.strip()
    )
    if args.pees_rank < 0 or args.pees_tokens < args.seq_len or args.pees_ridge_ratio < 0:
        raise ValueError("Invalid PEES rank, token count, or ridge ratio")
    if args.pees_rank > 0:
        if args.defect_bridge_mode != "none":
            raise ValueError("PEES and the block defect bridge are separate byte-budget experiments")
        if not pees_projections or len(set(pees_projections)) != len(pees_projections):
            raise ValueError("PEES requires a nonempty projection list without duplicates")
        if set(pees_projections) - set(PEES_DEFAULT_PROJECTIONS):
            raise ValueError(f"Unsupported PEES projections: {sorted(set(pees_projections) - set(PEES_DEFAULT_PROJECTIONS))}")
    elif pees_projections != PEES_DEFAULT_PROJECTIONS:
        raise ValueError("--pees-projections is only meaningful when --pees-rank is positive")
    if args.max_checkpoint_multiplier < 1.0:
        raise ValueError("--max-checkpoint-multiplier must be at least 1")
    if args.calibration_backend == "streaming_statistics":
        unsupported = []
        if args.calibration_activation_mode != "dense":
            unsupported.append("compressed-prefix activations")
        if args.construction_mode != "independent":
            unsupported.append("trajectory-aware construction")
        if args.representation != "mpo4":
            unsupported.append("non-MPO4 representation")
        if args.attention_gauge != "none" or args.mlp_gauge != "none":
            unsupported.append("pre-compression gauges")
        if args.mlp_objective != "independent":
            unsupported.append("product-aware MLP objective")
        if args.defect_bridge_mode != "none" or args.pees_rank > 0:
            unsupported.append("post-codec construction sidecars")
        if unsupported:
            raise ValueError(
                "Streaming statistics currently preserves only the active independent "
                f"zero-healing experiment; unsupported: {', '.join(unsupported)}"
            )
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
    if args.calibration_backend == "full_matrix":
        train, validation = train.to(args.device), validation.to(args.device)
    else:
        # Token-major state remains in host memory; only complete-window chunks
        # and the bounded covariance matrices enter the accelerator.
        train, validation = train.cpu(), validation.cpu()
    forward_options = dict(n_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads,
                           seq_len=args.seq_len, rms_norm_eps=config.rms_norm_eps)
    statistics_dtype = {
        "float32": torch.float32,
        "float64": torch.float64,
    }[args.statistics_dtype]

    def execute_block(block_weights, activations):
        if args.calibration_backend == "streaming_statistics":
            return forward_transformer_block_chunked(
                block_weights,
                activations,
                windows_per_chunk=args.calibration_windows_per_chunk,
                output_device="cpu",
                return_components=False,
                **forward_options,
            )
        return forward_transformer_block(block_weights, activations, **forward_options)
    compression = get_tier_config("high_fidelity_10x")
    compression["diagonal_prior_strength"] = args.diagonal_prior_strength
    compression["representation"] = args.representation
    compression["operator_schmidt_weight_mix"] = args.operator_schmidt_weight_mix
    compression["operator_schmidt_activation_mix"] = args.operator_schmidt_activation_mix
    compression["mpo_n_sweeps"] = args.mpo_sweeps
    compression["mpo_alpha_awt"] = args.mpo_alpha_awt
    per_projection_budgets = projection_budgets(
        args.budget_allocation, compression["r_bridge"], compression["n_outliers"],
    )
    compression_path = (
        "no_healing_independent"
        if args.calibration_activation_mode == "dense"
        else (
            "no_healing_prefix_calibrated"
            if args.construction_mode == "independent"
            else f"zero_healing_{args.construction_mode}"
        )
    )
    if args.defect_bridge_mode == "hybrid":
        compression_path = f"{compression_path}+hybrid_defect_bridge"
    if args.pees_rank > 0:
        compression_path = f"{compression_path}+pees_propagated_eigenspace"
    if args.representation == "operator_schmidt":
        compression_path = f"{compression_path}+operator_schmidt"
    identity = {
        "model_id": args.model_id, "model_revision": args.model_revision,
        "compression_path": compression_path,
        "calibration_activation_mode": args.calibration_activation_mode,
        "construction_mode": args.construction_mode,
        "config": compression,
        "attention_gauge": args.attention_gauge,
        "mlp_gauge": args.mlp_gauge,
        "mlp_objective": args.mlp_objective,
        "mlp_product_ridge_ratio": args.mlp_product_ridge_ratio,
        "budget_allocation": args.budget_allocation,
        "projection_budgets": per_projection_budgets,
        "representation": args.representation,
        "operator_schmidt_weight_mix": args.operator_schmidt_weight_mix,
        "operator_schmidt_activation_mix": args.operator_schmidt_activation_mix,
        "mpo_n_sweeps": args.mpo_sweeps,
        "mpo_alpha_awt": args.mpo_alpha_awt,
        "calibration_backend": args.calibration_backend,
        "calibration_windows_per_chunk": args.calibration_windows_per_chunk,
        "statistics_dtype": args.statistics_dtype,
        "defect_bridge_mode": args.defect_bridge_mode,
        "defect_bridge_linear_rank": args.defect_bridge_linear_rank,
        "defect_bridge_memory_size": args.defect_bridge_memory_size,
        "defect_bridge_memory_top_k": args.defect_bridge_memory_top_k,
        "defect_bridge_ridge_ratio": args.defect_bridge_ridge_ratio,
        "max_checkpoint_multiplier": args.max_checkpoint_multiplier,
        "pees_rank": args.pees_rank,
        "pees_projections": list(pees_projections if args.pees_rank > 0 else PEES_DEFAULT_PROJECTIONS),
        "pees_tokens": args.pees_tokens,
        "pees_ridge_ratio": args.pees_ridge_ratio,
        "mpo_site_order": args.site_order,
        "forward_options": forward_options, "input_data": data,
        "layers": list(range(args.start_layer, args.end_layer + 1)),
        "source_fingerprint": source_files_fingerprint([
            Path(__file__), ROOT / "nox/compiler.py", ROOT / "nox/block_compiler.py", ROOT / "nox/gauge.py",
            ROOT / "nox/mlp_gauge.py", ROOT / "nox/sequential_compiler.py", ROOT / "nox/data.py", ROOT / "nox/utils.py",
            ROOT / "nox/evaluation.py",
            ROOT / "nox/tensor_order.py", ROOT / "nox/allocation.py",
            ROOT / "nox/defect_bridge.py",
            ROOT / "nox/pees.py",
            ROOT / "nox/operator_schmidt.py",
            ROOT / "nox/calibration.py",
        ]),
    }
    if args.resume:
        manifest = json.loads((args.output / "model_manifest.json").read_text())
        if not resume_identity_compatible(manifest["run_identity"], identity):
            raise ValueError("Existing checkpoint identity does not match requested resumed run")
        manifest.setdefault("resume_events", []).append({
            "previous_run_fingerprint": manifest["run_fingerprint"],
            "previous_source_fingerprint": manifest["run_identity"].get("source_fingerprint"),
            "resumed_source_fingerprint": identity["source_fingerprint"],
            "previous_layers": manifest["run_identity"].get("layers"),
            "resumed_layers": identity["layers"],
            "reason": "resume after completed layer artifacts; source repair or layer-prefix extension",
        })
        manifest["run_identity"] = identity
        manifest["run_fingerprint"] = fingerprint(identity)
        manifest["layers_compressed"] = sorted(set(manifest.get("layers_compressed", [])))
        manifest.setdefault("checkpoint_file_sizes", {})
        manifest.setdefault("checkpoint_file_sha256", {})
        manifest.setdefault("intermediate_files", {})
    else:
        manifest = {"run_identity": identity, "run_fingerprint": fingerprint(identity),
                    "tier": "high_fidelity_10x", "compression_path": compression_path,
                    "calibration_activation_mode": args.calibration_activation_mode,
                    "construction_mode": args.construction_mode,
                    "mpo_site_order": args.site_order,
                    "rms_norm_eps": config.rms_norm_eps, "history": [], "layers_compressed": [],
                    "serialized_checkpoint_bytes": 0, "checkpoint_file_sizes": {},
                    "checkpoint_file_sha256": {}, "intermediate_files": {}}
    # A manifest must exist before expensive layer compilation begins: it is
    # the recovery point if a worker is preempted during the first layer.
    atomic_json_save(manifest, args.output / "model_manifest.json")
    compressed_validation = None
    compressed_calibration = None
    for index in range(args.end_layer + 1):
        started = time.monotonic()
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        dense = weights.layer(index, args.device)
        gauge_metadata = None
        mlp_gauge_metadata = None
        if index == args.start_layer and args.attention_gauge == "balanced":
            dense, gauge_metadata = balanced_attention_gauge(
                dense, config.num_attention_heads, config.num_key_value_heads,
            )
            train_next, _, _ = execute_block(dense, train)
            validation_next, _, _ = execute_block(dense, validation)
            # Verify the exact symmetry against the original pinned dense block
            # before any lossy compression is attempted.
            original = weights.layer(index, args.device)
            original_train, _, _ = execute_block(original, train)
            original_validation, _, _ = execute_block(original, validation)
            gauge_metadata["train_parity_relative_error"] = compute_metrics(original_train, train_next)["relative_error"]
            gauge_metadata["validation_parity_relative_error"] = compute_metrics(original_validation, validation_next)["relative_error"]
            if max(gauge_metadata["train_parity_relative_error"],
                   gauge_metadata["validation_parity_relative_error"]) > 2e-5:
                raise RuntimeError(f"Exact attention gauge parity failed: {gauge_metadata}")
            del original, original_train, original_validation
        else:
            train_next, _, _ = execute_block(dense, train)
            validation_next, _, _ = execute_block(dense, validation)
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
            train_next, _, _ = execute_block(dense, train)
            validation_next, _, _ = execute_block(dense, validation)
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
                accepted_fingerprints = {manifest["run_fingerprint"]}
                accepted_fingerprints.update(
                    event.get("previous_run_fingerprint")
                    for event in manifest.get("resume_events", [])
                )
                if artifact.get("layer_idx") != index or artifact.get("run_fingerprint") not in accepted_fingerprints:
                    raise ValueError(f"Incompatible resumed artifact: {path}")
                file_size = path.stat().st_size
                file_hash = sha256_file(path)
                expected_size = manifest["checkpoint_file_sizes"].get(path.name)
                expected_hash = manifest["checkpoint_file_sha256"].get(path.name)
                if expected_size is not None and int(expected_size) != file_size:
                    raise ValueError(f"Resumed artifact size mismatch: {path}")
                if expected_hash is not None and expected_hash != file_hash:
                    raise ValueError(f"Resumed artifact checksum mismatch: {path}")
                was_manifested = index in manifest["layers_compressed"]
                manifest["checkpoint_file_sizes"][path.name] = file_size
                manifest["checkpoint_file_sha256"][path.name] = file_hash
                manifest["layers_compressed"] = sorted(set(manifest["layers_compressed"] + [index]))
                manifest["serialized_checkpoint_bytes"] = sum(
                    int(size) for size in manifest["checkpoint_file_sizes"].values()
                )
                if not was_manifested:
                    manifest.setdefault("recovered_artifacts", []).append({
                        "layer": index,
                        "sha256": file_hash,
                        "serialized_bytes": file_size,
                        "metrics_available": False,
                    })
                atomic_json_save(manifest, args.output / "model_manifest.json")
                reconstructed = dict(dense)
                reconstructed.update({
                    name: reconstruct_from_artifact(value, device=args.device)
                    for name, value in artifact["compressed_artifacts"].items()
                })
                if artifact.get("cross_layer_bridge") is not None:
                    reconstructed["cross_layer_bridge"] = artifact["cross_layer_bridge"]
                compressed_validation, _, _ = execute_block(reconstructed, compressed_validation)
                if args.calibration_activation_mode == "compressed_prefix":
                    compressed_calibration, _, _ = execute_block(reconstructed, compressed_calibration)
                train, validation = train_next, validation_next
                del dense, reconstructed, artifact
                gc.collect()
                torch.cuda.empty_cache()
                print(f"layer={index} resumed_from={path}", flush=True)
                continue
            intermediate_payloads = {
                f"dense_input_layer_{index}.pt": {"X_test": validation.cpu(), "rms_norm_eps": config.rms_norm_eps},
                f"calibration_input_layer_{index}.pt": calibration_input.cpu(),
                f"layer_input_validation_layer_{index}.pt": validation_calibration_input.cpu(),
            }
            for filename, payload in intermediate_payloads.items():
                intermediate_path = args.output / filename
                atomic_torch_save(payload, intermediate_path)
                manifest["intermediate_files"][filename] = {
                    "bytes": intermediate_path.stat().st_size,
                    "sha256": sha256_file(intermediate_path),
                }
            atomic_json_save(manifest, args.output / "model_manifest.json")
            if args.calibration_backend == "streaming_statistics":
                compiled = compile_independent_transformer_block_streaming(
                    dense,
                    calibration_input,
                    windows_per_chunk=args.calibration_windows_per_chunk,
                    ranks=compression["ranks"],
                    r_bridge=compression["r_bridge"],
                    n_outliers=compression["n_outliers"],
                    return_artifacts=True,
                    diagonal_prior_strength=args.diagonal_prior_strength,
                    projection_budgets=per_projection_budgets,
                    mpo_n_sweeps=args.mpo_sweeps,
                    mpo_alpha_awt=args.mpo_alpha_awt,
                    accumulation_dtype=statistics_dtype,
                    statistics_device=args.device,
                    output_device="cpu",
                    verbose=True,
                    site_order=args.site_order,
                    **forward_options,
                )
            else:
                compiled = compile_transformer_block(
                    dense, calibration_input, X_calib_dense=reference_input, causal_healing=False,
                    ranks=compression["ranks"], r_bridge=compression["r_bridge"],
                    n_outliers=compression["n_outliers"], return_artifacts=True,
                    diagonal_prior_strength=args.diagonal_prior_strength,
                    trajectory_mode=("none" if args.construction_mode == "independent"
                                     else args.construction_mode.removeprefix("trajectory_aware_")),
                    mlp_product_aware=(
                        args.mlp_objective == "product_aware"
                        or args.construction_mode == "trajectory_aware_product"
                    ),
                    mlp_product_ridge_ratio=args.mlp_product_ridge_ratio,
                    projection_budgets=per_projection_budgets,
                    representation=args.representation,
                    operator_schmidt_weight_mix=args.operator_schmidt_weight_mix,
                    operator_schmidt_activation_mix=args.operator_schmidt_activation_mix,
                    mpo_n_sweeps=args.mpo_sweeps,
                    mpo_alpha_awt=args.mpo_alpha_awt,
                    verbose=True, site_order=args.site_order, **forward_options,
                )
            artifact = {
                "layer_idx": index, "compression_path": compression_path,
                "construction_mode": args.construction_mode,
                "source_fingerprint": identity["source_fingerprint"],
                "mpo_site_order": args.site_order,
                "attention_gauge": gauge_metadata,
                "mlp_gauge": mlp_gauge_metadata,
                "mlp_objective": args.mlp_objective,
                "mlp_product_aware": compiled["mlp_product_aware"],
                "budget_allocation": args.budget_allocation,
                "projection_budgets": per_projection_budgets,
                "representation": args.representation,
                "mpo_n_sweeps": args.mpo_sweeps,
                "mpo_alpha_awt": args.mpo_alpha_awt,
                "calibration_backend": args.calibration_backend,
                "calibration_windows_per_chunk": args.calibration_windows_per_chunk,
                "statistics_dtype": args.statistics_dtype,
                "rms_norm_eps": config.rms_norm_eps, "run_fingerprint": manifest["run_fingerprint"],
                "compressed_artifacts": compiled["compressed_artifacts"],
                "input_layernorm": dense["input_layernorm"].cpu(),
                "post_attention_layernorm": dense["post_attention_layernorm"].cpu(),
                "total_kb": compiled["total_kb"],
            }
            pees_base_checkpoint_bytes = None
            pees_tensor_bytes = 0
            if args.pees_rank > 0:
                base_buffer = io.BytesIO()
                torch.save(artifact, base_buffer)
                pees_base_checkpoint_bytes = len(base_buffer.getbuffer())
                del base_buffer

                per_rank_bytes = sum(
                    (dense[name].shape[0] + dense[name].shape[1]) * 2
                    for name in pees_projections
                )
                sidecar_budget_bytes = (
                    pees_base_checkpoint_bytes
                    * (args.max_checkpoint_multiplier - 1.0)
                    * 0.98
                )
                budgeted_rank = int(sidecar_budget_bytes // per_rank_bytes)
                effective_rank = min(args.pees_rank, budgeted_rank)
                if effective_rank <= 0:
                    raise RuntimeError(
                        "The configured checkpoint multiplier leaves no room for a PEES sidecar"
                    )

                base_reconstructed = dict(dense)
                base_reconstructed.update({
                    name: reconstruct_from_artifact(value, device=args.device)
                    for name, value in artifact["compressed_artifacts"].items()
                })
                pees_metadata = {
                    "objective": "propagated_defect_ridge_whitened_eigenspace",
                    "requested_rank_per_projection": int(args.pees_rank),
                    "effective_rank_per_projection": int(effective_rank),
                    "projections": list(pees_projections),
                    "sample_tokens": min(
                        max(args.seq_len, (args.pees_tokens // args.seq_len) * args.seq_len),
                        (train.shape[1] // args.seq_len) * args.seq_len,
                    ),
                    "ridge_ratio": float(args.pees_ridge_ratio),
                    "base_checkpoint_bytes": int(pees_base_checkpoint_bytes),
                    "max_checkpoint_multiplier": float(args.max_checkpoint_multiplier),
                    "sidecar_budget_bytes_with_safety_margin": int(sidecar_budget_bytes),
                    "conditional_projection_order": list(pees_projections),
                    "projection_metrics": {},
                }
                for name in pees_projections:
                    # Recompute the compressed contexts after each correction.
                    # Later projections therefore compensate the remaining
                    # defect after earlier PEES factors have already been
                    # applied, avoiding duplicate cancellation of the same
                    # upstream error.
                    trajectory_contexts = layer_projection_trajectories(
                        dense,
                        base_reconstructed,
                        train,
                        calibration_input,
                        seq_len=args.seq_len,
                        n_heads=config.num_attention_heads,
                        n_kv_heads=config.num_key_value_heads,
                        rms_norm_eps=config.rms_norm_eps,
                        max_tokens=args.pees_tokens,
                        projections=(name,),
                    )
                    dense_context, compressed_context = trajectory_contexts[name]
                    left, right, projection_metrics = fit_propagated_eigenspace_shield(
                        dense_context,
                        compressed_context,
                        dense[name],
                        base_reconstructed[name],
                        rank=effective_rank,
                        ridge_ratio=args.pees_ridge_ratio,
                    )
                    added_bytes = append_shield_to_projection_artifact(
                        artifact["compressed_artifacts"][name], left, right,
                    )
                    pees_tensor_bytes += added_bytes
                    projection_metrics["added_tensor_bytes"] = added_bytes
                    pees_metadata["projection_metrics"][name] = projection_metrics
                    base_reconstructed[name] = reconstruct_from_artifact(
                        artifact["compressed_artifacts"][name], device=args.device,
                    )
                    del trajectory_contexts
                artifact["pees"] = pees_metadata
                del base_reconstructed

            # Re-decode the serialized artifacts and compare them with the
            # compiler's in-memory matrices before releasing those references.
            compiled.pop("X_out_comp", None)
            compiled.pop("X_out_dense", None)
            reconstructed = dict(dense)
            reconstructed.update({name: reconstruct_from_artifact(value, device=args.device)
                                  for name, value in artifact["compressed_artifacts"].items()})
            projection_serialization_audit = {}
            for name, decoded in (
                (name, reconstructed[name])
                for name in artifact["compressed_artifacts"]
            ):
                reference = compiled["compressed_weights"][name]
                if decoded.shape != reference.shape or not torch.isfinite(decoded).all():
                    raise FloatingPointError(
                        f"Layer {index} {name}: serialized reconstruction has bad shape or non-finite values"
                    )
                relative_error = float(
                    (torch.linalg.vector_norm(decoded.float() - reference.float())
                     / torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)).item()
                )
                if not math.isfinite(relative_error) or relative_error > 1e-2:
                    raise RuntimeError(
                        f"Layer {index} {name}: serialized artifact reconstruction disagrees "
                        f"with compiler matrix (relative Frobenius error {relative_error})"
                    )
                projection_serialization_audit[name] = {
                    "shape": list(decoded.shape),
                    "relative_frobenius_error": relative_error,
                }
            compiled.pop("compressed_weights", None)
            # The bridge learns the complete composed defect on calibration
            # trajectories: dense next-layer state minus the lossy block output
            # at the actual (possibly compressed-prefix) input.
            train_output, _, _ = execute_block(reconstructed, calibration_input)
            if args.defect_bridge_mode == "hybrid":
                baseline_buffer = io.BytesIO()
                torch.save(artifact, baseline_buffer)
                baseline_buffer_bytes = len(baseline_buffer.getbuffer())
                del baseline_buffer
                bridge = fit_hybrid_defect_bridge(
                    calibration_input,
                    train_next - train_output,
                    linear_rank=args.defect_bridge_linear_rank,
                    memory_size=args.defect_bridge_memory_size,
                    ridge_ratio=args.defect_bridge_ridge_ratio,
                    memory_top_k=args.defect_bridge_memory_top_k,
                    storage_dtype=torch.float16,
                )
                artifact["cross_layer_bridge"] = bridge
                artifact["cross_layer_bridge_metadata"] = dict(bridge["metadata"])
                reconstructed["cross_layer_bridge"] = bridge
                artifact["cross_layer_bridge_metadata"]["base_checkpoint_bytes_before_bridge"] = int(
                    baseline_buffer_bytes
                )
                artifact["cross_layer_bridge_metadata"]["max_checkpoint_multiplier"] = float(
                    args.max_checkpoint_multiplier
                )

            dense_same_input, _, _ = execute_block(dense, validation_calibration_input)
            same_input, _, _ = execute_block(reconstructed, validation_calibration_input)
            compressed_validation, _, _ = execute_block(reconstructed, compressed_validation)
            calibration_output, _, _ = execute_block(reconstructed, calibration_input)
            calibration_target = train_next
            calibration_metrics = compute_metrics(calibration_target, calibration_output)
            validation_output, _, _ = execute_block(reconstructed, validation_calibration_input)
            validation_metrics = compute_metrics(validation_next, validation_output)
            if args.defect_bridge_mode == "hybrid":
                artifact["cross_layer_bridge_metadata"].update({
                    "validation_relative_error": validation_metrics["relative_error"],
                    "validation_cosine_similarity": validation_metrics["cosine_similarity"],
                })
            atomic_torch_save(artifact, path)
            actual_serialized_bytes = path.stat().st_size
            if args.pees_rank > 0:
                max_allowed_bytes = pees_base_checkpoint_bytes * args.max_checkpoint_multiplier
                if actual_serialized_bytes > max_allowed_bytes:
                    path.unlink()
                    raise RuntimeError(
                        f"PEES exceeds checkpoint budget: {actual_serialized_bytes} > "
                        f"{max_allowed_bytes:.0f} ({args.max_checkpoint_multiplier:.3f}x)"
                    )
            if args.defect_bridge_mode == "hybrid":
                max_allowed_bytes = baseline_buffer_bytes * args.max_checkpoint_multiplier
                if actual_serialized_bytes > max_allowed_bytes:
                    path.unlink()
                    raise RuntimeError(
                        f"Defect bridge exceeds checkpoint budget: {actual_serialized_bytes} > "
                        f"{max_allowed_bytes:.0f} ({args.max_checkpoint_multiplier:.3f}x)"
                    )
            record = {
                "layer": index, "calibration_activation_mode": args.calibration_activation_mode,
                "construction_mode": args.construction_mode,
                "mpo_site_order": args.site_order,
                "same_input_validation": compute_metrics(dense_same_input, same_input),
                "cascade_validation": compute_metrics(validation_next, compressed_validation),
                "calibration": calibration_metrics,
                "heldout_block_output": validation_metrics,
                "attention_gauge": gauge_metadata,
                "mlp_gauge": mlp_gauge_metadata,
                "mlp_objective": args.mlp_objective,
                "mlp_product_aware": compiled["mlp_product_aware"],
                "budget_allocation": args.budget_allocation,
                "projection_budgets": per_projection_budgets,
                "representation": args.representation,
                "calibration_backend": args.calibration_backend,
                "calibration_windows_per_chunk": args.calibration_windows_per_chunk,
                "statistics_dtype": args.statistics_dtype,
                "projection_serialization_audit": projection_serialization_audit,
                "serialized_bytes": path.stat().st_size,
                "base_checkpoint_bytes_before_bridge": (
                    baseline_buffer_bytes if args.defect_bridge_mode == "hybrid" else None
                ),
                "bridge_tensor_bytes": (
                    bridge_tensor_bytes(artifact["cross_layer_bridge"])
                    if args.defect_bridge_mode == "hybrid" else 0
                ),
                "pees": artifact.get("pees"),
                "base_checkpoint_bytes_before_pees": pees_base_checkpoint_bytes,
                "pees_tensor_bytes": pees_tensor_bytes,
                "pees_checkpoint_multiplier": (
                    path.stat().st_size / pees_base_checkpoint_bytes
                    if pees_base_checkpoint_bytes is not None else 1.0
                ),
                "checkpoint_multiplier": (
                    path.stat().st_size / baseline_buffer_bytes
                    if args.defect_bridge_mode == "hybrid" else 1.0
                ),
                "dense_fp16_tensor_bytes": sum(t.numel() * 2 for t in dense.values()),
                "seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                    if torch.device(args.device).type == "cuda" else 0
                ),
                "peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(torch.device(args.device)))
                    if torch.device(args.device).type == "cuda" else 0
                ),
                "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            }
            manifest["history"].append(record)
            manifest["layers_compressed"].append(index)
            manifest["checkpoint_file_sizes"][path.name] = path.stat().st_size
            manifest["checkpoint_file_sha256"][path.name] = sha256_file(path)
            manifest["serialized_checkpoint_bytes"] += path.stat().st_size
            atomic_json_save(manifest, args.output / "model_manifest.json")
            print(json.dumps(record), flush=True)
            if args.calibration_activation_mode == "compressed_prefix":
                compressed_calibration, _, _ = execute_block(reconstructed, compressed_calibration)
            del compiled, reconstructed, same_input, train_output, artifact
        train, validation = train_next, validation_next
        del dense
        gc.collect()
        torch.cuda.empty_cache()
        print(f"layer={index} elapsed={time.monotonic() - started:.2f}s", flush=True)
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
