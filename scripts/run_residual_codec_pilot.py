#!/usr/bin/env python3
"""Matched two-layer rate--distortion pilot for packed residual codecs.

This is an offline representation experiment: it reuses the existing NEC
layer-0/1 artifacts and never changes their saved factors.  Candidate codec
artifacts are produced before evaluation, with activation statistics taken
only from the saved training calibration contexts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import (
    compute_perplexity,
    inject_nox_checkpoints,
    load_wikitext2_test_text,
)
from nox.artifacts import checkpoint_file_sizes
from nox.compiler import reconstruct_from_artifact
from nox.pees import layer_projection_trajectories
from nox.residual_codec import codec_payload_bytes, decode_groupwise, encode_groupwise
from nox.sequential_compiler import forward_transformer_block
from nox.tensor_order import IDENTITY_SITE_ORDER, mode_index_permutation, permute_projection


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
DEFAULT_MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def norm_relative(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(numerator) / torch.linalg.vector_norm(denominator).clamp_min(1e-12))


def module_weights(layer) -> dict:
    return {
        "q_proj": layer.self_attn.q_proj.weight,
        "k_proj": layer.self_attn.k_proj.weight,
        "v_proj": layer.self_attn.v_proj.weight,
        "o_proj": layer.self_attn.o_proj.weight,
        "gate_proj": layer.mlp.gate_proj.weight,
        "up_proj": layer.mlp.up_proj.weight,
        "down_proj": layer.mlp.down_proj.weight,
        "input_layernorm": layer.input_layernorm.weight,
        "post_attention_layernorm": layer.post_attention_layernorm.weight,
    }


def get_dense_weights(model, layer_index: int) -> dict:
    layer = model.model.layers[layer_index]
    return {name: tensor.detach() for name, tensor in module_weights(layer).items()}


def artifact_spectrum_tails(weight: torch.Tensor, artifact: dict) -> list[dict]:
    """Compute cut tails of the activation-preconditioned pre-MPO remainder."""
    device = weight.device
    out_modes = tuple(int(core.shape[1]) for core in artifact["cores"])
    in_modes = tuple(int(core.shape[2]) for core in artifact["cores"])
    site_order = tuple(artifact.get("site_order", IDENTITY_SITE_ORDER))
    remainder = weight.float()
    if site_order != IDENTITY_SITE_ORDER:
        out_perm = mode_index_permutation(tuple(artifact["original_o_modes"]), site_order, device=device)
        in_perm = mode_index_permutation(tuple(artifact["original_i_modes"]), site_order, device=device)
        remainder = permute_projection(remainder, out_perm, in_perm)

    base_structured = torch.zeros_like(weight, dtype=torch.float32)
    if artifact.get("A_bridge") is not None and artifact.get("B_bridge") is not None:
        base_structured.add_(artifact["A_bridge"].to(device=device, dtype=torch.float32)
                             @ artifact["B_bridge"].to(device=device, dtype=torch.float32))
    if artifact.get("sparse_indices") is not None and artifact.get("sparse_values") is not None:
        indices = artifact["sparse_indices"].to(device=device, dtype=torch.long)
        values = artifact["sparse_values"].to(device=device, dtype=torch.float32)
        base_structured[indices[:, 0], indices[:, 1]] += values
    remainder = remainder - base_structured
    d_awt = artifact["d_awt"].to(device=device, dtype=torch.float32)
    conditioned = remainder * d_awt.unsqueeze(0)

    cores = artifact["cores"]
    tensor = conditioned.reshape(*out_modes, *in_modes)
    tensor = tensor.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    site_sizes = [out_modes[i] * in_modes[i] for i in range(4)]
    result = []
    for cut in range(1, 4):
        left = math.prod(site_sizes[:cut])
        matrix = tensor.reshape(left, -1)
        spectral_matrix = matrix if matrix.shape[0] <= matrix.shape[1] else matrix.T
        singular_values = torch.linalg.svdvals(spectral_matrix)
        rank = int(cores[cut - 1].shape[-1])
        total_energy = singular_values.square().sum().clamp_min(1e-20)
        tail = singular_values[rank:].square().sum() / total_energy
        result.append({
            "cut": cut,
            "left_dimension": int(matrix.shape[0]),
            "right_dimension": int(matrix.shape[1]),
            "configured_bond_rank": rank,
            "discarded_energy_fraction": float(tail),
        })
        del matrix, spectral_matrix, singular_values
    del base_structured, remainder, conditioned, tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def read_checkpoint(checkpoint_dir: Path, layer_index: int) -> dict:
    path = checkpoint_dir / f"compressed_layer_{layer_index}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_activation_tensor(path: Path) -> torch.Tensor:
    """Load either a raw activation matrix or a saved ``X_test`` validation dict."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, dict) and torch.is_tensor(value.get("X_test")):
        return value["X_test"].float()
    raise ValueError(f"Unsupported saved activation payload at {path}")


def compute_layer_contexts(dense_weights: dict, input_path: Path, seq_len: int, heads: int,
                           kv_heads: int, rms_norm_eps: float, device: torch.device) -> dict:
    inputs = load_activation_tensor(input_path).to(device)
    contexts = layer_projection_trajectories(
        dense_weights=dense_weights,
        compressed_weights=dense_weights,
        dense_input=inputs,
        compressed_input=inputs,
        seq_len=seq_len,
        n_heads=heads,
        n_kv_heads=kv_heads,
        rms_norm_eps=rms_norm_eps,
        max_tokens=inputs.shape[1],
        projections=PROJECTIONS,
    )
    # The utility returns (same-input context, same-input context) pairs.
    result = {name: pair[0] for name, pair in contexts.items()}
    del inputs, contexts
    return result


def write_candidate(
    *,
    method: str,
    bits: int,
    alpha: float,
    group_size: int,
    baseline_dir: Path,
    output_dir: Path,
    layer_indices: list[int],
    contexts: dict[int, dict],
    validation_contexts: dict[int, dict],
    dense_weights: dict[int, dict],
    device: torch.device,
) -> tuple[dict, dict]:
    if output_dir.exists():
        raise FileExistsError(f"Candidate directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    codec_stats = {}
    candidate_weights = {}
    total_payload_bytes = 0

    for layer_index in layer_indices:
        layer_data = read_checkpoint(baseline_dir, layer_index)
        artifacts = layer_data["compressed_artifacts"]
        new_artifacts = {}
        layer_weights = {}
        for projection in PROJECTIONS:
            base_artifact = artifacts[projection]
            dense_weight = dense_weights[layer_index][projection].to(device=device, dtype=torch.float32)
            base_weight = reconstruct_from_artifact(base_artifact, device=device, dtype=torch.float32)
            calibration_context = contexts[layer_index][projection].to(device=device, dtype=torch.float32)
            validation_context = validation_contexts[layer_index][projection].to(device=device, dtype=torch.float32)
            activation_rms = torch.sqrt(torch.mean(calibration_context.square(), dim=1))

            if method == "direct_quant":
                quantized = encode_groupwise(
                    dense_weight,
                    bits=bits,
                    group_size=group_size,
                    activation_rms=activation_rms,
                    alpha=alpha,
                )
                candidate_artifact = {**quantized, "codec_kind": "direct_quant"}
                reconstructed = decode_groupwise(quantized, device=device, dtype=torch.float32)
            elif method == "nec_hybrid":
                residual = dense_weight - base_weight
                quantized = encode_groupwise(
                    residual,
                    bits=bits,
                    group_size=group_size,
                    activation_rms=activation_rms,
                    alpha=alpha,
                )
                candidate_artifact = {
                    "codec_type": "nec_hybrid_residual_v1",
                    "codec_kind": "nec_hybrid_residual",
                    "base_artifact": base_artifact,
                    "residual_codec": quantized,
                    "orig_shape": tuple(dense_weight.shape),
                    "residual_payload_bytes": codec_payload_bytes(quantized),
                }
                reconstructed = base_weight + decode_groupwise(
                    quantized, device=device, dtype=torch.float32,
                )
                del residual
            else:
                raise ValueError(f"unsupported method {method!r}")

            target_output = dense_weight @ validation_context
            candidate_output = reconstructed @ validation_context
            error = candidate_output - target_output
            projection_record = {
                "activation_weighted_relative_error": norm_relative(error, target_output),
                "weight_relative_frobenius_error": norm_relative(reconstructed - dense_weight, dense_weight),
                "codec_payload_bytes": int(
                    codec_payload_bytes(quantized)
                    if method == "direct_quant"
                    else codec_payload_bytes(quantized)
                ),
                "train_context_tokens": int(calibration_context.shape[1]),
                "validation_context_tokens": int(validation_context.shape[1]),
                "activation_alpha": float(alpha),
                "quantization_bits": int(bits),
                "calibration_split": "train",
                "measurement_split": "validation",
            }
            codec_stats[f"layer_{layer_index}/{projection}"] = projection_record
            total_payload_bytes += projection_record["codec_payload_bytes"]
            new_artifacts[projection] = candidate_artifact
            layer_weights[projection] = reconstructed.to(torch.float16)
            del dense_weight, base_weight, calibration_context, validation_context
            del activation_rms, reconstructed, target_output, candidate_output, error

        layer_data["compressed_artifacts"] = new_artifacts
        layer_data["codec_experiment"] = {
            "method": method,
            "bits": int(bits),
            "group_size": int(group_size),
            "alpha": float(alpha),
            "gradient_healing": False,
            "offline_representation_diagnostic": True,
            "qualifying_zero_healing_artifact": False,
            "construction": "offline replacement using saved NEC checkpoint; not a newly compiled NEC artifact",
        }
        candidate_weights[layer_index] = layer_weights
        torch.save(layer_data, output_dir / f"compressed_layer_{layer_index}.pt")
        del layer_data, artifacts, new_artifacts, layer_weights
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    file_sizes = checkpoint_file_sizes(output_dir, layer_indices)
    summary = {
        "candidate": output_dir.name,
        "method": method,
        "bits": int(bits),
        "group_size": int(group_size),
        "activation_alpha": float(alpha),
        "checkpoint_file_bytes": file_sizes,
        "serialized_checkpoint_bytes": int(sum(file_sizes.values())),
        "codec_payload_bytes": int(total_payload_bytes),
        "projection_metrics": codec_stats,
    }
    (output_dir / "candidate_manifest.json").write_text(json.dumps(summary, indent=2))
    return summary, candidate_weights


@torch.no_grad()
def block_error_metrics(candidate_weights: dict, dense_weights: dict, validation_inputs: dict,
                        *, layer_indices: list[int], seq_len: int, heads: int,
                        kv_heads: int, rms_norm_eps: float) -> dict:
    local = {}
    for layer_index in layer_indices:
        dense = {
            name: tensor.to(torch.float32)
            for name, tensor in dense_weights[layer_index].items()
        }
        candidate = dict(dense)
        candidate.update({
            name: tensor.to(torch.float32)
            for name, tensor in candidate_weights[layer_index].items()
        })
        local_input = validation_inputs[layer_index].to(
            device=dense["q_proj"].device, dtype=torch.float32,
        )
        dense_out, dense_attn, dense_mlp = forward_transformer_block(
            dense, local_input, n_heads=heads, n_kv_heads=kv_heads,
            seq_len=seq_len, rms_norm_eps=rms_norm_eps,
        )
        candidate_out, candidate_attn, candidate_mlp = forward_transformer_block(
            candidate, local_input, n_heads=heads, n_kv_heads=kv_heads,
            seq_len=seq_len, rms_norm_eps=rms_norm_eps,
        )
        dense_active = dense_attn.float() + dense_mlp.float()
        candidate_active = candidate_attn.float() + candidate_mlp.float()
        local[str(layer_index)] = {
            "same_input_block_output_relative_error": norm_relative(
                candidate_out.float() - dense_out.float(), dense_out.float(),
            ),
            "same_input_active_transformation_relative_error": norm_relative(
                candidate_active - dense_active, dense_active,
            ),
        }
        if layer_index == 0:
            cascade_dense_input = dense_out
            cascade_candidate_input = candidate_out
        del local_input, candidate, dense_attn, dense_mlp, candidate_attn, candidate_mlp

    dense_state = validation_inputs[layer_indices[0]].to(
        device=dense_weights[layer_indices[0]]["q_proj"].device, dtype=torch.float32,
    )
    candidate_state = dense_state
    for layer_index in layer_indices:
        dense = {
            name: tensor.to(torch.float32)
            for name, tensor in dense_weights[layer_index].items()
        }
        candidate = dict(dense)
        candidate.update({
            name: tensor.to(torch.float32)
            for name, tensor in candidate_weights[layer_index].items()
        })
        dense_state, _, _ = forward_transformer_block(
            dense, dense_state, n_heads=heads, n_kv_heads=kv_heads,
            seq_len=seq_len, rms_norm_eps=rms_norm_eps,
        )
        candidate_state, _, _ = forward_transformer_block(
            candidate, candidate_state, n_heads=heads, n_kv_heads=kv_heads,
            seq_len=seq_len, rms_norm_eps=rms_norm_eps,
        )
    key = f"layers_{layer_indices[0]}_{layer_indices[-1]}_composed"
    local[key] = {
        "relative_error": norm_relative(candidate_state - dense_state, dense_state),
        "candidate_total_hidden_drift_from_dense": norm_relative(candidate_state - dense_state, dense_state),
    }
    return local


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Contiguous zero-based compressed layer prefix; default uses the baseline manifest")
    parser.add_argument("--candidate-set", choices=("comparison", "direct3_a05"), default="comparison",
                        help="Run all six matched codecs or only direct 3-bit alpha=0.5 for a stack extension")
    parser.add_argument("--max-eval-tokens", type=int, default=10000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This research pilot must run on the authorized remote CUDA host")
    if not args.baseline_dir.is_dir():
        raise FileNotFoundError(args.baseline_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"CUDA device: {torch.cuda.get_device_name(0)}")

    baseline_manifest = json.loads((args.baseline_dir / "model_manifest.json").read_text())
    layer_indices = args.layers or list(baseline_manifest["run_identity"]["layers"])
    if not layer_indices or layer_indices != list(range(layer_indices[0], layer_indices[-1] + 1)):
        raise ValueError("layers must be a nonempty contiguous range")
    if layer_indices[0] != 0:
        raise ValueError("this pilot evaluates a prefix and currently requires the prefix to start at layer 0")
    forward = baseline_manifest["run_identity"]["forward_options"]
    seq_len_cal = int(forward["seq_len"])
    heads = int(forward["n_heads"])
    kv_heads = int(forward["n_kv_heads"])
    rms_norm_eps = float(forward["rms_norm_eps"])
    from scripts.compile_high_fidelity import PinnedWeights

    pinned = PinnedWeights(args.model_id, args.model_revision)
    dense_weights = {}
    train_contexts = {}
    validation_contexts = {}
    validation_inputs = {}
    spectrum_report = {}
    validation_activation_sources = {}
    for layer_index in layer_indices:
        log(f"Loading pinned dense layer {layer_index} weights and saved activation windows")
        layer_weights = pinned.layer(layer_index, str(device))
        dense_weights[layer_index] = layer_weights
        train_path = args.baseline_dir / f"calibration_input_layer_{layer_index}.pt"
        validation_path = args.baseline_dir / f"layer_input_validation_layer_{layer_index}.pt"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing saved training calibration input for layer {layer_index}")
        if not validation_path.exists():
            validation_path = args.baseline_dir / f"dense_input_layer_{layer_index}.pt"
        if not validation_path.exists():
            raise FileNotFoundError(f"Missing saved held-out activation input for layer {layer_index}")
        validation_activation_sources[layer_index] = str(validation_path)
        train_contexts[layer_index] = compute_layer_contexts(
            layer_weights, train_path, seq_len_cal, heads, kv_heads, rms_norm_eps, device,
        )
        validation_contexts[layer_index] = compute_layer_contexts(
            layer_weights, validation_path, seq_len_cal, heads, kv_heads, rms_norm_eps, device,
        )
        validation_inputs[layer_index] = load_activation_tensor(validation_path).to(device)
        source_checkpoint = read_checkpoint(args.baseline_dir, layer_index)
        spectrum_report[str(layer_index)] = {}
        for projection in PROJECTIONS:
            spectrum_report[str(layer_index)][projection] = artifact_spectrum_tails(
                layer_weights[projection], source_checkpoint["compressed_artifacts"][projection],
            )
        del source_checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    all_candidate_specs = [
        ("direct_quant", 3, 0.0, "direct_3b_a0"),
        ("direct_quant", 3, 0.5, "direct_3b_a05"),
        ("direct_quant", 4, 0.0, "direct_4b_a0"),
        ("direct_quant", 4, 0.5, "direct_4b_a05"),
        ("nec_hybrid", 2, 0.0, "nec_residual_2b_a0"),
        ("nec_hybrid", 2, 0.5, "nec_residual_2b_a05"),
    ]
    candidate_specs = (
        [all_candidate_specs[1]]
        if args.candidate_set == "direct3_a05"
        else all_candidate_specs
    )
    baseline_projection_metrics = {}
    baseline_candidate_weights = {}
    for layer_index in layer_indices:
        source_checkpoint = read_checkpoint(args.baseline_dir, layer_index)
        baseline_candidate_weights[layer_index] = {}
        for projection in PROJECTIONS:
            dense_weight = dense_weights[layer_index][projection].float()
            base_artifact = source_checkpoint["compressed_artifacts"][projection]
            base_weight = reconstruct_from_artifact(base_artifact, device=device, dtype=torch.float32)
            validation_context = validation_contexts[layer_index][projection]
            target_output = dense_weight @ validation_context
            actual_output = base_weight @ validation_context
            baseline_projection_metrics[f"layer_{layer_index}/{projection}"] = {
                "activation_weighted_relative_error": norm_relative(
                    actual_output - target_output, target_output,
                ),
                "validation_context_tokens": int(validation_context.shape[1]),
            }
            baseline_candidate_weights[layer_index][projection] = base_weight.to(torch.float16)
            del dense_weight, base_weight, validation_context, target_output, actual_output
        del source_checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    baseline_block_metrics = block_error_metrics(
        baseline_candidate_weights,
        dense_weights,
        validation_inputs,
        layer_indices=layer_indices,
        seq_len=seq_len_cal,
        heads=heads,
        kv_heads=kv_heads,
        rms_norm_eps=rms_norm_eps,
    )
    del baseline_candidate_weights
    gc.collect()
    torch.cuda.empty_cache()

    candidate_summaries = {}
    for method, bits, alpha, name in candidate_specs:
        log(f"Constructing {name} from saved NEC checkpoints")
        summary, weights = write_candidate(
            method=method,
            bits=bits,
            alpha=alpha,
            group_size=args.group_size,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir / name,
            layer_indices=layer_indices,
            contexts=train_contexts,
            validation_contexts=validation_contexts,
            dense_weights=dense_weights,
            device=device,
        )
        summary["same_input_block_and_cascade_metrics"] = block_error_metrics(
            weights,
            dense_weights,
            validation_inputs,
            layer_indices=layer_indices,
            seq_len=seq_len_cal,
            heads=heads,
            kv_heads=kv_heads,
            rms_norm_eps=rms_norm_eps,
        )
        candidate_summaries[name] = summary
        (args.output_dir / name / "candidate_manifest.json").write_text(json.dumps(summary, indent=2))
        del weights
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "experiment": "packed_groupwise_residual_codec_two_layer_pilot",
        "custom_block_diagnostics_dtype": "float32",
        "language_model_evaluation_dtype": "float16",
        "source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "baseline_checkpoint_dir": str(args.baseline_dir),
        "calibration_split": baseline_manifest["run_identity"]["input_data"]["calibration_source"],
        "heldout_split": baseline_manifest["run_identity"]["input_data"]["heldout_source"],
        "heldout_activation_sources": validation_activation_sources,
        "test_protocol": {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1/test",
            "max_input_tokens": int(args.max_eval_tokens),
            "sequence_length": int(args.seq_len),
            "stride": int(args.stride),
        },
        "group_size": int(args.group_size),
        "layers": layer_indices,
        "gradient_healing": False,
        "offline_representation_diagnostic": True,
        "qualifying_zero_healing_artifact": False,
        "spectral_cut_diagnostics": spectrum_report,
        "nec_mpo_baseline_projection_metrics": baseline_projection_metrics,
        "nec_mpo_baseline_block_and_cascade_metrics": baseline_block_metrics,
        "candidates": candidate_summaries,
    }
    (args.output_dir / "representation_pilot_pre_eval.json").write_text(json.dumps(report, indent=2))

    # Release calibration-time dense weights and activations before loading the
    # full 7B model for the matched held-out PPL pass.
    del dense_weights, train_contexts, validation_contexts, validation_inputs, pinned
    gc.collect()
    torch.cuda.empty_cache()

    # Full model inference is delayed until all candidate codec artifacts exist.
    log("Loading pinned HF model once for matched dense/NEC/candidate PPL")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    text, test_metadata = load_wikitext2_test_text(return_metadata=True)
    dense_ppl, labels, dense_details = compute_perplexity(
        model, tokenizer, text, seq_len=args.seq_len, stride=args.stride,
        max_eval_tokens=args.max_eval_tokens, device=str(device), return_details=True,
    )
    evaluation = {
        "dense": {
            "ppl": dense_ppl,
            "mean_nll": dense_details["mean_nll"],
            "next_token_accuracy": dense_details["next_token_accuracy"],
        },
        "scored_labels": int(labels),
        "test_split": test_metadata,
        "candidates": {},
    }
    for name, directory in [("nec_mpo", args.baseline_dir)] + [
        (candidate, args.output_dir / candidate) for _, _, _, candidate in candidate_specs
    ]:
        log(f"Evaluating {name} on raw WikiText test")
        layers = inject_nox_checkpoints(model, directory, device=str(device), dtype=torch.float16)
        ppl, count, details = compute_perplexity(
            model, tokenizer, text, seq_len=args.seq_len, stride=args.stride,
            max_eval_tokens=args.max_eval_tokens, device=str(device), return_details=True,
        )
        if count != labels or details["top1_predictions"].shape != dense_details["top1_predictions"].shape:
            raise RuntimeError(f"Evaluation token alignment mismatch for {name}")
        record = {
            "compressed_layers": layers,
            "ppl": ppl,
            "mean_nll": details["mean_nll"],
            "next_token_accuracy": details["next_token_accuracy"],
            "dense_top1_agreement": float((
                details["top1_predictions"] == dense_details["top1_predictions"]
            ).float().mean()),
            "scored_labels": int(count),
            "serialized_checkpoint_bytes": int(sum(checkpoint_file_sizes(directory).values())),
            "checkpoint_file_bytes": checkpoint_file_sizes(directory),
        }
        evaluation["candidates"][name] = record
        if name != "nec_mpo":
            candidate_summaries[name]["wikitext_test"] = record
            (directory / "candidate_manifest.json").write_text(json.dumps(candidate_summaries[name], indent=2))
        log(
            f"{name}: PPL={ppl:.6f}, NLL={details['mean_nll']:.6f}, "
            f"next_acc={details['next_token_accuracy']:.6f}, "
            f"dense_agreement={record['dense_top1_agreement']:.6f}, "
            f"bytes={record['serialized_checkpoint_bytes']}"
        )

    evaluation["candidate_summary"] = {
        name: {
            key: value
            for key, value in record.items()
            if key in {
                "ppl", "mean_nll", "next_token_accuracy", "dense_top1_agreement",
                "serialized_checkpoint_bytes",
            }
        }
        for name, record in evaluation["candidates"].items()
    }
    evaluation["candidate_projection_and_block_metrics"] = {
        name: {
            "projection_metrics": candidate_summaries[name]["projection_metrics"],
            "same_input_block_and_cascade_metrics": candidate_summaries[name]["same_input_block_and_cascade_metrics"],
        }
        for name in candidate_summaries
    }
    (args.output_dir / "evaluation_wikitext.json").write_text(json.dumps(evaluation, indent=2))
    (args.output_dir / "representation_pilot.json").write_text(json.dumps({**report, "evaluation": evaluation}, indent=2))
    log(f"Saved pilot report to {args.output_dir / 'representation_pilot.json'}")


if __name__ == "__main__":
    main()
