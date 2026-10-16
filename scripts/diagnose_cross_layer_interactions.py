#!/usr/bin/env python3
"""Diagnose projection interactions and error propagation in a compressed layer pair.

This is an oracle analysis. Projection restores use dense weights and therefore do
not constitute a compressed candidate. The script never modifies a checkpoint.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import compute_perplexity
from nox.block_compiler import RMSNorm, compute_llama2_rotary_emb, apply_llama2_rope
from nox.compiler import reconstruct_from_artifact
from nox.data import load_wikitext2_raw_split
from nox.utils import compute_metrics


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GROUPS = {
    "q": ("q_proj",),
    "k": ("k_proj",),
    "v": ("v_proj",),
    "o": ("o_proj",),
    "gate": ("gate_proj",),
    "up": ("up_proj",),
    "down": ("down_proj",),
    "qk": ("q_proj", "k_proj"),
    "gate_up": ("gate_proj", "up_proj"),
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def _json_metrics(reference, candidate):
    return {key: float(value) for key, value in compute_metrics(reference, candidate).items()}


def _module_for_projection(layer, name):
    return getattr(layer.self_attn if name in PROJECTIONS[:4] else layer.mlp, name)


def _dense_layer_weights(model, layer_idx, *, dtype=torch.float32):
    layer = model.model.layers[layer_idx]
    weights = {
        name: _module_for_projection(layer, name).weight.detach().to(dtype=dtype).clone()
        for name in PROJECTIONS
    }
    weights["input_layernorm"] = layer.input_layernorm.weight.detach().to(dtype=dtype).clone()
    weights["post_attention_layernorm"] = layer.post_attention_layernorm.weight.detach().to(dtype=dtype).clone()
    return weights


def _load_compressed_layer(checkpoint_path, device, *, dtype=torch.float32):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = {
        name: reconstruct_from_artifact(artifact, device=device, dtype=dtype)
        for name, artifact in checkpoint["compressed_artifacts"].items()
    }
    weights["input_layernorm"] = checkpoint["input_layernorm"].to(device=device, dtype=dtype)
    weights["post_attention_layernorm"] = checkpoint["post_attention_layernorm"].to(
        device=device, dtype=dtype
    )
    return checkpoint, weights


def _attention(Q, K, V, n_heads, n_kv_heads, seq_len):
    d_model, n_tokens = Q.shape
    d_head = d_model // n_heads
    n_sequences = max(1, n_tokens // seq_len)
    actual_len = n_tokens // n_sequences
    used = n_sequences * actual_len
    Q = Q[:, :used].reshape(n_heads, d_head, n_sequences, actual_len).permute(2, 0, 3, 1)
    K = K[:, :used].reshape(n_kv_heads, d_head, n_sequences, actual_len).permute(2, 0, 3, 1)
    V = V[:, :used].reshape(n_kv_heads, d_head, n_sequences, actual_len).permute(2, 0, 3, 1)
    cos, sin = compute_llama2_rotary_emb(d_head, actual_len, Q.device, dtype=Q.dtype)
    Q, K = apply_llama2_rope(Q, K, cos, sin)
    if n_kv_heads < n_heads:
        repeat = n_heads // n_kv_heads
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)
    mask = torch.triu(
        torch.full((actual_len, actual_len), float("-inf"), device=Q.device, dtype=Q.dtype),
        diagonal=1,
    )
    probabilities = F.softmax(scores + mask[None, None], dim=-1)
    context = torch.matmul(probabilities, V)
    context = context.permute(1, 3, 0, 2).reshape(d_model, used)
    if used < n_tokens:
        context = F.pad(context, (0, n_tokens - used))
    return context, probabilities


@torch.no_grad()
def forward_details(weights, X, *, n_heads, n_kv_heads, seq_len, rms_norm_eps):
    norm1 = RMSNorm(X.shape[0], eps=rms_norm_eps, weight=weights["input_layernorm"])
    norm2 = RMSNorm(X.shape[0], eps=rms_norm_eps, weight=weights["post_attention_layernorm"])
    normalized = norm1(X)
    Q = weights["q_proj"] @ normalized
    K = weights["k_proj"] @ normalized
    V = weights["v_proj"] @ normalized
    context, attention = _attention(Q, K, V, n_heads, n_kv_heads, seq_len)
    attention_output = weights["o_proj"] @ context
    residual1 = X + attention_output
    mlp_input = norm2(residual1)
    gate = weights["gate_proj"] @ mlp_input
    up = weights["up_proj"] @ mlp_input
    swiglu = F.silu(gate) * up
    mlp_output = weights["down_proj"] @ swiglu
    output = residual1 + mlp_output
    return {
        "output": output,
        "attention_output": attention_output,
        "mlp_output": mlp_output,
        "active_transform": attention_output + mlp_output,
        "attention_context": context,
        "attention_probabilities": attention,
        "residual1": residual1,
        "mlp_input": mlp_input,
        "swiglu": swiglu,
    }


def _js_divergence(left, right):
    left = left.float().clamp_min(1e-12)
    right = right.float().clamp_min(1e-12)
    midpoint = 0.5 * (left + right)
    value = 0.5 * (
        (left * (left.log() - midpoint.log())).sum(dim=-1)
        + (right * (right.log() - midpoint.log())).sum(dim=-1)
    )
    return {
        "mean": float(value.mean()),
        "p95": float(torch.quantile(value.reshape(-1), 0.95)),
        "max": float(value.max()),
    }


def _error_components(dense_input, compressed_input, dense_weights, compressed_weights, options):
    dense_clean = forward_details(dense_weights, dense_input, **options)
    dense_corrupted = forward_details(dense_weights, compressed_input, **options)
    compressed_clean = forward_details(compressed_weights, dense_input, **options)
    compressed_corrupted = forward_details(compressed_weights, compressed_input, **options)

    inherited = dense_corrupted["output"] - dense_clean["output"]
    local_on_corrupted = compressed_corrupted["output"] - dense_corrupted["output"]
    total = compressed_corrupted["output"] - dense_clean["output"]
    inherited_norm = torch.linalg.vector_norm(inherited.float())
    local_norm = torch.linalg.vector_norm(local_on_corrupted.float())
    alignment = F.cosine_similarity(
        inherited.reshape(1, -1).float(), local_on_corrupted.reshape(1, -1).float()
    ).item()
    denominator = torch.linalg.vector_norm(dense_clean["output"].float()).clamp_min(1e-30)
    input_error = torch.linalg.vector_norm((compressed_input - dense_input).float())
    propagated_error = torch.linalg.vector_norm(inherited.float())
    return {
        "intrinsic_same_input_output": _json_metrics(dense_clean["output"], compressed_clean["output"]),
        "intrinsic_same_input_active_transform": _json_metrics(
            dense_clean["active_transform"], compressed_clean["active_transform"]
        ),
        "dense_propagated_inherited_error_relative_to_output": float(propagated_error / denominator),
        "dense_input_to_output_error_amplification": float(
            propagated_error / input_error.clamp_min(1e-30)
        ),
        "local_error_on_corrupted_input_relative_to_output": float(local_norm / denominator),
        "total_composed_error_relative_to_output": float(torch.linalg.vector_norm(total.float()) / denominator),
        "inherited_local_error_cosine": alignment,
        "inherited_plus_local_identity_relative_residual": float(
            torch.linalg.vector_norm((total - inherited - local_on_corrupted).float())
            / torch.linalg.vector_norm(total.float()).clamp_min(1e-30)
        ),
        "composed_active_transform": _json_metrics(
            dense_clean["active_transform"], compressed_corrupted["active_transform"]
        ),
        "attention_js_divergence": _js_divergence(
            dense_clean["attention_probabilities"],
            compressed_corrupted["attention_probabilities"],
        ),
    }, dense_clean, compressed_corrupted


def _rowspace_and_ridge_oracle(A_train, R_train, A_validation, R_validation, max_samples=512):
    count = min(max_samples, A_train.shape[1], A_validation.shape[1])
    A_train = A_train[:, :count].float()
    R_train = R_train[:, :count].float()
    A_validation = A_validation[:, :count].float()
    R_validation = R_validation[:, :count].float()
    feature_scale = float(A_train.shape[0])
    # Accumulate the large products in FP32, then solve the small sample-space
    # problem in FP64. The SwiGLU Gram matrix is highly ill-conditioned, and an
    # FP32 solve can otherwise report a residual larger than the target itself.
    gram = ((A_train.T @ A_train) / feature_scale).double()
    cross = ((A_train.T @ A_validation) / feature_scale).double()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    nonnegative = eigenvalues.clamp_min(0.0)
    largest = nonnegative[-1].clamp_min(1e-30)
    rowspace_results = {}
    R_train64 = R_train.double()
    R_validation64 = R_validation.double()
    for relative_threshold in (1e-6, 1e-8):
        keep = nonnegative > largest * relative_threshold
        basis = eigenvectors[:, keep]
        projected = (R_train64 @ basis) @ basis.T if basis.shape[1] else torch.zeros_like(R_train64)
        outside = (
            torch.linalg.vector_norm(R_train64 - projected)
            / torch.linalg.vector_norm(R_train64).clamp_min(1e-30)
        )
        rowspace_results[f"{relative_threshold:.0e}"] = {
            "effective_row_rank": int(keep.sum()),
            "outside_fraction": float(outside),
            "condition_number": float(largest / nonnegative[keep][0]) if keep.any() else None,
        }
    ridge_results = {}
    mean_diagonal = gram.diagonal().mean().clamp_min(1e-30)
    for relative_ridge in (1e-6, 1e-4, 1e-2):
        inverse = 1.0 / (nonnegative + relative_ridge * mean_diagonal)
        solve_gram = (eigenvectors * inverse.unsqueeze(0)) @ eigenvectors.T
        train_coefficients = solve_gram @ gram
        validation_coefficients = solve_gram @ cross
        train_prediction = R_train64 @ train_coefficients
        validation_prediction = R_train64 @ validation_coefficients
        ridge_results[f"{relative_ridge:.0e}"] = {
            "train_relative_residual": float(
                torch.linalg.vector_norm(train_prediction - R_train64)
                / torch.linalg.vector_norm(R_train64).clamp_min(1e-30)
            ),
            "heldout_relative_residual": float(
                torch.linalg.vector_norm(validation_prediction - R_validation64)
                / torch.linalg.vector_norm(R_validation64).clamp_min(1e-30)
            ),
        }
    return {
        "samples": count,
        "effective_rowspace": rowspace_results,
        "ridge_correction_oracle": ridge_results,
    }


def _hybrid_weights(dense, compressed, restored):
    return {
        name: (dense[name] if name in restored else compressed[name])
        for name in (*PROJECTIONS, "input_layernorm", "post_attention_layernorm")
    }


@torch.no_grad()
def trajectory_diagnostics(model, checkpoint_dir, dense_weights, compressed_weights, options):
    calibration0 = torch.load(
        checkpoint_dir / "calibration_input_layer_0.pt", map_location=model.device, weights_only=True
    ).float()
    validation_payload = torch.load(
        checkpoint_dir / "dense_input_layer_0.pt", map_location=model.device, weights_only=True
    )
    validation0 = validation_payload["X_test"].float()

    dense0_train = forward_details(dense_weights[0], calibration0, **options)
    compressed0_train = forward_details(compressed_weights[0], calibration0, **options)
    dense0_validation = forward_details(dense_weights[0], validation0, **options)
    compressed0_validation = forward_details(compressed_weights[0], validation0, **options)

    layer0_decomposition, _, _ = _error_components(
        validation0, validation0, dense_weights[0], compressed_weights[0], options
    )
    layer1_decomposition, dense1_validation, compressed1_validation = _error_components(
        dense0_validation["output"], compressed0_validation["output"],
        dense_weights[1], compressed_weights[1], options,
    )
    dense1_train = forward_details(dense_weights[1], dense0_train["output"], **options)
    compressed1_train = forward_details(
        compressed_weights[1], compressed0_train["output"], **options
    )

    attention_train_correction = (
        dense1_train["attention_output"] - compressed1_train["attention_output"]
    )
    attention_validation_correction = (
        dense1_validation["attention_output"] - compressed1_validation["attention_output"]
    )
    desired_mlp_train = dense1_train["output"] - compressed1_train["residual1"]
    desired_mlp_validation = dense1_validation["output"] - compressed1_validation["residual1"]
    mlp_train_correction = desired_mlp_train - compressed1_train["mlp_output"]
    mlp_validation_correction = desired_mlp_validation - compressed1_validation["mlp_output"]

    hybrid_errors = {}
    for layer_idx in (0, 1):
        for group_name, restored in GROUPS.items():
            layer0 = (
                _hybrid_weights(dense_weights[0], compressed_weights[0], restored)
                if layer_idx == 0 else compressed_weights[0]
            )
            layer1 = (
                _hybrid_weights(dense_weights[1], compressed_weights[1], restored)
                if layer_idx == 1 else compressed_weights[1]
            )
            after0 = forward_details(layer0, validation0, **options)
            after1 = forward_details(layer1, after0["output"], **options)
            hybrid_errors[f"restore_layer{layer_idx}_{group_name}"] = {
                "final_output": _json_metrics(dense1_validation["output"], after1["output"]),
                "final_active_transform": _json_metrics(
                    dense1_validation["active_transform"], after1["active_transform"]
                ),
            }

    return {
        "layer0": layer0_decomposition,
        "layer1": layer1_decomposition,
        "pair": {
            "final_output": _json_metrics(dense1_validation["output"], compressed1_validation["output"]),
            "final_active_transform": _json_metrics(
                dense1_validation["active_transform"], compressed1_validation["active_transform"]
            ),
        },
        "hybrid_heldout_errors": hybrid_errors,
        "layer1_correction_representability": {
            "attention_output": _rowspace_and_ridge_oracle(
                compressed1_train["attention_context"], attention_train_correction,
                compressed1_validation["attention_context"], attention_validation_correction,
            ),
            "mlp_down_output_and_residual": _rowspace_and_ridge_oracle(
                compressed1_train["swiglu"], mlp_train_correction,
                compressed1_validation["swiglu"], mlp_validation_correction,
            ),
        },
    }


def _copy_variant_into_model(model, dense_cpu, compressed_cpu, restored_by_layer):
    for layer_idx in sorted(dense_cpu):
        layer = model.model.layers[layer_idx]
        restored = restored_by_layer.get(layer_idx, ())
        for name in PROJECTIONS:
            source = dense_cpu[layer_idx][name] if name in restored else compressed_cpu[layer_idx][name]
            module = _module_for_projection(layer, name)
            module.weight.data.copy_(source.to(device=module.weight.device, dtype=module.weight.dtype))


def _ppl_variants():
    variants = {"compressed": {0: (), 1: ()}, "dense_layer0": {0: PROJECTIONS, 1: ()},
                "dense_layer1": {0: (), 1: PROJECTIONS}}
    for group_name, projections in GROUPS.items():
        variants[f"restore_both_{group_name}"] = {0: projections, 1: projections}
    for layer_idx in (0, 1):
        for group_name in ("gate_up", "down", "attention", "mlp"):
            projections = GROUPS[group_name]
            variants[f"restore_layer{layer_idx}_{group_name}"] = {
                0: projections if layer_idx == 0 else (),
                1: projections if layer_idx == 1 else (),
            }
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d")
    parser.add_argument("--max-eval-tokens", type=int, default=10000)
    parser.add_argument("--trajectory-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint_files = [args.checkpoint_dir / f"compressed_layer_{idx}.pt" for idx in (0, 1)]
    if any(not path.exists() for path in checkpoint_files):
        raise FileNotFoundError("The diagnostic requires compressed_layer_0.pt and compressed_layer_1.pt")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map="auto",
    ).eval()
    if str(model.device) != args.device and args.device.startswith("cuda"):
        print(f"model primary device is {model.device}; requested {args.device}", flush=True)
    options = {
        "n_heads": model.config.num_attention_heads,
        "n_kv_heads": model.config.num_key_value_heads,
        "seq_len": 128,
        "rms_norm_eps": model.config.rms_norm_eps,
    }
    dense_gpu = {idx: _dense_layer_weights(model, idx) for idx in (0, 1)}
    loaded = [_load_compressed_layer(path, model.device) for path in checkpoint_files]
    checkpoint_metadata = {idx: loaded[idx][0] for idx in (0, 1)}
    compressed_gpu = {idx: loaded[idx][1] for idx in (0, 1)}
    dense_cpu = {
        idx: {name: value.half().cpu() for name, value in weights.items() if name in PROJECTIONS}
        for idx, weights in dense_gpu.items()
    }
    compressed_cpu = {
        idx: {name: value.half().cpu() for name, value in weights.items() if name in PROJECTIONS}
        for idx, weights in compressed_gpu.items()
    }

    results = {
        "kind": "diagnostic_dense_restore_oracle",
        "qualifying_compressed_candidate": False,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "checkpoint_run_fingerprints": {
            str(idx): checkpoint_metadata[idx].get("run_fingerprint") for idx in (0, 1)
        },
        "trajectory": trajectory_diagnostics(
            model, args.checkpoint_dir, dense_gpu, compressed_gpu, options
        ),
        "ppl_protocol": {
            "dataset": "Salesforce/wikitext/wikitext-2-raw-v1 test",
            "max_eval_tokens": args.max_eval_tokens,
            "seq_len": 2048,
            "stride": 512,
        },
        "ppl_ablations": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    del dense_gpu, compressed_gpu
    torch.cuda.empty_cache()
    if not args.trajectory_only:
        test_text, source = load_wikitext2_raw_split("test")
        results["ppl_protocol"]["source"] = source
        _copy_variant_into_model(model, dense_cpu, compressed_cpu, {0: PROJECTIONS, 1: PROJECTIONS})
        dense_ppl, dense_count, dense_details = compute_perplexity(
            model, tokenizer, test_text, max_eval_tokens=args.max_eval_tokens,
            device=args.device, return_details=True,
        )
        dense_predictions = dense_details.pop("top1_predictions")
        results["dense"] = {"ppl": dense_ppl, "scored_labels": dense_count, **dense_details}
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        for name, restored_by_layer in _ppl_variants().items():
            _copy_variant_into_model(model, dense_cpu, compressed_cpu, restored_by_layer)
            ppl, count, details = compute_perplexity(
                model, tokenizer, test_text, max_eval_tokens=args.max_eval_tokens,
                device=args.device, return_details=True,
            )
            predictions = details.pop("top1_predictions")
            record = {
                "ppl": ppl,
                "mean_nll": details["mean_nll"],
                "next_token_accuracy": details["next_token_accuracy"],
                "dense_top1_agreement": float((predictions == dense_predictions).float().mean()),
                "scored_labels": count,
                "restored_dense_projections": {
                    str(idx): list(restored_by_layer.get(idx, ())) for idx in (0, 1)
                },
            }
            results["ppl_ablations"][name] = record
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(name, json.dumps(record), flush=True)
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
