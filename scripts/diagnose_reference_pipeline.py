#!/usr/bin/env python3
"""Measure true raw/legacy WikiText quality and check custom dense-layer parity."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import compute_perplexity, inject_nox_checkpoints
from nox.data import load_wikitext2_processed_split, load_wikitext2_raw_split
from nox.sequential_compiler import forward_transformer_block


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def layer_parity(model, tokenizer, validation_text, weights_dir, layer_indices=(0, 15)):
    token_ids = tokenizer(validation_text, return_tensors="pt").input_ids[:, :128].to(model.device)
    with torch.no_grad():
        states = model(token_ids, output_hidden_states=True, use_cache=False).hidden_states
    checks = {}
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        x = states[layer_idx].squeeze(0).T.float()
        reference = states[layer_idx + 1].squeeze(0).T.float()
        weights = {}
        cache_errors = {}
        for name in PROJECTIONS:
            module = getattr(layer.self_attn if name in PROJECTIONS[:4] else layer.mlp, name)
            dense = module.weight.detach().float()
            weights[name] = dense
            cached = torch.load(weights_dir / f"layer_{layer_idx}_{name}.pt", map_location="cpu", weights_only=True).to(dense.device)
            cache_errors[name] = (cached - dense).norm().item() / dense.norm().item()
        for name in ("input_layernorm", "post_attention_layernorm"):
            dense = getattr(layer, name).weight.detach().float()
            weights[name] = dense
            cached = torch.load(weights_dir / f"layer_{layer_idx}_{name}.pt", map_location="cpu", weights_only=True).to(dense.device)
            cache_errors[name] = (cached - dense).norm().item() / dense.norm().item()
        custom, _, _ = forward_transformer_block(
            weights, x, n_heads=model.config.num_attention_heads,
            n_kv_heads=model.config.num_key_value_heads, seq_len=128,
            rms_norm_eps=1e-6,  # Explicit legacy control, even after the default is fixed.
        )
        relative_error = ((custom - reference).norm() / reference.norm()).item()
        configured, _, _ = forward_transformer_block(
            weights, x, n_heads=model.config.num_attention_heads,
            n_kv_heads=model.config.num_key_value_heads, seq_len=128,
            rms_norm_eps=model.config.rms_norm_eps,
        )
        configured_error = ((configured - reference).norm() / reference.norm()).item()
        checks[str(layer_idx)] = {
            "custom_vs_hf_relative_error": relative_error,
            "custom_vs_hf_max_abs": (custom - reference).abs().max().item(),
            "max_cache_weight_relative_error": max(cache_errors.values()),
            "cache_weight_relative_errors": cache_errors,
            "configured_epsilon": model.config.rms_norm_eps,
            "configured_epsilon_relative_error": configured_error,
        }
        print(f"dense_layer_parity layer={layer_idx} relative_error={relative_error:.8f} "
              f"max_cache_error={max(cache_errors.values()):.8f}", flush=True)
        print(f"configured_epsilon={model.config.rms_norm_eps} relative_error={configured_error:.8f}", flush=True)
    return checks


def score(model, tokenizer, text, max_eval_tokens, device):
    ppl, count, details = compute_perplexity(
        model, tokenizer, text, max_eval_tokens=max_eval_tokens,
        device=device, return_details=True,
    )
    return {
        "ppl": ppl, "scored_labels": count,
        "next_token_accuracy": details["next_token_accuracy"],
        "mean_nll": details["mean_nll"],
        "input_tokens": details["input_tokens"],
    }, details["top1_predictions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d")
    parser.add_argument("--max-eval-tokens", type=int, default=10000)
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, device_map="auto",
    ).eval()
    device = str(model.device)
    raw_validation, validation_meta = load_wikitext2_raw_split("validation")
    parity = layer_parity(model, tokenizer, raw_validation, args.weights_dir)
    if args.parity_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump({"layer_parity": parity, "validation_source": validation_meta}, output, indent=2)
        return
    if any(check["max_cache_weight_relative_error"] > 1e-7 for check in parity.values()):
        raise AssertionError("Cached layer weights disagree with pinned Hugging Face model")

    raw_text, raw_meta = load_wikitext2_raw_split("test")
    old_text, old_meta = load_wikitext2_processed_split("test")
    results = {
        "model_id": args.model_id, "model_revision": args.model_revision,
        "checkpoint_dir": str(args.checkpoint_dir), "max_eval_tokens": args.max_eval_tokens,
        "seq_len": 2048, "stride": 512, "layer_parity": parity,
        "validation_source": validation_meta,
        "sources": {"raw": raw_meta, "processed": old_meta}, "scores": {},
    }
    dense_predictions = {}
    for name, text in (("raw", raw_text), ("processed", old_text)):
        dense_score, dense_predictions[name] = score(model, tokenizer, text, args.max_eval_tokens, device)
        results["scores"][name] = {"dense": dense_score}
        print(f"{name} dense PPL={dense_score['ppl']:.4f} accuracy={dense_score['next_token_accuracy']:.4f}", flush=True)

    results["injected_layers"] = inject_nox_checkpoints(model, str(args.checkpoint_dir), device=device)
    for name, text in (("raw", raw_text), ("processed", old_text)):
        compressed_score, predictions = score(model, tokenizer, text, args.max_eval_tokens, device)
        if predictions.shape != dense_predictions[name].shape:
            raise AssertionError(f"Different token counts for {name} dense/compressed runs")
        compressed_score["dense_top1_agreement"] = (
            (predictions == dense_predictions[name]).float().mean().item()
        )
        results["scores"][name]["compressed"] = compressed_score
        print(f"{name} compressed PPL={compressed_score['ppl']:.4f} "
              f"agreement={compressed_score['dense_top1_agreement']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
