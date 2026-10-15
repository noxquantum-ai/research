"""Day 4-5 Benchmark: Language Model Quality & Perplexity Evaluation.

Evaluates WikiText-2 perplexity and autoregressive generation quality for LLaMA models
with Nox-compressed checkpoint layers substituted in-place.
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nox.compiler import reconstruct_from_artifact


def log_step(msg):
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def load_wikitext2_test_text():
    """
    Loads WikiText-2 test text using datasets or direct raw fallback.
    """
    try:
        from datasets import load_dataset
        log_step("Loading WikiText-2 test set via datasets library...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        return text
    except Exception as e:
        log_step(f"datasets load failed ({e}), falling back to direct raw HTTP download...")
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "NoxResearch/1.0"})
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode("utf-8")
        return text


@torch.no_grad()
def compute_perplexity(model, tokenizer, text, seq_len=2048, stride=512, max_eval_tokens=10000, device="cuda"):
    """
    Computes sliding-window language model perplexity across tokens.
    """
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    total_tokens = encodings.input_ids.size(1)
    eval_limit = min(total_tokens, max_eval_tokens) if max_eval_tokens > 0 else total_tokens

    log_step(f"Computing Perplexity over {eval_limit} tokens (seq_len={seq_len}, stride={stride})...")

    nlls = []
    total_evaluated_tokens = 0

    for i in range(0, eval_limit, stride):
        begin_loc = max(i + stride - seq_len, 0)
        end_loc = min(i + stride, eval_limit)
        trg_len = end_loc - i
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        total_evaluated_tokens += trg_len

        if end_loc >= eval_limit:
            break

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / total_evaluated_tokens).item()
    return ppl, total_evaluated_tokens


@torch.no_grad()
def generate_samples(model, tokenizer, prompts, max_new_tokens=64, device="cuda"):
    """
    Runs greedy generation on sample prompts to verify qualitative language capability.
    """
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append((prompt, gen_text))
    return results


def inject_nox_checkpoints(model, checkpoint_dir, device="cuda", dtype=torch.float16):
    """
    Loads saved Nox layer checkpoints and injects reconstructed weights directly into the model.
    """
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "compressed_layer_*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No 'compressed_layer_*.pt' files found in '{checkpoint_dir}'")

    log_step(f"Found {len(ckpt_files)} compressed layer checkpoints in '{checkpoint_dir}'. Injecting weights...")

    injected_layers = []

    for ckpt_path in ckpt_files:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        l_idx = data["layer_idx"]
        arts = data.get("compressed_artifacts", None)

        if arts is None:
            log_step(f"Warning: Checkpoint '{ckpt_path}' contains no compressed artifacts; skipping.")
            continue

        target_layer = model.model.layers[l_idx]

        # Reconstruct and assign attention projections
        target_layer.self_attn.q_proj.weight.data = reconstruct_from_artifact(arts["q_proj"], device=device, dtype=dtype)
        target_layer.self_attn.k_proj.weight.data = reconstruct_from_artifact(arts["k_proj"], device=device, dtype=dtype)
        target_layer.self_attn.v_proj.weight.data = reconstruct_from_artifact(arts["v_proj"], device=device, dtype=dtype)
        target_layer.self_attn.o_proj.weight.data = reconstruct_from_artifact(arts["o_proj"], device=device, dtype=dtype)

        # Reconstruct and assign MLP projections
        target_layer.mlp.gate_proj.weight.data = reconstruct_from_artifact(arts["gate_proj"], device=device, dtype=dtype)
        target_layer.mlp.up_proj.weight.data = reconstruct_from_artifact(arts["up_proj"], device=device, dtype=dtype)
        target_layer.mlp.down_proj.weight.data = reconstruct_from_artifact(arts["down_proj"], device=device, dtype=dtype)

        # Assign layer norms if present
        if data.get("input_layernorm") is not None:
            target_layer.input_layernorm.weight.data = data["input_layernorm"].to(device=device, dtype=dtype)
        if data.get("post_attention_layernorm") is not None:
            target_layer.post_attention_layernorm.weight.data = data["post_attention_layernorm"].to(device=device, dtype=dtype)

        injected_layers.append(l_idx)
        log_step(f"  Layer {l_idx:02d}: Reconstructed and injected ({data.get('total_kb', 0.0):.1f} KB artifact)")

    return injected_layers


def main():
    parser = argparse.ArgumentParser(description="Day 4-5 Benchmark: Perplexity & Language Model Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to directory containing compressed_layer_*.pt")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="Base HuggingFace model ID")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto cuda if available, else cpu)")
    parser.add_argument("--eval_dense", action="store_true", help="Also compute baseline perplexity on dense uncompressed model")
    parser.add_argument("--seq_len", type=int, default=2048, help="Context sequence length (default: 2048)")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride (default: 512)")
    parser.add_argument("--max_eval_tokens", type=int, default=10000, help="Maximum tokens to evaluate from WikiText-2 (default: 10000, 0 for full)")
    parser.add_argument("--run_generation", action="store_true", help="Run text generation quality verification")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pt_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_step(f"NVIDIA GPU Acceleration enabled: {torch.cuda.get_device_name(0)}")

    log_step(f"Loading tokenizer from '{args.model_id}'...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    log_step(f"Loading base model from '{args.model_id}' ({pt_dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=pt_dtype,
        device_map=device,
    )

    test_text = load_wikitext2_test_text()

    dense_ppl = None
    if args.eval_dense:
        log_step("Evaluating Dense Uncompressed Baseline Perplexity...")
        dense_ppl, dense_tokens = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=test_text,
            seq_len=args.seq_len,
            stride=args.stride,
            max_eval_tokens=args.max_eval_tokens,
            device=device,
        )
        log_step(f"Dense Baseline Perplexity: {dense_ppl:.2f} ({dense_tokens} tokens)")

    # Inject Nox compressed layers
    injected_layers = inject_nox_checkpoints(model, args.checkpoint_dir, device=device, dtype=pt_dtype)

    # Read manifest if available
    manifest_path = os.path.join(args.checkpoint_dir, "model_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    log_step(f"Evaluating Nox-Compressed Model Perplexity ({len(injected_layers)} compressed layers)...")
    nox_ppl, eval_tokens = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        text=test_text,
        seq_len=args.seq_len,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
        device=device,
    )
    log_step(f"Nox Compressed Model Perplexity: {nox_ppl:.2f} ({eval_tokens} tokens)")

    sample_prompts = [
        "The theory of quantum entanglement describes",
        "To implement a binary search algorithm in Python,",
        "The capital of France is",
    ]
    gen_results = []
    if args.run_generation:
        log_step("Running qualitative generation sanity checks...")
        gen_results = generate_samples(model, tokenizer, sample_prompts, max_new_tokens=64, device=device)

    print("\n" + "=" * 110)
    print(" DAY 4-5 BENCHMARK SCORECARD: WIKITEXT-2 PERPLEXITY & LANGUAGE QUALITY")
    print("=" * 110)
    print(f"Model ID:              {args.model_id}")
    print(f"Checkpoint Directory:  {args.checkpoint_dir}")
    print(f"Compression Tier:      {manifest.get('tier', 'Custom / Unspecified')}")
    print(f"Compressed Layers:     {injected_layers}")
    print(f"Total Checkpoint Size: {manifest.get('total_model_mb', 'N/A'):.2f} MB" if 'total_model_mb' in manifest else f"Total Layers: {len(injected_layers)}")
    print(f"Tokens Evaluated:      {eval_tokens}")
    if dense_ppl is not None:
        print(f"Dense FP16 Baseline PPL: {dense_ppl:.2f}")
        print(f"Nox Compressed PPL:      {nox_ppl:.2f}")
        delta_ppl = nox_ppl - dense_ppl
        pct_increase = (delta_ppl / dense_ppl) * 100.0
        print(f"PPL Degradation (Delta): +{delta_ppl:.2f} (+{pct_increase:.1f}%)")
    else:
        print(f"Nox Compressed PPL:      {nox_ppl:.2f}")
    print("=" * 110)

    if gen_results:
        print("\n" + "=" * 110)
        print(" AUTOREGRESSIVE GENERATION SANITY CHECKS (Greedy Decoding, 64 tokens)")
        print("=" * 110)
        for idx, (prompt, completion) in enumerate(gen_results, 1):
            print(f"\n[Prompt {idx}]: {prompt}")
            print(f"[Completion]:\n{completion}")
            print("-" * 110)


if __name__ == "__main__":
    main()
