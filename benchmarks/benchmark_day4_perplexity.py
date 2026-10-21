"""Day 4-5 Benchmark: Language Model Quality & Perplexity Evaluation.

Evaluates WikiText-2 perplexity and autoregressive generation quality for LLaMA models
with Nox-compressed checkpoint layers substituted in-place.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nox.compiler import reconstruct_from_artifact
from nox.artifacts import checkpoint_file_sizes
from nox.defect_bridge import apply_defect_bridge_sequence
from nox.data import load_wikitext2_raw_split


def log_step(msg):
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def load_wikitext2_test_text(return_metadata=False):
    """Load the explicit raw WikiText-2 test split and report its checksum."""
    text, metadata = load_wikitext2_raw_split("test")
    log_step(
        f"Loaded WikiText-2 raw test split: {metadata['source_bytes']} bytes, "
        f"sha256={metadata['source_sha256']} from {metadata['source_url']}"
    )
    return (text, metadata) if return_metadata else text


def load_wikitext2_text(split="test", return_metadata=False):
    """Load a pinned raw WikiText-2 split; partial-prefix work uses validation."""
    text, metadata = load_wikitext2_raw_split(split)
    log_step(
        f"Loaded WikiText-2 raw {split} split: {metadata['source_bytes']} bytes, "
        f"sha256={metadata['source_sha256']} from {metadata['source_url']}"
    )
    return (text, metadata) if return_metadata else text


@torch.no_grad()
def sliding_window_specifications(num_tokens, seq_len=2048, stride=512, max_eval_tokens=10000):
    """Return non-duplicating strided causal-LM scoring windows.

    Each corpus target after the first token is scored exactly once.  The first
    pass uses a full context window; later passes score only the newly exposed
    ``stride`` targets while retaining up to ``seq_len - stride`` prior tokens
    as context.  This follows the corrected token-weighted Transformers
    perplexity reference implementation.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be nonnegative")
    if seq_len < 2 or stride < 1 or stride >= seq_len:
        raise ValueError("Require seq_len >= 2 and 1 <= stride < seq_len")
    if max_eval_tokens < 0:
        raise ValueError("max_eval_tokens must be nonnegative")
    eval_limit = min(num_tokens, max_eval_tokens) if max_eval_tokens > 0 else num_tokens
    windows = []
    previous_end = 0
    for begin in range(0, eval_limit, stride):
        end = min(begin + seq_len, eval_limit)
        target_length = end - previous_end
        if target_length <= 0:
            break
        # Causal-LM loss shifts labels left internally, so the first label in
        # the first window has no preceding prediction and is not scored.
        scored = target_length - int(begin == 0)
        windows.append({
            "begin": begin,
            "end": end,
            "target_length": target_length,
            "scored_tokens": scored,
        })
        previous_end = end
        if end == eval_limit:
            break
    return windows, eval_limit


@torch.inference_mode()
def compute_perplexity(
    model,
    tokenizer,
    text,
    seq_len=2048,
    stride=512,
    max_eval_tokens=10000,
    device="cuda",
    return_details=False,
    verify_reference_loss=False,
    batch_size=1,
    return_token_nll=False,
    progress_interval=0,
):
    """
    Computes sliding-window language model perplexity across tokens.
    """
    model.eval()
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if return_token_nll and not return_details:
        raise ValueError("return_token_nll requires return_details=True")
    if not isinstance(progress_interval, int) or progress_interval < 0:
        raise ValueError("progress_interval must be a nonnegative integer")
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    total_tokens = encodings.input_ids.size(1)
    if total_tokens < 2:
        raise ValueError("Tokenized evaluation text must contain at least two tokens")
    if encodings.input_ids.min().item() < 0 or encodings.input_ids.max().item() >= len(tokenizer):
        raise ValueError("Tokenizer emitted a token ID outside its declared vocabulary")
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and int(encodings.input_ids[0, 0]) != int(bos_token_id):
        raise ValueError("Raw-corpus tokenization did not begin with the pinned tokenizer BOS ID")
    windows, eval_limit = sliding_window_specifications(
        total_tokens, seq_len=seq_len, stride=stride,
        max_eval_tokens=max_eval_tokens,
    )

    log_step(
        f"Computing Perplexity over {eval_limit} tokens "
        f"(seq_len={seq_len}, stride={stride}, batch_size={batch_size})..."
    )

    total_nll = 0.0
    total_evaluated_tokens = 0
    predictions = []
    token_nlls = []
    correct = 0
    maximum_logit_abs = 0.0
    reference_loss_crosscheck = None

    forward_passes = 0
    processed_windows = 0
    evaluation_started = time.monotonic()
    window_index = 0
    while window_index < len(windows):
        window_length = windows[window_index]["end"] - windows[window_index]["begin"]
        group = []
        while window_index < len(windows) and len(group) < batch_size:
            candidate = windows[window_index]
            candidate_length = candidate["end"] - candidate["begin"]
            if candidate_length != window_length:
                break
            group.append(candidate)
            window_index += 1
        input_ids = torch.cat([
            encodings.input_ids[:, window["begin"]:window["end"]]
            for window in group
        ], dim=0).to(device)
        target_ids = input_ids.clone()
        for row, window in enumerate(group):
            target_ids[row, :-window["target_length"]] = -100

        scored_tokens = int((target_ids[:, 1:] != -100).sum().item())
        expected_group_scored = sum(window["scored_tokens"] for window in group)
        if scored_tokens != expected_group_scored:
            raise RuntimeError("Internal sliding-window target count mismatch")
        if scored_tokens == 0:
            continue
        outputs = model(input_ids)
        forward_passes += 1
        labels = target_ids[:, 1:]
        logits = outputs.logits[:, :-1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Model returned non-finite logits (possible inference overflow)")
        maximum_logit_abs = max(maximum_logit_abs, float(logits.abs().max().item()))
        mask = labels != -100
        if return_token_nll:
            per_token_nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(labels)
            nll = per_token_nll.sum()
        else:
            per_token_nll = None
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        window_nll = float(nll.item())
        if not math.isfinite(window_nll):
            raise FloatingPointError("Model returned non-finite shifted-token NLL")
        if verify_reference_loss and reference_loss_crosscheck is None:
            first_labels = labels[:1]
            first_logits = logits[:1]
            first_mask = first_labels != -100
            safe_labels = first_labels.clamp_min(0)
            target_logits = first_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            reference_per_token = torch.logsumexp(first_logits, dim=-1) - target_logits
            reference_nll = float(reference_per_token[first_mask].sum().item())
            cross_entropy_nll = float(F.cross_entropy(
                first_logits.reshape(-1, first_logits.shape[-1]),
                first_labels.reshape(-1), ignore_index=-100, reduction="sum",
            ).item())
            gap = abs(cross_entropy_nll - reference_nll)
            tolerance = 1e-4 + 1e-6 * max(abs(cross_entropy_nll), abs(reference_nll))
            if not math.isfinite(reference_nll) or gap > tolerance:
                raise RuntimeError(
                    "Cross-entropy NLL disagrees with independent logsumexp-minus-target-logit check: "
                    f"{cross_entropy_nll} vs {reference_nll}"
                )
            reference_loss_crosscheck = {
                "passed": True,
                "method": "sum(logsumexp(logits)-target_logit) over first window scored labels",
                "scored_tokens": int(first_mask.sum().item()),
                "cross_entropy_sum": cross_entropy_nll,
                "independent_formula_sum": reference_nll,
                "absolute_difference": gap,
                "absolute_relative_tolerance": tolerance,
            }
        total_nll += window_nll
        if return_token_nll:
            # Preserve scored losses in original corpus order for paired
            # comparisons; non-capture calls keep the original sum-reduction path.
            token_nlls.append(per_token_nll[mask].detach().to(device="cpu", dtype=torch.float32))
        if return_details:
            predicted = logits.argmax(dim=-1)[mask]
            correct += int((predicted == labels[mask]).sum().item())
            predictions.append(predicted.cpu())

        total_evaluated_tokens += scored_tokens
        processed_windows += len(group)
        if progress_interval and (
            processed_windows % progress_interval == 0 or processed_windows == len(windows)
        ):
            log_step(
                f"Evaluation progress: windows={processed_windows}/{len(windows)} "
                f"scored_tokens={total_evaluated_tokens} nll={total_nll:.4f} "
                f"elapsed_s={time.monotonic() - evaluation_started:.1f}"
            )

    if total_evaluated_tokens == 0:
        raise ValueError("Perplexity requires at least two input tokens")
    mean_nll = total_nll / total_evaluated_tokens
    ppl = math.exp(mean_nll)
    if return_details:
        token_ids = encodings.input_ids[0, :eval_limit].to(torch.int64).contiguous()
        token_ids_sha256 = hashlib.sha256(token_ids.numpy().tobytes()).hexdigest()
        details = {
            "top1_predictions": torch.cat(predictions),
            "next_token_accuracy": correct / total_evaluated_tokens,
            "mean_nll": mean_nll,
            "input_tokens": eval_limit,
            "token_ids_sha256": token_ids_sha256,
            "bos_token_id": bos_token_id,
            "maximum_absolute_logit": maximum_logit_abs,
            "nll_formula_crosscheck": reference_loss_crosscheck,
            "window_count": len(windows),
            "inference_batch_size": batch_size,
            "forward_passes": forward_passes,
            "window_specification": windows,
        }
        if return_token_nll:
            details["token_nll"] = torch.cat(token_nlls) if token_nlls else torch.empty(0)
            if details["token_nll"].numel() != total_evaluated_tokens:
                raise RuntimeError("Per-token NLL sidecar length disagrees with scored-token count")
            per_token_sum = float(details["token_nll"].double().sum().item())
            tolerance = 1e-4 + 1e-6 * max(abs(total_nll), 1.0)
            if abs(per_token_sum - total_nll) > tolerance:
                raise RuntimeError("Per-token NLL sidecar sum disagrees with aggregate loss")
        return ppl, total_evaluated_tokens, details
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


def inject_nox_checkpoints(
    model,
    checkpoint_dir,
    device="cuda",
    dtype=torch.float16,
    diagnostics: list[dict] | None = None,
    expected_layers: list[int] | None = None,
):
    """
    Loads NEC MPO, NEC-plus-packed-residual, or direct packed-codec checkpoints
    and injects their reconstructed weights into the model.

    Codec dispatch lives in ``reconstruct_from_artifact`` so legacy NEC files
    retain their original loading path and codec checkpoints share evaluation.
    """
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "compressed_layer_*.pt")))
    if not ckpt_files:
        raise FileNotFoundError(f"No 'compressed_layer_*.pt' files found in '{checkpoint_dir}'")
    file_layers = sorted(
        int(Path(path).stem.removeprefix("compressed_layer_")) for path in ckpt_files
    )
    if expected_layers is not None and file_layers != list(expected_layers):
        raise ValueError(
            f"Checkpoint files contain layers {file_layers}, expected exactly {expected_layers}"
        )

    log_step(f"Found {len(ckpt_files)} compressed layer checkpoints in '{checkpoint_dir}'. Injecting weights...")

    injected_layers = []

    for ckpt_path in ckpt_files:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        l_idx = data["layer_idx"]
        arts = data.get("compressed_artifacts", None)

        if arts is None:
            log_step(f"Warning: Checkpoint '{ckpt_path}' contains no compressed artifacts; skipping.")
            continue

        target_layer = model.model.layers[l_idx]
        modules = {
            "q_proj": target_layer.self_attn.q_proj,
            "k_proj": target_layer.self_attn.k_proj,
            "v_proj": target_layer.self_attn.v_proj,
            "o_proj": target_layer.self_attn.o_proj,
            "gate_proj": target_layer.mlp.gate_proj,
            "up_proj": target_layer.mlp.up_proj,
            "down_proj": target_layer.mlp.down_proj,
        }
        for name, module in modules.items():
            source_weight = module.weight.detach().float()
            if not torch.isfinite(source_weight).all():
                raise FloatingPointError(f"Layer {l_idx} {name} source weight contains non-finite values")
            reconstructed = reconstruct_from_artifact(
                arts[name], device=module.weight.device, dtype=module.weight.dtype,
            )
            if tuple(reconstructed.shape) != tuple(module.weight.shape):
                raise ValueError(f"Layer {l_idx} {name} reconstruction shape mismatch")
            if not torch.isfinite(reconstructed).all():
                raise FloatingPointError(f"Layer {l_idx} {name} reconstructed non-finite values")
            if reconstructed.dtype != module.weight.dtype:
                raise TypeError(
                    f"Layer {l_idx} {name} reconstructed dtype {reconstructed.dtype} "
                    f"does not match model dtype {module.weight.dtype}"
                )
            if diagnostics is not None:
                decoded = reconstructed.float()
                source_norm = torch.linalg.vector_norm(source_weight)
                error = decoded - source_weight
                nonzero = decoded != 0
                subnormal = nonzero & (decoded.abs() < torch.finfo(torch.float16).tiny)
                diagnostics.append({
                    "layer": int(l_idx),
                    "projection": name,
                    "shape": list(module.weight.shape),
                    "source_dtype": str(module.weight.dtype),
                    "reconstructed_dtype": str(reconstructed.dtype),
                    "source_rms": float(source_weight.square().mean().sqrt().item()),
                    "reconstructed_rms": float(decoded.square().mean().sqrt().item()),
                    "source_max_abs": float(source_weight.abs().max().item()),
                    "reconstructed_max_abs": float(decoded.abs().max().item()),
                    "relative_frobenius_error": float(
                        (torch.linalg.vector_norm(error) / source_norm.clamp_min(1e-30)).item()
                    ),
                    "cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(
                            source_weight.reshape(-1), decoded.reshape(-1), dim=0,
                        ).item()
                    ),
                    "source_nonzero_to_reconstruction_zero_count": int(
                        ((source_weight != 0) & ~nonzero).sum().item()
                    ),
                    "reconstruction_zero_fraction": float((~nonzero).float().mean().item()),
                    "reconstruction_fp16_subnormal_count": int(subnormal.sum().item()),
                })
            module.weight.data.copy_(reconstructed)

        # Assign layer norms if present
        if data.get("input_layernorm") is not None:
            module = target_layer.input_layernorm
            module.weight.data.copy_(data["input_layernorm"].to(device=module.weight.device, dtype=module.weight.dtype))
        if data.get("post_attention_layernorm") is not None:
            module = target_layer.post_attention_layernorm
            module.weight.data.copy_(data["post_attention_layernorm"].to(device=module.weight.device, dtype=module.weight.dtype))

        bridge = data.get("cross_layer_bridge")
        if bridge is not None:
            bridge = {
                key: (value.to(device=w_dev) if torch.is_tensor(value) else value)
                for key, value in bridge.items()
            }
            original_forward = target_layer.forward

            def forward_with_bridge(hidden_states, *args, _forward=original_forward,
                                    _bridge=bridge, **kwargs):
                output = _forward(hidden_states, *args, **kwargs)
                corrected = hidden_states + apply_defect_bridge_sequence(hidden_states, _bridge)
                if torch.is_tensor(output):
                    return output + (corrected - hidden_states)
                if isinstance(output, tuple):
                    return (output[0] + (corrected - hidden_states), *output[1:])
                if isinstance(output, list):
                    return [output[0] + (corrected - hidden_states), *output[1:]]
                raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")

            target_layer.forward = forward_with_bridge

        injected_layers.append(l_idx)
        log_step(f"  Layer {l_idx:02d}: Reconstructed and injected ({data.get('total_kb', 0.0):.1f} KB artifact)")

    if expected_layers is not None and sorted(injected_layers) != list(expected_layers):
        raise RuntimeError(
            f"Injected layer set {sorted(injected_layers)} differs from expected {expected_layers}"
        )
    return injected_layers


def main():
    parser = argparse.ArgumentParser(description="Day 4-5 Benchmark: Perplexity & Language Model Evaluation")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to directory containing compressed_layer_*.pt")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="Base HuggingFace model ID")
    parser.add_argument("--model_revision", type=str, default="main", help="HuggingFace revision (use a commit SHA for repeatable runs)")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)

    log_step(f"Loading base model from '{args.model_id}' ({pt_dtype})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=pt_dtype,
            revision=args.model_revision,
            low_cpu_mem_usage=True,
            device_map="auto" if str(device).startswith("cuda") else None,
        )
        if not str(device).startswith("cuda"):
            model = model.to(device)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=pt_dtype,
            revision=args.model_revision,
            low_cpu_mem_usage=True,
        ).to(device)

    test_text, test_data_metadata = load_wikitext2_test_text(return_metadata=True)

    dense_ppl = None
    dense_details = None
    if args.eval_dense:
        log_step("Evaluating Dense Uncompressed Baseline Perplexity...")
        dense_ppl, dense_tokens, dense_details = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            text=test_text,
            seq_len=args.seq_len,
            stride=args.stride,
            max_eval_tokens=args.max_eval_tokens,
            device=device,
            return_details=True,
        )
        log_step(
            f"Dense Baseline Perplexity: {dense_ppl:.4f} ({dense_tokens} labels), "
            f"next-token accuracy={dense_details['next_token_accuracy']:.6f}"
        )

    # Inject Nox compressed layers
    injected_layers = inject_nox_checkpoints(model, args.checkpoint_dir, device=device, dtype=pt_dtype)

    # Read manifest if available
    manifest_path = os.path.join(args.checkpoint_dir, "model_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    checkpoint_sizes = checkpoint_file_sizes(args.checkpoint_dir)
    serialized_checkpoint_bytes = sum(checkpoint_sizes.values())

    log_step(f"Evaluating Nox-Compressed Model Perplexity ({len(injected_layers)} compressed layers)...")
    nox_ppl, eval_tokens, nox_details = compute_perplexity(
        model=model,
        tokenizer=tokenizer,
        text=test_text,
        seq_len=args.seq_len,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
        device=device,
        return_details=True,
    )
    dense_top1_agreement = None
    if dense_details is not None:
        if dense_details["top1_predictions"].shape != nox_details["top1_predictions"].shape:
            raise RuntimeError("Dense and compressed evaluation scored different target positions")
        dense_top1_agreement = float(
            (dense_details["top1_predictions"] == nox_details["top1_predictions"]).float().mean().item()
        )
    eval_record = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "checkpoint_dir": os.path.abspath(args.checkpoint_dir),
        "compressed_layers": injected_layers,
        "dense": None if dense_details is None else {
            "ppl": dense_ppl,
            "mean_nll": dense_details["mean_nll"],
            "next_token_accuracy": dense_details["next_token_accuracy"],
        },
        "compressed": {
            "ppl": nox_ppl,
            "mean_nll": nox_details["mean_nll"],
            "next_token_accuracy": nox_details["next_token_accuracy"],
            "dense_top1_agreement": dense_top1_agreement,
        },
        "input_tokens": nox_details["input_tokens"],
        "scored_labels": eval_tokens,
        "ppl_context_length": args.seq_len,
        "sliding_window_stride": args.stride,
        "test_split": test_data_metadata,
        "compressed_layer_checkpoint_bytes": sum(
            size for name, size in checkpoint_sizes.items()
            if os.path.basename(name).startswith("compressed_layer_")
        ),
        "all_checkpoint_directory_bytes": serialized_checkpoint_bytes,
    }
    eval_record_path = os.path.join(args.checkpoint_dir, "evaluation_wikitext.json")
    with open(eval_record_path, "w", encoding="utf-8") as result_file:
        json.dump(eval_record, result_file, indent=2)
    log_step(
        f"Nox Compressed Model PPL: {nox_ppl:.4f}; mean NLL={nox_details['mean_nll']:.6f}; "
        f"next-token accuracy={nox_details['next_token_accuracy']:.6f}; "
        f"dense top-1 agreement={dense_top1_agreement}"
    )

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
    print(f"Model revision:         {args.model_revision}")
    print(f"Checkpoint Directory:  {args.checkpoint_dir}")
    print(f"Compression Tier:      {manifest.get('tier', 'Custom / Unspecified')}")
    print(f"Compressed Layers:     {injected_layers}")
    print(
        f"Serialized layer checkpoints: {serialized_checkpoint_bytes / (1024.0 * 1024.0):.2f} MiB "
        f"({serialized_checkpoint_bytes} bytes; {len(checkpoint_sizes)} files)"
    )
    if "total_model_mb" in manifest:
        print(
            "Legacy estimated tensor payload: "
            f"{manifest['total_model_mb']:.2f} MB (not serialized file size)"
        )
    print(
        f"WikiText-2 split:       {test_data_metadata['split']} "
        f"sha256={test_data_metadata['source_sha256']}"
    )
    print(f"Scored next-token labels: {eval_tokens}")
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
