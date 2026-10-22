#!/usr/bin/env python3
"""Repair saved NEC attention operators with sequential Schmidt corrections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import sample_calibration_offsets
from nox.block_compiler import scaled_dot_product_attention_2d
from nox.data import load_wikitext2_raw_split
from nox.schmidt_correction import compile_schmidt_correction


ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")


def rms_norm(value, weight, eps):
    return value * torch.rsqrt(value.square().mean(dim=0, keepdim=True) + eps) * weight[:, None]


def relative_error(reference, candidate):
    return float((reference - candidate).norm() / reference.norm().clamp_min(1e-30))


def token_digest(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--calibration-tokens", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--capture-batch-windows", type=int, default=4)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument(
        "--qkv-rank", type=int, default=None,
        help="Schmidt correction rank for Q/K/V; defaults to --rank.",
    )
    parser.add_argument("--ridge-ratio", type=float, default=0.02)
    parser.add_argument("--drift-compensation", type=float, default=1.0)
    parser.add_argument(
        "--teacher-target",
        choices=("same_input", "clean_trajectory"),
        default="same_input",
        help=("Evaluate the dense attention teacher on the compressed-prefix input, or "
              "target the separately captured clean trajectory."),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Schmidt output directory must be empty")
    if args.calibration_tokens % args.sequence_length:
        raise ValueError("calibration tokens must be divisible by sequence length")
    qkv_rank = args.rank if args.qkv_rank is None else args.qkv_rank
    if args.capture_batch_windows <= 0 or args.rank <= 0 or qkv_rank <= 0:
        raise ValueError("batch window count and rank must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()
    layers = model.model.layers

    text, source_metadata = load_wikitext2_raw_split("train")
    corpus = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0]
    offsets = sample_calibration_offsets(
        int(corpus.numel()), args.calibration_tokens, args.sequence_length, 42,
    )
    windows = torch.stack([corpus[start:start + args.sequence_length] for start in offsets])
    captures: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        def capture(_module, values, layer_index=index):
            captures[layer_index].append(values[0].detach().to(device="cpu", dtype=torch.float16))
        handles.append(layer.register_forward_pre_hook(capture))
    for begin in range(0, windows.shape[0], args.capture_batch_windows):
        batch = windows[begin:begin + args.capture_batch_windows].to(device)
        model(batch, use_cache=False)
        print(f"capture windows={min(begin + args.capture_batch_windows, windows.shape[0])}/{windows.shape[0]}", flush=True)
    for handle in handles:
        handle.remove()
    del corpus

    dense_inputs = [
        torch.cat(parts, dim=0).reshape(-1, model.config.hidden_size).T.contiguous()
        for parts in captures
    ]
    compressed_hidden = dense_inputs[0].to(device=device, dtype=torch.float32)
    layer_records = []
    n_heads = int(model.config.num_attention_heads)
    n_kv_heads = int(model.config.num_key_value_heads)
    eps = float(model.config.rms_norm_eps)

    for index, layer in enumerate(layers):
        layer_started = time.monotonic()
        source_path = args.checkpoint_dir / f"compressed_layer_{index}.pt"
        source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
        source_artifacts = source_checkpoint["compressed_artifacts"]
        dense_hidden = dense_inputs[index].to(device=device, dtype=torch.float32)
        norm1_weight = layer.input_layernorm.weight.detach().float()
        compressed_norm = rms_norm(compressed_hidden, norm1_weight, eps)

        if args.teacher_target == "same_input":
            teacher_hidden = compressed_hidden
            teacher_norm = compressed_norm
        else:
            teacher_hidden = dense_hidden
            teacher_norm = rms_norm(dense_hidden, norm1_weight, eps)

        dense_q = layer.self_attn.q_proj.weight.detach().float() @ teacher_norm
        dense_k = layer.self_attn.k_proj.weight.detach().float() @ teacher_norm
        dense_v = layer.self_attn.v_proj.weight.detach().float() @ teacher_norm
        dense_context = scaled_dot_product_attention_2d(
            dense_q, dense_k, dense_v, n_heads, n_kv_heads, args.sequence_length,
        )
        dense_attention = layer.self_attn.o_proj.weight.detach().float() @ dense_context

        qkv_artifacts = {}
        qkv_outputs = {}
        for projection_offset, (name, target) in enumerate((
            ("q_proj", dense_q), ("k_proj", dense_k), ("v_proj", dense_v),
        )):
            artifact, output = compile_schmidt_correction(
                source_artifacts[name],
                compressed_norm,
                target,
                rank=qkv_rank,
                ridge_ratio=args.ridge_ratio,
                seed=10_000 + 3 * index + projection_offset,
            )
            qkv_artifacts[name] = artifact
            qkv_outputs[name] = output
        compressed_q = qkv_outputs["q_proj"]
        compressed_k = qkv_outputs["k_proj"]
        compressed_v = qkv_outputs["v_proj"]
        compressed_context = scaled_dot_product_attention_2d(
            compressed_q, compressed_k, compressed_v,
            n_heads, n_kv_heads, args.sequence_length,
        )
        attention_target = dense_attention + args.drift_compensation * (
            teacher_hidden - compressed_hidden
        )
        corrected_o, compressed_attention = compile_schmidt_correction(
            source_artifacts["o_proj"],
            compressed_context,
            attention_target,
            rank=args.rank,
            ridge_ratio=args.ridge_ratio,
            seed=42 + index,
        )

        residual_hidden = compressed_hidden + compressed_attention
        norm2_weight = layer.post_attention_layernorm.weight.detach().float()
        mlp_input = rms_norm(residual_hidden, norm2_weight, eps)
        gate = layer.mlp.gate_proj.weight.detach().float() @ mlp_input
        up = layer.mlp.up_proj.weight.detach().float() @ mlp_input
        mlp_output = layer.mlp.down_proj.weight.detach().float() @ (F.silu(gate) * up)
        compressed_hidden = residual_hidden + mlp_output

        artifacts = {
            **qkv_artifacts,
            "o_proj": corrected_o,
        }
        output_checkpoint = {
            "layer_idx": index,
            "compressed_artifacts": artifacts,
            "input_layernorm": source_checkpoint.get("input_layernorm"),
            "post_attention_layernorm": source_checkpoint.get("post_attention_layernorm"),
            "compression_path": "nec_sequential_attention_schmidt",
        }
        output_path = args.output / f"compressed_layer_{index}.pt"
        torch.save(output_checkpoint, output_path)
        next_error = None
        if index + 1 < len(dense_inputs):
            next_error = relative_error(
                dense_inputs[index + 1].to(device=device, dtype=torch.float32),
                compressed_hidden,
            )
        record = {
            "layer": index,
            "rank": corrected_o["rank"],
            "qkv_rank": qkv_rank,
            "qkv_fit_error_after": {
                name: qkv_artifacts[name]["relative_output_error_after"]
                for name in ("q_proj", "k_proj", "v_proj")
            },
            "o_payload_bytes": corrected_o["payload_bytes"],
            "attention_fit_error_before": corrected_o["relative_output_error_before"],
            "attention_fit_error_after": corrected_o["relative_output_error_after"],
            "next_hidden_relative_error": next_error,
            "checkpoint_bytes": output_path.stat().st_size,
            "seconds": time.monotonic() - layer_started,
        }
        layer_records.append(record)
        print("LAYER " + json.dumps(record), flush=True)
        del dense_hidden, teacher_norm, compressed_norm, qkv_outputs
        torch.cuda.empty_cache() if device.type == "cuda" else None

    manifest = {
        "schema": "nec_sequential_attention_schmidt_v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "source_checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "compressed_projections": list(ATTENTION_PROJECTIONS),
        "dense_projections": ["gate_proj", "up_proj", "down_proj"],
        "calibration": {
            "tokens": args.calibration_tokens,
            "sequence_length": args.sequence_length,
            "token_ids_sha256": token_digest(windows),
            "source": source_metadata,
        },
        "schmidt_rank": args.rank,
        "qkv_schmidt_rank": qkv_rank,
        "ridge_ratio": args.ridge_ratio,
        "drift_compensation": args.drift_compensation,
        "teacher_target": args.teacher_target,
        "layers": layer_records,
        "artifact_bytes": sum(path.stat().st_size for path in args.output.glob("compressed_layer_*.pt")),
        "seconds": time.monotonic() - started,
    }
    (args.output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
