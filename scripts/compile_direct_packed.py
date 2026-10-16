#!/usr/bin/env python3
"""Construct a complete zero-healing packed direct-quantization checkpoint.

The codec is fitted once from pinned WikiText-2 training activations while the
dense reference trajectory is propagated layer by layer.  No written artifact
is subsequently optimized.  The output directory becomes visible only after
all layer, base-weight, tokenizer, and manifest files have been written.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import build_real_token_calibration_data
from nox.artifacts import checkpoint_file_sizes, fingerprint, source_files_fingerprint
from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d
from nox.residual_codec import CODEC_TYPE, codec_payload_bytes, encode_groupwise
from scripts.compile_high_fidelity import PinnedWeights


PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
DEFAULT_MODEL_ID = "NousResearch/Llama-2-7b-hf"
DEFAULT_MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


@torch.no_grad()
def projection_contexts_and_output(
    weights: dict,
    hidden: torch.Tensor,
    *,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    rms_norm_eps: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return all seven dense projection inputs and the dense block output."""
    dim = hidden.shape[0]
    norm1 = RMSNorm(dim, eps=rms_norm_eps, weight=weights["input_layernorm"])(hidden)
    q = weights["q_proj"] @ norm1
    k = weights["k_proj"] @ norm1
    v = weights["v_proj"] @ norm1
    attention = scaled_dot_product_attention_2d(
        q, k, v, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len,
    )
    attention_residual = hidden + weights["o_proj"] @ attention
    norm2 = RMSNorm(
        dim, eps=rms_norm_eps, weight=weights["post_attention_layernorm"],
    )(attention_residual)
    gate = weights["gate_proj"] @ norm2
    up = weights["up_proj"] @ norm2
    swiglu = F.silu(gate) * up
    output = attention_residual + weights["down_proj"] @ swiglu
    contexts = {
        "q_proj": norm1,
        "k_proj": norm1,
        "v_proj": norm1,
        "o_proj": attention,
        "gate_proj": norm2,
        "up_proj": norm2,
        "down_proj": swiglu,
    }
    return contexts, output


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--bits", type=int, default=3, choices=(2, 3, 4))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--activation-alpha", type=float, default=0.5)
    parser.add_argument("--n-calib", type=int, default=1024)
    parser.add_argument("--n-validation", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Full packed construction must run on the authorized CUDA host")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.group_size <= 0 or args.activation_alpha < 0:
        raise ValueError("Invalid codec configuration")
    if args.seq_len < 2 or any(
        count < args.seq_len or count % args.seq_len
        for count in (args.n_calib, args.n_validation)
    ):
        raise ValueError("Calibration counts must be positive multiples of sequence length")

    staging = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        device = torch.device(args.device)
        config = AutoConfig.from_pretrained(args.model_id, revision=args.model_revision)
        if config.num_hidden_layers != 32:
            raise ValueError(f"Expected 32 transformer layers, found {config.num_hidden_layers}")
        torch.manual_seed(42)
        log(f"CUDA device: {torch.cuda.get_device_name(0)}")
        weights = PinnedWeights(args.model_id, args.model_revision)
        embedding = weights.tensor("model.embed_tokens.weight")
        train, validation, data = build_real_token_calibration_data(
            embedding,
            args.model_id,
            n_calib=args.n_calib,
            n_test=args.n_validation,
            seq_len=args.seq_len,
            revision=args.model_revision,
            calibration_sampling="uniform",
            validation_sampling="uniform",
        )
        del validation
        train = train.to(device=device, dtype=torch.float32)
        forward_options = {
            "seq_len": int(args.seq_len),
            "n_heads": int(config.num_attention_heads),
            "n_kv_heads": int(config.num_key_value_heads),
            "rms_norm_eps": float(config.rms_norm_eps),
        }
        identity = {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            # Kept compatible with the strict zero-healing acceptance harness.
            "compression_path": "no_healing_independent",
            "representation": "direct_groupwise_packed",
            "codec_type": CODEC_TYPE,
            "bits": int(args.bits),
            "group_size": int(args.group_size),
            "activation_alpha": float(args.activation_alpha),
            "calibration_activation_mode": "dense",
            "construction_mode": "initial_training_free_quantization",
            "forward_options": forward_options,
            "input_data": data,
            "layers": list(range(config.num_hidden_layers)),
            "source_fingerprint": source_files_fingerprint([
                Path(__file__), ROOT / "nox/residual_codec.py",
                ROOT / "nox/compiler.py", ROOT / "nox/block_compiler.py",
                ROOT / "nox/data.py",
            ]),
        }
        run_fingerprint = fingerprint(identity)
        layer_payload_bytes: dict[str, int] = {}
        layer_seconds: dict[str, float] = {}

        for layer_index in range(config.num_hidden_layers):
            layer_started = time.monotonic()
            dense = weights.layer(layer_index, str(device))
            contexts, train_next = projection_contexts_and_output(
                dense, train, **forward_options,
            )
            artifacts = {}
            payload_bytes = 0
            for projection in PROJECTIONS:
                context = contexts[projection].float()
                activation_rms = torch.sqrt(torch.mean(context.square(), dim=1))
                artifact = encode_groupwise(
                    dense[projection],
                    bits=args.bits,
                    group_size=args.group_size,
                    activation_rms=activation_rms,
                    alpha=args.activation_alpha,
                )
                artifact["codec_kind"] = "direct_quant"
                artifacts[projection] = artifact
                payload_bytes += codec_payload_bytes(artifact)
                del context, activation_rms

            checkpoint = {
                "layer_idx": layer_index,
                "compressed_artifacts": artifacts,
                "input_layernorm": dense["input_layernorm"].to(torch.float16).cpu(),
                "post_attention_layernorm": dense["post_attention_layernorm"].to(torch.float16).cpu(),
                "total_kb": payload_bytes / 1024.0,
                "compression_path": "no_healing_independent",
                "run_fingerprint": run_fingerprint,
                "codec_construction": {
                    "method": "direct_groupwise_packed",
                    "bits": int(args.bits),
                    "group_size": int(args.group_size),
                    "activation_alpha": float(args.activation_alpha),
                    "calibration_split": "train",
                    "gradient_healing": False,
                    "post_serialization_updates": False,
                },
            }
            destination = staging / f"compressed_layer_{layer_index}.pt"
            atomic_torch_save(checkpoint, destination)
            layer_payload_bytes[str(layer_index)] = int(payload_bytes)
            layer_seconds[str(layer_index)] = time.monotonic() - layer_started
            train = train_next
            del dense, contexts, artifacts, checkpoint, train_next
            gc.collect()
            torch.cuda.empty_cache()
            log(
                f"Layer {layer_index:02d}/31: payload={payload_bytes} bytes, "
                f"file={destination.stat().st_size} bytes, seconds={layer_seconds[str(layer_index)]:.2f}"
            )

        # The projection layer files already contain all per-layer norms.  The
        # base bundle adds every other learned tensor required for inference.
        base_tensors = {
            "model.embed_tokens.weight": embedding.to(torch.float16),
            "model.norm.weight": weights.tensor("model.norm.weight").to(torch.float16),
            "lm_head.weight": weights.tensor("lm_head.weight").to(torch.float16),
        }
        base_payload_bytes = sum(t.numel() * t.element_size() for t in base_tensors.values())
        atomic_torch_save(
            {
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "tensors": base_tensors,
                "tensor_payload_bytes": int(base_payload_bytes),
            },
            staging / "base_model_tensors.pt",
        )
        metadata_dir = staging / "metadata"
        metadata_dir.mkdir()
        config.save_pretrained(metadata_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
        tokenizer.save_pretrained(metadata_dir)

        index_payload = json.loads(Path(weights.download("model.safetensors.index.json")).read_text())
        source_tensor_payload_bytes = int(index_payload["metadata"]["total_size"])
        source_shards = sorted(set(index_payload["weight_map"].values()))
        source_shard_file_bytes = sum(Path(weights.download(name)).stat().st_size for name in source_shards)
        layer_files = checkpoint_file_sizes(staging, range(config.num_hidden_layers))
        metadata_bytes = file_tree_bytes(metadata_dir)
        manifest = {
            "run_identity": identity,
            "run_fingerprint": run_fingerprint,
            "tier": "direct_packed_3bit" if args.bits == 3 else f"direct_packed_{args.bits}bit",
            "compression_path": "no_healing_independent",
            "representation": "direct_groupwise_packed",
            "layers_compressed": list(range(config.num_hidden_layers)),
            "history": [],
            "gradient_healing": False,
            "post_serialization_updates": False,
            "checkpoint_file_sizes": layer_files,
            "serialized_checkpoint_bytes": int(sum(layer_files.values())),
            "layer_codec_payload_bytes": layer_payload_bytes,
            "base_tensor_payload_bytes": int(base_payload_bytes),
            "base_tensor_file_bytes": int((staging / "base_model_tensors.pt").stat().st_size),
            "metadata_file_bytes": int(metadata_bytes),
            "source_tensor_payload_bytes": source_tensor_payload_bytes,
            "source_weight_shard_file_bytes": int(source_shard_file_bytes),
            "source_weight_shards": source_shards,
            "construction_seconds": time.monotonic() - started,
            "layer_construction_seconds": layer_seconds,
            "storage_note": (
                "Serialized bundle includes all packed projection layers, layer norms, embeddings, "
                "final norm, LM head, config, tokenizer, and manifest. Runtime evaluation reconstructs "
                "dense weights and does not demonstrate reduced runtime memory."
            ),
        }
        (staging / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest["serialized_bundle_bytes_before_final_manifest_rewrite"] = file_tree_bytes(staging)
        (staging / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
        os.replace(staging, args.output)
        final_bundle_bytes = file_tree_bytes(args.output)
        log(
            f"COMPLETE: {args.output}; layer_files={sum(layer_files.values())}; "
            f"full_bundle={final_bundle_bytes}; source_shards={source_shard_file_bytes}; "
            f"serialized_ratio={source_shard_file_bytes / final_bundle_bytes:.4f}x; "
            f"seconds={time.monotonic() - started:.2f}"
        )
    except BaseException:
        log(f"FAILED: partial construction preserved at {staging}")
        raise


if __name__ == "__main__":
    main()
