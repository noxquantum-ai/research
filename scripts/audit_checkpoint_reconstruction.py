#!/usr/bin/env python3
"""Compare serialized Nox reconstructions to dense layer weights."""

import argparse
import json
import sys
from pathlib import Path

import torch
torch.set_num_threads(min(torch.get_num_threads(), 4))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.artifacts import checkpoint_storage_bytes
from nox.compiler import reconstruct_from_artifact


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def compare_weights(reference, reconstructed):
    if reference.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: reference={tuple(reference.shape)} "
            f"reconstructed={tuple(reconstructed.shape)}"
        )
    # Checkpoints often store FP16 tensors.  Norms and dot products must be
    # accumulated in FP32; FP16 reductions can overflow and even report a
    # cosine similarity greater than one.
    reference = reference.float()
    reconstructed = reconstructed.float()
    diff = reconstructed - reference
    reference_norm = reference.norm()
    denominator = reference_norm.clamp_min(torch.finfo(reference.dtype).eps)
    cosine = torch.nn.functional.cosine_similarity(
        reconstructed.flatten(), reference.flatten(), dim=0
    ).clamp(-1.0, 1.0)
    return {
        "relative_frobenius_error": diff.norm().item() / denominator.item(),
        "cosine_similarity": cosine.item(),
        "max_absolute_error": diff.abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dense-weights-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--activation-file",
        help="Optional .pt tensor [d_model, tokens] of held-out dense layer inputs; "
        "also measures dense vs reconstructed layer outputs.",
    )
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--rms-norm-eps", type=float, default=None)
    parser.add_argument(
        "--allow-missing-dense",
        action="store_true",
        help="Print partial weight comparisons when some dense reference files are absent or empty.",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    dense_dir = Path(args.dense_weights_dir)
    checkpoint_path = checkpoint_dir / f"compressed_layer_{args.layer}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    rms_norm_eps = args.rms_norm_eps if args.rms_norm_eps is not None else checkpoint.get("rms_norm_eps", 1e-6)
    print(f"audit_rms_norm_eps={rms_norm_eps} (legacy checkpoints default to their original 1e-6)")
    if checkpoint.get("layer_idx") != args.layer:
        raise ValueError(
            f"Checkpoint layer_idx={checkpoint.get('layer_idx')} does not match requested layer {args.layer}"
        )
    artifacts = checkpoint.get("compressed_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"No compressed_artifacts in {checkpoint_path}")

    sizes = checkpoint_storage_bytes(checkpoint_dir)
    print(f"checkpoint_dir_bytes={sizes} ({sizes / (1024 * 1024):.2f} MiB)")
    print(f"layer_{args.layer}_checkpoint_bytes={checkpoint_path.stat().st_size}")
    print(f"compiler_payload_estimate_kb={checkpoint.get('total_kb', 'unknown')}")
    manifest_path = checkpoint_dir / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"manifest_estimated_model_mb={manifest.get('total_model_mb', 'unknown')}")
        print(f"manifest_serialized_checkpoint_bytes={manifest.get('serialized_checkpoint_bytes', 'not recorded')}")

    dense_weights = {}
    reconstructed_weights = {}
    missing = []
    print(f"layer={args.layer} reconstruction comparison")
    for name in PROJECTIONS:
        source_path = dense_dir / f"layer_{args.layer}_{name}.pt"
        if not source_path.is_file() or source_path.stat().st_size == 0:
            missing.append(name)
            print(f"{name}: MISSING_DENSE_REFERENCE ({source_path})")
            continue

        reference = torch.load(source_path, map_location="cpu", weights_only=True).float()
        reconstructed = reconstruct_from_artifact(
            artifacts[name], device="cpu", dtype=torch.float32
        )
        scores = compare_weights(reference, reconstructed)
        print(
            f"{name}: shape={tuple(reference.shape)} "
            f"rel_fro={scores['relative_frobenius_error']:.6f} "
            f"cosine={scores['cosine_similarity']:.6f} "
            f"max_abs={scores['max_absolute_error']:.6g}"
        )
        if args.activation_file:
            dense_weights[name] = reference
            reconstructed_weights[name] = reconstructed
        else:
            del reference, reconstructed

    for name in ("input_layernorm", "post_attention_layernorm"):
        source_path = dense_dir / f"layer_{args.layer}_{name}.pt"
        if source_path.is_file() and source_path.stat().st_size:
            if args.activation_file:
                dense_weights[name] = torch.load(source_path, map_location="cpu", weights_only=True).float()
                reconstructed_weights[name] = checkpoint[name].float()
        else:
            missing.append(name)
            print(f"{name}: MISSING_DENSE_REFERENCE ({source_path})")

    if args.activation_file:
        if missing:
            raise ValueError(
                "Layer output audit requires all seven dense projections and both norms; "
                f"missing: {', '.join(missing)}"
            )
        activation_data = torch.load(args.activation_file, map_location="cpu", weights_only=True)
        if isinstance(activation_data, dict):
            activation_data = activation_data.get("X_test", activation_data.get("activations"))
        if not torch.is_tensor(activation_data) or activation_data.ndim != 2:
            raise ValueError("Activation file must contain a tensor shaped [d_model, tokens]")

        from nox.sequential_compiler import forward_transformer_block

        x = activation_data.to(device=args.device, dtype=torch.float32)
        dense_weights = {k: v.to(device=args.device, dtype=torch.float32) for k, v in dense_weights.items()}
        reconstructed_weights = {k: v.to(device=args.device, dtype=torch.float32) for k, v in reconstructed_weights.items()}
        dense_out, _, _ = forward_transformer_block(
            dense_weights, x, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, seq_len=args.seq_len, rms_norm_eps=rms_norm_eps
        )
        reconstructed_out, _, _ = forward_transformer_block(
            reconstructed_weights, x, n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, seq_len=args.seq_len, rms_norm_eps=rms_norm_eps
        )
        scores = compare_weights(dense_out, reconstructed_out)
        print(
            "same_input_layer_output: "
            f"tokens={activation_data.shape[1]} "
            f"rel_fro={scores['relative_frobenius_error']:.6f} "
            f"cosine={scores['cosine_similarity']:.6f} "
            f"max_abs={scores['max_absolute_error']:.6g}"
        )
    elif not missing:
        print("layer_output_error=NOT_MEASURED (pass --activation-file with held-out dense inputs)")

    if missing:
        print(f"audit_status=PARTIAL missing_references={','.join(missing)}")
        if not args.allow_missing_dense:
            raise SystemExit(2)
    else:
        print("audit_status=COMPLETE")


if __name__ == "__main__":
    main()
