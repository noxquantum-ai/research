#!/usr/bin/env python3
"""Run an exploratory post-compression factor fine-tuning diagnostic.

This updates a saved checkpoint after the compressor has finished, so it is
healing and is excluded from all zero-healing acceptance results. It preserves
artifact shapes, uses the existing grouped-int4 forward grid through an STE,
and selects the checkpoint by held-out layer-output error. Validation data is
used only for selection, never for optimizer updates.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nox.artifacts import checkpoint_file_sizes, fingerprint, source_files_fingerprint
from nox.block_distillation import (
    forward_block_with_grad,
    make_trainable_components,
    post_compression_refinement_metadata,
    reconstruct_trainable_artifact,
    relative_output_error,
)
from nox.compiler import (
    estimate_artifact_tensor_bytes,
    quantize_4bit_grouped,
    reconstruct_from_artifact,
)
from nox.sequential_compiler import forward_transformer_block


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read_activation(path, key=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        if key is None:
            raise ValueError(f"Expected a tensor at {path}")
        data = data[key]
    if not torch.is_tensor(data) or data.ndim != 2:
        raise ValueError(f"Expected [d_model, tokens] activation at {path}")
    return data.float()


def load_dense_weights(weights_dir, layer, device):
    result = {}
    for name in (*PROJECTIONS, "input_layernorm", "post_attention_layernorm"):
        path = weights_dir / f"layer_{layer}_{name}.pt"
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        result[name] = torch.load(path, map_location="cpu", weights_only=True).to(
            device=device, dtype=torch.float32
        )
    return result


def reconstruct_weights(artifacts, components, device):
    result = {
        name: reconstruct_trainable_artifact(artifacts[name], components[name], device)
        for name in PROJECTIONS
    }
    return result


def component_parameters(components):
    values = []
    for projection in PROJECTIONS:
        values.extend(components[projection]["cores"])
        for name in ("A_bridge", "B_bridge"):
            if components[projection][name] is not None:
                values.append(components[projection][name])
    return values


def snapshot_components(components):
    snapshot = {}
    for projection in PROJECTIONS:
        snapshot[projection] = {
            "cores": [value.detach().cpu().clone() for value in components[projection]["cores"]],
            "A_bridge": (components[projection]["A_bridge"].detach().cpu().clone()
                         if components[projection]["A_bridge"] is not None else None),
            "B_bridge": (components[projection]["B_bridge"].detach().cpu().clone()
                         if components[projection]["B_bridge"] is not None else None),
        }
    return snapshot


def restore_components(components, snapshot):
    with torch.no_grad():
        for projection in PROJECTIONS:
            for current, saved in zip(components[projection]["cores"], snapshot[projection]["cores"]):
                current.copy_(saved.to(current.device))
            for name in ("A_bridge", "B_bridge"):
                current, saved = components[projection][name], snapshot[projection][name]
                if current is not None:
                    current.copy_(saved.to(current.device))


def quantized_artifact(artifact, components):
    result = dict(artifact)
    result["cores"] = [quantize_4bit_grouped(value.detach()).half().cpu()
                       for value in components["cores"]]
    for name in ("A_bridge", "B_bridge"):
        if components[name] is not None:
            result[name] = quantize_4bit_grouped(components[name].detach()).half().cpu()
    result["total_kb"] = estimate_artifact_tensor_bytes(
        result["cores"], result["A_bridge"], result["B_bridge"], result["d_awt"],
        sparse_count=len(result["sparse_values"]) if result.get("sparse_values") is not None else 0,
    ) / 1024.0
    return result


@torch.no_grad()
def measure_validation(weights, x_validation, teacher, options):
    student = forward_block_with_grad(weights, x_validation, **options)
    return relative_output_error(teacher, student).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.learning_rate <= 0 or args.patience < 1:
        raise ValueError("Invalid optimizer settings")
    if not torch.cuda.is_available():
        raise RuntimeError("Layer-output refinement requires the remote CUDA GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    started = time.monotonic()

    checkpoint_path = args.checkpoint_dir / "compressed_layer_15.pt"
    manifest_path = args.checkpoint_dir / "model_manifest.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    if checkpoint.get("layer_idx") != 15:
        raise ValueError("This diagnostic currently targets layer 15 only")
    if manifest.get("run_fingerprint") != checkpoint.get("run_fingerprint"):
        raise ValueError("Checkpoint and manifest fingerprints do not match")
    if manifest.get("compression_path") != "no_healing_independent":
        raise ValueError("Refinement requires an independent zero-healing checkpoint")
    input_metadata = manifest["run_identity"]["input_data"]
    if input_metadata["calibration_split"] != "train" or input_metadata["heldout_split"] != "validation":
        raise ValueError("Expected distinct train and held-out validation activations")
    sequence_length = int(manifest["run_identity"]["forward_options"]["seq_len"])
    epsilon = float(manifest["rms_norm_eps"])
    x_train = read_activation(args.checkpoint_dir / "calibration_input_layer_15.pt").to(device)
    x_validation = read_activation(args.checkpoint_dir / "dense_input_layer_15.pt", "X_test").to(device)
    if x_train.shape[0] != 4096 or x_validation.shape[0] != 4096:
        raise ValueError("Expected Llama-2-7B layer input width 4096")
    if x_train.shape[1] % sequence_length or x_validation.shape[1] % sequence_length:
        raise ValueError("Activation tokens must contain whole sequence windows")

    dense_weights = load_dense_weights(args.weights_dir, 15, device)
    options = {
        "n_heads": 32,
        "n_kv_heads": 32,
        "seq_len": sequence_length,
        "rms_norm_eps": epsilon,
    }
    with torch.no_grad():
        teacher_train = forward_transformer_block(dense_weights, x_train, **options)[0]
        teacher_validation = forward_transformer_block(dense_weights, x_validation, **options)[0]

    artifacts = checkpoint["compressed_artifacts"]
    if set(artifacts) != set(PROJECTIONS):
        raise ValueError("Layer checkpoint must contain all seven projection artifacts")
    components = {name: make_trainable_components(artifacts[name], device) for name in PROJECTIONS}
    parameters = component_parameters(components)
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    train_windows = x_train.shape[1] // sequence_length
    validation_windows = x_validation.shape[1] // sequence_length
    patience = 0
    best_epoch = 0
    best_error = float("inf")
    best_snapshot = None

    with torch.no_grad():
        initial_weights = reconstruct_weights(artifacts, components, device)
        initial_weights["input_layernorm"] = dense_weights["input_layernorm"]
        initial_weights["post_attention_layernorm"] = dense_weights["post_attention_layernorm"]
        initial_error = measure_validation(initial_weights, x_validation, teacher_validation, options)
    best_error = initial_error
    best_snapshot = snapshot_components(components)
    print(f"initial heldout_layer_error={initial_error:.8f}; train_windows={train_windows}; "
          f"validation_windows={validation_windows}; steps_per_epoch={train_windows}", flush=True)

    for epoch in range(1, args.epochs + 1):
        order = list(range(train_windows))
        random.shuffle(order)
        for window in order:
            start = window * sequence_length
            stop = start + sequence_length
            optimizer.zero_grad(set_to_none=True)
            student_weights = reconstruct_weights(artifacts, components, device)
            student_weights["input_layernorm"] = dense_weights["input_layernorm"]
            student_weights["post_attention_layernorm"] = dense_weights["post_attention_layernorm"]
            prediction = forward_block_with_grad(student_weights, x_train[:, start:stop], **options)
            target = teacher_train[:, start:stop]
            loss = (prediction - target).square().sum() / target.square().sum().clamp_min(1e-12)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            del student_weights, prediction, target, loss

        with torch.no_grad():
            validation_weights = reconstruct_weights(artifacts, components, device)
            validation_weights["input_layernorm"] = dense_weights["input_layernorm"]
            validation_weights["post_attention_layernorm"] = dense_weights["post_attention_layernorm"]
            candidate_error = measure_validation(
                validation_weights, x_validation, teacher_validation, options
            )
        print(f"epoch={epoch} heldout_layer_error={candidate_error:.8f}", flush=True)
        if candidate_error < best_error - 1e-7:
            best_error = candidate_error
            best_epoch = epoch
            best_snapshot = snapshot_components(components)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early_stop_epoch={epoch}; patience={args.patience}", flush=True)
                break

    restore_components(components, best_snapshot)
    refined = dict(checkpoint)
    refined["compressed_artifacts"] = {
        name: quantized_artifact(artifacts[name], components[name]) for name in PROJECTIONS
    }
    source_run_fingerprint = manifest["run_fingerprint"]
    refinement = {
        "method": "quantization_aware_full_block_output_distillation",
        **post_compression_refinement_metadata(),
        "source_run_fingerprint": source_run_fingerprint,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_epochs": args.epochs,
        "best_epoch": best_epoch,
        "steps_per_epoch": train_windows,
        "calibration_tokens": x_train.shape[1],
        "validation_tokens": x_validation.shape[1],
        "sequence_length": sequence_length,
        "objective": "relative_frobenius_error_of_full_residual_block_output",
        "initial_validation_error": initial_error,
        "best_validation_error": best_error,
        "source_fingerprint": source_files_fingerprint([Path(__file__), ROOT / "nox/block_distillation.py"]),
    }
    refined["block_distillation"] = refinement
    refined["source_run_fingerprint"] = source_run_fingerprint

    args.output.mkdir(parents=True)
    output_checkpoint = args.output / "compressed_layer_15.pt"
    torch.save(refined, output_checkpoint)
    shutil.copy2(args.checkpoint_dir / "calibration_input_layer_15.pt", args.output)
    shutil.copy2(args.checkpoint_dir / "dense_input_layer_15.pt", args.output)

    saved = torch.load(output_checkpoint, map_location="cpu", weights_only=True)
    saved_weights = {
        name: reconstruct_from_artifact(saved["compressed_artifacts"][name], device=device, dtype=torch.float32)
        for name in PROJECTIONS
    }
    saved_weights["input_layernorm"] = dense_weights["input_layernorm"]
    saved_weights["post_attention_layernorm"] = dense_weights["post_attention_layernorm"]
    with torch.no_grad():
        saved_output = forward_transformer_block(saved_weights, x_validation, **options)[0]
        saved_error = relative_output_error(teacher_validation, saved_output).item()
    if abs(saved_error - best_error) > 1e-3:
        raise AssertionError(f"Saved quantized checkpoint changed held-out error: {best_error} -> {saved_error}")

    new_manifest = copy.deepcopy(manifest)
    identity = new_manifest["run_identity"]
    identity["refinement"] = refinement
    new_fingerprint = fingerprint(identity)
    refined["run_fingerprint"] = new_fingerprint
    torch.save(refined, output_checkpoint)
    new_manifest["parent_run_fingerprint"] = source_run_fingerprint
    new_manifest["run_identity"] = identity
    new_manifest["run_fingerprint"] = new_fingerprint
    new_manifest["history"].append({"layer": 15, "block_distillation": refinement,
                                    "saved_validation_error": saved_error})
    new_manifest["serialized_checkpoint_bytes"] = output_checkpoint.stat().st_size
    new_manifest["checkpoint_file_sizes"] = checkpoint_file_sizes(args.output, [15])
    (args.output / "model_manifest.json").write_text(json.dumps(new_manifest, indent=2))
    report = {
        "status": "COMPLETE",
        "model_id": manifest["run_identity"]["model_id"],
        "model_revision": manifest["run_identity"]["model_revision"],
        "dataset": input_metadata["dataset"],
        "calibration_split": input_metadata["calibration_split"],
        "calibration_tokens": x_train.shape[1],
        "heldout_split": input_metadata["heldout_split"],
        "heldout_tokens": x_validation.shape[1],
        "zero_healing": False,
        **{key: value for key, value in post_compression_refinement_metadata().items()
           if key != "healing"},
        "initial_heldout_layer_error": initial_error,
        "best_heldout_layer_error": best_error,
        "saved_heldout_layer_error": saved_error,
        "best_epoch": best_epoch,
        "serialized_checkpoint_bytes": output_checkpoint.stat().st_size,
        "run_fingerprint": new_fingerprint,
        "seconds": time.monotonic() - started,
    }
    with (args.output / "block_distillation_report.json").open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
