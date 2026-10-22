#!/usr/bin/env python3
"""Predict a NEC layer allocation from single-layer logit probes.

The search never loads prior multi-layer allocation results. It measures the
dense model and each eligible one-layer replacement on a selection sample,
fits a Fisher logit-interaction proxy, and either selects a fixed-count
allocation or searches counts under a maximum-dense constraint. The chosen
mask is then evaluated on a separate, untouched split.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmarks.benchmark_day4_perplexity import (
    inject_nox_checkpoints,
    sliding_window_specifications,
)
from benchmarks.evaluate_high_fidelity_nec import load_model
from nox.data import load_wikitext2_raw_split
from nox.evaluation import MODEL_ID, MODEL_REVISION, sha256_file
from nox.layer_allocator import fit_fisher_allocation_proxy, search_fixed_compression_count
from diagnose_prefix_failure import NORMS, PROJECTIONS, module as layer_module
from preflight_nec_compression_targets import inspect_dense_model, layer_compressible_payload


def install_allocation(model, dense, compressed, selected, int8_dense=None):
    """Install one candidate whole-layer mask into the HF model."""
    for layer in dense:
        if isinstance(selected, dict):
            choice = selected.get(layer, "fp16")
            if choice == "nec":
                source = compressed[layer]
            elif choice == "int8":
                if int8_dense is None or layer not in int8_dense:
                    raise ValueError(f"Missing dense INT8 weights for layer {layer}")
                source = int8_dense[layer]
            elif choice == "fp16":
                source = dense[layer]
            else:
                raise ValueError(f"Unknown layer storage choice {choice!r} for layer {layer}")
        else:
            source = compressed[layer] if layer in selected else dense[layer]
        for name in (*PROJECTIONS, *NORMS):
            destination = layer_module(model.model.layers[layer], name).weight
            destination.data.copy_(source[name].to(destination.device, destination.dtype))


def quantize_layer_int8(weights):
    """Build per-output-channel INT8 linear weights and FP16 normalization weights."""
    artifacts = {}
    dequantized = {}
    for name, weight in weights.items():
        source = weight.detach().cpu().float()
        if source.ndim == 2:
            scale = source.abs().amax(dim=1, keepdim=True) / 127.0
            scale = scale.clamp_min(1e-8).to(torch.float16)
            quantized = torch.round(source / scale.float()).clamp(-127, 127).to(torch.int8)
            artifacts[name] = {"qweight": quantized, "scale": scale}
            dequantized[name] = (quantized.float() * scale.float()).to(weight.dtype)
        else:
            # Keep the small normalization vectors in FP16; per-scalar INT8
            # quantization needlessly perturbs RMSNorm while saving few bytes.
            artifacts[name] = {"weight_fp16": source.to(torch.float16)}
            dequantized[name] = artifacts[name]["weight_fp16"].to(weight.dtype)
    payload = {
        "representation": "symmetric_per_output_channel_int8",
        "layer_tensors": artifacts,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return payload, dequantized, len(buffer.getbuffer())


def dequantize_layer_int8(payload, weights):
    """Materialize just one INT8 layer for the existing dense HF evaluator."""
    result = {}
    for name, item in payload["layer_tensors"].items():
        if "qweight" in item:
            value = item["qweight"].float() * item["scale"].float()
        else:
            value = item["weight_fp16"]
        result[name] = value.to(weights[name].dtype)
    return result


def allocation_action_ids(compressed_layers, int8_layers, model_layer_count):
    """Encode per-layer NEC and dense-INT8 choices for the Fisher option proxy."""
    return [
        *sorted(int(layer) for layer in compressed_layers),
        *(model_layer_count + int(layer) for layer in sorted(int8_layers)),
    ]


def allocation_record(compressed_layers, int8_layers=()):
    return {
        "compressed_layers": sorted(int(layer) for layer in compressed_layers),
        "int8_dense_layers": sorted(int(layer) for layer in int8_layers),
    }


@torch.inference_mode()
def collect_logits(model, ids, *, seq_len=2048, stride=512):
    """Collect one row for each unique causal next-token target."""
    windows, _ = sliding_window_specifications(
        ids.shape[1], seq_len=seq_len, stride=stride, max_eval_tokens=0,
    )
    logits_parts, target_parts = [], []
    for window in windows:
        input_ids = ids[:, window["begin"]:window["end"]].to(model.device)
        target_ids = input_ids.clone()
        if window["target_length"] < input_ids.shape[1]:
            target_ids[:, :-window["target_length"]] = -100
        labels = target_ids[:, 1:]
        mask = labels != -100
        if int(mask.sum().item()) != window["scored_tokens"]:
            raise RuntimeError("Selection window target count changed")
        logits = model(input_ids, use_cache=False).logits[:, :-1][mask]
        logits_parts.append(logits.detach().to(device="cpu", dtype=torch.float16))
        target_parts.append(labels[mask].detach().to(device="cpu", dtype=torch.long))
    return torch.cat(logits_parts), torch.cat(target_parts)


def token_logit_metrics(teacher_logits, student_logits, targets, *, device, chunk_tokens=64):
    """Compute token-weighted PPL, KL, accuracy, and dense top-1 agreement."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and student logits must have matching shapes")
    if teacher_logits.shape[0] != targets.numel():
        raise ValueError("Logit rows and target IDs have different lengths")
    total_nll = 0.0
    total_kl = 0.0
    student_correct = 0
    top1_agreement = 0
    total = int(targets.numel())
    for begin in range(0, total, chunk_tokens):
        end = min(begin + chunk_tokens, total)
        teacher = teacher_logits[begin:end].to(device=device, dtype=torch.float32)
        student = student_logits[begin:end].to(device=device, dtype=torch.float32)
        labels = targets[begin:end].to(device=device, dtype=torch.long)
        log_teacher = F.log_softmax(teacher, dim=-1)
        log_student = F.log_softmax(student, dim=-1)
        rows = torch.arange(end - begin, device=device)
        total_nll += float((-log_student[rows, labels]).double().sum())
        total_kl += float(F.kl_div(
            log_student, log_teacher, reduction="none", log_target=True,
        ).sum(dim=-1).double().sum())
        teacher_top1 = teacher.argmax(dim=-1)
        student_top1 = student.argmax(dim=-1)
        student_correct += int((student_top1 == labels).sum())
        top1_agreement += int((student_top1 == teacher_top1).sum())
    mean_nll = total_nll / total
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "mean_kl_nats_per_token": total_kl / total,
        "next_token_accuracy": student_correct / total,
        "dense_top1_agreement": top1_agreement / total,
        "scored_tokens": total,
    }


@torch.inference_mode()
def evaluate_ppl_kl(model, ids, dense, compressed, selected, *, int8_dense=None,
                    max_eval_tokens=0, seq_len=2048, stride=512,
                    chunk_tokens=128, label="heldout"):
    """Stream a dense-teacher and selected-student evaluation over one split."""
    windows, eval_limit = sliding_window_specifications(
        ids.shape[1], seq_len=seq_len, stride=stride,
        max_eval_tokens=max_eval_tokens,
    )
    totals = {key: 0.0 for key in ("student_nll", "teacher_nll", "kl")}
    student_correct = 0
    top1_agreement = 0
    total_tokens = 0
    started = time.monotonic()
    for index, window in enumerate(windows, start=1):
        input_ids = ids[:, window["begin"]:window["end"]].to(model.device)
        target_ids = input_ids.clone()
        if window["target_length"] < input_ids.shape[1]:
            target_ids[:, :-window["target_length"]] = -100
        labels = target_ids[:, 1:]
        mask = labels != -100
        if int(mask.sum().item()) != window["scored_tokens"]:
            raise RuntimeError("Held-out target count changed")

        install_allocation(model, dense, compressed, set())
        teacher_logits = model(input_ids, use_cache=False).logits[:, :-1][mask]
        install_allocation(
            model, dense, compressed, selected,
            int8_dense=int8_dense,
        )
        student_logits = model(input_ids, use_cache=False).logits[:, :-1][mask]
        teacher_logits = teacher_logits.detach().to(dtype=torch.float32)
        student_logits = student_logits.detach().to(dtype=torch.float32)
        scored_labels = labels[mask]
        for begin in range(0, window["scored_tokens"], chunk_tokens):
            end = min(begin + chunk_tokens, window["scored_tokens"])
            teacher = teacher_logits[begin:end]
            student = student_logits[begin:end]
            chunk_labels = scored_labels[begin:end]
            log_teacher = F.log_softmax(teacher, dim=-1)
            log_student = F.log_softmax(student, dim=-1)
            rows = torch.arange(end - begin, device=model.device)
            totals["student_nll"] += float((-log_student[rows, chunk_labels]).double().sum())
            totals["teacher_nll"] += float((-log_teacher[rows, chunk_labels]).double().sum())
            totals["kl"] += float(F.kl_div(
                log_student, log_teacher, reduction="none", log_target=True,
            ).sum(dim=-1).double().sum())
            teacher_top1 = teacher.argmax(dim=-1)
            student_top1 = student.argmax(dim=-1)
            student_correct += int((student_top1 == chunk_labels).sum())
            top1_agreement += int((student_top1 == teacher_top1).sum())
        total_tokens += window["scored_tokens"]
        if index == 1 or index % 10 == 0 or index == len(windows):
            print(
                f"{label} progress windows={index}/{len(windows)} "
                f"tokens={total_tokens}/{sum(w['scored_tokens'] for w in windows)} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )
        del input_ids, target_ids, labels, mask, teacher_logits, student_logits

    if total_tokens != sum(window["scored_tokens"] for window in windows):
        raise RuntimeError("Held-out evaluation did not score the requested token count")
    mean_nll = totals["student_nll"] / total_tokens
    teacher_mean_nll = totals["teacher_nll"] / total_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "teacher_perplexity": math.exp(teacher_mean_nll),
        "teacher_mean_nll": teacher_mean_nll,
        "teacher_kl_nats_per_token": totals["kl"] / total_tokens,
        "next_token_accuracy": student_correct / total_tokens,
        "dense_top1_agreement": top1_agreement / total_tokens,
        "scored_tokens": total_tokens,
        "input_tokens": eval_limit,
        "context": seq_len,
        "stride": stride,
        "seconds": time.monotonic() - started,
    }


def sample_ids(tokenizer, split, start, count):
    text, metadata = load_wikitext2_raw_split(split)
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if start < 0 or count < 2 or start + count > token_ids.shape[1]:
        raise ValueError(
            f"Requested {split} token range [{start}, {start + count}) is outside "
            f"the {token_ids.shape[1]} available tokens"
        )
    ids = token_ids[:, start:start + count].contiguous()
    token_hash = hashlib.sha256(ids.to(torch.int64).numpy().tobytes()).hexdigest()
    return ids, metadata, token_hash, int(token_ids.shape[1])


def storage_summary(
    checkpoint_dir, dense, layers, selected, source_model_info=None,
    *, int8_layers=(), int8_artifact_bytes=None,
):
    """Count actual candidate artifacts and estimate the whole-model package bound."""
    int8_layers = set(int8_layers)
    int8_artifact_bytes = int8_artifact_bytes or {}
    per_layer = {}
    dense_total = 0
    allocated_total = 0
    retained_layer_bytes = 0
    for layer in layers:
        dense_bytes = sum(
            int(dense[layer][name].numel() * dense[layer][name].element_size())
            for name in (*PROJECTIONS, *NORMS)
        )
        checkpoint_bytes = int(
            (checkpoint_dir / f"compressed_layer_{layer}.pt").stat().st_size
        )
        if layer in selected:
            allocated_bytes = checkpoint_bytes
            representation = "NEC"
        elif layer in int8_layers:
            allocated_bytes = int(int8_artifact_bytes[layer])
            representation = "dense_int8"
        else:
            allocated_bytes = dense_bytes
            representation = "dense_fp16"
        dense_total += dense_bytes
        allocated_total += allocated_bytes
        if layer not in selected:
            retained_layer_bytes += allocated_bytes
        per_layer[str(layer)] = {
            "dense_projection_and_norm_bytes": dense_bytes,
            "serialized_checkpoint_bytes": checkpoint_bytes,
            "selected_as_compressed": layer in selected,
            "selected_representation": representation,
            "dense_int8_serialized_bytes": int8_artifact_bytes.get(layer),
            "allocated_bytes": allocated_bytes,
        }
    summary = {
        "scope": "candidate layers only; dense FP16 tensor payload plus actual serialized NEC checkpoint files",
        "candidate_layer_ids": list(layers),
        "compressed_layer_ids": sorted(selected),
        "int8_dense_layer_ids": sorted(int8_layers),
        "dense_fp16_layer_ids": sorted(set(layers) - set(selected) - int8_layers),
        "dense_layers_retained_in_scope": len(layers) - len(selected),
        "candidate_dense_bytes": dense_total,
        "allocated_candidate_bytes": allocated_total,
        "bytes_saved_vs_dense_candidates": dense_total - allocated_total,
        "retained_non_nec_layer_bytes": retained_layer_bytes,
        "retained_non_nec_layer_byte_fraction": retained_layer_bytes / dense_total if dense_total else 0.0,
        "non_nec_layer_fraction_in_scope": (
            (len(layers) - len(selected)) / len(layers) if layers else 0.0
        ),
        "scoped_compression_ratio": dense_total / allocated_total if allocated_total else None,
        "whole_deployable_package_measured": False,
        "scope_ratio_note": (
            "Scoped weight-storage ratio only. It uses exact dense tensor payload "
            "bytes and actual checkpoint file sizes; package-level metadata and "
            "loader/runtime files are outside this layer-weight accounting."
        ),
        "layers": per_layer,
    }
    if source_model_info is not None:
        payload_by_name = source_model_info["tensor_payload_bytes"]
        replaced_layers = set(selected) | int8_layers
        replaced_payload = sum(
            layer_compressible_payload(payload_by_name, layer) for layer in replaced_layers
        )
        package_lower_bound = (
            source_model_info["model_bytes"] - replaced_payload + sum(
                int((checkpoint_dir / f"compressed_layer_{layer}.pt").stat().st_size)
                for layer in selected
            ) + sum(int(int8_artifact_bytes[layer]) for layer in int8_layers)
        )
        summary.update({
            "source_complete_model_serialized_bytes": source_model_info["model_bytes"],
            "replaced_dense_tensor_payload_bytes": replaced_payload,
            "mixed_package_bytes_lower_bound": package_lower_bound,
            "whole_model_compression_ratio_upper_bound": (
                source_model_info["model_bytes"] / package_lower_bound
                if package_lower_bound else None
            ),
            "whole_deployable_package_measured": False,
            "whole_model_ratio_note": (
                "Preflight bound only. It subtracts replaced tensor payloads from "
                "the serialized source package and adds actual NEC checkpoint files; "
                "the mixed package has not been materialized or byte-counted."
            ),
        })
    return summary


def materialize_deployable_allocation_package(
    source_model_dir, checkpoint_dir, output_dir, layers, compressed_layers,
    int8_layers, int8_artifact_dir, model_revision,
):
    """Build and byte-count a mixed HF/NEC package with replaced tensors removed."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_model_dir = Path(source_model_dir)
    package_dir = Path(output_dir) / "deployable_model"
    if package_dir.exists():
        raise FileExistsError(package_dir)
    package_dir.mkdir(parents=True)

    index_path = source_model_dir / "model.safetensors.index.json"
    source_index = json.loads(index_path.read_text())
    weight_map = source_index["weight_map"]
    replaced_layers = set(compressed_layers) | set(int8_layers)
    replaced_names = set()
    for layer in replaced_layers:
        prefix = f"model.layers.{layer}."
        for projection in PROJECTIONS:
            module = "self_attn" if projection in PROJECTIONS[:4] else "mlp"
            replaced_names.add(f"{prefix}{module}.{projection}.weight")
        for norm in NORMS:
            replaced_names.add(f"{prefix}{norm}.weight")
    missing = replaced_names - set(weight_map)
    if missing:
        raise ValueError(
            "Source model index lacks tensors being replaced: "
            f"{sorted(missing)[:8]}"
        )

    updated_weight_map = {}
    updated_total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        source_shard = source_model_dir / shard_name
        destination_shard = package_dir / shard_name
        retained_tensors = {}
        with safe_open(source_shard, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name in replaced_names:
                    continue
                tensor = shard.get_tensor(name).contiguous()
                retained_tensors[name] = tensor
                updated_weight_map[name] = shard_name
                updated_total_size += tensor.numel() * tensor.element_size()
        if not retained_tensors:
            raise RuntimeError(f"No retained tensors remain in source shard {shard_name}")
        save_file(retained_tensors, destination_shard, metadata={"format": "pt"})
        del retained_tensors

    metadata = dict(source_index.get("metadata", {}))
    metadata["total_size"] = updated_total_size
    (package_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": metadata,
        "weight_map": updated_weight_map,
    }, indent=2) + "\n")

    shard_names = set(weight_map.values())
    for source_path in source_model_dir.rglob("*"):
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source_model_dir)
        if relative.as_posix() == "model.safetensors.index.json":
            continue
        if relative.as_posix() in shard_names or source_path.suffix == ".safetensors":
            continue
        if source_path.name.startswith("pytorch_model") and source_path.suffix == ".bin":
            continue
        destination = package_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    nec_dir = package_dir / "nec_checkpoints"
    nec_dir.mkdir()
    for layer in sorted(compressed_layers):
        shutil.copy2(
            checkpoint_dir / f"compressed_layer_{layer}.pt",
            nec_dir / f"compressed_layer_{layer}.pt",
        )
    shutil.copy2(checkpoint_dir / "model_manifest.json", package_dir / "nec_compile_manifest.json")
    if int8_layers:
        dense_int8_dir = package_dir / "int8_dense_checkpoints"
        dense_int8_dir.mkdir()
        for layer in sorted(int8_layers):
            shutil.copy2(
                int8_artifact_dir / f"int8_layer_{layer}.pt",
                dense_int8_dir / f"int8_layer_{layer}.pt",
            )
    shutil.copytree(
        ROOT / "nox", package_dir / "nox",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
    )

    package_loader = r'''#!/usr/bin/env python3
"""Load the custom mixed NEC/INT8 Llama package in this directory."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from nox.compiler import reconstruct_from_artifact

ROOT = Path(__file__).resolve().parent
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm")

def load_model(device_map="auto", torch_dtype=torch.float16):
    model = AutoModelForCausalLM.from_pretrained(
        ROOT, device_map=device_map, torch_dtype=torch_dtype, local_files_only=True,
    ).eval()
    for path in sorted((ROOT / "nec_checkpoints").glob("compressed_layer_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        layer = model.model.layers[int(data["layer_idx"])]
        modules = {
            "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }
        for name, module in modules.items():
            weight = reconstruct_from_artifact(
                data["compressed_artifacts"][name],
                device=module.weight.device, dtype=module.weight.dtype,
            )
            module.weight.data.copy_(weight)
        for name in NORMS:
            if data.get(name) is None:
                continue
            layer_module = getattr(layer, name)
            layer_module.weight.data.copy_(data[name].to(
                device=layer_module.weight.device, dtype=layer_module.weight.dtype,
            ))
    for path in sorted((ROOT / "int8_dense_checkpoints").glob("int8_layer_*.pt")):
        layer_id = int(path.stem.removeprefix("int8_layer_"))
        data = torch.load(path, map_location="cpu", weights_only=True)
        layer = model.model.layers[layer_id]
        modules = {
            "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
        }
        for name, item in data["layer_tensors"].items():
            if "qweight" in item:
                weight = item["qweight"].float() * item["scale"].float()
            else:
                weight = item["weight_fp16"]
            destination = modules[name].weight
            destination.data.copy_(weight.to(
                device=destination.device, dtype=destination.dtype,
            ))
    return model
'''
    (package_dir / "load_nec_package.py").write_text(package_loader)
    (package_dir / "NEC_PACKAGE_README.md").write_text(
        "# Mixed NEC model package\n\n"
        "Load with `load_nec_package.py`; this loads the full Llama architecture, "
        "then reconstructs NEC layers and dequantizes retained INT8 layers. "
        "Install PyTorch, Transformers, and safetensors in the runtime environment.\n"
    )
    package_manifest = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "quality_evaluation_architecture_layers": len(layers),
        "compressed_layers": sorted(compressed_layers),
        "int8_dense_layers": sorted(int8_layers),
        "fp16_dense_layers": sorted(set(layers) - set(compressed_layers) - set(int8_layers)),
        "replaced_tensor_names": sorted(replaced_names),
        "runtime_loader": "load_nec_package.py",
        "nonstandard_weight_loading_required": True,
    }
    (package_dir / "nec_allocation_manifest.json").write_text(
        json.dumps(package_manifest, indent=2) + "\n"
    )

    # Check the rebuilt index against the headers before reporting package bytes.
    rebuilt_keys = set()
    for shard_name in sorted(set(updated_weight_map.values())):
        with safe_open(package_dir / shard_name, framework="pt", device="cpu") as shard:
            rebuilt_keys.update(shard.keys())
    if rebuilt_keys != set(updated_weight_map):
        raise RuntimeError("Rebuilt safetensors shards do not match their index")
    file_sizes = {
        str(path.relative_to(package_dir)): path.stat().st_size
        for path in package_dir.rglob("*") if path.is_file()
    }
    return {
        "path": str(package_dir),
        "serialized_bytes": sum(file_sizes.values()),
        "file_count": len(file_sizes),
        "source_model_serialized_bytes": sum(
            path.stat().st_size for path in source_model_dir.rglob("*") if path.is_file()
        ),
        "compression_ratio": (
            sum(path.stat().st_size for path in source_model_dir.rglob("*") if path.is_file())
            / sum(file_sizes.values())
        ),
        "repacked_dense_tensor_payload_bytes": updated_total_size,
        "file_sizes": file_sizes,
    }


def ensure_selection_disjoint(manifest, split, start, count):
    """Prevent reuse of the text windows used to calibrate these checkpoints."""
    input_data = manifest.get("run_identity", {}).get("input_data", {})
    calibration_split = input_data.get("calibration_split")
    if split != calibration_split:
        return {
            "calibration_split": calibration_split,
            "selection_split": split,
            "overlap_count": 0,
            "overlap_check": "different canonical dataset splits",
        }
    offsets = input_data.get("calibration_window_offsets", [])
    width = int(input_data.get("activation_sequence_length", 128))
    sample_begin, sample_end = start, start + count
    overlaps = [
        int(offset) for offset in offsets
        if sample_begin < int(offset) + width and int(offset) < sample_end
    ]
    if overlaps:
        raise ValueError(
            "Selection token range overlaps checkpoint calibration windows at "
            f"offsets {overlaps[:8]}"
        )
    return {
        "calibration_split": calibration_split,
        "selection_split": split,
        "calibration_sequence_length": width,
        "calibration_window_count": len(offsets),
        "overlap_count": 0,
        "selection_token_start": start,
        "selection_token_end_exclusive": sample_end,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compress-count", type=int)
    parser.add_argument(
        "--keep-dense-layers", type=int,
        help="Model-wide dense-layer count. Requires enough candidate checkpoints.",
    )
    parser.add_argument(
        "--optimize-storage", action="store_true",
        help="Search compression counts and maximize measured savings subject to --max-ppl",
    )
    parser.add_argument(
        "--max-dense-layers", type=int,
        help="Hard model-wide dense-layer ceiling for --optimize-storage",
    )
    parser.add_argument("--max-ppl", type=float, default=10.0)
    parser.add_argument(
        "--protect-boundaries", action="store_true",
        help="Keep transformer layer 0 and the final layer dense when eligible.",
    )
    parser.add_argument("--search-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--search-start-token", type=int, default=4096)
    parser.add_argument("--search-tokens", type=int, default=1024)
    parser.add_argument("--refine-split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--refine-start-token", type=int, default=8192)
    parser.add_argument("--refine-tokens", type=int, default=4096)
    parser.add_argument("--refine-candidates", type=int, default=2)
    parser.add_argument(
        "--refine-candidates-per-count", type=int, default=2,
        help="Proxy shortlist size per feasible count in --optimize-storage mode",
    )
    parser.add_argument("--heldout-split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--heldout-start-token", type=int, default=0)
    parser.add_argument("--heldout-tokens", type=int, default=4096,
                        help="Use 0 for the full held-out split")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--proxy-chunk-tokens", type=int, default=32)
    parser.add_argument("--proxy-device", default="cuda")
    parser.add_argument("--exact-proxy-limit", type=int, default=256)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=1729)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    fixed_count_requested = args.compress_count is not None or args.keep_dense_layers is not None
    if args.optimize_storage:
        if fixed_count_requested:
            parser.error("--optimize-storage cannot be combined with a fixed-count option")
        if args.max_dense_layers is None:
            parser.error("--optimize-storage requires --max-dense-layers")
        if args.max_dense_layers < 0:
            parser.error("--max-dense-layers must be nonnegative")
    else:
        if args.max_dense_layers is not None:
            parser.error("--max-dense-layers requires --optimize-storage")
        if args.compress_count is not None and args.keep_dense_layers is not None:
            raise ValueError("Specify --compress-count or --keep-dense-layers, not both")
        if args.compress_count is None and args.keep_dense_layers is None:
            parser.error("Specify --compress-count or --keep-dense-layers")
    if args.heldout_split in {args.search_split, args.refine_split}:
        raise ValueError("Held-out data split must differ from both selection splits")
    if args.refine_candidates < 1:
        raise ValueError("--refine-candidates must be positive")
    if args.refine_candidates_per_count < 1:
        raise ValueError("--refine-candidates-per-count must be positive")
    if args.max_ppl <= 0:
        raise ValueError("--max-ppl must be positive")
    args.output.mkdir(parents=True)

    started = time.monotonic()
    manifest_path = args.checkpoint_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    layers = tuple(sorted(int(i) for i in manifest["run_identity"]["layers"]))
    checkpoint_hashes = {}
    for layer in layers:
        path = args.checkpoint_dir / f"compressed_layer_{layer}.pt"
        checkpoint_hashes[path.name] = sha256_file(path)
        expected = manifest.get("checkpoint_file_sha256", {}).get(path.name)
        if expected and expected != checkpoint_hashes[path.name]:
            raise ValueError(f"Checkpoint hash mismatch: {path.name}")
    if not layers:
        raise ValueError("Checkpoint manifest contains no candidate layers")
    calibration_range = ensure_selection_disjoint(
        manifest, args.search_split, args.search_start_token, args.search_tokens,
    )
    refine_range = ensure_selection_disjoint(
        manifest, args.refine_split, args.refine_start_token, args.refine_tokens,
    )
    if args.refine_split == args.search_split:
        search_begin = args.search_start_token
        search_end = search_begin + args.search_tokens
        refine_begin = args.refine_start_token
        refine_end = refine_begin + args.refine_tokens
        if search_begin < refine_end and refine_begin < search_end:
            raise ValueError("Proxy and exact-refinement token ranges overlap")

    print("PHASE=load_dense_model", flush=True)
    model, tokenizer, _, compatibility, _ = load_model(
        None, offload_folder=args.output / "offload", variant="dense",
    )
    model_layer_count = len(model.model.layers)
    if args.keep_dense_layers is not None:
        if not 0 <= args.keep_dense_layers <= model_layer_count:
            raise ValueError("--keep-dense-layers is outside the model layer range")
        compress_count = model_layer_count - args.keep_dense_layers
    elif args.optimize_storage:
        if args.max_dense_layers > model_layer_count:
            raise ValueError("--max-dense-layers is outside the model layer range")
        expected_candidates = tuple(range(model_layer_count))
        if layers != expected_candidates:
            raise ValueError(
                "Variable storage search requires a checkpoint for every model layer; "
                f"found {len(layers)} candidates for {model_layer_count} layers"
            )
        compress_count = None
    else:
        compress_count = args.compress_count
    protected_layers = (
        {0, model_layer_count - 1} if args.protect_boundaries else set()
    )
    eligible_layers = tuple(layer for layer in layers if layer not in protected_layers)
    if args.optimize_storage:
        minimum_compressed_count = max(0, model_layer_count - args.max_dense_layers)
        maximum_compressed_count = len(eligible_layers)
        if minimum_compressed_count > maximum_compressed_count:
            raise ValueError(
                f"A maximum of {len(eligible_layers)} layers can be compressed after "
                f"constraints, but at least {minimum_compressed_count} are required"
            )
        compression_counts = tuple(range(minimum_compressed_count, maximum_compressed_count + 1))
    else:
        if not 0 <= compress_count <= len(eligible_layers):
            raise ValueError(
                f"Requested {compress_count} compressed layers, but only candidate "
                f"layers {list(eligible_layers)} remain eligible after constraints"
            )
        compression_counts = (compress_count,)

    print("PHASE=inspect_source_storage", flush=True)
    # The runner caches the tokenizer/config/weight shards, but often omits
    # repository metadata such as README and license files. snapshot_download
    # rejects that otherwise usable cache as incomplete; resolve just the
    # pinned safetensors index needed by inspect_dense_model instead.
    from huggingface_hub import hf_hub_download
    model_index = Path(hf_hub_download(
        repo_id=MODEL_ID,
        filename="model.safetensors.index.json",
        revision=MODEL_REVISION,
        local_files_only=True,
    ))
    source_snapshot = model_index.parent
    source_model_info = inspect_dense_model(source_snapshot)

    dense = {
        layer: {
            name: layer_module(model.model.layers[layer], name).weight.detach().cpu().clone()
            for name in (*PROJECTIONS, *NORMS)
        }
        for layer in layers
    }
    injected = inject_nox_checkpoints(
        model, str(args.checkpoint_dir), expected_layers=list(layers),
    )
    if sorted(injected) != list(layers):
        raise RuntimeError(f"Injected layers {injected}; expected {list(layers)}")
    compressed = {
        layer: {
            name: layer_module(model.model.layers[layer], name).weight.detach().cpu().clone()
            for name in (*PROJECTIONS, *NORMS)
        }
        for layer in layers
    }

    search_ids, search_metadata, search_hash, search_split_tokens = sample_ids(
        tokenizer, args.search_split, args.search_start_token, args.search_tokens,
    )
    setup_seconds = time.monotonic() - started
    selection_started = time.monotonic()
    print(
        f"PHASE=single_layer_probes split={args.search_split} "
        f"token_range=[{args.search_start_token},{args.search_start_token + args.search_tokens})",
        flush=True,
    )
    t0 = time.monotonic()
    install_allocation(model, dense, compressed, set())
    dense_logits, targets = collect_logits(
        model, search_ids, seq_len=args.seq_len, stride=args.stride,
    )
    dense_seconds = time.monotonic() - t0
    singleton_logits = {}
    singleton_seconds = {}
    singleton_metrics = {}
    int8_payload_paths = {}
    int8_candidate_dir = Path(os.environ.get(
        "NEC_INT8_CANDIDATE_DIR", str(args.output / ".int8_candidates"),
    ))
    int8_artifact_bytes = {}
    int8_singleton_logits = {}
    int8_singleton_metrics = {}
    for layer in eligible_layers:
        if args.optimize_storage:
            payload, dequantized, serialized_bytes = quantize_layer_int8(dense[layer])
            int8_candidate_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = int8_candidate_dir / f"int8_layer_{layer}.pt"
            torch.save(payload, artifact_path)
            int8_payload_paths[layer] = artifact_path
            int8_artifact_bytes[layer] = artifact_path.stat().st_size
        t0 = time.monotonic()
        install_allocation(model, dense, compressed, {layer})
        logits, candidate_targets = collect_logits(
            model, search_ids, seq_len=args.seq_len, stride=args.stride,
        )
        if not torch.equal(targets, candidate_targets):
            raise RuntimeError("Dense and singleton probes used different target tokens")
        singleton_logits[layer] = logits
        singleton_metrics[str(layer)] = token_logit_metrics(
            dense_logits, logits, targets, device=model.device,
        )
        singleton_seconds[str(layer)] = time.monotonic() - t0
        print(
            f"SINGLE layer={layer} kl={singleton_metrics[str(layer)]['mean_kl_nats_per_token']:.6g} "
            f"ppl={singleton_metrics[str(layer)]['perplexity']:.6g} "
            f"seconds={singleton_seconds[str(layer)]:.1f}",
            flush=True,
        )

        if args.optimize_storage:
            t0 = time.monotonic()
            install_allocation(
                model, dense, compressed, {layer: "int8"},
                int8_dense={layer: dequantized},
            )
            logits, candidate_targets = collect_logits(
                model, search_ids, seq_len=args.seq_len, stride=args.stride,
            )
            if not torch.equal(targets, candidate_targets):
                raise RuntimeError("Dense and INT8 singleton probes used different target tokens")
            int8_singleton_logits[model_layer_count + layer] = logits
            int8_singleton_metrics[str(layer)] = token_logit_metrics(
                dense_logits, logits, targets, device=model.device,
            )
            singleton_seconds[f"int8_{layer}"] = time.monotonic() - t0
            print(
                f"SINGLE_INT8 layer={layer} "
                f"kl={int8_singleton_metrics[str(layer)]['mean_kl_nats_per_token']:.6g} "
                f"ppl={int8_singleton_metrics[str(layer)]['perplexity']:.6g} "
                f"artifact_bytes={serialized_bytes} "
                f"seconds={singleton_seconds[f'int8_{layer}']:.1f}",
                flush=True,
            )
            del payload, dequantized

    nec_proxy = fit_fisher_allocation_proxy(
        dense_logits, singleton_logits, targets,
        device=args.proxy_device,
        token_chunk_size=args.proxy_chunk_tokens,
    )
    if args.optimize_storage:
        combined_singleton_logits = {
            **singleton_logits,
            **int8_singleton_logits,
        }
        allocation_proxy = fit_fisher_allocation_proxy(
            dense_logits, combined_singleton_logits, targets,
            device=args.proxy_device,
            token_chunk_size=args.proxy_chunk_tokens,
        )
    else:
        allocation_proxy = nec_proxy
    del singleton_logits
    if args.optimize_storage:
        del int8_singleton_logits, combined_singleton_logits
    proxy_search_results = {}
    proxy_ranked_allocations = []
    proxy_shortlist = []
    proxy_seen = set()
    proxy_allocations_scored = 0
    proxy_search_strategies = {}
    proxy_optimal_within_search = True
    if args.optimize_storage:
        possible_allocation_count = sum(
            math.comb(len(eligible_layers), count)
            * (2 ** (len(layers) - count))
            for count in compression_counts
        )
        nec_artifact_savings = {}
        for layer in eligible_layers:
            dense_bytes = sum(
                int(dense[layer][name].numel() * dense[layer][name].element_size())
                for name in (*PROJECTIONS, *NORMS)
            )
            checkpoint_bytes = (args.checkpoint_dir / f"compressed_layer_{layer}.pt").stat().st_size
            nec_artifact_savings[layer] = dense_bytes - checkpoint_bytes

        for count in compression_counts:
            count_search = search_fixed_compression_count(
                nec_proxy, count, max_ppl=args.max_ppl,
                exact_proxy_limit=args.exact_proxy_limit,
                beam_width=args.beam_width,
                random_seed=args.random_seed + count,
            )
            proxy_search_results[str(count)] = {
                "strategy": count_search.strategy,
                "allocations_scored": count_search.proxy_allocations_scored,
                "proxy_optimal_within_search": count_search.is_proxy_optimal,
                "top_allocations": list(count_search.ranked_allocations[:max(8, args.refine_candidates_per_count * 2)]),
            }
            proxy_allocations_scored += count_search.proxy_allocations_scored
            proxy_search_strategies[str(count)] = count_search.strategy
            proxy_optimal_within_search = proxy_optimal_within_search and count_search.is_proxy_optimal
            proxy_ranked_allocations.extend(
                {**row, "compressed_count": count}
                for row in count_search.ranked_allocations[:max(8, args.refine_candidates_per_count * 2)]
            )
            sensitivity_order = sorted(
                eligible_layers,
                key=lambda layer: (
                    singleton_metrics[str(layer)]["mean_kl_nats_per_token"], layer,
                ),
            )
            sensitivity_mask = tuple(sorted(sensitivity_order[:count]))
            storage_mask = tuple(sorted(
                eligible_layers,
                key=lambda layer: (-nec_artifact_savings[layer], layer),
            )[:count])
            base_masks = {
                tuple(row["compressed_layers"])
                for row in count_search.ranked_allocations[
                    :args.refine_candidates_per_count
                ]
            }
            base_masks.update((sensitivity_mask, storage_mask))

            allocation_rows = []
            for mask in sorted(base_masks):
                remaining = tuple(layer for layer in layers if layer not in mask)
                for bitmask in range(1 << len(remaining)):
                    int8_layers = tuple(
                        layer for index, layer in enumerate(remaining)
                        if bitmask & (1 << index)
                    )
                    actions = allocation_action_ids(mask, int8_layers, model_layer_count)
                    estimate = allocation_proxy.predict(actions)
                    storage = storage_summary(
                        args.checkpoint_dir, dense, layers, set(mask), source_model_info,
                        int8_layers=int8_layers,
                        int8_artifact_bytes=int8_artifact_bytes,
                    )
                    allocation_rows.append({
                        **estimate,
                        **allocation_record(mask, int8_layers),
                        "compressed_count": count,
                        "predicted_quality_feasible": (
                            estimate["predicted_perplexity"] <= args.max_ppl
                        ),
                        "estimated_candidate_bytes_saved": storage[
                            "bytes_saved_vs_dense_candidates"
                        ],
                        "predicted_whole_model_compression_ratio_upper_bound": storage[
                            "whole_model_compression_ratio_upper_bound"
                        ],
                        "storage": storage,
                    })

            allocation_rows.sort(key=lambda row: (
                0 if row["predicted_quality_feasible"] else 1,
                -row["estimated_candidate_bytes_saved"]
                if row["predicted_quality_feasible"]
                else row["predicted_perplexity"],
                row["predicted_perplexity"],
                row["compressed_layers"],
                row["int8_dense_layers"],
            ))
            best_predicted = allocation_rows[:args.refine_candidates_per_count]
            best_storage = max(
                allocation_rows,
                key=lambda row: row["estimated_candidate_bytes_saved"],
            )
            selected_rows = list(best_predicted)
            if best_storage not in selected_rows:
                selected_rows.append(best_storage)
            proxy_ranked_allocations.extend(allocation_rows[:max(
                8, args.refine_candidates_per_count * 2,
            )])
            proxy_allocations_scored += len(allocation_rows)
            for row in selected_rows:
                key = (
                    tuple(row["compressed_layers"]),
                    tuple(row["int8_dense_layers"]),
                )
                if key not in proxy_seen:
                    proxy_shortlist.append({
                        "compressed_layers": key[0],
                        "int8_dense_layers": key[1],
                    })
                    proxy_seen.add(key)

        # The independent-sensitivity comparator chooses NEC layers by their
        # singleton KL and uses the option proxy to decide which remaining
        # dense layers to store as INT8.
        independent_candidates = []
        sensitivity_order = sorted(
            eligible_layers,
            key=lambda layer: (singleton_metrics[str(layer)]["mean_kl_nats_per_token"], layer),
        )
        for count in compression_counts:
            mask = tuple(sorted(sensitivity_order[:count]))
            remaining = tuple(layer for layer in layers if layer not in mask)
            variants = []
            for bitmask in range(1 << len(remaining)):
                int8_layers = tuple(
                    layer for index, layer in enumerate(remaining)
                    if bitmask & (1 << index)
                )
                estimate = allocation_proxy.predict(
                    allocation_action_ids(mask, int8_layers, model_layer_count),
                )
                storage = storage_summary(
                    args.checkpoint_dir, dense, layers, set(mask), source_model_info,
                    int8_layers=int8_layers,
                    int8_artifact_bytes=int8_artifact_bytes,
                )
                variants.append({
                    **estimate,
                    **allocation_record(mask, int8_layers),
                    "compressed_count": count,
                    "predicted_quality_feasible": estimate["predicted_perplexity"] <= args.max_ppl,
                    "estimated_bytes_saved": storage["bytes_saved_vs_dense_candidates"],
                })
            variants.sort(key=lambda row: (
                0 if row["predicted_quality_feasible"] else 1,
                -row["estimated_bytes_saved"]
                if row["predicted_quality_feasible"] else row["predicted_perplexity"],
                row["predicted_perplexity"],
            ))
            independent_candidates.extend(variants[:1])
            proxy_allocations_scored += len(variants)
        independent_choice = max(
            (row for row in independent_candidates if row["predicted_quality_feasible"]),
            key=lambda row: (row["estimated_bytes_saved"], -row["predicted_kl_nats_per_token"]),
            default=None,
        )
        if independent_choice is None:
            independent_choice = min(
                independent_candidates,
                key=lambda row: (row["predicted_perplexity"], -row["estimated_bytes_saved"]),
            )
        independent_sensitivity_layers = tuple(independent_choice["compressed_layers"])
        independent_sensitivity_int8_layers = tuple(independent_choice["int8_dense_layers"])
        independent_key = (independent_sensitivity_layers, independent_sensitivity_int8_layers)
        if independent_key not in proxy_seen:
            proxy_shortlist.append({
                "compressed_layers": independent_key[0],
                "int8_dense_layers": independent_key[1],
            })
            proxy_seen.add(independent_key)
        proxy_ranked_allocations.extend(independent_candidates)
    else:
        search = search_fixed_compression_count(
            nec_proxy, compress_count, max_ppl=args.max_ppl,
            exact_proxy_limit=args.exact_proxy_limit,
            beam_width=args.beam_width,
            random_seed=args.random_seed,
        )
        possible_allocation_count = math.comb(len(eligible_layers), compress_count)
        if args.refine_candidates >= possible_allocation_count:
            raise ValueError(
                "Exact refinement must evaluate fewer than all possible allocations; "
                "reduce --refine-candidates"
            )
        proxy_shortlist = [
            tuple(row["compressed_layers"])
            for row in search.ranked_allocations[:args.refine_candidates]
        ]
        independent_sensitivity_layers = tuple(sorted(
            eligible_layers,
            key=lambda layer: singleton_metrics[str(layer)]["mean_kl_nats_per_token"],
        )[:compress_count])
        independent_candidates = [{
            "compressed_layers": independent_sensitivity_layers,
            "compressed_count": compress_count,
            "predicted_perplexity": nec_proxy.predict(independent_sensitivity_layers)["predicted_perplexity"],
            "predicted_kl_nats_per_token": nec_proxy.predict(independent_sensitivity_layers)["predicted_kl_nats_per_token"],
            "predicted_quality_feasible": (
                nec_proxy.predict(independent_sensitivity_layers)["predicted_perplexity"] <= args.max_ppl
            ),
        }]
        proxy_ranked_allocations = list(search.ranked_allocations)
        proxy_allocations_scored = search.proxy_allocations_scored
        proxy_search_strategies[str(compress_count)] = search.strategy
        proxy_optimal_within_search = search.is_proxy_optimal
    if not args.optimize_storage and len(proxy_shortlist) >= possible_allocation_count:
        raise ValueError(
            "Exact refinement must evaluate fewer than all possible allocations; "
            "reduce the shortlist size"
        )
    selection_seconds = time.monotonic() - selection_started

    refine_ids, refine_metadata, refine_hash, refine_split_tokens = sample_ids(
        tokenizer, args.refine_split, args.refine_start_token, args.refine_tokens,
    )
    refinement_started = time.monotonic()
    print(
        f"PHASE=exact_shortlist_refinement split={args.refine_split} "
        f"candidates={len(proxy_shortlist)} token_range="
        f"[{args.refine_start_token},{args.refine_start_token + args.refine_tokens})",
        flush=True,
    )
    t0 = time.monotonic()
    install_allocation(model, dense, compressed, set())
    refine_teacher_logits, refine_targets = collect_logits(
        model, refine_ids, seq_len=args.seq_len, stride=args.stride,
    )
    refine_dense_seconds = time.monotonic() - t0
    refinement_rows = []
    refinement_seconds = {}
    for candidate in proxy_shortlist:
        if isinstance(candidate, dict):
            mask = tuple(candidate["compressed_layers"])
            int8_layers = tuple(candidate["int8_dense_layers"])
        else:
            mask = tuple(candidate)
            int8_layers = ()
        t0 = time.monotonic()
        choices = {layer: "nec" for layer in mask}
        choices.update({layer: "int8" for layer in int8_layers})
        candidate_int8_dense = {}
        for layer in int8_layers:
            payload = torch.load(int8_payload_paths[layer], map_location="cpu")
            candidate_int8_dense[layer] = dequantize_layer_int8(payload, dense[layer])
            del payload
        install_allocation(
            model, dense, compressed, choices,
            int8_dense=candidate_int8_dense if args.optimize_storage else None,
        )
        student_logits, student_targets = collect_logits(
            model, refine_ids, seq_len=args.seq_len, stride=args.stride,
        )
        if not torch.equal(refine_targets, student_targets):
            raise RuntimeError("Exact refinement candidates used different target tokens")
        metrics = token_logit_metrics(
            refine_teacher_logits, student_logits, refine_targets, device=model.device,
        )
        del candidate_int8_dense
        row = {
            **allocation_record(mask, int8_layers),
            "metrics": metrics,
        }
        row["quality_feasible"] = metrics["perplexity"] <= args.max_ppl
        row["storage"] = storage_summary(
            args.checkpoint_dir, dense, layers, set(mask), source_model_info,
            int8_layers=int8_layers,
            int8_artifact_bytes=int8_artifact_bytes,
        )
        refinement_rows.append(row)
        refinement_seconds[
            "nec_" + "_".join(str(i) for i in mask)
            + "_int8_" + "_".join(str(i) for i in int8_layers)
        ] = time.monotonic() - t0
        print(
            f"REFINED NEC={list(mask)} INT8={list(int8_layers)} "
            f"ppl={metrics['perplexity']:.6g} "
            f"kl={metrics['mean_kl_nats_per_token']:.6g} "
            f"seconds={refinement_seconds['nec_' + '_'.join(str(i) for i in mask) + '_int8_' + '_'.join(str(i) for i in int8_layers)]:.1f}",
            flush=True,
        )

    def refinement_key(row):
        metrics = row["metrics"]
        if args.optimize_storage:
            if row["quality_feasible"]:
                return (
                    0.0,
                    -row["storage"]["bytes_saved_vs_dense_candidates"],
                    -row["storage"]["whole_model_compression_ratio_upper_bound"],
                    metrics["perplexity"],
                    metrics["mean_kl_nats_per_token"],
                    row["compressed_layers"],
                    row["int8_dense_layers"],
                )
            return (
                1.0,
                metrics["perplexity"],
                metrics["mean_kl_nats_per_token"],
                row["compressed_layers"],
                row["int8_dense_layers"],
            )
        if row["quality_feasible"]:
            return (
                0.0, metrics["perplexity"], metrics["mean_kl_nats_per_token"],
                row["compressed_layers"], row["int8_dense_layers"],
            )
        return (
            1.0, metrics["perplexity"], metrics["mean_kl_nats_per_token"],
            row["compressed_layers"], row["int8_dense_layers"],
        )

    chosen_refinement = min(refinement_rows, key=refinement_key)
    selected = set(chosen_refinement["compressed_layers"])
    selected_int8 = set(chosen_refinement["int8_dense_layers"])
    for row in refinement_rows:
        row["selected_after_refinement"] = row is chosen_refinement
    selected_proxy = allocation_proxy.predict(
        allocation_action_ids(selected, selected_int8, model_layer_count),
    )
    selected_proxy.update({
        "compressed_layers": sorted(selected),
        "int8_dense_layers": sorted(selected_int8),
        "predicted_quality_feasible": selected_proxy["predicted_perplexity"] <= args.max_ppl,
    })

    int8_output_dir = args.output / "int8_dense_checkpoints"
    int8_artifact_files = {}
    if selected_int8:
        int8_output_dir.mkdir(parents=True, exist_ok=True)
        for layer in sorted(selected_int8):
            artifact_path = int8_output_dir / f"int8_layer_{layer}.pt"
            shutil.move(int8_payload_paths[layer], artifact_path)
            int8_payload_paths[layer] = artifact_path
            actual_bytes = artifact_path.stat().st_size
            if actual_bytes != int8_artifact_bytes[layer]:
                int8_artifact_bytes[layer] = actual_bytes
            int8_artifact_files[str(layer)] = {
                "path": str(artifact_path.relative_to(args.output)),
                "serialized_bytes": actual_bytes,
                "sha256": sha256_file(artifact_path),
            }
    if int8_candidate_dir.exists():
        shutil.rmtree(int8_candidate_dir)
    refinement_wall_seconds = time.monotonic() - refinement_started

    validation_ids, heldout_metadata, heldout_hash, heldout_split_tokens = sample_ids(
        tokenizer, args.heldout_split, args.heldout_start_token,
        args.heldout_tokens if args.heldout_tokens > 0 else 2,
    )
    if args.heldout_tokens == 0:
        full_text, heldout_metadata = load_wikitext2_raw_split(args.heldout_split)
        all_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=True).input_ids
        validation_ids = all_ids[:, args.heldout_start_token:].contiguous()
        heldout_split_tokens = int(all_ids.shape[1])
        heldout_hash = hashlib.sha256(
            validation_ids.to(torch.int64).numpy().tobytes()
        ).hexdigest()

    print(
        f"PHASE=independent_heldout split={args.heldout_split} "
        f"NEC={sorted(selected)} INT8={sorted(selected_int8)} "
        f"tokens={validation_ids.shape[1]}",
        flush=True,
    )
    heldout_allocation = {layer: "nec" for layer in selected}
    heldout_allocation.update({layer: "int8" for layer in selected_int8})
    selected_int8_dense = {}
    for layer in selected_int8:
        payload = torch.load(int8_payload_paths[layer], map_location="cpu")
        selected_int8_dense[layer] = dequantize_layer_int8(payload, dense[layer])
        del payload
    heldout = evaluate_ppl_kl(
        model, validation_ids, dense, compressed, heldout_allocation,
        int8_dense=selected_int8_dense if args.optimize_storage else None,
        max_eval_tokens=0, seq_len=args.seq_len, stride=args.stride,
        label="heldout",
    )
    allocation_feasibility = {
        "max_ppl": args.max_ppl,
        "refinement_ppl_passed": chosen_refinement["quality_feasible"],
        "heldout_ppl_passed": heldout["perplexity"] <= args.max_ppl,
        "dense_layer_cap_passed": (
            len(layers) - len(selected) <= args.max_dense_layers
            if args.optimize_storage else True
        ),
        "status": (
            "FEASIBLE"
            if chosen_refinement["quality_feasible"]
            and heldout["perplexity"] <= args.max_ppl
            and (not args.optimize_storage or len(layers) - len(selected) <= args.max_dense_layers)
            else "INFEASIBLE"
        ),
    }
    selected_storage = storage_summary(
        args.checkpoint_dir, dense, layers, selected, source_model_info,
        int8_layers=selected_int8,
        int8_artifact_bytes=int8_artifact_bytes,
    )
    if args.optimize_storage:
        # Free the retained CPU copies before repacking the source safetensors.
        del dense, compressed, selected_int8_dense
        gc.collect()
        package_measurement = materialize_deployable_allocation_package(
            source_snapshot, args.checkpoint_dir, args.output,
            layers, selected, selected_int8, int8_output_dir, MODEL_REVISION,
        )
        selected_storage.update({
            "whole_deployable_package_measured": True,
            "actual_deployable_package": package_measurement,
            "actual_whole_model_compression_ratio": package_measurement["compression_ratio"],
            "ten_x_whole_model_target_met": package_measurement["compression_ratio"] >= 10.0,
            "greater_than_three_x_whole_model_ratio": package_measurement["compression_ratio"] > 3.0,
            "three_x_is_success_target": False,
            "compression_goal_assessment": (
                "10x aspiration met"
                if package_measurement["compression_ratio"] >= 10.0 else
                "10x aspiration not met; report the measured ratio and current representation limit"
            ),
            "whole_model_ratio_note": (
                "Measured by summing every serialized file in the rebuilt deployable package, "
                "including retained model tensors, NEC and INT8 artifacts, metadata, and loader code."
            ),
        })

    report = {
        "status": "COMPLETE",
        "scope": (
            "variable-count full-model mixed NEC / dense-INT8 / dense-FP16 allocation predicted from single-layer end-to-end logit probes"
            if args.optimize_storage else
            "fixed-count NEC layer allocation predicted from single-layer end-to-end logit probes"
        ),
        "method": {
            "name": "Fisher quadratic interaction proxy with conditional beam, swap search, and exact shortlist refinement",
            "proxy_objective": "second-order dense-teacher KL and NLL Taylor expansion in logit space",
            "allocation_objective": (
                "maximize actual serialized NEC plus dense-INT8 artifact savings subject to measured refinement PPL and the hard non-NEC layer cap; compare candidates on independent held-out data"
                if args.optimize_storage else
                "minimize measured refinement PPL under the fixed layer count"
            ),
            "variable_compression_counts_searched": list(compression_counts),
            "maximum_dense_layers": args.max_dense_layers,
            "search_strategy_by_compressed_count": proxy_search_strategies,
            "search_details_by_compressed_count": proxy_search_results,
            "uses_prior_multi_layer_rankings": False,
            "multi_layer_full_model_evaluations_during_proxy_search": 0,
            "multi_layer_full_model_evaluations_during_selection": len(proxy_shortlist),
            "selection_model_evaluation_count": (
                1 + len(eligible_layers) * (2 if args.optimize_storage else 1)
            ),
            "selection_model_evaluations": [
                "dense reference",
                *[f"NEC_layer_{i}_only" for i in eligible_layers],
                *([f"INT8_layer_{i}_only" for i in eligible_layers] if args.optimize_storage else []),
            ],
            "exact_refinement_model_evaluation_count": 1 + len(proxy_shortlist),
            "exact_refinement_candidate_count": len(proxy_shortlist),
            "total_possible_allocations_across_tested_counts": possible_allocation_count,
            "refinement_is_strict_shortlist": len(proxy_shortlist) < possible_allocation_count,
            "exact_refinement_evaluations": [
                allocation_record(
                    candidate["compressed_layers"], candidate["int8_dense_layers"],
                ) if isinstance(candidate, dict)
                else allocation_record(candidate)
                for candidate in proxy_shortlist
            ],
            "heldout_model_evaluation_count": 2,
            "proxy_allocations_scored": proxy_allocations_scored,
            "proxy_search_strategy": (
                "conditional fixed-cardinality Fisher searches across allowed counts"
                if args.optimize_storage else proxy_search_strategies[str(compress_count)]
            ),
            "proxy_optimal_within_search": proxy_optimal_within_search,
            "max_ppl_constraint_applied_to": "proxy and exact-refinement selection samples; held-out PPL is the independent check",
            "exact_refinement_selection_rule": (
                "among candidates meeting max_ppl on the refinement sample, maximize bytes saved using actual checkpoint file sizes; if none pass, select the lowest-PPL candidate and mark infeasible"
                if args.optimize_storage else
                "choose the lowest-PPL shortlisted candidate that meets max_ppl; if none pass, choose the lowest-PPL candidate and mark infeasible"
            ),
            "total_model_configuration_runs_before_heldout": (
                1 + len(eligible_layers) * (2 if args.optimize_storage else 1)
                + 1 + len(proxy_shortlist)
            ),
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_compatibility": compatibility,
        "quality_evaluation_model_layer_count": model_layer_count,
        "allocation_scope_layers": list(layers),
        "allocation_scope_layer_count": len(layers),
        "eligible_candidate_layers": list(eligible_layers),
        "protected_layers": sorted(protected_layers),
        "requested_compressed_count": compress_count,
        "requested_max_dense_layers": args.max_dense_layers,
        "compressed_layer_count_in_scope": len(selected),
        "dense_layer_count_in_scope": len(layers) - len(selected),
        "dense_layer_count_in_model": model_layer_count - len(selected),
        "max_ppl": args.max_ppl,
        "selected_allocation": {
            "compressed_layers": sorted(selected),
            "int8_dense_layers": sorted(selected_int8),
            "fp16_dense_layers": sorted(set(layers) - selected - selected_int8),
            "proxy_estimate": selected_proxy,
            "exact_refinement_metrics": chosen_refinement["metrics"],
            "exact_refinement_quality_feasible": chosen_refinement["quality_feasible"],
            "int8_artifact_files": int8_artifact_files,
        },
        "storage": storage_summary(
            args.checkpoint_dir, dense, layers, selected, source_model_info,
            int8_layers=selected_int8,
            int8_artifact_bytes=int8_artifact_bytes,
        ) if not args.optimize_storage else selected_storage,
        "allocation_feasibility": allocation_feasibility,
        "overall_assessment": (
            "FIXED_COUNT_DIAGNOSTIC"
            if not args.optimize_storage else
            "QUALITY_AND_DENSE_CAP_FEASIBLE; 10X_COMPRESSION_ASPIRATION_MET"
            if allocation_feasibility["status"] == "FEASIBLE"
            and selected_storage["actual_whole_model_compression_ratio"] >= 10.0 else
            "QUALITY_AND_DENSE_CAP_FEASIBLE; 10X_COMPRESSION_ASPIRATION_NOT_MET"
            if allocation_feasibility["status"] == "FEASIBLE" else
            "QUALITY_OR_DENSE_CAP_CONSTRAINT_INFEASIBLE"
        ),
        "independent_sensitivity_baseline": {
            "compressed_layers": list(independent_sensitivity_layers),
            "int8_dense_layers": list(independent_sensitivity_int8_layers)
            if args.optimize_storage else [],
            "selection_rule": (
                "for each allowed count, choose the individually lowest singleton-KL layers; among proxy-feasible masks, select the largest exact serialized-byte savings"
                if args.optimize_storage else
                "choose the individually lowest measured singleton KL layers"
            ),
            "candidate_counts": independent_candidates,
            "warning": "baseline ignores interactions and is not the allocator's objective",
        },
        "fisher_interaction_proxy": {
            "action_order": [
                {"action_id": layer, "choice": "NEC", "layer": layer}
                for layer in eligible_layers
            ] + [
                {"action_id": model_layer_count + layer, "choice": "INT8", "layer": layer}
                for layer in (eligible_layers if args.optimize_storage else ())
            ],
            "single_action_nll_gradient": list(allocation_proxy.linear_nll),
            "pairwise_fisher_gram": [list(row) for row in allocation_proxy.fisher_gram],
            "scored_tokens": allocation_proxy.scored_tokens,
        },
        "single_layer_probe_metrics": singleton_metrics,
        "single_layer_int8_probe_metrics": int8_singleton_metrics,
        "proxy_ranked_allocations": proxy_ranked_allocations,
        "exact_refinement": {
            "split": args.refine_split,
            "split_metadata": refine_metadata,
            "token_ids_sha256": refine_hash,
            "split_token_count": refine_split_tokens,
            "token_start": args.refine_start_token,
            "token_end_exclusive": args.refine_start_token + args.refine_tokens,
            "scored_tokens": int(refine_targets.numel()),
            "checkpoint_calibration_disjointness": refine_range,
            "measured_candidates": refinement_rows,
            "selected_layers": sorted(selected),
            "selected_int8_dense_layers": sorted(selected_int8),
        },
        "selection_data": {
            "split": args.search_split,
            "split_metadata": search_metadata,
            "token_ids_sha256": search_hash,
            "split_token_count": search_split_tokens,
            "scored_tokens": int(targets.numel()),
            "token_start": args.search_start_token,
            "token_end_exclusive": args.search_start_token + args.search_tokens,
            "checkpoint_calibration_disjointness": calibration_range,
        },
        "heldout_data": {
            "split": args.heldout_split,
            "split_metadata": heldout_metadata,
            "token_ids_sha256": heldout_hash,
            "split_token_count": heldout_split_tokens,
            "token_start": args.heldout_start_token,
            "input_token_count": int(validation_ids.shape[1]),
            "is_disjoint_from_selection_splits": (
                args.heldout_split not in {args.search_split, args.refine_split}
            ),
            "metrics": heldout,
            "ppl_constraint_passed": heldout["perplexity"] <= args.max_ppl,
        },
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": checkpoint_hashes,
        "script_sha256": sha256_file(__file__),
        "allocator_module_sha256": sha256_file(ROOT / "nox/layer_allocator.py"),
        "selection_runtime": {
            "setup_seconds": setup_seconds,
            "dense_probe_seconds": dense_seconds,
            "singleton_probe_seconds": singleton_seconds,
            "selection_probe_wall_seconds": selection_seconds,
            "refinement_dense_seconds": refine_dense_seconds,
            "refinement_candidate_seconds": refinement_seconds,
            "refinement_wall_seconds": refinement_wall_seconds,
        },
        "protocol": {
            "context": args.seq_len,
            "stride": args.stride,
            "logits_dtype": "model output dtype, stored as FP16",
            "proxy_distribution_dtype": "FP32",
            "proxy_chunk_tokens": args.proxy_chunk_tokens,
        },
        "warning": (
            "The Fisher proxy models pairwise interactions between single-layer "
            "logit deltas but omits higher-order hidden-state interactions. This "
            "result is a proof of concept and does not establish 32-layer quality."
        ),
    }
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"COMPLETE selected={sorted(selected)} "
        f"heldout_ppl={heldout['perplexity']:.6g} "
        f"heldout_kl={heldout['teacher_kl_nats_per_token']:.6g} "
        f"report={args.output / 'report.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
