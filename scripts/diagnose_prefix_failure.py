#!/usr/bin/env python3
"""Read-only checkpoint investigation; no compression, fitting, or model saving.

Short validation PPLs are diagnostic controls, not replacements for the archived
full-split evaluation. Every variant starts from the same original dense weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import (
    compute_perplexity,
    inject_nox_checkpoints,
    sliding_window_specifications,
)
from benchmarks.evaluate_high_fidelity_nec import load_model
from nox.compiler import reconstruct_from_artifact
from nox.data import load_wikitext2_raw_split
from nox.evaluation import MODEL_ID, MODEL_REVISION, sha256_file
from nox.sequential_compiler import forward_transformer_block

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
NORMS = ("input_layernorm", "post_attention_layernorm")


def module(layer, name):
    if name in NORMS:
        return getattr(layer, name)
    return getattr(layer.self_attn if name in PROJECTIONS[:4] else layer.mlp, name)


def relative(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    return float((candidate - reference).norm() / reference.norm().clamp_min(1e-30))


def summarize(reference, candidate):
    """Rows are tokens. Report energy-weighted and equally weighted token errors."""
    reference = reference.detach().float().cpu().reshape(-1, reference.shape[-1])
    candidate = candidate.detach().float().cpu().reshape_as(reference)
    error = candidate - reference
    norms = reference.norm(dim=-1)
    token_errors = error.norm(dim=-1) / norms.clamp_min(1e-30)
    energy = reference.square().sum(dim=-1)
    channel_energy = reference.square().sum(dim=0)
    top = channel_energy.topk(min(8, reference.shape[-1]))
    return {
        "relative_error": relative(reference, candidate),
        "token_relative_error_mean": float(token_errors.mean()),
        "token_relative_error_quantiles": [float(v) for v in torch.quantile(token_errors, torch.tensor([0., .5, .9, .99, 1.], device=reference.device))],
        "token_norm_quantiles": [float(v) for v in torch.quantile(norms, torch.tensor([0., .5, .9, .99, 1.], device=reference.device))],
        "largest_token_energy_fraction": float(energy.max() / energy.sum().clamp_min(1e-30)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0)),
        "mean_token_cosine": float(torch.nn.functional.cosine_similarity(reference, candidate, dim=-1).mean()),
        "norm_ratio": float(candidate.norm() / reference.norm().clamp_min(1e-30)),
        "largest_channels": top.indices.tolist(),
        "largest_channel_energy_fractions": (top.values / channel_energy.sum().clamp_min(1e-30)).tolist(),
    }


@torch.inference_mode()
def install(model, dense, compressed, selected):
    for index in range(4):
        for name in (*PROJECTIONS, *NORMS):
            source = compressed[index] if name in selected.get(index, ()) else dense[index]
            destination = module(model.model.layers[index], name).weight
            destination.copy_(source[name].to(destination.device, destination.dtype))


@torch.inference_mode()
def capture(model, ids):
    states = {}
    handles = []
    for index in range(4):
        def hook(_module, args, kwargs, output, index=index):
            inp = args[0] if args else kwargs["hidden_states"]
            out = output[0] if isinstance(output, tuple) else output
            states[index] = {"input": inp.detach().cpu(), "output": out.detach().cpu()}
        handles.append(model.model.layers[index].register_forward_hook(hook, with_kwargs=True))
    try:
        model(ids.to(model.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return states


def columns(tensor, device):
    return tensor.reshape(-1, tensor.shape[-1]).T.contiguous().to(device=device, dtype=torch.float32)


@torch.inference_mode()
def teacher_student_kl(model, ids, dense, compressed, selected, max_eval_tokens,
                       seq_len=2048, stride=512, chunk_tokens=128):
    """Measure mean tokenwise KL(P_dense || P_selected) on the PPL windows."""
    windows, eval_limit = sliding_window_specifications(
        ids.shape[1], seq_len=seq_len, stride=stride,
        max_eval_tokens=max_eval_tokens,
    )
    token_kls = []
    for window in windows:
        input_ids = ids[:, window["begin"]:window["end"]].to(model.device)
        targets = input_ids.clone()
        if window["target_length"] < input_ids.shape[1]:
            targets[:, :-window["target_length"]] = -100
        labels = targets[:, 1:]
        mask = labels != -100
        if int(mask.sum().item()) != window["scored_tokens"]:
            raise RuntimeError("Teacher-student KL target count differs from the PPL window")

        install(model, dense, compressed, {})
        teacher_logits = model(input_ids, use_cache=False).logits[:, :-1][mask]
        teacher_logits = teacher_logits.detach().to(device="cpu", dtype=torch.float16)
        install(model, dense, compressed, selected)
        student_logits = model(input_ids, use_cache=False).logits[:, :-1][mask]
        student_logits = student_logits.detach().to(device="cpu", dtype=torch.float16)
        if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
            raise FloatingPointError("Non-finite logits in teacher-student KL evaluation")

        for begin in range(0, teacher_logits.shape[0], chunk_tokens):
            end = min(begin + chunk_tokens, teacher_logits.shape[0])
            teacher = teacher_logits[begin:end].to(device=model.device, dtype=torch.float32)
            student = student_logits[begin:end].to(device=model.device, dtype=torch.float32)
            log_teacher = F.log_softmax(teacher, dim=-1)
            log_student = F.log_softmax(student, dim=-1)
            per_token = F.kl_div(
                log_student, log_teacher, reduction="none", log_target=True,
            ).sum(dim=-1)
            token_kls.append(per_token.detach().cpu())
        del input_ids, targets, labels, mask, teacher_logits, student_logits

    values = torch.cat(token_kls).float()
    if values.numel() != sum(window["scored_tokens"] for window in windows):
        raise RuntimeError("Teacher-student KL token count differs from the PPL protocol")
    return {
        "direction": "KL(dense teacher || student)",
        "reduction": "mean over non-duplicated causal next-token targets",
        "mean_nats_per_token": float(values.mean()),
        "token_kl_quantiles_nats": [
            float(value) for value in torch.quantile(values, torch.tensor([0., .5, .9, .99, 1.]))
        ],
        "scored_tokens": int(values.numel()),
        "input_tokens": int(eval_limit),
        "token_ids_sha256": hashlib.sha256(
            ids[0, :eval_limit].to(torch.int64).contiguous().cpu().numpy().tobytes()
        ).hexdigest(),
        "context": seq_len,
        "stride": stride,
        "teacher_logits_dtype": "model output dtype, accumulated in float32",
        "distribution_computation_dtype": "float32",
    }


def storage_accounting(checkpoint_dir, dense, variants):
    """Compare four-layer dense tensor payloads with serialized layer checkpoints."""
    layers = {}
    dense_total = 0
    for index in range(4):
        dense_bytes = sum(
            int(dense[index][name].numel() * dense[index][name].element_size())
            for name in (*PROJECTIONS, *NORMS)
        )
        compressed_bytes = int(
            (checkpoint_dir / f"compressed_layer_{index}.pt").stat().st_size
        )
        dense_total += dense_bytes
        layers[str(index)] = {
            "dense_projection_and_norm_tensor_bytes": dense_bytes,
            "serialized_compressed_checkpoint_bytes": compressed_bytes,
            "bytes_saved_when_compressed": dense_bytes - compressed_bytes,
        }

    allocations = {}
    for name, selected in variants.items():
        compressed_layers = [
            index for index in range(4)
            if tuple(selected.get(index, ())) == PROJECTIONS
        ]
        partially_compressed = [
            index for index in range(4)
            if selected.get(index) and tuple(selected[index]) != PROJECTIONS
        ]
        if partially_compressed:
            continue
        variant_bytes = sum(
            layers[str(index)]["serialized_compressed_checkpoint_bytes"]
            if index in compressed_layers
            else layers[str(index)]["dense_projection_and_norm_tensor_bytes"]
            for index in range(4)
        )
        allocations[name] = {
            "compressed_layers": compressed_layers,
            "dense_layers": [index for index in range(4) if index not in compressed_layers],
            "four_layer_payload_bytes": variant_bytes,
            "bytes_saved_vs_all_dense": dense_total - variant_bytes,
        }
    return {
        "scope": "layers 0-3 only; suffix layers 4-31 are identical dense weights",
        "dense_payload_definition": "seven projection tensors and two norm tensors per layer",
        "compressed_payload_definition": "actual serialized compressed_layer_N.pt file size",
        "all_dense_four_layer_payload_bytes": dense_total,
        "layers": layers,
        "whole_layer_allocations": allocations,
    }


@torch.inference_mode()
def trace(model, dense, compressed, ids):
    install(model, dense, compressed, {})
    clean = capture(model, ids)
    install(model, dense, compressed, {i: PROJECTIONS for i in range(4)})
    altered = capture(model, ids)
    result = {}
    for index in range(4):
        options = dict(n_heads=model.config.num_attention_heads, n_kv_heads=model.config.num_key_value_heads,
                       seq_len=ids.shape[1], rms_norm_eps=model.config.rms_norm_eps)
        x = columns(clean[index]["input"], model.device)
        z = columns(altered[index]["input"], model.device)
        wd = {k: v.to(device=model.device, dtype=torch.float32) for k, v in dense[index].items()}
        wc = {k: v.to(device=model.device, dtype=torch.float32) for k, v in compressed[index].items()}
        dd, da, dm = forward_transformer_block(wd, x, **options)
        dc, _, _ = forward_transformer_block(wd, z, **options)
        cd, ca, cm = forward_transformer_block(wc, x, **options)
        cc, _, _ = forward_transformer_block(wc, z, **options)
        inherited, local = dc - dd, cc - dc
        denominator = dd.norm().clamp_min(1e-30)
        record = {
            "hf_input": summarize(clean[index]["input"], altered[index]["input"]),
            "hf_output": summarize(clean[index]["output"], altered[index]["output"]),
            "custom_dense_vs_hf": relative(columns(clean[index]["output"], model.device), dd),
            "custom_compressed_vs_hf": relative(columns(altered[index]["output"], model.device), cc),
            "same_clean_input_full_block": summarize(dd.T, cd.T),
            "same_clean_input_active_transform": summarize((da + dm).T, (ca + cm).T),
            "same_clean_input_attention": summarize(da.T, ca.T),
            "same_clean_input_mlp": summarize(dm.T, cm.T),
            "inherited_error_relative_to_clean_output": float(inherited.norm() / denominator),
            "local_error_on_changed_input_relative_to_clean_output": float(local.norm() / denominator),
            "total_error_relative_to_clean_output": float((cc - dd).norm() / denominator),
            "inherited_local_cosine": float(torch.nn.functional.cosine_similarity(inherited.flatten(), local.flatten(), dim=0)),
            "decomposition_identity_residual": float(((cc - dd) - (inherited + local)).norm() / denominator),
            "active_transform_to_full_output_norm_ratio": float((da + dm).norm() / denominator),
        }
        # The aggregate norm can hide errors on ordinary tokens behind a few large tokens.
        shape = clean[index]["output"].shape
        if shape[1] > 1:
            record["hf_output_excluding_first_position"] = summarize(clean[index]["output"][:, 1:], altered[index]["output"][:, 1:])
            energy = clean[index]["output"].float().square().sum(-1)
            record["first_position_energy_fraction"] = float(energy[:, 0].sum() / energy.sum())
        result[str(index)] = record
        print(f"TRACE layer={index} context={ids.shape[1]} same={record['same_clean_input_full_block']['relative_error']:.6g} active={record['same_clean_input_active_transform']['relative_error']:.6g} cascade={record['hf_output']['relative_error']:.6g} token_mean={record['hf_output']['token_relative_error_mean']:.6g}", flush=True)
        del wd, wc, x, z, dd, dc, cd, cc, da, dm, ca, cm, inherited, local
    return result, clean


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    manifest_path = args.checkpoint_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {}
    for i in range(4):
        path = args.checkpoint_dir / f"compressed_layer_{i}.pt"
        hashes[path.name] = sha256_file(path)
        assert hashes[path.name] == manifest["checkpoint_file_sha256"][path.name]
    record = {"status": "RUNNING", "scope": "saved four-layer checkpoint diagnostic; no fitting",
              "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "diagnostic_script_sha256": sha256_file(__file__), "torch": torch.__version__,
              "checkpoint_manifest_sha256": sha256_file(manifest_path), "checkpoint_sha256": hashes,
              "ppl_protocol": {"split": "validation", "context": 2048, "stride": 512, "max_eval_tokens": args.max_eval_tokens, "batch_size": 1},
              "decoding": {}, "ppl": {}, "teacher_student_kl": {}, "traces": {}}

    def save():
        record["elapsed_seconds"] = time.monotonic() - start
        temporary = args.output / "report.partial.json"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(args.output / "report.json")

    print("PHASE=load_dense_model", flush=True)
    model, tokenizer, _, compatibility, _ = load_model(None, variant="dense", offload_folder=args.output / "offload")
    record["compatibility"] = compatibility
    record["gpu"] = torch.cuda.get_device_name()
    dense = {i: {name: module(model.model.layers[i], name).weight.detach().cpu().clone()
                 for name in (*PROJECTIONS, *NORMS)} for i in range(4)}
    injected = inject_nox_checkpoints(model, str(args.checkpoint_dir), expected_layers=[0, 1, 2, 3])
    assert injected == [0, 1, 2, 3]
    compressed = {i: {name: module(model.model.layers[i], name).weight.detach().cpu().clone()
                      for name in (*PROJECTIONS, *NORMS)} for i in range(4)}
    fp32decoded = {}
    for i in range(4):
        artifact = torch.load(args.checkpoint_dir / f"compressed_layer_{i}.pt", map_location="cpu", weights_only=True)
        assert artifact["layer_idx"] == i
        assert not artifact.get("cross_layer_bridge")
        assert all(torch.equal(dense[i][n], compressed[i][n]) for n in NORMS)
        fp32decoded[i] = {n: dense[i][n] for n in NORMS}
        for name in PROJECTIONS:
            original = artifact["compressed_artifacts"][name]
            decoded16 = reconstruct_from_artifact(original, device=model.device, dtype=torch.float16)
            assert torch.equal(decoded16.cpu(), compressed[i][name]), (i, name)
            decoded32 = reconstruct_from_artifact(original, device=model.device, dtype=torch.float32)
            fp32decoded[i][name] = decoded32.half().cpu()
            record["decoding"][f"{i}.{name}"] = {"installed_weights_equal_fp16_decode": True,
                "fp16_vs_fp32_decode_relative_error": relative(decoded32, decoded16),
                "finite": bool(torch.isfinite(decoded16).all())}
            del decoded16, decoded32
    variants = {
        "dense": {},
        **{f"layer{i}_only": {i: PROJECTIONS} for i in range(4)},
        **{f"prefix{n}": {i: PROJECTIONS for i in range(n)} for n in (2, 3, 4)},
        "prefix4_layer0_dense": {i: PROJECTIONS for i in range(1, 4)},
        "prefix4_layer3_dense": {i: PROJECTIONS for i in range(3)},
        "prefix4_boundaries_dense": {i: PROJECTIONS for i in (1, 2)},
        "prefix4_attention_only": {i: PROJECTIONS[:4] for i in range(4)},
        "prefix4_mlp_only": {i: PROJECTIONS[4:] for i in range(4)},
        "prefix4_fp32_decode_then_cast": {i: PROJECTIONS for i in range(4)},
    }
    record["storage"] = storage_accounting(args.checkpoint_dir, dense, variants)
    kl_variants = {
        "prefix4",
        "prefix4_layer0_dense",
        "prefix4_layer3_dense",
        "prefix4_boundaries_dense",
    }
    save()
    text, data = load_wikitext2_raw_split("validation")
    record["dataset"] = data
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    record["token_ids_sha256"] = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    dense_predictions = None
    for name, selected in variants.items():
        print(f"PHASE=ppl variant={name}", flush=True)
        install(model, dense, fp32decoded if name.endswith("then_cast") else compressed, selected)
        t = time.monotonic()
        ppl, count, details = compute_perplexity(model, tokenizer, text, seq_len=2048, stride=512,
            max_eval_tokens=args.max_eval_tokens, device=str(model.device), return_details=True,
            verify_reference_loss=True, return_token_nll=True, progress_interval=1)
        predictions = details.pop("top1_predictions")
        nll = details.pop("token_nll")
        if dense_predictions is None:
            dense_predictions = predictions
        torch.save({"token_nll": nll, "predictions": predictions}, args.output / f"{name}.pt")
        record["ppl"][name] = {"perplexity": ppl, "scored_tokens": count, "seconds": time.monotonic() - t,
            "mean_nll": details["mean_nll"], "next_token_accuracy": details["next_token_accuracy"],
            "dense_top1_agreement": float((predictions == dense_predictions).float().mean()),
            "nll_formula_crosscheck": details.get("nll_formula_crosscheck"),
            "selected": {str(k): list(v) for k, v in selected.items()}}
        allocation = record["storage"]["whole_layer_allocations"].get(name)
        if allocation is not None:
            record["ppl"][name]["storage"] = allocation
        if name in kl_variants:
            print(f"PHASE=teacher_student_kl variant={name}", flush=True)
            kl = teacher_student_kl(
                model, ids, dense, compressed, selected,
                max_eval_tokens=args.max_eval_tokens,
            )
            if kl["scored_tokens"] != count or kl["token_ids_sha256"] != details["token_ids_sha256"]:
                raise RuntimeError("Teacher-student KL and PPL use different scored tokens")
            record["teacher_student_kl"][name] = kl
            record["ppl"][name]["teacher_student_kl"] = kl
            print(f"RESULT teacher_student_kl {name} {json.dumps(kl)}", flush=True)
        print(f"RESULT {name} {json.dumps(record['ppl'][name])}", flush=True)
        save()
    del fp32decoded
    offsets = manifest["run_identity"]["input_data"]["validation_window_offsets"]
    saved_ids = torch.stack([ids[0, j:j + 128] for j in offsets])
    for name, trace_ids in (("saved_validation_128", saved_ids), ("first2048_long", ids[:, :2048]),
                            ("first2048_reset128", ids[:, :2048].reshape(-1, 128))):
        print(f"PHASE=trace name={name}", flush=True)
        traced, clean = trace(model, dense, compressed, trace_ids)
        record["traces"][name] = traced
        if name == "saved_validation_128":
            record["saved_input_parity"] = {}
            for i in range(4):
                cached = torch.load(args.checkpoint_dir / f"dense_input_layer_{i}.pt", map_location="cpu", weights_only=True)["X_test"]
                record["saved_input_parity"][str(i)] = summarize(cached.T, clean[i]["input"].reshape(-1, cached.shape[0]))
        save()
    record["status"] = "COMPLETE"
    record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save()
    print(f"COMPLETE report={args.output / 'report.json'} elapsed={record['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
