#!/usr/bin/env python3
"""Predict a NEC layer allocation from single-layer logit probes.

The search never loads prior multi-layer allocation results. It measures the
dense model and each eligible one-layer replacement on a selection sample,
fits a Fisher logit-interaction proxy, and chooses a fixed-count allocation.
The chosen mask is then evaluated on a separate, untouched split.
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


def install_allocation(model, dense, compressed, selected):
    """Install one candidate whole-layer mask into the HF model."""
    for layer in dense:
        source = compressed[layer] if layer in selected else dense[layer]
        for name in (*PROJECTIONS, *NORMS):
            destination = layer_module(model.model.layers[layer], name).weight
            destination.data.copy_(source[name].to(destination.device, destination.dtype))


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
def evaluate_ppl_kl(model, ids, dense, compressed, selected, *, max_eval_tokens=0,
                    seq_len=2048, stride=512, chunk_tokens=128, label="heldout"):
    """Stream a dense-teacher and selected-student evaluation over one split."""
    windows, eval_limit = sliding_window_specifications(
        ids.shape[1], seq_len=seq_len, stride=stride,
        max_eval_tokens=max_eval_tokens,
    )
    totals = {key: 0.0 for key in ("nll", "kl")}
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
        install_allocation(model, dense, compressed, set(selected))
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
            totals["nll"] += float((-log_student[rows, chunk_labels]).double().sum())
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
    mean_nll = totals["nll"] / total_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
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


def storage_summary(checkpoint_dir, dense, layers, selected):
    """Count serialized NEC bytes versus dense tensor payload for candidates."""
    per_layer = {}
    dense_total = 0
    allocated_total = 0
    for layer in layers:
        dense_bytes = sum(
            int(dense[layer][name].numel() * dense[layer][name].element_size())
            for name in (*PROJECTIONS, *NORMS)
        )
        checkpoint_bytes = int(
            (checkpoint_dir / f"compressed_layer_{layer}.pt").stat().st_size
        )
        allocated_bytes = checkpoint_bytes if layer in selected else dense_bytes
        dense_total += dense_bytes
        allocated_total += allocated_bytes
        per_layer[str(layer)] = {
            "dense_projection_and_norm_bytes": dense_bytes,
            "serialized_checkpoint_bytes": checkpoint_bytes,
            "selected_as_compressed": layer in selected,
            "allocated_bytes": allocated_bytes,
        }
    return {
        "scope": "candidate layers only; all other model layers remain dense",
        "candidate_dense_bytes": dense_total,
        "allocated_candidate_bytes": allocated_total,
        "bytes_saved_vs_dense_candidates": dense_total - allocated_total,
        "layers": per_layer,
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
    if args.compress_count is not None and args.keep_dense_layers is not None:
        raise ValueError("Specify --compress-count or --keep-dense-layers, not both")
    if args.compress_count is None and args.keep_dense_layers is None:
        parser.error("Specify --compress-count or --keep-dense-layers")
    if args.heldout_split in {args.search_split, args.refine_split}:
        raise ValueError("Held-out data split must differ from both selection splits")
    if args.refine_candidates < 1:
        raise ValueError("--refine-candidates must be positive")
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
    else:
        compress_count = args.compress_count
    protected_layers = (
        {0, model_layer_count - 1} if args.protect_boundaries else set()
    )
    eligible_layers = tuple(layer for layer in layers if layer not in protected_layers)
    if not 0 <= compress_count <= len(eligible_layers):
        raise ValueError(
            f"Requested {compress_count} compressed layers, but only candidate "
            f"layers {list(eligible_layers)} remain eligible after constraints"
        )

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
    if injected != list(layers):
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
    for layer in eligible_layers:
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

    proxy = fit_fisher_allocation_proxy(
        dense_logits, singleton_logits, targets,
        device=args.proxy_device,
        token_chunk_size=args.proxy_chunk_tokens,
    )
    search = search_fixed_compression_count(
        proxy, compress_count, max_ppl=args.max_ppl,
        exact_proxy_limit=args.exact_proxy_limit,
        beam_width=args.beam_width,
        random_seed=args.random_seed,
    )
    selection_seconds = time.monotonic() - selection_started
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
    for mask in proxy_shortlist:
        t0 = time.monotonic()
        install_allocation(model, dense, compressed, set(mask))
        student_logits, student_targets = collect_logits(
            model, refine_ids, seq_len=args.seq_len, stride=args.stride,
        )
        if not torch.equal(refine_targets, student_targets):
            raise RuntimeError("Exact refinement candidates used different target tokens")
        metrics = token_logit_metrics(
            refine_teacher_logits, student_logits, refine_targets, device=model.device,
        )
        row = {"compressed_layers": list(mask), "metrics": metrics}
        row["quality_feasible"] = metrics["perplexity"] <= args.max_ppl
        refinement_rows.append(row)
        refinement_seconds["_".join(str(i) for i in mask)] = time.monotonic() - t0
        print(
            f"REFINED layers={list(mask)} ppl={metrics['perplexity']:.6g} "
            f"kl={metrics['mean_kl_nats_per_token']:.6g} "
            f"seconds={refinement_seconds['_'.join(str(i) for i in mask)]:.1f}",
            flush=True,
        )

    def refinement_key(row):
        metrics = row["metrics"]
        if row["quality_feasible"]:
            return (0.0, metrics["perplexity"], metrics["mean_kl_nats_per_token"], row["compressed_layers"])
        return (1.0, metrics["perplexity"], metrics["mean_kl_nats_per_token"], row["compressed_layers"])

    chosen_refinement = min(refinement_rows, key=refinement_key)
    selected = set(chosen_refinement["compressed_layers"])
    for row in refinement_rows:
        row["selected_after_refinement"] = row is chosen_refinement
    selected_proxy = proxy.predict(selected)
    selected_proxy.update({
        "compressed_layers": sorted(selected),
        "predicted_quality_feasible": selected_proxy["predicted_perplexity"] <= args.max_ppl,
    })
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
        f"selected={sorted(selected)} tokens={validation_ids.shape[1]}",
        flush=True,
    )
    heldout = evaluate_ppl_kl(
        model, validation_ids, dense, compressed, selected,
        max_eval_tokens=0, seq_len=args.seq_len, stride=args.stride,
        label="heldout",
    )

    report = {
        "status": "COMPLETE",
        "scope": "fixed-count NEC layer allocation predicted from single-layer end-to-end logit probes",
        "method": {
            "name": "Fisher quadratic interaction proxy with conditional beam, swap search, and exact shortlist refinement",
            "proxy_objective": "second-order dense-teacher KL and NLL Taylor expansion in logit space",
            "uses_prior_multi_layer_rankings": False,
            "multi_layer_full_model_evaluations_during_proxy_search": 0,
            "multi_layer_full_model_evaluations_during_selection": len(proxy_shortlist),
            "selection_model_evaluation_count": 1 + len(eligible_layers),
            "selection_model_evaluations": ["dense reference", *[f"layer_{i}_only" for i in eligible_layers]],
            "exact_refinement_model_evaluation_count": 1 + len(proxy_shortlist),
            "exact_refinement_candidate_count": len(proxy_shortlist),
            "total_possible_allocations": possible_allocation_count,
            "refinement_is_strict_shortlist": True,
            "exact_refinement_evaluations": [list(mask) for mask in proxy_shortlist],
            "heldout_model_evaluation_count": 2,
            "proxy_allocations_scored": search.proxy_allocations_scored,
            "proxy_search_strategy": search.strategy,
            "proxy_optimal_within_search": search.is_proxy_optimal,
            "max_ppl_constraint_applied_to": "proxy and exact-refinement selection samples; held-out PPL is the independent check",
            "total_model_configuration_runs_before_heldout": (
                1 + len(eligible_layers) + 1 + len(proxy_shortlist)
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
        "model_layer_count": model_layer_count,
        "candidate_layers": list(layers),
        "eligible_candidate_layers": list(eligible_layers),
        "protected_layers": sorted(protected_layers),
        "requested_compressed_count": compress_count,
        "resulting_dense_layer_count": model_layer_count - compress_count,
        "max_ppl": args.max_ppl,
        "selected_allocation": {
            "compressed_layers": sorted(selected),
            "proxy_estimate": selected_proxy,
            "exact_refinement_metrics": chosen_refinement["metrics"],
            "exact_refinement_quality_feasible": chosen_refinement["quality_feasible"],
        },
        "storage": storage_summary(args.checkpoint_dir, dense, layers, selected),
        "independent_sensitivity_baseline": {
            "compressed_layers": list(independent_sensitivity_layers),
            "selection_rule": "choose the individually lowest measured singleton KL layers",
            "warning": "baseline ignores interactions and is not the allocator's objective",
        },
        "fisher_interaction_proxy": {
            "layer_order": list(proxy.layers),
            "single_layer_nll_gradient": list(proxy.linear_nll),
            "pairwise_fisher_gram": [list(row) for row in proxy.fisher_gram],
            "scored_tokens": proxy.scored_tokens,
        },
        "single_layer_probe_metrics": singleton_metrics,
        "proxy_ranked_allocations": list(search.ranked_allocations),
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
