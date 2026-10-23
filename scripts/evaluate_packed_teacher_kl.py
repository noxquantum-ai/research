#!/usr/bin/env python3
"""Measure dense-teacher KL for a complete packed NEC checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import (
    inject_nox_checkpoints,
    sliding_window_specifications,
)
from nox.data import load_wikitext2_raw_split


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--max-eval-tokens", type=int, default=10_000)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--kl-chunk-tokens", type=int, default=128)
    parser.add_argument(
        "--projections",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated NEC projections to inject; all other weights remain dense.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.kl_chunk_tokens <= 0:
        raise ValueError("--kl-chunk-tokens must be positive")
    if args.token_offset < 0:
        raise ValueError("--token-offset must be nonnegative")
    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if str(args.device).startswith("cuda") else None,
    )
    if not str(args.device).startswith("cuda"):
        model.to(args.device)
    model.eval()

    text, split_metadata = load_wikitext2_raw_split(args.split)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if args.token_offset >= ids.shape[1] - 1:
        raise ValueError("--token-offset leaves fewer than two input tokens")
    ids = ids[:, args.token_offset:]
    windows, input_tokens = sliding_window_specifications(
        ids.shape[1],
        seq_len=args.seq_len,
        stride=args.stride,
        max_eval_tokens=args.max_eval_tokens,
    )

    teacher_parts: list[torch.Tensor] = []
    labels_parts: list[torch.Tensor] = []
    started = time.monotonic()
    for index, window in enumerate(windows, start=1):
        input_ids = ids[:, window["begin"]:window["end"]].to(args.device)
        targets = input_ids.clone()
        if window["target_length"] < input_ids.shape[1]:
            targets[:, :-window["target_length"]] = -100
        labels = targets[:, 1:]
        mask = labels != -100
        logits = model(input_ids, use_cache=False).logits[:, :-1][mask]
        teacher_parts.append(logits.to(device="cpu", dtype=torch.float16))
        labels_parts.append(labels[mask].to(device="cpu", dtype=torch.long))
        if index == 1 or index % 10 == 0 or index == len(windows):
            print(f"teacher windows={index}/{len(windows)}", flush=True)

    projections = tuple(part.strip() for part in args.projections.split(",") if part.strip())
    injected = inject_nox_checkpoints(
        model,
        args.checkpoint_dir,
        device=args.device,
        dtype=dtype,
        projections=projections,
    )
    totals = {"teacher_nll": 0.0, "student_nll": 0.0, "kl": 0.0}
    student_correct = 0
    teacher_correct = 0
    top1_agreement = 0
    scored_tokens = 0

    for index, (window, teacher_cpu, labels_cpu) in enumerate(
        zip(windows, teacher_parts, labels_parts), start=1,
    ):
        input_ids = ids[:, window["begin"]:window["end"]].to(args.device)
        targets = input_ids.clone()
        if window["target_length"] < input_ids.shape[1]:
            targets[:, :-window["target_length"]] = -100
        labels = targets[:, 1:]
        mask = labels != -100
        student_logits = model(input_ids, use_cache=False).logits[:, :-1][mask].float()
        teacher_logits = teacher_cpu.to(device=args.device, dtype=torch.float32)
        scored_labels = labels_cpu.to(device=args.device)
        if teacher_logits.shape != student_logits.shape:
            raise RuntimeError("teacher and student logit shapes differ")

        for begin in range(0, scored_labels.numel(), args.kl_chunk_tokens):
            end = min(begin + args.kl_chunk_tokens, scored_labels.numel())
            teacher = teacher_logits[begin:end]
            student = student_logits[begin:end]
            chunk_labels = scored_labels[begin:end]
            log_teacher = F.log_softmax(teacher, dim=-1)
            log_student = F.log_softmax(student, dim=-1)
            rows = torch.arange(end - begin, device=args.device)
            totals["teacher_nll"] += float((-log_teacher[rows, chunk_labels]).double().sum())
            totals["student_nll"] += float((-log_student[rows, chunk_labels]).double().sum())
            totals["kl"] += float(F.kl_div(
                log_student, log_teacher, reduction="none", log_target=True,
            ).sum(dim=-1).double().sum())
            teacher_top1 = teacher.argmax(dim=-1)
            student_top1 = student.argmax(dim=-1)
            teacher_correct += int((teacher_top1 == chunk_labels).sum())
            student_correct += int((student_top1 == chunk_labels).sum())
            top1_agreement += int((student_top1 == teacher_top1).sum())
        scored_tokens += int(scored_labels.numel())
        if index == 1 or index % 10 == 0 or index == len(windows):
            print(f"student windows={index}/{len(windows)}", flush=True)

    teacher_nll = totals["teacher_nll"] / scored_tokens
    student_nll = totals["student_nll"] / scored_tokens
    report = {
        "schema": "nec_packed_teacher_kl_v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "compressed_layers": sorted(int(index) for index in injected),
        "compressed_projections": list(projections),
        "test_split": split_metadata,
        "token_offset": args.token_offset,
        "input_token_ids_sha256": hashlib.sha256(
            ids[:, :input_tokens].contiguous().numpy().astype("<i8").tobytes()
        ).hexdigest(),
        "input_tokens": int(input_tokens),
        "scored_tokens": int(scored_tokens),
        "context": int(args.seq_len),
        "stride": int(args.stride),
        "teacher": {
            "perplexity": math.exp(teacher_nll),
            "mean_nll": teacher_nll,
            "next_token_accuracy": teacher_correct / scored_tokens,
        },
        "student": {
            "perplexity": math.exp(student_nll),
            "mean_nll": student_nll,
            "teacher_kl_nats_per_token": totals["kl"] / scored_tokens,
            "next_token_accuracy": student_correct / scored_tokens,
            "dense_top1_agreement": top1_agreement / scored_tokens,
        },
        "seconds": time.monotonic() - started,
    }
    output = args.output or args.checkpoint_dir / "evaluation_teacher_kl.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
