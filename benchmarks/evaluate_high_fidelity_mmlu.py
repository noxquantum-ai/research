#!/usr/bin/env python3
"""Pinned matched lm-evaluation-harness MMLU evaluation for dense or NEC Llama-2-7B."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.evaluate_high_fidelity_nec import MODEL_VARIANTS, load_model, source_commit
from nox.artifacts import fingerprint
from nox.evaluation import (
    DENSE_LLAMA2_7B_COEFFICIENTS,
    MMLU_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    sha256_file,
    validate_high_fidelity_checkpoint,
)
from nox.mmlu import load_full_mmlu


LM_EVAL_VERSION = "0.4.13"
LM_EVAL_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"


def install_pinned_mmlu_loader(datasets_module):
    """Force every harness load of ``cais/mmlu`` to the audited revision."""
    original = datasets_module.load_dataset
    calls = []

    def pinned(path, *args, **kwargs):
        if path == "cais/mmlu":
            supplied = kwargs.get("revision")
            if supplied not in (None, MMLU_REVISION):
                raise ValueError(f"Conflicting MMLU revision: {supplied}")
            kwargs["revision"] = MMLU_REVISION
            calls.append({"path": path, "name": kwargs.get("name"), "revision": MMLU_REVISION})
        return original(path, *args, **kwargs)

    datasets_module.load_dataset = pinned
    return calls, original


def summarize_mmlu_harness(results: dict, subjects: list[str], *, full_test: bool) -> dict:
    expected_tasks = {f"mmlu_{subject}" for subject in subjects}
    task_results = results.get("results", {})
    if not expected_tasks.issubset(task_results):
        missing = sorted(expected_tasks - set(task_results))
        raise ValueError(f"Harness omitted MMLU subjects: {missing}")
    samples = results.get("n-samples", {})
    per_subject = {}
    total_questions = 0
    total_correct_weight = 0.0
    for subject in sorted(subjects):
        task = f"mmlu_{subject}"
        accuracy = task_results[task].get("acc,none")
        if not isinstance(accuracy, (int, float)):
            raise ValueError(f"Missing accuracy for {task}")
        sample_record = samples.get(task, {})
        count = sample_record.get("effective", sample_record.get("original"))
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"Missing evaluated sample count for {task}")
        per_subject[subject] = {"accuracy": float(accuracy), "questions": count}
        total_questions += count
        total_correct_weight += float(accuracy) * count
    overall = total_correct_weight / total_questions
    group_accuracy = results.get("groups", {}).get("mmlu", {}).get("acc,none")
    if group_accuracy is not None and abs(float(group_accuracy) - overall) > 1e-12:
        raise ValueError("Harness group accuracy disagrees with question-weighted subject accuracy")
    if full_test and total_questions != 14_042:
        raise ValueError(f"Full MMLU must score 14,042 questions, got {total_questions}")
    return {
        "overall_accuracy": overall,
        "total_questions": total_questions,
        "per_subject": per_subject,
    }


def mmlu_sample_identity(results: dict, subjects: list[str]) -> dict:
    """Hash only harness input identities, excluding model responses and metrics."""
    samples = results.get("samples", {})
    rows = []
    for subject in sorted(subjects):
        task = f"mmlu_{subject}"
        task_samples = samples.get(task)
        if not isinstance(task_samples, list) or not task_samples:
            raise ValueError(f"Harness did not log samples for {task}")
        for sample in task_samples:
            identity = {
                "task": task,
                "doc_id": sample.get("doc_id"),
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
                "target_hash": sample.get("target_hash"),
            }
            if not isinstance(identity["doc_id"], int) or any(
                not isinstance(identity[key], str) or not identity[key]
                for key in ("doc_hash", "prompt_hash", "target_hash")
            ):
                raise ValueError(f"Harness sample identity is incomplete for {task}")
            rows.append(identity)
    rows.sort(key=lambda item: (item["task"], item["doc_id"]))
    return {"samples": len(rows), "sha256": fingerprint(rows)}


def audit_mmlu_response_finiteness(results: dict, subjects: list[str]) -> dict:
    """Reject NaN/Inf likelihoods in every logged MMLU model response."""
    checked = 0

    def inspect(value) -> None:
        nonlocal checked
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                raise FloatingPointError("MMLU harness logged a non-finite model response")
            checked += 1
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    for subject in subjects:
        for sample in results.get("samples", {}).get(f"mmlu_{subject}", []):
            inspect(sample.get("resps"))
            inspect(sample.get("filtered_resps"))
    if checked == 0:
        raise RuntimeError("MMLU response audit found no numerical likelihood values")
    return {"status": "PASS", "finite_numerical_values_checked": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=MODEL_VARIANTS, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--smoke-limit", type=int, default=0,
                        help="Nonzero is adapter validation only and cannot produce a complete score")
    parser.add_argument("--smoke-first", action="store_true",
                        help="Run and verify one example per subject before the full benchmark")
    args = parser.parse_args()
    if args.variant == "high_fidelity_nec" and args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required for --variant high_fidelity_nec")
    if args.variant == "dense" and args.checkpoint_dir is not None:
        parser.error("--checkpoint-dir is forbidden for --variant dense")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    raw_path = args.output.with_name(args.output.stem + ".lm_eval_raw.json")
    if raw_path.exists():
        raise FileExistsError(f"Refusing to overwrite {raw_path}")
    if args.batch_size < 1 or args.smoke_limit < 0:
        raise ValueError("Invalid batch size or smoke limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    installed = importlib.metadata.version("lm_eval")
    if installed != LM_EVAL_VERSION:
        raise RuntimeError(f"Expected lm-eval {LM_EVAL_VERSION}, found {installed}")
    import datasets
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    loader_calls, original_loader = install_pinned_mmlu_loader(datasets)
    try:
        splits, pinned_data = load_full_mmlu()
        subjects = sorted({row["subject"] for row in splits["test"]})
        if len(subjects) != 57:
            raise RuntimeError("Pinned MMLU audit did not find 57 subjects")
        audit = (
            validate_high_fidelity_checkpoint(args.checkpoint_dir)
            if args.variant == "high_fidelity_nec" else None
        )
        model, tokenizer, injected, compatibility, _ = load_model(
            args.checkpoint_dir,
            offload_folder=args.output.parent / "evaluation_offload",
            variant=args.variant,
        )
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=4096,
            logits_cache=True,
        )
        started = time.monotonic()
        smoke_record = None
        if args.smoke_first:
            smoke = evaluator.simple_evaluate(
                model=lm,
                tasks=["mmlu"],
                num_fewshot=5,
                batch_size=args.batch_size,
                device="cuda:0",
                limit=1,
                bootstrap_iters=0,
                check_integrity=True,
                log_samples=False,
                apply_chat_template=False,
                fewshot_as_multiturn=False,
                random_seed=0,
                numpy_random_seed=1234,
                torch_random_seed=1234,
                fewshot_random_seed=1234,
            )
            if smoke is None:
                raise RuntimeError("MMLU adapter smoke returned no rank-zero result")
            smoke_summary = summarize_mmlu_harness(smoke, subjects, full_test=False)
            if smoke_summary["total_questions"] != 57:
                raise RuntimeError("MMLU adapter smoke did not evaluate one question per subject")
            smoke_record = {
                "status": "PASS",
                "subjects": 57,
                "questions": 57,
                "metric_disclosed": False,
            }
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=["mmlu"],
            num_fewshot=5,
            batch_size=args.batch_size,
            device="cuda:0",
            limit=args.smoke_limit or None,
            bootstrap_iters=0,
            check_integrity=True,
            log_samples=True,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
        elapsed = time.monotonic() - started
    finally:
        datasets.load_dataset = original_loader
    if results is None:
        raise RuntimeError("lm-evaluation-harness returned no rank-zero results")
    if not loader_calls or any(call["revision"] != MMLU_REVISION for call in loader_calls):
        raise RuntimeError("Harness did not use the pinned MMLU dataset loader")
    full_test = args.smoke_limit == 0
    summary = summarize_mmlu_harness(results, subjects, full_test=full_test)
    sample_identity = mmlu_sample_identity(results, subjects)
    response_finiteness = audit_mmlu_response_finiteness(results, subjects)
    if sample_identity["samples"] != summary["total_questions"]:
        raise RuntimeError("MMLU logged-sample count disagrees with the evaluated question count")

    raw_temp = raw_path.with_suffix(raw_path.suffix + ".partial")
    raw_temp.write_text(json.dumps(results, indent=2, default=str) + "\n")
    raw_temp.replace(raw_path)
    protocol_identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "tokenizer": compatibility["tokenizer"],
        "framework": {
            "name": "EleutherAI/lm-evaluation-harness",
            "version": installed,
            "git_commit": LM_EVAL_COMMIT,
            "task_versions": results.get("versions"),
        },
        "dataset": pinned_data,
        "shots": 5,
        "fewshot_split": "dev",
        "test_split": "test",
        "choice_scoring": "lm-eval multiple_choice loglikelihood over A/B/C/D",
        "prompt_template": "{question.strip()}\\nA. {choices[0]}\\nB. {choices[1]}\\nC. {choices[2]}\\nD. {choices[3]}\\nAnswer:",
        "answer_choices": ["A", "B", "C", "D"],
        "fewshot_sampler": "first_n",
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "batch_size": args.batch_size,
        "max_context_length": 4096,
        "use_cache": False,
        "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
    }
    is_dense = args.variant == "dense"
    report = {
        "status": "COMPLETE" if full_test else "SMOKE_ONLY",
        "evaluation": f"{args.variant}_mmlu_5shot",
        "model_variant": args.variant,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tokenizer_compatibility": compatibility,
        "source_commit": source_commit(),
        "checkpoint_manifest_sha256": audit["manifest_sha256"] if audit else None,
        "compressed_layers": injected,
        "independent_numerical_coefficients": (
            DENSE_LLAMA2_7B_COEFFICIENTS if is_dense
            else audit["independent_numerical_coefficients"]
        ),
        "structural_parameter_reduction": 0.0 if is_dense else audit["structural_parameter_reduction"],
        "parameter_compression_ratio": 1.0 if is_dense else audit["parameter_compression_ratio"],
        "framework": {
            "name": "EleutherAI/lm-evaluation-harness",
            "version": installed,
            "git_commit": LM_EVAL_COMMIT,
            "task": "mmlu",
            "task_versions": results.get("versions"),
        },
        "dataset": pinned_data,
        "protocol_identity": protocol_identity,
        "protocol_fingerprint": fingerprint(protocol_identity),
        "evaluation_input_identity": sample_identity,
        "response_finiteness_audit": response_finiteness,
        "dataset_revision_enforcement_calls": loader_calls,
        "protocol": {
            "shots": 5,
            "fewshot_split": "dev",
            "test_split": "test",
            "choice_scoring": "lm-eval multiple_choice loglikelihood over A/B/C/D",
            "prompt_template": "{question.strip()}\\nA. {choices[0]}\\nB. {choices[1]}\\nC. {choices[2]}\\nD. {choices[3]}\\nAnswer:",
            "answer_choices": ["A", "B", "C", "D"],
            "fewshot_sampler": "first_n",
            "apply_chat_template": False,
            "fewshot_as_multiturn": False,
            "batch_size": args.batch_size,
            "max_context_length": 4096,
            "use_cache": False,
            "smoke_limit_per_subject": args.smoke_limit or None,
            "seeds": {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        },
        "overall_5shot_accuracy": summary["overall_accuracy"],
        "total_questions": summary["total_questions"],
        "per_subject": summary["per_subject"],
        "runtime_seconds": elapsed,
        "total_runtime_seconds": time.monotonic() - started,
        "adapter_smoke": smoke_record,
        "raw_harness_results": str(raw_path.resolve()),
        "raw_harness_results_sha256": sha256_file(raw_path),
        "dense_model_evaluated": is_dense,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "accuracy": report["overall_5shot_accuracy"],
        "questions": report["total_questions"],
        "seconds": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
