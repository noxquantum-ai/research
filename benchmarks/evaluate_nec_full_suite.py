#!/usr/bin/env python3
"""Run the five full capability benchmarks on the exported NEC model only.

The package reconstructs NEC weights for inference. This evaluation does not
measure compressed execution speed, and never evaluates a dense baseline.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sqlite3
from pathlib import Path
import sys
import time
import traceback


TASKS = (
    ("boolq", 0, "acc,none", 3270),
    ("hellaswag", 0, "acc_norm,none", 10042),
    ("mmlu", 5, "acc,none", 14042),
    ("triviaqa", 0, "exact_match,remove_whitespace", None),
    ("gsm8k", 5, "exact_match,strict-match", 1319),
)
HARNESS_VERSION = "0.4.11"


def serializable(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(data, indent=2, default=serializable) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", default="auto:4")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/nec-bench-response-cache"))
    parser.add_argument("--tasks", nargs="+", choices=[t[0] for t in TASKS])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # SQLite performs many tiny reads/writes: keep its hot working set on local
    # scratch, and take consistent durable snapshots after each benchmark.
    for source in args.output_dir.glob("response_cache_rank*.db"):
        destination = args.cache_dir / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    status_path = args.output_dir / "status.json"
    status = {"schema": "nec_full_capability_suite_v1", "phase": "loading",
              "package_dir": str(args.package_dir.resolve()), "dense_evaluation": False,
              "full_splits": True, "completed": {}, "failed": {}}
    for task, *_ in TASKS:
        path = args.output_dir / f"{task}_summary.json"
        if path.exists():
            status["completed"][task] = json.loads(path.read_text())
    write_json(status_path, status)
    if importlib.metadata.version("lm_eval") != HARNESS_VERSION:
        raise RuntimeError(f"Require lm_eval=={HARNESS_VERSION}")
    import torch
    import datasets
    from huggingface_hub import HfApi
    # Resolve and record immutable dataset revisions before task construction.
    # The wrapper also records configurations, split fingerprints and row counts.
    original_loader = datasets.load_dataset
    revisions, loads = {}, []

    def pinned_loader(path, *positional, **kwargs):
        revision = kwargs.get("revision")
        key = (str(path), revision)
        if key not in revisions:
            revisions[key] = HfApi().dataset_info(path, revision=revision).sha
        kwargs["revision"] = revisions[key]
        result = original_loader(path, *positional, **kwargs)
        splits = result.items() if hasattr(result, "items") else [(kwargs.get("split"), result)]
        loads.append({"path": path, "name": kwargs.get("name", positional[0] if positional else None),
                      "revision": kwargs["revision"], "splits": {
                          str(k): {"rows": len(v), "fingerprint": v._fingerprint} for k, v in splits}})
        write_json(args.output_dir / "datasets.json", loads)
        return result

    datasets.load_dataset = pinned_loader
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer
    sys.path.insert(0, str(args.package_dir.resolve()))
    from load_nec_package import load_model
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.manual_seed(1234)
    started = time.monotonic()
    model = load_model(device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.package_dir, local_files_only=True)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size,
              max_batch_size=args.max_batch_size, max_length=4096, logits_cache=True)
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "datasets", "lm_eval", "accelerate", "safetensors")}
    write_json(args.output_dir / "protocol.json", {
        "schema": "nec_full_capability_protocol_v1", "versions": versions,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_dir": str(args.package_dir.resolve()), "inference": "NEC reconstructed FP16",
        "model_load_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(), "batch_size": args.batch_size,
        "max_batch_size": args.max_batch_size, "context": 4096,
        "apply_chat_template": False, "limit": None,
        "seeds": {"random": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234},
        "tasks": [dict(task=t, fewshot=s, primary_metric=m, expected_samples=n)
                  for t, s, m, n in TASKS],
        "external_reference_protocol_match": "not established",
        "dense_evaluation": False,
    })
    for task, fewshot, metric, expected in TASKS:
        if task in status["completed"] or (args.tasks and task not in args.tasks):
            continue
        status.update(phase="evaluating", current_task=task)
        write_json(status_path, status)
        began = time.monotonic()
        print(f"START FULL {task} fewshot={fewshot}", flush=True)
        try:
            result = evaluator.simple_evaluate(
                model=lm, tasks=[task], num_fewshot=fewshot,
                batch_size=args.batch_size, device="cuda:0", limit=None,
                use_cache=str(args.cache_dir / "response_cache"),
                bootstrap_iters=0, log_samples=True, apply_chat_template=False,
                random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                fewshot_random_seed=1234,
            )
            if result is None:
                raise RuntimeError("Harness returned no result")
            raw = args.output_dir / f"{task}_full.json.gz"
            temporary = raw.with_suffix(".partial")
            with gzip.open(temporary, "wt") as stream:
                json.dump(result, stream, default=serializable)
            temporary.replace(raw)
            sample_counts = result["n-samples"]
            leaf_tasks = list(result["samples"])
            evaluated = sum(sample_counts[k]["effective"] for k in leaf_tasks)
            for name in leaf_tasks:
                count = sample_counts[name]
                if count["effective"] != count["original"]:
                    raise RuntimeError(f"Incomplete evaluation for {name}: {count}")
                if len(result["samples"][name]) != count["effective"]:
                    raise RuntimeError(f"Logged sample count mismatch for {name}")
            if expected is not None and evaluated != expected:
                raise RuntimeError(f"Expected {expected} samples, got {evaluated}")
            metrics = result["results"].get(task, result.get("groups", {}).get(task, {}))
            value = metrics.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(f"Missing/invalid prespecified metric {metric}: {metrics}")
            summary = {"task": task, "full_split": True, "fewshot": fewshot,
                       "samples": evaluated, "sample_counts": sample_counts,
                       "primary_metric": metric, "score": value,
                       "score_percent": float(value) * 100, "all_metrics": metrics,
                       "seconds": time.monotonic() - began,
                       "raw_result_sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
            write_json(args.output_dir / f"{task}_summary.json", summary)
            status["completed"][task] = summary
            status["failed"].pop(task, None)
            print("COMPLETE " + json.dumps(summary), flush=True)
            del result
            for source in args.cache_dir.glob("response_cache_rank*.db"):
                target = args.output_dir / source.name
                with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
                    src.backup(dst)
        except Exception:
            failure = traceback.format_exc()
            status["failed"][task] = failure
            print(failure, flush=True)
        write_json(status_path, status)
    status.update(phase="complete" if len(status["completed"]) == 5 else "incomplete",
                  current_task=None)
    write_json(status_path, status)
    if status["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
