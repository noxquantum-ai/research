#!/usr/bin/env python3
"""Evaluate saved layer-0 NEC down-projection stages in an otherwise dense Llama-2.

Every variant is run with the same pinned model, tokenizer, raw WikiText-2 test
tokens, context windows, causal shift and FP16 inference.  Only layer 0's
``mlp.down_proj.weight`` differs. Results are checkpointed stage-by-stage so a
pod interruption does not discard completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day4_perplexity import (  # noqa: E402
    compute_perplexity,
    load_wikitext2_test_text,
)
from benchmarks.evaluate_high_fidelity_nec import (  # noqa: E402
    load_model,
    source_commit,
)
from nox.evaluation import MODEL_ID, MODEL_REVISION  # noqa: E402


STAGE_NAMES = (
    "sparse",
    "sparse_plus_bridge",
    "tt_warmstart",
    "sweep_1",
    "sweep_2",
    "serialized_artifact",
)
STAGE_LABELS = {"dense": "dense", **{stage: stage for stage in STAGE_NAMES}}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> dict:
    paths = (
        "benchmarks/benchmark_day4_perplexity.py",
        "benchmarks/evaluate_high_fidelity_nec.py",
        "benchmarks/evaluate_single_projection_stages.py",
        "nox/data.py",
        "nox/evaluation.py",
    )
    entries = {name: sha256_file(ROOT / name) for name in paths}
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {"files": entries, "sha256": sha256_bytes(payload)}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npy(path: Path, values: np.ndarray, dtype="<f4") -> dict:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        array = np.asarray(values, dtype=dtype)
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "dtype": str(array.dtype),
        "count": int(values.size),
        "bytes": int(path.stat().st_size),
    }


def load_stage_weights(run_output: Path, report: dict, expected_shape: tuple[int, int]) -> dict:
    if report.get("status") != "PASS":
        raise ValueError("A completed PASS qualification report is required")
    identity = report.get("qualification_identity", {})
    if (
        identity.get("model_id") != MODEL_ID
        or identity.get("model_revision") != MODEL_REVISION
        or identity.get("layer") != 0
        or identity.get("projection") != "down_proj"
        or report.get("layer") != 0
        or report.get("projection") != "down_proj"
        or tuple(report.get("weight_shape", ())) != expected_shape
    ):
        raise ValueError("Stage report does not describe the pinned layer-0 down_proj")
    snapshots = report.get("stage_snapshots", {})
    if set(snapshots) != set(STAGE_NAMES):
        raise ValueError(f"Expected all {STAGE_NAMES} snapshots, got {sorted(snapshots)}")

    weights = {}
    for stage in STAGE_NAMES:
        metadata = snapshots[stage]
        path = run_output / "stage_weights" / f"{stage}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != metadata.get("bytes") or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Stage checksum/size mismatch: {stage}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("stage") != stage
            or payload.get("algorithm_identity") != report.get("algorithm_identity")
            or payload.get("qualification_identity") != identity
        ):
            raise ValueError(f"Stage provenance mismatch: {stage}")
        weight = payload.get("weight")
        if (
            not torch.is_tensor(weight)
            or tuple(weight.shape) != expected_shape
            or not torch.isfinite(weight).all()
        ):
            raise ValueError(f"Stage weight has invalid shape or non-finite values: {stage}")
        weights[stage] = {"path": path, "weight": weight, "sha256": metadata["sha256"]}
    return weights


def block_bootstrap_paired_delta(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    block_tokens: int = 512,
    repetitions: int = 5000,
    seed: int = 20261021,
) -> dict:
    """Paired contiguous-block bootstrap for mean token-NLL difference.

    The block bootstrap accounts approximately for local dependence in text and
    sliding-context scores. It is descriptive for this fixed benchmark corpus,
    not a claim of independent document sampling.
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if reference.size != candidate.size or reference.size == 0:
        raise ValueError("Paired NLL arrays must have equal, nonzero lengths")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Paired NLL arrays must be finite")
    if block_tokens < 1 or repetitions < 100:
        raise ValueError("block_tokens must be positive and repetitions >= 100")

    delta = candidate - reference
    sums, lengths = [], []
    for start in range(0, delta.size, block_tokens):
        chunk = delta[start : start + block_tokens]
        sums.append(float(chunk.sum()))
        lengths.append(int(chunk.size))
    sums = np.asarray(sums, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    n_blocks = int(sums.size)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_blocks, size=(repetitions, n_blocks))
    bootstrap_means = sums[sampled].sum(axis=1) / lengths[sampled].sum(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    mean_delta = float(delta.mean())
    return {
        "mean_nll_delta_candidate_minus_dense": mean_delta,
        "perplexity_ratio_candidate_over_dense": math.exp(mean_delta),
        "paired_token_delta_std": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
        "paired_token_delta_median": float(np.median(delta)),
        "fraction_tokens_candidate_higher_nll": float(np.mean(delta > 0)),
        "block_tokens": block_tokens,
        "block_count": n_blocks,
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "mean_nll_delta_ci95": [float(low), float(high)],
        "perplexity_ratio_ci95": [math.exp(float(low)), math.exp(float(high))],
        "interval_method": "paired contiguous token-block bootstrap; approximate local-dependence adjustment",
    }


def read_existing_result(output_dir: Path, identity_sha256: str, stage: str) -> tuple[dict, np.ndarray] | None:
    result_path = output_dir / "progress.json"
    if not result_path.is_file():
        return None
    record = json.loads(result_path.read_text())
    if record.get("run_identity_sha256") != identity_sha256:
        raise ValueError("Existing progress belongs to a different model, stage set, or protocol")
    entry = record.get("stages", {}).get(stage)
    if not entry:
        return None
    sidecar = Path(entry["token_nll_sidecar"]["path"])
    metadata = entry["token_nll_sidecar"]
    prediction_metadata = entry["top1_prediction_sidecar"]
    prediction_sidecar = Path(prediction_metadata["path"])
    if (
        not sidecar.is_file()
        or sha256_file(sidecar) != metadata.get("sha256")
        or sidecar.stat().st_size != metadata.get("bytes")
        or not prediction_sidecar.is_file()
        or sha256_file(prediction_sidecar) != prediction_metadata.get("sha256")
        or prediction_sidecar.stat().st_size != prediction_metadata.get("bytes")
    ):
        raise ValueError(f"Existing {stage} token-NLL sidecar failed integrity check")
    return record, np.load(sidecar, allow_pickle=False)


def save_stage_result(
    output_dir: Path,
    record: dict,
    stage: str,
    metrics: dict,
    token_nll: np.ndarray,
    token_predictions: np.ndarray,
    token_ids_sha256: str,
) -> None:
    nll_metadata = atomic_npy(output_dir / f"{stage}.token_nll.f32.npy", token_nll)
    pred_metadata = atomic_npy(
        output_dir / f"{stage}.top1_predictions.i32.npy", token_predictions, dtype="<i4",
    )
    record["stages"][stage] = {
        **metrics,
        "token_nll_sidecar": nll_metadata,
        "top1_prediction_sidecar": pred_metadata,
        "token_ids_sha256": token_ids_sha256,
    }
    atomic_json(output_dir / "progress.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True,
                        help="Qualification directory containing report.json and stage_weights/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--block-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--offload-folder", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        # Only the exact resumable progress file/sidecars are accepted below.
        allowed = {"progress.json", "result.json", "last_execution.json"}
        allowed.update(f"{name}.token_nll.f32.npy" for name in STAGE_LABELS)
        allowed.update(f"{name}.top1_predictions.i32.npy" for name in STAGE_LABELS)
        unexpected = {path.name for path in args.output_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"Refusing to overwrite unexpected output files: {sorted(unexpected)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.context_length != 2048 or args.stride != 512:
        raise ValueError("Protocol is fixed at context_length=2048 and stride=512")
    if args.batch_size != 1:
        # One-window execution is the reference path; batch equivalence is a
        # separate implementation concern and is not needed for this experiment.
        raise ValueError("This paired stage experiment fixes inference batch size at 1")

    qualification_path = args.run_output / "report.json"
    qualification = json.loads(qualification_path.read_text())
    shape = tuple(qualification.get("weight_shape", ()))
    if len(shape) != 2:
        raise ValueError("Qualification report has no valid projection shape")
    stage_weights = load_stage_weights(args.run_output, qualification, shape)
    stage_hashes = {stage: entry["sha256"] for stage, entry in stage_weights.items()}
    source = source_fingerprint()
    run_identity = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "qualification_source_commit": qualification.get("source", {}).get("commit"),
        "qualification_identity": qualification["qualification_identity"],
        "qualification_report_sha256": sha256_file(qualification_path),
        "stage_snapshot_sha256": stage_hashes,
        "evaluator_source": source,
        "inference_dtype": "float16",
        "attention_implementation": "sdpa",
        "context_length": args.context_length,
        "stride": args.stride,
        "inference_batch_size": args.batch_size,
        "dataset": "Salesforce/wikitext wikitext-2-raw-v1 pinned revision b08601e04326c79dfdd32d625aee71d232d685c3, test",
        "primary_prefix_tokens": 10_000,
        "scoring": "logits[:-1] against labels[1:], each target once, token-weighted CE",
    }
    run_identity_sha = sha256_bytes(
        json.dumps(run_identity, sort_keys=True, separators=(",", ":")).encode()
    )
    progress_path = args.output_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("run_identity_sha256") != run_identity_sha:
            raise ValueError("Existing progress is incompatible with the current code/config/artifacts")
    else:
        progress = {
            "status": "IN_PROGRESS",
            "created_at_unix": time.time(),
            "source_commit": source_commit(),
            "run_identity": run_identity,
            "run_identity_sha256": run_identity_sha,
            "qualification": {
                "source_commit": qualification.get("source", {}).get("commit"),
                "report": str(qualification_path.resolve()),
                "report_sha256": sha256_file(qualification_path),
                "stage_snapshot_sha256": stage_hashes,
                "configuration": qualification.get("configuration"),
                "input_data": qualification.get("input_data"),
            },
            "protocol": {
                "primary_prefix_input_tokens": 10_000,
                "context_length": args.context_length,
                "stride": args.stride,
                "tokenization": "pinned model tokenizer; raw corpus text joined by two newlines; special tokens enabled",
                "loss": "float32 shifted cross entropy, summed and divided by exact scored target count",
                "scored_targets": "each token after initial BOS scored exactly once",
                "primary_prefix_from_full_run": "causal equivalence: future tokens in the final full-test window cannot affect earlier logits",
                "precision": "model FP16, logits/loss FP32",
                "batch_size": args.batch_size,
            },
            "dataset": None,
            "model_tokenizer_compatibility": None,
            "runtime": {},
            "stages": {},
        }
        atomic_json(progress_path, progress)

    completed_path = args.output_dir / "result.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text())
        if completed.get("run_identity_sha256") != run_identity_sha or completed.get("status") != "COMPLETE":
            raise ValueError("Existing final result is incompatible with current source/config/artifacts")
        print(f"ALREADY_COMPLETE output={completed_path.resolve()}", flush=True)
        return

    # Load exactly one pinned dense model. Each saved stage replaces only this
    # one weight, then the original dense tensor is restored before the next.
    print("LOADING_PINNED_DENSE_LLAMA2", flush=True)
    model, tokenizer, injected, compatibility, _ = load_model(
        None, offload_folder=args.offload_folder, variant="dense",
    )
    if injected:
        raise RuntimeError("Dense model unexpectedly received compressed checkpoints")
    target = model.model.layers[0].mlp.down_proj.weight
    if tuple(target.shape) != shape:
        raise ValueError(f"Model projection shape {tuple(target.shape)} != saved stage shape {shape}")
    if target.device.type != "cuda":
        raise RuntimeError(f"Expected layer-0 down_proj on CUDA, found {target.device}")
    dense_weight = target.detach().to(device="cpu", dtype=torch.float16).clone()
    print(f"MODEL_READY projection_shape={shape} projection_device={target.device}", flush=True)
    text, dataset = load_wikitext2_test_text(return_metadata=True)
    corpus_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids[0].to(torch.int64)
    corpus_targets = corpus_ids[1:].numpy().astype(np.int64, copy=False)
    if corpus_ids.numel() < 10_000:
        raise ValueError(f"Pinned raw test split tokenized to only {corpus_ids.numel()} tokens")
    primary_token_ids_sha = sha256_bytes(
        corpus_ids[:10_000].contiguous().numpy().astype("<i8", copy=False).tobytes()
    )
    corpus_token_ids_sha = sha256_bytes(
        corpus_ids.contiguous().numpy().astype("<i8", copy=False).tobytes()
    )
    progress["dataset"] = dataset
    progress["model_tokenizer_compatibility"] = compatibility
    progress["runtime"]["package_versions"] = {
        package: getattr(__import__(package), "__version__", "unknown")
        for package in ("torch", "transformers", "accelerate", "huggingface_hub")
    }
    progress["runtime"]["python"] = sys.version
    progress["runtime"]["platform"] = platform.platform()
    progress["runtime"]["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    atomic_json(progress_path, progress)

    dense_nll = None
    dense_predictions = None
    baseline_start = time.monotonic()
    for stage in ("dense", *STAGE_NAMES):
        existing = read_existing_result(args.output_dir, run_identity_sha, stage)
        if existing is not None:
            saved_record, saved_nll = existing
            progress["stages"][stage] = saved_record["stages"][stage]
            if stage == "dense":
                dense_nll = saved_nll.astype(np.float64)
                dense_predictions = np.load(
                    Path(progress["stages"][stage]["top1_prediction_sidecar"]["path"]),
                    allow_pickle=False,
                ).astype(np.int64)
            print(f"RESUME_REUSE stage={stage} nll_tokens={saved_nll.size}", flush=True)
            continue

        if stage == "dense":
            target.data.copy_(dense_weight.to(device=target.device, dtype=target.dtype))
        else:
            stage_entry = stage_weights[stage]
            candidate = stage_entry["weight"]
            target.data.copy_(candidate.to(device=target.device, dtype=target.dtype, non_blocking=True))
        torch.cuda.synchronize()
        print(f"EVALUATING_STAGE stage={stage} elapsed_s={time.monotonic()-baseline_start:.1f}", flush=True)
        started = time.monotonic()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
        ppl_full, scored, details = compute_perplexity(
            model,
            tokenizer,
            text,
            seq_len=args.context_length,
            stride=args.stride,
            max_eval_tokens=0,
            device=str(model.get_input_embeddings().weight.device),
            return_details=True,
            verify_reference_loss=True,
            batch_size=args.batch_size,
            return_token_nll=True,
            progress_interval=10,
        )
        torch.cuda.synchronize()
        nll = details["token_nll"].numpy().astype(np.float64, copy=False)
        predictions = details["top1_predictions"].numpy().astype(np.int64, copy=False)
        if scored != details["input_tokens"] - 1 or nll.size != scored or predictions.size != scored:
            raise RuntimeError(f"Incorrect causal target accounting at stage {stage}")
        if details["token_ids_sha256"] != corpus_token_ids_sha:
            raise RuntimeError(f"Evaluator token IDs disagree with the pinned tokenizer at {stage}")
        if not np.isfinite(nll).all() or not np.isfinite(ppl_full):
            raise FloatingPointError(f"Non-finite language-model result at stage {stage}")
        if abs(math.exp(float(nll.mean())) - ppl_full) > 1e-8 * max(1.0, ppl_full):
            raise RuntimeError(f"Per-token NLL and aggregate PPL disagree at stage {stage}")
        if stage == "dense":
            dense_nll = nll
            dense_predictions = predictions
        if dense_nll is None or dense_predictions is None:
            raise RuntimeError("Dense reference must be evaluated before NEC stages")
        if nll.size != dense_nll.size or nll.size != corpus_targets.size:
            raise RuntimeError(f"Scored-token sequence length changed at stage {stage}")
        dense_token_hash = progress["stages"].get("dense", {}).get("token_ids_sha256")
        if stage != "dense" and details["token_ids_sha256"] != dense_token_hash:
            raise RuntimeError(f"Tokenization changed relative to dense baseline at stage {stage}")

        prefix_count = min(9_999, nll.size)
        primary_nll = float(nll[:prefix_count].mean())
        full_nll = float(details["mean_nll"])
        tokenwise_sidecar_mean_nll = float(nll.mean())
        if abs(tokenwise_sidecar_mean_nll - full_nll) > 1e-6 * max(1.0, abs(full_nll)):
            raise RuntimeError(f"Saved per-token NLL mean disagrees with aggregate NLL at {stage}")
        metrics = {
            "status": "COMPLETE",
            "perplexity": ppl_full,
            "mean_nll": full_nll,
            "tokenwise_sidecar_mean_nll": tokenwise_sidecar_mean_nll,
            "next_token_accuracy": float(np.mean(predictions == corpus_targets)),
            "scored_tokens": int(scored),
            "input_tokens": int(details["input_tokens"]),
            "token_ids_sha256": details["token_ids_sha256"],
            "bos_token_id": details["bos_token_id"],
            "window_count": details["window_count"],
            "window_specification": details["window_specification"],
            "nll_formula_crosscheck": details["nll_formula_crosscheck"],
            "full_test": {
                "perplexity": ppl_full,
                "mean_nll": full_nll,
                "next_token_accuracy": float(np.mean(predictions == corpus_targets)),
                "scored_tokens": int(scored),
            },
            "primary_10k_prefix": {
                "input_tokens": min(10_000, int(details["input_tokens"])),
                "scored_tokens": prefix_count,
                "mean_nll": primary_nll,
                "perplexity": math.exp(primary_nll),
                "token_ids_sha256": primary_token_ids_sha,
                "next_token_accuracy": float(np.mean(
                    predictions[:prefix_count] == corpus_targets[:prefix_count],
                )),
            },
            "elapsed_seconds": time.monotonic() - started,
            "nll_formula": "tokenwise cross entropy after causal shift; exact token-weighted aggregation",
            "cuda_peak_allocated_bytes": {
                str(index): int(torch.cuda.max_memory_allocated(index))
                for index in range(torch.cuda.device_count())
            },
            "cuda_peak_reserved_bytes": {
                str(index): int(torch.cuda.max_memory_reserved(index))
                for index in range(torch.cuda.device_count())
            },
        }
        if abs(metrics["next_token_accuracy"] - details["next_token_accuracy"]) > 1e-12:
            raise RuntimeError(f"Token accuracy side calculation disagrees at stage {stage}")
        if stage != "dense":
            metrics["paired_vs_dense_full_test"] = block_bootstrap_paired_delta(
                dense_nll, nll, block_tokens=args.block_tokens,
                repetitions=args.bootstrap_repetitions,
            )
            metrics["paired_vs_dense_primary_10k"] = block_bootstrap_paired_delta(
                dense_nll[:prefix_count], nll[:prefix_count],
                block_tokens=args.block_tokens,
                repetitions=args.bootstrap_repetitions,
            )
            metrics["one_projection_replacement"] = {
                "layer": 0,
                "projection": "mlp.down_proj",
                "other_layers_and_projections": "pinned dense model unchanged",
                "saved_stage_sha256": stage_entry["sha256"],
                "stage_file": str(stage_entry["path"].resolve()),
                "stage_source_commit": qualification.get("source", {}).get("commit"),
            }
        save_stage_result(
            args.output_dir, progress, stage, metrics, nll,
            predictions, details["token_ids_sha256"],
        )
        print(
            f"STAGE_COMPLETE stage={stage} full_ppl={metrics['perplexity']:.6f} "
            f"mean_nll={metrics['mean_nll']:.6f} scored={scored} "
            f"primary_10k_ppl={metrics['primary_10k_prefix']['perplexity']:.6f} "
            f"elapsed_s={metrics['elapsed_seconds']:.1f}",
            flush=True,
        )
        # Restore the dense weight before evaluating another stage. This also
        # makes stage semantics explicit even if a future error path changes.
        target.data.copy_(dense_weight.to(device=target.device, dtype=target.dtype))

    if set(progress["stages"]) != set(STAGE_LABELS):
        raise RuntimeError(f"Incomplete stage set: {sorted(progress['stages'])}")
    dense_tokens_sha = progress["stages"]["dense"]["token_ids_sha256"]
    for stage, result in progress["stages"].items():
        if result["token_ids_sha256"] != dense_tokens_sha:
            raise RuntimeError(f"Token IDs differ from dense baseline for {stage}")
    progress["status"] = "COMPLETE"
    progress["completed_at_unix"] = time.time()
    progress["dense_model_evaluation_scope"] = (
        "one dense full-model reference plus six variants with only layer-0 mlp.down_proj replaced"
    )
    atomic_json(args.output_dir / "result.json", progress)
    print(json.dumps({
        "status": progress["status"],
        "output": str((args.output_dir / "result.json").resolve()),
        "stages": {
            stage: {
                "full_test_ppl": entry["perplexity"],
                "mean_nll": entry["mean_nll"],
                "primary_10k_ppl": entry["primary_10k_prefix"]["perplexity"],
                "scored_tokens": entry["scored_tokens"],
            }
            for stage, entry in progress["stages"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
