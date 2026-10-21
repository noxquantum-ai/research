#!/usr/bin/env python3
"""Paired, checksum-verified comparison of dense and NEC PPL result files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _primary_metrics(result: dict) -> dict:
    metrics = result.get("metrics", {})
    primary = metrics.get("primary", metrics.get("primary_10k"))
    if not isinstance(primary, dict):
        raise ValueError("Missing primary evaluation metrics")
    return primary


def _load_sidecar(result: dict, name: str, *, dtype: str):
    metadata = _primary_metrics(result)
    sidecar = metadata.get(name)
    if not isinstance(sidecar, dict):
        raise ValueError(f"Missing {name} metadata")
    path = Path(sidecar.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != sidecar.get("sha256"):
        raise ValueError(f"SHA-256 mismatch for {name}")
    expected_dtype = "float32_little_endian_npy" if dtype == "float32" else "int32_little_endian"
    if sidecar.get("dtype") != expected_dtype:
        raise ValueError(f"Unexpected dtype metadata for {name}")
    if name == "token_nll_sidecar":
        values = np.load(path, allow_pickle=False)
        if values.dtype != np.float32 or values.ndim != 1:
            raise ValueError("Token NLL sidecar must be one-dimensional float32")
    else:
        values = np.frombuffer(payload, dtype="<i4")
    if values.size != sidecar.get("count") or not np.isfinite(values).all():
        raise ValueError(f"Invalid count or non-finite data in {name}")
    return values


def compare_results(dense: dict, nec: dict, *, block_tokens=512, repetitions=5000, seed=20261021):
    if dense.get("status") != "COMPLETE" or nec.get("status") != "COMPLETE":
        raise ValueError("Both inputs must be complete evaluation results")
    if dense.get("model_variant") != "dense" or nec.get("model_variant") != "high_fidelity_nec":
        raise ValueError("Expected dense and High-Fidelity NEC result records")
    keys = (
        "model_id", "model_revision", "source_commit", "dataset_split",
        "protocol_fingerprint",
    )
    for key in keys:
        if key not in dense or key not in nec:
            raise ValueError(f"Missing required protocol field {key}")
        if dense.get(key) != nec.get(key):
            raise ValueError(f"Dense/NEC mismatch for {key}")
    if dense.get("dataset") != nec.get("dataset"):
        raise ValueError("Dense/NEC dataset source metadata differs")
    dense_metrics = _primary_metrics(dense)
    nec_metrics = _primary_metrics(nec)
    for key in ("input_tokens", "scored_tokens", "token_ids_sha256", "window_count"):
        if dense_metrics.get(key) != nec_metrics.get(key):
            raise ValueError(f"Dense/NEC token protocol mismatch for {key}")
    if dense_metrics.get("token_ids_sha256") != nec_metrics.get("token_ids_sha256"):
        raise ValueError("Dense/NEC token ID streams differ")

    dense_nll = _load_sidecar(dense, "token_nll_sidecar", dtype="float32").astype(np.float64)
    nec_nll = _load_sidecar(nec, "token_nll_sidecar", dtype="float32").astype(np.float64)
    dense_top1 = _load_sidecar(dense, "top1_prediction_sidecar", dtype="int32")
    nec_top1 = _load_sidecar(nec, "top1_prediction_sidecar", dtype="int32")
    expected_count = dense_metrics.get("scored_tokens")
    if any(values.size != expected_count for values in (dense_nll, nec_nll, dense_top1, nec_top1)):
        raise ValueError("Sidecar count differs from scored-token count")

    delta = nec_nll - dense_nll
    if block_tokens <= 0 or repetitions < 100:
        raise ValueError("block_tokens must be positive and repetitions >= 100")
    block_sums = np.asarray([
        delta[start : start + block_tokens].sum(dtype=np.float64)
        for start in range(0, delta.size, block_tokens)
    ])
    block_lengths = np.asarray([
        min(block_tokens, delta.size - start)
        for start in range(0, delta.size, block_tokens)
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, block_sums.size, size=(repetitions, block_sums.size))
    bootstrap_means = block_sums[choices].sum(axis=1) / block_lengths[choices].sum(axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    dense_nll_mean = float(dense_nll.mean())
    nec_nll_mean = float(nec_nll.mean())
    mean_delta = float(delta.mean())
    return {
        "status": "COMPLETE",
        "model_id": dense["model_id"],
        "model_revision": dense["model_revision"],
        "source_commit": dense["source_commit"],
        "dataset": dense["dataset"],
        "protocol_fingerprint": dense["protocol_fingerprint"],
        "compressed_layers": nec.get("compressed_layers"),
        "dense": {
            "perplexity": dense_metrics["perplexity"],
            "mean_nll": dense_nll_mean,
            "next_token_accuracy": dense_metrics["next_token_accuracy"],
        },
        "nec": {
            "perplexity": nec_metrics["perplexity"],
            "mean_nll": nec_nll_mean,
            "next_token_accuracy": nec_metrics["next_token_accuracy"],
        },
        "paired": {
            "scored_tokens": int(expected_count),
            "mean_nll_delta_nec_minus_dense": mean_delta,
            "mean_nll_delta_95pct_block_bootstrap_ci": [float(low), float(high)],
            "perplexity_ratio_nec_over_dense": float(np.exp(nec_nll_mean - dense_nll_mean)),
            "next_token_accuracy_delta": float(
                nec_metrics["next_token_accuracy"] - dense_metrics["next_token_accuracy"]
            ),
            "top1_agreement_with_dense": float(np.mean(nec_top1 == dense_top1)),
            "bootstrap": {
                "block_tokens": block_tokens,
                "blocks": int(block_sums.size),
                "repetitions": repetitions,
                "seed": seed,
                "interval": "percentile 95%; contiguous NLL-difference blocks",
            },
        },
        "dense_result_sha256": hashlib.sha256(
            json.dumps(dense, sort_keys=True).encode()
        ).hexdigest(),
        "nec_result_sha256": hashlib.sha256(
            json.dumps(nec, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--nec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    result = compare_results(
        json.loads(args.dense.read_text()),
        json.loads(args.nec.read_text()),
        block_tokens=args.block_tokens,
        repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != result:
            raise FileExistsError(
                f"Existing paired comparison differs from current results: {args.output}"
            )
        print("Reusing checksum-verified identical paired comparison", flush=True)
        print(json.dumps(result["paired"], indent=2), flush=True)
        return
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["paired"], indent=2), flush=True)


if __name__ == "__main__":
    main()
