#!/usr/bin/env python3
"""Fail-closed RunPod preflight with real dense-model GPU qualification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nox.evaluation import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_PROBE_IDS,
    TOKENIZER_PROBE_TEXT,
    validate_llama2_config,
)
from nox.data import load_wikitext2_raw_split  # noqa: E402
from runpod_execution.nec_high_fidelity.gpu_qualification import qualify_gpu  # noqa: E402
from runpod_execution.nec_high_fidelity.runpod_driver import (  # noqa: E402
    CONFIG_PATH,
    evaluation_batch_sizes,
    evaluation_memory_limits,
    load_config,
)


EXPECTED_PACKAGES = {
    "torch": "2.11.0+cu128",
    "transformers": "5.16.1",
    "huggingface-hub": "1.29.0",
    "safetensors": "0.8.0",
    "accelerate": "1.14.0",
    "datasets": "4.8.5",
    "pyarrow": "23.0.1",
    "numpy": "2.1.3",
    "scipy": "1.16.3",
    "lm_eval": "0.4.13",
}


def effective_ram_bytes() -> int:
    """Use cgroup limits when present instead of overstating container RAM."""
    physical = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    candidates = []
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            limit = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 < limit < 2**60:
            candidates.append(limit)
    return min([physical, *candidates])


def package_versions() -> dict[str, str]:
    names = dict.fromkeys(EXPECTED_PACKAGES, None)
    return {
        name: importlib.metadata.version(name)
        for name in names
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sys.version_info[:3] != (3, 13, 15):
        raise RuntimeError(f"Expected Python 3.13.15, found {platform.python_version()}")
    workspace = Path(os.environ.get("NEC_WORKSPACE_ROOT", "/workspace")).resolve()
    run_root = Path(os.environ.get(
        "NEC_RUN_ROOT", str(workspace / "runs/nec-high-fidelity-quality-max-20261019"),
    )).resolve()
    if not run_root.is_relative_to(workspace):
        raise RuntimeError(f"Run root {run_root} must be under persistent volume {workspace}")
    if not os.path.ismount(workspace):
        raise RuntimeError(f"{workspace} is not a mounted persistent RunPod volume")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("RunPod preflight requires at least one CUDA GPU")

    # The 65,536-token calibration matrices and exact SVD path are intentionally
    # restricted to B200/B300-class memory.  Overrides may only make this gate
    # stricter; silently attempting this profile on a smaller GPU is unsafe.
    min_gpu_gib = max(160, int(os.environ.get("NEC_MIN_GPU_MEMORY_GIB", "160")))
    min_gpu_free_gib = max(150, int(os.environ.get("NEC_MIN_GPU_FREE_GIB", "150")))
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        total_gib = properties.total_memory / (1024**3)
        free_gib = free_bytes / (1024**3)
        if total_gib < min_gpu_gib:
            raise RuntimeError(
                f"GPU {index} has {total_gib:.1f} GiB; need >= {min_gpu_gib} GiB"
            )
        if free_gib < min_gpu_free_gib:
            raise RuntimeError(
                f"GPU {index} has {free_gib:.1f} GiB free; need >= {min_gpu_free_gib} GiB free"
            )
        devices.append({
            "index": index,
            "name": properties.name,
            "total_vram_bytes": properties.total_memory,
            "free_vram_bytes_at_preflight": free_bytes,
            "cuda_reported_total_bytes_at_preflight": total_bytes,
            "compute_capability": [properties.major, properties.minor],
        })

    ram_bytes = effective_ram_bytes()
    if ram_bytes < 128 * 1024**3:
        raise RuntimeError(f"System RAM is {ram_bytes / 1024**3:.1f} GiB; need >= 128 GiB")
    usage = shutil.disk_usage(workspace)
    if usage.free < 150 * 1024**3:
        raise RuntimeError(
            f"Persistent volume free space is {usage.free / 1024**3:.1f} GiB; need >= 150 GiB"
        )

    versions = package_versions()
    for name, expected in EXPECTED_PACKAGES.items():
        actual = versions[name]
        if actual != expected:
            raise RuntimeError(f"Pinned package mismatch for {name}: {actual}, expected {expected}")
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, text=True, check=False,
    )
    if pip_check.returncode:
        raise RuntimeError(
            "Pinned runtime dependency check failed before model loading: "
            + (pip_check.stdout + pip_check.stderr).strip()
        )
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected CUDA 12.8 PyTorch build, got {torch.version.cuda}")

    run_config = load_config()
    run_config_hash = sha256_file(CONFIG_PATH)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    config_audit = validate_llama2_config(config)
    if config_audit["model_config_commit_hash"] not in (None, MODEL_REVISION):
        raise RuntimeError(
            "Model config resolved to an unexpected revision: "
            f"{config_audit['model_config_commit_hash']}"
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    special_ids = {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "unk": tokenizer.unk_token_id,
        "pad": tokenizer.pad_token_id,
    }
    expected_special_ids = {"bos": 1, "eos": 2, "unk": 0, "pad": 0}
    probe_ids = [int(value) for value in tokenizer(
        TOKENIZER_PROBE_TEXT, add_special_tokens=True,
    )["input_ids"]]
    if len(tokenizer) != 32_000 or tokenizer.vocab_size != 32_000:
        raise RuntimeError("Pinned tokenizer vocabulary is not 32,000 tokens")
    if special_ids != expected_special_ids:
        raise RuntimeError(f"Pinned tokenizer special IDs mismatch: {special_ids}")
    if probe_ids != TOKENIZER_PROBE_IDS:
        raise RuntimeError(f"Pinned tokenizer probe mismatch: {probe_ids}")
    calibration_config = run_config["calibration"]
    calibration_splits = {}
    for split, requested_tokens in (
        ("train", calibration_config["calibration_tokens"]),
        ("validation", calibration_config["heldout_tokens"]),
    ):
        split_text, source = load_wikitext2_raw_split(split)
        token_count = len(tokenizer.encode(split_text, add_special_tokens=True))
        if token_count < requested_tokens:
            raise RuntimeError(
                f"Pinned WikiText-2 {split} has {token_count} tokens; need {requested_tokens}"
            )
        if source["dataset_revision"] != calibration_config["dataset_revision"]:
            raise RuntimeError(f"Unexpected WikiText-2 {split} source revision")
        calibration_splits[split] = {
            "source": source,
            "tokenized_tokens_available": token_count,
            "tokens_requested": requested_tokens,
            "sampling": calibration_config["sampling"],
            "sequence_length": calibration_config["sequence_length"],
            "complete_windows_available": token_count // calibration_config["sequence_length"],
            "complete_windows_requested": requested_tokens // calibration_config["sequence_length"],
        }

    run_root.mkdir(parents=True, exist_ok=True)
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("Repository must be clean and committed before starting the run")
    config_snapshot = run_root / "run_config.json"
    if config_snapshot.exists():
        if sha256_file(config_snapshot) != run_config_hash:
            raise RuntimeError("Persistent run directory already contains a different run configuration")
    else:
        snapshot_temporary = config_snapshot.with_suffix(".json.partial")
        snapshot_temporary.write_bytes(CONFIG_PATH.read_bytes())
        snapshot_temporary.replace(config_snapshot)

    available_gpu_memory = [
        torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())
    ]
    gpu_cap, cpu_cap = evaluation_memory_limits(
        available_gpu_memory, ram_bytes, os.environ,
    )
    os.environ["NEC_EVAL_GPU_MEMORY_GIB"] = str(gpu_cap)
    os.environ["NEC_EVAL_CPU_MEMORY_GIB"] = str(cpu_cap)
    requested_ppl_batch, requested_mmlu_batch = evaluation_batch_sizes(
        available_gpu_memory, os.environ,
    )
    qualification = qualify_gpu(
        run_root=run_root,
        source_commit=repo_commit,
        run_config_sha256=run_config_hash,
        requested_ppl_batch=requested_ppl_batch,
        requested_mmlu_batch=requested_mmlu_batch,
        compression=run_config["compression"],
        calibration_tokens=run_config["calibration"]["calibration_tokens"],
        model_id=run_config["model"]["id"],
        model_revision=run_config["model"]["revision"],
    )

    record = {
        "status": "PASS",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": repo_commit,
        "source_tree_clean": True,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "pip_check": (pip_check.stdout + pip_check.stderr).strip() or "No broken requirements found.",
        "torch_cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": devices,
        "ram_bytes": ram_bytes,
        "persistent_volume": {
            "mount": str(workspace),
            "run_root": str(run_root),
            "total_bytes": usage.total,
            "free_bytes": usage.free,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "run_config": {
            "path": str(CONFIG_PATH),
            "persistent_snapshot": str(config_snapshot),
            "sha256": run_config_hash,
            "experiment": run_config["experiment"],
            "calibration": run_config["calibration"],
        },
        "calibration_dataset_audit": calibration_splits,
        "model_config": config_audit,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "length": len(tokenizer),
            "special_token_ids": special_ids,
            "probe_text": TOKENIZER_PROBE_TEXT,
            "probe_ids": probe_ids,
            "probe_ids_sha256": hashlib.sha256(
                json.dumps(probe_ids, separators=(",", ":")).encode(),
            ).hexdigest(),
            "revision": MODEL_REVISION,
        },
        "dense_model_weights_loaded_during_preflight": True,
        "gpu_qualification": qualification,
        "evaluation_memory_caps_gib": {"gpu_per_device": gpu_cap, "cpu": cpu_cap},
        "qualified_ppl_batch_size": qualification["ppl_batch_equivalence"]["qualified_batch_size"],
        "qualified_mmlu_batch_size": qualification["mmlu_gpu_qualification"]["qualified_batch_size"],
        "matched_dense_and_nec_benchmarks_planned": True,
    }
    target = run_root / "runpod_preflight.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
