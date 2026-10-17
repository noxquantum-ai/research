#!/usr/bin/env python3
"""Resumable, stage-bounded High-Fidelity NEC compilation on Kaggle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL_ID = "NousResearch/Llama-2-7b-hf"
MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
HF_DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WORK = Path("/kaggle/working/nec_high_fidelity")
SOURCE = Path("/kaggle/working/nec_source")
CHECKPOINT = WORK / "checkpoint"
STAGE_SIZE = 10


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardware_snapshot() -> dict:
    gpus = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpus.append({
            "index": index,
            "name": properties.name,
            "total_vram_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        })
    memory = shutil.disk_usage("/kaggle/working")
    ram = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    packages = {}
    for package in (
        "transformers", "huggingface-hub", "safetensors", "accelerate",
        "datasets", "pyarrow", "numpy", "scipy",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": packages,
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "ram_bytes": ram,
        "working_disk_total_bytes": memory.total,
        "working_disk_free_bytes": memory.free,
    }


def verify_source_tree(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("Source bundle manifest has no source files")
    for relative, record in source_files.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe source-bundle path: {relative}")
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe source-bundle file: {relative}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"Source-bundle size mismatch: {relative}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Source-bundle checksum mismatch: {relative}")
    return manifest


def get_source_bundle() -> tuple[Path, str, str]:
    input_root = Path("/kaggle/input")
    archives = sorted(input_root.rglob("nec_source_bundle.tar.gz"))
    manifests = sorted(input_root.rglob("bundle_manifest.json"))
    if len(archives) + len(manifests) != 1:
        raise FileNotFoundError(
            "Expected exactly one packed bundle or extracted bundle manifest in "
            f"Kaggle inputs; found {len(archives)} archive(s), {len(manifests)} manifest(s)"
        )
    SOURCE.mkdir(parents=True, exist_ok=True)
    if archives:
        source_input = archives[0]
        with tarfile.open(source_input, "r:gz") as bundle:
            members = bundle.getmembers()
            for member in members:
                target = (SOURCE / member.name).resolve()
                if not target.is_relative_to(SOURCE.resolve()):
                    raise ValueError(f"Unsafe source archive path: {member.name}")
            bundle.extractall(SOURCE, filter="data")
    else:
        manifest_path = manifests[0]
        input_root = manifest_path.parent.resolve()
        manifest = verify_source_tree(input_root)
        shutil.copy2(manifest_path, SOURCE / "bundle_manifest.json")
        for relative in manifest["source_files"]:
            source = (input_root / relative).resolve()
            target = (SOURCE / relative).resolve()
            if not target.is_relative_to(SOURCE.resolve()):
                raise ValueError(f"Unsafe source-bundle path: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    manifest = verify_source_tree(SOURCE)
    manifest_hash = sha256_file(SOURCE / "bundle_manifest.json")
    return source_input, str(manifest["source_commit"]), manifest_hash


def locate_prior_checkpoint() -> tuple[Path | None, dict | None]:
    found = []
    for path in Path("/kaggle/input").rglob("model_manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        identity = manifest.get("run_identity", {})
        if (
            identity.get("model_id") == MODEL_ID
            and identity.get("model_revision") == MODEL_REVISION
            and identity.get("compression_path") == "no_healing_independent"
        ):
            found.append((path.parent, manifest))
    if len(found) > 1:
        raise RuntimeError(f"Ambiguous prior NEC checkpoint inputs: {[str(x[0]) for x in found]}")
    return found[0] if found else (None, None)


def copy_prior_checkpoint(prior: Path, manifest: dict) -> int:
    identity_layers = manifest.get("run_identity", {}).get("layers")
    completed = sorted(set(manifest.get("layers_compressed", [])))
    if (
        not identity_layers
        or identity_layers != list(range(len(identity_layers)))
        or len(identity_layers) > 32
        or completed != list(range(len(completed)))
        or len(completed) > len(identity_layers)
    ):
        raise ValueError("Prior checkpoint is not a contiguous, resumable layer prefix")
    completed_files = {f"compressed_layer_{index}.pt" for index in completed}
    expected_files = {f"compressed_layer_{index}.pt" for index in identity_layers}
    if set(manifest.get("checkpoint_file_sizes", {})) != expected_files:
        if set(manifest.get("checkpoint_file_sizes", {})) != completed_files:
            raise ValueError("Prior manifest size records disagree with completed layers")
    if set(manifest.get("checkpoint_file_sha256", {})) != expected_files:
        if set(manifest.get("checkpoint_file_sha256", {})) != completed_files:
            raise ValueError("Prior manifest checksums disagree with completed layers")
    if sum(manifest["checkpoint_file_sizes"].values()) != manifest.get("serialized_checkpoint_bytes"):
        raise ValueError("Prior manifest byte total is inconsistent")
    intermediate = manifest.get("intermediate_files", {})
    for filename, metadata in intermediate.items():
        src = prior / filename
        if not src.is_file() or src.stat().st_size != int(metadata["bytes"]):
            raise ValueError(f"Prior intermediate file size mismatch: {filename}")
        if sha256_file(src) != metadata["sha256"]:
            raise ValueError(f"Prior intermediate file checksum mismatch: {filename}")
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prior / "model_manifest.json", CHECKPOINT / "model_manifest.json")
    actual_artifacts = {
        path.name for path in prior.glob("compressed_layer_*.pt")
        if path.stem.removeprefix("compressed_layer_").isdigit()
    }
    if actual_artifacts - expected_files:
        raise ValueError("Prior output contains a layer artifact outside its declared run prefix")
    if not completed_files.issubset(actual_artifacts):
        raise ValueError("Prior manifest declares a missing completed layer artifact")
    for filename in sorted(completed_files):
        src = prior / filename
        if src.stat().st_size != int(manifest["checkpoint_file_sizes"][filename]):
            raise ValueError(f"Prior checkpoint size mismatch: {filename}")
        if sha256_file(src) != manifest["checkpoint_file_sha256"][filename]:
            raise ValueError(f"Prior checkpoint checksum mismatch: {filename}")
        shutil.copy2(src, CHECKPOINT / filename)
    for filename in sorted(actual_artifacts - completed_files):
        shutil.copy2(prior / filename, CHECKPOINT / filename)
    for filename in sorted(intermediate):
        shutil.copy2(prior / filename, CHECKPOINT / filename)
    return len(completed)


def monitor_gpu(stop: threading.Event) -> None:
    while not stop.wait(60):
        try:
            usage = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ], text=True).strip()
            free = shutil.disk_usage("/kaggle/working").free
            memory = {}
            with Path("/proc/meminfo").open() as stream:
                for line in stream:
                    key, value, *_ = line.split()
                    if key in {"MemTotal:", "MemAvailable:"}:
                        memory[key[:-1].lower() + "_bytes"] = int(value) * 1024
            print(
                f"RESOURCE gpu={usage!r} disk_free_bytes={free} ram={memory}",
                flush=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            print(f"RESOURCE monitor_error={type(error).__name__}: {error}", flush=True)


def run_compile(source_commit: str, end_layer: int, resume: bool, run_config: dict) -> None:
    calibration = run_config["calibration"]
    compression = run_config["compression"]
    model = run_config["model"]
    command = [
        sys.executable,
        str(SOURCE / "scripts/compile_high_fidelity.py"),
        "--output", str(CHECKPOINT),
        "--start-layer", "0",
        "--end-layer", str(end_layer),
        "--n-calib", str(calibration["calibration_tokens"]),
        "--n-validation", str(calibration["heldout_tokens"]),
        "--seq-len", str(calibration["sequence_length"]),
        "--calibration-sampling", calibration["sampling"],
        "--validation-sampling", calibration["sampling"],
        "--calibration-activation-mode", "dense",
        "--construction-mode", compression["construction_mode"],
        "--diagonal-prior-strength", str(compression["diagonal_prior_strength"]),
        "--mpo-sweeps", str(compression["mpo_sweeps"]),
        "--mpo-alpha-awt", str(compression["activation_scale_exponent"]),
        "--representation", compression["representation"],
        "--attention-gauge", compression["attention_gauge"],
        "--mlp-gauge", compression["mlp_gauge"],
        "--mlp-objective", compression["mlp_objective"],
        "--budget-allocation", compression["budget_allocation"],
        "--defect-bridge-mode", compression["defect_bridge"],
        "--pees-rank", str(compression["pees_rank"]),
        "--model-id", model["id"],
        "--model-revision", model["revision"],
        "--device", "cuda:0",
    ]
    if resume:
        command.append("--resume")
    env = os.environ.copy()
    env["NEC_SOURCE_COMMIT"] = source_commit
    env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    log_path = WORK / "compile.log"
    stop = threading.Event()
    watcher = threading.Thread(target=monitor_gpu, args=(stop,), daemon=True)
    watcher.start()
    started = time.monotonic()
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\nCOMMAND: {json.dumps(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=SOURCE,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        if status:
            raise subprocess.CalledProcessError(status, command)
    finally:
        stop.set()
        watcher.join(timeout=5)
    print(f"STAGE_COMPILE_SECONDS={time.monotonic() - started:.2f}", flush=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Kaggle kernel has no CUDA accelerator")
    if not (WORK / "progress.json").exists():
        atomic_json(WORK / "progress.json", {"status": "STARTING", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        source_input, source_commit, manifest_hash = get_source_bundle()
        run_config = json.loads((SOURCE / "kaggle_execution/nec_high_fidelity/run_config.json").read_text())
        if run_config["model"]["id"] != MODEL_ID or run_config["model"]["revision"] != MODEL_REVISION:
            raise ValueError("Kaggle run configuration does not use the pinned Llama-2-7B snapshot")
        compression = run_config["compression"]
        if (
            compression.get("tier") != "high_fidelity_10x"
            or compression.get("ranks") != [1, 64, 128, 64, 1]
            or compression.get("bridge_rank") != 128
            or compression.get("sparse_outlier_budget") != 524288
            or compression.get("mpo_sweeps") != 2
            or compression.get("activation_scale_exponent") != 0.5
            or compression.get("post_compression_healing") is not False
            or compression.get("construction_mode") != "independent"
        ):
            raise ValueError("Kaggle run configuration differs from the required High-Fidelity NEC settings")
        if run_config["calibration"]["dataset_revision"] != HF_DATASET_REVISION:
            raise ValueError("Kaggle run configuration has unpinned calibration data")
        prior, previous_manifest = locate_prior_checkpoint()
        if prior is None:
            completed = 0
            end_layer = 0  # The first single-layer runtime is the real-engine smoke.
            resume = False
        else:
            completed = copy_prior_checkpoint(prior, previous_manifest)
            if completed == 32:
                raise RuntimeError("All 32 layers are already compiled; refusing to rerun")
            prior_requested_end = len(previous_manifest["run_identity"]["layers"]) - 1
            end_layer = min(31, max(prior_requested_end, completed + STAGE_SIZE - 1))
            resume = True
        stage = {
            "status": "RUNNING",
            "source_commit": source_commit,
            "source_input": str(source_input),
            "source_bundle_manifest_sha256": manifest_hash,
            "runner_sha256": sha256_file(Path(__file__)),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "dataset_revision": HF_DATASET_REVISION,
            "prior_kernel_output": str(prior) if prior else None,
            "completed_layers_before": completed,
            "target_end_layer": end_layer,
            "resume": resume,
            "runtime": hardware_snapshot(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(WORK / "progress.json", stage)
        print(json.dumps(stage, indent=2), flush=True)
        run_compile(source_commit, end_layer, resume, run_config)
        manifest = json.loads((CHECKPOINT / "model_manifest.json").read_text())
        layers = manifest.get("layers_compressed", [])
        expected = list(range(end_layer + 1))
        if sorted(set(layers)) != expected:
            raise RuntimeError(f"Stage ended with unexpected layer set: {layers}")
        stage.update({
            "status": "COMPLETE",
            "layers_compressed": expected,
            "checkpoint_manifest_sha256": sha256_file(CHECKPOINT / "model_manifest.json"),
            "serialized_checkpoint_bytes": manifest["serialized_checkpoint_bytes"],
            "ended_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(WORK / "progress.json", stage)
        print(json.dumps(stage, indent=2), flush=True)
    except BaseException as error:
        failure = {
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(WORK / "failure.json", failure)
        atomic_json(WORK / "progress.json", failure)
        raise


if __name__ == "__main__":
    main()
