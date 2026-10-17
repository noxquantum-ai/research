#!/usr/bin/env python3
"""Resumable RunPod orchestration for the existing High-Fidelity NEC run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from nox.evaluation import validate_high_fidelity_checkpoint  # noqa: E402

CONFIG_PATH = ROOT / "kaggle_execution/nec_high_fidelity/run_config.json"
COMPILER = ROOT / "scripts/compile_high_fidelity.py"
PPL_EVALUATOR = ROOT / "benchmarks/evaluate_high_fidelity_nec.py"
MMLU_EVALUATOR = ROOT / "benchmarks/evaluate_high_fidelity_mmlu.py"
LM_EVAL_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"
MODEL_ID = "NousResearch/Llama-2-7b-hf"
MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text())
    model = config["model"]
    compression = config["compression"]
    calibration = config["calibration"]
    if (model["id"], model["revision"], model["layers"]) != (
        MODEL_ID, MODEL_REVISION, 32,
    ):
        raise RuntimeError("run_config.json model identity differs from pinned Llama-2-7B")
    if model["projections_per_layer"] != [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ]:
        raise RuntimeError("run_config.json does not include all seven supported projections")
    expected = {
        "tier": "high_fidelity_10x",
        "ranks": [1, 64, 128, 64, 1],
        "bridge_rank": 128,
        "sparse_outlier_budget": 524288,
        "mpo_sweeps": 2,
        "activation_scale_exponent": 0.5,
        "construction_mode": "independent",
        "representation": "mpo4",
        "post_compression_healing": False,
    }
    for key, value in expected.items():
        if compression.get(key) != value:
            raise RuntimeError(f"run_config.json compression.{key}={compression.get(key)!r}, expected {value!r}")
    if (
        model.get("rms_norm_epsilon") != 1e-5
        or model.get("hidden_size") != 4096
        or model.get("intermediate_size") != 11008
        or model.get("attention_heads") != 32
        or model.get("key_value_heads") != 32
        or model.get("vocabulary_size") != 32000
    ):
        raise RuntimeError("run_config.json Llama-2 architecture settings differ from the pinned model")
    if (
        calibration["dataset_revision"] != WIKITEXT_REVISION
        or calibration["dataset"] != "Salesforce/wikitext/wikitext-2-raw-v1"
        or calibration["calibration_split"] != "train"
        or calibration["heldout_split"] != "validation"
        or calibration["calibration_tokens"] != 256
        or calibration["heldout_tokens"] != 128
        or calibration["sequence_length"] != 128
        or calibration["sampling"] != "prefix"
        or calibration["calibration_seed"] != 42
        or calibration["heldout_seed"] != 43
        or calibration["target"] != "independent dense model activations"
    ):
        raise RuntimeError("run_config.json calibration settings differ from the verified run")
    wikitext = config["wikitext2"]
    if (
        wikitext["dataset"] != "Salesforce/wikitext/wikitext-2-raw-v1"
        or wikitext["dataset_revision"] != WIKITEXT_REVISION
        or wikitext["split"] != "test"
        or wikitext["context_length"] != 2048
        or wikitext["stride"] != 512
        or wikitext["primary_input_tokens"] != 10000
        or wikitext["also_evaluate_full_split"] is not True
        or wikitext["score_each_next_token_once"] is not True
        or wikitext["tokenizer_revision"] != MODEL_REVISION
        or wikitext["text_join"] != "two newline characters"
        or wikitext["token_loss_precision"] != "float32_cross_entropy_from_logits"
        or wikitext["chat_template"] is not False
    ):
        raise RuntimeError("run_config.json WikiText-2 revision is not pinned")
    mmlu = config["mmlu"]
    if (
        mmlu["dataset"] != "cais/mmlu"
        or mmlu["dataset_revision"] != MMLU_REVISION
        or mmlu["framework"] != "EleutherAI/lm-evaluation-harness"
        or mmlu["framework_version"] != "0.4.13"
        or mmlu["framework_commit"] != LM_EVAL_COMMIT
        or mmlu["shots"] != 5
        or mmlu["subjects"] != 57
        or mmlu["expected_test_questions"] != 14042
        or mmlu["batch_size"] != 1
        or mmlu["fewshot_split"] != "dev"
        or mmlu["test_split"] != "test"
        or mmlu["prompt_template"] != "question + four labeled choices + Answer:"
        or mmlu["answer_scoring"] != "multiple-choice likelihood of A/B/C/D"
        or mmlu["seeds"] != {"python": 0, "numpy": 1234, "torch": 1234, "fewshot": 1234}
        or mmlu["run_one_per_subject_adapter_smoke_first"] is not True
    ):
        raise RuntimeError("run_config.json MMLU revision is not pinned")
    quantization = compression["quantization"]
    if quantization != {
        "enabled": True,
        "bits": 4,
        "group_size": 64,
        "signed_code_range": [-8, 7],
        "serialization_dtype": "float16",
        "packed_codes": False,
    }:
        raise RuntimeError("run_config.json quantization settings differ from High-Fidelity NEC")
    for key, expected_value in {
        "attention_gauge": "none",
        "mlp_gauge": "none",
        "mlp_objective": "independent",
        "budget_allocation": "uniform",
        "defect_bridge": "none",
        "pees_rank": 0,
    }.items():
        if compression.get(key) != expected_value:
            raise RuntimeError(f"run_config.json compression.{key} differs from High-Fidelity NEC")
    return config


def verify_checkpoint_matches_run_config(checkpoint: Path, config: dict) -> None:
    """Check the compiler manifest's effective settings against the pinned run config."""
    manifest = json.loads((checkpoint / "model_manifest.json").read_text())
    identity = manifest["run_identity"]
    compiler_config = identity["config"]
    compression = config["compression"]
    for actual, expected in {
        "ranks": compression["ranks"],
        "r_bridge": compression["bridge_rank"],
        "n_outliers": compression["sparse_outlier_budget"],
        "mpo_n_sweeps": compression["mpo_sweeps"],
        "mpo_alpha_awt": compression["activation_scale_exponent"],
        "diagonal_prior_strength": compression["diagonal_prior_strength"],
        "representation": compression["representation"],
        "epsilon": 0.005,
    }.items():
        value = compiler_config.get(actual)
        if actual == "ranks":
            value = list(value or [])
        if value != expected:
            raise RuntimeError(
                f"Checkpoint compiler setting {actual}={value!r}, expected {expected!r}"
            )
    for actual, expected in {
        "attention_gauge": compression["attention_gauge"],
        "mlp_gauge": compression["mlp_gauge"],
        "mlp_objective": compression["mlp_objective"],
        "budget_allocation": compression["budget_allocation"],
        "defect_bridge_mode": compression["defect_bridge"],
        "pees_rank": compression["pees_rank"],
        "mpo_site_order": compression["site_order"],
        "calibration_activation_mode": "dense",
        "construction_mode": compression["construction_mode"],
    }.items():
        if identity.get(actual) != expected:
            raise RuntimeError(
                f"Checkpoint run identity {actual}={identity.get(actual)!r}, expected {expected!r}"
            )
    inputs = identity["input_data"]
    for actual, expected in {
        "calibration_token_count": config["calibration"]["calibration_tokens"],
        "heldout_token_count": config["calibration"]["heldout_tokens"],
        "activation_sequence_length": config["calibration"]["sequence_length"],
        "calibration_sampling": config["calibration"]["sampling"],
        "calibration_seed": config["calibration"]["calibration_seed"],
        "validation_sampling": config["calibration"]["sampling"],
        "validation_seed": config["calibration"]["heldout_seed"],
        "tokenizer_model_id": config["model"]["id"],
        "tokenizer_model_revision": config["model"]["revision"],
    }.items():
        if inputs.get(actual) != expected:
            raise RuntimeError(
                f"Checkpoint calibration setting {actual}={inputs.get(actual)!r}, expected {expected!r}"
            )


def roots() -> tuple[Path, Path]:
    workspace = Path(os.environ.get("NEC_WORKSPACE_ROOT", "/workspace")).resolve()
    run_root = Path(os.environ.get(
        "NEC_RUN_ROOT", str(workspace / "runs/nec-high-fidelity-20261017"),
    )).resolve()
    if not run_root.is_relative_to(workspace):
        raise RuntimeError(f"Run root {run_root} must be under persistent volume {workspace}")
    if not os.path.ismount(workspace):
        raise RuntimeError(f"{workspace} is not a mounted persistent volume")
    return workspace, run_root


def source_commit() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Repository must be clean and committed before running NEC")
    return commit


def environment_snapshot(workspace: Path, run_root: Path, commit: str) -> dict:
    packages = {}
    for package in (
        "torch", "transformers", "huggingface-hub", "safetensors", "accelerate",
        "datasets", "pyarrow", "numpy", "scipy", "lm_eval",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    disk = shutil.disk_usage(workspace)
    ram = effective_ram_bytes()
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": packages,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [{
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_vram_bytes": torch.cuda.get_device_properties(index).total_memory,
        } for index in range(torch.cuda.device_count())],
        "ram_bytes": ram,
        "workspace_total_bytes": disk.total,
        "workspace_free_bytes": disk.free,
        "run_root": str(run_root),
        "dense_model_evaluated": False,
    }


def evaluation_memory_limits(
    available_gpu_bytes: list[int], ram_bytes: int,
    overrides: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Choose conservative Accelerate caps while leaving runtime headroom."""
    if not available_gpu_bytes or any(value <= 0 for value in available_gpu_bytes):
        raise ValueError("At least one positive available GPU memory size is required")
    if ram_bytes <= 0:
        raise ValueError("Positive system RAM is required")
    env = os.environ if overrides is None else overrides
    gpu_default = min(60, int(min(available_gpu_bytes) / (1024**3) * 0.75))
    cpu_default = min(48, int(ram_bytes / (1024**3) * 0.75))
    gpu_limit = int(env.get("NEC_EVAL_GPU_MEMORY_GIB", str(gpu_default)))
    cpu_limit = int(env.get("NEC_EVAL_CPU_MEMORY_GIB", str(cpu_default)))
    if gpu_limit < 1 or cpu_limit < 1:
        raise ValueError("Evaluation memory caps must be positive integer GiB values")
    return gpu_limit, cpu_limit


def effective_ram_bytes() -> int:
    """Return the smaller of host RAM and any Linux cgroup memory limit."""
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


def resource_monitor(stop: threading.Event, run_root: Path) -> None:
    target = run_root / "logs/resources.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    while not stop.wait(60):
        record = {"utc": datetime.now(timezone.utc).isoformat()}
        try:
            record["nvidia_smi"] = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ], text=True).strip().splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            record["nvidia_smi_error"] = f"{type(error).__name__}: {error}"
        disk = shutil.disk_usage(run_root)
        record["run_root_free_bytes"] = disk.free
        try:
            with Path("/proc/meminfo").open() as source:
                for line in source:
                    key, value, *_ = line.split()
                    if key in {"MemTotal:", "MemAvailable:"}:
                        record[key[:-1].lower() + "_bytes"] = int(value) * 1024
        except OSError as error:
            record["meminfo_error"] = str(error)
        with target.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
        print("RESOURCE " + json.dumps(record, sort_keys=True), flush=True)


def run_logged(command: list[str], log_path: Path, env: dict[str, str], on_line=None) -> None:
    print("COMMAND " + json.dumps(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nSTART_UTC {datetime.now(timezone.utc).isoformat()}\n")
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
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
            if on_line is not None:
                on_line(line.rstrip("\n"))
        status = process.wait()
        log.write(f"END_UTC {datetime.now(timezone.utc).isoformat()}\nEXIT_STATUS {status}\n")
        log.flush()
    if status:
        raise subprocess.CalledProcessError(status, command)


def compile_command(config: dict, checkpoint: Path, resume: bool) -> list[str]:
    calibration = config["calibration"]
    compression = config["compression"]
    command = [
        sys.executable, str(COMPILER),
        "--output", str(checkpoint),
        "--start-layer", "0", "--end-layer", "31",
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
        "--model-id", config["model"]["id"],
        "--model-revision", config["model"]["revision"],
        "--device", "cuda:0",
    ]
    if resume:
        command.append("--resume")
    return command


def run_compile(workspace: Path, run_root: Path, commit: str, config: dict) -> Path:
    checkpoint = run_root / "checkpoint"
    manifest_path = checkpoint / "model_manifest.json"
    progress_path = run_root / "compile_progress.json"
    checkpoint.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(CONFIG_PATH)
    env = os.environ.copy()
    env["NEC_SOURCE_COMMIT"] = commit
    env["HF_HOME"] = str(workspace / "cache/huggingface")
    env["HF_HUB_CACHE"] = str(workspace / "cache/huggingface/hub")
    env["HF_DATASETS_CACHE"] = str(workspace / "cache/huggingface/datasets")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("layers_compressed") == list(range(32)):
            # Do not trust the manifest's layer list alone: re-hash and inspect
            # all 32 serialized layers and their calibration artifacts on resume.
            checkpoint_audit = validate_high_fidelity_checkpoint(
                checkpoint, reconstruct=False,
            )
            verify_checkpoint_matches_run_config(checkpoint, config)
            record = {
                "status": "COMPLETE",
                "reason": "all 32 layers and persisted inputs passed checkpoint audit; compile skipped",
                "source_commit": commit,
                "run_config_sha256": config_hash,
                "checkpoint_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_audit": checkpoint_audit,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(progress_path, record)
            print(json.dumps(record, indent=2), flush=True)
            return checkpoint
    else:
        existing = list(checkpoint.glob("compressed_layer_*.pt"))
        if existing:
            raise RuntimeError("Checkpoint files exist without model_manifest.json; refusing to guess resume state")

    attempt = utc_id()
    progress = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "run_config_sha256": config_hash,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "calibration_dataset_revision": WIKITEXT_REVISION,
        "target_layers": list(range(32)),
        "checkpoint_dir": str(checkpoint),
        "resume": manifest_path.exists(),
        "runtime": environment_snapshot(workspace, run_root, commit),
    }
    atomic_json(progress_path, progress)
    started = time.monotonic()

    def update_compile_progress(line: str) -> None:
        layer_number = None
        if line.startswith("layer="):
            try:
                layer_number = int(line.split("=", 1)[1].split()[0])
            except (ValueError, IndexError):
                pass
        elif line.startswith("{"):
            try:
                record = json.loads(line)
                if "layer" in record:
                    layer_number = int(record["layer"])
                    progress["last_layer_metrics"] = record
            except (ValueError, TypeError):
                pass
        if layer_number is not None:
            progress["current_layer"] = layer_number
            try:
                current_manifest = json.loads(manifest_path.read_text())
                progress["layers_compressed"] = current_manifest.get("layers_compressed", [])
            except (OSError, json.JSONDecodeError):
                pass
            progress["last_update_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(progress_path, progress)

    try:
        run_logged(
            compile_command(config, checkpoint, manifest_path.exists()),
            run_root / f"logs/compile-{attempt}.log",
            env,
            on_line=update_compile_progress,
        )
    except BaseException as error:
        progress.update({
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(progress_path, progress)
        raise

    if not manifest_path.is_file():
        raise RuntimeError("Compiler exited without a checkpoint manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("layers_compressed") != list(range(32)):
        raise RuntimeError(f"Expected all 32 layers, found {manifest.get('layers_compressed')}")
    for index in range(32):
        name = f"compressed_layer_{index}.pt"
        path = checkpoint / name
        if not path.is_file():
            raise RuntimeError(f"Missing compressed layer artifact: {name}")
        expected_size = manifest["checkpoint_file_sizes"].get(name)
        expected_hash = manifest["checkpoint_file_sha256"].get(name)
        if path.stat().st_size != int(expected_size or -1) or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Layer {index} failed persisted size/SHA-256 verification")
    checkpoint_audit = validate_high_fidelity_checkpoint(checkpoint, reconstruct=False)
    verify_checkpoint_matches_run_config(checkpoint, config)
    progress.update({
        "status": "COMPLETE",
        "layers_compressed": list(range(32)),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "serialized_checkpoint_bytes": manifest["serialized_checkpoint_bytes"],
        "checkpoint_audit": checkpoint_audit,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    })
    atomic_json(progress_path, progress)
    print(json.dumps({
        "status": progress["status"],
        "layers": len(progress["layers_compressed"]),
        "checkpoint_bytes": progress["serialized_checkpoint_bytes"],
        "checkpoint_manifest_sha256": progress["checkpoint_manifest_sha256"],
        "elapsed_seconds": progress["elapsed_seconds"],
    }, indent=2), flush=True)
    return checkpoint


def verify_result(
    path: Path, manifest_hash: str, required_status: str = "COMPLETE",
    source_commit_hash: str | None = None,
) -> dict | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("status") != required_status:
        return None
    if record.get("checkpoint_manifest_sha256") != manifest_hash:
        raise RuntimeError(f"Existing result at {path} belongs to a different checkpoint")
    if source_commit_hash is not None and record.get("source_commit") != source_commit_hash:
        raise RuntimeError(f"Existing result at {path} belongs to a different source commit")
    return record


def publish_json(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite completed result: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def install_harness(env: dict[str, str], log_path: Path) -> None:
    specification = (
        "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@"
        + LM_EVAL_COMMIT
    )
    run_logged(
        [sys.executable, "-m", "pip", "install", specification],
        log_path,
        env,
    )
    actual = importlib.metadata.version("lm_eval")
    if actual != "0.4.13":
        raise RuntimeError(f"Expected lm-evaluation-harness 0.4.13, found {actual}")


def run_evaluation(workspace: Path, run_root: Path, commit: str, checkpoint: Path) -> None:
    manifest_path = checkpoint / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("layers_compressed") != list(range(32)):
        raise RuntimeError("Refusing evaluation: checkpoint is not complete through layer 31")
    checkpoint_hash = sha256_file(manifest_path)
    results = run_root / "evaluation/results" / checkpoint_hash / commit
    results.mkdir(parents=True, exist_ok=True)
    ppl_result = results / "wikitext2.json"
    mmlu_result = results / "mmlu_5shot.json"
    env = os.environ.copy()
    env["NEC_SOURCE_COMMIT"] = commit
    available_gpu_memory = [
        torch.cuda.mem_get_info(index)[0]
        for index in range(torch.cuda.device_count())
    ]
    ram_bytes = effective_ram_bytes()
    gpu_cap, cpu_cap = evaluation_memory_limits(available_gpu_memory, ram_bytes, env)
    env["NEC_EVAL_GPU_MEMORY_GIB"] = str(gpu_cap)
    env["NEC_EVAL_CPU_MEMORY_GIB"] = str(cpu_cap)
    env["HF_HOME"] = str(workspace / "cache/huggingface")
    env["HF_HUB_CACHE"] = str(workspace / "cache/huggingface/hub")
    env["HF_DATASETS_CACHE"] = str(workspace / "cache/huggingface/datasets")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    evaluation_progress = run_root / "evaluation/evaluation_progress.json"
    progress = {
        "status": "RUNNING",
        "source_commit": commit,
        "checkpoint_manifest_sha256": checkpoint_hash,
        "gpu_memory_cap_gib": gpu_cap,
        "cpu_memory_cap_gib": cpu_cap,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    atomic_json(evaluation_progress, progress)
    try:
        if verify_result(ppl_result, checkpoint_hash, source_commit_hash=commit) is None:
            attempt = run_root / "evaluation/attempts" / f"ppl-{utc_id()}"
            attempt.mkdir(parents=True, exist_ok=False)
            output = attempt / "wikitext2.json"
            progress["stages"]["wikitext2"] = {
                "status": "RUNNING", "attempt": str(attempt),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(evaluation_progress, progress)
            run_logged([
                sys.executable, str(PPL_EVALUATOR),
                "--checkpoint-dir", str(checkpoint),
                "--offload-folder", str(attempt / "offload"),
                "--output", str(output),
                "--context-length", "2048",
                "--stride", "512",
                "--max-eval-tokens", "10000",
                "--full-test",
                "--skip-reconstruct-audit",
            ], attempt / "wikitext2.log", env)
            evaluated = json.loads(output.read_text())
            if evaluated.get("status") != "COMPLETE" or evaluated.get("dense_model_evaluated") is not False:
                raise RuntimeError("PPL evaluator did not produce a complete NEC-only report")
            if (
                evaluated.get("checkpoint_manifest_sha256") != checkpoint_hash
                or evaluated.get("source_commit") != commit
            ):
                raise RuntimeError("PPL report checkpoint/source identity mismatch")
            for label in ("primary_10k", "full_test"):
                metric = evaluated["metrics"].get(label)
                if metric is None or metric["scored_tokens"] != metric["input_tokens"] - 1:
                    raise RuntimeError(f"PPL report failed scored-token checks for {label}")
                if metric["nll_formula_crosscheck"].get("passed") is not True:
                    raise RuntimeError(f"PPL report independent NLL cross-check failed for {label}")
            publish_json(output, ppl_result)
            progress["stages"]["wikitext2"] = {
                "status": "COMPLETE", "attempt": str(attempt),
                "report": str(ppl_result), "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(evaluation_progress, progress)
        else:
            progress["stages"]["wikitext2"] = {"status": "REUSED_VERIFIED_RESULT"}
            atomic_json(evaluation_progress, progress)

        if verify_result(mmlu_result, checkpoint_hash, source_commit_hash=commit) is None:
            attempt = run_root / "evaluation/attempts" / f"mmlu-{utc_id()}"
            attempt.mkdir(parents=True, exist_ok=False)
            progress["stages"]["mmlu_5shot"] = {
                "status": "INSTALLING_HARNESS", "attempt": str(attempt),
                "started_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(evaluation_progress, progress)
            install_harness(env, attempt / "install_lm_eval.log")
            output = attempt / "mmlu_5shot.json"
            progress["stages"]["mmlu_5shot"]["status"] = "RUNNING"
            atomic_json(evaluation_progress, progress)
            run_logged([
                sys.executable, str(MMLU_EVALUATOR),
                "--checkpoint-dir", str(checkpoint),
                "--output", str(output),
                "--batch-size", "1",
                "--smoke-first",
            ], attempt / "mmlu_5shot.log", env)
            evaluated = json.loads(output.read_text())
            if (
                evaluated.get("status") != "COMPLETE"
                or evaluated.get("dense_model_evaluated") is not False
                or evaluated.get("total_questions") != 14_042
                or len(evaluated.get("per_subject", {})) != 57
                or evaluated.get("adapter_smoke", {}).get("status") != "PASS"
            ):
                raise RuntimeError("MMLU report is incomplete or its adapter smoke failed")
            if (
                evaluated.get("checkpoint_manifest_sha256") != checkpoint_hash
                or evaluated.get("source_commit") != commit
            ):
                raise RuntimeError("MMLU report checkpoint/source identity mismatch")
            publish_json(output, mmlu_result)
            progress["stages"]["mmlu_5shot"] = {
                "status": "COMPLETE", "attempt": str(attempt),
                "report": str(mmlu_result), "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
            atomic_json(evaluation_progress, progress)
        else:
            progress["stages"]["mmlu_5shot"] = {"status": "REUSED_VERIFIED_RESULT"}
            atomic_json(evaluation_progress, progress)
    except BaseException as error:
        progress.update({
            "status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "failed_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(evaluation_progress, progress)
        raise

    ppl = verify_result(ppl_result, checkpoint_hash, source_commit_hash=commit)
    mmlu = verify_result(mmlu_result, checkpoint_hash, source_commit_hash=commit)
    if ppl is None or mmlu is None:
        raise RuntimeError("Evaluation output missing one or more complete result files")
    final = {
        "status": "COMPLETE",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_commit": commit,
        "run_config_sha256": sha256_file(CONFIG_PATH),
        "checkpoint_manifest_sha256": checkpoint_hash,
        "wikitext2_report": str(ppl_result),
        "wikitext2_report_sha256": sha256_file(ppl_result),
        "mmlu_5shot_report": str(mmlu_result),
        "mmlu_5shot_report_sha256": sha256_file(mmlu_result),
        "wikitext2_primary_10k": ppl["metrics"]["primary_10k"],
        "wikitext2_full_test": ppl["metrics"]["full_test"],
        "mmlu_5shot_accuracy": mmlu["overall_5shot_accuracy"],
        "mmlu_total_questions": mmlu["total_questions"],
        "mmlu_per_subject": mmlu["per_subject"],
        "dense_model_evaluated": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    final_path = results / "final_results.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        expected_hashes = {
            "wikitext2_report_sha256": sha256_file(ppl_result),
            "mmlu_5shot_report_sha256": sha256_file(mmlu_result),
        }
        if (
            existing.get("checkpoint_manifest_sha256") != checkpoint_hash
            or existing.get("source_commit") != commit
            or any(existing.get(key) != value for key, value in expected_hashes.items())
        ):
            raise RuntimeError("Existing final result does not match checkpoint/source/report identities")
    else:
        atomic_json(final_path, final)
    progress.update({
        "status": "COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "final_result": str(final_path),
    })
    atomic_json(evaluation_progress, progress)
    print(json.dumps(final, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile", "evaluate", "all"), default="all")
    args = parser.parse_args()
    workspace, run_root = roots()
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".nec-run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another NEC runner holds {lock_path}") from error
        config = load_config()
        commit = source_commit()
        atomic_json(run_root / "runtime.json", environment_snapshot(workspace, run_root, commit))
        checkpoint = run_root / "checkpoint"
        stop = threading.Event()
        watcher = threading.Thread(target=resource_monitor, args=(stop, run_root), daemon=True)
        watcher.start()
        try:
            if args.mode in ("compile", "all"):
                checkpoint = run_compile(workspace, run_root, commit, config)
            if args.mode in ("evaluate", "all"):
                run_evaluation(workspace, run_root, commit, checkpoint)
        finally:
            stop.set()
            watcher.join(timeout=5)


if __name__ == "__main__":
    main()
