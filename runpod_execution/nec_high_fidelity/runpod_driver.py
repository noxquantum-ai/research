#!/usr/bin/env python3
"""Resumable RunPod orchestration for the existing High-Fidelity NEC run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
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
        or wikitext["inference_batch_size_policy"] != "shared_hardware_adaptive"
        or wikitext["batch_size_tiers_by_minimum_free_vram_gib"] != {
            "70": 8, "42": 4, "28": 2, "0": 1,
        }
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
        or mmlu["batch_size_policy"] != "shared_hardware_adaptive"
        or mmlu["batch_size_tiers_by_minimum_free_vram_gib"] != {
            "70": 16, "42": 8, "28": 4, "0": 2,
        }
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
        "dense_model_evaluation_planned": True,
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


def evaluation_batch_sizes(
    available_gpu_bytes: list[int], overrides: dict[str, str] | None = None,
) -> tuple[int, int]:
    """Choose one hardware-sized PPL/MMLU batch pair for both model variants."""
    if not available_gpu_bytes or any(value <= 0 for value in available_gpu_bytes):
        raise ValueError("At least one positive available GPU memory size is required")
    minimum_free_gib = min(available_gpu_bytes) / (1024**3)
    if minimum_free_gib >= 70:
        defaults = (8, 16)
    elif minimum_free_gib >= 42:
        defaults = (4, 8)
    elif minimum_free_gib >= 28:
        defaults = (2, 4)
    else:
        defaults = (1, 2)
    env = os.environ if overrides is None else overrides
    ppl_batch = int(env.get("NEC_PPL_BATCH_SIZE", str(defaults[0])))
    mmlu_batch = int(env.get("NEC_MMLU_BATCH_SIZE", str(defaults[1])))
    if ppl_batch < 1 or mmlu_batch < 1:
        raise ValueError("Evaluation batch sizes must be positive integers")
    return ppl_batch, mmlu_batch


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
    path: Path, manifest_hash: str | None, required_status: str = "COMPLETE",
    source_commit_hash: str | None = None, model_variant: str | None = None,
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
    if model_variant is not None and record.get("model_variant") != model_variant:
        raise RuntimeError(f"Existing result at {path} belongs to a different model variant")
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
    expected_runtime = {
        "torch": "2.11.0+cu128",
        "transformers": "5.16.1",
        "huggingface-hub": "1.29.0",
        "safetensors": "0.8.0",
        "accelerate": "1.14.0",
        "datasets": "4.8.5",
        "pyarrow": "23.0.1",
        "numpy": "2.1.3",
        "scipy": "1.16.3",
    }
    for package, expected in expected_runtime.items():
        installed = importlib.metadata.version(package)
        if installed != expected:
            raise RuntimeError(
                f"Installing lm-eval changed pinned {package}: {installed}, expected {expected}"
            )
    run_logged([sys.executable, "-m", "pip", "check"], log_path, env)


def validate_ppl_report(record: dict, variant: str, checkpoint_hash: str | None, commit: str) -> None:
    if record.get("status") != "COMPLETE" or record.get("model_variant") != variant:
        raise RuntimeError(f"{variant} PPL evaluator did not produce a complete report")
    if record.get("dense_model_evaluated") is not (variant == "dense"):
        raise RuntimeError(f"{variant} PPL report has an inconsistent dense-model marker")
    if record.get("checkpoint_manifest_sha256") != checkpoint_hash or record.get("source_commit") != commit:
        raise RuntimeError(f"{variant} PPL report checkpoint/source identity mismatch")
    if not isinstance(record.get("protocol_fingerprint"), str):
        raise RuntimeError(f"{variant} PPL report has no protocol fingerprint")
    protocol_batch = record.get("protocol_identity", {}).get("inference_batch_size")
    if not isinstance(protocol_batch, int) or protocol_batch < 1:
        raise RuntimeError(f"{variant} PPL report has an invalid batch size")
    for label in ("primary_10k", "full_test"):
        metric = record.get("metrics", {}).get(label)
        if metric is None or metric["scored_tokens"] != metric["input_tokens"] - 1:
            raise RuntimeError(f"{variant} PPL report failed scored-token checks for {label}")
        if metric.get("nll_formula_crosscheck", {}).get("passed") is not True:
            raise RuntimeError(f"{variant} PPL independent NLL cross-check failed for {label}")
        if not math.isfinite(metric["mean_nll"]) or not math.isfinite(metric["perplexity"]):
            raise RuntimeError(f"{variant} PPL report contains non-finite metrics for {label}")
        if abs(metric["perplexity"] - math.exp(metric["mean_nll"])) > 1e-9 * max(1.0, metric["perplexity"]):
            raise RuntimeError(f"{variant} PPL is inconsistent with exp(mean NLL) for {label}")
        if not 0.0 <= metric["next_token_accuracy"] <= 1.0:
            raise RuntimeError(f"{variant} next-token accuracy is invalid for {label}")
        if metric.get("inference_batch_size") != protocol_batch:
            raise RuntimeError(f"{variant} PPL batch metadata is inconsistent for {label}")
        if not 0 < metric.get("forward_passes", 0) <= metric.get("window_count", 0):
            raise RuntimeError(f"{variant} PPL forward-pass count is invalid for {label}")


def validate_mmlu_report(record: dict, variant: str, checkpoint_hash: str | None, commit: str) -> None:
    if (
        record.get("status") != "COMPLETE"
        or record.get("model_variant") != variant
        or record.get("dense_model_evaluated") is not (variant == "dense")
        or record.get("checkpoint_manifest_sha256") != checkpoint_hash
        or record.get("source_commit") != commit
        or record.get("total_questions") != 14_042
        or len(record.get("per_subject", {})) != 57
        or record.get("adapter_smoke", {}).get("status") != "PASS"
        or not isinstance(record.get("protocol_fingerprint"), str)
        or record.get("evaluation_input_identity", {}).get("samples") != 14_042
        or not isinstance(record.get("evaluation_input_identity", {}).get("sha256"), str)
        or record.get("response_finiteness_audit", {}).get("status") != "PASS"
        or record.get("response_finiteness_audit", {}).get("finite_numerical_values_checked", 0) <= 0
    ):
        raise RuntimeError(f"{variant} MMLU report is incomplete or inconsistent")
    accuracy = record.get("overall_5shot_accuracy")
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise RuntimeError(f"{variant} MMLU accuracy is invalid")
    if sum(item["questions"] for item in record["per_subject"].values()) != 14_042:
        raise RuntimeError(f"{variant} per-subject MMLU counts do not sum to 14,042")
    raw_path = Path(record.get("raw_harness_results", ""))
    if (
        not raw_path.is_file()
        or sha256_file(raw_path) != record.get("raw_harness_results_sha256")
    ):
        raise RuntimeError(f"{variant} raw MMLU harness result is missing or corrupt")


def validate_matched_evaluations(
    dense_ppl: dict, nec_ppl: dict, dense_mmlu: dict, nec_mmlu: dict,
) -> dict:
    """Prove that both variants used the same model inputs and benchmark protocols."""
    if dense_ppl["protocol_fingerprint"] != nec_ppl["protocol_fingerprint"]:
        raise RuntimeError("Dense and NEC WikiText-2 protocol fingerprints differ")
    for label in ("primary_10k", "full_test"):
        dense_metric = dense_ppl["metrics"][label]
        nec_metric = nec_ppl["metrics"][label]
        for key in (
            "input_tokens", "scored_tokens", "token_ids_sha256", "bos_token_id",
            "inference_batch_size", "window_specification",
        ):
            if dense_metric[key] != nec_metric[key]:
                raise RuntimeError(f"Dense and NEC WikiText-2 {label} differ in {key}")
    if dense_mmlu["protocol_fingerprint"] != nec_mmlu["protocol_fingerprint"]:
        raise RuntimeError("Dense and NEC MMLU protocol fingerprints differ")
    if dense_mmlu["evaluation_input_identity"] != nec_mmlu["evaluation_input_identity"]:
        raise RuntimeError("Dense and NEC MMLU prompt/document identities differ")
    if set(dense_mmlu["per_subject"]) != set(nec_mmlu["per_subject"]):
        raise RuntimeError("Dense and NEC MMLU subject sets differ")
    for subject in dense_mmlu["per_subject"]:
        if dense_mmlu["per_subject"][subject]["questions"] != nec_mmlu["per_subject"][subject]["questions"]:
            raise RuntimeError(f"Dense and NEC MMLU question counts differ for {subject}")
    for variant, ppl, mmlu in (
        ("dense", dense_ppl, dense_mmlu),
        ("high_fidelity_nec", nec_ppl, nec_mmlu),
    ):
        if ppl["model_tokenizer_compatibility"] != mmlu["model_tokenizer_compatibility"]:
            raise RuntimeError(f"{variant} PPL and MMLU model/tokenizer audits differ")
        if ppl["independent_numerical_coefficients"] != mmlu["independent_numerical_coefficients"]:
            raise RuntimeError(f"{variant} PPL and MMLU coefficient counts differ")
    return {
        "status": "PASS",
        "same_model_id_and_revision": True,
        "same_fp16_loading_and_sdpa_backend": True,
        "same_tokenizer_and_architecture_validation": True,
        "wikitext2_protocol_fingerprint": dense_ppl["protocol_fingerprint"],
        "wikitext2_identical_token_streams": True,
        "mmlu_protocol_fingerprint": dense_mmlu["protocol_fingerprint"],
        "mmlu_evaluation_input_sha256": dense_mmlu["evaluation_input_identity"]["sha256"],
        "mmlu_identical_subject_question_counts": True,
        "separate_processes_prevent_cross_variant_weight_contamination": True,
    }


def run_evaluation(workspace: Path, run_root: Path, commit: str, checkpoint: Path) -> None:
    manifest_path = checkpoint / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("layers_compressed") != list(range(32)):
        raise RuntimeError("Refusing evaluation: checkpoint is not complete through layer 31")
    checkpoint_hash = sha256_file(manifest_path)
    results = run_root / "evaluation/results" / checkpoint_hash / commit
    results.mkdir(parents=True, exist_ok=True)
    report_paths = {
        ("dense", "ppl"): results / "dense_wikitext2.json",
        ("high_fidelity_nec", "ppl"): results / "nec_wikitext2.json",
        ("dense", "mmlu"): results / "dense_mmlu_5shot.json",
        ("high_fidelity_nec", "mmlu"): results / "nec_mmlu_5shot.json",
    }
    env = os.environ.copy()
    env["NEC_SOURCE_COMMIT"] = commit
    available_gpu_memory = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    gpu_cap, cpu_cap = evaluation_memory_limits(available_gpu_memory, effective_ram_bytes(), env)
    ppl_batch, mmlu_batch = evaluation_batch_sizes(available_gpu_memory, env)
    env.update({
        "NEC_EVAL_GPU_MEMORY_GIB": str(gpu_cap),
        "NEC_EVAL_CPU_MEMORY_GIB": str(cpu_cap),
        "HF_HOME": str(workspace / "cache/huggingface"),
        "HF_HUB_CACHE": str(workspace / "cache/huggingface/hub"),
        "HF_DATASETS_CACHE": str(workspace / "cache/huggingface/datasets"),
        "PYTHONPATH": str(ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
        "PYTHONUNBUFFERED": "1",
    })
    evaluation_progress = run_root / "evaluation/evaluation_progress.json"
    progress = {
        "status": "RUNNING",
        "source_commit": commit,
        "checkpoint_manifest_sha256": checkpoint_hash,
        "gpu_memory_cap_gib": gpu_cap,
        "cpu_memory_cap_gib": cpu_cap,
        "ppl_batch_size": ppl_batch,
        "mmlu_batch_size": mmlu_batch,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }
    atomic_json(evaluation_progress, progress)
    try:
        for variant in ("dense", "high_fidelity_nec"):
            expected_hash = None if variant == "dense" else checkpoint_hash
            result_path = report_paths[(variant, "ppl")]
            stage = f"{variant}_wikitext2"
            existing = verify_result(
                result_path, expected_hash, source_commit_hash=commit, model_variant=variant,
            )
            if existing is None:
                attempt = run_root / "evaluation/attempts" / f"{stage}-{utc_id()}"
                attempt.mkdir(parents=True, exist_ok=False)
                output = attempt / result_path.name
                progress["stages"][stage] = {
                    "status": "RUNNING", "attempt": str(attempt),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(evaluation_progress, progress)
                command = [
                    sys.executable, str(PPL_EVALUATOR), "--variant", variant,
                    "--offload-folder", str(attempt / "offload"),
                    "--output", str(output), "--context-length", "2048", "--stride", "512",
                    "--max-eval-tokens", "10000", "--batch-size", str(ppl_batch), "--full-test",
                ]
                if variant == "high_fidelity_nec":
                    command.extend(["--checkpoint-dir", str(checkpoint), "--skip-reconstruct-audit"])
                run_logged(command, attempt / "wikitext2.log", env)
                evaluated = json.loads(output.read_text())
                validate_ppl_report(evaluated, variant, expected_hash, commit)
                publish_json(output, result_path)
                progress["stages"][stage] = {
                    "status": "COMPLETE", "attempt": str(attempt), "report": str(result_path),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            else:
                validate_ppl_report(existing, variant, expected_hash, commit)
                progress["stages"][stage] = {"status": "REUSED_VERIFIED_RESULT"}
            atomic_json(evaluation_progress, progress)

        missing_mmlu = any(
            verify_result(
                report_paths[(variant, "mmlu")], None if variant == "dense" else checkpoint_hash,
                source_commit_hash=commit, model_variant=variant,
            ) is None
            for variant in ("dense", "high_fidelity_nec")
        )
        if missing_mmlu:
            install_attempt = run_root / "evaluation/attempts" / f"install-mmlu-{utc_id()}"
            install_attempt.mkdir(parents=True, exist_ok=False)
            progress["stages"]["install_mmlu_harness"] = {
                "status": "RUNNING", "attempt": str(install_attempt),
            }
            atomic_json(evaluation_progress, progress)
            install_harness(env, install_attempt / "install_lm_eval.log")
            progress["stages"]["install_mmlu_harness"]["status"] = "COMPLETE"
            atomic_json(evaluation_progress, progress)

        for variant in ("dense", "high_fidelity_nec"):
            expected_hash = None if variant == "dense" else checkpoint_hash
            result_path = report_paths[(variant, "mmlu")]
            stage = f"{variant}_mmlu_5shot"
            existing = verify_result(
                result_path, expected_hash, source_commit_hash=commit, model_variant=variant,
            )
            if existing is None:
                attempt = run_root / "evaluation/attempts" / f"{stage}-{utc_id()}"
                attempt.mkdir(parents=True, exist_ok=False)
                output = attempt / result_path.name
                progress["stages"][stage] = {
                    "status": "RUNNING", "attempt": str(attempt),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(evaluation_progress, progress)
                command = [
                    sys.executable, str(MMLU_EVALUATOR), "--variant", variant,
                    "--output", str(output), "--batch-size", str(mmlu_batch), "--smoke-first",
                ]
                if variant == "high_fidelity_nec":
                    command.extend(["--checkpoint-dir", str(checkpoint)])
                run_logged(command, attempt / "mmlu_5shot.log", env)
                evaluated = json.loads(output.read_text())
                validate_mmlu_report(evaluated, variant, expected_hash, commit)
                publish_json(output, result_path)
                progress["stages"][stage] = {
                    "status": "COMPLETE", "attempt": str(attempt), "report": str(result_path),
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                }
            else:
                validate_mmlu_report(existing, variant, expected_hash, commit)
                progress["stages"][stage] = {"status": "REUSED_VERIFIED_RESULT"}
            atomic_json(evaluation_progress, progress)
    except BaseException as error:
        progress.update({
            "status": "FAILED", "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "failed_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(evaluation_progress, progress)
        raise

    reports = {
        key: verify_result(
            path, None if key[0] == "dense" else checkpoint_hash,
            source_commit_hash=commit, model_variant=key[0],
        )
        for key, path in report_paths.items()
    }
    if any(value is None for value in reports.values()):
        raise RuntimeError("Evaluation output is missing one or more complete result files")
    dense_ppl = reports[("dense", "ppl")]
    nec_ppl = reports[("high_fidelity_nec", "ppl")]
    dense_mmlu = reports[("dense", "mmlu")]
    nec_mmlu = reports[("high_fidelity_nec", "mmlu")]
    matched_audit = validate_matched_evaluations(dense_ppl, nec_ppl, dense_mmlu, nec_mmlu)

    dense_primary = dense_ppl["metrics"]["primary_10k"]
    nec_primary = nec_ppl["metrics"]["primary_10k"]
    dense_full = dense_ppl["metrics"]["full_test"]
    nec_full = nec_ppl["metrics"]["full_test"]
    dense_mmlu_accuracy = dense_mmlu["overall_5shot_accuracy"]
    nec_mmlu_accuracy = nec_mmlu["overall_5shot_accuracy"]
    report_hashes = {f"{variant}_{benchmark}_sha256": sha256_file(path) for (variant, benchmark), path in report_paths.items()}
    final = {
        "status": "COMPLETE",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_commit": commit,
        "run_config_sha256": sha256_file(CONFIG_PATH),
        "checkpoint_manifest_sha256": checkpoint_hash,
        "matched_protocol_audit": matched_audit,
        "reports": {f"{variant}_{benchmark}": str(path) for (variant, benchmark), path in report_paths.items()},
        "report_hashes": report_hashes,
        "dense": {
            "independent_numerical_coefficients": dense_ppl["independent_numerical_coefficients"],
            "structural_parameter_reduction": 0.0,
            "parameter_compression_ratio": 1.0,
            "wikitext2_primary_10k": dense_primary,
            "wikitext2_full_test": dense_full,
            "mmlu_5shot_accuracy": dense_mmlu_accuracy,
            "mmlu_total_questions": dense_mmlu["total_questions"],
            "mmlu_per_subject": dense_mmlu["per_subject"],
        },
        "high_fidelity_nec": {
            "independent_numerical_coefficients": nec_ppl["independent_numerical_coefficients"],
            "structural_parameter_reduction": nec_ppl["structural_parameter_reduction"],
            "parameter_compression_ratio": nec_ppl["parameter_compression_ratio"],
            "wikitext2_primary_10k": nec_primary,
            "wikitext2_full_test": nec_full,
            "mmlu_5shot_accuracy": nec_mmlu_accuracy,
            "mmlu_total_questions": nec_mmlu["total_questions"],
            "mmlu_per_subject": nec_mmlu["per_subject"],
        },
        "comparison": {
            "primary_10k_ppl_ratio_nec_over_dense": nec_primary["perplexity"] / dense_primary["perplexity"],
            "primary_10k_mean_nll_delta_nec_minus_dense": nec_primary["mean_nll"] - dense_primary["mean_nll"],
            "primary_10k_next_token_accuracy_retention": nec_primary["next_token_accuracy"] / dense_primary["next_token_accuracy"],
            "full_test_ppl_ratio_nec_over_dense": nec_full["perplexity"] / dense_full["perplexity"],
            "full_test_mean_nll_delta_nec_minus_dense": nec_full["mean_nll"] - dense_full["mean_nll"],
            "mmlu_accuracy_retention": nec_mmlu_accuracy / dense_mmlu_accuracy,
            "mmlu_accuracy_percentage_point_delta": 100.0 * (nec_mmlu_accuracy - dense_mmlu_accuracy),
            "mmlu_preserves_at_least_90_percent": nec_mmlu_accuracy / dense_mmlu_accuracy >= 0.90,
        },
        "dense_model_evaluated": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    final_path = results / "final_results.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        if (
            existing.get("checkpoint_manifest_sha256") != checkpoint_hash
            or existing.get("source_commit") != commit
            or existing.get("report_hashes") != report_hashes
        ):
            raise RuntimeError("Existing final result does not match checkpoint/source/report identities")
        final = existing
    else:
        atomic_json(final_path, final)
    progress.update({
        "status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "matched_protocol_audit": matched_audit, "final_result": str(final_path),
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
