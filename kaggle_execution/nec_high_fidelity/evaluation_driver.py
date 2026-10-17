#!/usr/bin/env python3
"""Run complete High-Fidelity NEC WikiText-2 and MMLU evaluations on Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch


MODEL_ID = "NousResearch/Llama-2-7b-hf"
MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
WORK = Path("/kaggle/working/nec_hf_evaluation")
SOURCE = Path("/kaggle/working/nec_source")
LM_EVAL_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"


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
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Source-bundle checksum mismatch: {relative}")
    return manifest


def choose_extracted_source_manifest(manifests: list[Path]) -> Path:
    if not manifests:
        raise FileNotFoundError("No extracted source bundle manifest found in Kaggle inputs")
    by_hash: dict[str, list[Path]] = {}
    for path in manifests:
        verify_source_tree(path.parent)
        by_hash.setdefault(sha256_file(path), []).append(path)
    if len(by_hash) != 1:
        raise ValueError("Kaggle inputs contain conflicting verified NEC source bundles")
    identical = next(iter(by_hash.values()))
    return sorted(
        identical,
        key=lambda path: ("datasets" not in path.parts, len(str(path)), str(path)),
    )[0]


def find_source_input(input_root: Path = Path("/kaggle/input")) -> Path:
    archives = sorted(input_root.rglob("nec_source_bundle.tar.gz"))
    manifests = sorted(input_root.rglob("bundle_manifest.json"))
    if len(archives) > 1 or (archives and manifests):
        raise FileNotFoundError(
            "Expected a single packed bundle or extracted bundles, not mixed inputs; "
            f"found {len(archives)} archive(s), {len(manifests)} manifest(s)"
        )
    if archives:
        return archives[0]
    return choose_extracted_source_manifest(manifests)


def extract_source(source_input: Path) -> dict:
    SOURCE.mkdir(parents=True, exist_ok=True)
    if source_input.name == "bundle_manifest.json":
        input_root = source_input.parent.resolve()
        manifest = verify_source_tree(input_root)
        shutil.copy2(source_input, SOURCE / "bundle_manifest.json")
        for relative in manifest["source_files"]:
            source = (input_root / relative).resolve()
            target = (SOURCE / relative).resolve()
            if not target.is_relative_to(SOURCE.resolve()):
                raise ValueError(f"Unsafe source-bundle path: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    else:
        with tarfile.open(source_input, "r:gz") as bundle:
            for member in bundle.getmembers():
                if not (SOURCE / member.name).resolve().is_relative_to(SOURCE.resolve()):
                    raise ValueError(f"Unsafe source archive path: {member.name}")
            bundle.extractall(SOURCE, filter="data")
        manifest = json.loads((SOURCE / "bundle_manifest.json").read_text())
    return verify_source_tree(SOURCE)


def find_complete_checkpoint() -> Path:
    candidates = []
    for manifest_path in Path("/kaggle/input").rglob("model_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        identity = manifest.get("run_identity", {})
        if (
            identity.get("model_id") == MODEL_ID
            and identity.get("model_revision") == MODEL_REVISION
            and identity.get("layers") == list(range(32))
            and manifest.get("layers_compressed") == list(range(32))
        ):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one complete NEC checkpoint input, found {candidates}"
        )
    return candidates[0]


def install_harness() -> None:
    specification = (
        "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@"
        + LM_EVAL_COMMIT
    )
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet", specification,
    ], check=True)


def run_logged(command: list[str], log_path: Path, env: dict) -> None:
    print(f"COMMAND: {json.dumps(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
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


def package_versions() -> dict:
    import importlib.metadata

    result = {}
    for name in (
        "torch", "transformers", "huggingface-hub", "safetensors", "accelerate",
        "pyarrow", "datasets", "lm_eval", "numpy",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main() -> None:
    started = time.monotonic()
    WORK.mkdir(parents=True, exist_ok=True)
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("Kaggle evaluation requires CUDA")
        source_input = find_source_input()
        source_manifest = extract_source(source_input)
        source_commit = source_manifest["source_commit"]
        prior = find_complete_checkpoint()
        os.environ["NEC_SOURCE_COMMIT"] = source_commit
        os.environ["PYTHONPATH"] = str(SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", "")

        record = {
            "status": "RUNNING",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_commit": source_commit,
            "source_input": str(source_input),
            "source_bundle_manifest_sha256": sha256_file(SOURCE / "bundle_manifest.json"),
            "checkpoint_manifest_sha256": sha256_file(prior / "model_manifest.json"),
            "checkpoint_source": str(prior),
            "runner_sha256": sha256_file(Path(__file__)),
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "working_disk_free_bytes": shutil.disk_usage("/kaggle/working").free,
            },
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "dense_model_evaluated": False,
        }
        atomic_json(WORK / "progress.json", record)
        env = os.environ.copy()
        env["NEC_SOURCE_COMMIT"] = source_commit
        env["PYTHONPATH"] = str(SOURCE) + os.pathsep + env.get("PYTHONPATH", "")

        ppl_path = WORK / "wikitext2.json"
        run_logged([
            sys.executable,
            str(SOURCE / "benchmarks/evaluate_high_fidelity_nec.py"),
            "--checkpoint-dir", str(prior),
            "--offload-folder", str(WORK / "evaluation_offload"),
            "--output", str(ppl_path),
            "--context-length", "2048",
            "--stride", "512",
            "--max-eval-tokens", "10000",
            "--full-test",
            "--skip-reconstruct-audit",
        ], WORK / "wikitext2.log", env)
        ppl = json.loads(ppl_path.read_text())
        if ppl.get("status") != "COMPLETE" or ppl.get("dense_model_evaluated") is not False:
            raise RuntimeError("WikiText-2 evaluator did not produce a valid NEC-only result")
        for name, metrics in (
            ("primary_10k", ppl["metrics"]["primary_10k"]),
            ("full_test", ppl["metrics"]["full_test"]),
        ):
            if metrics is None:
                raise RuntimeError(f"Missing required PPL result: {name}")
            if metrics.get("nll_formula_crosscheck", {}).get("passed") is not True:
                raise RuntimeError(f"{name}: independent NLL formula cross-check did not pass")
            expected_count = metrics["input_tokens"] - 1
            if metrics["scored_tokens"] != expected_count:
                raise RuntimeError(f"{name}: scored-token count mismatch")
            if abs(metrics["perplexity"] - float(torch.exp(torch.tensor(metrics["mean_nll"], dtype=torch.float64)))) > 1e-9 * max(1.0, metrics["perplexity"]):
                raise RuntimeError(f"{name}: PPL != exp(mean NLL)")
        record["status"] = "WIKITEXT2_COMPLETE_MMLU_RUNNING"
        record["wikitext2"] = str(ppl_path)
        record["wikitext2_sha256"] = sha256_file(ppl_path)
        atomic_json(WORK / "progress.json", record)

        install_harness()
        mmlu_path = WORK / "mmlu_5shot.json"
        run_logged([
            sys.executable,
            str(SOURCE / "benchmarks/evaluate_high_fidelity_mmlu.py"),
            "--checkpoint-dir", str(prior),
            "--output", str(mmlu_path),
            "--batch-size", "1",
            "--smoke-first",
        ], WORK / "mmlu.log", env)
        mmlu = json.loads(mmlu_path.read_text())
        if mmlu.get("status") != "COMPLETE" or mmlu.get("dense_model_evaluated") is not False:
            raise RuntimeError("MMLU evaluator did not produce a complete NEC-only result")
        if mmlu.get("total_questions") != 14_042 or len(mmlu.get("per_subject", {})) != 57:
            raise RuntimeError("MMLU result is incomplete")
        if mmlu.get("adapter_smoke", {}).get("status") != "PASS":
            raise RuntimeError("MMLU adapter smoke did not pass before the full run")

        record.update({
            "status": "COMPLETE",
            "wikitext2_sha256": sha256_file(ppl_path),
            "mmlu_5shot": str(mmlu_path),
            "mmlu_5shot_sha256": sha256_file(mmlu_path),
            "package_versions": package_versions(),
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
        })
        atomic_json(WORK / "progress.json", record)
        print(json.dumps({
            "status": "COMPLETE",
            "wikitext2_primary_10k": ppl["metrics"]["primary_10k"],
            "wikitext2_full_test": ppl["metrics"]["full_test"],
            "mmlu_5shot_accuracy": mmlu["overall_5shot_accuracy"],
            "mmlu_questions": mmlu["total_questions"],
            "elapsed_seconds": record["elapsed_seconds"],
        }, indent=2), flush=True)
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
