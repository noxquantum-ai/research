#!/usr/bin/env python3
"""Refresh the self-contained Kaggle source bundle and its provenance digest."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "kaggle_execution" / "function_aware_tn"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    shutil.copytree(REPO / "nox", BUNDLE / "nox", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    benchmark_dir = BUNDLE / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "benchmarks" / "benchmark_day3_cascade.py",
                 benchmark_dir / "benchmark_day3_cascade.py")
    (benchmark_dir / "__init__.py").touch()
    # Keep the frozen execution bundle clean without editing any historical
    # source files in the repository.
    for source in BUNDLE.rglob("*.py"):
        normalized = "\n".join(line.rstrip() for line in source.read_text().splitlines()).rstrip("\n") + "\n"
        source.write_text(normalized)
    for cached_bytecode in BUNDLE.rglob("*.pyc"):
        cached_bytecode.unlink()

    relative_files = [
        Path("driver.py"), Path("kernel-metadata.json"), Path("requirements.txt"),
        Path("README.md"), Path("benchmarks/__init__.py"),
        Path("benchmarks/benchmark_day3_cascade.py"),
    ]
    relative_files.extend(
        path.relative_to(BUNDLE)
        for path in sorted((BUNDLE / "nox").rglob("*.py"))
    )
    records = [
        {"path": path.as_posix(), "sha256": sha256(BUNDLE / path)}
        for path in relative_files
    ]
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
    ).strip()
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "base_commit": base_commit,
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True,
        ).strip(),
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": records,
    }
    (BUNDLE / "code_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({
        "bundle": str(BUNDLE), "base_commit": base_commit,
        "bundle_sha256": manifest["bundle_sha256"], "source_file_count": len(records),
    }, indent=2))


if __name__ == "__main__":
    main()
