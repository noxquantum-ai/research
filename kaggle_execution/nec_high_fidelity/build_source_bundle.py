#!/usr/bin/env python3
"""Create a deterministic source-only tarball for a private Kaggle dataset."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    "benchmarks/__init__.py",
    "benchmarks/benchmark_day3_cascade.py",
    "benchmarks/benchmark_day4_perplexity.py",
    "benchmarks/evaluate_high_fidelity_nec.py",
    "benchmarks/evaluate_high_fidelity_mmlu.py",
    "scripts/compile_high_fidelity.py",
    "kaggle_execution/nec_high_fidelity/run_config.json",
)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_paths() -> list[Path]:
    paths = [ROOT / path for path in SOURCE_FILES]
    paths.extend(sorted((ROOT / "nox").rglob("*.py")))
    paths = [path for path in paths if "__pycache__" not in path.parts]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    files = source_paths()
    entries = {
        path.relative_to(ROOT).as_posix(): path.read_bytes()
        for path in files
    }
    manifest = {
        "source_commit": commit,
        "source_files": {
            name: {"bytes": len(content), "sha256": sha256(content)}
            for name, content in sorted(entries.items())
        },
        "bundle_format": 1,
    }
    entries["bundle_manifest.json"] = (json.dumps(
        manifest, indent=2, sort_keys=True,
    ) + "\n").encode()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0, compresslevel=9) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, content in sorted(entries.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    archive.addfile(info, io.BytesIO(content))
    print(json.dumps({
        **manifest,
        "archive": str(args.output.resolve()),
        "archive_bytes": args.output.stat().st_size,
        "archive_sha256": sha256(args.output.read_bytes()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
