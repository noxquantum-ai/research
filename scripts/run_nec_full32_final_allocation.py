#!/usr/bin/env python3
"""Wait for the staged full-32 run, then select and validate its final allocation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_count(checkpoint_dir: Path) -> int:
    count = 0
    while (checkpoint_dir / f"compressed_layer_{count}.pt").is_file():
        count += 1
    return count


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--runner-pid-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")
    report = args.output / "report.json"
    if report.is_file():
        print(f"Final allocation already exists: {report}", flush=True)
        return
    if args.output.exists():
        raise FileExistsError(
            f"Final allocation output exists without report.json: {args.output}"
        )

    pid = int(args.runner_pid_file.read_text().strip())
    while process_exists(pid):
        count = checkpoint_count(args.checkpoint_dir)
        print(f"WAIT staged_runner_pid={pid} contiguous_checkpoints={count}/32", flush=True)
        time.sleep(args.poll_seconds)
    count = checkpoint_count(args.checkpoint_dir)
    if count != 32:
        raise RuntimeError(
            f"Staged runner exited with only {count}/32 checkpoints; refusing final allocation"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", "scripts/search_nec_layer_allocation.py",
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--output", str(args.output),
        "--optimize-storage", "--max-dense-layers", "4", "--max-ppl", "10",
        "--heldout-split", "test", "--heldout-start-token", "0",
        "--heldout-tokens", "0",
    ]
    with args.log.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"Final allocation exited with {result.returncode}; inspect {args.log}"
        )
    print(f"COMPLETE final_report={report}", flush=True)


if __name__ == "__main__":
    main()
