#!/usr/bin/env python3
"""Push one private Kaggle compile stage or the final evaluation kernel."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_OWNER = "toheebogunade"
DEFAULT_SOURCE = f"{DEFAULT_OWNER}/nec-high-fidelity-source-20261017"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile", "evaluate"), default="compile")
    parser.add_argument("--owner", default=DEFAULT_OWNER)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--previous-kernel", default=None)
    args = parser.parse_args()
    if args.stage < 0:
        raise ValueError("Stage must be nonnegative")
    if args.mode == "evaluate" and not args.previous_kernel:
        raise ValueError("Evaluation requires --previous-kernel with all 32 compiled layers")
    if args.mode == "compile" and args.stage > 0 and not args.previous_kernel:
        raise ValueError("Follow-up compile stages require the preceding successful kernel")

    with tempfile.TemporaryDirectory(prefix="nec-kaggle-kernel-") as temporary:
        folder = Path(temporary)
        if args.mode == "compile":
            slug = f"nec-high-fidelity-compile-stage-{args.stage:02d}"
            title = f"NEC High Fidelity Compile Stage {args.stage:02d}"
            code_file = "driver.py"
            shutil.copy2(HERE / "driver.py", folder / code_file)
        else:
            slug = "nec-high-fidelity-independent-evaluation"
            title = "NEC High Fidelity Independent Evaluation"
            code_file = "evaluation_driver.py"
            shutil.copy2(HERE / code_file, folder / code_file)

        metadata = {
            "id": f"{args.owner}/{slug}",
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [args.source_dataset],
            "competition_sources": [],
            "kernel_sources": [args.previous_kernel] if args.previous_kernel else [],
            "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        command = [
            "kaggle", "kernels", "push", "--path", str(folder),
            "--timeout", "43200", "--accelerator", "NvidiaTeslaT4",
        ]
        print(f"Pushing private Kaggle kernel {metadata['id']}", flush=True)
        subprocess.run(command, check=True)
        print(json.dumps({
            "kernel": metadata["id"],
            "mode": args.mode,
            "stage": args.stage if args.mode == "compile" else None,
            "attached_previous_kernel": args.previous_kernel,
            "attached_source_dataset": args.source_dataset,
            "timeout_seconds": 43200,
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
