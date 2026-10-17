#!/usr/bin/env python3
"""Record the hardware exposed by the authenticated Kaggle GPU runtime."""

from __future__ import annotations

import json
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    disk = shutil.disk_usage("/kaggle/working")
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "provider_limits": {
            "maximum_kernel_timeout_seconds": 43200,
            "persistent_outputs_attachable_to_later_kernels": True,
            "kernel_output_visibility": "private",
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {},
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    for package in (
        "transformers", "huggingface-hub", "safetensors", "accelerate",
        "pyarrow", "datasets", "lm_eval", "pytest",
    ):
        try:
            record["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record["packages"][package] = None
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        record["gpus"].append({
            "index": index,
            "name": props.name,
            "total_vram_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
        })
    if torch.cuda.is_available():
        record["nvidia_smi"] = command(
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ).splitlines()
    output = Path("/kaggle/working/preflight.json")
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
