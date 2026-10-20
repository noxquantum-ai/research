#!/usr/bin/env python3
"""Persist periodic resource and progress telemetry for a qualification stage."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def read_progress(path):
    try:
        value = json.loads(path.read_text())
        return {
            "status": value.get("status"),
            "phase": value.get("phase"),
            "updated_at_utc": value.get("updated_at_utc"),
        }
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None


def read_gpu():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15,
        )
        rows = []
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 7:
                rows.append({
                    "index": int(fields[0]),
                    "name": fields[1],
                    "utilization_percent": float(fields[2]),
                    "memory_used_mib": float(fields[3]),
                    "memory_total_mib": float(fields[4]),
                    "temperature_c": float(fields[5]),
                    "power_draw_w": (
                        None if fields[6] in {"[N/A]", "N/A"} else float(fields[6])
                    ),
                })
        return {"status": "OK", "devices": rows}
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as error:
        return {"status": "UNAVAILABLE", "error": f"{type(error).__name__}: {error}"}


def read_memory():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[f"{key.lower()}_bytes"] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return values


def snapshot(stage, progress_path, workspace, started):
    disk = shutil.disk_usage(workspace)
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "elapsed_seconds": time.monotonic() - started,
        "progress": read_progress(progress_path),
        "gpu": read_gpu(),
        "host_memory": read_memory(),
        "disk": {
            "path": str(workspace),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "load_average": list(os.getloadavg()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    while not STOP:
        record = snapshot(args.stage, args.progress, args.workspace, started)
        line = json.dumps(record, sort_keys=True)
        with args.output.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        phase = (record["progress"] or {}).get("phase") or "not-reported"
        gpu = record["gpu"].get("devices", [])
        gpu_text = ", ".join(
            f"gpu{item['index']}={item['utilization_percent']:.0f}% "
            f"vram={item['memory_used_mib']:.0f}/{item['memory_total_mib']:.0f}MiB"
            for item in gpu
        ) or record["gpu"]["status"]
        print(
            f"[heartbeat] stage={args.stage} elapsed={record['elapsed_seconds']:.0f}s "
            f"phase={phase} {gpu_text}",
            flush=True,
        )
        deadline = time.monotonic() + args.interval_seconds
        while not STOP and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))


if __name__ == "__main__":
    main()
