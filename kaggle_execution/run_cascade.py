"""Kaggle Runner for NoxQuantum Day 3 Cascade Benchmark.

Executes 4-Layer Cascade (Layers 15 -> 18) under both Extreme (77x) and Production (18x) tiers
on free NVIDIA T4 GPU compute with zero cloud costs.
"""

import os
import subprocess
import sys


def run_cmd(cmd):
    print(f"\n[Kaggle Runner] Executing: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, text=True)
    if res.returncode != 0:
        print(f"[Kaggle Runner] Warning: Command returned non-zero exit code {res.returncode}", flush=True)
    return res.returncode


def main():
    print("[Kaggle Runner] Starting NoxQuantum Day 3 Multi-Layer Cascade Benchmark...", flush=True)

    # 1. Clone or pull latest research repo
    if not os.path.exists("research"):
        run_cmd("git clone https://github.com/noxquantum-ai/research.git")
    else:
        run_cmd("cd research && git pull origin main")

    work_dir = "research" if os.path.exists("research") else "."

    # 2. Benchmark Tier 1: Extreme Stress-Test (77x Compression)
    print("\n" + "=" * 90, flush=True)
    print(" EXECUTING TIER 1: EXTREME STRESS-TEST (77x COMPRESSION, ~5.1 MB/LAYER)", flush=True)
    print("=" * 90, flush=True)
    run_cmd(
        f"python3 {work_dir}/benchmarks/benchmark_day3_cascade.py "
        f"--start_layer 15 --end_layer 18 --tier extreme_77x"
    )

    # 3. Benchmark Tier 2: Production CompactifAI-Killer (18x Compression)
    print("\n" + "=" * 90, flush=True)
    print(" EXECUTING TIER 2: PRODUCTION COMPACTIFAI-KILLER (18x COMPRESSION, ~20 MB/LAYER)", flush=True)
    print("=" * 90, flush=True)
    run_cmd(
        f"python3 {work_dir}/benchmarks/benchmark_day3_cascade.py "
        f"--start_layer 15 --end_layer 18 --tier production_18x"
    )

    print("\n[Kaggle Runner] All Day 3 Cascade benchmarks completed successfully.", flush=True)


if __name__ == "__main__":
    main()
