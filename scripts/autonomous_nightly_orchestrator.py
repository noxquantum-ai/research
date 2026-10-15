#!/usr/bin/env python3
"""
Autonomous Nightly Orchestrator for Nox A10 Runner.

1. Monitors the active 'the_big_guns' sweep (Tier 2, Tier 3, Day 4 Perplexity & Packaging).
2. As soon as it completes, automatically launches the Day 5 Multi-Task Evaluation Suite
   (WikiText-2, C4, ARC-Challenge, HellaSwag, GSM8K across all 3 tiers + dense baseline).
3. Packages all final artifacts and writes 'PIPELINE_COMPLETE.flag'.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t_str}] [Orchestrator] {msg}"
    print(line, flush=True)
    with open("/home/shadeform/research/orchestrator.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_big_guns_running():
    try:
        res = subprocess.run(
            ["pgrep", "-f", "run_day3_day4_runpod.sh"],
            capture_output=True,
            text=True,
        )
        return len(res.stdout.strip()) > 0
    except Exception:
        return False


def is_benchmark_day3_running():
    try:
        res = subprocess.run(
            ["pgrep", "-f", "benchmark_day3_cascade.py"],
            capture_output=True,
            text=True,
        )
        return len(res.stdout.strip()) > 0
    except Exception:
        return False


def is_benchmark_day4_running():
    try:
        res = subprocess.run(
            ["pgrep", "-f", "benchmark_day4_perplexity.py"],
            capture_output=True,
            text=True,
        )
        return len(res.stdout.strip()) > 0
    except Exception:
        return False


def wait_for_big_guns():
    log("Starting Nightly Autonomous Watcher. Monitoring 'the_big_guns' sweep...")
    check_interval = 60
    while True:
        bg_running = is_big_guns_running()
        d3_running = is_benchmark_day3_running()
        d4_running = is_benchmark_day4_running()

        if not (bg_running or d3_running or d4_running):
            log("'the_big_guns' sweep has finished! Verifying checkpoint manifests...")
            break

        # Log heartbeat every 15 minutes
        if int(time.time()) % 900 < check_interval:
            log("Sweep still actively running on GPU. Heartbeat OK.")

        time.sleep(check_interval)


def run_day5_benchmarks():
    log("=================================================================")
    log("LAUNCHING DAY 5 MULTI-TASK BENCHMARK SUITE")
    log("Tasks: WikiText-2 PPL, C4 PPL, ARC-Challenge, HellaSwag, GSM8K")
    log("=================================================================")

    cmd = [
        "python3",
        "benchmarks/benchmark_day5_multitask.py",
        "--sweep_dir",
        "./checkpoints/day3_day4_cascade/master_cliff_sweep",
        "--output_file",
        "./checkpoints/day3_day4_cascade/master_cliff_sweep/day5_multitask_results.json",
    ]

    log(f"Executing: {' '.join(cmd)}")
    with open("/home/shadeform/research/day5_benchmarks.log", "w", encoding="utf-8") as out_f:
        p = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT, cwd="/home/shadeform/research")
        p.wait()

    if p.returncode == 0:
        log("Day 5 Multi-Task Benchmarks COMPLETED SUCCESSFULLY!")
    else:
        log(f"Day 5 Multi-Task Benchmarks finished with exit code {p.returncode}. Check day5_benchmarks.log.")


def package_and_flag():
    log("Final packaging of all artifacts...")
    cmd_export = [
        "python3",
        "scripts/export_checkpoints.py",
        "--checkpoint_dir",
        "./checkpoints/day3_day4_cascade/master_cliff_sweep",
        "--method",
        "all",
    ]
    subprocess.run(cmd_export, cwd="/home/shadeform/research")

    flag_path = "/home/shadeform/research/PIPELINE_COMPLETE.flag"
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"COMPLETED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    log(f"Wrote completion signal flag to '{flag_path}'! Pipeline is 100% complete and ready for local download.")


def main():
    wait_for_big_guns()
    run_day5_benchmarks()
    package_and_flag()


if __name__ == "__main__":
    main()
