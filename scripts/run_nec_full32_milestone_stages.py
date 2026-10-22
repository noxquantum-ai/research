#!/usr/bin/env python3
"""Resume the 32-layer NEC compiler with full-model checks every eight layers."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MILESTONES = (8, 16, 24, 32)


def contiguous_checkpoint_count(checkpoint_dir: Path) -> int:
    count = 0
    while (checkpoint_dir / f"compressed_layer_{count}.pt").is_file():
        count += 1
    return count


def compile_args(
    checkpoint_dir: Path,
    stop_after_layer: int,
    *,
    mpo_sweeps: int = 2,
    mpo_cg_iterations: int = 48,
    resume: bool = True,
) -> list[str]:
    command = [
        sys.executable, "-u", "scripts/compile_high_fidelity.py",
        "--output", str(checkpoint_dir),
        "--start-layer", "0", "--end-layer", "31",
        "--stop-after-layer", str(stop_after_layer),
        "--n-calib", "4096", "--n-validation", "1024", "--seq-len", "128",
        "--calibration-sampling", "uniform", "--validation-sampling", "uniform",
        "--calibration-backend", "full_matrix", "--calibration-windows-per-chunk", "4",
        "--statistics-dtype", "float64", "--calibration-activation-mode", "dense",
        "--construction-mode", "independent", "--mpo-sweeps", str(mpo_sweeps),
        "--mpo-alpha-awt", "0.5", "--mpo-quantization-schedule", "post_sweep",
        "--mpo-cg-threshold", "4096", "--mpo-cg-iterations", str(mpo_cg_iterations),
        "--representation", "mpo4", "--operator-schmidt-weight-mix", "1.0",
        "--operator-schmidt-activation-mix", "1.0", "--attention-gauge", "none",
        "--mlp-gauge", "none", "--mlp-objective", "independent",
        "--mlp-product-ridge-ratio", "0.05", "--budget-allocation", "uniform",
        "--defect-bridge-mode", "none", "--pees-rank", "0", "--pees-tokens", "4096",
        "--pees-ridge-ratio", "0.01", "--max-checkpoint-multiplier", "1.25",
        "--site-order", "0,1,2,3", "--device", "cuda:0",
    ]
    if resume:
        command.append("--resume")
    return command


def run_compile(
    checkpoint_dir: Path,
    log_path: Path,
    milestone: int,
    *,
    mpo_sweeps: int,
    mpo_cg_iterations: int,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resume = (checkpoint_dir / "model_manifest.json").is_file()
    command = compile_args(
        checkpoint_dir,
        stop_after_layer=milestone - 1,
        mpo_sweeps=mpo_sweeps,
        mpo_cg_iterations=mpo_cg_iterations,
        resume=resume,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== compile stage through milestone {milestone} ===\n")
        log.flush()
        result = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
        return_code = result.returncode
        log.write(f"\n=== compiler exit code {return_code} ===\n")
        log.flush()

    count = contiguous_checkpoint_count(checkpoint_dir)
    if count < milestone:
        raise RuntimeError(
            f"Compiler stopped at {count} contiguous checkpoints before milestone {milestone}; "
            f"inspect {log_path}"
        )
    if milestone == 32 and return_code != 0:
        raise RuntimeError(f"Final 32-layer compile failed with code {return_code}; inspect {log_path}")
    if return_code != 0:
        raise RuntimeError(f"Compiler failed before milestone {milestone} with code {return_code}")


def run_milestone_evaluation(
    checkpoint_dir: Path, output_dir: Path, milestone: int, log_dir: Path,
) -> None:
    output = output_dir / f"compiled_{milestone:02d}"
    log_path = log_dir / f"quality_milestone_{milestone:02d}.log"
    command = [
        sys.executable, "-u", "scripts/evaluate_nec_compile_milestone.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--candidate-count", str(milestone),
        "--output", str(output),
        "--eval-tokens", "4096",
        "--context-length", "2048",
        "--stride", "512",
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        (output_dir / f"compiled_{milestone:02d}.FAILED.txt").write_text(
            f"Milestone evaluator exited {result.returncode}; inspect {log_path}\n"
        )
        print(
            f"MILESTONE {milestone}: evaluation failed with code {result.returncode}; "
            "continuing the compile",
            flush=True,
        )
    else:
        print(f"MILESTONE {milestone}: quality report written to {output / 'report.json'}", flush=True)


def run_final_allocation(checkpoint_dir: Path, milestone_dir: Path, log_dir: Path) -> None:
    output = milestone_dir.parent / "final_allocation"
    report = output / "report.json"
    if report.is_file():
        print(f"FINAL ALLOCATION already recorded at {report}", flush=True)
        return
    if output.exists():
        raise FileExistsError(
            f"Final allocation output exists without a report; inspect {output} before retrying"
        )
    log_path = log_dir / "nec-full32-adaptive-allocation.log"
    command = [
        sys.executable, "-u", "scripts/search_nec_layer_allocation.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--output", str(output),
        "--optimize-storage", "--max-dense-layers", "4", "--max-ppl", "10",
        "--heldout-split", "test", "--heldout-start-token", "0",
        "--heldout-tokens", "0",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUNNING final mixed NEC/INT8/FP16 allocation with full test split", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(
            f"Final adaptive allocation failed with code {result.returncode}; inspect {log_path}"
        )
    print(f"FINAL ALLOCATION report written to {report}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--mpo-sweeps", type=int, default=2)
    parser.add_argument("--mpo-cg-iterations", type=int, default=48)
    parser.add_argument(
        "--allow-fresh", action="store_true",
        help="Start a new checkpoint directory instead of requiring the four saved layer artifacts.",
    )
    args = parser.parse_args()
    initial_count = contiguous_checkpoint_count(args.checkpoint_dir)
    has_manifest = (args.checkpoint_dir / "model_manifest.json").is_file()
    if not has_manifest and not args.allow_fresh:
        raise FileNotFoundError("The resumable NEC checkpoint manifest is missing")
    if not has_manifest and initial_count:
        raise ValueError("A fresh checkpoint directory must not contain partial layer artifacts")
    if has_manifest and initial_count < 4:
        raise ValueError(f"Expected the four existing layer checkpoints; found {initial_count}")
    if args.mpo_sweeps < 0 or args.mpo_cg_iterations <= 0:
        raise ValueError("MPO sweeps must be nonnegative and CG iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    for milestone in MILESTONES:
        if contiguous_checkpoint_count(args.checkpoint_dir) < milestone:
            print(f"COMPILING through checkpoint {milestone}/32", flush=True)
            run_compile(
                args.checkpoint_dir,
                args.log_dir / "layer0-31-staged-compile.log",
                milestone,
                mpo_sweeps=args.mpo_sweeps,
                mpo_cg_iterations=args.mpo_cg_iterations,
            )
        print(f"EVALUATING full model at {milestone}/32 compiled candidates", flush=True)
        run_milestone_evaluation(
            args.checkpoint_dir, args.output_dir, milestone, args.log_dir,
        )

    run_final_allocation(args.checkpoint_dir, args.output_dir, args.log_dir)
    print("COMPLETE: 8, 16, 24, and 32 candidate checkpoints evaluated; final allocation selected", flush=True)


if __name__ == "__main__":
    main()
