#!/usr/bin/env python3
"""Deterministic CPU audit of direct-SVD versus streaming-statistics NEC.

This diagnostic deliberately uses an ill-conditioned calibration matrix.  It
separates four effects that the real A40 qualification combined:

1. the Gram identity used to recover singular coordinates;
2. unquantized bridge and DMRG differences;
3. historical in-sweep Q4 amplification; and
4. Q4 applied once after a completed, gauge-fixed sweep.

It is a numerical control, not evidence about Llama model quality.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from nox.calibration import accumulate_calibration_statistics
from nox.compiler import (
    _left_singular_system_from_gram,
    nox_hybrid_dmrg_decompose,
    nox_hybrid_dmrg_decompose_from_statistics,
)


ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_SOURCES = (
    "benchmarks/diagnose_streaming_numerics.py",
    "nox/calibration.py",
    "nox/compiler.py",
)


def provenance():
    files = {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in NUMERICAL_SOURCES
    }
    digest = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "working_tree_clean": not dirty,
        "dirty_paths": dirty,
        "numerical_source_sha256": digest,
        "files": files,
    }


def relative_difference(actual, reference):
    return float(
        torch.linalg.vector_norm(actual - reference)
        .div(torch.linalg.vector_norm(reference).clamp_min(1e-30))
        .item()
    )


def functional_disagreement(actual, reference, inputs):
    return relative_difference(actual @ inputs, reference @ inputs)


def reconstruction_error(operator, dense_weight, inputs):
    return relative_difference(operator @ inputs, dense_weight @ inputs)


def run_compiler(
    weight,
    inputs,
    statistics,
    *,
    backend,
    quantize,
    schedule,
    cg_iterations,
):
    stages = {}
    solvers = []

    def record_stage(name, value):
        stages[name] = value.detach().clone()

    common = {
        "W": weight,
        "o_modes": (2, 2, 2, 2),
        "i_modes": (2, 2, 2, 2),
        "ranks": (1, 4, 8, 4, 1),
        "r_bridge": 4,
        "n_outliers": 12,
        "n_sweeps": 2,
        "prior_gamma": 0.01,
        "alpha_awt": 0.5,
        "chunk_size": 7,
        # Force the two interior sites through the same six-iteration
        # matrix-free CG branch used by real High-Fidelity Llama projections.
        "cg_threshold": 32,
        "cg_iterations": cg_iterations,
        "return_artifacts": True,
        "verbose": False,
        "quantize_components": quantize,
        "quantization_schedule": schedule,
        "diagonal_prior_strength": 0.0,
        "diagnostic_callback": record_stage,
        "solver_diagnostic_callback": solvers.append,
    }
    if backend == "direct_svd":
        result = nox_hybrid_dmrg_decompose(
            X_calib=inputs,
            Y_dense=weight @ inputs,
            **common,
        )
    else:
        result = nox_hybrid_dmrg_decompose_from_statistics(
            calibration_statistics=statistics,
            **common,
        )
    stages["final"] = result[0].detach().clone()
    return {"stages": stages, "solvers": solvers}


def compare_pair(reference, streamed, weight, calibration, validation):
    stages = {}
    for stage in reference["stages"]:
        reference_weight = reference["stages"][stage]
        streamed_weight = streamed["stages"][stage]
        stages[stage] = {
            "relative_weight_difference": relative_difference(
                streamed_weight, reference_weight,
            ),
            "calibration_functional_disagreement": functional_disagreement(
                streamed_weight, reference_weight, calibration,
            ),
            "validation_functional_disagreement": functional_disagreement(
                streamed_weight, reference_weight, validation,
            ),
            "reference_calibration_error": reconstruction_error(
                reference_weight, weight, calibration,
            ),
            "streaming_calibration_error": reconstruction_error(
                streamed_weight, weight, calibration,
            ),
            "reference_validation_error": reconstruction_error(
                reference_weight, weight, validation,
            ),
            "streaming_validation_error": reconstruction_error(
                streamed_weight, weight, validation,
            ),
        }
    return {
        "stages": stages,
        "reference_solver_diagnostics": reference["solvers"],
        "streaming_solver_diagnostics": streamed["solvers"],
    }


def singular_identity(inputs, targets):
    direct_u, direct_s, direct_vh = torch.linalg.svd(
        inputs.double(), full_matrices=False,
    )
    gram_u, gram_s, diagnostics = _left_singular_system_from_gram(
        inputs.double() @ inputs.double().T,
        inputs.shape[1],
        torch.float64,
    )
    signs = torch.where(
        torch.sum(direct_u * gram_u, dim=0) < 0,
        -torch.ones_like(gram_s),
        torch.ones_like(gram_s),
    )
    gram_u = gram_u * signs.unsqueeze(0)
    gram_yv = (
        (targets.double() @ inputs.double().T) @ gram_u
    ) / gram_s.unsqueeze(0)
    direct_yv = targets.double() @ direct_vh.T
    return {
        "singular_value_relative_difference": relative_difference(gram_s, direct_s),
        "left_singular_vector_relative_difference_after_sign_alignment": (
            relative_difference(gram_u, direct_u)
        ),
        "projected_target_relative_difference": relative_difference(
            gram_yv, direct_yv,
        ),
        "gram_diagnostics": diagnostics,
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(1234)
    dimension = 16
    left, _ = torch.linalg.qr(torch.randn(
        dimension, dimension, dtype=torch.float64,
    ))
    right, _ = torch.linalg.qr(torch.randn(
        dimension, dimension, dtype=torch.float64,
    ))
    singular_values = torch.logspace(0, -7, dimension, dtype=torch.float64)
    calibration = (left @ torch.diag(singular_values) @ right.T).float()
    weight = torch.randn(dimension, dimension)
    validation = torch.randn(dimension, 64)
    targets = weight @ calibration
    statistics = accumulate_calibration_statistics(
        calibration,
        targets,
        chunk_size=5,
        accumulation_dtype=torch.float64,
    )

    experiments = {}
    for name, quantize, schedule, cg_iterations in (
        ("unquantized_6cg", False, "in_sweep", 6),
        ("historical_in_sweep_q4_6cg", True, "in_sweep", 6),
        ("post_sweep_q4_6cg_control", True, "post_sweep", 6),
        ("post_sweep_q4_48cg_control", True, "post_sweep", 48),
    ):
        reference = run_compiler(
            weight,
            calibration,
            statistics,
            backend="direct_svd",
            quantize=quantize,
            schedule=schedule,
            cg_iterations=cg_iterations,
        )
        streamed = run_compiler(
            weight,
            calibration,
            statistics,
            backend="streaming_statistics",
            quantize=quantize,
            schedule=schedule,
            cg_iterations=cg_iterations,
        )
        experiments[name] = compare_pair(
            reference, streamed, weight, calibration, validation,
        )

    unquantized = experiments["unquantized_6cg"]["stages"]["final"]
    historical = experiments["historical_in_sweep_q4_6cg"]["stages"]["final"]
    post_sweep_48 = experiments["post_sweep_q4_48cg_control"]["stages"]["final"]
    post_sweep_48_residual = max(
        diagnostic["relative_normal_residual"]
        for side in ("reference_solver_diagnostics", "streaming_solver_diagnostics")
        for diagnostic in experiments["post_sweep_q4_48cg_control"][side]
    )
    report = {
        "schema_version": 1,
        "scope": "deterministic CPU numerical control; not a model-quality benchmark",
        "provenance": provenance(),
        "seed": 1234,
        "dimension": dimension,
        "calibration_samples": int(calibration.shape[1]),
        "validation_samples": int(validation.shape[1]),
        "prescribed_condition_number": 1e7,
        "torch_version": torch.__version__,
        "singular_identity": singular_identity(calibration, targets),
        "experiments": experiments,
        "acceptance": {
            "unquantized_functional_disagreement_below_1e-4": (
                unquantized["calibration_functional_disagreement"] < 1e-4
            ),
            "historical_q4_exhibits_amplification_above_1e-2": (
                historical["calibration_functional_disagreement"] > 1e-2
            ),
            "post_sweep_48cg_validation_disagreement_below_5e-3": (
                post_sweep_48["validation_functional_disagreement"] < 5e-3
            ),
            "post_sweep_48cg_normal_residual_below_1e-3": (
                post_sweep_48_residual < 1e-3
            ),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if not all(report["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
