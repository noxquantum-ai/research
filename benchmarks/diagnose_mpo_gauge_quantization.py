#!/usr/bin/env python3
"""Deterministic MPO gauge-sensitivity and per-core Q4 diagnostic."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import quantize_4bit_grouped  # noqa: E402
from nox.utils import mpo_to_dense  # noqa: E402


def apply_bond_gauges(cores, gauges):
    """Apply G_i R_i and R_i^-1 G_{i+1} without changing the MPO."""
    transformed = [core.clone() for core in cores]
    if len(gauges) != len(transformed) - 1:
        raise ValueError("Expected one gauge for every internal MPO bond")
    for index, gauge in enumerate(gauges):
        bond = transformed[index].shape[-1]
        if tuple(gauge.shape) != (bond, bond):
            raise ValueError(f"Gauge {index} does not match bond dimension {bond}")
        if not torch.isfinite(gauge).all():
            raise ValueError("Gauge contains non-finite values")
        inverse = torch.linalg.inv(gauge)
        transformed[index] = torch.einsum(
            "...a,ab->...b", transformed[index], gauge,
        )
        transformed[index + 1] = torch.einsum(
            "ab,b...->a...", inverse, transformed[index + 1],
        )
    return transformed


def left_canonicalize(cores):
    canonical = [core.clone() for core in cores]
    for index in range(len(canonical) - 1):
        shape = canonical[index].shape
        matrix = canonical[index].reshape(-1, shape[-1])
        q, r = torch.linalg.qr(matrix, mode="reduced")
        canonical[index] = q.reshape(*shape[:-1], q.shape[1])
        canonical[index + 1] = torch.einsum(
            "ab,b...->a...", r, canonical[index + 1],
        )
    return canonical


def right_canonicalize(cores):
    canonical = [core.clone() for core in cores]
    for index in range(len(canonical) - 1, 0, -1):
        shape = canonical[index].shape
        matrix = canonical[index].reshape(shape[0], -1)
        q, r = torch.linalg.qr(matrix.T, mode="reduced")
        canonical[index] = q.T.reshape(q.shape[1], *shape[1:])
        canonical[index - 1] = torch.einsum(
            "...a,ab->...b", canonical[index - 1], r.T,
        )
    return canonical


def quantize_cores(cores, selected=None):
    selected = set(range(len(cores))) if selected is None else set(selected)
    return [
        quantize_4bit_grouped(core, group_size=64) if index in selected else core.clone()
        for index, core in enumerate(cores)
    ]


def relative_output_error(reference, candidate, inputs):
    reference_output = reference @ inputs
    return float(
        torch.linalg.vector_norm((candidate @ inputs) - reference_output)
        .div(torch.linalg.vector_norm(reference_output).clamp_min(1e-30))
        .item()
    )


def core_set_metrics(cores, reference_operator, calibration, validation):
    operator = mpo_to_dense(cores)
    return {
        "operator_relative_error": float(
            torch.linalg.vector_norm(operator - reference_operator)
            .div(torch.linalg.vector_norm(reference_operator).clamp_min(1e-30))
            .item()
        ),
        "calibration_relative_error": relative_output_error(
            reference_operator, operator, calibration,
        ),
        "validation_relative_error": relative_output_error(
            reference_operator, operator, validation,
        ),
    }


def orthogonal(size, generator, dtype):
    q, r = torch.linalg.qr(torch.randn(size, size, generator=generator, dtype=dtype))
    signs = torch.where(torch.diag(r) < 0, -torch.ones(size, dtype=dtype), torch.ones(size, dtype=dtype))
    return q * signs.unsqueeze(0)


def candidate_gauges(cores, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bonds = [core.shape[-1] for core in cores[:-1]]
    identity = [torch.eye(size, dtype=cores[0].dtype) for size in bonds]
    candidates = [("identity", [g.clone() for g in identity])]
    for sample in range(4):
        signs = []
        rotations = []
        for size in bonds:
            diagonal = torch.randint(0, 2, (size,), generator=generator) * 2 - 1
            signs.append(torch.diag(diagonal.to(dtype=cores[0].dtype)))
            rotations.append(orthogonal(size, generator, cores[0].dtype))
        candidates.append((f"sign_{sample}", signs))
        candidates.append((f"orthogonal_{sample}", rotations))
    for exponent in (0.5, 1.0, 2.0):
        scales = []
        for size in bonds:
            values = torch.linspace(-exponent, exponent, size, dtype=cores[0].dtype)
            scales.append(torch.diag(torch.exp(values)))
        candidates.append((f"diagonal_exp_{exponent:g}", scales))
    return candidates


def provenance():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    return {"git_commit": commit, "working_tree_clean": not dirty, "dirty_paths": dirty}


@torch.no_grad()
def run_diagnostic(seed=20261021):
    torch.manual_seed(seed)
    # Every bond is within the exact left/right unfolding rank bound, so the
    # canonical controls do not change rank merely by removing redundancy.
    ranks = (1, 4, 6, 4, 1)
    cores = [
        torch.randn(ranks[i], 2, 2, ranks[i + 1]) / math.sqrt(ranks[i] * 4)
        for i in range(4)
    ]
    reference = mpo_to_dense(cores)
    calibration = torch.randn(reference.shape[1], 96)
    validation = torch.randn(reference.shape[1], 192)
    validation.mul_(torch.linspace(0.5, 1.5, reference.shape[1]).unsqueeze(1))

    per_core = {}
    for index in range(4):
        per_core[f"core_{index}"] = core_set_metrics(
            quantize_cores(cores, {index}), reference, calibration, validation,
        )
    cumulative = {}
    for stop in range(1, 5):
        cumulative[f"cores_0_through_{stop - 1}"] = core_set_metrics(
            quantize_cores(cores, set(range(stop))),
            reference,
            calibration,
            validation,
        )

    candidates = []
    fixed_gauges = candidate_gauges(cores, seed + 1)
    for name, gauges in fixed_gauges:
        transformed = apply_bond_gauges(cores, gauges)
        invariance = core_set_metrics(
            transformed, reference, calibration, validation,
        )
        quantized = core_set_metrics(
            quantize_cores(transformed), reference, calibration, validation,
        )
        candidates.append({
            "name": name,
            "pre_quantization": invariance,
            "post_q4": quantized,
        })
    for name, transformed in (
        ("left_canonical", left_canonicalize(cores)),
        ("right_canonical", right_canonicalize(cores)),
    ):
        candidates.append({
            "name": name,
            "pre_quantization": core_set_metrics(
                transformed, reference, calibration, validation,
            ),
            "post_q4": core_set_metrics(
                quantize_cores(transformed), reference, calibration, validation,
            ),
        })

    selected = min(
        candidates, key=lambda item: item["post_q4"]["calibration_relative_error"],
    )
    identity = next(item for item in candidates if item["name"] == "identity")
    max_invariance_error = max(
        item["pre_quantization"]["operator_relative_error"] for item in candidates
    )
    q4_validation_errors = [
        item["post_q4"]["validation_relative_error"] for item in candidates
    ]
    return {
        "schema_version": 1,
        "scope": "small deterministic fixed-MPO gauge/Q4 sensitivity diagnostic",
        "seed": seed,
        "ranks": list(ranks),
        "mode_dimensions": [[2, 2]] * 4,
        "selection_rule": (
            "Choose the candidate with minimum calibration-output error; "
            "validation is measured only after selection."
        ),
        "per_core_q4": per_core,
        "cumulative_q4": cumulative,
        "gauge_candidates": candidates,
        "selected_candidate": selected,
        "identity_candidate": identity,
        "summary": {
            "maximum_pre_quantization_operator_error": max_invariance_error,
            "minimum_q4_validation_error_across_candidates": min(q4_validation_errors),
            "maximum_q4_validation_error_across_candidates": max(q4_validation_errors),
            "identity_q4_validation_error": identity["post_q4"]["validation_relative_error"],
            "selected_q4_validation_error": selected["post_q4"]["validation_relative_error"],
            "selected_improves_calibration": bool(
                selected["post_q4"]["calibration_relative_error"]
                <= identity["post_q4"]["calibration_relative_error"]
            ),
            "selected_improves_validation": bool(
                selected["post_q4"]["validation_relative_error"]
                <= identity["post_q4"]["validation_relative_error"]
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    report = run_diagnostic(args.seed)
    report["provenance"] = provenance()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    if report["summary"]["maximum_pre_quantization_operator_error"] > 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
