#!/usr/bin/env python3
"""Run a reproducible CPU-only synthetic matched-budget functional comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.function_aware_tn import (  # noqa: E402
    activation_weighted_svd_factors,
    apply_factorized_swiglu,
    apply_selected_swiglu,
    build_factorized_swiglu,
    build_selected_swiglu,
    budgeted_swiglu_rank,
    ordinary_svd_factors,
    rank_allocations,
    swiglu_factor_coefficient_count,
)


def metrics(reference, candidate):
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(1e-30)
    cand_norm = torch.linalg.vector_norm(cand)
    return {
        "relative_error": float(torch.linalg.vector_norm(delta) / ref_norm),
        "cosine_similarity": float(F.cosine_similarity(ref.T, cand.T, dim=1).mean()),
        "norm_ratio": float(cand_norm / ref_norm),
    }


def direct_factor_artifact(gate, up, down, ranks, method, inputs, ridge_ratio=0.01):
    if method == "ordinary_svd":
        gate_factors = ordinary_svd_factors(gate, ranks[0], seed=101)
        up_factors = ordinary_svd_factors(up, ranks[1], seed=102)
        hidden = F.silu(gate @ inputs) * (up @ inputs)
        down_factors = ordinary_svd_factors(down, ranks[2], seed=103)
    elif method == "activation_weighted_svd":
        gate_factors = activation_weighted_svd_factors(
            gate, inputs, ranks[0], ridge_ratio=ridge_ratio, seed=101,
        )
        up_factors = activation_weighted_svd_factors(
            up, inputs, ranks[1], ridge_ratio=ridge_ratio, seed=102,
        )
        hidden = F.silu(gate @ inputs) * (up @ inputs)
        down_factors = activation_weighted_svd_factors(
            down, hidden, ranks[2], ridge_ratio=ridge_ratio, seed=103,
        )
    else:
        raise ValueError(method)
    return {
        "gate_left": gate_factors[0], "gate_right": gate_factors[1],
        "up_left": up_factors[0], "up_right": up_factors[1],
        "down_left": down_factors[0], "down_right": down_factors[1],
        "coefficient_count": swiglu_factor_coefficient_count(gate.shape[1], gate.shape[0], ranks),
        "construction": method,
        "ranks": list(ranks),
    }


def correlated_inputs(d_model, count, generator, mixing):
    latent = torch.randn(10, count, generator=generator)
    noise = torch.randn(d_model, count, generator=generator) * 0.15
    return mixing @ latent + noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment_artifacts/function_aware_tn/synthetic_cpu_results.json"),
    )
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    d_model, intermediate = 32, 48
    gate = torch.randn(intermediate, d_model, generator=generator) / d_model**0.5
    up = torch.randn(intermediate, d_model, generator=generator) / d_model**0.5
    down = torch.randn(d_model, intermediate, generator=generator) / intermediate**0.5
    mixing = torch.randn(d_model, 10, generator=generator) / 3
    fit_x = correlated_inputs(d_model, 192, generator, mixing)
    selection_x = correlated_inputs(d_model, 64, generator, mixing)
    heldout_x = correlated_inputs(d_model, 256, generator, mixing)
    dense = lambda x: down @ (F.silu(gate @ x) * (up @ x))

    results = {
        "experiment": "synthetic_cpu_function_aware_swiglu_matched_budget",
        "status": "COMPLETE",
        "model_identity": "synthetic; not Llama-2-7B evidence",
        "code_revision": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "seed": args.seed,
        "device": "cpu",
        "dtype": "float32",
        "dimensions": {"d_model": d_model, "intermediate": intermediate},
        "split_counts": {
            "fit": fit_x.shape[1],
            "internal_selection": selection_x.shape[1],
            "heldout": heldout_x.shape[1],
        },
        "budgets": {},
    }
    for fraction in (0.75, 0.50, 0.25):
        budget = budgeted_swiglu_rank(d_model, intermediate, fraction)
        ranks = budget["ranks"]
        results["budgets"][str(fraction)] = {
            **budget,
            "candidates": {},
        }
        trials = {name: [] for name in (
            "ordinary_svd", "activation_weighted_svd", "function_aware", "modegpt_inspired",
        )}
        for allocation, trial_ranks in rank_allocations(sum(ranks)).items():
            for ridge_ratio in (0.0, 0.01, 0.1):
                trials["ordinary_svd"].append((
                    allocation, ridge_ratio,
                    direct_factor_artifact(gate, up, down, trial_ranks, "ordinary_svd", fit_x),
                ))
                trials["activation_weighted_svd"].append((
                    allocation, ridge_ratio,
                    direct_factor_artifact(
                        gate, up, down, trial_ranks, "activation_weighted_svd", fit_x,
                        ridge_ratio=ridge_ratio,
                    ),
                ))
                trials["function_aware"].append((
                    allocation, ridge_ratio,
                    build_factorized_swiglu(
                        gate, up, down, fit_x, trial_ranks, construction="function_aware",
                        ridge_ratio=ridge_ratio, storage_dtype=torch.float32, seed=200,
                    ),
                ))
                trials["modegpt_inspired"].append((
                    allocation, ridge_ratio,
                    build_selected_swiglu(
                        gate, up, down, fit_x, budget["coefficient_budget"],
                        ridge_ratio=max(1e-4, ridge_ratio), storage_dtype=torch.float32,
                    ),
                ))

        for name, candidates in trials.items():
            ranked = []
            for allocation, ridge_ratio, artifact in candidates:
                apply_candidate = (
                    apply_selected_swiglu if name == "modegpt_inspired" else apply_factorized_swiglu
                )
                selection_prediction = apply_candidate(artifact, selection_x)
                selection_error = metrics(dense(selection_x), selection_prediction)["relative_error"]
                ranked.append((selection_error, allocation, ridge_ratio, artifact))
            ranked.sort(key=lambda item: item[0])
            selected_error, selected_allocation, selected_ridge, artifact = ranked[0]
            entry = {
                "ranks": list(artifact["ranks"]) if "ranks" in artifact else None,
                "selected_neuron_count": artifact.get("neuron_count"),
                "coefficient_count": artifact["coefficient_count"],
                "dense_residual": False,
                "uses_gradients": False,
                "selected_allocation": selected_allocation,
                "selected_ridge_ratio": selected_ridge,
                "selected_internal_selection_error": selected_error,
                "selection_trials": [
                    {
                        "allocation": allocation,
                        "ridge_ratio": ridge_ratio,
                        "relative_error": error,
                    }
                    for error, allocation, ridge_ratio, _ in ranked
                ],
                "metrics": {},
            }
            for split_name, inputs in (
                ("fit", fit_x),
                ("internal_selection", selection_x),
                ("heldout", heldout_x),
            ):
                expected = dense(inputs)
                predicted = apply_candidate(artifact, inputs)
                entry["metrics"][split_name] = {
                    "mlp_output": metrics(expected, predicted),
                    "residual_block_output": metrics(inputs + expected, inputs + predicted),
                }
            results["budgets"][str(fraction)]["candidates"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(results, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "status": results["status"],
        "heldout_errors": {
            fraction: {
                method: record["metrics"]["heldout"]["mlp_output"]["relative_error"]
                for method, record in budget["candidates"].items()
            }
            for fraction, budget in results["budgets"].items()
        },
        "coefficients": {
            fraction: {
                method: record["coefficient_count"]
                for method, record in budget["candidates"].items()
            }
            for fraction, budget in results["budgets"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
