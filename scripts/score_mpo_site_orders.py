#!/usr/bin/env python3
"""Weight-only screen of shared four-site MPO orders for high-fidelity runs."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compile_high_fidelity import PinnedWeights  # noqa: E402
from nox.tensor_order import IDENTITY_SITE_ORDER, tt_svd_relative_error  # noqa: E402
from nox.sequential_compiler import get_tier_config  # noqa: E402
from nox.utils import get_modes_for_projection  # noqa: E402


def parse_int_list(value, name):
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return values


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=lambda v: parse_int_list(v, "layers"), default=(0, 3))
    parser.add_argument("--projections", default="q_proj,o_proj")
    parser.add_argument("--model-id", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--model-revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    projections = tuple(name.strip() for name in args.projections.split(",") if name.strip())
    if not projections:
        parser.error("at least one projection is required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA requested but unavailable: {args.device}")

    orders = list(itertools.permutations(range(4)))
    tier = get_tier_config("high_fidelity_10x")
    per_order = {order: [] for order in orders}
    weights = PinnedWeights(args.model_id, args.model_revision)

    for layer_idx in args.layers:
        layer = weights.layer(layer_idx, args.device)
        for projection in projections:
            if projection not in layer:
                parser.error(f"unknown projection {projection!r}; choose from {sorted(layer)}")
            W = layer[projection]
            o_modes, i_modes = get_modes_for_projection(*W.shape)
            for order in orders:
                relative_error = tt_svd_relative_error(
                    W, o_modes, i_modes, tier["ranks"], order,
                )
                per_order[order].append({
                    "layer": layer_idx,
                    "projection": projection,
                    "relative_tt_svd_error_bound": relative_error,
                })
            del W
        del layer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []
    for order, measurements in per_order.items():
        values = [item["relative_tt_svd_error_bound"] for item in measurements]
        rows.append({
            "site_order": order,
            "mean_relative_tt_svd_error_bound": sum(values) / len(values),
            "max_relative_tt_svd_error_bound": max(values),
            "measurements": measurements,
        })
    rows.sort(key=lambda row: (row["mean_relative_tt_svd_error_bound"],
                               row["max_relative_tt_svd_error_bound"]))
    identity = next(row for row in rows if tuple(row["site_order"]) == IDENTITY_SITE_ORDER)
    best = rows[0]
    result = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "layers": args.layers,
        "projections": projections,
        "ranks": tier["ranks"],
        "selection_metric": "mean normalized TT-SVD truncation bound; screening only, not end-to-end quality",
        "identity_order": identity,
        "best_order": best,
        "relative_mean_improvement_over_identity": (
            0.0 if identity["mean_relative_tt_svd_error_bound"] == 0 else
            1.0 - best["mean_relative_tt_svd_error_bound"] /
            identity["mean_relative_tt_svd_error_bound"]
        ),
        "all_orders": rows,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
