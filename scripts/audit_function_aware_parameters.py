#!/usr/bin/env python3
"""Recompute stored-scalar counts from the committed Llama-2-7B layouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.sequential_compiler import get_tier_config
from nox.spectra import llama_projection_modes, tt_parameter_count


HIDDEN = 4096
INTERMEDIATE = 11008
VOCAB = 32000
LAYERS = 32

PROJECTIONS = {
    "q_proj": (HIDDEN, HIDDEN),
    "k_proj": (HIDDEN, HIDDEN),
    "v_proj": (HIDDEN, HIDDEN),
    "o_proj": (HIDDEN, HIDDEN),
    "gate_proj": (INTERMEDIATE, HIDDEN),
    "up_proj": (INTERMEDIATE, HIDDEN),
    "down_proj": (HIDDEN, INTERMEDIATE),
}


def dense_model_coefficients():
    layer_projection = sum(d_out * d_in for d_out, d_in in PROJECTIONS.values())
    layer_norms = 2 * HIDDEN
    final_norm = HIDDEN
    embedding = VOCAB * HIDDEN
    lm_head = VOCAB * HIDDEN
    return {
        "projection_coefficients_per_layer": layer_projection,
        "norm_coefficients_per_layer": layer_norms,
        "final_norm_coefficients": final_norm,
        "embedding_coefficients": embedding,
        "lm_head_coefficients": lm_head,
        "layers": LAYERS,
        "total_coefficients": LAYERS * (layer_projection + layer_norms) + final_norm + embedding + lm_head,
    }


def nec_layer_coefficients(config):
    core = 0
    bridges = 0
    sparse_values = 0
    input_scales = 0
    details = {}
    for name, (d_out, d_in) in PROJECTIONS.items():
        o_modes, i_modes = llama_projection_modes(d_out, d_in)
        physical = tuple(o * i for o, i in zip(o_modes, i_modes))
        core_count = tt_parameter_count(physical, tuple(config["ranks"]))
        bridge_count = (d_out + d_in) * int(config["r_bridge"])
        sparse_count = int(config["n_outliers"])
        scale_count = d_in
        details[name] = {
            "tt_core_coefficients": core_count,
            "bridge_coefficients": bridge_count,
            "sparse_value_coefficients": sparse_count,
            "input_scale_coefficients": scale_count,
            "total_coefficients": core_count + bridge_count + sparse_count + scale_count,
        }
        core += core_count
        bridges += bridge_count
        sparse_values += sparse_count
        input_scales += scale_count
    return {
        "projection_details": details,
        "tt_core_coefficients": core,
        "bridge_coefficients": bridges,
        "sparse_value_coefficients": sparse_values,
        "input_scale_coefficients": input_scales,
        "layer_projection_coefficients": core + bridges + sparse_values + input_scales,
        "layer_norm_coefficients": 2 * HIDDEN,
        "layer_total_coefficients": core + bridges + sparse_values + input_scales + 2 * HIDDEN,
    }


def tier_counts(tier_name):
    config = get_tier_config(tier_name)
    if tier_name == "production_18x":
        boundary = get_tier_config("high_fidelity_10x")
    elif tier_name == "extreme_77x":
        boundary = get_tier_config("production_18x")
    else:
        boundary = config
    interior_counts = nec_layer_coefficients(config)
    boundary_counts = nec_layer_coefficients(boundary)
    dense = dense_model_coefficients()
    compressed = (
        28 * interior_counts["layer_total_coefficients"]
        + 4 * boundary_counts["layer_total_coefficients"]
        + dense["embedding_coefficients"]
        + dense["lm_head_coefficients"]
        + dense["final_norm_coefficients"]
    )
    return {
        "tier": tier_name,
        "configuration": config,
        "boundary_configuration": boundary,
        "interior_layer": interior_counts,
        "boundary_layer": boundary_counts,
        "whole_model_coefficients": compressed,
        "dense_whole_model_coefficients": dense["total_coefficients"],
        "parameter_compression_ratio": dense["total_coefficients"] / compressed,
        "parameter_reduction_fraction": 1.0 - compressed / dense["total_coefficients"],
        "assumptions": [
            "All FP16/quantized TT, bridge, sparse-value, and diagonal-scale entries count as one numerical model coefficient.",
            "Sparse coordinate indices and container metadata are not numerical coefficients; they remain part of checkpoint bytes.",
            "Counts are configuration-derived; dynamic rank reductions, threshold ties, and unavailable historical checkpoint tensors are not observed here.",
        ],
    }


def grouped_count(rank=2612):
    dense = dense_model_coefficients()
    dense_mlp_per_layer = 3 * HIDDEN * INTERMEDIATE
    compressed_mlp_per_layer = 3 * int(rank) * (HIDDEN + INTERMEDIATE)
    compressed = dense["total_coefficients"] - LAYERS * (
        dense_mlp_per_layer - compressed_mlp_per_layer
    )
    return {
        "rank": int(rank),
        "dense_mlp_coefficients_per_layer": dense_mlp_per_layer,
        "grouped_mlp_coefficients_per_layer": compressed_mlp_per_layer,
        "mlp_retained_fraction": compressed_mlp_per_layer / dense_mlp_per_layer,
        "whole_model_dense_coefficients": dense["total_coefficients"],
        "whole_model_compressed_coefficients": compressed,
        "whole_model_parameter_compression_ratio": dense["total_coefficients"] / compressed,
        "whole_model_parameter_reduction_fraction": 1.0 - compressed / dense["total_coefficients"],
        "assumption": "Attention, embeddings, output head, and norms remain dense; factor coefficients are counted once as stored.",
    }


def build_report():
    return {
        "model_id": "NousResearch/Llama-2-7b-hf",
        "revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
        "dense": dense_model_coefficients(),
        "legacy_nec_config_derived": {
            name: tier_counts(name)
            for name in ("high_fidelity_10x", "production_18x", "extreme_77x")
        },
        "grouped_swiglu_0875": grouped_count(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment_artifacts/function_aware_tn/derived_parameter_counts.json"),
    )
    args = parser.parse_args()
    result = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "dense_coefficients": result["dense"]["total_coefficients"],
        "legacy_tier_ratios": {
            key: value["parameter_compression_ratio"]
            for key, value in result["legacy_nec_config_derived"].items()
        },
        "grouped_0875": result["grouped_swiglu_0875"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
