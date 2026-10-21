#!/usr/bin/env python3
"""Diagnose NEC's direct-SVD and Gram-derived bridges on real activations.

This is deliberately narrower than projection qualification: it reconstructs
only the sparse shield and rank-128 bridge, then compares the two mathematical
paths before and after Q4.  It does not run TT-SVD or DMRG.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_day3_cascade import build_real_token_calibration_data  # noqa: E402
from benchmarks.qualify_streaming_projection import (  # noqa: E402
    LAYER,
    PROJECTION,
    accumulate_down_covariance,
    down_projection_input,
    materialize_down_inputs,
)
from nox.compiler import quantize_4bit_grouped  # noqa: E402
from nox.evaluation import MODEL_ID, MODEL_REVISION, validate_llama2_config  # noqa: E402
from scripts.compile_high_fidelity import PinnedWeights, atomic_json_save  # noqa: E402


def relative_error(actual, reference):
    return float(
        torch.linalg.vector_norm((actual - reference).double())
        .div(torch.linalg.vector_norm(reference.double()).clamp_min(1e-30))
        .item()
    )


@torch.no_grad()
def functional_difference(left, right, inputs, chunk_size=512):
    difference_sq = 0.0
    reference_sq = 0.0
    for start in range(0, inputs.shape[1], chunk_size):
        chunk = inputs[:, start : start + chunk_size]
        reference = left @ chunk
        difference = (right @ chunk) - reference
        reference_sq += float(reference.double().square().sum().item())
        difference_sq += float(difference.double().square().sum().item())
    return math.sqrt(difference_sq) / max(math.sqrt(reference_sq), 1e-30)


@torch.no_grad()
def construct_bridge(residual, u_x, singular_values, projected_target, rank):
    u_y, _, _ = torch.linalg.svd(projected_target, full_matrices=False)
    u_r = u_y[:, :rank]
    regularized_inverse = singular_values / (
        singular_values.square() + (0.02 * singular_values[0]).square()
    )
    b_raw = (
        (u_r.T @ projected_target)
        * regularized_inverse.unsqueeze(0)
        @ u_x.T
    )
    column_norms = torch.sqrt(
        torch.linalg.vector_norm(u_r, dim=0)
        / (torch.linalg.vector_norm(b_raw, dim=1) + 1e-8)
    )
    a = u_r / column_norms.unsqueeze(0)
    b = b_raw * column_norms.unsqueeze(1)
    a_q = quantize_4bit_grouped(a, group_size=64)
    b_q = quantize_4bit_grouped(b, group_size=64)
    return a, b, a @ b, a_q @ b_q


@torch.no_grad()
def canonicalize_low_rank_factors(a, b):
    """Return balanced SVD-gauge factors without forming the dense product."""
    q_a, r_a = torch.linalg.qr(a, mode="reduced")
    q_b, r_b = torch.linalg.qr(b.T, mode="reduced")
    u, singular_values, vh = torch.linalg.svd(r_a @ r_b.T, full_matrices=False)
    root = torch.sqrt(singular_values.clamp_min(0))
    a_canonical = (q_a @ u) * root.unsqueeze(0)
    b_canonical = root.unsqueeze(1) * (vh @ q_b.T)
    return a_canonical, b_canonical


@torch.no_grad()
def anchor_low_rank_factors(a, b, seed=20261021):
    """Fix the factor gauge against a deterministic ambient input sketch.

    For C=AB and fixed P, this returns (CP, (BP)^-1 B).  Under any invertible
    gauge change A'=AQ, B'=Q^-1B, both returned factors are unchanged in exact
    arithmetic.  The condition number is reported because an unlucky sketch
    must not be used silently.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sketch = torch.randn(
        b.shape[1], b.shape[0], generator=generator, dtype=torch.float32,
    ).div_(math.sqrt(b.shape[1])).to(device=b.device, dtype=b.dtype)
    transform = b @ sketch
    condition = float(torch.linalg.cond(transform.float()).item())
    a_anchored = a @ transform
    b_anchored = torch.linalg.solve(transform, b)
    column_norms = torch.sqrt(
        torch.linalg.vector_norm(a_anchored, dim=0)
        / (torch.linalg.vector_norm(b_anchored, dim=1) + 1e-8)
    )
    return (
        a_anchored / column_norms.unsqueeze(0),
        b_anchored * column_norms.unsqueeze(1),
        condition,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-calib", type=int, default=4096)
    parser.add_argument("--n-validation", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--windows-per-chunk", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    validate_llama2_config(config)
    pinned = PinnedWeights(MODEL_ID, MODEL_REVISION)
    embedding = pinned.tensor("model.embed_tokens.weight")
    train_hidden, validation_hidden, input_data = build_real_token_calibration_data(
        embedding,
        MODEL_ID,
        args.n_calib,
        args.n_validation,
        args.seq_len,
        revision=MODEL_REVISION,
        calibration_sampling="uniform",
        validation_sampling="uniform",
    )
    del embedding
    weights = pinned.layer(LAYER, device)
    del pinned
    forward_options = {
        "seq_len": args.seq_len,
        "n_heads": config.num_attention_heads,
        "n_kv_heads": config.num_key_value_heads,
        "rms_norm_eps": config.rms_norm_eps,
    }
    x = materialize_down_inputs(
        weights,
        train_hidden,
        windows_per_chunk=args.windows_per_chunk,
        forward_options=forward_options,
    )
    x_validation = materialize_down_inputs(
        weights,
        validation_hidden,
        windows_per_chunk=args.windows_per_chunk,
        forward_options=forward_options,
    )
    covariance = accumulate_down_covariance(
        weights,
        train_hidden,
        windows_per_chunk=args.windows_per_chunk,
        forward_options=forward_options,
        accumulation_dtype=torch.float64,
    ).xx

    sparse_payload = torch.load(
        args.reference_output / "stage_weights" / "sparse.pt",
        map_location="cpu",
        weights_only=False,
    )
    sparse = sparse_payload["weight"].to(device=device, dtype=weights[PROJECTION].dtype)
    residual = weights[PROJECTION] - sparse

    print("BRIDGE_DIAGNOSTIC_PHASE=direct_svd", flush=True)
    u_direct, s_direct, vh_direct = torch.linalg.svd(x, full_matrices=False)
    print("BRIDGE_DIAGNOSTIC_PHASE=gram_eigh", flush=True)
    eigenvalues, u_gram = torch.linalg.eigh((covariance + covariance.T) * 0.5)
    eigenvalues = eigenvalues.flip(0)[: x.shape[1]]
    u_gram = u_gram.flip(1)[:, : x.shape[1]]
    if torch.any(eigenvalues <= 0):
        raise FloatingPointError("FP64 Gram eigendecomposition lost positive modes")
    s_gram = torch.sqrt(eigenvalues).to(dtype=x.dtype)
    u_gram = u_gram.to(dtype=x.dtype)

    active_rank = 256
    ud = u_direct[:, :active_rank]
    ug = u_gram[:, :active_rank]
    sd = s_direct[:active_rank]
    sg = s_gram[:active_rank]
    overlap = ud.T @ ug
    cosines = torch.linalg.svdvals(overlap.float()).clamp(0, 1)
    normalized_projector_distance = math.sqrt(
        max(0.0, 1.0 - float(overlap.double().square().sum().item()) / active_rank)
    )

    # Match the two production formulas exactly.  The direct path uses V,
    # whereas the statistics path recovers YV from (YX^T-SXX^T)U Sigma^-1.
    y_remaining = residual @ x
    projected_direct = y_remaining @ vh_direct[:active_rank].T
    h_remaining = residual.to(torch.float64) @ covariance
    projected_gram = (
        h_remaining @ ug.to(torch.float64)
    ) / sg.to(torch.float64).unsqueeze(0)
    projected_gram = projected_gram.to(dtype=x.dtype)

    print("BRIDGE_DIAGNOSTIC_PHASE=constructing_bridges", flush=True)
    a_direct, b_direct, bridge_direct, bridge_direct_q4 = construct_bridge(
        residual, ud, sd, projected_direct, rank=128,
    )
    a_gram, b_gram, bridge_gram, bridge_gram_q4 = construct_bridge(
        residual, ug, sg, projected_gram, rank=128,
    )
    a_direct_canonical, b_direct_canonical = canonicalize_low_rank_factors(
        a_direct, b_direct,
    )
    a_gram_canonical, b_gram_canonical = canonicalize_low_rank_factors(
        a_gram, b_gram,
    )
    bridge_direct_canonical_q4 = quantize_4bit_grouped(
        a_direct_canonical, group_size=64,
    ) @ quantize_4bit_grouped(b_direct_canonical, group_size=64)
    bridge_gram_canonical_q4 = quantize_4bit_grouped(
        a_gram_canonical, group_size=64,
    ) @ quantize_4bit_grouped(b_gram_canonical, group_size=64)
    a_direct_anchored, b_direct_anchored, direct_anchor_condition = (
        anchor_low_rank_factors(a_direct, b_direct)
    )
    a_gram_anchored, b_gram_anchored, gram_anchor_condition = (
        anchor_low_rank_factors(a_gram, b_gram)
    )
    bridge_direct_anchored_q4 = quantize_4bit_grouped(
        a_direct_anchored, group_size=64,
    ) @ quantize_4bit_grouped(b_direct_anchored, group_size=64)
    bridge_gram_anchored_q4 = quantize_4bit_grouped(
        a_gram_anchored, group_size=64,
    ) @ quantize_4bit_grouped(b_gram_anchored, group_size=64)

    report = {
        "scope": "real Llama-2-7B layer-0 down_proj bridge-only diagnostic",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_data": input_data,
        "shape": list(x.shape),
        "active_rank": active_rank,
        "bridge_rank": 128,
        "spectrum": {
            "relative_singular_value_difference_top_active": relative_error(sg, sd),
            "active_cutoff_ratio": float((s_direct[active_rank - 1] / s_direct[0]).item()),
            "cutoff_relative_gap": float(
                ((s_direct[active_rank - 1] - s_direct[active_rank])
                 / s_direct[active_rank - 1]).item()
            ),
            "smallest_principal_cosine": float(cosines[-1].item()),
            "largest_principal_angle_degrees": float(
                torch.rad2deg(torch.acos(cosines[-1])).item()
            ),
            "normalized_projector_distance": normalized_projector_distance,
        },
        "projected_target_relative_difference": relative_error(
            projected_gram, projected_direct,
        ),
        "bridge": {
            "unquantized_weight_relative_difference": relative_error(
                bridge_gram, bridge_direct,
            ),
            "q4_weight_relative_difference": relative_error(
                bridge_gram_q4, bridge_direct_q4,
            ),
            "unquantized_calibration_functional_difference": functional_difference(
                bridge_direct, bridge_gram, x,
            ),
            "unquantized_validation_functional_difference": functional_difference(
                bridge_direct, bridge_gram, x_validation,
            ),
            "q4_calibration_functional_difference": functional_difference(
                bridge_direct_q4, bridge_gram_q4, x,
            ),
            "q4_validation_functional_difference": functional_difference(
                bridge_direct_q4, bridge_gram_q4, x_validation,
            ),
            "canonical_q4_weight_relative_difference": relative_error(
                bridge_gram_canonical_q4, bridge_direct_canonical_q4,
            ),
            "canonical_q4_calibration_functional_difference": functional_difference(
                bridge_direct_canonical_q4, bridge_gram_canonical_q4, x,
            ),
            "canonical_q4_validation_functional_difference": functional_difference(
                bridge_direct_canonical_q4, bridge_gram_canonical_q4, x_validation,
            ),
            "direct_canonicalization_preserves_unquantized_product": relative_error(
                a_direct_canonical @ b_direct_canonical, bridge_direct,
            ),
            "gram_canonicalization_preserves_unquantized_product": relative_error(
                a_gram_canonical @ b_gram_canonical, bridge_gram,
            ),
            "anchored_q4_weight_relative_difference": relative_error(
                bridge_gram_anchored_q4, bridge_direct_anchored_q4,
            ),
            "anchored_q4_calibration_functional_difference": functional_difference(
                bridge_direct_anchored_q4, bridge_gram_anchored_q4, x,
            ),
            "anchored_q4_validation_functional_difference": functional_difference(
                bridge_direct_anchored_q4, bridge_gram_anchored_q4, x_validation,
            ),
            "direct_anchoring_preserves_unquantized_product": relative_error(
                a_direct_anchored @ b_direct_anchored, bridge_direct,
            ),
            "gram_anchoring_preserves_unquantized_product": relative_error(
                a_gram_anchored @ b_gram_anchored, bridge_gram,
            ),
            "direct_anchor_condition_number": direct_anchor_condition,
            "gram_anchor_condition_number": gram_anchor_condition,
        },
        "reference_stage_source": str(args.reference_output),
    }
    atomic_json_save(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
