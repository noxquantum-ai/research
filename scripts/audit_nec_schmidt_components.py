#!/usr/bin/env python3
"""Measure how NEC's sparse, bridge, and scale steps change MPO cut spectra.

This is a read-only diagnostic.  It compares the same three NEC tensor cuts
for W, W-S, W-S-AB, and (W-S-AB)D without fitting a replacement operator.
Large central cuts use a reproducible randomized range SVD by default and
retain unresolved Frobenius energy in every tail estimate.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.artifacts import source_files_fingerprint
from nox.spectra import named_tt_spectrum_audit
from nox.tensor_order import (
    IDENTITY_SITE_ORDER,
    mode_index_permutation,
    unpermute_projection,
    validate_site_order,
)
from nox.utils import get_modes_for_projection


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_components(weight, artifact, *, device):
    if artifact.get("codec_type") not in (None, "nec_mpo_v1"):
        raise ValueError("component spectrum audit requires an NEC MPO artifact")
    rows, columns = map(int, artifact.get("orig_shape", weight.shape))
    if (rows, columns) != tuple(weight.shape):
        raise ValueError("artifact and dense weight shapes differ")

    dtype = torch.float32
    sparse = torch.zeros((rows, columns), device=device, dtype=dtype)
    if artifact.get("sparse_indices") is not None:
        indices = artifact["sparse_indices"].to(device=device, dtype=torch.long)
        values = artifact["sparse_values"].to(device=device, dtype=dtype)
        sparse[indices[:, 0], indices[:, 1]] = values

    bridge = torch.zeros_like(sparse)
    if artifact.get("A_bridge") is not None:
        bridge = (
            artifact["A_bridge"].to(device=device, dtype=dtype)
            @ artifact["B_bridge"].to(device=device, dtype=dtype)
        )

    output_modes, input_modes = get_modes_for_projection(rows, columns)
    output_modes = tuple(map(int, artifact.get("original_o_modes", output_modes)))
    input_modes = tuple(map(int, artifact.get("original_i_modes", input_modes)))
    site_order = validate_site_order(artifact.get("site_order", IDENTITY_SITE_ORDER))
    scale = artifact["d_awt"].to(device=device, dtype=dtype)
    if scale.ndim != 1 or scale.numel() != columns or not torch.isfinite(scale).all():
        raise ValueError("artifact input scale is malformed")

    # Artifacts with a non-identity site order store components and scale in
    # reordered coordinates.  Restore original coordinates before applying
    # the common audit tensorization below.
    if site_order != IDENTITY_SITE_ORDER:
        out_perm = mode_index_permutation(output_modes, site_order, device=device)
        in_perm = mode_index_permutation(input_modes, site_order, device=device)
        sparse = unpermute_projection(sparse, out_perm, in_perm)
        bridge = unpermute_projection(bridge, out_perm, in_perm)
        scale = scale.index_select(0, torch.argsort(in_perm))

    return sparse, bridge, scale, output_modes, input_modes, site_order


def _successive_deltas(audit):
    names = list(audit["variants"])
    comparisons = []
    for before, after in zip(names, names[1:]):
        cuts = []
        for before_cut, after_cut in zip(
            audit["variants"][before], audit["variants"][after],
        ):
            cuts.append({
                "cut_after_site": before_cut["cut_after_site"],
                "configured_rank_residual_energy_delta": (
                    after_cut["configured_rank_residual_energy_fraction"]
                    - before_cut["configured_rank_residual_energy_fraction"]
                ),
                "reported_partial_entropy_delta_nats": (
                    after_cut["reported_spectrum_partial_entropy_nats"]
                    - before_cut["reported_spectrum_partial_entropy_nats"]
                ),
            })
        comparisons.append({"before": before, "after": after, "cuts": cuts})
    return comparisons


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-weight", type=Path, required=True)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-rank", type=int, default=512)
    parser.add_argument("--oversample", type=int, default=32)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument(
        "--exact-large-cuts", action="store_true",
        help="Compute full FP64 Gram spectra for central cuts; expensive for Llama matrices.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.max_rank <= 0 or args.oversample < 0 or args.power_iterations < 0:
        raise ValueError("invalid randomized spectrum settings")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    artifacts = checkpoint.get("compressed_artifacts", checkpoint)
    if args.projection not in artifacts:
        raise KeyError(f"projection {args.projection!r} is absent from checkpoint")
    artifact = artifacts[args.projection]
    weight = torch.load(args.dense_weight, map_location="cpu", weights_only=True)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("dense weight file must contain one tensor")
    weight = weight.to(device=device, dtype=torch.float32)
    sparse, bridge, scale, output_modes, input_modes, site_order = _artifact_components(
        weight, artifact, device=device,
    )
    residual_sparse = weight - sparse
    residual_bridge = residual_sparse - bridge
    variants = {
        "weight": weight,
        "weight_minus_sparse": residual_sparse,
        "weight_minus_sparse_bridge": residual_bridge,
        "conditioned_residual": residual_bridge * scale.unsqueeze(0),
    }
    ranks = tuple(map(int, artifact["ranks"]))
    audit = named_tt_spectrum_audit(
        variants, output_modes, input_modes, ranks,
        site_order=site_order,
        exact_large_cuts=args.exact_large_cuts,
        seed=args.seed,
        max_rank=args.max_rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
    )
    report = {
        "diagnostic": "nec_component_operator_schmidt_spectra",
        "projection": args.projection,
        "checkpoint": str(args.checkpoint),
        "dense_weight": str(args.dense_weight),
        "input_sha256": {
            "checkpoint": _sha256(args.checkpoint),
            "dense_weight": _sha256(args.dense_weight),
        },
        "source_fingerprint": source_files_fingerprint([
            Path(__file__), ROOT / "nox/spectra.py", ROOT / "nox/utils.py",
            ROOT / "nox/tensor_order.py",
        ]),
        "component_norms": {
            "weight": float(torch.linalg.vector_norm(weight)),
            "sparse": float(torch.linalg.vector_norm(sparse)),
            "bridge": float(torch.linalg.vector_norm(bridge)),
            "weight_minus_sparse": float(torch.linalg.vector_norm(residual_sparse)),
            "weight_minus_sparse_bridge": float(torch.linalg.vector_norm(residual_bridge)),
            "conditioned_residual": float(torch.linalg.vector_norm(variants["conditioned_residual"])),
        },
        "spectrum_audit": audit,
        "successive_deltas": _successive_deltas(audit),
        "interpretation_rule": (
            "A positive configured-rank residual-energy delta means the named NEC step made "
            "that specific cut harder at the configured rank. Entropy deltas are interpretable "
            "as full-spectrum changes only for exact cuts; randomized partial entropy is descriptive."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
