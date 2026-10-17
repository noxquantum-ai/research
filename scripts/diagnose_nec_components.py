#!/usr/bin/env python3
"""Audit saved NEC operator components and a matched operator-Schmidt fit.

This diagnostic is read-only with respect to the source checkpoint.  The
candidate is written only to the JSON report; it is never inserted into the
model or optimized after construction.
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.compiler import reconstruct_from_artifact
from nox.operator_schmidt import (
    _truncated_svd,
    four_site_parameter_count,
    nox_operator_schmidt_decompose,
)
from nox.utils import compute_metrics, get_modes_for_projection
from nox.tensor_order import IDENTITY_SITE_ORDER, mode_index_permutation
from scripts.audit_projection_fit import projection_inputs
from scripts.diagnose_cross_layer_interactions import PROJECTIONS, forward_details


RESTORE_GROUPS = {
    "attention": PROJECTIONS[:4],
    "mlp": PROJECTIONS[4:],
    **{name.removesuffix("_proj"): (name,) for name in PROJECTIONS},
}


def _metrics(reference, candidate):
    result = compute_metrics(reference, candidate)
    return {name: float(value) for name, value in result.items()}


def _dense_component(artifact, key, *, device, dtype):
    rows, columns = tuple(artifact["orig_shape"])
    result = torch.zeros((rows, columns), device=device, dtype=dtype)
    if key == "bridge" and artifact.get("A_bridge") is not None:
        result = artifact["A_bridge"].to(device=device, dtype=dtype) @ artifact[
            "B_bridge"
        ].to(device=device, dtype=dtype)
    elif key == "sparse" and artifact.get("sparse_indices") is not None:
        indices = artifact["sparse_indices"].to(device=device, dtype=torch.long)
        values = artifact["sparse_values"].to(device=device, dtype=dtype)
        result[indices[:, 0], indices[:, 1]] = values
    elif key == "core":
        core_artifact = dict(artifact)
        core_artifact["A_bridge"] = None
        core_artifact["B_bridge"] = None
        core_artifact["sparse_indices"] = None
        core_artifact["sparse_values"] = None
        result = reconstruct_from_artifact(core_artifact, device=device, dtype=dtype)
    return result


def _legacy_cut_spectrum(weight, artifact, projection_name, *, device):
    original_o_modes, original_i_modes = get_modes_for_projection(*weight.shape)
    site_order = tuple(artifact.get("site_order", IDENTITY_SITE_ORDER))
    o_modes = tuple(original_o_modes[index] for index in site_order)
    i_modes = tuple(original_i_modes[index] for index in site_order)
    ranks = tuple(int(value) for value in artifact["ranks"])
    core = _dense_component(artifact, "core", device=device, dtype=torch.float32)
    dense = weight.float()
    # The original core approximates the residual after the stored sparse and
    # bridge channels, in the diagonal AWT coordinates.
    sparse = _dense_component(artifact, "sparse", device=device, dtype=torch.float32)
    bridge = _dense_component(artifact, "bridge", device=device, dtype=torch.float32)
    residual = (dense - sparse - bridge) * artifact["d_awt"].to(
        device=device, dtype=torch.float32
    )[None, :]
    if site_order != IDENTITY_SITE_ORDER:
        out_perm = mode_index_permutation(original_o_modes, site_order, device=device)
        in_perm = mode_index_permutation(original_i_modes, site_order, device=device)
        residual = residual.index_select(0, out_perm).index_select(1, in_perm)
    del core
    tensor = residual.reshape(*o_modes, *i_modes).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    remainder = tensor.reshape(1, *tensor.shape)
    cuts = []
    previous = 1
    for site in range(3):
        next_unfolding = remainder.reshape(
            previous * o_modes[site] * i_modes[site], -1
        )
        retained_rank = min(ranks[site + 1], *next_unfolding.shape)
        total_energy = next_unfolding.square().sum().clamp_min(1e-30)
        left, singular_values, right = _truncated_svd(
            next_unfolding, retained_rank, oversample=8,
            power_iterations=1, seed=site,
        )
        retained = singular_values.square().sum()
        cuts.append({
            "cut_after_site": site,
            "unfolding_shape": tuple(next_unfolding.shape),
            "bond_rank": retained_rank,
            "retained_squared_spectral_energy": float(retained / total_energy),
            "discarded_squared_spectral_energy": float(1.0 - retained / total_energy),
        })
        remainder = singular_values[:, None] * right
        previous = retained_rank
    return cuts


def _projection_component_ablation(weight, artifact, train_input, valid_input, *, device):
    decoded = reconstruct_from_artifact(artifact, device=device, dtype=torch.float32)
    components = {
        name: _dense_component(artifact, name, device=device, dtype=torch.float32)
        for name in ("core", "bridge", "sparse")
    }
    variants = {
        "full": tuple(components),
        "core_only": ("core",),
        "bridge_only": ("bridge",),
        "sparse_only": ("sparse",),
        "core_plus_bridge": ("core", "bridge"),
        "core_plus_sparse": ("core", "sparse"),
        "bridge_plus_sparse": ("bridge", "sparse"),
    }
    result = {}
    for variant, names in variants.items():
        matrix = sum((components[name] for name in names), torch.zeros_like(weight))
        item = {
            "relative_weight_error": float(
                torch.linalg.vector_norm(weight - matrix)
                / torch.linalg.vector_norm(weight).clamp_min(1e-30)
            ),
            "train_output": _metrics(weight @ train_input, matrix @ train_input),
            "heldout_output": _metrics(weight @ valid_input, matrix @ valid_input),
        }
        if variant == "full":
            item["saved_checkpoint_reconstruction_difference"] = float(
                torch.linalg.vector_norm(decoded - matrix)
                / torch.linalg.vector_norm(matrix).clamp_min(1e-30)
            )
        result[variant] = item
    del decoded, components
    return result


def _layer_restoration_ablation(
    dense, compressed, validation_input, options, *, operator_schmidt_artifacts=None,
):
    dense_result = forward_details(dense, validation_input, **options)
    compressed_result = forward_details(compressed, validation_input, **options)
    result = {
        "compressed_baseline": {
            "block_output": _metrics(dense_result["output"], compressed_result["output"]),
            "active_transform": _metrics(
                dense_result["active_transform"], compressed_result["active_transform"]
            ),
            "attention_output": _metrics(
                dense_result["attention_output"], compressed_result["attention_output"]
            ),
            "mlp_output": _metrics(
                dense_result["mlp_output"], compressed_result["mlp_output"]
            ),
        }
    }
    if operator_schmidt_artifacts:
        operator_schmidt = dict(compressed)
        operator_schmidt.update({
            name: reconstruct_from_artifact(artifact, device=validation_input.device,
                                            dtype=torch.float32)
            for name, artifact in operator_schmidt_artifacts.items()
        })
        outputs = forward_details(operator_schmidt, validation_input, **options)
        result["operator_schmidt_candidate_block"] = {
            "replaced_projections": list(operator_schmidt_artifacts),
            "block_output": _metrics(dense_result["output"], outputs["output"]),
            "active_transform": _metrics(
                dense_result["active_transform"], outputs["active_transform"]
            ),
            "attention_output": _metrics(
                dense_result["attention_output"], outputs["attention_output"]
            ),
            "mlp_output": _metrics(
                dense_result["mlp_output"], outputs["mlp_output"]
            ),
        }
    for group, names in RESTORE_GROUPS.items():
        variant = dict(compressed)
        variant.update({name: dense[name] for name in names})
        outputs = forward_details(variant, validation_input, **options)
        result[f"restore_dense_{group}"] = {
            "restored_projections": list(names),
            "block_output": _metrics(dense_result["output"], outputs["output"]),
            "active_transform": _metrics(
                dense_result["active_transform"], outputs["active_transform"]
            ),
            "attention_output": _metrics(
                dense_result["attention_output"], outputs["attention_output"]
            ),
            "mlp_output": _metrics(
                dense_result["mlp_output"], outputs["mlp_output"]
            ),
        }
    return result


def _serialized_bytes(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return stream.tell()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dense-weights-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projections", default=",".join(PROJECTIONS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=32)
    parser.add_argument("--rms-norm-eps", type=float)
    parser.add_argument("--weight-mix", type=float, default=1.0)
    parser.add_argument("--activation-mix", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = tuple(name.strip() for name in args.projections.split(",") if name.strip())
    if not selected or set(selected) - set(PROJECTIONS) or len(set(selected)) != len(selected):
        raise ValueError(f"projections must be unique names from {PROJECTIONS}")

    checkpoint_path = args.checkpoint_dir / f"compressed_layer_{args.layer}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    epsilon = args.rms_norm_eps or checkpoint.get("rms_norm_eps", 1e-5)
    names = list(checkpoint["compressed_artifacts"])
    dense = {
        name: torch.load(
            args.dense_weights_dir / f"layer_{args.layer}_{name}.pt",
            map_location="cpu", weights_only=True,
        ).to(device=device, dtype=torch.float32)
        for name in names
    }
    for norm in ("input_layernorm", "post_attention_layernorm"):
        dense[norm] = checkpoint[norm].to(device=device, dtype=torch.float32)
    train_layer_input = torch.load(
        args.checkpoint_dir / f"calibration_input_layer_{args.layer}.pt",
        map_location="cpu", weights_only=True,
    ).to(device=device, dtype=torch.float32)
    validation_path = args.checkpoint_dir / f"layer_input_validation_layer_{args.layer}.pt"
    if validation_path.exists():
        valid_layer_input = torch.load(validation_path, map_location="cpu", weights_only=True)
    else:
        payload = torch.load(
            args.checkpoint_dir / f"dense_input_layer_{args.layer}.pt",
            map_location="cpu", weights_only=True,
        )
        valid_layer_input = payload["X_test"]
    valid_layer_input = valid_layer_input.to(device=device, dtype=torch.float32)

    train_contexts = projection_inputs(
        dense, train_layer_input, epsilon, args.seq_len, args.n_heads, args.n_kv_heads,
    )
    valid_contexts = projection_inputs(
        dense, valid_layer_input, epsilon, args.seq_len, args.n_heads, args.n_kv_heads,
    )
    compressed = dict(dense)
    compressed.update({
        name: reconstruct_from_artifact(artifact, device=device, dtype=torch.float32)
        for name, artifact in checkpoint["compressed_artifacts"].items()
    })
    report = {
        "diagnostic": "saved_nec_component_and_operator_schmidt_comparison",
        "layer": args.layer,
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "device": str(device),
        "rms_norm_eps": epsilon,
        "calibration_tokens": int(train_layer_input.shape[1]),
        "heldout_tokens": int(valid_layer_input.shape[1]),
        "objective": {
            "weight_mix": args.weight_mix,
            "activation_mix": args.activation_mix,
            "formula": "weight_mix*||W-W_hat||^2/||W||^2 + activation_mix*||(W-W_hat)X||^2/||WX||^2",
        },
        "block_restoration_ablation": {},
        "projections": {},
    }
    candidate_artifacts = {}

    for name in selected:
        weight = dense[name]
        artifact = checkpoint["compressed_artifacts"][name]
        train_input = train_contexts[name].float()
        valid_input = valid_contexts[name].float()
        baseline = _projection_component_ablation(
            weight, artifact, train_input, valid_input, device=device,
        )
        o_modes, i_modes = get_modes_for_projection(*weight.shape)
        ranks = tuple(int(value) for value in artifact["ranks"])
        bridge_rank = 0 if artifact.get("A_bridge") is None else artifact["A_bridge"].shape[1]
        sparse_count = (
            0 if artifact.get("sparse_values") is None else int(artifact["sparse_values"].numel())
        )
        fit = nox_operator_schmidt_decompose(
            W=weight,
            X_calib=train_input,
            Y_dense=weight @ train_input,
            o_modes=o_modes,
            i_modes=i_modes,
            ranks=ranks,
            r_bridge=bridge_rank,
            n_outliers=sparse_count,
            return_artifacts=True,
            verbose=True,
            projection_name=name,
            n_heads=args.n_heads,
            weight_mix=args.weight_mix,
            activation_mix=args.activation_mix,
        )
        candidate_artifact = fit[4]
        candidate_artifacts[name] = candidate_artifact
        candidate = reconstruct_from_artifact(
            candidate_artifact, device=device, dtype=torch.float32,
        )
        report["projections"][name] = {
            "original_nec": baseline,
            "original_nec_tensor_cut_spectrum": _legacy_cut_spectrum(
                weight, artifact, name, device=device,
            ),
            "original_core_coefficients": sum(core.numel() for core in artifact["cores"]),
            "candidate_operator_schmidt": {
                "selected_layout": candidate_artifact["selected_layout"],
                "layout_candidates": candidate_artifact["layout_candidates"],
                "core_coefficients": candidate_artifact["core_coefficient_count"],
                "core_coefficient_budget": four_site_parameter_count(o_modes, i_modes, ranks),
                "payload_bytes": candidate_artifact["payload_bytes"],
                "serialized_artifact_bytes": _serialized_bytes(candidate_artifact),
                "joint_component_gains": candidate_artifact["component_gains"],
                "joint_gain_condition": candidate_artifact["joint_gain_diagnostics"],
                "mixed_objective": candidate_artifact["objective"],
                "relative_weight_error": float(
                    torch.linalg.vector_norm(weight - candidate)
                    / torch.linalg.vector_norm(weight).clamp_min(1e-30)
                ),
                "train_output": _metrics(weight @ train_input, candidate @ train_input),
                "heldout_output": _metrics(weight @ valid_input, candidate @ valid_input),
            },
        }
        del fit, candidate, candidate_artifact
        if device.type == "cuda":
            torch.cuda.empty_cache()

    candidate_checkpoint = dict(checkpoint)
    candidate_checkpoint["representation"] = "operator_schmidt"
    candidate_layer_artifacts = dict(checkpoint["compressed_artifacts"])
    candidate_layer_artifacts.update(candidate_artifacts)
    candidate_checkpoint["compressed_artifacts"] = candidate_layer_artifacts
    candidate_checkpoint["total_kb"] = sum(
        float(artifact.get("total_kb", 0.0))
        for artifact in candidate_layer_artifacts.values()
    )
    candidate_checkpoint_bytes = _serialized_bytes(candidate_checkpoint)
    report["storage_comparison"] = {
        "original_layer_checkpoint_bytes": checkpoint_path.stat().st_size,
        "candidate_layer_checkpoint_serialized_bytes": candidate_checkpoint_bytes,
        "candidate_to_original_layer_bytes_ratio": (
            candidate_checkpoint_bytes / checkpoint_path.stat().st_size
        ),
        "candidate_projection_payload_bytes": sum(
            int(artifact["payload_bytes"])
            for artifact in candidate_artifacts.values()
        ),
        "candidate_projection_serialized_artifact_bytes": sum(
            _serialized_bytes(artifact) for artifact in candidate_artifacts.values()
        ),
        "replaced_projections": list(candidate_artifacts),
        "whole_model_size_included": False,
        "note": "Layer-only comparison; unchanged embeddings, output head, and other layers are excluded.",
    }

    report["block_restoration_ablation"] = _layer_restoration_ablation(
        dense, compressed, valid_layer_input, {
            "n_heads": args.n_heads,
            "n_kv_heads": args.n_kv_heads,
            "seq_len": args.seq_len,
            "rms_norm_eps": epsilon,
        },
        operator_schmidt_artifacts=candidate_artifacts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"results={args.output}", flush=True)


if __name__ == "__main__":
    main()
