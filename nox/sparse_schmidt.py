"""Schmidt backbone plus sparse coordinate excitation amplitudes.

The residual can have full algebraic rank even when the MPO bond is small.
This is explicitly a hybrid tensor-network/sparsity representation, not a
claim that conventional pruning or quantization alone is a new NEC method.
Masks occupy one actual serialized bit per original matrix coordinate; all
amplitude values and scales count toward the structural coefficient cap.
"""

from __future__ import annotations

import math

import torch

from nox.functional_schmidt import _prior_anchored_pcg
from nox.residual_codec import _pack_codes, _unpack_codes, decode_groupwise, encode_groupwise


CODEC_TYPE = "nec_sparse_excitation_schmidt_v1"


def excitation_indices(artifact, device):
    rows, columns = artifact["orig_shape"]
    mask = _unpack_codes(artifact["support_bits"].to(device), rows * columns, 1)
    return mask.reshape(rows, columns).bool().nonzero(as_tuple=False)[:, 1].reshape(rows, -1)


def excitation_values(artifact, device):
    return decode_groupwise(artifact["amplitude_codec"], device=device)


def reconstruct_sparse_schmidt(artifact, *, device="cpu", dtype=torch.float32):
    from nox.packed_schmidt import reconstruct_packed_schmidt
    base = reconstruct_packed_schmidt(artifact["base_artifact"], device=device)
    indices = excitation_indices(artifact, device)
    return base.scatter_add(1, indices, excitation_values(artifact, device)).to(dtype)


@torch.no_grad()
def compile_sparse_schmidt(
    weight, *, byte_budget, coefficient_budget, calibration_inputs,
    calibration_targets=None, backbone_fraction=.10, amplitude_bits=3,
    excitation_cg_iterations=8, excitation_ridge_ratio=.01,
    excitation_sequence_length=128, **options,
):
    """Select support and solve exact constrained linear normal equations."""
    from nox.packed_schmidt import (
        compile_packed_schmidt, packed_matrix_coefficient_count,
        packed_matrix_payload_bytes, packed_schmidt_coefficient_count_from_artifact,
        packed_schmidt_payload_bytes_from_artifact,
    )
    if not 0 < backbone_fraction < 1:
        raise ValueError("backbone fraction must lie strictly between zero and one")
    if excitation_cg_iterations < 1 or excitation_ridge_ratio <= 0:
        raise ValueError("invalid sparse amplitude construction controls")
    rows, columns = weight.shape
    group_size = int(options.get("group_size", 64))
    mask_bytes = math.ceil(rows * columns / 8)
    base_cap = min(math.floor(byte_budget * backbone_fraction), byte_budget - mask_bytes)
    if base_cap <= 0:
        raise ValueError("coordinate support mask does not fit in the operator budget")
    base, decoded_base, base_diagnostics = compile_packed_schmidt(
        weight, byte_budget=base_cap,
        coefficient_budget=math.floor(coefficient_budget * backbone_fraction),
        calibration_inputs=calibration_inputs, calibration_targets=calibration_targets,
        **options,
    )
    base_bytes = packed_schmidt_payload_bytes_from_artifact(base)
    base_coefficients = packed_schmidt_coefficient_count_from_artifact(base)
    low, high = 0, columns
    while low < high:
        middle = (low + high + 1) // 2
        fits = (
            base_bytes + mask_bytes + packed_matrix_payload_bytes(rows, middle, amplitude_bits, group_size) <= byte_budget
            and base_coefficients + packed_matrix_coefficient_count(rows, middle, group_size) <= coefficient_budget
        )
        if fits:
            low = middle
        else:
            high = middle - 1
    kept = low
    if kept < 1:
        raise ValueError("no excitation amplitudes fit after the MPO backbone and support mask")
    x = calibration_inputs.float()
    if excitation_sequence_length < 1 or x.shape[1] % excitation_sequence_length:
        raise ValueError("sparse construction requires complete sequence windows")
    windows = x.shape[1] // excitation_sequence_length
    if windows < 2:
        raise ValueError("sparse construction requires at least two windows")
    split = (windows - max(1, windows // 4)) * excitation_sequence_length
    fit_x, guard_x = x[:, :split], x[:, split:]
    target = weight.float() @ x if calibration_targets is None else calibration_targets.float()
    reference_residual = weight.float() - decoded_base
    # The mask depends only on construction inputs and source weights. It
    # preserves high-impact coordinate defects rather than truncating another
    # Schmidt cut. Each row has the same number of stored amplitudes.
    score = reference_residual.abs() * fit_x.square().mean(dim=1).clamp_min(1e-20).sqrt()[None, :]
    indices = score.topk(kept, dim=1, sorted=False).indices.sort(dim=1).values
    del score
    mask = torch.zeros_like(weight, dtype=torch.bool).scatter_(1, indices, True)
    prior = reference_residual.gather(1, indices)
    amplitude_codec = encode_groupwise(prior, bits=amplitude_bits, group_size=group_size)
    prior = decode_groupwise(amplitude_codec, device=x.device)
    residual_target = target - decoded_base @ x
    fit_y, guard_y = residual_target[:, :split], residual_target[:, split:]
    covariance = fit_x @ fit_x.T / split
    ridge = excitation_ridge_ratio * covariance.diagonal().mean().clamp_min(1e-20)

    def dense(values):
        return torch.zeros_like(weight, dtype=torch.float32).scatter_(1, indices, values)

    def errors(values):
        matrix = dense(values)
        return tuple(float((matrix @ a - b).square().sum() / b.square().sum().clamp_min(1e-30))
                     for a, b in ((fit_x, fit_y), (guard_x, guard_y)))

    initial = errors(prior)
    rhs = ((fit_y - dense(prior) @ fit_x) @ fit_x.T / split).gather(1, indices)
    diagonal = covariance.diagonal()[indices] + ridge
    delta, history = _prior_anchored_pcg(
        lambda values: (dense(values) @ covariance).gather(1, indices) + ridge * values,
        rhs, diagonal, iterations=excitation_cg_iterations, tolerance=1e-3,
    )
    best_fit, best_guard = initial
    accepted_fraction = None
    trials = []
    for fraction in (1., .5, .25):
        candidate = encode_groupwise(prior + fraction * delta, bits=amplitude_bits, group_size=group_size)
        candidate_values = decode_groupwise(candidate, device=x.device)
        fit_error, guard_error = errors(candidate_values)
        trials.append({"fraction": fraction, "fit_relative_mse": fit_error, "guard_relative_mse": guard_error})
        if fit_error < initial[0] and guard_error < best_guard:
            amplitude_codec = candidate
            best_fit, best_guard, accepted_fraction = fit_error, guard_error, fraction
    artifact = {
        "codec_type": CODEC_TYPE, "orig_shape": tuple(weight.shape),
        "representation": "schmidt_mpo_plus_sparse_coordinate_excitations",
        "base_artifact": base, "support_bits": _pack_codes(mask, 1).cpu(),
        "amplitude_codec": amplitude_codec, "amplitudes_per_row": kept,
        "byte_budget": int(byte_budget), "coefficient_budget": int(coefficient_budget),
        "post_compression_healing": False,
    }
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    artifact["logical_coefficient_count"] = packed_schmidt_coefficient_count_from_artifact(artifact)
    if artifact["payload_bytes"] > byte_budget or artifact["logical_coefficient_count"] > coefficient_budget:
        raise RuntimeError("sparse excitation representation exceeded its hard budget")
    decoded = decoded_base + dense(decode_groupwise(amplitude_codec, device=x.device))
    return artifact, decoded, {
        "representation": artifact["representation"], "backbone_fraction": backbone_fraction,
        "sparse_density": kept / columns, "amplitude_bits": amplitude_bits,
        "support_mask_bytes": mask_bytes, "payload_bytes": artifact["payload_bytes"],
        "logical_coefficient_count": artifact["logical_coefficient_count"],
        "initial_fit_relative_mse": initial[0], "initial_guard_relative_mse": initial[1],
        "final_fit_relative_mse": best_fit, "final_guard_relative_mse": best_guard,
        "accepted_fraction": accepted_fraction, "cg_history": history, "trials": trials,
        "base_construction": base_diagnostics, "uses_nonlinear_derivatives": False,
    }
