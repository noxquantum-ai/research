"""Direct-sum Schmidt sectors that protect complete nonlinear feature paths.

The protected and bulk neuron sets are disjoint coordinate sectors. A common
selection of Gate/Up rows and Down columns commutes exactly with coordinatewise
SwiGLU, while the bulk remains represented by NEC MPOs. Protected values,
indices, scales and bulk cores are charged explicitly to their budgets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nox.residual_codec import (
    codec_payload_bytes, decode_groupwise, encode_groupwise,
)


CODEC_TYPE = "nec_packed_sector_schmidt_v1"


@torch.no_grad()
def select_swiglu_sectors(
    gate, up, down, probe_gate, probe_up, inputs, protected_count, *, chunk_tokens=512,
):
    """Rank exact single-neuron output-error gains on construction data.

    For feature defect dh_j and total error e=D dh, restoring neuron j changes
    squared error by 2 <dh_j, D_j^T e> - ||D_j||^2 ||dh_j||^2. This is an exact
    one-neuron intervention score, not an assertion that gains add across a set.
    """

    width = gate.shape[0]
    if not 0 < protected_count < width:
        raise ValueError("protected sector must be nonempty and smaller than neuron width")
    gate, up, down = gate.float(), up.float(), down.float()
    probe_gate, probe_up = probe_gate.float(), probe_up.float()
    scores = torch.zeros(width, device=inputs.device, dtype=torch.float64)
    down_norm = down.square().sum(dim=0)
    for begin in range(0, inputs.shape[1], chunk_tokens):
        x = inputs[:, begin:begin + chunk_tokens].float()
        teacher = F.silu(gate @ x) * (up @ x)
        probe = F.silu(probe_gate @ x) * (probe_up @ x)
        defect = teacher - probe
        error = down @ defect
        alignment = (defect * (down.T @ error)).sum(dim=1)
        energy = down_norm * defect.square().sum(dim=1)
        scores += (2 * alignment - energy).double()
    selected = torch.argsort(scores, descending=True, stable=True)[:protected_count]
    return selected.sort().values, {
        "method": "single_neuron_exact_readout_error_gain",
        "protected_neurons": protected_count,
        "total_neurons": width,
        "fit_tokens": inputs.shape[1],
        "uses_nonlinear_derivatives": False,
        "selection_uses_validation": False,
        "selected_score_sum": float(scores.index_select(0, selected).sum()),
        "score_sum_is_joint_error_reduction": False,
    }


@torch.no_grad()
def compile_sector_schmidt(
    weight, *, protected_indices, neuron_axis, protected_bits,
    byte_budget, coefficient_budget, calibration_inputs, calibration_targets=None,
    **construction_options,
):
    """Compile protected coordinates and a smaller bulk MPO under both caps."""

    from nox.packed_schmidt import (
        compile_packed_schmidt, packed_codec_coefficient_count,
        packed_schmidt_coefficient_count_from_artifact,
        packed_schmidt_payload_bytes_from_artifact,
    )

    if neuron_axis not in (0, 1):
        raise ValueError("neuron_axis must be zero for Gate/Up or one for Down")
    indices = torch.as_tensor(protected_indices, device=weight.device, dtype=torch.long)
    width = weight.shape[neuron_axis]
    if indices.numel() == 0 or indices.numel() >= width:
        raise ValueError("sector dimensions must be positive")
    if torch.unique(indices).numel() != indices.numel() or int(indices.min()) < 0 or int(indices.max()) >= width:
        raise ValueError("protected indices must be unique valid neuron coordinates")
    mask = torch.ones(width, device=weight.device, dtype=torch.bool)
    mask[indices] = False
    bulk_indices = torch.arange(width, device=weight.device)[mask]
    group_size = int(construction_options.get("group_size", 64))
    protected_weight = weight.index_select(neuron_axis, indices)
    protected_codec = encode_groupwise(
        protected_weight, bits=protected_bits, group_size=group_size,
    )
    protected_decoded = decode_groupwise(protected_codec, device=weight.device)
    index_bytes = width * 4
    protected_bytes = codec_payload_bytes(protected_codec) + index_bytes
    protected_coefficients = packed_codec_coefficient_count(protected_codec)
    bulk_bytes = byte_budget - protected_bytes
    bulk_coefficients = coefficient_budget - protected_coefficients
    if min(bulk_bytes, bulk_coefficients) <= 0:
        raise ValueError("protected sector consumes the entire operator budget")
    targets = (
        weight.float() @ calibration_inputs.float()
        if calibration_targets is None else calibration_targets.float()
    )
    if neuron_axis == 0:
        bulk_inputs = calibration_inputs
        bulk_targets = targets.index_select(0, bulk_indices)
    else:
        bulk_inputs = calibration_inputs.index_select(0, bulk_indices)
        bulk_targets = targets - protected_decoded @ calibration_inputs.index_select(0, indices).float()
    base, _, diagnostics = compile_packed_schmidt(
        weight.index_select(neuron_axis, bulk_indices),
        byte_budget=bulk_bytes, coefficient_budget=bulk_coefficients,
        calibration_inputs=bulk_inputs, calibration_targets=bulk_targets,
        **construction_options,
    )
    artifact = {
        "codec_type": CODEC_TYPE,
        "representation": "direct_sum_native_neuron_and_bulk_schmidt_mpo",
        "orig_shape": tuple(weight.shape),
        "neuron_axis": neuron_axis,
        "protected_indices": indices.to(device="cpu", dtype=torch.int32),
        "bulk_indices": bulk_indices.to(device="cpu", dtype=torch.int32),
        "protected_codec": protected_codec,
        "base_artifact": base,
        "byte_budget": int(byte_budget),
        "coefficient_budget": int(coefficient_budget),
        "post_compression_healing": False,
    }
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    artifact["logical_coefficient_count"] = packed_schmidt_coefficient_count_from_artifact(artifact)
    if artifact["payload_bytes"] > byte_budget or artifact["logical_coefficient_count"] > coefficient_budget:
        raise RuntimeError("direct-sum sectors exceeded the assigned budgets")
    decoded = reconstruct_sector_schmidt(artifact, device=weight.device)
    return artifact, decoded, {
        "representation": artifact["representation"],
        "protected_neurons": indices.numel(), "bulk_neurons": bulk_indices.numel(),
        "protected_coefficients": protected_coefficients,
        "protected_payload_bytes": protected_bytes,
        "payload_bytes": artifact["payload_bytes"],
        "logical_coefficient_count": artifact["logical_coefficient_count"],
        "relative_calibration_output_error": float(
            (decoded @ calibration_inputs.float() - targets).norm()
            / targets.norm().clamp_min(1e-30)
        ),
        "bulk_construction": diagnostics,
    }


def reconstruct_sector_schmidt(artifact, *, device="cpu", dtype=torch.float32):
    from nox.packed_schmidt import reconstruct_packed_schmidt

    base = reconstruct_packed_schmidt(artifact["base_artifact"], device=device)
    protected = decode_groupwise(artifact["protected_codec"], device=device)
    indices = artifact["protected_indices"].to(device=device, dtype=torch.long)
    bulk_indices = artifact["bulk_indices"].to(device=device, dtype=torch.long)
    dense = torch.empty(artifact["orig_shape"], device=device, dtype=torch.float32)
    dense.index_copy_(int(artifact["neuron_axis"]), indices, protected)
    dense.index_copy_(int(artifact["neuron_axis"]), bulk_indices, base)
    return dense.to(dtype=dtype)
