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
CANONICAL_CODEC_TYPE = "nec_sparse_excitation_schmidt_v2"


def excitation_basis_scale(artifact, indices, device):
    if artifact.get('codec_type') != CANONICAL_CODEC_TYPE:
        return None
    if artifact.get('amplitude_basis') != 'backbone_input_scale':
        raise ValueError('unknown canonical excitation basis')
    scale = artifact['base_artifact']['input_scale'].to(device=device, dtype=torch.float32)
    if not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError('canonical excitation input scale must be positive and finite')
    return scale[indices].reciprocal()


@torch.no_grad()
def reconstruct_excitation_operator(weight, artifact, inputs, targets, *, sequence_length=2048,
                                    methods=("pcg_mse", "pcg_nonzero", "feedback_fixed", "feedback_support", "feedback_nonzero", "pcg_canonical", "feedback_canonical")):
    """Recompile an existing NEC operator without increasing its tensor budget.

    Only training construction and disjoint training guard windows select the
    constructor. The MPO and all branch shapes remain unchanged.
    """
    from nox.packed_schmidt import (reconstruct_packed_schmidt,
        packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact)
    allowed={'pcg_mse','pcg_nonzero','feedback_fixed','feedback_support','feedback_nonzero','pcg_canonical','feedback_canonical'}
    if set(methods)-allowed:
        raise ValueError('unknown excitation construction method')
    windows = inputs.shape[1] // sequence_length
    if inputs.shape[1] % sequence_length or windows < 2:
        raise ValueError("operator reconstruction requires complete fit/guard windows")
    split = (windows - max(1, windows // 4)) * sequence_length
    x, guard_x = inputs[:, :split].float(), inputs[:, split:].float()
    y, guard_y = targets[:, :split].float(), targets[:, split:].float()
    base = reconstruct_packed_schmidt(artifact['base_artifact'], device=inputs.device)
    residual_y, guard_residual = y - base @ x, guard_y - base @ guard_x
    indices = excitation_indices(artifact, inputs.device)
    prior = excitation_values(artifact, inputs.device, indices=indices)

    input_scale=artifact['base_artifact']['input_scale'].to(device=inputs.device,dtype=torch.float32)

    def scores(support, codec, canonical=False):
        values=decode_groupwise(codec,device=inputs.device)
        if canonical: values=values/input_scale[support]
        sparse = torch.zeros_like(base).scatter_(1, support, values)
        return (float((sparse @ x - residual_y).square().sum() / y.square().sum().clamp_min(1e-30)),
                float((sparse @ guard_x - guard_residual).square().sum() / guard_y.square().sum().clamp_min(1e-30)))

    initial_fit, initial_guard = scores(indices, artifact['amplitude_codec'], artifact.get('codec_type')==CANONICAL_CODEC_TYPE)
    best, best_fit, best_guard, selected = artifact, initial_fit, initial_guard, 'baseline'
    trials = []
    delta = None
    if any(m.startswith('pcg_') for m in methods):
        _, delta, _, _ = refit_sparse_excitations(artifact, inputs, targets, split=split)
    for method in methods:
        canonical=method.endswith('_canonical')
        if method.startswith('pcg_'):
            grid = 'nonzero' if method == 'pcg_nonzero' else 'symmetric'
            candidates = [(indices, encode_groupwise((prior + fraction * delta)*(input_scale[indices] if canonical else 1.),
                bits=artifact['amplitude_codec']['bits'], group_size=artifact['amplitude_codec']['group_size'],
                scale_policy='mse', quant_grid=grid, entry_weights=x.square().mean(dim=1)[indices]/(input_scale[indices].square() if canonical else 1.)),
                {'fraction': fraction}) for fraction in (.25, .5, 1.)]
        else:
            support, codec, details = covariance_feedback_excitations(
                weight.float() - base, x, residual_y, kept=artifact['amplitudes_per_row'],
                bits=artifact['amplitude_codec']['bits'], group_size=artifact['amplitude_codec']['group_size'],
                fixed_indices=indices if method == 'feedback_fixed' else None,
                quant_grid='nonzero' if method in ('feedback_nonzero','feedback_canonical') else 'symmetric',
                amplitude_input_scale=input_scale if canonical else None)
            candidates = [(support, codec, details)]
        for support, codec, details in candidates:
            fit, guard = scores(support, codec, canonical)
            trials.append({'method': method, 'fit_relative_mse': fit, 'guard_relative_mse': guard, **details})
            if fit < initial_fit and guard < best_guard:
                mask = torch.zeros_like(base, dtype=torch.bool).scatter_(1, support, True)
                best = {**artifact, 'support_bits': _pack_codes(mask, 1).cpu(), 'amplitude_codec': codec}
                best['codec_type']=CANONICAL_CODEC_TYPE if canonical else CODEC_TYPE
                if canonical: best['amplitude_basis']='backbone_input_scale'
                else: best.pop('amplitude_basis',None)
                best_fit, best_guard, selected = fit, guard, method
        del candidates
    if (packed_schmidt_coefficient_count_from_artifact(best) != packed_schmidt_coefficient_count_from_artifact(artifact)
            or packed_schmidt_payload_bytes_from_artifact(best) != packed_schmidt_payload_bytes_from_artifact(artifact)):
        raise RuntimeError('excitation reconstruction changed the tensor budget')
    return best, reconstruct_packed_schmidt(best, device=inputs.device), {
        'selected': selected, 'initial_fit_relative_mse': initial_fit, 'initial_guard_relative_mse': initial_guard,
        'final_fit_relative_mse': best_fit, 'final_guard_relative_mse': best_guard,
        'fit_tokens': split, 'guard_tokens': inputs.shape[1] - split, 'trials': trials,
        'logical_coefficients_added': 0, 'tensor_payload_bytes_added': 0,
        'validation_used_for_selection': False, 'uses_autograd': False,
    }


@torch.no_grad()
def refit_sparse_excitations(artifact, inputs, targets, *, split, scale_policy="mse",
                             quant_grid="symmetric", ridge_ratio=.01, iterations=16):
    """Exact supported normal-equation solve with channel-relative anchoring."""
    from nox.packed_schmidt import reconstruct_packed_schmidt
    base = reconstruct_packed_schmidt(artifact["base_artifact"], device=inputs.device)
    indices = excitation_indices(artifact, inputs.device)
    prior = excitation_values(artifact, inputs.device, indices=indices)
    x = inputs[:, :split].float()
    y = targets[:, :split].float() - base @ x
    covariance = x @ x.T / split
    variance = covariance.diagonal().clamp_min(1e-20)

    def dense(values):
        return torch.zeros_like(base).scatter_(1, indices, values)

    rhs = ((y - dense(prior) @ x) @ x.T / split).gather(1, indices)
    anchoring = ridge_ratio * variance[indices]
    delta, history = _prior_anchored_pcg(
        lambda values: (dense(values) @ covariance).gather(1, indices) + anchoring * values,
        rhs, variance[indices] + anchoring, iterations=iterations, tolerance=1e-4,
    )
    return prior, delta, indices, {"cg_history": history,
                                 "ridge_policy": "channel_relative",
                                 "scale_policy": scale_policy, "quant_grid": quant_grid}


@torch.no_grad()
def covariance_feedback_excitations(
    reference_residual, fit_inputs, fit_targets, *, kept, bits=3,
    group_size=64, ridge_ratio=.01, block_size=128, quant_grid="symmetric",
    scale_policy="mse", fixed_indices=None,
    amplitude_input_scale=None,
):
    """Construct sparse amplitudes with analytical correlated-error feedback.

    Canonicalize the input covariance by channel RMS before regularization.
    This is a temporary construction gauge: no transformation, covariance or
    inverse is stored at inference. Zeroed and rounded amplitudes both pass
    their error into undecided coordinates through the inverse metric.
    The packed support has exactly ``kept`` coordinates per row.
    """
    rows, columns = reference_residual.shape
    if not 0 < kept <= columns or block_size < 1 or ridge_ratio <= 0:
        raise ValueError("invalid covariance feedback construction controls")
    x = fit_inputs.float()
    if x.shape[0] != columns or fit_targets.shape != (rows, x.shape[1]):
        raise ValueError("construction inputs/targets have incompatible shapes")
    rms = x.square().mean(dim=1).sqrt().clamp_min(1e-8)
    normalized_x = x / rms[:, None]
    covariance = normalized_x @ normalized_x.T / x.shape[1]
    ridge = ridge_ratio * covariance.diagonal().mean().clamp_min(1e-20)
    covariance.diagonal().add_(ridge)
    cholesky = torch.linalg.cholesky(covariance)
    work = reference_residual.float() * rms[None, :]
    discrepancy = (fit_targets.float() - work @ normalized_x) @ normalized_x.T / x.shape[1]
    work.add_(torch.cholesky_solve(discrepancy.T.contiguous(), cholesky).T)
    inverse = torch.cholesky_inverse(cholesky)
    if fixed_indices is None:
        # Single-coordinate constrained quadratic saliency under the full
        # covariance, rather than an independent activation-energy score.
        scores = work.square() / inverse.diagonal().clamp_min(1e-20)[None, :]
        indices = scores.topk(kept, dim=1, sorted=False).indices.sort(dim=1).values
        del scores
    else:
        indices = fixed_indices
        if tuple(indices.shape) != (rows, kept):
            raise ValueError("fixed support does not match the coefficient budget")
    upper = torch.linalg.cholesky(inverse, upper=True)
    del inverse, cholesky, covariance, discrepancy, normalized_x
    mask = torch.zeros_like(work, dtype=torch.bool).scatter_(1, indices, True)
    positions = mask.cumsum(dim=1, dtype=torch.int32).sub_(1).clamp_min_(0)
    initial_values = (work / rms[None, :]).gather(1, indices)
    importance=rms.square()[indices]
    if amplitude_input_scale is not None:
        amplitude_input_scale=amplitude_input_scale.to(device=x.device,dtype=torch.float32)
        if amplitude_input_scale.shape!=(columns,) or torch.any(amplitude_input_scale<=0) or not torch.isfinite(amplitude_input_scale).all():
            raise ValueError('amplitude input scale must match the operator input channels')
        initial_values=initial_values*amplitude_input_scale[indices]
        importance=importance/amplitude_input_scale[indices].square()
    codec = encode_groupwise(initial_values, bits=bits, group_size=group_size,
                             scale_policy=scale_policy, quant_grid=quant_grid,
                             entry_weights=importance)
    scales = codec["group_scales"].to(device=x.device, dtype=torch.float32)
    quantized = torch.empty_like(work)
    qmin, qmax = int(codec["qmin"]), int(codec["qmax"])
    # Only a block of column decisions is sequential. Error outside that block
    # is propagated in one GPU matrix multiplication.
    for begin in range(0, columns, block_size):
        end = min(begin + block_size, columns)
        current = work[:, begin:end].clone()
        errors = torch.empty_like(current)
        local_upper = upper[begin:end, begin:end]
        for offset in range(end - begin):
            column = begin + offset
            group = torch.div(positions[:, column], group_size, rounding_mode="floor")
            scale = scales.gather(1, group[:, None]).squeeze(1).clamp_min(1e-30)
            original_units = current[:, offset] / rms[column]
            if amplitude_input_scale is not None:
                original_units=original_units*amplitude_input_scale[column]
            if quant_grid == "nonzero":
                codes = ((original_units / scale - qmin) * .5).round().clamp(0, (1 << bits) - 1)
                values = (codes * 2 + qmin) * scale
            else:
                values = (original_units / scale).round().clamp(qmin, qmax) * scale
            if amplitude_input_scale is not None:
                values=values/amplitude_input_scale[column]
            response = torch.where(mask[:, column], values * rms[column], 0.)
            quantized[:, column] = response / rms[column]
            error = (current[:, offset] - response) / local_upper[offset, offset]
            errors[:, offset] = error
            current[:, offset:] -= error[:, None] * local_upper[offset, offset:][None, :]
        if end < columns:
            work[:, end:] -= errors @ upper[begin:end, end:]
    values = quantized.gather(1, indices)
    if amplitude_input_scale is not None:
        values=values*amplitude_input_scale[indices]
    padded = torch.nn.functional.pad(values, (0, codec["groups_per_row"] * group_size - kept))
    normalized = padded.reshape(rows, -1, group_size) / scales[:, :, None].clamp_min(1e-30)
    codes = (((normalized - qmin) * .5).round().clamp(0, (1 << bits) - 1)
             if quant_grid == "nonzero" else (normalized.round().clamp(qmin, qmax) - qmin))
    codec["packed_codes"] = _pack_codes(codes.to(torch.uint8), bits).cpu()
    return indices, codec, {
        "construction_gauge": "unit_rms_input_covariance",
        "support": "inverse_covariance_saliency" if fixed_indices is None else "fixed",
        "ridge_ratio": ridge_ratio, "feedback_block_size": block_size,
        "quant_grid": quant_grid, "scale_policy": scale_policy,
        "uses_autograd": False, "stored_covariance_coefficients": 0,
    }


def excitation_indices(artifact, device):
    rows, columns = artifact["orig_shape"]
    mask = _unpack_codes(artifact["support_bits"].to(device), rows * columns, 1)
    return mask.reshape(rows, columns).bool().nonzero(as_tuple=False)[:, 1].reshape(rows, -1)


def excitation_values(artifact, device, *, indices=None):
    values=decode_groupwise(artifact["amplitude_codec"], device=device)
    if artifact.get('codec_type')==CANONICAL_CODEC_TYPE:
        if indices is None:
            indices=excitation_indices(artifact,device)
        values=values*excitation_basis_scale(artifact,indices,device)
    return values


def reconstruct_sparse_schmidt(artifact, *, device="cpu", dtype=torch.float32):
    from nox.packed_schmidt import reconstruct_packed_schmidt
    base = reconstruct_packed_schmidt(artifact["base_artifact"], device=device)
    indices = excitation_indices(artifact, device)
    return base.scatter_add(1, indices, excitation_values(artifact, device, indices=indices)).to(dtype)


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
    amplitude_basis=options.pop('amplitude_basis','native')
    if amplitude_basis not in ('native','backbone_input_scale'):
        raise ValueError('unknown sparse amplitude basis')
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
    basis_scale=(base['input_scale'].to(device=x.device,dtype=torch.float32)[indices]
                 if amplitude_basis=='backbone_input_scale' else None)
    encode=lambda values:encode_groupwise(values if basis_scale is None else values*basis_scale,
        bits=amplitude_bits,group_size=group_size,
        scale_policy='maxabs' if basis_scale is None else 'mse')
    decode=lambda codec:(decode_groupwise(codec,device=x.device) if basis_scale is None
                          else decode_groupwise(codec,device=x.device)/basis_scale)
    amplitude_codec = encode(prior)
    prior = decode(amplitude_codec)
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
        candidate = encode(prior + fraction * delta)
        candidate_values = decode(candidate)
        fit_error, guard_error = errors(candidate_values)
        trials.append({"fraction": fraction, "fit_relative_mse": fit_error, "guard_relative_mse": guard_error})
        if fit_error < initial[0] and guard_error < best_guard:
            amplitude_codec = candidate
            best_fit, best_guard, accepted_fraction = fit_error, guard_error, fraction
    artifact = {
        "codec_type": CODEC_TYPE if basis_scale is None else CANONICAL_CODEC_TYPE, "orig_shape": tuple(weight.shape),
        "representation": "schmidt_mpo_plus_sparse_coordinate_excitations",
        "base_artifact": base, "support_bits": _pack_codes(mask, 1).cpu(),
        "amplitude_codec": amplitude_codec, "amplitudes_per_row": kept,
        "byte_budget": int(byte_budget), "coefficient_budget": int(coefficient_budget),
        "post_compression_healing": False,
    }
    if basis_scale is not None:
        artifact['amplitude_basis']='backbone_input_scale'
    artifact["payload_bytes"] = packed_schmidt_payload_bytes_from_artifact(artifact)
    artifact["logical_coefficient_count"] = packed_schmidt_coefficient_count_from_artifact(artifact)
    if artifact["payload_bytes"] > byte_budget or artifact["logical_coefficient_count"] > coefficient_budget:
        raise RuntimeError("sparse excitation representation exceeded its hard budget")
    decoded = decoded_base + dense(decode(amplitude_codec))
    return artifact, decoded, {
        "representation": artifact["representation"], "backbone_fraction": backbone_fraction,
        "sparse_density": kept / columns, "amplitude_bits": amplitude_bits,
        "amplitude_basis": amplitude_basis,
        "support_mask_bytes": mask_bytes, "payload_bytes": artifact["payload_bytes"],
        "logical_coefficient_count": artifact["logical_coefficient_count"],
        "initial_fit_relative_mse": initial[0], "initial_guard_relative_mse": initial[1],
        "final_fit_relative_mse": best_fit, "final_guard_relative_mse": best_guard,
        "accepted_fraction": accepted_fraction, "cg_history": history, "trials": trials,
        "base_construction": base_diagnostics, "uses_nonlinear_derivatives": False,
    }
