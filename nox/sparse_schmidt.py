"""Schmidt backbone plus sparse coordinate excitation amplitudes.

The residual can have full algebraic rank even when the MPO bond is small.
This is explicitly a hybrid tensor-network/sparsity representation, not a
claim that conventional pruning or quantization alone is a new NEC method.
Masks occupy one actual serialized bit per original matrix coordinate; all
amplitude values and scales count toward the structural coefficient cap.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch

from nox.functional_schmidt import _prior_anchored_pcg
from nox.residual_codec import _pack_codes, _unpack_codes, decode_groupwise, encode_groupwise


CODEC_TYPE = "nec_sparse_excitation_schmidt_v1"
CANONICAL_CODEC_TYPE = "nec_sparse_excitation_schmidt_v2"
GLOBAL_CODEC_TYPE = "nec_global_excitation_schmidt_v1"


@torch.no_grad()
def reconstruct_global_excitations(weight, artifact, inputs, targets, sensitivity, *,
                                   sequence_length=2048, minimum_per_row=64):
    """Allocate coordinate excitations globally using nonlinear output sensitivity.

    The MPO is unchanged. Global groups of 64 values remove per-row padding;
    every value and scale counts against the old amplitude coefficient cap.
    Training guards select only packed operators; no gradients are used.
    """
    from nox.packed_schmidt import (reconstruct_packed_schmidt, packed_codec_coefficient_count,
        packed_schmidt_coefficient_count_from_artifact,packed_schmidt_payload_bytes_from_artifact)
    base=reconstruct_packed_schmidt(artifact['base_artifact'],device=inputs.device)
    old=reconstruct_packed_schmidt(artifact,device=inputs.device)
    windows=inputs.shape[1]//sequence_length
    split=(windows-max(1,windows//4))*sequence_length
    x,y=inputs[:,:split].float(),targets[:,:split].float();rows,columns=weight.shape
    cap=packed_codec_coefficient_count(artifact['amplitude_codec'])
    count=(cap//65)*64
    minimum=min(minimum_per_row,columns,count//rows)
    variance=x.square().mean(1).clamp_min(1e-20)
    impact=(weight.float()-base).square()*variance[None,:]*sensitivity[:,None]
    mask=torch.zeros_like(weight,dtype=torch.bool)
    local=impact.topk(minimum,dim=1,sorted=False).indices
    mask.scatter_(1,local,True);impact.masked_fill_(mask,-1)
    remaining=count-rows*minimum
    if remaining:
        chosen=impact.flatten().topk(remaining,sorted=False).indices
        mask.flatten()[chosen]=True
    del impact
    coordinates=mask.nonzero(as_tuple=False);ri,ci=coordinates[:,0],coordinates[:,1]
    prior=(weight.float()-base)[ri,ci].reshape(1,-1)
    dense=lambda values:torch.zeros_like(base).index_put_((ri,ci),values.reshape(-1))
    with construction_matmul_precision('high'):
        covariance=x@x.T/split;diagonal=covariance.diagonal().clamp_min(1e-20)
        rhs=(((y-base@x-dense(prior)@x)@x.T)/split)[ri,ci].reshape_as(prior)
        ridge=.01*diagonal[ci].reshape_as(prior)
        normal=lambda values:(dense(values)@covariance)[ri,ci].reshape_as(prior)+ridge*values
        delta,_=_prior_anchored_pcg(normal,rhs,diagonal[ci].reshape_as(prior)+ridge,iterations=16,tolerance=1e-4)
    sensitivity=sensitivity.clamp_min(1e-20)
    def errors(matrix):
        return tuple(float(((matrix@a-b).square()*sensitivity[:,None]).sum()/
                           (b.square()*sensitivity[:,None]).sum().clamp_min(1e-30))
                     for a,b in ((inputs[:,:split],targets[:,:split]),(inputs[:,split:],targets[:,split:])))
    initial=errors(old);best,decoded=artifact,old;best_guard=initial[1];records=[]
    for fraction in (0.,.5,1.):
        codec=encode_groupwise(prior+fraction*delta,bits=artifact['amplitude_codec']['bits'],
            group_size=64,scale_policy='mse',entry_weights=variance[ci].reshape_as(prior))
        physical=base+dense(decode_groupwise(codec,device=inputs.device))
        fit,guard=errors(physical);records.append({'fraction':fraction,'fit_error':fit,'guard_error':guard})
        if fit<initial[0] and guard<best_guard:
            best={'codec_type':GLOBAL_CODEC_TYPE,'orig_shape':tuple(weight.shape),
                  'base_artifact':artifact['base_artifact'],'support_bits':_pack_codes(mask,1).cpu(),
                  'amplitude_codec':codec,'post_compression_healing':False}
            best['payload_bytes']=packed_schmidt_payload_bytes_from_artifact(best)
            best['logical_coefficient_count']=packed_schmidt_coefficient_count_from_artifact(best)
            if best['logical_coefficient_count']>packed_schmidt_coefficient_count_from_artifact(artifact):
                raise RuntimeError('global allocation exceeded coefficient cap')
            decoded=physical;best_guard=guard
    return best,decoded,{'method':'global_output_sensitive_excitation_allocation','fit_tokens':split,
        'guard_tokens':inputs.shape[1]-split,'guard_used_for_support':False,'uses_autograd':False,
        'initial_errors':initial,'selected':best.get('codec_type')==GLOBAL_CODEC_TYPE,
        'trials':records,'maximum_coefficients':packed_schmidt_coefficient_count_from_artifact(artifact)}


def reconstruct_global_sparse_schmidt(artifact,*,device='cpu',dtype=torch.float32):
    from nox.packed_schmidt import reconstruct_packed_schmidt
    base=reconstruct_packed_schmidt(artifact['base_artifact'],device=device)
    mask=excitation_mask(artifact,device)
    values=decode_groupwise(artifact['amplitude_codec'],device=device).reshape(-1)
    if int(mask.sum())!=values.numel():raise ValueError('global support and amplitude counts differ')
    base[mask]+=values
    return base.to(dtype)


def regularized_teacher_target(symmetric_target, teacher_target, prior_prediction, *, fit_tokens, precision=None):
    """Fit-only closed-form interpolation; arXiv:2602.05902 Proposition 3.2."""
    if symmetric_target.shape != teacher_target.shape or prior_prediction.shape != teacher_target.shape:
        raise ValueError('target interpolation shapes differ')
    if fit_tokens < 1 or fit_tokens > teacher_target.shape[1]:
        raise ValueError('invalid target interpolation fit range')
    mismatch=teacher_target-symmetric_target
    u=mismatch[:,:fit_tokens];e=(symmetric_target-prior_prediction)[:,:fit_tokens]
    measured=u if precision is None else precision@u
    denominator=(u*measured).sum(dtype=torch.float64)
    alpha=0. if denominator<=1e-20 else float((-(e*measured).sum(dtype=torch.float64)/denominator).clamp(0,1))
    return symmetric_target+alpha*mismatch, {'alpha':alpha,'fit_tokens':fit_tokens,
        'guard_used_for_target_interpolation':False,'inference_coefficients_added':0,
        'method':'closed_form_regularized_asymmetric_target','reference':'https://arxiv.org/abs/2602.05902'}


@torch.no_grad()
def block_shrunk_output_precision(targets, *, fit_tokens, sequence_length):
    """Training-only full output geometry with window-level covariance shrinkage.

    Separate complete windows estimate uncertainty in the covariance, avoiding
    the assumption that adjacent tokens are independent. No metric is stored
    in the inference representation.
    """
    x=targets[:,:fit_tokens].float()
    if fit_tokens%sequence_length or fit_tokens//sequence_length<2:
        raise ValueError('output covariance needs at least two complete fit windows')
    x=x-x.mean(1,keepdim=True)
    variance=x.square().mean(1);floor=variance.median().clamp_min(1e-20)*.01
    scale=variance.clamp_min(floor).rsqrt();z=x*scale[:,None]
    with construction_matmul_precision('high'):
        covariance=z@z.T/fit_tokens
        windows=fit_tokens//sequence_length;uncertainty=torch.zeros((),device=x.device,dtype=torch.float64)
        for first in range(0,fit_tokens,sequence_length):
            chunk=z[:,first:first+sequence_length]
            difference=chunk@chunk.T/sequence_length-covariance
            uncertainty+=difference.square().sum(dtype=torch.float64)
        identity=torch.eye(len(scale),device=x.device)
        distance=(covariance-identity).square().sum(dtype=torch.float64).clamp_min(1e-20)
        shrinkage=float((uncertainty/(windows*(windows-1))/distance).clamp(1e-4,1))
        shrunk=(1-shrinkage)*covariance+shrinkage*identity
        precision=torch.cholesky_inverse(torch.linalg.cholesky(shrunk))
        precision*=scale[:,None]*scale[None,:]
        precision/=precision.diagonal().mean().clamp_min(1e-20)
    return precision,{'shrinkage':shrinkage,'fit_windows':windows,'fit_tokens':fit_tokens,
                      'guard_used':False,'stored_coefficients_added':0}


@contextmanager
def construction_matmul_precision(precision):
    """Scope tensor-core arithmetic to construction, never final decoding."""
    if precision not in ('highest', 'high'):
        raise ValueError('construction precision must be highest or high')
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


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
                                    methods=("pcg_mse", "pcg_nonzero", "feedback_fixed", "feedback_support", "feedback_nonzero", "pcg_canonical", "feedback_canonical"),
                                    output_bits=None, matmul_precision='highest', output_precision=None):
    """Recompile an existing NEC operator without increasing its tensor budget.

    Only training construction and disjoint training guard windows select the
    constructor. The MPO and all branch shapes remain unchanged.
    """
    from nox.packed_schmidt import (reconstruct_packed_schmidt,
        packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact)
    allowed={'pcg_mse','pcg_nonzero','feedback_fixed','feedback_support','feedback_nonzero','pcg_canonical','feedback_canonical'}
    if set(methods)-allowed:
        raise ValueError('unknown excitation construction method')
    if output_bits is None:
        output_bits = int(artifact['amplitude_codec']['bits'])
        allow_precision_reallocation = False
    else:
        from nox.residual_codec import SUPPORTED_BITS
        if output_bits not in SUPPORTED_BITS:
            raise ValueError('unsupported excitation precision')
        allow_precision_reallocation = True
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
        def energy(value):
            return value.square().sum() if output_precision is None else (value*(output_precision@value)).sum()
        return (float(energy(sparse @ x - residual_y) / energy(y).clamp_min(1e-30)),
                float(energy(sparse @ guard_x - guard_residual) / energy(guard_y).clamp_min(1e-30)))

    initial_fit, initial_guard = scores(indices, artifact['amplitude_codec'], artifact.get('codec_type')==CANONICAL_CODEC_TYPE)
    best, best_fit, best_guard, selected = artifact, initial_fit, initial_guard, 'baseline'
    trials = []
    delta = None
    if any(m.startswith('pcg_') for m in methods):
        with construction_matmul_precision(matmul_precision):
            _, delta, _, _ = refit_sparse_excitations(artifact, inputs, targets, split=split,
                                                     base_weight=base,output_precision=output_precision)
    for method in methods:
        canonical=method.endswith('_canonical')
        if method.startswith('pcg_'):
            grid = 'nonzero' if method == 'pcg_nonzero' else 'symmetric'
            candidates = [(indices, encode_groupwise((prior + fraction * delta)*(input_scale[indices] if canonical else 1.),
                bits=output_bits, group_size=artifact['amplitude_codec']['group_size'],
                scale_policy='mse', quant_grid=grid, entry_weights=x.square().mean(dim=1)[indices]/(input_scale[indices].square() if canonical else 1.)),
                {'fraction': fraction}) for fraction in (.25, .5, 1.)]
        else:
            with construction_matmul_precision(matmul_precision):
                support, codec, details = covariance_feedback_excitations(
                    weight.float() - base, x, residual_y, kept=artifact['amplitudes_per_row'],
                    bits=output_bits, group_size=artifact['amplitude_codec']['group_size'],
                    fixed_indices=indices if method == 'feedback_fixed' else None,
                    quant_grid='nonzero' if method in ('feedback_nonzero','feedback_canonical') else 'symmetric',
                    amplitude_input_scale=input_scale if canonical else None)
            candidates = [(support, codec, details)]
        for support, codec, details in candidates:
            fit, guard = scores(support, codec, canonical)
            trials.append({'method': method, 'fit_relative_mse': fit, 'guard_relative_mse': guard, **details})
            if fit < initial_fit and guard < best_guard:
                mask = torch.zeros_like(base, dtype=torch.bool).scatter_(1, support, True)
                best = {**artifact, 'support_bits': _pack_codes(mask, 1).cpu(), 'amplitude_codec': codec,
                        'support_layout': 'row_major'}
                best['codec_type']=CANONICAL_CODEC_TYPE if canonical else CODEC_TYPE
                if canonical: best['amplitude_basis']='backbone_input_scale'
                else: best.pop('amplitude_basis',None)
                best_fit, best_guard, selected = fit, guard, method
        del candidates
    added_bytes = packed_schmidt_payload_bytes_from_artifact(best) - packed_schmidt_payload_bytes_from_artifact(artifact)
    if (packed_schmidt_coefficient_count_from_artifact(best) != packed_schmidt_coefficient_count_from_artifact(artifact)
            or (added_bytes != 0 and not allow_precision_reallocation)):
        raise RuntimeError('excitation reconstruction changed the tensor budget')
    if added_bytes:
        payload = packed_schmidt_payload_bytes_from_artifact(best)
        best = {**best, 'payload_bytes': payload,
                'prior_byte_budget': artifact.get('byte_budget'),
                'byte_budget': max(int(artifact.get('byte_budget', 0)), payload),
                'global_storage_reallocation_requires_export': True}
    return best, reconstruct_packed_schmidt(best, device=inputs.device), {
        'selected': selected, 'initial_fit_relative_mse': initial_fit, 'initial_guard_relative_mse': initial_guard,
        'final_fit_relative_mse': best_fit, 'final_guard_relative_mse': best_guard,
        'fit_tokens': split, 'guard_tokens': inputs.shape[1] - split, 'trials': trials,
        'logical_coefficients_added': 0, 'tensor_payload_bytes_added': added_bytes,
        'explicit_precision_reallocation': allow_precision_reallocation,
        'temporary_construction_matmul_precision': matmul_precision,
        'complete_serialized_budget_requires_measurement': bool(added_bytes),
        'validation_used_for_selection': False, 'uses_autograd': False,
    }


@torch.no_grad()
def refit_sparse_excitations(artifact, inputs, targets, *, split, scale_policy="mse",
                             quant_grid="symmetric", ridge_ratio=.01, iterations=16,
                             base_weight=None, output_precision=None):
    """Exact supported normal-equation solve with channel-relative anchoring."""
    from nox.packed_schmidt import reconstruct_packed_schmidt
    base = (reconstruct_packed_schmidt(artifact["base_artifact"], device=inputs.device)
            if base_weight is None else base_weight)
    indices = excitation_indices(artifact, inputs.device)
    prior = excitation_values(artifact, inputs.device, indices=indices)
    x = inputs[:, :split].float()
    y = targets[:, :split].float() - base @ x
    covariance = x @ x.T / split
    variance = covariance.diagonal().clamp_min(1e-20)

    def dense(values):
        return torch.zeros_like(base).scatter_(1, indices, values)

    measured=lambda values:values if output_precision is None else output_precision@values
    rhs = measured((y - dense(prior) @ x) @ x.T / split).gather(1, indices)
    output_diagonal=1. if output_precision is None else output_precision.diagonal()[:,None]
    diagonal=variance[indices]*output_diagonal
    anchoring = ridge_ratio * diagonal
    delta, history = _prior_anchored_pcg(
        lambda values: measured(dense(values) @ covariance).gather(1, indices) + anchoring * values,
        rhs, diagonal + anchoring, iterations=iterations, tolerance=1e-4,
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


def excitation_mask(artifact, device):
    rows, columns = artifact["orig_shape"]
    mask = _unpack_codes(artifact["support_bits"].to(device), rows * columns, 1)
    layout=artifact.get('support_layout','row_major')
    if layout=='row_major':return mask.reshape(rows,columns).bool()
    if layout=='column_major':return mask.reshape(columns,rows).T.bool()
    raise ValueError('unknown excitation support layout')


def reorder_excitation_support(artifact, layout):
    """Losslessly reorder support bits; amplitudes retain logical row order."""
    import numpy as np
    if layout not in ('row_major','column_major'):raise ValueError('unknown support layout')
    mask=excitation_mask(artifact,'cpu').numpy()
    flat=mask.reshape(-1) if layout=='row_major' else mask.T.reshape(-1)
    changed={**artifact,'support_bits':torch.from_numpy(np.packbits(flat,bitorder='little')),
             'support_layout':layout}
    if not torch.equal(excitation_mask(changed,'cpu'),torch.from_numpy(mask)):
        raise RuntimeError('lossless support reorder verification failed')
    return changed


def excitation_indices(artifact, device):
    rows, _ = artifact['orig_shape']
    return excitation_mask(artifact,device).nonzero(as_tuple=False)[:,1].reshape(rows,-1)


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
