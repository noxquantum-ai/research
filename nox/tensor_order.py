"""Utilities for changing the physical-site order of a four-site MPO.

The permutation changes the MPO factorization, not the represented dense
operator.  A single order can be shared by all residual-width projections in a
layer stack so neighboring layers use the same hidden-channel site layout.
"""

import math

import torch


IDENTITY_SITE_ORDER = (0, 1, 2, 3)


def validate_site_order(site_order):
    order = tuple(int(axis) for axis in site_order)
    if len(order) != 4 or sorted(order) != [0, 1, 2, 3]:
        raise ValueError(f"site_order must be a permutation of (0, 1, 2, 3), got {order}")
    return order


def mode_index_permutation(mode_sizes, site_order, device=None):
    """Return indices p such that reordered[j] = original[p[j]]."""
    mode_sizes = tuple(int(size) for size in mode_sizes)
    order = validate_site_order(site_order)
    if len(mode_sizes) != 4 or any(size <= 0 for size in mode_sizes):
        raise ValueError(f"Expected four positive mode sizes, got {mode_sizes}")
    flat = torch.arange(math.prod(mode_sizes), device=device)
    return flat.reshape(mode_sizes).permute(order).reshape(-1)


def permute_projection(W, X, Y, o_modes, i_modes, site_order):
    """Reindex a projection and its calibration tensors into an MPO site order."""
    order = validate_site_order(site_order)
    if order == IDENTITY_SITE_ORDER:
        return W, X, Y, tuple(o_modes), tuple(i_modes), None, None

    out_perm = mode_index_permutation(o_modes, order, device=W.device)
    in_perm = mode_index_permutation(i_modes, order, device=W.device)
    W_ordered = W.index_select(0, out_perm).index_select(1, in_perm)
    X_ordered = X.index_select(0, in_perm)
    Y_ordered = Y.index_select(0, out_perm)
    ordered_o_modes = tuple(o_modes[axis] for axis in order)
    ordered_i_modes = tuple(i_modes[axis] for axis in order)
    return W_ordered, X_ordered, Y_ordered, ordered_o_modes, ordered_i_modes, out_perm, in_perm


def unpermute_projection(W_ordered, out_perm, in_perm):
    """Restore a matrix from reordered row and column coordinates."""
    if out_perm is None or in_perm is None:
        return W_ordered
    out_inverse = torch.argsort(out_perm)
    in_inverse = torch.argsort(in_perm)
    return W_ordered.index_select(0, out_inverse).index_select(1, in_inverse)


@torch.no_grad()
def tt_svd_relative_error(W, o_modes, i_modes, ranks, site_order):
    """Measure the TT-SVD truncation bound for one projection and site order.

    This is a weight-only screening metric. It does not include Nox's bridge,
    sparse shield, activation-weighted DMRG refinement, or quantization, so it
    must not be reported as an end-to-end compression result.
    """
    order = validate_site_order(site_order)
    out_perm = mode_index_permutation(o_modes, order, device=W.device)
    in_perm = mode_index_permutation(i_modes, order, device=W.device)
    ordered = W.index_select(0, out_perm).index_select(1, in_perm)
    oo = tuple(o_modes[axis] for axis in order)
    ii = tuple(i_modes[axis] for axis in order)
    tensor = ordered.reshape(*oo, *ii).permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    bond_ranks = tuple(ranks[1:4]) if len(ranks) == 5 else tuple(ranks)
    if len(bond_ranks) != 3 or any(int(rank) <= 0 for rank in bond_ranks):
        raise ValueError(f"Expected three positive TT bond ranks, got {ranks}")

    t = tensor
    r_prev = 1
    residual_sq = torch.zeros((), device=W.device, dtype=torch.float64)
    for site in range(3):
        left_dim = r_prev * oo[site] * ii[site]
        t = t.reshape(left_dim, -1)
        _, singular_values, vh = torch.linalg.svd(t, full_matrices=False)
        rank = min(int(bond_ranks[site]), singular_values.numel())
        residual_sq += singular_values[rank:].double().square().sum()
        t = singular_values[:rank].unsqueeze(1) * vh[:rank]
        r_prev = rank

    denominator = torch.linalg.vector_norm(W.double()).square().clamp_min(1e-30)
    return torch.sqrt(residual_sq / denominator).item()

