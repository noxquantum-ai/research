"""Activation-Aware Weight Tensorization (AWT) baseline (Beatini et al., arXiv:2610.10085)."""

import torch
from nox.utils import quantize_dequantize_groupwise


def awt_tt_svd(
    W,
    X_comp,
    o_modes,
    i_modes,
    ranks=None,
    alpha=0.5,
    eps=0.01,
    max_rank=64,
    min_rank=4,
    core_bits=4,
):
    """
    Activation-Aware Weight Tensorization (AWT) TT-SVD decomposition.
    
    Preconditions the weight columns with diagonal activation norms:
        W_scaled = W @ diag(d_alpha)
    Runs Frobenius TT-SVD on W_scaled, then folds the separable rank-1 approximation
    of diag(d_alpha)^{-1} back into the input modes of the tensor cores.
    Post-hoc core quantization is applied to the final cores.
    
    Args:
        W: 2D weight matrix [d_out, d_in]
        X_comp: input calibration activations [d_in, N]
        o_modes: tuple of 4 output mode sizes
        i_modes: tuple of 4 input mode sizes
        ranks: optional explicit bond dimensions [r1, r2, r3]
        alpha: activation power exponent (default 0.5)
        eps: truncation energy threshold if ranks is None
        max_rank: maximum bond dimension limit
        min_rank: minimum bond dimension limit
        core_bits: bit width for post-hoc core quantization
        
    Returns:
        cores: list of 4 MPO cores with absorbed scaling
        chosen_ranks: list of ranks [1, r1, r2, r3, 1]
    """
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes

    act_l2 = torch.norm(X_comp, p=2, dim=1) + 1e-8
    d_scale = (act_l2 / (X_comp.shape[1] ** 0.5)) ** alpha

    W_scaled = W * d_scale.unsqueeze(0)

    tensor_w = W_scaled.reshape(o0, o1, o2, o3, i0, i1, i2, i3)
    permuted = tensor_w.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()

    cores = []
    r_prev = 1
    t = permuted

    for m in range(3):
        t = t.reshape(r_prev * o_modes[m] * i_modes[m], -1)
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)

        if ranks is not None and m < len(ranks):
            r_next = ranks[m]
        else:
            energy = S ** 2
            total_energy = torch.sum(energy) + 1e-12
            cum_energy = torch.cumsum(energy, dim=0)
            discarded_ratio = 1.0 - (cum_energy / total_energy)
            valid_indices = torch.where(discarded_ratio <= eps)[0]
            if len(valid_indices) > 0:
                r_next = int(valid_indices[0].item()) + 1
            else:
                r_next = S.shape[0]
            r_next = max(min_rank, min(max_rank, min(r_next, S.shape[0])))

        U_truncated = U[:, :r_next].reshape(r_prev, o_modes[m], i_modes[m], r_next)
        cores.append(U_truncated)
        t = S[:r_next].unsqueeze(1) * Vh[:r_next, :]
        r_prev = r_next

    cores.append(t.reshape(r_prev, o3, i3, 1))

    inv_scale = (1.0 / d_scale).reshape(i0, i1, i2, i3)
    for m in range(4):
        reduce_dims = [d for d in range(4) if d != m]
        m_scale = inv_scale.mean(dim=reduce_dims)
        if m == 0:
            m_scale = m_scale * (inv_scale.mean() / (m_scale.mean() + 1e-8))
        cores[m] = cores[m] * m_scale.view(1, 1, -1, 1)

    if core_bits < 16:
        for idx in range(len(cores)):
            cores[idx] = quantize_dequantize_groupwise(cores[idx], bits=core_bits)

    chosen_ranks = [1, cores[1].shape[0], cores[2].shape[0], cores[3].shape[0], 1]
    return cores, chosen_ranks
