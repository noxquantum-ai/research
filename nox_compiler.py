"""Nox Entanglement Compiler (NEC): 4-step closed-form MPO solver with zero backpropagation."""

import torch
from utils import quantize_dequantize_groupwise


def adaptive_tt_svd_4site(
    W,
    o_modes,
    i_modes,
    ranks=None,
    eps=0.01,
    max_rank=64,
    min_rank=4,
):
    """
    Computes initial 4-site MPO cores via TT-SVD.
    Optionally sizes each bond dimension dynamically via spectral energy threshold eps.
    """
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes

    tensor_w = W.reshape(o0, o1, o2, o3, i0, i1, i2, i3)
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

        cores.append(U[:, :r_next].reshape(r_prev, o_modes[m], i_modes[m], r_next))
        t = S[:r_next].unsqueeze(1) * Vh[:r_next, :]
        r_prev = r_next

    cores.append(t.reshape(r_prev, o3, i3, 1))
    return cores


def local_ridge_solve(Z_env, Y_site, o_m, dim_param, ridge=1e-4, device=None):
    """
    Solves local site ridge regression: G_m* = Y_site @ Z_env (Z_env^T Z_env + lambda I)^(-1).
    Runs in milliseconds for small local dimensions.
    """
    if device is None:
        device = Z_env.device
    H = Z_env.T @ Z_env
    reg = ridge * (torch.trace(H) / H.shape[0] + 1e-8) * torch.eye(dim_param, device=device)
    B = Z_env.T @ Y_site
    return torch.linalg.solve(H + reg, B).T


def nox_4step_compiler(
    W,
    X_comp,
    Y_dense,
    o_modes,
    i_modes,
    ranks=None,
    eps=0.01,
    max_rank=64,
    min_rank=4,
    outlier_pct=0.005,
    r_subspace=256,
    n_sweeps=3,
    core_bits=4,
    ridge=1e-4,
):
    """
    Complete 4-Step Nox Entanglement Compiler for a 2D linear weight matrix:
      - Step 1: Operator-Norm Outlier Decoupling (W = S + W_core)
      - Step 2: Low-Rank Activation Density Projection (X_tilde, Y_tilde)
      - Step 3: Adaptive-Rank 1-Site DMRG / ALS Sweep with Isometric QR Gauge
      - Step 4.1: In-Loop Quantization Error Absorption on Final Forward Sweep
      
    Args:
        W: 2D weight matrix [d_out, d_in]
        X_comp: input activations [d_in, N]
        Y_dense: target uncompressed layer output [d_out, N]
        o_modes: tuple of 4 output mode sizes
        i_modes: tuple of 4 input mode sizes
        ranks: optional explicit bond dimensions [r1, r2, r3]
        eps: spectral energy threshold for bond sizing
        max_rank: maximum bond dimension limit
        min_rank: minimum bond dimension limit
        outlier_pct: fraction of entries to decouple into sparse buffer S (e.g. 0.005 = 0.5%)
        r_subspace: number of principal modes in activation projection
        n_sweeps: number of full DMRG sweeps (default 3)
        core_bits: target quantization bit-width for cores (default 4)
        ridge: ridge regularization factor
        
    Returns:
        final_cores: list of 4 optimized (and quantized) MPO cores
        S_sparse: sparse outlier matrix [d_out, d_in]
        chosen_ranks: list of ranks [1, r1, r2, r3, 1]
    """
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes
    device = W.device

    # Step 1: Operator-Norm Outlier Decoupling
    act_l2 = torch.norm(X_comp, p=2, dim=1) + 1e-8
    impact_scores = torch.abs(W) * act_l2.unsqueeze(0)
    k_outliers = int(outlier_pct * W.numel())
    if k_outliers > 0:
        thresh = torch.kthvalue(impact_scores.flatten(), W.numel() - k_outliers).values
        mask = impact_scores >= thresh
        S_sparse = torch.where(mask, W, torch.zeros_like(W))
        W_core = W - S_sparse
    else:
        S_sparse = torch.zeros_like(W)
        W_core = W.clone()

    Y_target = Y_dense - (S_sparse @ X_comp)

    # Step 2: Low-Rank Activation Density Projection
    U_x, S_x, Vh_x = torch.linalg.svd(X_comp, full_matrices=False)
    r = min(r_subspace, S_x.shape[0])
    X_tilde = (U_x[:, :r] * S_x[:r].unsqueeze(0)).reshape(i0, i1, i2, i3, r)
    Y_tilde = (Y_target @ Vh_x[:r, :].T).reshape(o0, o1, o2, o3, r)

    # Step 3.1: AWT Warm-Start + Dynamic Bond Sizing
    d_scale = (act_l2 / (X_comp.shape[1] ** 0.5)) ** 0.5
    cores = adaptive_tt_svd_4site(
        W_core * d_scale.unsqueeze(0),
        o_modes,
        i_modes,
        ranks=ranks,
        eps=eps,
        max_rank=max_rank,
        min_rank=min_rank,
    )

    inv_scale = (1.0 / d_scale).reshape(i0, i1, i2, i3)
    for m in range(4):
        reduce_dims = [d for d in range(4) if d != m]
        m_scale = inv_scale.mean(dim=reduce_dims)
        if m == 0:
            m_scale = m_scale * (inv_scale.mean() / (m_scale.mean() + 1e-8))
        cores[m] = cores[m] * m_scale.view(1, 1, -1, 1)

    c0 = cores[0].squeeze(0)
    c1 = cores[1]
    c2 = cores[2]
    c3 = cores[3].squeeze(-1)

    # Step 3.2 & Step 4.1: DMRG Sweeps with QR Gauge Fixing and Final-Pass Quantization
    for sweep in range(n_sweeps):
        is_final_sweep = (sweep == n_sweeps - 1)

        # Site 0: c0 (o0, i0, r1)
        r1 = c1.shape[0]
        Z0 = torch.einsum("bcjd,dekf,fgl,ijklr->cegrbi", c1, c2, c3, X_tilde).reshape(-1, r1 * i0)
        Y0 = Y_tilde.permute(1, 2, 3, 4, 0).reshape(-1, o0)
        c0 = local_ridge_solve(Z0, Y0, o0, r1 * i0, ridge=ridge, device=device).reshape(o0, r1, i0).permute(0, 2, 1).contiguous()
        if is_final_sweep:
            c0 = quantize_dequantize_groupwise(c0, bits=core_bits)
        else:
            Q, R = torch.linalg.qr(c0.reshape(o0 * i0, r1))
            c0 = Q.reshape(o0, i0, -1)
            c1 = torch.einsum("ab,bcjd->acjd", R, c1)

        # Site 1: c1 (r1, o1, i1, r2)
        r1, _, _, r2 = c1.shape
        Z1 = torch.einsum("aib,dekf,fgl,ijklr->aegrbdj", c0, c2, c3, X_tilde).reshape(-1, r1 * r2 * i1)
        Y1 = Y_tilde.permute(0, 2, 3, 4, 1).reshape(-1, o1)
        c1 = local_ridge_solve(Z1, Y1, o1, r1 * r2 * i1, ridge=ridge, device=device).reshape(o1, r1, r2, i1).permute(1, 0, 3, 2).contiguous()
        if is_final_sweep:
            c1 = quantize_dequantize_groupwise(c1, bits=core_bits)
        else:
            Q, R = torch.linalg.qr(c1.reshape(r1 * o1 * i1, r2))
            c1 = Q.reshape(r1, o1, i1, -1)
            c2 = torch.einsum("ab,bcjd->acjd", R, c2)

        # Site 2: c2 (r2, o2, i2, r3)
        r2, _, _, r3 = c2.shape
        Z2 = torch.einsum("aib,bcjd,fgl,ijklr->acgrdfk", c0, c1, c3, X_tilde).reshape(-1, r2 * r3 * i2)
        Y2 = Y_tilde.permute(0, 1, 3, 4, 2).reshape(-1, o2)
        c2 = local_ridge_solve(Z2, Y2, o2, r2 * r3 * i2, ridge=ridge, device=device).reshape(o2, r2, r3, i2).permute(1, 0, 3, 2).contiguous()
        if is_final_sweep:
            c2 = quantize_dequantize_groupwise(c2, bits=core_bits)
        else:
            Q, R = torch.linalg.qr(c2.reshape(r2 * o2 * i2, r3))
            c2 = Q.reshape(r2, o2, i2, -1)
            c3 = torch.einsum("ab,bcj->acj", R, c3)

        # Site 3: c3 (r3, o3, i3)
        r3 = c3.shape[0]
        Z3 = torch.einsum("aib,bcjd,dekf,ijklr->acerfl", c0, c1, c2, X_tilde).reshape(-1, r3 * i3)
        Y3 = Y_tilde.permute(0, 1, 2, 4, 3).reshape(-1, o3)
        c3 = local_ridge_solve(Z3, Y3, o3, r3 * i3, ridge=ridge, device=device).reshape(o3, r3, i3).permute(1, 0, 2).contiguous()
        if is_final_sweep:
            c3 = quantize_dequantize_groupwise(c3, bits=core_bits)

    final_cores = [c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]
    chosen_ranks = [1, c1.shape[0], c2.shape[0], c3.shape[0], 1]
    return final_cores, S_sparse, chosen_ranks
