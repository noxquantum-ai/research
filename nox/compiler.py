"""Nox Entanglement Compiler (NEC): 4-step closed-form MPO solver with zero backpropagation."""

import torch
from nox.utils import quantize_dequantize_groupwise, quantize_4bit_per_channel


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

    if ranks is not None and len(ranks) == 5:
        ranks = ranks[1:4]

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


def nox_extreme_800x_decompose(
    W,
    X_comp,
    Y_dense,
    o_modes=(8, 8, 8, 8),
    i_modes=(8, 8, 8, 8),
    ranks=(1, 8, 48, 8, 1),
    r_bridge=4,
    n_sweeps=3,
    core_bits=4,
    ridge=1e-4,
    verbose=True,
):
    """
    Nox Extreme ~800x Closed-Form Zero-Backprop Decomposition:
      - Closed-Form Activation-Projected Principal Outlier Bridge (r_bridge)
      - Entanglement-Localizing Channel Permutation (sort by energy)
      - Asymmetric Area-Law MPO Cores (ranks e.g. [1, 8, 48, 8, 1])
      - QR-Gauge DMRG sweeps with in-loop quantization error absorption
      
    Total Footprint at 4-bit (ranks=[1, 8, 48, 8, 1], r_bridge=4 on 4096 x 4096):
      - Asymmetric Area-Law MPO Cores: 50,176 params = 24.5 KB
      - Closed-Form Activation-Subspace Bridge: 32,768 params = 16.0 KB
      - Total Size: 40.5 KB (809.1x compression from 32,768 KB FP16)
    """
    d_out, d_in = W.shape
    device = W.device

    # Step 1: Closed-Form Activation-Projected Principal Outlier Bridge
    U_x, S_x, Vh_x = torch.linalg.svd(X_comp, full_matrices=False)
    r_cov = min(128, S_x.shape[0])
    X_red_flat = U_x[:, :r_cov] * S_x[:r_cov].unsqueeze(0)
    Y_red_flat = Y_dense @ Vh_x[:r_cov, :].T

    if r_bridge > 0:
        U_y, S_y, Vh_y = torch.linalg.svd(Y_red_flat, full_matrices=False)
        r_b = min(r_bridge, S_y.shape[0])
        A_bridge = U_y[:, :r_b] * S_y[:r_b].unsqueeze(0)
        B_bridge = Vh_y[:r_b, :] @ (
            (1.0 / (S_x[:r_cov] + 1e-5)).unsqueeze(1) * U_x[:, :r_cov].T
        )
        A_q = quantize_4bit_per_channel(A_bridge, dim=0, bits=core_bits)
        B_q = quantize_4bit_per_channel(B_bridge, dim=1, bits=core_bits)
        S_lowrank = A_q @ B_q
        bridge = (A_q, B_q)
        if verbose:
            bridge_params = A_q.numel() + B_q.numel()
            bridge_kb = bridge_params * (core_bits / 8.0) / 1024.0
            print(f"    [Nox Step 1] Constructed rank-{r_b} Activation Bridge ({bridge_params} params = {bridge_kb:.1f} KB at {core_bits}-bit)", flush=True)
    else:
        S_lowrank = torch.zeros_like(W)
        bridge = None

    W_core = W - S_lowrank
    Y_target_red = Y_red_flat - (S_lowrank @ X_red_flat)

    # Step 2: Entanglement-Localizing Channel Permutation
    in_energy = torch.norm(X_red_flat, dim=1) * torch.norm(W_core, dim=0)
    out_energy = torch.norm(Y_target_red, dim=1)
    perm_in = torch.argsort(in_energy, descending=True)
    perm_out = torch.argsort(out_energy, descending=True)
    inv_perm_in = torch.argsort(perm_in)
    inv_perm_out = torch.argsort(perm_out)
    perm_info = (perm_in, perm_out, inv_perm_in, inv_perm_out)

    W_perm = W_core[perm_out][:, perm_in]
    X_perm = X_red_flat[perm_in, :]
    Y_perm = Y_target_red[perm_out, :]

    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes
    X_tilde = X_perm.reshape(i0, i1, i2, i3, r_cov)
    Y_tilde = Y_perm.reshape(o0, o1, o2, o3, r_cov)

    if verbose:
        print(f"    [Nox Step 2] Channel permutation applied (top-energy channels localized into tensor index 0)", flush=True)

    # Step 3.1: AWT Warm-Start + Dynamic / Asymmetric Area-Law Bond Sizing
    act_l2 = torch.norm(X_perm, p=2, dim=1) + 1e-8
    d_scale = (act_l2 / (r_cov ** 0.5)) ** 0.5

    bond_ranks = ranks[1:4] if len(ranks) == 5 else ranks
    cores = adaptive_tt_svd_4site(
        W_perm * d_scale.unsqueeze(0),
        o_modes,
        i_modes,
        ranks=bond_ranks,
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

    if verbose:
        print(f"    [Nox Step 3.1] Asymmetric Area-Law Ranks initialized: [1, {c1.shape[0]}, {c2.shape[0]}, {c3.shape[0]}, 1]", flush=True)

    # Step 3.2 & Step 4: DMRG Sweeps with QR Gauge Fixing and Final-Pass Quantization
    for sweep in range(n_sweeps):
        is_final_sweep = (sweep == n_sweeps - 1)
        if verbose:
            sweep_mode = f"Final Pass (in-loop {core_bits}-bit quantization)" if is_final_sweep else "QR isometric gauge-fixing"
            print(f"    [Nox Step 3.2] Starting DMRG Sweep {sweep+1}/{n_sweeps} [{sweep_mode}]...", flush=True)

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
    return final_cores, bridge, perm_info, chosen_ranks


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
    r_bridge=4,
    use_bridge=True,
    use_channel_perm=True,
    outlier_pct=0.005,
    r_subspace=128,
    n_sweeps=3,
    core_bits=4,
    ridge=1e-4,
    verbose=False,
):
    """
    Unified 4-Step Nox Entanglement Compiler:
      - When use_bridge=True: uses Activation-Projected Low-Rank Bridge + Channel Permutation
      - When use_bridge=False: falls back to coordinate CSR buffer S for legacy tests
    """
    if use_bridge:
        target_ranks = ranks if ranks is not None else (1, 8, 48, 8, 1)
        return nox_extreme_800x_decompose(
            W=W,
            X_comp=X_comp,
            Y_dense=Y_dense,
            o_modes=o_modes,
            i_modes=i_modes,
            ranks=target_ranks,
            r_bridge=r_bridge,
            n_sweeps=n_sweeps,
            core_bits=core_bits,
            ridge=ridge,
            verbose=verbose,
        )

    # Legacy coordinate-sparse fallback
    d_out, d_in = W.shape
    device = W.device
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
    U_x, S_x, Vh_x = torch.linalg.svd(X_comp, full_matrices=False)
    r = min(r_subspace, S_x.shape[0])
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes
    X_tilde = (U_x[:, :r] * S_x[:r].unsqueeze(0)).reshape(i0, i1, i2, i3, r)
    Y_tilde = (Y_target @ Vh_x[:r, :].T).reshape(o0, o1, o2, o3, r)

    d_scale = (act_l2 / (X_comp.shape[1] ** 0.5)) ** 0.5
    bond_ranks = ranks[1:4] if (ranks is not None and len(ranks) == 5) else ranks
    cores = adaptive_tt_svd_4site(
        W_core * d_scale.unsqueeze(0),
        o_modes,
        i_modes,
        ranks=bond_ranks,
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

    for sweep in range(n_sweeps):
        is_final_sweep = (sweep == n_sweeps - 1)
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

        r3 = c3.shape[0]
        Z3 = torch.einsum("aib,bcjd,dekf,ijklr->acerfl", c0, c1, c2, X_tilde).reshape(-1, r3 * i3)
        Y3 = Y_tilde.permute(0, 1, 2, 4, 3).reshape(-1, o3)
        c3 = local_ridge_solve(Z3, Y3, o3, r3 * i3, ridge=ridge, device=device).reshape(o3, r3, i3).permute(1, 0, 2).contiguous()
        if is_final_sweep:
            c3 = quantize_dequantize_groupwise(c3, bits=core_bits)

    final_cores = [c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]
    chosen_ranks = [1, c1.shape[0], c2.shape[0], c3.shape[0], 1]
    return final_cores, S_sparse, chosen_ranks
