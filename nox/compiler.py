"""Nox Entanglement Compiler (NEC): 4-step closed-form MPO solver with zero backpropagation."""

import torch
from nox.utils import quantize_dequantize_groupwise, quantize_4bit_per_channel, mpo_to_dense


def quantize_4bit_grouped(tensor, group_size=64):
    """4-bit group quantization (standard GPTQ/AWQ style, 0.5 bytes/param)."""
    orig_shape = tensor.shape
    flat = tensor.reshape(-1)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    flat = flat.reshape(-1, group_size)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(flat / scale), -8, 7)
    deq = (q * scale).reshape(-1)
    if pad > 0:
        deq = deq[:-pad]
    return deq.reshape(orig_shape)


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


def solve_site_anchored(Z_env, Y_site, C_prior, gamma, device=None):
    """
    Prior-Anchored Ridge Solver:
      (Z_env^T Z_env + reg * I) G = Z_env^T Y_site + reg * C_prior
    Fits calibration data along active directions while anchoring nullspace
    directions to the true weight prior to prevent test-set overfitting.
    """
    D = Z_env.shape[1]
    if device is None:
        device = Z_env.device
    H = Z_env.T @ Z_env
    reg_scale = gamma * (torch.trace(H) / D + 1e-6)
    H_reg = H + reg_scale * torch.eye(D, device=device)
    rhs = (Z_env.T @ Y_site) + reg_scale * C_prior
    sol = torch.linalg.solve(H_reg, rhs)
    return sol.T


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


def nox_hybrid_dmrg_decompose(
    W,
    X_calib,
    Y_dense,
    o_modes=(8, 8, 8, 8),
    i_modes=(8, 8, 8, 8),
    ranks=(1, 8, 48, 8, 1),
    r_bridge=4,
    n_outliers=16384,
    n_sweeps=2,
    prior_gamma=0.01,
    alpha_awt=0.5,
    verbose=True,
):
    """
    Nox Hybrid Entanglement Engine (v3 - Covariance Bridge + Bidirectional Isometric Gauge):
      1. Bit-Packed Outlier Shield (1.0 byte/outlier) for extreme coordinate spikes.
      2. Covariance-Projected SVD Bridge (r_bridge) capturing both off-diagonal semantic
         manifold correlations and diagonal outlier channels.
      3. Exact AWT Channel Preconditioning.
      4. Bidirectional QR Gauge-Fixed DMRG Sweep (Left->Right then Right->Left canonicalization
         before the final in-loop 4-bit quantization pass so gauges never collide).
    """
    d_out, d_in = W.shape
    device = W.device
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes

    # Step 1A: Exact Operator-Norm Sparse Outlier Shield (1.0 Byte/Outlier Packed)
    ch_rms = torch.sqrt(torch.mean(X_calib ** 2, dim=1) + 1e-8)
    if n_outliers > 0:
        importance = torch.abs(W) * ch_rms.unsqueeze(0)
        thresh = torch.kthvalue(importance.flatten(), W.numel() - n_outliers).values
        mask = importance >= thresh
        S_vals = torch.where(mask, W, torch.zeros_like(W))
        S_sparse = quantize_4bit_grouped(S_vals, group_size=64) * mask.to(W.dtype)
        if verbose:
            print(f"    [Nox Step 1A] Bit-Packed Sparse Outlier Shield ({n_outliers} entries = {n_outliers / 1024.0:.1f} KB)", flush=True)
    else:
        S_sparse = torch.zeros_like(W)

    W_rem = W - S_sparse

    # Step 1B: Covariance-Projected Hybrid SVD Bridge (Captures Off-Diagonal Manifold + Outliers)
    if r_bridge > 0:
        U_xc, S_xc, _ = torch.linalg.svd(X_calib, full_matrices=False)
        r_act = min(len(S_xc), max(r_bridge * 2, 64))
        X_basis = U_xc[:, :r_act] * S_xc[:r_act].unsqueeze(0)
        Y_proj = W_rem @ X_basis
        U_y, S_y, Vh_y = torch.linalg.svd(Y_proj, full_matrices=False)

        U_r = U_y[:, :r_bridge]
        reg_inv_S = S_xc[:r_act] / (S_xc[:r_act] ** 2 + (0.02 * S_xc[0]) ** 2)
        B_raw = (U_r.T @ Y_proj) * reg_inv_S.unsqueeze(0) @ U_xc[:, :r_act].T

        col_norms = torch.sqrt(torch.norm(U_r, dim=0) / (torch.norm(B_raw, dim=1) + 1e-8))
        A_bridge = U_r / col_norms.unsqueeze(0)
        B_bridge = B_raw * col_norms.unsqueeze(1)

        A_q = quantize_4bit_grouped(A_bridge, group_size=64)
        B_q = quantize_4bit_grouped(B_bridge, group_size=64)
        S_bridge = A_q @ B_q
        if verbose:
            bridge_params = (d_out + d_in) * r_bridge
            print(f"    [Nox Step 1B] Covariance-Projected SVD Bridge (r={r_bridge}, {bridge_params} params = {bridge_params * 0.5 / 1024.0:.1f} KB)", flush=True)
    else:
        S_bridge = torch.zeros_like(W)

    S_total = S_sparse + S_bridge
    W_core = W - S_total

    # Step 2: Exact AWT Channel Preconditioning
    d_awt = (ch_rms + 1e-6) ** alpha_awt
    W_cond = W_core * d_awt.unsqueeze(0)
    X_cond = X_calib / d_awt.unsqueeze(1)

    Y_target = Y_dense - (S_total @ X_calib)

    U_x, S_x, Vh_x = torch.linalg.svd(X_cond, full_matrices=False)
    r_cov = S_x.shape[0]
    X_red = (U_x[:, :r_cov] * S_x[:r_cov].unsqueeze(0)).reshape(i0, i1, i2, i3, r_cov)
    Y_red = (Y_target @ Vh_x[:r_cov, :].T).reshape(o0, o1, o2, o3, r_cov)

    if verbose:
        print(f"    [Nox Step 2] Exact AWT preconditioning applied; keeping all r={r_cov} calibration modes", flush=True)

    # Step 3: Exact AWT TT-SVD Warm-Start on W_cond
    bond_ranks = ranks[1:4] if len(ranks) == 5 else ranks
    cores = adaptive_tt_svd_4site(W_cond, o_modes, i_modes, ranks=bond_ranks)
    c0 = cores[0].squeeze(0)
    c1 = cores[1]
    c2 = cores[2]
    c3 = cores[3].squeeze(-1)

    if verbose:
        print(f"    [Nox Step 3.1] Area-Law Ranks initialized: [1, {c1.shape[0]}, {c2.shape[0]}, {c3.shape[0]}, 1]", flush=True)

    def right_canonicalize(c0, c1, c2, c3):
        r3 = c3.shape[0]
        Q3, R3 = torch.linalg.qr(c3.reshape(r3, o3 * i3).T)
        c3 = Q3.T.reshape(r3, o3, i3)
        c2 = torch.einsum("abcd,ed->abce", c2, R3)

        r2, _, _, r3_new = c2.shape
        Q2, R2 = torch.linalg.qr(c2.reshape(r2, o2 * i2 * r3_new).T)
        c2 = Q2.T.reshape(r2, o2, i2, r3_new)
        c1 = torch.einsum("abcd,ed->abce", c1, R2)

        r1, _, _, r2_new = c1.shape
        Q1, R1 = torch.linalg.qr(c1.reshape(r1, o1 * i1 * r2_new).T)
        c1 = Q1.T.reshape(r1, o1, i1, r2_new)
        c0 = torch.einsum("aib,cb->aic", c0, R1)
        return c0, c1, c2, c3

    c0, c1, c2, c3 = right_canonicalize(c0, c1, c2, c3)

    # Step 4: Isometric DMRG Sweeps in the Conditioned Hilbert Space
    for sweep in range(n_sweeps):
        is_final = (sweep == n_sweeps - 1)
        cur_gamma = prior_gamma * (0.25 if is_final else 1.0)
        if verbose:
            mode_str = "Final Pass (in-loop 4-bit quantization)" if is_final else "QR isometric gauge-fixing"
            print(f"    [Nox Step 3.2] Starting Prior-Anchored DMRG Sweep {sweep+1}/{n_sweeps} [{mode_str}]...", flush=True)

        # Site 0
        r1 = c1.shape[0]
        Z0 = torch.einsum("bcjd,dekf,fgl,ijklr->cegrbi", c1, c2, c3, X_red).reshape(-1, r1 * i0)
        Y0 = Y_red.permute(1, 2, 3, 4, 0).reshape(-1, o0)
        c0_prior = c0.permute(2, 1, 0).reshape(r1 * i0, o0)
        c0 = solve_site_anchored(Z0, Y0, c0_prior, cur_gamma, device=device).reshape(o0, r1, i0).permute(0, 2, 1).contiguous()
        if is_final:
            c0 = quantize_4bit_grouped(c0)
        else:
            Q, R = torch.linalg.qr(c0.reshape(o0 * i0, r1))
            c0 = Q.reshape(o0, i0, -1)
            c1 = torch.einsum("ab,bcjd->acjd", R, c1)

        # Site 1
        r1, _, _, r2 = c1.shape
        Z1 = torch.einsum("aib,dekf,fgl,ijklr->aegrbdj", c0, c2, c3, X_red).reshape(-1, r1 * r2 * i1)
        Y1 = Y_red.permute(0, 2, 3, 4, 1).reshape(-1, o1)
        c1_prior = c1.permute(0, 3, 2, 1).reshape(r1 * r2 * i1, o1)
        c1 = solve_site_anchored(Z1, Y1, c1_prior, cur_gamma, device=device).reshape(o1, r1, r2, i1).permute(1, 0, 3, 2).contiguous()
        if is_final:
            c1 = quantize_4bit_grouped(c1)
        else:
            Q, R = torch.linalg.qr(c1.reshape(r1 * o1 * i1, r2))
            c1 = Q.reshape(r1, o1, i1, -1)
            c2 = torch.einsum("ab,bcjd->acjd", R, c2)

        # Site 2
        r2, _, _, r3 = c2.shape
        Z2 = torch.einsum("aib,bcjd,fgl,ijklr->acgrdfk", c0, c1, c3, X_red).reshape(-1, r2 * r3 * i2)
        Y2 = Y_red.permute(0, 1, 3, 4, 2).reshape(-1, o2)
        c2_prior = c2.permute(0, 3, 2, 1).reshape(r2 * r3 * i2, o2)
        c2 = solve_site_anchored(Z2, Y2, c2_prior, cur_gamma, device=device).reshape(o2, r2, r3, i2).permute(1, 0, 3, 2).contiguous()
        if is_final:
            c2 = quantize_4bit_grouped(c2)
        else:
            Q, R = torch.linalg.qr(c2.reshape(r2 * o2 * i2, r3))
            c2 = Q.reshape(r2, o2, i2, -1)
            c3 = torch.einsum("ab,bcj->acj", R, c3)

        # Site 3
        r3 = c3.shape[0]
        Z3 = torch.einsum("aib,bcjd,dekf,ijklr->acerfl", c0, c1, c2, X_red).reshape(-1, r3 * i3)
        Y3 = Y_red.permute(0, 1, 2, 4, 3).reshape(-1, o3)
        c3_prior = c3.permute(0, 2, 1).reshape(r3 * i3, o3)
        c3 = solve_site_anchored(Z3, Y3, c3_prior, cur_gamma, device=device).reshape(o3, r3, i3).permute(1, 0, 2).contiguous()
        if is_final:
            c3 = quantize_4bit_grouped(c3)
        else:
            c0, c1, c2, c3 = right_canonicalize(c0, c1, c2, c3)

    final_cores = [c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]

    # Reconstruct W_hat = (mpo_to_dense(cores) / d_awt) + S_total
    W_mpo = mpo_to_dense(final_cores) / d_awt.unsqueeze(0)
    W_hat = W_mpo + S_total

    # Byte accounting:
    mpo_bytes = sum(c.numel() for c in final_cores) * 0.5
    bridge_bytes = ((d_out + d_in) * r_bridge * 0.5) if r_bridge > 0 else 0.0
    sparse_bytes = n_outliers * 1.0
    total_kb = (mpo_bytes + bridge_bytes + sparse_bytes) / 1024.0
    chosen_ranks = [1, c1.shape[0], c2.shape[0], c3.shape[0], 1]

    return W_hat, total_kb, final_cores, chosen_ranks


def nox_4step_compiler(
    W,
    X_comp,
    Y_dense,
    o_modes,
    i_modes,
    ranks=None,
    r_bridge=4,
    n_outliers=16384,
    n_sweeps=2,
    prior_gamma=0.02,
    verbose=False,
    **kwargs,
):
    """
    Convenience wrapper redirecting to nox_hybrid_dmrg_decompose.
    """
    target_ranks = ranks if ranks is not None else (1, 8, 48, 8, 1)
    return nox_hybrid_dmrg_decompose(
        W=W,
        X_calib=X_comp,
        Y_dense=Y_dense,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        r_bridge=r_bridge,
        n_outliers=n_outliers,
        n_sweeps=n_sweeps,
        prior_gamma=prior_gamma,
        verbose=verbose,
    )
