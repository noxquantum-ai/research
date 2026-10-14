"""Nox Entanglement Compiler (NEC): 4-step closed-form MPO solver with zero backpropagation."""

import torch
from nox.utils import quantize_dequantize_groupwise, quantize_4bit_per_channel


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


def nox_stabilized_decompose(
    W,
    X_calib,
    Y_dense,
    o_modes=(8, 8, 8, 8),
    i_modes=(8, 8, 8, 8),
    ranks=(1, 8, 48, 8, 1),
    r_bridge=4,
    n_sweeps=2,
    prior_gamma=0.20,
    verbose=True,
):
    """
    Stabilized Nox Engine:
      1. Activation-Weighted Exact SVD Bridge (sees all 4096 columns of W, zero X^+ pseudo-inverse noise).
      2. Preserves native Llama-2 attention head locality (no cross-head scrambling).
      3. Prior-Anchored Isometric DMRG Sweep: prevents nullspace overfitting to calibration tokens.
    """
    d_out, d_in = W.shape
    device = W.device
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes

    # Step 1: Activation-Weighted Exact SVD Bridge
    ch_rms = torch.sqrt(torch.mean(X_calib ** 2, dim=1) + 1e-8)
    iso_floor = 0.05 * ch_rms.mean()
    s_scale = ch_rms + iso_floor

    if r_bridge > 0:
        W_scaled = W * s_scale.unsqueeze(0)
        U_b, S_b, V_b = torch.svd_lowrank(W_scaled, q=r_bridge + 8, niter=4)
        U_r = U_b[:, :r_bridge]
        S_r = S_b[:r_bridge]
        V_r = V_b[:, :r_bridge]

        sqrt_S = torch.sqrt(S_r).unsqueeze(0)
        A_bridge = U_r * sqrt_S
        B_bridge = (V_r * sqrt_S).T / s_scale.unsqueeze(0)

        A_q = quantize_4bit_grouped(A_bridge, group_size=64)
        B_q = quantize_4bit_grouped(B_bridge, group_size=64)
        S_lowrank = A_q @ B_q
        bridge = (A_q, B_q)

        if verbose:
            bridge_params = A_q.numel() + B_q.numel()
            bridge_kb = bridge_params * 0.5 / 1024.0
            print(f"    [Nox Step 1] Activation-Weighted Exact SVD Bridge (r={r_bridge}, {bridge_params} params = {bridge_kb:.1f} KB)", flush=True)
    else:
        S_lowrank = torch.zeros_like(W)
        bridge = None

    W_core = W - S_lowrank

    # Step 2: Low-Rank Activation Subspace Projection (Truncated to clean modes)
    Y_target = Y_dense - (S_lowrank @ X_calib)
    U_x, S_x, Vh_x = torch.linalg.svd(X_calib, full_matrices=False)
    r_cov = min(64, S_x.shape[0])
    X_red = (U_x[:, :r_cov] * S_x[:r_cov].unsqueeze(0)).reshape(i0, i1, i2, i3, r_cov)
    Y_red = (Y_target @ Vh_x[:r_cov, :].T).reshape(o0, o1, o2, o3, r_cov)

    if verbose:
        print(f"    [Nox Step 2] Preserved native head locality; projected to r={r_cov} principal activation modes", flush=True)

    # Step 3: AWT Warm-Start on W_core (Fold-in rank-1 positive tensor scaling)
    d_awt = (s_scale) ** 0.5
    bond_ranks = ranks[1:4] if len(ranks) == 5 else ranks
    cores_scaled = adaptive_tt_svd_4site(W_core * d_awt.unsqueeze(0), o_modes, i_modes, ranks=bond_ranks)

    inv_d_tensor = (1.0 / d_awt).reshape(i0, i1, i2, i3)
    log_inv = torch.log(inv_d_tensor)
    mean_total = log_inv.mean()
    factors = [
        torch.exp(log_inv.mean(dim=(1, 2, 3)) - 0.75 * mean_total),
        torch.exp(log_inv.mean(dim=(0, 2, 3)) - 0.75 * mean_total),
        torch.exp(log_inv.mean(dim=(0, 1, 3)) - 0.75 * mean_total),
        torch.exp(log_inv.mean(dim=(0, 1, 2)) - 0.75 * mean_total),
    ]
    c0 = cores_scaled[0].squeeze(0) * factors[0].view(1, i0, 1)
    c1 = cores_scaled[1] * factors[1].view(1, 1, i1, 1)
    c2 = cores_scaled[2] * factors[2].view(1, 1, i2, 1)
    c3 = cores_scaled[3].squeeze(-1) * factors[3].view(1, 1, i3)

    if verbose:
        print(f"    [Nox Step 3.1] Area-Law Ranks initialized: [1, {c1.shape[0]}, {c2.shape[0]}, {c3.shape[0]}, 1]", flush=True)

    # Step 4: Prior-Anchored DMRG Sweeps with In-Loop 4-Bit Quantization
    for sweep in range(n_sweeps):
        is_final = (sweep == n_sweeps - 1)
        if verbose:
            mode_str = "Final Pass (in-loop 4-bit quantization)" if is_final else "QR isometric gauge-fixing"
            print(f"    [Nox Step 3.2] Starting Prior-Anchored DMRG Sweep {sweep+1}/{n_sweeps} [{mode_str}]...", flush=True)

        # Site 0: c0 is (o0, i0, r1) -> parameter order (r1, i0)
        r1 = c1.shape[0]
        Z0 = torch.einsum("bcjd,dekf,fgl,ijklr->cegrbi", c1, c2, c3, X_red).reshape(-1, r1 * i0)
        Y0 = Y_red.permute(1, 2, 3, 4, 0).reshape(-1, o0)
        c0_prior = c0.permute(2, 1, 0).reshape(r1 * i0, o0)
        c0 = solve_site_anchored(Z0, Y0, c0_prior, prior_gamma, device=device).reshape(o0, r1, i0).permute(0, 2, 1).contiguous()
        if is_final:
            c0 = quantize_4bit_grouped(c0)
        else:
            Q, R = torch.linalg.qr(c0.reshape(o0 * i0, r1))
            c0 = Q.reshape(o0, i0, -1)
            c1 = torch.einsum("ab,bcjd->acjd", R, c1)

        # Site 1: c1 is (r1, o1, i1, r2) -> parameter order (r1, r2, i1)
        r1, _, _, r2 = c1.shape
        Z1 = torch.einsum("aib,dekf,fgl,ijklr->aegrbdj", c0, c2, c3, X_red).reshape(-1, r1 * r2 * i1)
        Y1 = Y_red.permute(0, 2, 3, 4, 1).reshape(-1, o1)
        c1_prior = c1.permute(0, 3, 2, 1).reshape(r1 * r2 * i1, o1)
        c1 = solve_site_anchored(Z1, Y1, c1_prior, prior_gamma, device=device).reshape(o1, r1, r2, i1).permute(1, 0, 3, 2).contiguous()
        if is_final:
            c1 = quantize_4bit_grouped(c1)
        else:
            Q, R = torch.linalg.qr(c1.reshape(r1 * o1 * i1, r2))
            c1 = Q.reshape(r1, o1, i1, -1)
            c2 = torch.einsum("ab,bcjd->acjd", R, c2)

        # Site 2: c2 is (r2, o2, i2, r3) -> parameter order (r2, r3, i2)
        r2, _, _, r3 = c2.shape
        Z2 = torch.einsum("aib,bcjd,fgl,ijklr->acgrdfk", c0, c1, c3, X_red).reshape(-1, r2 * r3 * i2)
        Y2 = Y_red.permute(0, 1, 3, 4, 2).reshape(-1, o2)
        c2_prior = c2.permute(0, 3, 2, 1).reshape(r2 * r3 * i2, o2)
        c2 = solve_site_anchored(Z2, Y2, c2_prior, prior_gamma, device=device).reshape(o2, r2, r3, i2).permute(1, 0, 3, 2).contiguous()
        if is_final:
            c2 = quantize_4bit_grouped(c2)
        else:
            Q, R = torch.linalg.qr(c2.reshape(r2 * o2 * i2, r3))
            c2 = Q.reshape(r2, o2, i2, -1)
            c3 = torch.einsum("ab,bcj->acj", R, c3)

        # Site 3: c3 is (r3, o3, i3) -> parameter order (r3, i3)
        r3 = c3.shape[0]
        Z3 = torch.einsum("aib,bcjd,dekf,ijklr->acerfl", c0, c1, c2, X_red).reshape(-1, r3 * i3)
        Y3 = Y_red.permute(0, 1, 2, 4, 3).reshape(-1, o3)
        c3_prior = c3.permute(0, 2, 1).reshape(r3 * i3, o3)
        c3 = solve_site_anchored(Z3, Y3, c3_prior, prior_gamma, device=device).reshape(o3, r3, i3).permute(1, 0, 2).contiguous()
        if is_final:
            c3 = quantize_4bit_grouped(c3)

    final_cores = [c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]
    chosen_ranks = [1, c1.shape[0], c2.shape[0], c3.shape[0], 1]
    return final_cores, bridge, chosen_ranks


def nox_4step_compiler(
    W,
    X_comp,
    Y_dense,
    o_modes,
    i_modes,
    ranks=None,
    r_bridge=4,
    n_sweeps=2,
    prior_gamma=0.20,
    core_bits=4,
    verbose=False,
    **kwargs,
):
    """
    Convenience wrapper redirecting to nox_stabilized_decompose.
    """
    target_ranks = ranks if ranks is not None else (1, 8, 48, 8, 1)
    return nox_stabilized_decompose(
        W=W,
        X_calib=X_comp,
        Y_dense=Y_dense,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        r_bridge=r_bridge,
        n_sweeps=n_sweeps,
        prior_gamma=prior_gamma,
        verbose=verbose,
    )
