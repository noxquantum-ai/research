import gc
import time
import torch
from nox.calibration import CalibrationStatistics
from nox.utils import quantize_dequantize_groupwise, quantize_4bit_per_channel, mpo_to_dense
from nox.tensor_order import (
    IDENTITY_SITE_ORDER,
    mode_index_permutation,
    permute_projection,
    unpermute_projection,
    validate_site_order,
)
from nox.residual_codec import CODEC_TYPE, decode_groupwise


def _site1_factorized_normal_matvec(c0, x, right23, vector):
    """Apply the site-1 normal matrix without materializing its design matrix."""
    a, i, b = c0.shape
    i_x, j, k, l, r = x.shape
    d, e, g, k_r, l_r = right23.shape
    if (i_x, k_r, l_r) != (i, k, l):
        raise ValueError("Site-1 factor dimensions do not match")
    if vector.shape[0] != b * d * j:
        raise ValueError("Site-1 vector width does not match the local core")
    outputs = vector.shape[1]

    c0_matrix = c0.reshape(a * i, b)
    vector4 = vector.reshape(b, d, j, outputs)
    left = c0_matrix @ vector4.reshape(b, d * j * outputs)
    left = left.reshape(a, i, d, j, outputs)

    x_matrix = x.reshape(i * j, k * l * r)
    middle = (
        left.permute(0, 2, 4, 1, 3).contiguous()
        .reshape(a * d * outputs, i * j)
        @ x_matrix
    )
    middle = middle.reshape(a, d, outputs, k, l, r)

    right_matrix = (
        right23.permute(0, 3, 4, 1, 2).contiguous()
        .reshape(d * k * l, e * g)
    )
    projected = (
        middle.permute(0, 2, 5, 1, 3, 4).contiguous()
        .reshape(a * outputs * r, d * k * l)
        @ right_matrix
    )

    middle_adjoint = projected @ right_matrix.T
    middle_adjoint = (
        middle_adjoint.reshape(a, outputs, r, d, k, l)
        .permute(0, 3, 1, 4, 5, 2).contiguous()
        .reshape(a * d * outputs, k * l * r)
    )
    left_adjoint = middle_adjoint @ x_matrix.T
    left_adjoint = (
        left_adjoint.reshape(a, d, outputs, i, j)
        .permute(0, 3, 1, 4, 2).contiguous()
        .reshape(a * i, d * j * outputs)
    )
    result = c0_matrix.T @ left_adjoint
    return result.reshape(b * d * j, outputs)


def _site2_factorized_normal_matvec(left01, x, c3, vector):
    """Apply the site-2 normal matrix without materializing its design matrix."""
    a, c, i, j, d = left01.shape
    i_x, j_x, k, l, r = x.shape
    f, g, l_c = c3.shape
    if (i_x, j_x, l_c) != (i, j, l):
        raise ValueError("Site-2 factor dimensions do not match")
    if vector.shape[0] != d * f * k:
        raise ValueError("Site-2 vector width does not match the local core")
    outputs = vector.shape[1]

    vector4 = vector.reshape(d, f, k, outputs)
    c3_matrix = c3.reshape(f, g * l)
    right = (
        vector4.permute(0, 2, 3, 1).contiguous()
        .reshape(d * k * outputs, f)
        @ c3_matrix
    )
    right = right.reshape(d, k, outputs, g, l)

    x_matrix = (
        x.permute(2, 3, 0, 1, 4).contiguous()
        .reshape(k * l, i * j * r)
    )
    middle = (
        right.permute(0, 2, 3, 1, 4).contiguous()
        .reshape(d * outputs * g, k * l)
        @ x_matrix
    )
    middle = middle.reshape(d, outputs, g, i, j, r)

    left_matrix = (
        left01.permute(4, 2, 3, 0, 1).contiguous()
        .reshape(d * i * j, a * c)
    )
    projected = (
        middle.permute(1, 2, 5, 0, 3, 4).contiguous()
        .reshape(outputs * g * r, d * i * j)
        @ left_matrix
    )

    middle_adjoint = projected @ left_matrix.T
    middle_adjoint = (
        middle_adjoint.reshape(outputs, g, r, d, i, j)
        .permute(3, 0, 1, 4, 5, 2).contiguous()
        .reshape(d * outputs * g, i * j * r)
    )
    right_adjoint = middle_adjoint @ x_matrix.T
    right_adjoint = (
        right_adjoint.reshape(d, outputs, g, k, l)
        .permute(0, 3, 1, 2, 4).contiguous()
        .reshape(d * k * outputs, g * l)
    )
    result = right_adjoint @ c3_matrix.T
    return (
        result.reshape(d, k, outputs, f)
        .permute(0, 3, 1, 2).contiguous()
        .reshape(d * f * k, outputs)
    )


def _site1_factorized_adjoint(c0, x, right23, projected):
    """Apply the transpose of the site-1 design matrix."""
    a, i, b = c0.shape
    i_x, j, k, l, r = x.shape
    d, e, g, k_r, l_r = right23.shape
    if (i_x, k_r, l_r) != (i, k, l):
        raise ValueError("Site-1 factor dimensions do not match")
    if projected.shape[:4] != (a, e, g, r):
        raise ValueError("Site-1 projected tensor dimensions do not match")
    outputs = projected.shape[4]

    right_matrix = (
        right23.permute(0, 3, 4, 1, 2).contiguous()
        .reshape(d * k * l, e * g)
    )
    middle_adjoint = (
        projected.permute(0, 4, 3, 1, 2).contiguous()
        .reshape(a * outputs * r, e * g)
        @ right_matrix.T
    )
    middle_adjoint = (
        middle_adjoint.reshape(a, outputs, r, d, k, l)
        .permute(0, 3, 1, 4, 5, 2).contiguous()
        .reshape(a * d * outputs, k * l * r)
    )
    x_matrix = x.reshape(i * j, k * l * r)
    left_adjoint = middle_adjoint @ x_matrix.T
    left_adjoint = (
        left_adjoint.reshape(a, d, outputs, i, j)
        .permute(0, 3, 1, 4, 2).contiguous()
        .reshape(a * i, d * j * outputs)
    )
    result = c0.reshape(a * i, b).T @ left_adjoint
    return result.reshape(b * d * j, outputs)


def _site2_factorized_adjoint(left01, x, c3, projected):
    """Apply the transpose of the site-2 design matrix."""
    a, c, i, j, d = left01.shape
    i_x, j_x, k, l, r = x.shape
    f, g, l_c = c3.shape
    if (i_x, j_x, l_c) != (i, j, l):
        raise ValueError("Site-2 factor dimensions do not match")
    if projected.shape[:4] != (a, c, g, r):
        raise ValueError("Site-2 projected tensor dimensions do not match")
    outputs = projected.shape[4]

    left_matrix = (
        left01.permute(4, 2, 3, 0, 1).contiguous()
        .reshape(d * i * j, a * c)
    )
    middle_adjoint = (
        projected.permute(4, 2, 3, 0, 1).contiguous()
        .reshape(outputs * g * r, a * c)
        @ left_matrix.T
    )
    middle_adjoint = (
        middle_adjoint.reshape(outputs, g, r, d, i, j)
        .permute(3, 0, 1, 4, 5, 2).contiguous()
        .reshape(d * outputs * g, i * j * r)
    )
    x_matrix = (
        x.permute(2, 3, 0, 1, 4).contiguous()
        .reshape(k * l, i * j * r)
    )
    right_adjoint = middle_adjoint @ x_matrix.T
    right_adjoint = (
        right_adjoint.reshape(d, outputs, g, k, l)
        .permute(0, 3, 1, 2, 4).contiguous()
        .reshape(d * k * outputs, g * l)
    )
    result = right_adjoint @ c3.reshape(f, g * l).T
    return (
        result.reshape(d, k, outputs, f)
        .permute(0, 3, 1, 2).contiguous()
        .reshape(d * f * k, outputs)
    )


def _factorized_design_frobenius_squared(x_samples, left_gram, right_gram, sample_chunk=1024):
    """Evaluate sum x.T (left_gram kron right_gram) x in bounded memory."""
    total = torch.zeros((), device=x_samples.device, dtype=torch.float64)
    for start in range(0, x_samples.shape[0], sample_chunk):
        sample = x_samples[start:start + sample_chunk]
        weighted = torch.matmul(left_gram, sample)
        weighted = torch.matmul(weighted, right_gram)
        total.add_(torch.sum(sample * weighted))
    return total


def _site1_design_frobenius_squared(c0, x, right23):
    """Compute the exact site-1 design Frobenius norm without building it."""
    _, i, _ = c0.shape
    i_x, j, k, l, r = x.shape
    d, e, g, k_r, l_r = right23.shape
    if (i_x, k_r, l_r) != (i, k, l):
        raise ValueError("Site-1 factor dimensions do not match")
    left_gram = torch.einsum("aib,ajb->ij", c0, c0)
    right_matrix = right23.reshape(d * e * g, k * l)
    right_gram = right_matrix.T @ right_matrix
    samples = (
        x.permute(1, 4, 0, 2, 3).contiguous()
        .reshape(j * r, i, k * l)
    )
    return _factorized_design_frobenius_squared(samples, left_gram, right_gram)


def _site2_design_frobenius_squared(left01, x, c3):
    """Compute the exact site-2 design Frobenius norm without building it."""
    a, c, i, j, d = left01.shape
    i_x, j_x, k, l, r = x.shape
    f, g, l_c = c3.shape
    if (i_x, j_x, l_c) != (i, j, l):
        raise ValueError("Site-2 factor dimensions do not match")
    left_matrix = left01.permute(0, 1, 4, 2, 3).reshape(a * c * d, i * j)
    left_gram = left_matrix.T @ left_matrix
    right_matrix = c3.reshape(f * g, l)
    right_gram = right_matrix.T @ right_matrix
    samples = (
        x.permute(2, 4, 0, 1, 3).contiguous()
        .reshape(k * r, i * j, l)
    )
    return _factorized_design_frobenius_squared(samples, left_gram, right_gram)


def quantize_4bit_grouped(tensor, group_size=64, return_diagnostics=False):
    """Return 4-bit quantized/dequantized values in the input tensor's dtype.

    This simulates quantization for fitting and reconstruction. The current
    artifact writer stores these values as FP16 tensors; it does not bit-pack
    them into a half-byte representation.
    """
    orig_shape = tensor.shape
    original_numel = tensor.numel()
    flat = tensor.reshape(-1)
    pad = (group_size - (flat.numel() % group_size)) % group_size
    if pad > 0:
        flat = torch.nn.functional.pad(flat, (0, pad))
    flat = flat.reshape(-1, group_size)
    scale = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    rounded = torch.round(flat / scale)
    q = torch.clamp(rounded, -8, 7)
    deq = (q * scale).reshape(-1)
    if pad > 0:
        deq = deq[:-pad]
    deq = deq.reshape(orig_shape)
    if not return_diagnostics:
        return deq

    rounded = rounded.reshape(-1)[:original_numel]
    q = q.reshape(-1)[:original_numel]
    original = tensor.reshape(-1)
    stored_fp16 = deq.to(dtype=torch.float16)
    stored_flat = stored_fp16.reshape(-1)
    fp16_tiny = torch.finfo(torch.float16).tiny
    fp16_max = torch.finfo(torch.float16).max
    # Aggregate on-device and synchronize once per component.  The counts
    # distinguish intended 4-bit rounding-to-zero from FP16 storage underflow
    # and detect any actual clamp/overflow instead of inferring them later.
    counts = torch.stack((
        ((rounded < -8) | (rounded > 7)).sum(),
        ((original != 0) & (q == 0)).sum(),
        (q == 7).sum(),
        (q == -8).sum(),
        ((deq.reshape(-1) != 0) & (stored_flat == 0)).sum(),
        ((stored_flat != 0) & (stored_flat.abs() < fp16_tiny)).sum(),
        (deq.abs() > fp16_max).sum(),
        (~torch.isfinite(stored_fp16)).sum(),
    )).to(device="cpu", dtype=torch.int64).tolist()
    clipped, rounded_to_zero, positive_max_code, negative_min_code, underflow, subnormal, overflow, nonfinite = counts
    diagnostics = {
        "group_size": int(group_size),
        "input_coefficients": int(original_numel),
        "q4_clipped_code_count": int(clipped),
        "q4_nonzero_input_rounded_to_zero_count": int(rounded_to_zero),
        "q4_positive_max_code_count": int(positive_max_code),
        "q4_negative_min_code_count": int(negative_min_code),
        "fp16_nonzero_value_underflow_to_zero_count": int(underflow),
        "fp16_subnormal_count": int(subnormal),
        "fp16_overflow_count": int(overflow),
        "fp16_nonfinite_count": int(nonfinite),
        "input_max_abs": float(tensor.abs().max().item()),
        "dequantized_max_abs": float(deq.abs().max().item()),
        "stored_fp16_max_abs": float(stored_fp16.abs().max().item()),
    }
    return deq, diagnostics


def _low_rank_product_relative_error(a, b, target_a, target_b):
    """Compute ||AB-A0B0||_F/||A0B0||_F without materializing either product."""
    # The factors are small in their shared dimension but may be very tall or
    # wide.  All dense work below is therefore rank-by-rank.
    a_work = a.double()
    b_work = b.double()
    target_a_work = target_a.double()
    target_b_work = target_b.double()
    norm = torch.sum((a_work.T @ a_work) * (b_work @ b_work.T).T)
    target_norm = torch.sum(
        (target_a_work.T @ target_a_work)
        * (target_b_work @ target_b_work.T).T
    )
    inner = torch.trace(
        (a_work.T @ target_a_work) @ (target_b_work @ b_work.T)
    )
    squared_error = (norm + target_norm - 2.0 * inner).clamp_min(0)
    return torch.sqrt(squared_error / target_norm.clamp_min(1e-30))


@torch.no_grad()
def quantize_low_rank_bridge_operator_refit(
    a,
    b,
    *,
    group_size=64,
    iterations=4,
    return_diagnostics=False,
):
    """Q4-quantize a low-rank product using alternating operator refits.

    The historical path rounds ``a`` and ``b`` independently.  Here each
    candidate factor is first refit against the represented product ``a @ b``
    while the other Q4 factor is held fixed.  Candidate selection minimizes
    exact Frobenius operator error using only rank-sized Gram matrices.  The
    historical pair is always retained as a candidate, so this routine cannot
    knowingly return a worse pair under its stated objective.

    This is an experimental quantizer.  It does not change rank or coefficient
    count and performs no gradient optimization.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected low-rank factors A[d_out,r] and B[r,d_in]")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")

    target_a = a
    target_b = b

    def quantized(tensor):
        return quantize_4bit_grouped(tensor, group_size=group_size)

    best_raw_a = a
    best_raw_b = b
    baseline_a = quantized(a)
    baseline_b = quantized(b)
    best_a = baseline_a
    best_b = baseline_b
    baseline_error = _low_rank_product_relative_error(
        best_a, best_b, target_a, target_b,
    )
    best_error = baseline_error
    selected_step = "independent"
    current_a = best_a
    current_b = best_b
    rank = int(a.shape[1])
    identity = torch.eye(rank, device=a.device, dtype=torch.float32)

    for iteration in range(iterations):
        # argmin_B ||current_a B - target_a target_b||_F
        left = current_a.float()
        normal = left.T @ left
        ridge = torch.trace(normal).div(max(rank, 1)).mul(1e-8).clamp_min(1e-12)
        try:
            transfer = torch.linalg.solve(
                normal + ridge * identity,
                left.T @ target_a.float(),
            )
            raw_b = (transfer @ target_b.float()).to(dtype=b.dtype)
        except torch.linalg.LinAlgError:
            raw_b = None
        if raw_b is not None and torch.isfinite(raw_b).all():
            candidate_b = quantized(raw_b)
            candidate_error = _low_rank_product_relative_error(
                current_a, candidate_b, target_a, target_b,
            )
            if torch.isfinite(candidate_error) and candidate_error < best_error:
                best_raw_a = current_a
                best_raw_b = raw_b
                best_a = current_a
                best_b = candidate_b
                best_error = candidate_error
                selected_step = f"iteration_{iteration + 1}_right"
            current_b = candidate_b

        # argmin_A ||A current_b - target_a target_b||_F
        right = current_b.float()
        normal = right @ right.T
        ridge = torch.trace(normal).div(max(rank, 1)).mul(1e-8).clamp_min(1e-12)
        try:
            transfer = (target_b.float() @ right.T) @ torch.linalg.solve(
                normal + ridge * identity,
                identity,
            )
            raw_a = (target_a.float() @ transfer).to(dtype=a.dtype)
        except torch.linalg.LinAlgError:
            raw_a = None
        if raw_a is not None and torch.isfinite(raw_a).all():
            candidate_a = quantized(raw_a)
            candidate_error = _low_rank_product_relative_error(
                candidate_a, current_b, target_a, target_b,
            )
            if torch.isfinite(candidate_error) and candidate_error < best_error:
                best_raw_a = raw_a
                best_raw_b = current_b
                best_a = candidate_a
                best_b = current_b
                best_error = candidate_error
                selected_step = f"iteration_{iteration + 1}_left"
            current_a = candidate_a

    if not return_diagnostics:
        return best_a, best_b

    # Requantize the selected pre-rounding tensors once to obtain diagnostics
    # for the factors actually returned.  When a fixed factor came from a prior
    # iteration, it is already on the Q4 grid and requantization is idempotent.
    best_a, a_diagnostics = quantize_4bit_grouped(
        best_raw_a, group_size=group_size, return_diagnostics=True,
    )
    best_b, b_diagnostics = quantize_4bit_grouped(
        best_raw_b, group_size=group_size, return_diagnostics=True,
    )
    final_error = _low_rank_product_relative_error(
        best_a, best_b, target_a, target_b,
    )
    if not torch.isfinite(final_error) or final_error > baseline_error:
        best_a, a_diagnostics = quantize_4bit_grouped(
            a, group_size=group_size, return_diagnostics=True,
        )
        best_b, b_diagnostics = quantize_4bit_grouped(
            b, group_size=group_size, return_diagnostics=True,
        )
        final_error = baseline_error
        selected_step = "independent_fallback"
    return best_a, best_b, a_diagnostics, b_diagnostics, {
        "objective": "relative Frobenius error of the represented bridge operator",
        "iterations": int(iterations),
        "selected_step": selected_step,
        "independent_q4_relative_operator_error": float(baseline_error.item()),
        "selected_q4_relative_operator_error": float(final_error.item()),
    }


def augment_diagonal_weight_prior(X, Y, W, channel_scale, strength):
    """Add N*strength*||(W_hat-W) diag(channel_scale)||_F^2 to fit loss.

    These deterministic basis columns encode a dense-weight prior, not extra
    corpus examples. They never use validation activations or labels.
    """
    if strength < 0:
        raise ValueError("Weight prior strength must be nonnegative")
    if strength == 0:
        return X, Y
    scale = channel_scale * (X.shape[1] * strength) ** 0.5
    return (torch.cat((X, torch.diag(scale)), dim=1),
            torch.cat((Y, W * scale.unsqueeze(0)), dim=1))


def estimate_artifact_tensor_bytes(cores, A_bridge, B_bridge, d_awt, sparse_count):
    """Estimate stored tensor payload bytes using artifact dtypes.

    Components are FP16, except each sparse value has two int32 coordinates.
    This excludes torch.save container overhead and layer-level norm tensors.
    """
    bytes_fp16 = 2
    total = sum(core.numel() * bytes_fp16 for core in cores)
    if A_bridge is not None:
        total += A_bridge.numel() * bytes_fp16
    if B_bridge is not None:
        total += B_bridge.numel() * bytes_fp16
    total += d_awt.numel() * bytes_fp16
    total += int(sparse_count) * (2 * 4 + bytes_fp16)
    return int(total)


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
            r_next = min(ranks[m], S.shape[0])
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
    Uses Symmetric Positive-Definite Cholesky with automatic fallback.
    """
    D = Z_env.shape[1]
    if device is None:
        device = Z_env.device
    H = Z_env.T @ Z_env
    rhs = Z_env.T @ Y_site
    reg_scale = gamma * (torch.trace(H) / D + 1e-6)
    H.diagonal().add_(reg_scale)
    rhs.add_(reg_scale * C_prior)
    try:
        L, info = torch.linalg.cholesky_ex(H)
        if info == 0:
            sol = torch.cholesky_solve(rhs, L)
        else:
            sol = torch.linalg.solve(H, rhs)
    except Exception:
        sol = torch.linalg.solve(H, rhs)
    del H, rhs
    return sol.T


def solve_accumulated_system(
    H,
    rhs,
    C_prior,
    gamma,
    D,
    use_cg_threshold=4096,
    n_cg_iters=6,
    return_diagnostics=False,
):
    """
    Solves (H + reg*I) sol = rhs + reg*C_prior for chunked DMRG site systems.
    When D <= 4096: Uses SPD Cholesky factorization (cholesky_ex + cholesky_solve).
    When D > 4096: Uses Warm-Started Conjugate Gradient on H initialized at sol_0 = C_prior.
    Eliminates 12 GB memory leaks, avoids OOM, and takes 260x fewer FLOPs!
    """
    if C_prior.dtype != H.dtype or C_prior.device != H.device:
        C_prior = C_prior.to(device=H.device, dtype=H.dtype)
    reg_scale = gamma * (torch.trace(H) / D + 1e-6)
    H.diagonal().add_(reg_scale)
    rhs.add_(reg_scale * C_prior)

    solver = "cholesky"
    iterations = 0
    if D <= use_cg_threshold:
        L, info = torch.linalg.cholesky_ex(H)
        if info == 0:
            sol = torch.cholesky_solve(rhs, L)
        else:
            # Activation-conditioned SwiGLU contexts can be effectively low
            # rank. The old fallback retried the same singular direct solve.
            # Escalating diagonal jitter keeps the solve on device and only
            # changes cases where the regularized normal equation is not SPD.
            diagonal_scale = H.diagonal().abs().mean().clamp_min(1e-6)
            sol = None
            for relative_jitter in (1e-6, 1e-4, 1e-2):
                H.diagonal().add_(diagonal_scale * relative_jitter)
                L, info = torch.linalg.cholesky_ex(H)
                if info == 0:
                    sol = torch.cholesky_solve(rhs, L)
                    solver = "cholesky_with_jitter"
                    break
            if sol is None:
                sol = torch.linalg.pinv(H, hermitian=True, rtol=1e-5) @ rhs
                solver = "hermitian_pinv"
    else:
        solver = "conjugate_gradient"
        HC = H @ C_prior
        R = rhs - HC
        del HC
        P = R.clone()
        Delta = torch.zeros_like(C_prior)
        rs_old = torch.sum(R * R, dim=0, keepdim=True)

        for iteration in range(n_cg_iters):
            HP = H @ P
            pAp = torch.sum(P * HP, dim=0, keepdim=True).clamp(min=1e-12)
            alpha = rs_old / pAp
            Delta.addcmul_(alpha, P)
            R.addcmul_(-alpha, HP)
            del HP
            rs_new = torch.sum(R * R, dim=0, keepdim=True)
            iterations = iteration + 1
            if torch.max(rs_new).item() < 1e-8:
                break
            beta = rs_new / rs_old.clamp(min=1e-12)
            P = R + beta * P
            rs_old = rs_new
        del R, P
        sol = C_prior + Delta
        del Delta

    diagnostics = None
    if return_diagnostics:
        normal_residual = H @ sol - rhs
        diagnostics = {
            "solver": solver,
            "iterations": int(iterations),
            "regularization_scale": float(reg_scale.item()),
            "relative_normal_residual": float(
                torch.linalg.vector_norm(normal_residual)
                .div(torch.linalg.vector_norm(rhs).clamp_min(1e-30))
                .item()
            ),
        }
        del normal_residual
    del H, rhs
    result = sol.T
    if return_diagnostics:
        return result, diagnostics
    return result


def reconstruct_from_artifact(artifact, device="cpu", dtype=torch.float32):
    """
    Reconstruct a dense matrix from a legacy NEC, direct low-bit, or hybrid
    NEC-plus-residual artifact.
    """
    if artifact.get("encoding") == "nox_int4_csr_v1":
        # The packed representation is the on-disk form of the quantum-inspired
        # MPO/sparse/bridge operator.  Decode it at the reconstruction boundary
        # so packed NEC bases can participate in composite codecs without an
        # exported FP16 round-trip checkpoint.
        from nox.packing import unpack_artifact
        artifact = unpack_artifact(artifact)

    codec_type = artifact.get("codec_type")
    if codec_type == "nec_packed_operator_schmidt_v1":
        from nox.packed_schmidt import reconstruct_packed_schmidt
        return reconstruct_packed_schmidt(artifact, device=device, dtype=dtype)
    if codec_type == "nec_operator_schmidt_v1":
        from nox.operator_schmidt import reconstruct_operator_schmidt
        return reconstruct_operator_schmidt(artifact, device=device, dtype=dtype)
    if codec_type == "nec_activation_schmidt_correction_v1":
        from nox.schmidt_correction import reconstruct_schmidt_correction
        return reconstruct_schmidt_correction(artifact, device=device, dtype=dtype)
    if codec_type == CODEC_TYPE:
        return decode_groupwise(artifact, device=device, dtype=dtype)
    if codec_type == "nec_hybrid_residual_v1":
        base = reconstruct_from_artifact(artifact["base_artifact"], device=device, dtype=dtype)
        residual = decode_groupwise(artifact["residual_codec"], device=device, dtype=dtype)
        if tuple(base.shape) != tuple(residual.shape):
            raise ValueError("NEC base and packed residual shapes differ")
        return base + residual
    if codec_type not in (None, "nec_mpo_v1"):
        raise ValueError(f"Unsupported compressed artifact codec: {codec_type!r}")

    cores = [c.to(device=device, dtype=dtype) for c in artifact["cores"]]
    d_awt = artifact["d_awt"].to(device=device, dtype=dtype)
    if len(cores) == 1:
        # A single-site operator is a valid degenerate MPO and is useful for
        # compact composite-codec controls and tiny layer tests.
        W_rec = cores[0].squeeze(0).squeeze(-1) / d_awt.unsqueeze(0)
    else:
        W_rec = mpo_to_dense(cores) / d_awt.unsqueeze(0)
    if artifact.get("A_bridge") is not None and artifact.get("B_bridge") is not None:
        A_b = artifact["A_bridge"].to(device=device, dtype=dtype)
        B_b = artifact["B_bridge"].to(device=device, dtype=dtype)
        W_rec.addmm_(A_b, B_b)
    if artifact.get("sparse_indices") is not None and artifact.get("sparse_values") is not None:
        idx = artifact["sparse_indices"]
        val = artifact["sparse_values"].to(device=device, dtype=dtype)
        W_rec[idx[:, 0].long(), idx[:, 1].long()] += val
    site_order = artifact.get("site_order")
    if site_order is not None and tuple(site_order) != IDENTITY_SITE_ORDER:
        out_modes = tuple(artifact["original_o_modes"])
        in_modes = tuple(artifact["original_i_modes"])
        out_perm = mode_index_permutation(out_modes, site_order, device=device)
        in_perm = mode_index_permutation(in_modes, site_order, device=device)
        W_rec = unpermute_projection(W_rec, out_perm, in_perm)
    return W_rec


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


def _left_singular_system_from_gram(xx, sample_count, output_dtype):
    """Recover the nonzero left singular system of X from ``xx = X X^T``.

    This is algebraically equivalent to an SVD in exact arithmetic.  In finite
    precision it is less robust because forming a Gram matrix squares the
    condition number.  Accumulators can therefore retain float64 statistics;
    the returned reduced calibration coordinates are converted to the model
    dtype before the DMRG stage.
    """
    symmetric = (xx + xx.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)
    maximum_modes = min(int(sample_count), int(xx.shape[0]))
    eigenvalues = eigenvalues[:maximum_modes]
    eigenvectors = eigenvectors[:, :maximum_modes]
    largest = eigenvalues[0].clamp_min(0)
    # Apply the conventional SVD rank threshold in singular-value units.
    # Since these are squared singular values, the corresponding eigenvalue
    # threshold is squared as well.  Negative Gram eigenvalues are never
    # promoted to valid directions.
    singular_tolerance = torch.sqrt(largest) * torch.finfo(eigenvalues.dtype).eps * max(
        int(sample_count), int(xx.shape[0]),
    )
    tolerance = singular_tolerance.square()
    negative_eigenvalues = int(torch.count_nonzero(eigenvalues < 0).item())
    positive_modes = int(torch.count_nonzero(eigenvalues > 0).item())
    conventional_numerical_rank = int(torch.count_nonzero(eigenvalues > tolerance).item())
    below_conventional_cutoff = positive_modes - conventional_numerical_rank
    # The reference SVD keeps min(N,d) modes; silently applying the usual SVD
    # rank heuristic here would change that algorithm (notably, eps*N is
    # 0.0078125 for FP32 and N=65536). Retain every expected positive mode and
    # fail closed when Gram roundoff makes that impossible. A stable TSQR/SVD
    # path, not silent truncation, is required for such a covariance.
    if positive_modes < maximum_modes:
        raise FloatingPointError(
            "Calibration Gram factorization lost positive modes: "
            f"expected {maximum_modes}, observed {positive_modes}, "
            f"negative={negative_eigenvalues}, dtype={xx.dtype}. "
            "Retry with float64 statistics or the full-matrix SVD backend."
        )
    retained = maximum_modes
    singular_values = torch.sqrt(eigenvalues)
    return (
        eigenvectors.to(dtype=output_dtype),
        singular_values.to(dtype=output_dtype),
        {
            "maximum_modes": maximum_modes,
            "numerical_rank": retained,
            "conventional_numerical_rank": conventional_numerical_rank,
            "positive_modes_below_conventional_cutoff": below_conventional_cutoff,
            "eigenvalue_tolerance": float(tolerance.item()),
            "singular_value_tolerance": float(singular_tolerance.item()),
            "largest_eigenvalue": float(largest.item()),
            "negative_eigenvalues_in_candidate_range": negative_eigenvalues,
            "statistics_dtype": str(xx.dtype),
        },
    )


@torch.no_grad()
def nox_hybrid_dmrg_decompose(
    W,
    X_calib,
    Y_dense,
    o_modes=(8, 8, 8, 8),
    i_modes=(8, 8, 8, 8),
    ranks=(1, 24, 64, 24, 1),
    r_bridge=64,
    n_outliers=262144,
    n_sweeps=2,
    prior_gamma=0.01,
    alpha_awt=0.5,
    chunk_size=32,
    return_artifacts=False,
    verbose=True,
    quantize_components=True,
    diagnostic_callback=None,
    core_diagnostic_callback=None,
    diagonal_prior_strength=0.0,
    site_order=IDENTITY_SITE_ORDER,
    calibration_statistics=None,
    statistics_factorization_cache=None,
    quantization_schedule="in_sweep",
    bridge_quantization_mode="independent_q4",
    bridge_refit_iterations=4,
    solver_diagnostic_callback=None,
    cg_threshold=4096,
    cg_iterations=6,
):
    """
    Nox Hybrid Entanglement Engine (v3 - Covariance Bridge + Bidirectional Isometric Gauge):
      1. Sparse outlier shield; values are stored as FP16 with two int32
         coordinates per nonzero in the current artifact format.
      2. Covariance-Projected SVD Bridge (r_bridge) capturing both off-diagonal semantic
         manifold correlations and diagonal outlier channels.
      3. Exact AWT Channel Preconditioning.
      4. Bidirectional QR Gauge-Fixed DMRG Sweep (Left->Right then Right->Left canonicalization
         before the final in-loop 4-bit quantization pass so gauges never collide).
    """
    if calibration_statistics is not None and not isinstance(
        calibration_statistics, CalibrationStatistics,
    ):
        raise TypeError("calibration_statistics must be CalibrationStatistics")
    if calibration_statistics is None:
        if X_calib is None or Y_dense is None:
            raise ValueError("X_calib and Y_dense are required without calibration statistics")
        if X_calib.ndim != 2 or Y_dense.ndim != 2 or X_calib.shape[1] != Y_dense.shape[1]:
            raise ValueError("Expected X_calib[d_in,N] and Y_dense[d_out,N] with matching N")
    elif X_calib is not None or Y_dense is not None:
        raise ValueError("Pass either raw calibration tensors or statistics, not both")

    site_order = validate_site_order(site_order)
    if site_order != IDENTITY_SITE_ORDER:
        if calibration_statistics is None:
            W_ordered, X_ordered, Y_ordered, o_ordered, i_ordered, out_perm, in_perm = permute_projection(
                W, X_calib, Y_dense, o_modes, i_modes, site_order,
            )
            ordered_statistics = None
        else:
            out_perm = mode_index_permutation(o_modes, site_order, device=W.device)
            in_perm = mode_index_permutation(i_modes, site_order, device=W.device)
            W_ordered = W.index_select(0, out_perm).index_select(1, in_perm)
            X_ordered = Y_ordered = None
            o_ordered = tuple(o_modes[axis] for axis in site_order)
            i_ordered = tuple(i_modes[axis] for axis in site_order)
            ordered_statistics = calibration_statistics.permuted(
                out_perm.to(calibration_statistics.yx.device),
                in_perm.to(calibration_statistics.xx.device),
            )

        def restore_diagnostic(stage, value):
            if diagnostic_callback is not None:
                diagnostic_callback(stage, unpermute_projection(value, out_perm, in_perm))

        result = nox_hybrid_dmrg_decompose(
            W=W_ordered,
            X_calib=X_ordered,
            Y_dense=Y_ordered,
            o_modes=o_ordered,
            i_modes=i_ordered,
            ranks=ranks,
            r_bridge=r_bridge,
            n_outliers=n_outliers,
            n_sweeps=n_sweeps,
            prior_gamma=prior_gamma,
            alpha_awt=alpha_awt,
            chunk_size=chunk_size,
            return_artifacts=return_artifacts,
            verbose=verbose,
            quantize_components=quantize_components,
            diagnostic_callback=restore_diagnostic if diagnostic_callback is not None else None,
            core_diagnostic_callback=core_diagnostic_callback,
            diagonal_prior_strength=diagonal_prior_strength,
            site_order=IDENTITY_SITE_ORDER,
            calibration_statistics=ordered_statistics,
            statistics_factorization_cache=statistics_factorization_cache,
            quantization_schedule=quantization_schedule,
            bridge_quantization_mode=bridge_quantization_mode,
            bridge_refit_iterations=bridge_refit_iterations,
            solver_diagnostic_callback=solver_diagnostic_callback,
            cg_threshold=cg_threshold,
            cg_iterations=cg_iterations,
        )
        if return_artifacts:
            W_hat, total_kb, cores, chosen_ranks, artifact = result
            artifact["site_order"] = site_order
            artifact["original_o_modes"] = tuple(o_modes)
            artifact["original_i_modes"] = tuple(i_modes)
            return (
                unpermute_projection(W_hat, out_perm, in_perm),
                total_kb,
                cores,
                chosen_ranks,
                artifact,
            )
        W_hat, total_kb, cores, chosen_ranks = result
        return unpermute_projection(W_hat, out_perm, in_perm), total_kb, cores, chosen_ranks

    d_out, d_in = W.shape
    if calibration_statistics is not None:
        if (
            calibration_statistics.input_dimension != d_in
            or calibration_statistics.output_dimension != d_out
        ):
            raise ValueError("Calibration statistics do not match W dimensions")
    if diagonal_prior_strength < 0:
        raise ValueError("Weight prior strength must be nonnegative")
    if quantization_schedule not in ("in_sweep", "post_sweep"):
        raise ValueError(
            "quantization_schedule must be 'in_sweep' or 'post_sweep'"
        )
    if bridge_quantization_mode not in (
        "independent_q4", "operator_refit_q4", "fp16_control",
    ):
        raise ValueError(
            "bridge_quantization_mode must be 'independent_q4', "
            "'operator_refit_q4', or 'fp16_control'"
        )
    if bridge_refit_iterations < 0:
        raise ValueError("bridge_refit_iterations must be nonnegative")
    if cg_threshold < 1 or cg_iterations < 1:
        raise ValueError("cg_threshold and cg_iterations must be positive")
    in_sweep_quantization = (
        quantize_components and quantization_schedule == "in_sweep"
    )
    quantize = quantize_4bit_grouped if quantize_components else lambda tensor, group_size=64: tensor
    precision_diagnostics = {}

    def quantize_component(tensor, component, group_size=64):
        if not quantize_components:
            return tensor
        quantized, diagnostics = quantize_4bit_grouped(
            tensor, group_size=group_size, return_diagnostics=True,
        )
        precision_diagnostics[component] = diagnostics
        return quantized
    device = W.device
    o0, o1, o2, o3 = o_modes
    i0, i1, i2, i3 = i_modes

    if str(device).startswith("cuda"):
        gc.collect()
        torch.cuda.empty_cache()

    statistics_diagnostics = None
    covariance = cross_covariance = None
    if calibration_statistics is not None:
        # Keep higher-precision accumulation until a factorization is needed.
        # The DMRG implementation itself is float32, matching the reference.
        covariance = calibration_statistics.xx.to(device=device)
        cross_covariance = calibration_statistics.yx.to(device=device)
        calibration_count = int(calibration_statistics.sample_count)
        ch_rms = torch.sqrt(
            covariance.diagonal().to(dtype=W.dtype) / calibration_count + 1e-8,
        )
    else:
        calibration_count = int(X_calib.shape[1])
        ch_rms = torch.sqrt(torch.mean(X_calib ** 2, dim=1) + 1e-8)

    def gram_factorization(matrix, effective_samples, cache_key):
        if statistics_factorization_cache is not None and cache_key in statistics_factorization_cache:
            return statistics_factorization_cache[cache_key]
        result = _left_singular_system_from_gram(matrix, effective_samples, W.dtype)
        if statistics_factorization_cache is not None:
            statistics_factorization_cache[cache_key] = result
        return result

    # Step 1A: Exact Operator-Norm Sparse Outlier Shield
    mask = None
    if n_outliers > 0:
        importance = torch.abs(W) * ch_rms.unsqueeze(0)
        k_idx = max(1, min(W.numel() - n_outliers, W.numel()))
        thresh = torch.kthvalue(importance.flatten(), k_idx).values
        mask = importance >= thresh
        S_vals = torch.where(mask, W, torch.zeros_like(W))
        S_sparse = quantize_component(S_vals, "sparse_shield", group_size=64) * mask.to(W.dtype)
        sparse_count = int(torch.count_nonzero(S_sparse).item())
        nz_mask = (S_sparse != 0) if return_artifacts else None
        if verbose:
            print(
                f"    [Nox Step 1A] Sparse Outlier Shield ({sparse_count} stored entries; "
                f"{sparse_count * 10 / 1024.0:.1f} KiB tensor payload)",
                flush=True,
            )
    else:
        S_sparse = torch.zeros_like(W)
        nz_mask = None
        sparse_count = 0

    W_rem = W - S_sparse
    if diagnostic_callback is not None:
        diagnostic_callback("sparse", S_sparse)

    # Step 1B: Covariance-Projected Hybrid SVD Bridge (Captures Off-Diagonal Manifold + Outliers)
    A_q = None
    B_q = None
    if r_bridge > 0:
        if calibration_statistics is None:
            U_xc, S_xc, Vh_xc = torch.linalg.svd(X_calib, full_matrices=False)
            bridge_rank_diagnostics = None
            r_act = min(len(S_xc), max(r_bridge * 2, 64))
            Y_rem = Y_dense - (S_sparse @ X_calib)
            Y_proj = Y_rem @ Vh_xc[:r_act, :].T
            del Y_rem
        else:
            U_xc, S_xc, bridge_rank_diagnostics = gram_factorization(
                covariance,
                calibration_count,
                ("raw", int(covariance.data_ptr()), calibration_count, str(covariance.dtype)),
            )
            r_act = min(len(S_xc), max(r_bridge * 2, 64))
            # If X=U Sigma V^T, then (Y-SX)V=(YX^T-SXX^T)U Sigma^-1.
            h_remaining = (
                cross_covariance
                - S_sparse.to(dtype=cross_covariance.dtype) @ covariance
            )
            Y_proj = (
                h_remaining @ U_xc[:, :r_act].to(dtype=h_remaining.dtype)
            ) / S_xc[:r_act].to(dtype=h_remaining.dtype).unsqueeze(0)
            Y_proj = Y_proj.to(dtype=W.dtype)
            del h_remaining
        U_y, S_y, Vh_y = torch.linalg.svd(Y_proj, full_matrices=False)
        del Vh_y

        realized_bridge_rank = min(r_bridge, U_y.shape[1])
        U_r = U_y[:, :realized_bridge_rank]
        reg_inv_S = S_xc[:r_act] / (S_xc[:r_act] ** 2 + (0.02 * S_xc[0]) ** 2)
        B_raw = (U_r.T @ Y_proj) * reg_inv_S.unsqueeze(0) @ U_xc[:, :r_act].T
        del U_y, Y_proj, U_xc, S_xc
        if calibration_statistics is None:
            del Vh_xc

        col_norms = torch.sqrt(torch.norm(U_r, dim=0) / (torch.norm(B_raw, dim=1) + 1e-8))
        A_bridge = U_r / col_norms.unsqueeze(0)
        B_bridge = B_raw * col_norms.unsqueeze(1)

        if not quantize_components:
            A_q = A_bridge
            B_q = B_bridge
            bridge_quantization_diagnostics = {
                "mode": "unquantized_control",
                "iterations": 0,
            }
        elif bridge_quantization_mode == "fp16_control":
            # Match the tensors that inference reconstructs from the artifact.
            # Keeping FP32 factors during the sweeps and casting only while
            # serializing would test a different operator from the saved one.
            A_q = A_bridge.to(dtype=torch.float16).to(dtype=W.dtype)
            B_q = B_bridge.to(dtype=torch.float16).to(dtype=W.dtype)
            bridge_quantization_diagnostics = {
                "mode": "fp16_control",
                "iterations": 0,
            }
        elif bridge_quantization_mode == "independent_q4":
            A_q = quantize_component(A_bridge, "bridge_A", group_size=64)
            B_q = quantize_component(B_bridge, "bridge_B", group_size=64)
            bridge_quantization_diagnostics = {
                "mode": "independent_q4",
                "iterations": 0,
            }
        else:
            (
                A_q,
                B_q,
                a_diagnostics,
                b_diagnostics,
                bridge_quantization_diagnostics,
            ) = quantize_low_rank_bridge_operator_refit(
                A_bridge,
                B_bridge,
                group_size=64,
                iterations=bridge_refit_iterations,
                return_diagnostics=True,
            )
            precision_diagnostics["bridge_A"] = a_diagnostics
            precision_diagnostics["bridge_B"] = b_diagnostics
            bridge_quantization_diagnostics["mode"] = "operator_refit_q4"
        S_bridge = A_q @ B_q
        del A_bridge, B_bridge, U_r, B_raw
        if verbose:
            bridge_params = (d_out + d_in) * realized_bridge_rank
            print(f"    [Nox Step 1B] Covariance-Projected SVD Bridge (r={realized_bridge_rank}, {bridge_params} params = {bridge_params * 0.5 / 1024.0:.1f} KB)", flush=True)
    else:
        S_bridge = torch.zeros_like(W)

    S_total = S_sparse + S_bridge
    if diagnostic_callback is not None:
        diagnostic_callback("sparse_plus_bridge", S_total)
    W_core = W - S_total

    # Step 2: Exact AWT Channel Preconditioning
    d_awt = (ch_rms + 1e-6) ** alpha_awt
    W_cond = W_core * d_awt.unsqueeze(0)
    if calibration_statistics is None:
        X_cond = X_calib / d_awt.unsqueeze(1)
        Y_target = Y_dense - (S_total @ X_calib)

        X_cond, Y_target = augment_diagonal_weight_prior(
            X_cond, Y_target, W_cond, ch_rms / d_awt, diagonal_prior_strength,
        )

        U_x, S_x, Vh_x = torch.linalg.svd(X_cond, full_matrices=False)
        r_cov = S_x.shape[0]
        X_red = (U_x * S_x.unsqueeze(0)).reshape(i0, i1, i2, i3, r_cov)
        Y_red = (Y_target @ Vh_x.T).reshape(o0, o1, o2, o3, r_cov)
        del U_x, S_x, Vh_x, Y_target, X_cond
    else:
        d_stats = d_awt.to(dtype=covariance.dtype)
        inverse_d = d_stats.reciprocal()
        conditioned_covariance = (
            covariance * inverse_d.unsqueeze(0) * inverse_d.unsqueeze(1)
        )
        target_cross_covariance = (
            cross_covariance
            - S_total.to(dtype=cross_covariance.dtype) @ covariance
        ) * inverse_d.unsqueeze(0)

        if diagonal_prior_strength > 0:
            prior_scale = (
                ch_rms.to(dtype=covariance.dtype) / d_stats
            ) * (calibration_count * diagonal_prior_strength) ** 0.5
            prior_variance = prior_scale.square()
            conditioned_covariance.diagonal().add_(prior_variance)
            target_cross_covariance.add_(
                W_cond.to(dtype=target_cross_covariance.dtype) * prior_variance.unsqueeze(0),
            )

        U_x, S_x, statistics_diagnostics = gram_factorization(
            conditioned_covariance,
            calibration_count + (d_in if diagonal_prior_strength > 0 else 0),
            (
                "conditioned",
                int(covariance.data_ptr()),
                calibration_count,
                float(alpha_awt),
                float(diagonal_prior_strength),
                str(covariance.dtype),
            ),
        )
        r_cov = S_x.shape[0]
        # X_red=U Sigma and Y_red=(Y X^T) U Sigma^-1 preserve both
        # X_red X_red^T and Y_red X_red^T, which are the only data terms
        # consumed by every local normal equation below.
        Y_reduced_matrix = (
            target_cross_covariance
            @ U_x.to(dtype=target_cross_covariance.dtype)
        ) / S_x.to(dtype=target_cross_covariance.dtype).unsqueeze(0)
        X_red = (U_x * S_x.unsqueeze(0)).reshape(i0, i1, i2, i3, r_cov)
        Y_red = Y_reduced_matrix.to(dtype=W.dtype).reshape(o0, o1, o2, o3, r_cov)
        del U_x, S_x, Y_reduced_matrix, conditioned_covariance, target_cross_covariance
    del W_core

    if verbose:
        backend = "streamed sufficient statistics" if calibration_statistics is not None else "full calibration matrix"
        print(f"    [Nox Step 2] Exact AWT preconditioning applied from {backend}; keeping r={r_cov} calibration modes (chunk_size={chunk_size})", flush=True)

    # Step 3: Exact AWT TT-SVD Warm-Start on W_cond
    bond_ranks = ranks[1:4] if len(ranks) == 5 else ranks
    cores = adaptive_tt_svd_4site(W_cond, o_modes, i_modes, ranks=bond_ranks)
    del W_cond
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
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
    if diagnostic_callback is not None:
        diagnostic_callback("tt_warmstart", mpo_to_dense([c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]) / d_awt.unsqueeze(0) + S_total)

    def solve_site_system(H, rhs, prior, gamma, width, sweep_index, site_index):
        if solver_diagnostic_callback is None:
            return solve_accumulated_system(
                H,
                rhs,
                prior,
                gamma,
                width,
                use_cg_threshold=cg_threshold,
                n_cg_iters=cg_iterations,
            )
        solved, diagnostics = solve_accumulated_system(
            H,
            rhs,
            prior,
            gamma,
            width,
            use_cg_threshold=cg_threshold,
            n_cg_iters=cg_iterations,
            return_diagnostics=True,
        )
        diagnostics.update({
            "sweep": int(sweep_index + 1),
            "site": int(site_index),
            "local_width": int(width),
        })
        solver_diagnostic_callback(diagnostics)
        return solved

    # Step 4: Isometric DMRG Sweeps in the Conditioned Hilbert Space
    for sweep in range(n_sweeps):
        sweep_started = time.monotonic()
        is_final = (sweep == n_sweeps - 1)
        cur_gamma = prior_gamma * (0.25 if is_final else 1.0)
        if verbose:
            if not is_final:
                mode_str = "QR isometric gauge-fixing"
            elif in_sweep_quantization:
                mode_str = "Final Pass (in-loop 4-bit quantization)"
            elif quantize_components:
                mode_str = "Final optimization pass (4-bit quantization after sweep)"
            else:
                mode_str = "Final optimization pass (unquantized control)"
            print(f"    [Nox Step 3.2] Starting Prior-Anchored DMRG Sweep {sweep+1}/{n_sweeps} [{mode_str}]...", flush=True)

        # Site 0
        r1 = c1.shape[0]
        D0 = r1 * i0
        site_started = time.monotonic()
        if verbose:
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 0/3 start "
                f"(local_width={D0}, calibration_modes={r_cov})",
                flush=True,
            )
        R23 = torch.einsum("dekf,fgl->degkl", c2, c3)
        R123 = torch.einsum("bcjd,degkl->bcejkgl", c1, R23)
        del R23
        H0 = torch.zeros((D0, D0), device=device, dtype=torch.float32)
        rhs0 = torch.zeros((D0, o0), device=device, dtype=torch.float32)
        for r_start in range(0, r_cov, chunk_size):
            r_end = min(r_start + chunk_size, r_cov)
            Z0_c = torch.einsum("bcejkgl,ijklr->cegrbi", R123, X_red[..., r_start:r_end]).reshape(-1, D0)
            Y0_c = Y_red[..., r_start:r_end].permute(1, 2, 3, 4, 0).reshape(-1, o0)
            if str(device).startswith("cuda"):
                H0.add_(torch.mm(Z0_c.half().T, Z0_c.half()).float())
                rhs0.add_(torch.mm(Z0_c.half().T, Y0_c.half()).float())
            else:
                H0.addmm_(Z0_c.T, Z0_c)
                rhs0.addmm_(Z0_c.T, Y0_c)
            del Z0_c, Y0_c
        del R123
        c0_prior = c0.permute(2, 1, 0).reshape(D0, o0)
        c0 = solve_site_system(
            H0, rhs0, c0_prior, cur_gamma, D0, sweep, 0,
        ).reshape(o0, r1, i0).permute(0, 2, 1).contiguous()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        if is_final and in_sweep_quantization:
            c0 = quantize_component(c0, "mpo_site_0")
        else:
            Q, R = torch.linalg.qr(c0.reshape(o0 * i0, r1))
            c0 = Q.reshape(o0, i0, -1)
            c1 = torch.einsum("ab,bcjd->acjd", R, c1)
        if verbose:
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 0/3 complete "
                f"({time.monotonic() - site_started:.1f}s)",
                flush=True,
            )

        # Site 1
        r1, _, _, r2 = c1.shape
        D1 = r1 * r2 * i1
        site_started = time.monotonic()
        if verbose:
            solver = "matrix-free-CG" if D1 > cg_threshold else "normal-equation"
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 1/3 start "
                f"(local_width={D1}, solver={solver}, calibration_modes={r_cov})",
                flush=True,
            )
        R23 = torch.einsum("dekf,fgl->degkl", c2, c3)
        c1_prior = c1.permute(0, 3, 2, 1).reshape(D1, o1)
        # The factorized CG operator does not materialize the enormous local
        # design matrix, so it can use the normal calibration chunk size.  The
        # old eight-mode cap existed only to keep that design matrix below the
        # VRAM limit and caused hundreds of undersized GPU launches per step.
        step_chunk1 = chunk_size

        if D1 > cg_threshold:
            # Matrix-Free Warm-Started CG.  The RHS, norm, and normal operator
            # all retain their tensor-network factorization on the GPU.
            rhs1 = torch.zeros((D1, o1), device=device, dtype=torch.float32)
            # Keep the norm reduction on-device. Calling .item() for every
            # calibration chunk serializes the CUDA stream and materially
            # slows the matrix-free path.
            # FP64 preserves the previous Python-float accumulation precision;
            # only scalar additions use FP64, not the expensive contractions.
            sum_frob1 = _site1_design_frobenius_squared(c0, X_red, R23)
            for r_start in range(0, r_cov, step_chunk1):
                r_end = min(r_start + step_chunk1, r_cov)
                y1 = Y_red[..., r_start:r_end].permute(0, 2, 3, 4, 1)
                rhs1.add_(_site1_factorized_adjoint(
                    c0, X_red[..., r_start:r_end], R23, y1,
                ))

            reg_scale1 = cur_gamma * (sum_frob1 / D1 + 1e-6)
            rhs1.add_(reg_scale1 * c1_prior)
            rhs_norm1 = (
                torch.linalg.vector_norm(rhs1).clamp_min(1e-30)
                if solver_diagnostic_callback is not None else None
            )

            def matvec1(P_in):
                HP = reg_scale1 * P_in
                for r_start in range(0, r_cov, step_chunk1):
                    r_end = min(r_start + step_chunk1, r_cov)
                    HP.add_(_site1_factorized_normal_matvec(
                        c0, X_red[..., r_start:r_end], R23, P_in,
                    ))
                return HP

            HC1 = matvec1(c1_prior)
            R1 = rhs1 - HC1
            del HC1, rhs1
            P1 = R1.clone()
            Delta1 = torch.zeros_like(c1_prior)
            rs_old1 = torch.sum(R1 * R1, dim=0, keepdim=True)
            cg_iterations1 = 0
            for iteration in range(cg_iterations):
                HP1 = matvec1(P1)
                pAp1 = torch.sum(P1 * HP1, dim=0, keepdim=True).clamp(min=1e-12)
                alpha1 = rs_old1 / pAp1
                Delta1.addcmul_(alpha1, P1)
                R1.addcmul_(-alpha1, HP1)
                del HP1
                rs_new1 = torch.sum(R1 * R1, dim=0, keepdim=True)
                cg_iterations1 = iteration + 1
                if torch.max(rs_new1).item() < 1e-8:
                    break
                beta1 = rs_new1 / rs_old1.clamp(min=1e-12)
                P1 = R1 + beta1 * P1
                rs_old1 = rs_new1
            if solver_diagnostic_callback is not None:
                solver_diagnostic_callback({
                    "sweep": int(sweep + 1),
                    "site": 1,
                    "local_width": int(D1),
                    "solver": "matrix_free_conjugate_gradient",
                    "iterations": int(cg_iterations1),
                    "regularization_scale": float(reg_scale1.item()),
                    "relative_normal_residual": float(
                        torch.linalg.vector_norm(R1).div(rhs_norm1).item()
                    ),
                })
            del R1, P1
            c1 = (c1_prior + Delta1).T.reshape(o1, r1, r2, i1).permute(1, 0, 3, 2).contiguous()
            del Delta1, c1_prior
        else:
            H1 = torch.zeros((D1, D1), device=device, dtype=torch.float32)
            rhs1 = torch.zeros((D1, o1), device=device, dtype=torch.float32)
            for r_start in range(0, r_cov, step_chunk1):
                r_end = min(r_start + step_chunk1, r_cov)
                L0_c = torch.einsum("aib,ijklr->abjklr", c0, X_red[..., r_start:r_end])
                Z1_c = torch.einsum("abjklr,degkl->aegrbdj", L0_c, R23).reshape(-1, D1)
                del L0_c
                Y1_c = Y_red[..., r_start:r_end].permute(0, 2, 3, 4, 1).reshape(-1, o1)
                H1.addmm_(Z1_c.T, Z1_c)
                rhs1.addmm_(Z1_c.T, Y1_c)
                del Z1_c, Y1_c
            c1 = solve_site_system(
                H1, rhs1, c1_prior, cur_gamma, D1, sweep, 1,
            ).reshape(o1, r1, r2, i1).permute(1, 0, 3, 2).contiguous()
        del R23
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        if is_final and in_sweep_quantization:
            c1 = quantize_component(c1, "mpo_site_1")
        else:
            Q, R = torch.linalg.qr(c1.reshape(r1 * o1 * i1, r2))
            c1 = Q.reshape(r1, o1, i1, -1)
            c2 = torch.einsum("ab,bcjd->acjd", R, c2)
        if verbose:
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 1/3 complete "
                f"({time.monotonic() - site_started:.1f}s)",
                flush=True,
            )

        # Site 2
        r2, _, _, r3 = c2.shape
        D2 = r2 * r3 * i2
        site_started = time.monotonic()
        if verbose:
            solver = "matrix-free-CG" if D2 > cg_threshold else "normal-equation"
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 2/3 start "
                f"(local_width={D2}, solver={solver}, calibration_modes={r_cov})",
                flush=True,
            )
        L01 = torch.einsum("aib,bcjd->acijd", c0, c1)
        c2_prior = c2.permute(0, 3, 2, 1).reshape(D2, o2)
        step_chunk2 = chunk_size

        if D2 > cg_threshold:
            # Matrix-Free Warm-Started CG through the same factorized operator.
            rhs2 = torch.zeros((D2, o2), device=device, dtype=torch.float32)
            # As at site 1, avoid a host synchronization for every chunk.
            sum_frob2 = _site2_design_frobenius_squared(L01, X_red, c3)
            for r_start in range(0, r_cov, step_chunk2):
                r_end = min(r_start + step_chunk2, r_cov)
                y2 = Y_red[..., r_start:r_end].permute(0, 1, 3, 4, 2)
                rhs2.add_(_site2_factorized_adjoint(
                    L01, X_red[..., r_start:r_end], c3, y2,
                ))

            reg_scale2 = cur_gamma * (sum_frob2 / D2 + 1e-6)
            rhs2.add_(reg_scale2 * c2_prior)
            rhs_norm2 = (
                torch.linalg.vector_norm(rhs2).clamp_min(1e-30)
                if solver_diagnostic_callback is not None else None
            )

            def matvec2(P_in):
                HP = reg_scale2 * P_in
                for r_start in range(0, r_cov, step_chunk2):
                    r_end = min(r_start + step_chunk2, r_cov)
                    HP.add_(_site2_factorized_normal_matvec(
                        L01, X_red[..., r_start:r_end], c3, P_in,
                    ))
                return HP

            HC2 = matvec2(c2_prior)
            R2 = rhs2 - HC2
            del HC2, rhs2
            P2 = R2.clone()
            Delta2 = torch.zeros_like(c2_prior)
            rs_old2 = torch.sum(R2 * R2, dim=0, keepdim=True)
            cg_iterations2 = 0
            for iteration in range(cg_iterations):
                HP2 = matvec2(P2)
                pAp2 = torch.sum(P2 * HP2, dim=0, keepdim=True).clamp(min=1e-12)
                alpha2 = rs_old2 / pAp2
                Delta2.addcmul_(alpha2, P2)
                R2.addcmul_(-alpha2, HP2)
                del HP2
                rs_new2 = torch.sum(R2 * R2, dim=0, keepdim=True)
                cg_iterations2 = iteration + 1
                if torch.max(rs_new2).item() < 1e-8:
                    break
                beta2 = rs_new2 / rs_old2.clamp(min=1e-12)
                P2 = R2 + beta2 * P2
                rs_old2 = rs_new2
            if solver_diagnostic_callback is not None:
                solver_diagnostic_callback({
                    "sweep": int(sweep + 1),
                    "site": 2,
                    "local_width": int(D2),
                    "solver": "matrix_free_conjugate_gradient",
                    "iterations": int(cg_iterations2),
                    "regularization_scale": float(reg_scale2.item()),
                    "relative_normal_residual": float(
                        torch.linalg.vector_norm(R2).div(rhs_norm2).item()
                    ),
                })
            del R2, P2
            c2 = (c2_prior + Delta2).T.reshape(o2, r2, r3, i2).permute(1, 0, 3, 2).contiguous()
            del Delta2, c2_prior
        else:
            H2 = torch.zeros((D2, D2), device=device, dtype=torch.float32)
            rhs2 = torch.zeros((D2, o2), device=device, dtype=torch.float32)
            for r_start in range(0, r_cov, step_chunk2):
                r_end = min(r_start + step_chunk2, r_cov)
                L01_X_c = torch.einsum("acijd,ijklr->acdkrl", L01, X_red[..., r_start:r_end])
                Z2_c = torch.einsum("acdkrl,fgl->acgrdfk", L01_X_c, c3).reshape(-1, D2)
                del L01_X_c
                Y2_c = Y_red[..., r_start:r_end].permute(0, 1, 3, 4, 2).reshape(-1, o2)
                H2.addmm_(Z2_c.T, Z2_c)
                rhs2.addmm_(Z2_c.T, Y2_c)
                del Z2_c, Y2_c
            c2 = solve_site_system(
                H2, rhs2, c2_prior, cur_gamma, D2, sweep, 2,
            ).reshape(o2, r2, r3, i2).permute(1, 0, 3, 2).contiguous()
        del L01
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        if is_final and in_sweep_quantization:
            c2 = quantize_component(c2, "mpo_site_2")
        else:
            Q, R = torch.linalg.qr(c2.reshape(r2 * o2 * i2, r3))
            c2 = Q.reshape(r2, o2, i2, -1)
            c3 = torch.einsum("ab,bcj->acj", R, c3)
        if verbose:
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 2/3 complete "
                f"({time.monotonic() - site_started:.1f}s)",
                flush=True,
            )

        # Site 3
        r3 = c3.shape[0]
        D3 = r3 * i3
        site_started = time.monotonic()
        if verbose:
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 3/3 start "
                f"(local_width={D3}, calibration_modes={r_cov})",
                flush=True,
            )
        L01 = torch.einsum("aib,bcjd->acijd", c0, c1)
        L012 = torch.einsum("acijd,dekf->aceijkf", L01, c2)
        del L01
        H3 = torch.zeros((D3, D3), device=device, dtype=torch.float32)
        rhs3 = torch.zeros((D3, o3), device=device, dtype=torch.float32)
        for r_start in range(0, r_cov, chunk_size):
            r_end = min(r_start + chunk_size, r_cov)
            Z3_c = torch.einsum("aceijkf,ijklr->acerfl", L012, X_red[..., r_start:r_end]).reshape(-1, D3)
            Y3_c = Y_red[..., r_start:r_end].permute(0, 1, 2, 4, 3).reshape(-1, o3)
            if str(device).startswith("cuda"):
                H3.add_(torch.mm(Z3_c.half().T, Z3_c.half()).float())
                rhs3.add_(torch.mm(Z3_c.half().T, Y3_c.half()).float())
            else:
                H3.addmm_(Z3_c.T, Z3_c)
                rhs3.addmm_(Z3_c.T, Y3_c)
            del Z3_c, Y3_c
        del L012
        c3_prior = c3.permute(0, 2, 1).reshape(D3, o3)
        c3 = solve_site_system(
            H3, rhs3, c3_prior, cur_gamma, D3, sweep, 3,
        ).reshape(o3, r3, i3).permute(1, 0, 2).contiguous()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        if is_final and in_sweep_quantization:
            c3 = quantize_component(c3, "mpo_site_3")
        else:
            c0, c1, c2, c3 = right_canonicalize(c0, c1, c2, c3)
        if verbose:
            print(
                f"      [DMRG sweep {sweep+1}/{n_sweeps}] site 3/3 complete "
                f"({time.monotonic() - site_started:.1f}s); sweep complete "
                f"({time.monotonic() - sweep_started:.1f}s)",
                flush=True,
            )
        if diagnostic_callback is not None:
            diagnostic_callback(f"sweep_{sweep + 1}", mpo_to_dense([c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]) / d_awt.unsqueeze(0) + S_total)

    # The historical schedule quantizes each solved site immediately and then
    # builds the next local environment from that rounded core.  This makes a
    # final sweep discontinuous: a small factorization perturbation can cross
    # a Q4 boundary at an early site and be amplified by every later solve.
    # The post-sweep schedule is an explicitly separate numerical control.  It
    # completes the gauge-fixed sweep in floating point and quantizes every
    # final core exactly once, after no remaining local solve can consume it.
    if quantize_components and quantization_schedule == "post_sweep" and n_sweeps > 0:
        if core_diagnostic_callback is not None:
            core_diagnostic_callback(
                "pre_post_sweep_q4",
                (c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)),
                d_awt,
            )
        c0 = quantize_component(c0, "mpo_site_0")
        c1 = quantize_component(c1, "mpo_site_1")
        c2 = quantize_component(c2, "mpo_site_2")
        c3 = quantize_component(c3, "mpo_site_3")

    final_cores = [c0.unsqueeze(0), c1, c2, c3.unsqueeze(-1)]

    # Reconstruct W_hat = (mpo_to_dense(cores) / d_awt) + S_total
    W_mpo = mpo_to_dense(final_cores) / d_awt.unsqueeze(0)
    W_hat = W_mpo + S_total
    del W_mpo, S_total, X_red, Y_red
    if str(device).startswith("cuda"):
        gc.collect()
        torch.cuda.empty_cache()

    # Match the component dtypes and sparse index representation used on disk.
    payload_bytes = estimate_artifact_tensor_bytes(
        final_cores, A_q, B_q, d_awt, sparse_count
    )
    total_kb = payload_bytes / 1024.0
    chosen_ranks = [1, c1.shape[0], c2.shape[0], c3.shape[0], 1]

    if return_artifacts:
        if n_outliers > 0 and mask is not None:
            if nz_mask is None:
                nz_mask = S_sparse != 0
            sparse_idx = torch.nonzero(nz_mask, as_tuple=False).to(dtype=torch.int32, device="cpu")
            sparse_val = S_sparse[nz_mask].to(dtype=torch.float16, device="cpu")
        else:
            sparse_idx = None
            sparse_val = None

        artifact = {
            # With zero sweeps the TT-SVD cores never enter the historical
            # final in-loop quantization pass.  Record that distinction rather
            # than labeling an FP16 warm start as a simulated 4-bit core.
            "quantization_bits": 4 if quantize_components and n_sweeps > 0 else 16,
            "core_quantization_bits": 4 if quantize_components and n_sweeps > 0 else 16,
            "sparse_bridge_quantization_bits": (
                4
                if quantize_components and bridge_quantization_mode != "fp16_control"
                else (16 if not quantize_components else None)
            ),
            "sparse_quantization_bits": 4 if quantize_components else 16,
            "bridge_quantization_bits": (
                4
                if quantize_components and bridge_quantization_mode != "fp16_control"
                else 16
            ),
            "bridge_quantization_diagnostics": (
                bridge_quantization_diagnostics if r_bridge > 0 else None
            ),
            "cores": [c.detach().to(dtype=torch.float16, device="cpu") for c in final_cores],
            "A_bridge": A_q.detach().to(dtype=torch.float16, device="cpu") if r_bridge > 0 and A_q is not None else None,
            "B_bridge": B_q.detach().to(dtype=torch.float16, device="cpu") if r_bridge > 0 and B_q is not None else None,
            "d_awt": d_awt.detach().to(dtype=torch.float16, device="cpu"),
            "sparse_indices": sparse_idx,
            "sparse_values": sparse_val,
            "ranks": chosen_ranks,
            "precision_diagnostics": precision_diagnostics,
            "total_kb": total_kb,
            "orig_shape": (d_out, d_in),
            "compiler_hyperparameters": {
                "n_sweeps": int(n_sweeps),
                "prior_gamma": float(prior_gamma),
                "alpha_awt": float(alpha_awt),
                "chunk_size": int(chunk_size),
                "diagonal_prior_strength": float(diagonal_prior_strength),
                "requested_bridge_rank": int(r_bridge),
                "requested_sparse_budget": int(n_outliers),
                "realized_sparse_count": int(sparse_count),
                "quantize_components": bool(quantize_components),
                "quantization_schedule": quantization_schedule,
                "bridge_quantization_mode": bridge_quantization_mode,
                "bridge_refit_iterations": int(bridge_refit_iterations),
                "cg_threshold": int(cg_threshold),
                "cg_iterations": int(cg_iterations),
                "calibration_backend": (
                    "streaming_sufficient_statistics"
                    if calibration_statistics is not None else "full_matrix_svd"
                ),
                "calibration_sample_count": int(calibration_count),
            },
        }
        if statistics_diagnostics is not None:
            artifact["calibration_statistics_diagnostics"] = statistics_diagnostics
        if calibration_statistics is not None and r_bridge > 0:
            artifact["bridge_statistics_diagnostics"] = bridge_rank_diagnostics
        del S_sparse, d_awt
        del covariance, cross_covariance
        return W_hat, total_kb, final_cores, chosen_ranks, artifact
    else:
        del S_sparse, d_awt
        if calibration_statistics is not None:
            del covariance, cross_covariance
        return W_hat, total_kb, final_cores, chosen_ranks


def nox_hybrid_dmrg_decompose_from_statistics(
    W,
    calibration_statistics,
    **kwargs,
):
    """Fit one NEC operator from globally accumulated calibration statistics.

    The existing full-matrix compiler remains unchanged as the reference path.
    This wrapper makes the mutually exclusive input mode explicit to callers.
    """
    return nox_hybrid_dmrg_decompose(
        W=W,
        X_calib=None,
        Y_dense=None,
        calibration_statistics=calibration_statistics,
        **kwargs,
    )


def nox_4step_compiler(
    W,
    X_comp,
    Y_dense,
    o_modes,
    i_modes,
    ranks=None,
    r_bridge=64,
    n_outliers=262144,
    n_sweeps=2,
    prior_gamma=0.01,
    verbose=False,
    **kwargs,
):
    """
    Convenience wrapper redirecting to nox_hybrid_dmrg_decompose.
    """
    target_ranks = ranks if ranks is not None else (1, 24, 64, 24, 1)
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
