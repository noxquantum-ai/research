import gc
import math
import time
import torch
import torch.nn.functional as F
from nox.compiler import nox_hybrid_dmrg_decompose
from nox.operator_schmidt import nox_operator_schmidt_decompose
from nox.baselines.multiverse import multiverse_tt_svd
from nox.utils import compute_metrics, count_parameters, get_modes_for_projection, mpo_to_dense
from nox.tensor_order import IDENTITY_SITE_ORDER, validate_site_order


def _projection_budget(overrides, name, r_bridge, n_outliers):
    if overrides is None:
        return int(r_bridge), int(n_outliers)
    unknown = set(overrides) - {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    }
    if unknown:
        raise ValueError(f"Unknown projection budget keys: {sorted(unknown)}")
    config = overrides.get(name, {})
    return int(config.get("r_bridge", r_bridge)), int(config.get("n_outliers", n_outliers))


def _validate_mpo_fit_options(n_sweeps, alpha_awt):
    if not isinstance(n_sweeps, int) or n_sweeps < 0:
        raise ValueError("mpo_n_sweeps must be a nonnegative integer")
    if not math.isfinite(float(alpha_awt)):
        raise ValueError("mpo_alpha_awt must be finite")


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-5, weight=None):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = torch.nn.Parameter(weight.clone())
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=0, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            normed = normed * self.weight.unsqueeze(1)
        return normed


def rotate_half(x):
    """
    Rotates half the hidden dims of the input for LLaMA RoPE.
    Matches HuggingFace modeling_llama.py exactly.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_llama2_rotary_emb(dim, seq_len, device, base=10000.0, dtype=torch.float32):
    """
    Computes exact non-interleaved LLaMA-2 RoPE cos and sin embeddings.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype, device=device) / dim))
    t = torch.arange(seq_len, dtype=dtype, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, dim]
    return cos, sin


def apply_llama2_rope(Q, K, cos, sin):
    """
    Applies exact LLaMA-2 Rotary Position Embeddings to Q and K tensors of shape
    [num_seqs, n_heads, seq_len, d_head].
    """
    q_rot = (Q * cos) + (rotate_half(Q) * sin)
    k_rot = (K * cos) + (rotate_half(K) * sin)
    return q_rot, k_rot


def scaled_dot_product_attention_2d(Q, K, V, n_heads=32, n_kv_heads=32, seq_len=128):
    """
    Computes multi-head / grouped-query scaled dot-product attention
    for 2D token activation matrices of shape [d_model, N] with exact LLaMA-2 RoPE.
    
    Args:
        Q: Queries tensor [d_model, N]
        K: Keys tensor [d_model, N]
        V: Values tensor [d_model, N]
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value heads (supports GQA)
        seq_len: Sequence chunk length for attention masking and RoPE positions
        
    Returns:
        Context tensor of shape [d_model, N]
    """
    d_model, N = Q.shape
    d_head = d_model // n_heads
    num_seqs = max(1, N // seq_len)
    actual_len = N // num_seqs

    # Reshape to [num_seqs, heads, actual_len, d_head]
    Q_4d = Q[:, :num_seqs * actual_len].reshape(n_heads, d_head, num_seqs, actual_len).permute(2, 0, 3, 1)
    K_4d = K[:, :num_seqs * actual_len].reshape(n_kv_heads, d_head, num_seqs, actual_len).permute(2, 0, 3, 1)
    V_4d = V[:, :num_seqs * actual_len].reshape(n_kv_heads, d_head, num_seqs, actual_len).permute(2, 0, 3, 1)

    # Apply exact LLaMA-2 Rotary Position Embeddings (RoPE) before GQA expansion
    cos, sin = compute_llama2_rotary_emb(d_head, actual_len, device=Q.device, dtype=Q.dtype)
    Q_4d, K_4d = apply_llama2_rope(Q_4d, K_4d, cos, sin)

    if n_kv_heads < n_heads:
        repeat_factor = n_heads // n_kv_heads
        K_4d = K_4d.repeat_interleave(repeat_factor, dim=1)
        V_4d = V_4d.repeat_interleave(repeat_factor, dim=1)

    # Preserve the same causal attention map while allowing PyTorch's fused
    # backend to avoid a materialized [windows, heads, length, length] score
    # tensor. This matters for representative 2048-token calibration windows.
    context_4d = F.scaled_dot_product_attention(
        Q_4d, K_4d, V_4d, dropout_p=0.0, is_causal=True,
    )
    context_2d = context_4d.permute(1, 3, 0, 2).reshape(d_model, num_seqs * actual_len)

    if context_2d.shape[1] < N:
        pad_len = N - context_2d.shape[1]
        context_2d = F.pad(context_2d, (0, pad_len))

    return context_2d


def ridge_compensated_up_target(
    gate_dense,
    up_dense,
    gate_compressed,
    down_weight=None,
    ridge_ratio=0.05,
):
    """Closed-form up activations that compensate a fixed gate approximation.

    For each intermediate channel and token this minimizes

        (silu(g_hat) u - silu(g) u_dense)^2
        + lambda (u - u_dense)^2,

    weighted by the squared norm of the corresponding dense Down column, plus
    lambda * (u - u_dense)^2. The diagonal output metric approximates
    diag(D^T D), so channels with greater downstream effect receive stronger
    product correction. The ridge term makes the target finite around SiLU
    zeros.
    """
    if ridge_ratio <= 0:
        raise ValueError("ridge_ratio must be positive")
    if gate_dense.shape != up_dense.shape or gate_compressed.shape != gate_dense.shape:
        raise ValueError("gate and up activation tensors must have identical shapes")
    factor_dense = F.silu(gate_dense.float())
    factor_compressed = F.silu(gate_compressed.float())
    product_dense = factor_dense * up_dense.float()
    if down_weight is None:
        importance = torch.ones(
            gate_dense.shape[0], device=gate_dense.device, dtype=torch.float32,
        )
    else:
        if down_weight.ndim != 2 or down_weight.shape[1] != gate_dense.shape[0]:
            raise ValueError("down_weight must have shape [d_model, intermediate_size]")
        importance = down_weight.float().square().sum(dim=0)
        importance = importance / importance.mean().clamp_min(1e-12)
        importance = importance.clamp(0.05, 20.0)
    ridge = ridge_ratio * factor_compressed.square().mean().clamp_min(1e-8)
    weighted_factor = importance.unsqueeze(1) * factor_compressed
    target = (
        weighted_factor * product_dense + ridge * up_dense.float()
    ) / (importance.unsqueeze(1) * factor_compressed.square() + ridge)
    return target.to(dtype=up_dense.dtype), ridge, importance


@torch.no_grad()
def compile_attention_block(
    weights,
    X_in,
    X_in_dense=None,
    residual_drift=None,
    alpha_heal=0.0,
    ranks=(1, 24, 64, 24, 1),
    r_bridge=64,
    n_outliers=262144,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    causal_healing=True,
    return_artifacts=False,
    verbose=False,
    diagonal_prior_strength=0.0,
    site_order=IDENTITY_SITE_ORDER,
    conditional_context=False,
    projection_budgets=None,
    representation="mpo4",
    operator_schmidt_weight_mix=1.0,
    operator_schmidt_activation_mix=1.0,
    mpo_n_sweeps=2,
    mpo_alpha_awt=0.5,
    mpo_quantization_schedule="in_sweep",
    mpo_cg_threshold=4096,
    mpo_cg_iterations=6,
):
    """
    Compiles the 4 attention projections (W_q, W_k, W_v, W_o) with optional
    causal healing where W_o absorbs Q,K,V distortion and inter-layer residual drift.
    """
    site_order = validate_site_order(site_order)
    _validate_mpo_fit_options(mpo_n_sweeps, mpo_alpha_awt)
    if representation not in ("mpo4", "operator_schmidt"):
        raise ValueError(f"Unknown NEC representation: {representation}")

    def compile_projection(*, projection_name, **kwargs):
        if verbose:
            print(f"    [NEC projection] Starting {projection_name}...", flush=True)
        if representation == "mpo4":
            return nox_hybrid_dmrg_decompose(
                diagonal_prior_strength=diagonal_prior_strength,
                site_order=site_order,
                n_sweeps=mpo_n_sweeps,
                alpha_awt=mpo_alpha_awt,
                quantization_schedule=mpo_quantization_schedule,
                cg_threshold=mpo_cg_threshold,
                cg_iterations=mpo_cg_iterations,
                **kwargs,
            )
        if site_order != IDENTITY_SITE_ORDER:
            raise ValueError("MPO site order is not applicable to the operator-Schmidt engine")
        return nox_operator_schmidt_decompose(
            projection_name=projection_name,
            n_heads=n_heads,
            weight_mix=operator_schmidt_weight_mix,
            activation_mix=operator_schmidt_activation_mix,
            **kwargs,
        )
    W_q = weights["q_proj"]
    W_k = weights["k_proj"]
    W_v = weights["v_proj"]
    W_o = weights["o_proj"]
    r_bridge_q, n_outliers_q = _projection_budget(
        projection_budgets, "q_proj", r_bridge, n_outliers,
    )
    r_bridge_k, n_outliers_k = _projection_budget(
        projection_budgets, "k_proj", r_bridge, n_outliers,
    )
    r_bridge_v, n_outliers_v = _projection_budget(
        projection_budgets, "v_proj", r_bridge, n_outliers,
    )
    r_bridge_o_base, n_outliers_o_base = _projection_budget(
        projection_budgets, "o_proj", r_bridge, n_outliers,
    )
    d_model = W_o.shape[0]
    o_modes, i_modes = get_modes_for_projection(W_q.shape[0], W_q.shape[1])

    ref_in = X_in_dense if X_in_dense is not None else X_in
    Q_dense = W_q @ ref_in
    K_dense = W_k @ ref_in
    V_dense = W_v @ ref_in
    A_dense = scaled_dot_product_attention_2d(Q_dense, K_dense, V_dense, n_heads, n_kv_heads, seq_len)
    Y_attn_dense = W_o @ A_dense

    if verbose:
        print("  [Attention Sub-Block] Compressing Q, K, V projections...", flush=True)

    art_q = art_k = art_v = art_o = None
    if return_artifacts:
        W_hat_q, kb_q, cores_q, _, art_q = compile_projection(
            projection_name="q_proj",
            W=W_q, X_calib=X_in, Y_dense=Q_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_q, n_outliers=n_outliers_q, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_q, kb_q, cores_q, _ = compile_projection(
            projection_name="q_proj",
            W=W_q, X_calib=X_in, Y_dense=Q_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_q, n_outliers=n_outliers_q, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if return_artifacts:
        W_hat_k, kb_k, cores_k, _, art_k = compile_projection(
            projection_name="k_proj",
            W=W_k, X_calib=X_in, Y_dense=K_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_k, n_outliers=n_outliers_k, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_k, kb_k, cores_k, _ = compile_projection(
            projection_name="k_proj",
            W=W_k, X_calib=X_in, Y_dense=K_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_k, n_outliers=n_outliers_k, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if return_artifacts:
        W_hat_v, kb_v, cores_v, _, art_v = compile_projection(
            projection_name="v_proj",
            W=W_v, X_calib=X_in, Y_dense=V_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_v, n_outliers=n_outliers_v, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_v, kb_v, cores_v, _ = compile_projection(
            projection_name="v_proj",
            W=W_v, X_calib=X_in, Y_dense=V_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_v, n_outliers=n_outliers_v, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    Q_comp = W_hat_q @ X_in
    K_comp = W_hat_k @ X_in
    V_comp = W_hat_v @ X_in
    A_comp = scaled_dot_product_attention_2d(Q_comp, K_comp, V_comp, n_heads, n_kv_heads, seq_len)

    if causal_healing:
        X_o_calib = A_comp
        r_bridge_o = min(int(r_bridge_o_base * 1.5), max(64, int(X_in.shape[1] * 0.6)))
        n_outliers_o = n_outliers_o_base // 2
        if residual_drift is not None:
            Y_attn_target = Y_attn_dense + alpha_heal * residual_drift
        else:
            Y_attn_target = Y_attn_dense
        if verbose:
            print(f"  [Attention Sub-Block] Causal Healing active: Solving W_o on corrupted context A_comp (bridge: r={r_bridge_o}, outliers={n_outliers_o})...", flush=True)
    elif conditional_context:
        # Fit once on the context the saved Q/K/V factors actually produce.
        # Capacity stays identical to the independent zero-healing baseline.
        X_o_calib = A_comp
        r_bridge_o = r_bridge_o_base
        n_outliers_o = n_outliers_o_base
        Y_attn_target = Y_attn_dense
        if verbose:
            print(
                "  [Attention Sub-Block] Trajectory-aware construction: "
                "solving W_o on compressed context at the fixed budget...",
                flush=True,
            )
    else:
        if verbose:
            print("  [Attention Sub-Block] Independent mode: Solving W_o on uncorrupted context A_dense...", flush=True)
        X_o_calib = A_dense
        r_bridge_o = r_bridge_o_base
        n_outliers_o = n_outliers_o_base
        Y_attn_target = Y_attn_dense

    gamma_o = 0.002 if causal_healing else 0.01
    if return_artifacts:
        W_hat_o, kb_o, cores_o, _, art_o = compile_projection(
            projection_name="o_proj",
            W=W_o, X_calib=X_o_calib, Y_dense=Y_attn_target, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_o, n_outliers=n_outliers_o, prior_gamma=gamma_o, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_o, kb_o, cores_o, _ = compile_projection(
            projection_name="o_proj",
            W=W_o, X_calib=X_o_calib, Y_dense=Y_attn_target, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_o, n_outliers=n_outliers_o, prior_gamma=gamma_o, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    Y_attn_comp = W_hat_o @ A_comp
    total_kb = kb_q + kb_k + kb_v + kb_o

    res = {
        "W_hat_q": W_hat_q,
        "W_hat_k": W_hat_k,
        "W_hat_v": W_hat_v,
        "W_hat_o": W_hat_o,
        "Y_attn_comp": Y_attn_comp,
        "Y_attn_dense": Y_attn_dense,
        "A_comp": A_comp,
        "A_dense": A_dense,
        "total_kb": total_kb,
    }
    if return_artifacts:
        res["artifacts"] = {
            "q_proj": art_q,
            "k_proj": art_k,
            "v_proj": art_v,
            "o_proj": art_o,
        }
    return res


@torch.no_grad()
def compile_mlp_block(
    weights,
    M_in,
    M_in_dense=None,
    residual_drift=None,
    alpha_heal=0.0,
    ranks=(1, 24, 64, 24, 1),
    r_bridge=64,
    n_outliers=262144,
    causal_healing=True,
    return_artifacts=False,
    verbose=False,
    diagonal_prior_strength=0.0,
    site_order=IDENTITY_SITE_ORDER,
    conditional_context=False,
    product_aware=False,
    product_ridge_ratio=0.05,
    projection_budgets=None,
    representation="mpo4",
    operator_schmidt_weight_mix=1.0,
    operator_schmidt_activation_mix=1.0,
    mpo_n_sweeps=2,
    mpo_alpha_awt=0.5,
    mpo_quantization_schedule="in_sweep",
    mpo_cg_threshold=4096,
    mpo_cg_iterations=6,
):
    """
    Compiles the 3 SwiGLU MLP projections (W_gate, W_up, W_down) with optional
    causal healing where W_down absorbs non-linear distortion and residual drift.
    """
    site_order = validate_site_order(site_order)
    _validate_mpo_fit_options(mpo_n_sweeps, mpo_alpha_awt)
    if representation not in ("mpo4", "operator_schmidt"):
        raise ValueError(f"Unknown NEC representation: {representation}")

    def compile_projection(*, projection_name, **kwargs):
        if verbose:
            print(f"    [NEC projection] Starting {projection_name}...", flush=True)
        if representation == "mpo4":
            return nox_hybrid_dmrg_decompose(
                diagonal_prior_strength=diagonal_prior_strength,
                site_order=site_order,
                n_sweeps=mpo_n_sweeps,
                alpha_awt=mpo_alpha_awt,
                quantization_schedule=mpo_quantization_schedule,
                cg_threshold=mpo_cg_threshold,
                cg_iterations=mpo_cg_iterations,
                **kwargs,
            )
        if site_order != IDENTITY_SITE_ORDER:
            raise ValueError("MPO site order is not applicable to the operator-Schmidt engine")
        return nox_operator_schmidt_decompose(
            projection_name=projection_name,
            weight_mix=operator_schmidt_weight_mix,
            activation_mix=operator_schmidt_activation_mix,
            **kwargs,
        )
    W_gate = weights["gate_proj"]
    W_up = weights["up_proj"]
    W_down = weights["down_proj"]
    r_bridge_gate, n_outliers_gate = _projection_budget(
        projection_budgets, "gate_proj", r_bridge, n_outliers,
    )
    r_bridge_up, n_outliers_up = _projection_budget(
        projection_budgets, "up_proj", r_bridge, n_outliers,
    )
    r_bridge_down_base, n_outliers_down_base = _projection_budget(
        projection_budgets, "down_proj", r_bridge, n_outliers,
    )

    o_modes_up, i_modes_up = get_modes_for_projection(W_gate.shape[0], W_gate.shape[1])
    o_modes_down, i_modes_down = get_modes_for_projection(W_down.shape[0], W_down.shape[1])

    if M_in_dense is None:
        M_in_dense = M_in

    H_gate_dense = W_gate @ M_in_dense
    H_up_dense = W_up @ M_in_dense
    H_swiglu_dense = F.silu(H_gate_dense) * H_up_dense
    Y_mlp_dense = W_down @ H_swiglu_dense

    if verbose:
        print("  [SwiGLU MLP Sub-Block] Compressing Gate and Up projections...", flush=True)

    art_gate = art_up = art_down = None
    if return_artifacts:
        W_hat_gate, kb_gate, cores_gate, _, art_gate = compile_projection(
            projection_name="gate_proj",
            W=W_gate, X_calib=M_in, Y_dense=H_gate_dense, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge_gate, n_outliers=n_outliers_gate, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_gate, kb_gate, cores_gate, _ = compile_projection(
            projection_name="gate_proj",
            W=W_gate, X_calib=M_in, Y_dense=H_gate_dense, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge_gate, n_outliers=n_outliers_gate, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    H_gate_comp = W_hat_gate @ M_in
    product_metadata = None
    if product_aware:
        H_up_target, product_ridge, product_importance = ridge_compensated_up_target(
            H_gate_dense, H_up_dense, H_gate_comp,
            down_weight=W_down,
            ridge_ratio=product_ridge_ratio,
        )
        dense_product = F.silu(H_gate_dense) * H_up_dense
        target_product = F.silu(H_gate_comp) * H_up_target
        product_metadata = {
            "objective": "ridge_compensated_swiglu_product",
            "ridge_ratio": float(product_ridge_ratio),
            "mean_channel_ridge": float(product_ridge.float().mean()),
            "down_channel_importance_mean": float(product_importance.mean()),
            "down_channel_importance_p95": float(torch.quantile(product_importance, 0.95)),
            "pre_fit_target_relative_error": compute_metrics(
                dense_product, target_product,
            )["relative_error"],
        }
        if verbose:
            print(
                "  [SwiGLU MLP Sub-Block] Product-aware construction: "
                "fitting Up to a ridge-compensated compressed-gate target...",
                flush=True,
            )
    else:
        H_up_target = H_up_dense

    if return_artifacts:
        W_hat_up, kb_up, cores_up, _, art_up = compile_projection(
            projection_name="up_proj",
            W=W_up, X_calib=M_in, Y_dense=H_up_target, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge_up, n_outliers=n_outliers_up, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_up, kb_up, cores_up, _ = compile_projection(
            projection_name="up_proj",
            W=W_up, X_calib=M_in, Y_dense=H_up_target, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge_up, n_outliers=n_outliers_up, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    H_up_comp = W_hat_up @ M_in
    H_swiglu_comp = F.silu(H_gate_comp) * H_up_comp
    if product_metadata is not None:
        product_metadata["post_fit_product_relative_error"] = compute_metrics(
            H_swiglu_dense, H_swiglu_comp,
        )["relative_error"]

    if causal_healing:
        X_down_calib = H_swiglu_comp
        r_bridge_down = min(int(r_bridge_down_base * 1.5), max(64, int(H_swiglu_comp.shape[1] * 0.6)))
        n_outliers_down = n_outliers_down_base // 2
        if residual_drift is not None:
            Y_mlp_target = Y_mlp_dense + alpha_heal * residual_drift
        else:
            Y_mlp_target = Y_mlp_dense
        if verbose:
            print(f"  [SwiGLU MLP Sub-Block] Causal Healing active: Solving W_down on corrupted activation H_swiglu_comp (bridge: r={r_bridge_down}, outliers={n_outliers_down})...", flush=True)
    elif conditional_context:
        # Condition the initial down-projection construction on the SwiGLU
        # activation produced by the factors that will be serialized. This is
        # part of compilation and keeps the bridge/sparse/MPO budget fixed.
        X_down_calib = H_swiglu_comp
        r_bridge_down = r_bridge_down_base
        n_outliers_down = n_outliers_down_base
        Y_mlp_target = Y_mlp_dense
        if verbose:
            print(
                "  [SwiGLU MLP Sub-Block] Trajectory-aware construction: "
                "solving W_down on compressed SwiGLU context at the fixed budget...",
                flush=True,
            )
    else:
        if verbose:
            print("  [SwiGLU MLP Sub-Block] Independent mode: Solving W_down on uncorrupted activation H_swiglu_dense...", flush=True)
        X_down_calib = H_swiglu_dense
        r_bridge_down = r_bridge_down_base
        n_outliers_down = n_outliers_down_base
        Y_mlp_target = Y_mlp_dense

    if verbose:
        if causal_healing:
            mode_str = "Causal Healing active: Solving W_down on corrupted SwiGLU context"
        elif conditional_context:
            mode_str = "Trajectory-aware construction: Solving W_down on compressed SwiGLU context"
        else:
            mode_str = "Independent mode: Solving W_down on clean SwiGLU context"
        print(f"  [SwiGLU MLP Sub-Block] {mode_str} (bridge: r={r_bridge_down}, outliers={n_outliers_down})...", flush=True)

    gamma_down = 0.002 if causal_healing else 0.01
    if return_artifacts:
        W_hat_down, kb_down, cores_down, _, art_down = compile_projection(
            projection_name="down_proj",
            W=W_down, X_calib=X_down_calib, Y_dense=Y_mlp_target, o_modes=o_modes_down, i_modes=i_modes_down,
            ranks=ranks, r_bridge=r_bridge_down, n_outliers=n_outliers_down, prior_gamma=gamma_down, return_artifacts=True, verbose=verbose,
        )
    else:
        W_hat_down, kb_down, cores_down, _ = compile_projection(
            projection_name="down_proj",
            W=W_down, X_calib=X_down_calib, Y_dense=Y_mlp_target, o_modes=o_modes_down, i_modes=i_modes_down,
            ranks=ranks, r_bridge=r_bridge_down, n_outliers=n_outliers_down, prior_gamma=gamma_down, verbose=verbose,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    Y_mlp_comp = W_hat_down @ H_swiglu_comp
    total_kb = kb_gate + kb_up + kb_down

    res = {
        "W_hat_gate": W_hat_gate,
        "W_hat_up": W_hat_up,
        "W_hat_down": W_hat_down,
        "Y_mlp_comp": Y_mlp_comp,
        "Y_mlp_dense": Y_mlp_dense,
        "H_swiglu_comp": H_swiglu_comp,
        "H_swiglu_dense": H_swiglu_dense,
        "product_aware": product_metadata,
        "total_kb": total_kb,
    }
    if return_artifacts:
        res["artifacts"] = {
            "gate_proj": art_gate,
            "up_proj": art_up,
            "down_proj": art_down,
        }
    return res


@torch.no_grad()
def compile_transformer_block(
    layer_weights,
    X_calib,
    X_calib_dense=None,
    alpha_heal=0.0,
    ranks=(1, 24, 64, 24, 1),
    r_bridge=64,
    n_outliers=262144,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    causal_healing=True,
    return_artifacts=False,
    verbose=True,
    rms_norm_eps=1e-5,
    diagonal_prior_strength=0.0,
    site_order=IDENTITY_SITE_ORDER,
    trajectory_mode="none",
    mlp_product_aware=False,
    mlp_product_ridge_ratio=0.05,
    projection_budgets=None,
    representation="mpo4",
    operator_schmidt_weight_mix=1.0,
    operator_schmidt_activation_mix=1.0,
    mpo_n_sweeps=2,
    mpo_alpha_awt=0.5,
    mpo_quantization_schedule="in_sweep",
    mpo_cg_threshold=4096,
    mpo_cg_iterations=6,
):
    """
    Executes full Causal Inter-Layer & Intra-Block Propagation (Step 4.2) for a Transformer layer.
    
    1. Input RMSNorm: X_norm1 = RMSNorm_1(X_calib), X_norm1_dense = RMSNorm_1(X_calib_dense)
    2. Attention Projections: W_q, W_k, W_v compressed -> Q_comp, K_comp, V_comp
    3. Scaled Dot-Product Attention: A_comp
    4. Causal W_o Solve: W_o absorbs Q,K,V distortion + damped inter-layer residual drift:
       Y_attn_target = Y_attn_dense + alpha_heal * (X_calib_dense - X_calib)
    5. First Residual Add: X_res1_comp = X_calib + Y_attn_comp, X_res1_dense = X_calib_dense + Y_attn_dense
    6. Post-Attention RMSNorm: M_in = RMSNorm_2(X_res1_comp), M_in_dense = RMSNorm_2(X_res1_dense)
    7. SwiGLU Projections: W_gate, W_up compressed -> H_swiglu_comp
    8. Causal W_down Solve: W_down absorbs non-linear distortion + damped residual drift:
       Y_mlp_target = Y_mlp_dense + alpha_heal * (X_res1_dense - X_res1_comp)
    9. Second Residual Add: X_out_comp = X_res1_comp + Y_mlp_comp, X_out_dense = X_res1_dense + Y_mlp_dense
    """
    t_start = time.time()
    d_model = X_calib.shape[0]
    site_order = validate_site_order(site_order)
    if trajectory_mode not in ("none", "mlp", "all", "product"):
        raise ValueError(f"Unknown trajectory_mode: {trajectory_mode}")
    if causal_healing and trajectory_mode != "none":
        raise ValueError("trajectory_mode and causal_healing are mutually exclusive")

    norm1_weight = layer_weights.get("input_layernorm", None)
    norm2_weight = layer_weights.get("post_attention_layernorm", None)

    norm1 = RMSNorm(d_model, eps=rms_norm_eps, weight=norm1_weight)
    norm2 = RMSNorm(d_model, eps=rms_norm_eps, weight=norm2_weight)

    ref_calib_dense = X_calib_dense if X_calib_dense is not None else X_calib

    X_norm1_comp = norm1(X_calib)
    X_norm1_dense = norm1(ref_calib_dense)

    drift_attn = (ref_calib_dense - X_calib) if (X_calib_dense is not None and causal_healing) else None

    attn_res = compile_attention_block(
        diagonal_prior_strength=diagonal_prior_strength,
        weights=layer_weights,
        X_in=X_norm1_comp,
        X_in_dense=X_norm1_dense,
        residual_drift=drift_attn,
        alpha_heal=alpha_heal,
        ranks=ranks,
        r_bridge=r_bridge,
        n_outliers=n_outliers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        causal_healing=causal_healing,
        return_artifacts=return_artifacts,
        verbose=verbose,
        site_order=site_order,
        conditional_context=trajectory_mode in ("all", "product"),
        projection_budgets=projection_budgets,
        representation=representation,
        operator_schmidt_weight_mix=operator_schmidt_weight_mix,
        operator_schmidt_activation_mix=operator_schmidt_activation_mix,
        mpo_n_sweeps=mpo_n_sweeps,
        mpo_alpha_awt=mpo_alpha_awt,
        mpo_quantization_schedule=mpo_quantization_schedule,
        mpo_cg_threshold=mpo_cg_threshold,
        mpo_cg_iterations=mpo_cg_iterations,
    )

    X_res1_dense = ref_calib_dense + attn_res["Y_attn_dense"]
    X_res1_comp = X_calib + attn_res["Y_attn_comp"]

    X_norm2_dense = norm2(X_res1_dense)
    X_norm2_comp = (
        norm2(X_res1_comp)
        if causal_healing or trajectory_mode in ("mlp", "all", "product")
        else norm2(X_res1_dense)
    )

    drift_mlp = (X_res1_dense - X_res1_comp) if (X_calib_dense is not None and causal_healing) else None

    mlp_res = compile_mlp_block(
        diagonal_prior_strength=diagonal_prior_strength,
        weights=layer_weights,
        M_in=X_norm2_comp,
        M_in_dense=X_norm2_dense,
        residual_drift=drift_mlp,
        alpha_heal=alpha_heal,
        ranks=ranks,
        r_bridge=r_bridge,
        n_outliers=n_outliers,
        causal_healing=causal_healing,
        return_artifacts=return_artifacts,
        verbose=verbose,
        site_order=site_order,
        conditional_context=trajectory_mode in ("mlp", "all"),
        product_aware=mlp_product_aware,
        product_ridge_ratio=mlp_product_ridge_ratio,
        projection_budgets=projection_budgets,
        representation=representation,
        operator_schmidt_weight_mix=operator_schmidt_weight_mix,
        operator_schmidt_activation_mix=operator_schmidt_activation_mix,
        mpo_n_sweeps=mpo_n_sweeps,
        mpo_alpha_awt=mpo_alpha_awt,
        mpo_quantization_schedule=mpo_quantization_schedule,
        mpo_cg_threshold=mpo_cg_threshold,
        mpo_cg_iterations=mpo_cg_iterations,
    )

    X_out_dense = X_res1_dense + mlp_res["Y_mlp_dense"]
    X_out_comp = X_res1_comp + mlp_res["Y_mlp_comp"]

    total_time = time.time() - t_start
    total_kb = attn_res["total_kb"] + mlp_res["total_kb"]

    Delta_dense = attn_res["Y_attn_dense"] + mlp_res["Y_mlp_dense"]
    Delta_comp = attn_res["Y_attn_comp"] + mlp_res["Y_mlp_comp"]

    block_metrics = compute_metrics(X_out_dense, X_out_comp)
    transform_metrics = compute_metrics(Delta_dense, Delta_comp)
    attn_metrics = compute_metrics(attn_res["Y_attn_dense"], attn_res["Y_attn_comp"])
    mlp_metrics = compute_metrics(mlp_res["Y_mlp_dense"], mlp_res["Y_mlp_comp"])

    res = {
        "compressed_weights": {
            "q_proj": attn_res["W_hat_q"],
            "k_proj": attn_res["W_hat_k"],
            "v_proj": attn_res["W_hat_v"],
            "o_proj": attn_res["W_hat_o"],
            "gate_proj": mlp_res["W_hat_gate"],
            "up_proj": mlp_res["W_hat_up"],
            "down_proj": mlp_res["W_hat_down"],
        },
        "X_out_comp": X_out_comp,
        "X_out_dense": X_out_dense,
        "block_metrics": block_metrics,
        "transform_metrics": transform_metrics,
        "attn_metrics": attn_metrics,
        "mlp_metrics": mlp_metrics,
        "mlp_product_aware": mlp_res["product_aware"],
        "total_kb": total_kb,
        "time_sec": total_time,
        "causal_healing": causal_healing,
        "trajectory_mode": trajectory_mode,
    }
    if return_artifacts:
        res["compressed_artifacts"] = {
            **attn_res["artifacts"],
            **mlp_res["artifacts"],
        }
    return res


@torch.no_grad()
def evaluate_compiled_block(compiled_weights, layer_weights, X_test, n_heads=32, n_kv_heads=32, seq_len=128, rms_norm_eps=1e-5):
    """
    Evaluates a compiled Transformer block on unseen held-out test tokens.
    """
    d_model = X_test.shape[0]
    norm1 = RMSNorm(d_model, eps=rms_norm_eps, weight=layer_weights.get("input_layernorm", None))
    norm2 = RMSNorm(d_model, eps=rms_norm_eps, weight=layer_weights.get("post_attention_layernorm", None))

    cw = compiled_weights["compressed_weights"]
    lw = layer_weights

    # Dense reference path
    X_n1_d = norm1(X_test)
    Q_d = lw["q_proj"] @ X_n1_d
    K_d = lw["k_proj"] @ X_n1_d
    V_d = lw["v_proj"] @ X_n1_d
    A_d = scaled_dot_product_attention_2d(Q_d, K_d, V_d, n_heads, n_kv_heads, seq_len)
    Y_attn_d = lw["o_proj"] @ A_d
    X_res1_d = X_test + Y_attn_d

    X_n2_d = norm2(X_res1_d)
    H_g_d = lw["gate_proj"] @ X_n2_d
    H_u_d = lw["up_proj"] @ X_n2_d
    H_sw_d = F.silu(H_g_d) * H_u_d
    Y_mlp_d = lw["down_proj"] @ H_sw_d
    X_out_dense = X_res1_d + Y_mlp_d

    # Compressed model path
    X_n1_c = norm1(X_test)
    Q_c = cw["q_proj"] @ X_n1_c
    K_c = cw["k_proj"] @ X_n1_c
    V_c = cw["v_proj"] @ X_n1_c
    A_c = scaled_dot_product_attention_2d(Q_c, K_c, V_c, n_heads, n_kv_heads, seq_len)
    Y_attn_c = cw["o_proj"] @ A_c
    X_res1_c = X_test + Y_attn_c

    X_n2_c = norm2(X_res1_c)
    H_g_c = cw["gate_proj"] @ X_n2_c
    H_u_c = cw["up_proj"] @ X_n2_c
    H_sw_c = F.silu(H_g_c) * H_u_c
    Y_mlp_c = cw["down_proj"] @ H_sw_c
    X_out_comp = X_res1_c + Y_mlp_c
    Delta_dense = Y_attn_d + Y_mlp_d
    Delta_comp = Y_attn_c + Y_mlp_c

    return {
        "block_metrics": compute_metrics(X_out_dense, X_out_comp),
        "transform_metrics": compute_metrics(Delta_dense, Delta_comp),
        "attn_metrics": compute_metrics(Y_attn_d, Y_attn_c),
        "mlp_metrics": compute_metrics(Y_mlp_d, Y_mlp_c),
    }


def compute_isobyte_multiverse_ranks(target_kb, core_bits=4):
    """
    Computes MPO bond ranks [r, r_mid, r] such that the unhealed Multiverse baseline
    has an equal or slightly larger KB footprint than Nox (Iso-Byte fairness).
    """
    for r in range(16, 256, 4):
        r_mid = int(r * 1.67)
        bytes_attn = (128 * r + 128 * r * r_mid) * (core_bits / 8.0)
        bytes_mlp = (376 * r + 128 * r * r_mid) * (core_bits / 8.0)
        total_kb = (4 * bytes_attn + 3 * bytes_mlp) / 1024.0
        if total_kb >= target_kb:
            return (r, r_mid, r)
    return (96, 160, 96)


@torch.no_grad()
def compile_multiverse_block(
    layer_weights,
    ranks=(24, 24, 24),
    target_kb=None,
    core_bits=4,
    verbose=False,
):
    """
    Compiles an entire Transformer layer using the Multiverse (CompactifAI) TT-SVD baseline.
    All 7 projections are decomposed independently with pure Frobenius SVD and post-hoc core quantization.
    If target_kb is provided, automatically uses Iso-Byte bond ranks.
    """
    if target_kb is not None:
        ranks = compute_isobyte_multiverse_ranks(target_kb, core_bits=core_bits)

    t_start = time.time()
    compressed_weights = {}
    total_bytes = 0.0

    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for name in projs:
        W = layer_weights[name]
        o_modes, i_modes = get_modes_for_projection(W.shape[0], W.shape[1])
        if verbose:
            print(f"  [Multiverse Baseline] Decomposing {name}...", flush=True)
        cores, chosen_ranks = multiverse_tt_svd(
            W=W,
            o_modes=o_modes,
            i_modes=i_modes,
            ranks=list(ranks),
            core_bits=core_bits,
        )
        W_hat = mpo_to_dense(cores)
        compressed_weights[name] = W_hat
        fp = count_parameters(cores, bits=core_bits)
        total_bytes += fp["total_bytes"]

    total_time = time.time() - t_start
    return {
        "compressed_weights": compressed_weights,
        "total_kb": total_bytes / 1024.0,
        "total_bytes": total_bytes,
        "time_sec": total_time,
        "ranks": tuple(ranks),
    }
