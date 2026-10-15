import gc
import time
import torch
import torch.nn.functional as F
from nox.compiler import nox_hybrid_dmrg_decompose
from nox.baselines.multiverse import multiverse_tt_svd
from nox.utils import compute_metrics, count_parameters, get_modes_for_projection, mpo_to_dense


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6, weight=None):
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


def scaled_dot_product_attention_2d(Q, K, V, n_heads=32, n_kv_heads=32, seq_len=128):
    """
    Computes multi-head / grouped-query scaled dot-product attention
    for 2D token activation matrices of shape [d_model, N].
    
    Args:
        Q: Queries tensor [d_model, N]
        K: Keys tensor [d_model, N]
        V: Values tensor [d_model, N]
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value heads (supports GQA)
        seq_len: Sequence chunk length for attention masking
        
    Returns:
        Context tensor of shape [d_model, N]
    """
    d_model, N = Q.shape
    d_head = d_model // n_heads
    num_seqs = max(1, N // seq_len)
    actual_len = N // num_seqs

    Q_4d = Q[:, :num_seqs * actual_len].reshape(n_heads, d_head, num_seqs, actual_len).permute(2, 0, 3, 1)
    K_4d = K[:, :num_seqs * actual_len].reshape(n_kv_heads, d_head, num_seqs, actual_len).permute(2, 0, 3, 1)
    V_4d = V[:, :num_seqs * actual_len].reshape(n_kv_heads, d_head, num_seqs, actual_len).permute(2, 0, 3, 1)

    if n_kv_heads < n_heads:
        repeat_factor = n_heads // n_kv_heads
        K_4d = K_4d.repeat_interleave(repeat_factor, dim=1)
        V_4d = V_4d.repeat_interleave(repeat_factor, dim=1)

    scale = 1.0 / (d_head ** 0.5)
    scores = torch.matmul(Q_4d, K_4d.transpose(-2, -1)) * scale

    causal_mask = torch.triu(torch.full((actual_len, actual_len), float("-inf"), device=Q.device), diagonal=1)
    scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
    probs = F.softmax(scores, dim=-1)

    context_4d = torch.matmul(probs, V_4d)
    context_2d = context_4d.permute(1, 3, 0, 2).reshape(d_model, num_seqs * actual_len)

    if context_2d.shape[1] < N:
        pad_len = N - context_2d.shape[1]
        context_2d = F.pad(context_2d, (0, pad_len))

    return context_2d


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
):
    """
    Compiles the 4 attention projections (W_q, W_k, W_v, W_o) with optional
    causal healing where W_o absorbs Q,K,V distortion and inter-layer residual drift.
    """
    W_q = weights["q_proj"]
    W_k = weights["k_proj"]
    W_v = weights["v_proj"]
    W_o = weights["o_proj"]
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
        W_hat_q, kb_q, cores_q, _, art_q = nox_hybrid_dmrg_decompose(
            W=W_q, X_calib=X_in, Y_dense=Q_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_q, kb_q, cores_q, _ = nox_hybrid_dmrg_decompose(
            W=W_q, X_calib=X_in, Y_dense=Q_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, verbose=False,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if return_artifacts:
        W_hat_k, kb_k, cores_k, _, art_k = nox_hybrid_dmrg_decompose(
            W=W_k, X_calib=X_in, Y_dense=K_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_k, kb_k, cores_k, _ = nox_hybrid_dmrg_decompose(
            W=W_k, X_calib=X_in, Y_dense=K_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, verbose=False,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if return_artifacts:
        W_hat_v, kb_v, cores_v, _, art_v = nox_hybrid_dmrg_decompose(
            W=W_v, X_calib=X_in, Y_dense=V_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_v, kb_v, cores_v, _ = nox_hybrid_dmrg_decompose(
            W=W_v, X_calib=X_in, Y_dense=V_dense, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, verbose=False,
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
        r_bridge_o = min(r_bridge + 32, max(32, X_in.shape[1] // 2))
        n_outliers_o = n_outliers // 2
        if residual_drift is not None:
            Y_attn_target = Y_attn_dense + alpha_heal * residual_drift
        else:
            Y_attn_target = Y_attn_dense
        if verbose:
            print(f"  [Attention Sub-Block] Causal Healing active: Solving W_o on corrupted context A_comp (bridge: r={r_bridge_o}, outliers={n_outliers_o})...", flush=True)
    else:
        if verbose:
            print("  [Attention Sub-Block] Independent mode: Solving W_o on uncorrupted context A_dense...", flush=True)
        X_o_calib = A_dense
        r_bridge_o = r_bridge
        n_outliers_o = n_outliers
        Y_attn_target = Y_attn_dense

    gamma_o = 0.002 if causal_healing else 0.01
    if return_artifacts:
        W_hat_o, kb_o, cores_o, _, art_o = nox_hybrid_dmrg_decompose(
            W=W_o, X_calib=X_o_calib, Y_dense=Y_attn_target, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_o, n_outliers=n_outliers_o, prior_gamma=gamma_o, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_o, kb_o, cores_o, _ = nox_hybrid_dmrg_decompose(
            W=W_o, X_calib=X_o_calib, Y_dense=Y_attn_target, o_modes=o_modes, i_modes=i_modes,
            ranks=ranks, r_bridge=r_bridge_o, n_outliers=n_outliers_o, prior_gamma=gamma_o, verbose=False,
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
):
    """
    Compiles the 3 SwiGLU MLP projections (W_gate, W_up, W_down) with optional
    causal healing where W_down absorbs non-linear distortion and residual drift.
    """
    W_gate = weights["gate_proj"]
    W_up = weights["up_proj"]
    W_down = weights["down_proj"]

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
        W_hat_gate, kb_gate, cores_gate, _, art_gate = nox_hybrid_dmrg_decompose(
            W=W_gate, X_calib=M_in, Y_dense=H_gate_dense, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_gate, kb_gate, cores_gate, _ = nox_hybrid_dmrg_decompose(
            W=W_gate, X_calib=M_in, Y_dense=H_gate_dense, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, verbose=False,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if return_artifacts:
        W_hat_up, kb_up, cores_up, _, art_up = nox_hybrid_dmrg_decompose(
            W=W_up, X_calib=M_in, Y_dense=H_up_dense, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_up, kb_up, cores_up, _ = nox_hybrid_dmrg_decompose(
            W=W_up, X_calib=M_in, Y_dense=H_up_dense, o_modes=o_modes_up, i_modes=i_modes_up,
            ranks=ranks, r_bridge=r_bridge, n_outliers=n_outliers, verbose=False,
        )
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    H_gate_comp = W_hat_gate @ M_in
    H_up_comp = W_hat_up @ M_in
    H_swiglu_comp = F.silu(H_gate_comp) * H_up_comp

    if causal_healing:
        X_down_calib = H_swiglu_comp
        r_bridge_down = min(r_bridge + 32, max(32, H_swiglu_comp.shape[1] // 2))
        n_outliers_down = n_outliers // 2
        if residual_drift is not None:
            Y_mlp_target = Y_mlp_dense + alpha_heal * residual_drift
        else:
            Y_mlp_target = Y_mlp_dense
        if verbose:
            print(f"  [SwiGLU MLP Sub-Block] Causal Healing active: Solving W_down on corrupted activation H_swiglu_comp (bridge: r={r_bridge_down}, outliers={n_outliers_down})...", flush=True)
    else:
        if verbose:
            print("  [SwiGLU MLP Sub-Block] Independent mode: Solving W_down on uncorrupted activation H_swiglu_dense...", flush=True)
        X_down_calib = H_swiglu_dense
        r_bridge_down = r_bridge
        n_outliers_down = n_outliers
        Y_mlp_target = Y_mlp_dense

    gamma_down = 0.002 if causal_healing else 0.01
    if return_artifacts:
        W_hat_down, kb_down, cores_down, _, art_down = nox_hybrid_dmrg_decompose(
            W=W_down, X_calib=X_down_calib, Y_dense=Y_mlp_target, o_modes=o_modes_down, i_modes=i_modes_down,
            ranks=ranks, r_bridge=r_bridge_down, n_outliers=n_outliers_down, prior_gamma=gamma_down, return_artifacts=True, verbose=False,
        )
    else:
        W_hat_down, kb_down, cores_down, _ = nox_hybrid_dmrg_decompose(
            W=W_down, X_calib=X_down_calib, Y_dense=Y_mlp_target, o_modes=o_modes_down, i_modes=i_modes_down,
            ranks=ranks, r_bridge=r_bridge_down, n_outliers=n_outliers_down, prior_gamma=gamma_down, verbose=False,
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

    norm1_weight = layer_weights.get("input_layernorm", None)
    norm2_weight = layer_weights.get("post_attention_layernorm", None)

    norm1 = RMSNorm(d_model, weight=norm1_weight)
    norm2 = RMSNorm(d_model, weight=norm2_weight)

    ref_calib_dense = X_calib_dense if X_calib_dense is not None else X_calib

    X_norm1_comp = norm1(X_calib)
    X_norm1_dense = norm1(ref_calib_dense)

    drift_attn = (ref_calib_dense - X_calib) if (X_calib_dense is not None and causal_healing) else None

    attn_res = compile_attention_block(
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
    )

    X_res1_dense = ref_calib_dense + attn_res["Y_attn_dense"]
    X_res1_comp = X_calib + attn_res["Y_attn_comp"]

    X_norm2_dense = norm2(X_res1_dense)
    X_norm2_comp = norm2(X_res1_comp) if causal_healing else norm2(X_res1_dense)

    drift_mlp = (X_res1_dense - X_res1_comp) if (X_calib_dense is not None and causal_healing) else None

    mlp_res = compile_mlp_block(
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
        "total_kb": total_kb,
        "time_sec": total_time,
        "causal_healing": causal_healing,
    }
    if return_artifacts:
        res["compressed_artifacts"] = {
            **attn_res["artifacts"],
            **mlp_res["artifacts"],
        }
    return res


@torch.no_grad()
def evaluate_compiled_block(compiled_weights, layer_weights, X_test, n_heads=32, n_kv_heads=32, seq_len=128):
    """
    Evaluates a compiled Transformer block on unseen held-out test tokens.
    """
    d_model = X_test.shape[0]
    norm1 = RMSNorm(d_model, weight=layer_weights.get("input_layernorm", None))
    norm2 = RMSNorm(d_model, weight=layer_weights.get("post_attention_layernorm", None))

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


@torch.no_grad()
def compile_multiverse_block(
    layer_weights,
    ranks=(24, 24, 24),
    core_bits=4,
    verbose=False,
):
    """
    Compiles an entire Transformer layer using the Multiverse (CompactifAI) TT-SVD baseline.
    All 7 projections are decomposed independently with pure Frobenius SVD and post-hoc core quantization.
    """
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
    }
