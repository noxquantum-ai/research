"""Sequential Multi-Layer Compiler for NoxQuantum Architecture (Step 4.2 Inter-Layer Causal Healing).

Orchestrates layer-by-layer streaming compression across Transformer layers, ensuring
each downstream layer is calibrated on the actual corrupted activations arriving from
upstream compressed layers, thereby dampening compounding drift.
"""

import gc
import time
import torch
import torch.nn.functional as F
from nox.block_compiler import (
    RMSNorm,
    scaled_dot_product_attention_2d,
    compile_transformer_block,
    compile_multiverse_block,
)
from nox.utils import compute_metrics


@torch.no_grad()
def forward_transformer_block(
    weights,
    X,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
):
    """
    Executes a complete forward pass through one Transformer layer (Attention + SwiGLU MLP).
    
    Supports both dense weight dictionaries and compressed weight dictionaries.
    
    Args:
        weights: Dictionary containing q_proj, k_proj, v_proj, o_proj,
                 gate_proj, up_proj, down_proj, and optional norm weights.
        X: Input activation tensor of shape [d_model, N].
        n_heads: Number of query attention heads.
        n_kv_heads: Number of key/value heads.
        seq_len: Sequence chunk length for attention masking.
        
    Returns:
        Tuple of (X_out, Y_attn, Y_mlp)
    """
    d_model = X.shape[0]
    norm1 = RMSNorm(d_model, weight=weights.get("input_layernorm", None))
    norm2 = RMSNorm(d_model, weight=weights.get("post_attention_layernorm", None))

    # Attention sub-block
    X_norm1 = norm1(X)
    Q = weights["q_proj"] @ X_norm1
    K = weights["k_proj"] @ X_norm1
    V = weights["v_proj"] @ X_norm1
    A = scaled_dot_product_attention_2d(Q, K, V, n_heads, n_kv_heads, seq_len)
    Y_attn = weights["o_proj"] @ A
    X_res1 = X + Y_attn

    # SwiGLU MLP sub-block
    X_norm2 = norm2(X_res1)
    H_gate = weights["gate_proj"] @ X_norm2
    H_up = weights["up_proj"] @ X_norm2
    H_swiglu = F.silu(H_gate) * H_up
    Y_mlp = weights["down_proj"] @ H_swiglu
    X_out = X_res1 + Y_mlp

    return X_out, Y_attn, Y_mlp


def get_tier_config(tier_name="extreme_77x"):
    """
    Returns compression hyperparameter configurations for different operating tiers.
    
    Supported tiers:
      - extreme_77x: ~5 MB/layer (~77x compression, 0.2 bits/weight)
      - production_18x: ~20 MB/layer (~18x compression, 0.9 bits/weight)
    """
    if tier_name == "production_18x":
        return {
            "ranks": (1, 48, 128, 48, 1),
            "r_bridge": 256,
            "n_outliers": 524288,
            "description": "Production CompactifAI-Killer Tier (~18x compression, ~20 MB/layer)",
        }
    else:
        return {
            "ranks": (1, 24, 64, 24, 1),
            "r_bridge": 64,
            "n_outliers": 262144,
            "description": "Extreme Stress-Test Tier (~77x compression, ~5.1 MB/layer)",
        }


@torch.no_grad()
def compile_sequential_cascade(
    fetch_layer_fn,
    layer_indices,
    X_calib_init,
    X_test_init,
    tier="extreme_77x",
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    device="cpu",
    verbose=True,
):
    """
    Executes a multi-layer sequential cascade test across specified layer indices.
    
    Streams layers one at a time to keep peak memory below 2.0 GB.
    Propagates actual compressed activations forward so each layer heals
    upstream residual drift.
    
    Args:
        fetch_layer_fn: Callable(layer_idx) returning dictionary of layer weights.
        layer_indices: Iterable of integer layer indices (e.g. range(15, 19)).
        X_calib_init: Initial calibration activations [d_model, n_calib] entering first layer.
        X_test_init: Initial test activations [d_model, n_test] entering first layer.
        tier: Compression configuration tier ('extreme_77x' or 'production_18x').
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        seq_len: Sequence chunk length.
        device: Device to execute computation on ('cuda' or 'cpu').
        verbose: Whether to log step-by-step progress.
        
    Returns:
        Dictionary containing layer-by-layer cascade history and metrics.
    """
    cfg = get_tier_config(tier)
    ranks = cfg["ranks"]
    r_bridge = cfg["r_bridge"]
    n_outliers = cfg["n_outliers"]

    if verbose:
        print(f"\nInitializing Sequential Cascade across layers {list(layer_indices)}", flush=True)
        print(f"Configuration: {cfg['description']}", flush=True)

    # State tracking: activation streams
    X_d_calib = X_calib_init.clone().to(device)
    X_d_test = X_test_init.clone().to(device)

    X_mv_test = X_test_init.clone().to(device)
    X_indep_test = X_test_init.clone().to(device)

    X_healed_calib = X_calib_init.clone().to(device)
    X_healed_test = X_test_init.clone().to(device)

    history = []

    for l_idx in layer_indices:
        t0 = time.time()
        if verbose:
            print(f"\n--- Compiling Cascade Layer {l_idx} ---", flush=True)

        weights = fetch_layer_fn(l_idx)
        weights = {k: v.to(device) for k, v in weights.items()}

        # 1. Dense Reference forward pass
        X_d_calib_next, Y_attn_d_calib, Y_mlp_d_calib = forward_transformer_block(
            weights, X_d_calib, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        X_d_test_next, Y_attn_d_test, Y_mlp_d_test = forward_transformer_block(
            weights, X_d_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        Delta_d_test = Y_attn_d_test + Y_mlp_d_test

        # 2. Multiverse Baseline path
        mv_ranks = ranks[1:-1] if len(ranks) == 5 else ranks
        mv_block = compile_multiverse_block(
            layer_weights=weights,
            ranks=mv_ranks,
            core_bits=4,
            verbose=False,
        )
        mv_weights = dict(weights)
        mv_weights.update(mv_block["compressed_weights"])
        X_mv_test_next, Y_attn_mv, Y_mlp_mv = forward_transformer_block(
            mv_weights, X_mv_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        Delta_mv_test = Y_attn_mv + Y_mlp_mv
        mv_kb = mv_block["total_kb"]
        del mv_block, mv_weights, Y_attn_mv, Y_mlp_mv

        # 3. Nox Independent path (calibrated on uncorrupted dense stream)
        nox_indep_block = compile_transformer_block(
            layer_weights=weights,
            X_calib=X_d_calib,
            ranks=ranks,
            r_bridge=r_bridge,
            n_outliers=n_outliers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            seq_len=seq_len,
            causal_healing=False,
            verbose=False,
        )
        indep_weights = dict(weights)
        indep_weights.update(nox_indep_block["compressed_weights"])
        X_indep_test_next, Y_attn_indep, Y_mlp_indep = forward_transformer_block(
            indep_weights, X_indep_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        Delta_indep_test = Y_attn_indep + Y_mlp_indep
        indep_kb = nox_indep_block["total_kb"]
        del nox_indep_block, indep_weights, Y_attn_indep, Y_mlp_indep

        # 4. Nox Causally-Healed path (inter-layer: calibrated on corrupted X_healed_calib)
        nox_healed_block = compile_transformer_block(
            layer_weights=weights,
            X_calib=X_healed_calib,
            ranks=ranks,
            r_bridge=r_bridge,
            n_outliers=n_outliers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            seq_len=seq_len,
            causal_healing=True,
            verbose=False,
        )
        healed_weights = dict(weights)
        healed_weights.update(nox_healed_block["compressed_weights"])

        # Advance calibration stream for subsequent layer
        X_healed_calib_next, _, _ = forward_transformer_block(
            healed_weights, X_healed_calib, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        # Advance test stream for held-out evaluation
        X_healed_test_next, Y_attn_healed, Y_mlp_healed = forward_transformer_block(
            healed_weights, X_healed_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        Delta_healed_test = Y_attn_healed + Y_mlp_healed
        healed_kb = nox_healed_block["total_kb"]
        del nox_healed_block, healed_weights, Y_attn_healed, Y_mlp_healed

        # Compute cascading evaluation metrics
        mv_block_m = compute_metrics(X_d_test_next, X_mv_test_next)
        indep_block_m = compute_metrics(X_d_test_next, X_indep_test_next)
        healed_block_m = compute_metrics(X_d_test_next, X_healed_test_next)

        mv_delta_m = compute_metrics(Delta_d_test, Delta_mv_test)
        indep_delta_m = compute_metrics(Delta_d_test, Delta_indep_test)
        healed_delta_m = compute_metrics(Delta_d_test, Delta_healed_test)

        step_time = time.time() - t0

        layer_summary = {
            "layer_idx": l_idx,
            "time_sec": step_time,
            "multiverse": {
                "block": mv_block_m,
                "delta": mv_delta_m,
                "kb": mv_kb,
            },
            "independent": {
                "block": indep_block_m,
                "delta": indep_delta_m,
                "kb": indep_kb,
            },
            "causal_healed": {
                "block": healed_block_m,
                "delta": healed_delta_m,
                "kb": healed_kb,
            },
        }
        history.append(layer_summary)

        if verbose:
            print(f"Layer {l_idx} compiled in {step_time:.2f}s:")
            print(f"  [Full Residual X_out] Multiverse RelErr: {mv_block_m['relative_error']:.4f} | Indep: {indep_block_m['relative_error']:.4f} | Healed: {healed_block_m['relative_error']:.4f}")
            print(f"  [Delta_block]         Multiverse RelErr: {mv_delta_m['relative_error']:.4f} | Indep: {indep_delta_m['relative_error']:.4f} | Healed: {healed_delta_m['relative_error']:.4f}")
            print(f"  [Delta Cosine Sim]    Multiverse: {mv_delta_m['cosine_similarity']:.4f} | Indep: {indep_delta_m['cosine_similarity']:.4f} | Healed: {healed_delta_m['cosine_similarity']:.4f}")

        # Update streaming activation tensors
        X_d_calib = X_d_calib_next
        X_d_test = X_d_test_next
        X_mv_test = X_mv_test_next
        X_indep_test = X_indep_test_next
        X_healed_calib = X_healed_calib_next
        X_healed_test = X_healed_test_next

        # Free layer weights from memory immediately
        del weights, X_d_calib_next, X_d_test_next, X_mv_test_next, X_indep_test_next
        del X_healed_calib_next, X_healed_test_next, Delta_d_test, Delta_mv_test, Delta_indep_test, Delta_healed_test
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return {
        "history": history,
        "final_streams": {
            "X_dense_test": X_d_test,
            "X_mv_test": X_mv_test,
            "X_indep_test": X_indep_test,
            "X_healed_test": X_healed_test,
        },
        "tier": tier,
        "config": cfg,
    }
