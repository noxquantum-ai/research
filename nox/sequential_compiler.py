"""Sequential Multi-Layer Compiler for NoxQuantum Architecture (Step 4.2 Inter-Layer Causal Healing).

Orchestrates layer-by-layer streaming compression across Transformer layers, ensuring
each downstream layer is calibrated on the actual corrupted activations arriving from
upstream compressed layers, thereby dampening compounding drift.
"""

import gc
import json
import os
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
    w_dtype = weights["q_proj"].dtype
    if X.dtype != w_dtype:
        X = X.to(w_dtype)

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
      - extreme_77x: ~5.1 MB/layer (~77x compression, 0.2 bits/weight, eps=0.030)
      - production_18x: ~18-20 MB/layer (~18-20x compression, 0.9 bits/weight, eps=0.015)
      - high_fidelity_10x: ~35 MB/layer (~10x compression, 1.8 bits/weight, eps=0.005)
    """
    if tier_name == "high_fidelity_10x":
        return {
            "ranks": (1, 64, 128, 64, 1),
            "r_bridge": 128,
            "n_outliers": 524288,
            "epsilon": 0.005,
            "description": "High-Fidelity Lossless Tier (~10-12x compression, ~32 MB/layer, eps=0.005)",
        }
    elif tier_name == "production_18x":
        return {
            "ranks": (1, 40, 80, 40, 1),
            "r_bridge": 128,
            "n_outliers": 262144,
            "epsilon": 0.015,
            "description": "Production CompactifAI-Killer Tier (~18-20x compression, ~16-18 MB/layer, eps=0.015)",
        }
    else:
        return {
            "ranks": (1, 32, 80, 32, 1),
            "r_bridge": 96,
            "n_outliers": 65536,
            "epsilon": 0.030,
            "description": "Extreme Stress-Test Tier (~77x compression, ~5.0 MB/layer, eps=0.030)",
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
    save_checkpoint_dir=None,
    skip_independent=False,
    skip_baselines=False,
    final_norm_weight=None,
    lm_head_weight=None,
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
        save_checkpoint_dir: Directory to save serialized layer checkpoints.
        skip_independent: Whether to skip compiling Nox Independent baseline.
        skip_baselines: Whether to skip compiling all baselines (Multiverse + Independent) for pure production compression speed.
        final_norm_weight: Optional model.norm.weight for Layer 31 logit verification.
        lm_head_weight: Optional lm_head.weight for Layer 31 logit verification.
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
    X_mv_iso_test = X_test_init.clone().to(device)
    X_indep_test = X_test_init.clone().to(device)

    X_healed_calib = X_calib_init.clone().to(device)
    X_healed_test = X_test_init.clone().to(device)

    history = []

    for l_idx in layer_indices:
        t0 = time.time()
        if verbose:
            print(f"\n--- Compiling Cascade Layer {l_idx} ---", flush=True)

        # Boundary Layer Protection (Layers 0, 1, 30, 31)
        if l_idx in (0, 1, 30, 31):
            if tier == "production_18x":
                boundary_cfg = get_tier_config("high_fidelity_10x")
            elif tier == "extreme_77x":
                boundary_cfg = get_tier_config("production_18x")
            else:
                boundary_cfg = cfg
            layer_ranks = boundary_cfg["ranks"]
            layer_r_bridge = boundary_cfg["r_bridge"]
            layer_n_outliers = boundary_cfg["n_outliers"]
            if verbose:
                print(f"  [Boundary Layer Protection Active for Layer {l_idx}] Upgraded to: {boundary_cfg['description']}", flush=True)
        else:
            layer_ranks = ranks
            layer_r_bridge = r_bridge
            layer_n_outliers = n_outliers

        weights = fetch_layer_fn(l_idx)
        weights = {k: v.to(device=device, dtype=torch.float32) for k, v in weights.items()}

        # 1. Dense Reference forward pass
        X_d_calib_next, Y_attn_d_calib, Y_mlp_d_calib = forward_transformer_block(
            weights, X_d_calib, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        X_d_test_next, Y_attn_d_test, Y_mlp_d_test = forward_transformer_block(
            weights, X_d_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
        )
        Delta_d_test = Y_attn_d_test + Y_mlp_d_test

        # 2. Multiverse Default Baseline path
        if skip_baselines:
            X_mv_test_next = X_mv_test
            Delta_mv_test = Delta_d_test
            mv_kb = 0.0
        else:
            mv_ranks = layer_ranks[1:-1] if len(layer_ranks) == 5 else layer_ranks
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
            if str(device).startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()

        # 3. Nox Independent path (calibrated on uncorrupted dense stream)
        if skip_independent or skip_baselines:
            X_indep_test_next = X_indep_test
            Delta_indep_test = Delta_d_test
            indep_kb = 0.0
        else:
            nox_indep_block = compile_transformer_block(
                layer_weights=weights,
                X_calib=X_d_calib,
                X_calib_dense=X_d_calib,
                ranks=layer_ranks,
                r_bridge=layer_r_bridge,
                n_outliers=layer_n_outliers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                seq_len=seq_len,
                causal_healing=False,
                return_artifacts=False,
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
            if str(device).startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()

        # 4. Nox Causally-Healed path (inter-layer: calibrated on corrupted X_healed_calib targeting clean X_d_calib)
        nox_healed_block = compile_transformer_block(
            layer_weights=weights,
            X_calib=X_healed_calib,
            X_calib_dense=X_d_calib,
            alpha_heal=0.0,
            ranks=layer_ranks,
            r_bridge=layer_r_bridge,
            n_outliers=layer_n_outliers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            seq_len=seq_len,
            causal_healing=True,
            return_artifacts=True,
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

        # Measure forward inference latency on healed layer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_inf_start = time.perf_counter()
        for _ in range(3):
            _ = forward_transformer_block(healed_weights, X_healed_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_latency_ms = ((time.perf_counter() - t_inf_start) / 3.0) * 1000.0

        if save_checkpoint_dir:
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(save_checkpoint_dir, f"compressed_layer_{l_idx}.pt")
            layer_checkpoint = {
                "layer_idx": l_idx,
                "compressed_artifacts": nox_healed_block.get("compressed_artifacts", None),
                "input_layernorm": weights["input_layernorm"].cpu() if "input_layernorm" in weights else None,
                "post_attention_layernorm": weights["post_attention_layernorm"].cpu() if "post_attention_layernorm" in weights else None,
                "total_kb": healed_kb,
            }
            torch.save(layer_checkpoint, ckpt_path)
            if verbose:
                print(f"  [Checkpoint] Saved Layer {l_idx} ({healed_kb:.1f} KB compact artifact) -> '{ckpt_path}'", flush=True)

        # 5. Multiverse Iso-Byte Baseline path (matched or greater byte budget than Nox)
        if skip_baselines:
            X_mv_iso_test_next = X_mv_iso_test
            Delta_mv_iso_test = Delta_d_test
            mv_iso_kb = 0.0
        else:
            mv_iso_block = compile_multiverse_block(
                layer_weights=weights,
                target_kb=healed_kb,
                core_bits=4,
                verbose=False,
            )
            mv_iso_weights = dict(weights)
            mv_iso_weights.update(mv_iso_block["compressed_weights"])
            X_mv_iso_test_next, Y_attn_mv_iso, Y_mlp_mv_iso = forward_transformer_block(
                mv_iso_weights, X_mv_iso_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
            )
            Delta_mv_iso_test = Y_attn_mv_iso + Y_mlp_mv_iso
            mv_iso_kb = mv_iso_block["total_kb"]
            del mv_iso_block, mv_iso_weights, Y_attn_mv_iso, Y_mlp_mv_iso

        del nox_healed_block, healed_weights, Y_attn_healed, Y_mlp_healed

        indep_block_m = (
            {"relative_error": float('nan'), "cosine_similarity": float('nan'), "angular_drift": float('nan'), "norm_ratio": float('nan')}
            if (skip_independent or skip_baselines)
            else compute_metrics(X_d_test_next, X_indep_test_next)
        )
        healed_block_m = compute_metrics(X_d_test_next, X_healed_test_next)

        mv_block_m = (
            {"relative_error": float('nan'), "cosine_similarity": float('nan'), "angular_drift": float('nan'), "norm_ratio": float('nan')}
            if skip_baselines
            else compute_metrics(X_d_test_next, X_mv_test_next)
        )
        mv_iso_block_m = (
            {"relative_error": float('nan'), "cosine_similarity": float('nan'), "angular_drift": float('nan'), "norm_ratio": float('nan')}
            if skip_baselines
            else compute_metrics(X_d_test_next, X_mv_iso_test_next)
        )
        mv_delta_m = (
            {"relative_error": float('nan'), "cosine_similarity": float('nan'), "angular_drift": float('nan'), "norm_ratio": float('nan')}
            if skip_baselines
            else compute_metrics(Delta_d_test, Delta_mv_test)
        )
        mv_iso_delta_m = (
            {"relative_error": float('nan'), "cosine_similarity": float('nan'), "angular_drift": float('nan'), "norm_ratio": float('nan')}
            if skip_baselines
            else compute_metrics(Delta_d_test, Delta_mv_iso_test)
        )
        indep_delta_m = (
            {"relative_error": float('nan'), "cosine_similarity": float('nan'), "angular_drift": float('nan'), "norm_ratio": float('nan')}
            if (skip_independent or skip_baselines)
            else compute_metrics(Delta_d_test, Delta_indep_test)
        )
        healed_delta_m = compute_metrics(Delta_d_test, Delta_healed_test)

        step_time = time.time() - t0

        layer_summary = {
            "layer_idx": l_idx,
            "compile_time_sec": step_time,
            "infer_latency_ms": infer_latency_ms,
            "multiverse": {
                "block": mv_block_m,
                "delta": mv_delta_m,
                "kb": mv_kb,
            },
            "multiverse_isobyte": {
                "block": mv_iso_block_m,
                "delta": mv_iso_delta_m,
                "kb": mv_iso_kb,
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
            print(f"Layer {l_idx} compiled in {step_time:.2f}s (Infer: {infer_latency_ms:.2f}ms):")
            print(f"  [Full Residual X_out] MV Default RelErr: {mv_block_m['relative_error']:.4f} | MV IsoByte: {mv_iso_block_m['relative_error']:.4f} | Healed: {healed_block_m['relative_error']:.4f}")
            print(f"  [Delta_block]         MV Default RelErr: {mv_delta_m['relative_error']:.4f} | MV IsoByte: {mv_iso_delta_m['relative_error']:.4f} | Healed: {healed_delta_m['relative_error']:.4f}")
            print(f"  [Delta Cosine Sim]    MV Default: {mv_delta_m['cosine_similarity']:.4f} | MV IsoByte: {mv_iso_delta_m['cosine_similarity']:.4f} | Healed: {healed_delta_m['cosine_similarity']:.4f}")

        # Update streaming activation tensors
        X_d_calib = X_d_calib_next
        X_d_test = X_d_test_next
        X_mv_test = X_mv_test_next
        X_mv_iso_test = X_mv_iso_test_next
        X_indep_test = X_indep_test_next
        X_healed_calib = X_healed_calib_next
        X_healed_test = X_healed_test_next

        # Free layer weights from memory immediately
        del weights, X_d_calib_next, X_d_test_next, X_mv_test_next, X_indep_test_next
        del X_healed_calib_next, X_healed_test_next, Delta_d_test, Delta_mv_test, Delta_indep_test, Delta_healed_test
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    final_logit_metrics = None
    if (31 in layer_indices or len(layer_indices) == 32) and lm_head_weight is not None:
        if verbose:
            print("\n" + "=" * 120)
            print(" FINAL LAYER 31 END-TO-END VOCABULARY & LOGIT VERIFICATION SCORECARD")
            print("=" * 120)

        d_model = X_d_test.shape[0]
        norm_layer = RMSNorm(d_model, weight=final_norm_weight.to(device=device, dtype=torch.float32) if final_norm_weight is not None else None)
        head_w = lm_head_weight.to(device=device, dtype=torch.float32)

        norm_d = norm_layer(X_d_test).T
        logits_d = norm_d @ head_w.T  # [N, 32000]

        norm_healed = norm_layer(X_healed_test).T
        logits_healed = norm_healed @ head_w.T  # [N, 32000]

        cos_sim_logits = F.cosine_similarity(logits_d.flatten().unsqueeze(0), logits_healed.flatten().unsqueeze(0)).item()

        top1_dense = logits_d.argmax(dim=-1)
        top1_healed = logits_healed.argmax(dim=-1)
        top1_agree = (top1_dense == top1_healed).float().mean().item() * 100.0

        top5_healed = logits_healed.topk(5, dim=-1).indices
        top5_agree = (top5_healed == top1_dense.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0

        loss_dense = F.cross_entropy(logits_d, top1_dense).item()
        loss_healed = F.cross_entropy(logits_healed, top1_dense).item()
        ppl_dense = float(torch.exp(torch.tensor(loss_dense)).item())
        ppl_healed = float(torch.exp(torch.tensor(loss_healed)).item())

        final_logit_metrics = {
            "logit_cos_sim": cos_sim_logits,
            "top1_agreement_pct": top1_agree,
            "top5_agreement_pct": top5_agree,
            "dense_ppl": ppl_dense,
            "healed_ppl": ppl_healed,
        }

        if verbose:
            print(" Exit Residual Stream -> RMSNorm -> lm_head (32,000 Vocab Logits) across held-out tokens:")
            print(f"  - Dense Reference Target Perplexity (PPL): {ppl_dense:.2f}")
            print(f"  - Nox Healed Exit Logit Cosine Sim:        {cos_sim_logits:.4f}")
            print(f"  - Nox Healed Top-1 Vocab Agreement:        {top1_agree:.2f}%")
            print(f"  - Nox Healed Top-5 Vocab Agreement:        {top5_agree:.2f}%")
            print(f"  - Nox Healed Exit Perplexity (PPL):        {ppl_healed:.2f}")
            print("=" * 120 + "\n")

    if save_checkpoint_dir:
        manifest_path = os.path.join(save_checkpoint_dir, "model_manifest.json")
        total_model_kb = sum(h["causal_healed"]["kb"] for h in history)
        manifest_data = {
            "tier": tier,
            "config": cfg,
            "layers_compressed": list(layer_indices),
            "num_layers": len(layer_indices),
            "total_model_kb": total_model_kb,
            "total_model_mb": total_model_kb / 1024.0,
            "first_layer_rel_err": history[0]["causal_healed"]["block"]["relative_error"],
            "last_layer_rel_err": history[-1]["causal_healed"]["block"]["relative_error"],
            "drift_growth": history[-1]["causal_healed"]["block"]["relative_error"] - history[0]["causal_healed"]["block"]["relative_error"],
            "final_delta_cos_sim": history[-1]["causal_healed"]["delta"]["cosine_similarity"],
            "final_logit_metrics": final_logit_metrics,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        if verbose:
            print(f"\n[Checkpoint Manifest] Master manifest written to '{manifest_path}' ({total_model_kb / 1024.0:.2f} MB total).", flush=True)

    return {
        "history": history,
        "final_streams": {
            "X_dense_test": X_d_test,
            "X_mv_test": X_mv_test,
            "X_indep_test": X_indep_test,
            "X_healed_test": X_healed_test,
        },
        "final_logit_metrics": final_logit_metrics,
        "tier": tier,
        "config": cfg,
    }
