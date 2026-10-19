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
from nox.compiler import nox_hybrid_dmrg_decompose_from_statistics
from nox.utils import get_modes_for_projection
from nox.utils import compute_metrics
from nox.artifacts import checkpoint_file_sizes, checkpoint_storage_bytes, fingerprint
from nox.defect_bridge import apply_defect_bridge
from nox.calibration import InputCovarianceAccumulator


def compute_dense_argmax_metrics(logits_dense, logits_compressed):
    """Compare compressed logits against dense greedy labels; these are not PPL metrics."""
    if logits_dense.shape != logits_compressed.shape:
        raise ValueError(
            f"Logit shape mismatch: dense={tuple(logits_dense.shape)} "
            f"compressed={tuple(logits_compressed.shape)}"
        )

    targets = logits_dense.argmax(dim=-1)
    dense_ce = F.cross_entropy(logits_dense, targets).item()
    compressed_ce = F.cross_entropy(logits_compressed, targets).item()
    top1 = (logits_compressed.argmax(dim=-1) == targets).float().mean().item() * 100.0
    top_k = min(5, logits_compressed.shape[-1])
    top5 = (logits_compressed.topk(top_k, dim=-1).indices == targets.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0
    cosine = F.cosine_similarity(logits_dense.flatten().unsqueeze(0), logits_compressed.flatten().unsqueeze(0)).item()
    return {
        "target": "dense_argmax",
        "metric_semantics": "cross-entropy to dense greedy tokens; not language-model perplexity",
        "dense_argmax_cross_entropy_dense_logits": dense_ce,
        "dense_argmax_cross_entropy_compressed_logits": compressed_ce,
        "exp_dense_argmax_cross_entropy_dense_logits": float(torch.exp(torch.tensor(dense_ce)).item()),
        "exp_dense_argmax_cross_entropy_compressed_logits": float(torch.exp(torch.tensor(compressed_ce)).item()),
        "logit_cos_sim": cosine,
        "top1_agreement_pct": top1,
        "top5_agreement_pct": top5,
    }


@torch.no_grad()
def forward_transformer_block(
    weights,
    X,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    rms_norm_eps=1e-5,
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

    norm1 = RMSNorm(d_model, eps=rms_norm_eps, weight=weights.get("input_layernorm", None))
    norm2 = RMSNorm(d_model, eps=rms_norm_eps, weight=weights.get("post_attention_layernorm", None))

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
    cross_layer_bridge = weights.get("cross_layer_bridge")
    if cross_layer_bridge is not None:
        X_out = X_out + apply_defect_bridge(X, cross_layer_bridge)

    return X_out, Y_attn, Y_mlp


@torch.no_grad()
def forward_transformer_block_chunked(
    weights,
    X,
    *,
    windows_per_chunk,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    rms_norm_eps=1e-5,
    output_device="cpu",
    return_components=True,
):
    """Run independent calibration windows in bounded-memory batches.

    Chunk boundaries are required to align to complete ``seq_len`` windows.
    Each chunk is passed through the unchanged block forward, so causal masks,
    RoPE positions as represented by the cached hidden states, normalization,
    residuals, and SwiGLU operations have the same per-window semantics as the
    full call.  This function never permits attention across windows.
    """
    if windows_per_chunk <= 0:
        raise ValueError("windows_per_chunk must be positive")
    if X.ndim != 2 or X.shape[1] % seq_len:
        raise ValueError("Calibration columns must contain complete sequence windows")
    token_chunk = int(windows_per_chunk) * int(seq_len)
    outputs = []
    attention_outputs = [] if return_components else None
    mlp_outputs = [] if return_components else None
    device = weights["q_proj"].device
    for start in range(0, X.shape[1], token_chunk):
        stop = min(start + token_chunk, X.shape[1])
        chunk = X[:, start:stop].to(device=device)
        output, attention, mlp = forward_transformer_block(
            weights,
            chunk,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            seq_len=seq_len,
            rms_norm_eps=rms_norm_eps,
        )
        outputs.append(output.to(device=output_device))
        if return_components:
            attention_outputs.append(attention.to(device=output_device))
            mlp_outputs.append(mlp.to(device=output_device))
        del chunk, output, attention, mlp
    return (
        torch.cat(outputs, dim=1),
        torch.cat(attention_outputs, dim=1) if return_components else None,
        torch.cat(mlp_outputs, dim=1) if return_components else None,
    )


@torch.no_grad()
def collect_independent_block_calibration(
    weights,
    X,
    *,
    windows_per_chunk,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    rms_norm_eps=1e-5,
    statistics_device=None,
    accumulation_dtype=torch.float32,
    output_device="cpu",
):
    """Collect exact independent-NEC input covariances for all seven projections.

    The current RunPod experiment uses independent, dense-trajectory targets.
    Therefore q/k/v share the normalized residual-stream covariance, gate/up
    share the normalized post-attention covariance, and each dense target is
    exactly ``W @ X``.  Storing four input covariances is sufficient; projection
    cross-covariances are formed one at a time as ``W @ (X X^T)``.
    """
    if windows_per_chunk <= 0:
        raise ValueError("windows_per_chunk must be positive")
    if X.ndim != 2 or X.shape[1] % seq_len:
        raise ValueError("Calibration columns must contain complete sequence windows")
    device = weights["q_proj"].device
    stats_device = device if statistics_device is None else statistics_device
    d_model = int(weights["q_proj"].shape[1])
    d_intermediate = int(weights["down_proj"].shape[1])
    accumulators = {
        "qkv_input": InputCovarianceAccumulator(
            d_model, device=stats_device, accumulation_dtype=accumulation_dtype,
        ),
        "o_input": InputCovarianceAccumulator(
            d_model, device=stats_device, accumulation_dtype=accumulation_dtype,
        ),
        "gate_up_input": InputCovarianceAccumulator(
            d_model, device=stats_device, accumulation_dtype=accumulation_dtype,
        ),
        "down_input": InputCovarianceAccumulator(
            d_intermediate, device=stats_device, accumulation_dtype=accumulation_dtype,
        ),
    }
    norm1 = RMSNorm(
        d_model, eps=rms_norm_eps, weight=weights.get("input_layernorm", None),
    )
    norm2 = RMSNorm(
        d_model, eps=rms_norm_eps, weight=weights.get("post_attention_layernorm", None),
    )
    outputs = []
    attention_outputs = []
    mlp_outputs = []
    token_chunk = int(windows_per_chunk) * int(seq_len)
    for start in range(0, X.shape[1], token_chunk):
        stop = min(start + token_chunk, X.shape[1])
        x = X[:, start:stop].to(device=device, dtype=weights["q_proj"].dtype)
        x_norm1 = norm1(x)
        q = weights["q_proj"] @ x_norm1
        k = weights["k_proj"] @ x_norm1
        v = weights["v_proj"] @ x_norm1
        attention_input = scaled_dot_product_attention_2d(
            q, k, v, n_heads, n_kv_heads, seq_len,
        )
        attention_output = weights["o_proj"] @ attention_input
        residual1 = x + attention_output
        x_norm2 = norm2(residual1)
        gate = weights["gate_proj"] @ x_norm2
        up = weights["up_proj"] @ x_norm2
        down_input = F.silu(gate) * up
        mlp_output = weights["down_proj"] @ down_input
        output = residual1 + mlp_output

        accumulators["qkv_input"].update(x_norm1)
        accumulators["o_input"].update(attention_input)
        accumulators["gate_up_input"].update(x_norm2)
        accumulators["down_input"].update(down_input)
        outputs.append(output.to(device=output_device))
        attention_outputs.append(attention_output.to(device=output_device))
        mlp_outputs.append(mlp_output.to(device=output_device))
        del x, x_norm1, q, k, v, attention_input, attention_output, residual1, x_norm2, gate, up, down_input, mlp_output, output

    return {
        "covariances": {name: accumulator.finalize() for name, accumulator in accumulators.items()},
        "output": torch.cat(outputs, dim=1),
        "attention_output": torch.cat(attention_outputs, dim=1),
        "mlp_output": torch.cat(mlp_outputs, dim=1),
        "sample_count": int(X.shape[1]),
        "windows_per_chunk": int(windows_per_chunk),
        "seq_len": int(seq_len),
    }


@torch.no_grad()
def compile_independent_transformer_block_streaming(
    weights,
    X,
    *,
    windows_per_chunk,
    ranks=(1, 64, 128, 64, 1),
    r_bridge=128,
    n_outliers=524288,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
    rms_norm_eps=1e-5,
    diagonal_prior_strength=0.0,
    site_order=(0, 1, 2, 3),
    projection_budgets=None,
    mpo_n_sweeps=2,
    mpo_alpha_awt=0.5,
    accumulation_dtype=torch.float32,
    statistics_device=None,
    output_device="cpu",
    return_artifacts=True,
    verbose=True,
):
    """Compile the active independent High-Fidelity block from streamed data.

    This is intentionally narrow: it implements the current independent,
    dense-trajectory, MPO4, zero-healing experiment.  Trajectory-conditioned,
    product-aware, PEES, and defect-bridge experiments remain on their original
    paths because their targets are not reducible to these four covariances.
    """
    started = time.time()
    calibration = collect_independent_block_calibration(
        weights,
        X,
        windows_per_chunk=windows_per_chunk,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        rms_norm_eps=rms_norm_eps,
        statistics_device=statistics_device,
        accumulation_dtype=accumulation_dtype,
        output_device=output_device,
    )
    covariance_for_projection = {
        "q_proj": "qkv_input",
        "k_proj": "qkv_input",
        "v_proj": "qkv_input",
        "o_proj": "o_input",
        "gate_proj": "gate_up_input",
        "up_proj": "gate_up_input",
        "down_proj": "down_input",
    }
    compressed_weights = {}
    artifacts = {}
    total_kb = 0.0
    factorization_cache = {}
    active_covariance_name = None
    for name in covariance_for_projection:
        covariance_name = covariance_for_projection[name]
        if active_covariance_name is not None and covariance_name != active_covariance_name:
            factorization_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        active_covariance_name = covariance_name
        budget = {} if projection_budgets is None else projection_budgets.get(name, {})
        projection_bridge = int(budget.get("r_bridge", r_bridge))
        projection_outliers = int(budget.get("n_outliers", n_outliers))
        weight = weights[name]
        o_modes, i_modes = get_modes_for_projection(weight.shape[0], weight.shape[1])
        statistics = calibration["covariances"][
            covariance_for_projection[name]
        ].projection_statistics(weight)
        result = nox_hybrid_dmrg_decompose_from_statistics(
            W=weight,
            calibration_statistics=statistics,
            o_modes=o_modes,
            i_modes=i_modes,
            ranks=ranks,
            r_bridge=projection_bridge,
            n_outliers=projection_outliers,
            n_sweeps=mpo_n_sweeps,
            prior_gamma=0.01,
            alpha_awt=mpo_alpha_awt,
            return_artifacts=return_artifacts,
            verbose=verbose,
            diagonal_prior_strength=diagonal_prior_strength,
            site_order=site_order,
            statistics_factorization_cache=factorization_cache,
        )
        compressed_weights[name] = result[0]
        total_kb += float(result[1])
        if return_artifacts:
            artifacts[name] = result[4]
        del statistics, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    factorization_cache.clear()

    compressed_output, compressed_attention, compressed_mlp = forward_transformer_block_chunked(
        {**weights, **compressed_weights},
        X,
        windows_per_chunk=windows_per_chunk,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        rms_norm_eps=rms_norm_eps,
        output_device=output_device,
    )
    dense_output = calibration["output"]
    dense_attention = calibration["attention_output"]
    dense_mlp = calibration["mlp_output"]
    result = {
        "compressed_weights": compressed_weights,
        "X_out_comp": compressed_output,
        "X_out_dense": dense_output,
        "block_metrics": compute_metrics(dense_output, compressed_output),
        "transform_metrics": compute_metrics(
            dense_attention + dense_mlp, compressed_attention + compressed_mlp,
        ),
        "attn_metrics": compute_metrics(dense_attention, compressed_attention),
        "mlp_metrics": compute_metrics(dense_mlp, compressed_mlp),
        "mlp_product_aware": None,
        "total_kb": total_kb,
        "time_sec": time.time() - started,
        "causal_healing": False,
        "trajectory_mode": "none",
        "calibration_backend": "streaming_sufficient_statistics",
        "calibration_windows_per_chunk": int(windows_per_chunk),
    }
    if return_artifacts:
        result["compressed_artifacts"] = artifacts
    return result


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
            "description": "High-fidelity experimental configuration; quality and serialized size require measurement",
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
    save_independent_checkpoint_dir=None,
    save_validation_activations_dir=None,
    skip_independent=False,
    skip_baselines=False,
    final_norm_weight=None,
    lm_head_weight=None,
    input_data_metadata=None,
    run_identity=None,
    run_fingerprint=None,
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
        save_independent_checkpoint_dir: Optional directory for independently
            calibrated (no-healing) checkpoints, for downstream PPL evaluation.
        save_validation_activations_dir: Optional directory for dense held-out
            layer inputs used by checkpoint reconstruction/output audits.
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

        if save_validation_activations_dir:
            os.makedirs(save_validation_activations_dir, exist_ok=True)
            activation_path = os.path.join(
                save_validation_activations_dir,
                f"dense_input_layer_{l_idx}.pt",
            )
            torch.save(X_d_test.detach().to(device="cpu", dtype=torch.float32), activation_path)

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
            if verbose:
                print(f"  [Layer {l_idx}] Compiling Multiverse baseline block...", flush=True)
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
            if verbose:
                print(f"  [Layer {l_idx}] Compiling Nox Independent baseline block...", flush=True)
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
                return_artifacts=bool(save_independent_checkpoint_dir),
                verbose=False,
            )
            indep_weights = dict(weights)
            indep_weights.update(nox_indep_block["compressed_weights"])
            X_indep_test_next, Y_attn_indep, Y_mlp_indep = forward_transformer_block(
                indep_weights, X_indep_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len
            )
            Delta_indep_test = Y_attn_indep + Y_mlp_indep
            indep_kb = nox_indep_block["total_kb"]
            if save_independent_checkpoint_dir:
                os.makedirs(save_independent_checkpoint_dir, exist_ok=True)
                independent_checkpoint = {
                    "layer_idx": l_idx,
                    "compression_path": "no_healing_independent",
                    "compressed_artifacts": nox_indep_block["compressed_artifacts"],
                    "input_layernorm": weights["input_layernorm"].cpu() if "input_layernorm" in weights else None,
                    "post_attention_layernorm": weights["post_attention_layernorm"].cpu() if "post_attention_layernorm" in weights else None,
                    "total_kb": indep_kb,
                }
                independent_path = os.path.join(
                    save_independent_checkpoint_dir,
                    f"compressed_layer_{l_idx}.pt",
                )
                torch.save(independent_checkpoint, independent_path)
                if verbose:
                    print(
                        f"  [Checkpoint] Saved no-healing Layer {l_idx} "
                        f"({indep_kb:.1f} KB estimated payload) -> '{independent_path}'",
                        flush=True,
                    )
            del nox_indep_block, indep_weights, Y_attn_indep, Y_mlp_indep
            if str(device).startswith("cuda"):
                gc.collect()
                torch.cuda.empty_cache()

        # 4. Nox Causally-Healed path (inter-layer: calibrated on corrupted X_healed_calib targeting clean X_d_calib)
        if verbose:
            print(f"  [Layer {l_idx}] Compiling Nox Causal Healed block (inter-layer)...", flush=True)
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
            verbose=verbose,
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

        final_logit_metrics = compute_dense_argmax_metrics(logits_d, logits_healed)

        if verbose:
            print(" Exit Residual Stream -> RMSNorm -> lm_head (32,000 Vocab Logits) across held-out tokens:")
            print("  - Reference targets: dense model's greedy top-1 token (not dataset labels)")
            print(f"  - Dense self cross-entropy:                 {final_logit_metrics['dense_argmax_cross_entropy_dense_logits']:.4f}")
            print(f"  - Compressed vs dense-argmax cross-entropy: {final_logit_metrics['dense_argmax_cross_entropy_compressed_logits']:.4f}")
            print(f"  - Compressed top-1 agreement:               {final_logit_metrics['top1_agreement_pct']:.2f}%")
            print(f"  - Compressed top-5 agreement:               {final_logit_metrics['top5_agreement_pct']:.2f}%")
            print(f"  - Logit cosine similarity:                  {final_logit_metrics['logit_cos_sim']:.4f}")
            print("  - These are logit-agreement diagnostics, not language-model perplexity.")
            print("=" * 120 + "\n")

    if save_checkpoint_dir:
        manifest_path = os.path.join(save_checkpoint_dir, "model_manifest.json")
        total_model_kb = sum(h["causal_healed"]["kb"] for h in history)
        checkpoint_sizes = checkpoint_file_sizes(save_checkpoint_dir, layer_indices)
        serialized_checkpoint_bytes = checkpoint_storage_bytes(save_checkpoint_dir, layer_indices)
        manifest_data = {
            "tier": tier,
            "config": cfg,
            "run_identity": run_identity,
            "run_fingerprint": run_fingerprint,
            "input_data": input_data_metadata,
            "layers_compressed": list(layer_indices),
            "num_layers": len(layer_indices),
            "estimated_tensor_payload_kb": total_model_kb,
            "serialized_checkpoint_bytes": serialized_checkpoint_bytes,
            "serialized_checkpoint_mib": serialized_checkpoint_bytes / (1024.0 * 1024.0),
            "checkpoint_file_sizes": checkpoint_sizes,
            # Legacy field retained for reading older manifests. It is the compiler's
            # payload estimate, not the serialized bytes on disk.
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
            print(
                f"\n[Checkpoint Manifest] Wrote '{manifest_path}': estimated tensor payload "
                f"{total_model_kb / 1024.0:.2f} MiB; serialized checkpoint files "
                f"{serialized_checkpoint_bytes / (1024.0 * 1024.0):.2f} MiB.",
                flush=True,
            )

    if save_independent_checkpoint_dir:
        independent_identity = {
            **(run_identity or {}),
            "compression_path": "no_healing_independent",
        }
        independent_fingerprint = fingerprint(independent_identity)
        independent_sizes = checkpoint_file_sizes(save_independent_checkpoint_dir, layer_indices)
        independent_bytes = checkpoint_storage_bytes(save_independent_checkpoint_dir, layer_indices)
        independent_payload_kb = sum(h["independent"]["kb"] for h in history)
        independent_manifest = {
            "compression_path": "no_healing_independent",
            "tier": tier,
            "config": cfg,
            "run_identity": independent_identity,
            "run_fingerprint": independent_fingerprint,
            "input_data": input_data_metadata,
            "layers_compressed": list(layer_indices),
            "num_layers": len(layer_indices),
            "estimated_tensor_payload_kb": independent_payload_kb,
            "serialized_checkpoint_bytes": independent_bytes,
            "serialized_checkpoint_mib": independent_bytes / (1024.0 * 1024.0),
            "checkpoint_file_sizes": independent_sizes,
            "first_layer_rel_err": history[0]["independent"]["block"]["relative_error"],
            "last_layer_rel_err": history[-1]["independent"]["block"]["relative_error"],
            "drift_growth": (
                history[-1]["independent"]["block"]["relative_error"]
                - history[0]["independent"]["block"]["relative_error"]
            ),
        }
        manifest_path = os.path.join(save_independent_checkpoint_dir, "model_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(independent_manifest, f, indent=2)
        if verbose:
            print(
                f"\n[No-Healing Manifest] Wrote '{manifest_path}': serialized checkpoint files "
                f"{independent_bytes / (1024.0 * 1024.0):.2f} MiB.",
                flush=True,
            )

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
