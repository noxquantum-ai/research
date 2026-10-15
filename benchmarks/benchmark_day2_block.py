"""Day 2 Benchmark: Full Single-Block Causal Healing Test across Layer 15 (Step 4.2).

Demonstrates that Causal Intra-Block Propagation (W_o healing Q, K, V distortion
and W_down healing W_gate, W_up non-linear Hadamard distortion) slashes full-block
output drift compared to both independent matrix compression and unhealed Multiverse baseline.
"""

import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import struct
import sys
import time
import urllib.request
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, v = stripped.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

HF_TOKEN = os.getenv("HF_TOKEN")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nox.block_compiler import (
    RMSNorm,
    scaled_dot_product_attention_2d,
    compile_transformer_block,
    compile_multiverse_block,
    evaluate_compiled_block,
)
from nox.utils import compute_metrics, get_modes_for_projection


def log_step(msg):
    """Prints timestamped progress log with immediate stdout flush."""
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def fetch_layer_weights(
    repo_id="NousResearch/Llama-2-7b-hf",
    layer_idx=15,
    save_dir="./models/Llama-2-7b-hf",
    token=None,
):
    """
    Fetches all 7 projection matrices and 2 layer norms for a single Transformer layer
    directly via HTTP Range requests without downloading the full model.
    Caches tensors locally in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    tensor_mapping = {
        "q_proj": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        "k_proj": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        "v_proj": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        "gate_proj": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        "up_proj": f"model.layers.{layer_idx}.mlp.up_proj.weight",
        "down_proj": f"model.layers.{layer_idx}.mlp.down_proj.weight",
        "input_layernorm": f"model.layers.{layer_idx}.input_layernorm.weight",
        "post_attention_layernorm": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
    }

    layer_weights = {}
    missing_tensors = {}

    for name, tensor_key in tensor_mapping.items():
        cache_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}.pt")
        if os.path.exists(cache_path):
            log_step(f"Loading cached {name} from '{cache_path}'...")
            layer_weights[name] = torch.load(cache_path, weights_only=True)
        else:
            missing_tensors[name] = tensor_key

    if not missing_tensors:
        log_step(f"All Layer {layer_idx} weights loaded from local cache.")
        return layer_weights

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    log_step(f"Resolving safetensors shard index for '{repo_id}' to fetch missing tensors...")
    index_url = f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
    req_index = urllib.request.Request(index_url, headers=headers)
    with urllib.request.urlopen(req_index) as resp:
        index_json = json.loads(resp.read().decode("utf-8"))
    weight_map = index_json.get("weight_map", {})

    shard_headers = {}
    chunk_size = 16 * 1024 * 1024

    for name, tensor_key in missing_tensors.items():
        shard_name = weight_map.get(tensor_key, "model-00001-of-00002.safetensors")
        file_url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_name}"

        if shard_name not in shard_headers:
            log_step(f"Fetching safetensors header from '{shard_name}'...")
            req_header_len = urllib.request.Request(file_url, headers={**headers, "Range": "bytes=0-7"})
            with urllib.request.urlopen(req_header_len) as resp:
                header_len = struct.unpack("<Q", resp.read())[0]

            req_header = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes=8-{8 + header_len - 1}"})
            with urllib.request.urlopen(req_header) as resp:
                shard_headers[shard_name] = (json.loads(resp.read().decode("utf-8")), header_len)

        header_json, header_len = shard_headers[shard_name]
        if tensor_key not in header_json:
            raise KeyError(f"Tensor key '{tensor_key}' not found in {shard_name} header")

        meta = header_json[tensor_key]
        shape = meta["shape"]
        offsets = meta["data_offsets"]
        start_byte = 8 + header_len + offsets[0]
        end_byte = 8 + header_len + offsets[1] - 1
        total_bytes = end_byte - start_byte + 1
        size_mb = total_bytes / (1024 * 1024)

        log_step(f"Downloading {tensor_key} ({size_mb:.2f} MB via HTTP Range)...")
        raw_bytes = bytearray()
        curr_offset = start_byte
        while curr_offset <= end_byte:
            chunk_end = min(curr_offset + chunk_size - 1, end_byte)
            req_data = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes={curr_offset}-{chunk_end}"})
            with urllib.request.urlopen(req_data) as resp:
                raw_bytes.extend(resp.read())
            curr_offset = chunk_end + 1

        tensor = torch.frombuffer(raw_bytes, dtype=torch.float16).reshape(shape).clone().to(dtype=torch.float32)
        cache_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}.pt")
        torch.save(tensor, cache_path)
        layer_weights[name] = tensor
        log_step(f"Cached {name} (shape: {list(tensor.shape)}) to '{cache_path}'.")

    return layer_weights


def build_calibrated_layer_input(
    d_model=4096,
    n_calib=256,
    n_test=128,
    latent_rank=48,
    outlier_pct=0.02,
    outlier_factor=12.0,
    noise_level=0.08,
    seed=42,
):
    """
    Constructs realistic LLM layer input tokens modeling Llama-2-7B's token manifold:
      1. Correlated semantic manifold (rank-48 subspace across sequence).
      2. Heavy-tailed persistent outlier channels (2% of channels scaled 12x).
      3. Background thermal noise (8%).
    """
    g = torch.Generator().manual_seed(seed)
    n_total = n_calib + n_test

    basis = torch.randn(d_model, latent_rank, generator=g, dtype=torch.float32) / (latent_rank ** 0.5)
    mode_scales = torch.exp(torch.linspace(0.0, -2.5, latent_rank)).unsqueeze(1)
    coeffs = torch.randn(latent_rank, n_total, generator=g, dtype=torch.float32) * mode_scales
    X_manifold = basis @ coeffs

    X_noise = noise_level * torch.randn(d_model, n_total, generator=g, dtype=torch.float32)
    X = X_manifold + X_noise

    n_outliers = max(1, int(d_model * outlier_pct))
    outlier_channels = torch.randperm(d_model, generator=g)[:n_outliers]
    X[outlier_channels, :] *= outlier_factor
    X = X / torch.std(X)

    X_calib = X[:, :n_calib]
    X_test = X[:, n_calib:]

    log_step(
        f"Calibrated layer activation profile: [{d_model}, {n_calib}] calibration tokens, "
        f"[{d_model}, {n_test}] held-out test tokens ({n_outliers} outlier channels, rank-{latent_rank} manifold)."
    )

    return X_calib, X_test


def generate_synthetic_layer_weights(d_model=4096, d_mlp=11008, seed=42):
    """Generates synthetic layer weights with realistic spectral decay and outlier coordinates."""
    torch.manual_seed(seed)
    log_step(f"Generating synthetic Layer weights (d_model={d_model}, d_mlp={d_mlp})...")

    def make_weight(d_out, d_in):
        W = torch.randn(d_out, d_in) * (2.0 / (d_in + d_out)) ** 0.5
        n_spikes = max(1, int(0.005 * W.numel()))
        idx = torch.randperm(W.numel())[:n_spikes]
        W.view(-1)[idx] += torch.randn(n_spikes) * 3.0
        return W

    return {
        "q_proj": make_weight(d_model, d_model),
        "k_proj": make_weight(d_model, d_model),
        "v_proj": make_weight(d_model, d_model),
        "o_proj": make_weight(d_model, d_model),
        "gate_proj": make_weight(d_mlp, d_model),
        "up_proj": make_weight(d_mlp, d_model),
        "down_proj": make_weight(d_model, d_mlp),
        "input_layernorm": torch.ones(d_model),
        "post_attention_layernorm": torch.ones(d_model),
    }


def run_day2_benchmark(
    layer_weights,
    X_calib,
    X_test,
    ranks=(1, 32, 80, 32, 1),
    r_bridge=96,
    n_outliers=65536,
    n_heads=32,
    n_kv_heads=32,
    seq_len=128,
):
    """
    Executes the Day 2 Full Single-Block Benchmark:
      1. Multiverse Default Block Baseline (pure Frobenius MPO)
      2. Multiverse Iso-Byte Block Baseline (matched byte budget)
      3. Nox Independent Block (Nox compression without causal healing)
      4. Nox Causally-Healed Block (Step 4.2 Causal Intra-Block Propagation)
    """
    total_dense_params = sum(
        w.numel() for k, w in layer_weights.items() if k not in ["input_layernorm", "post_attention_layernorm"]
    )
    dense_bytes = total_dense_params * 2.0
    dense_mb = dense_bytes / (1024 * 1024)
    log_step(f"Dense Layer 15 weights footprint: {dense_mb:.2f} MB ({total_dense_params:,} parameters).")

    results = {}

    def measure_infer_latency(blk):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = evaluate_compiled_block(blk, layer_weights, X_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        n_runs = 5
        for _ in range(n_runs):
            _ = evaluate_compiled_block(blk, layer_weights, X_test, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return ((time.perf_counter() - t_start) / float(n_runs)) * 1000.0

    log_step("Compiling Baseline 1: Multiverse Unhealed Block (CompactifAI TT-SVD Default)...")
    mv_ranks = ranks[1:-1] if len(ranks) == 5 else ranks
    mv_block = compile_multiverse_block(
        layer_weights=layer_weights,
        ranks=mv_ranks,
        core_bits=4,
        verbose=False,
    )
    mv_eval = evaluate_compiled_block(
        compiled_weights=mv_block,
        layer_weights=layer_weights,
        X_test=X_test,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
    )
    mv_infer = measure_infer_latency(mv_block)
    results["Multiverse (Default)"] = {
        "block": mv_eval["block_metrics"],
        "transform": mv_eval["transform_metrics"],
        "attn": mv_eval["attn_metrics"],
        "mlp": mv_eval["mlp_metrics"],
        "kb": mv_block["total_kb"],
        "time": mv_block["time_sec"],
        "infer_ms": mv_infer,
    }
    log_step(
        f"Multiverse Default Block compiled: Rel Err={mv_eval['block_metrics']['relative_error']:.4f}, "
        f"Drift={mv_eval['block_metrics']['angular_drift']:.5f}, "
        f"Norm={mv_eval['block_metrics']['norm_ratio']:.4f} in {mv_block['time_sec']:.2f}s (Infer: {mv_infer:.2f}ms)."
    )
    del mv_block, mv_eval
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    log_step("Compiling Baseline 2: Nox Independent Block (causal_healing=False)...")
    nox_indep_block = compile_transformer_block(
        layer_weights=layer_weights,
        X_calib=X_calib,
        ranks=ranks,
        r_bridge=r_bridge,
        n_outliers=n_outliers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        causal_healing=False,
        verbose=True,
    )
    nox_indep_eval = evaluate_compiled_block(
        compiled_weights=nox_indep_block,
        layer_weights=layer_weights,
        X_test=X_test,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
    )
    indep_infer = measure_infer_latency(nox_indep_block)
    results["Nox Independent Block"] = {
        "block": nox_indep_eval["block_metrics"],
        "transform": nox_indep_eval["transform_metrics"],
        "attn": nox_indep_eval["attn_metrics"],
        "mlp": nox_indep_eval["mlp_metrics"],
        "kb": nox_indep_block["total_kb"],
        "time": nox_indep_block["time_sec"],
        "infer_ms": indep_infer,
    }
    log_step(
        f"Nox Independent Block compiled: Rel Err={nox_indep_eval['block_metrics']['relative_error']:.4f}, "
        f"Drift={nox_indep_eval['block_metrics']['angular_drift']:.5f}, "
        f"Norm={nox_indep_eval['block_metrics']['norm_ratio']:.4f} in {nox_indep_block['time_sec']:.2f}s (Infer: {indep_infer:.2f}ms)."
    )
    del nox_indep_block, nox_indep_eval
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    log_step("Compiling Method 3: Nox Causally-Healed Block (Step 4.2 Causal Intra-Block Propagation)...")
    nox_healed_block = compile_transformer_block(
        layer_weights=layer_weights,
        X_calib=X_calib,
        ranks=ranks,
        r_bridge=r_bridge,
        n_outliers=n_outliers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
        causal_healing=True,
        verbose=True,
    )
    nox_healed_eval = evaluate_compiled_block(
        compiled_weights=nox_healed_block,
        layer_weights=layer_weights,
        X_test=X_test,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
    )
    healed_infer = measure_infer_latency(nox_healed_block)
    healed_kb = nox_healed_block["total_kb"]
    results["Nox Causally-Healed Block"] = {
        "block": nox_healed_eval["block_metrics"],
        "transform": nox_healed_eval["transform_metrics"],
        "attn": nox_healed_eval["attn_metrics"],
        "mlp": nox_healed_eval["mlp_metrics"],
        "kb": healed_kb,
        "time": nox_healed_block["time_sec"],
        "infer_ms": healed_infer,
    }
    log_step(
        f"Nox Causally-Healed Block compiled: Rel Err={nox_healed_eval['block_metrics']['relative_error']:.4f}, "
        f"Drift={nox_healed_eval['block_metrics']['angular_drift']:.5f}, "
        f"Norm={nox_healed_eval['block_metrics']['norm_ratio']:.4f} in {nox_healed_block['time_sec']:.2f}s (Infer: {healed_infer:.2f}ms)."
    )
    del nox_healed_block, nox_healed_eval
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    log_step(f"Compiling Method 4: Multiverse Iso-Byte Block (Matched Byte Budget >= {healed_kb:.1f} KB)...")
    mv_iso_block = compile_multiverse_block(
        layer_weights=layer_weights,
        target_kb=healed_kb,
        core_bits=4,
        verbose=False,
    )
    mv_iso_eval = evaluate_compiled_block(
        compiled_weights=mv_iso_block,
        layer_weights=layer_weights,
        X_test=X_test,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        seq_len=seq_len,
    )
    mv_iso_infer = measure_infer_latency(mv_iso_block)
    results["Multiverse (Iso-Byte)"] = {
        "block": mv_iso_eval["block_metrics"],
        "transform": mv_iso_eval["transform_metrics"],
        "attn": mv_iso_eval["attn_metrics"],
        "mlp": mv_iso_eval["mlp_metrics"],
        "kb": mv_iso_block["total_kb"],
        "time": mv_iso_block["time_sec"],
        "infer_ms": mv_iso_infer,
    }
    log_step(
        f"Multiverse Iso-Byte Block compiled (ranks={mv_iso_block.get('ranks', 'N/A')}): "
        f"Rel Err={mv_iso_eval['block_metrics']['relative_error']:.4f}, "
        f"Drift={mv_iso_eval['block_metrics']['angular_drift']:.5f}, "
        f"Norm={mv_iso_eval['block_metrics']['norm_ratio']:.4f} in {mv_iso_block['time_sec']:.2f}s (Infer: {mv_iso_infer:.2f}ms)."
    )
    del mv_iso_block, mv_iso_eval
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    print("\nNote: Compile Time is a one-time offline compression cost (0 GPU fine-tuning epochs). Infer Latency is the live forward-pass execution time.")
    print("=" * 135)
    print(" TABLE 1: FULL BLOCK OUTPUT (X_out = X_in + Y_attn + Y_mlp) (HELD-OUT TEST SET)")
    print("=" * 135)
    print(f"{'Method':<28} | {'Rel Error':<10} | {'Angular Drift':<14} | {'Norm Ratio':<10} | {'Cosine Sim':<10} | {'Size (KB)':<10} | {'Comp Ratio':<10} | {'Compile Time (1-Time)':<21} | {'Infer Latency (ms)':<18}")
    print("-" * 135)

    for name, r in results.items():
        rel_err = f"{r['block']['relative_error']:.4f}"
        drift = f"{r['block']['angular_drift']:.5f}"
        ratio = f"{r['block']['norm_ratio']:.4f}"
        cos_sim = f"{r['block']['cosine_similarity']:.4f}"
        kb = f"{r['kb']:.1f} KB"
        comp = f"{dense_bytes / (r['kb'] * 1024.0):.1f}x"
        ctime = f"{r['time']:.2f}s"
        infer = f"{r['infer_ms']:.2f} ms"
        print(f"{name:<28} | {rel_err:<10} | {drift:<14} | {ratio:<10} | {cos_sim:<10} | {kb:<10} | {comp:<10} | {ctime:<21} | {infer:<18}")

    print("-" * 135)

    print("\n" + "=" * 110)
    print(" TABLE 2: BLOCK TRANSFORMATION DELTA (Delta_block = Y_attn + Y_mlp) (ISOLATED FROM SKIP-CONNECTION)")
    print("=" * 110)
    print(f"{'Method':<28} | {'Delta RelErr':<12} | {'Delta Drift':<14} | {'Delta Norm':<10} | {'Delta CosSim':<12} | {'Gate Status':<12}")
    print("-" * 110)

    for name, r in results.items():
        t_err = f"{r['transform']['relative_error']:.4f}"
        t_drift = f"{r['transform']['angular_drift']:.5f}"
        t_norm = f"{r['transform']['norm_ratio']:.4f}"
        t_cos = f"{r['transform']['cosine_similarity']:.4f}"
        gate = "BASELINE" if "Multiverse" in name else ("PASS" if r['transform']['relative_error'] < 0.35 else "REVIEW")
        print(f"{name:<28} | {t_err:<12} | {t_drift:<14} | {t_norm:<10} | {t_cos:<12} | {gate:<12}")

    print("-" * 110)

    print("\n" + "=" * 110)
    print(" TABLE 3: SUB-BLOCK COMPONENT DRILLDOWN (HELD-OUT TEST SET)")
    print("=" * 110)
    print(f"{'Method':<28} | {'Attn RelErr':<12} | {'Attn Drift':<12} | {'MLP RelErr':<12} | {'MLP Drift':<12} | {'MLP Norm':<10}")
    print("-" * 110)

    for name, r in results.items():
        a_err = f"{r['attn']['relative_error']:.4f}"
        a_drift = f"{r['attn']['angular_drift']:.5f}"
        m_err = f"{r['mlp']['relative_error']:.4f}"
        m_drift = f"{r['mlp']['angular_drift']:.5f}"
        m_norm = f"{r['mlp']['norm_ratio']:.4f}"
        print(f"{name:<28} | {a_err:<12} | {a_drift:<12} | {m_err:<12} | {m_drift:<12} | {m_norm:<10}")

    print("-" * 110)

    mv_trans_err = results["Multiverse (Default)"]["transform"]["relative_error"]
    mv_iso_trans_err = results["Multiverse (Iso-Byte)"]["transform"]["relative_error"]
    indep_trans_err = results["Nox Independent Block"]["transform"]["relative_error"]
    healed_trans_err = results["Nox Causally-Healed Block"]["transform"]["relative_error"]

    err_red_trans_vs_mv = (1.0 - (healed_trans_err / (mv_trans_err + 1e-12))) * 100.0
    err_red_trans_vs_mv_iso = (1.0 - (healed_trans_err / (mv_iso_trans_err + 1e-12))) * 100.0
    err_red_trans_vs_indep = (1.0 - (healed_trans_err / (indep_trans_err + 1e-12))) * 100.0

    mv_err = results["Multiverse (Default)"]["block"]["relative_error"]
    mv_iso_err = results["Multiverse (Iso-Byte)"]["block"]["relative_error"]
    indep_err = results["Nox Independent Block"]["block"]["relative_error"]
    healed_err = results["Nox Causally-Healed Block"]["block"]["relative_error"]

    err_red_vs_mv = (1.0 - (healed_err / (mv_err + 1e-12))) * 100.0
    err_red_vs_mv_iso = (1.0 - (healed_err / (mv_iso_err + 1e-12))) * 100.0
    err_red_vs_indep = (1.0 - (healed_err / (indep_err + 1e-12))) * 100.0

    print("\n[Causal Intra-Block Propagation Verification (Step 4.2)]")
    print("  Transformation Delta (Delta_block = Y_attn + Y_mlp):")
    print(f"    Multiverse Default Relative Error:      {mv_trans_err:.4f}")
    print(f"    Multiverse Iso-Byte Relative Error:     {mv_iso_trans_err:.4f} (Matched Footprint >= {healed_kb:.1f} KB)")
    print(f"    Nox Causally-Healed Relative Error:     {healed_trans_err:.4f}")
    print(f"    Transformation Error Reduction vs MV:   {err_red_trans_vs_mv:.1f}%")
    print(f"    Transformation Gain vs MV Iso-Byte:     {err_red_trans_vs_mv_iso:.1f}%")
    print(f"    Causal Healing Gain vs Independent:     {err_red_trans_vs_indep:.1f}%")
    print("  Full Layer Output (X_out = X_in + Delta_block):")
    print(f"    Multiverse Default Relative Error:      {mv_err:.4f}")
    print(f"    Multiverse Iso-Byte Relative Error:     {mv_iso_err:.4f}")
    print(f"    Nox Causally-Healed Relative Error:     {healed_err:.4f}")
    print(f"    Full-Layer Error Reduction vs MV:       {err_red_vs_mv:.1f}%")
    print(f"    Full-Layer Gain vs MV Iso-Byte:         {err_red_vs_mv_iso:.1f}%")
    print(f"    Full-Layer Gain vs Independent:         {err_red_vs_indep:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Day 2 Benchmark: Full Single-Block Causal Healing Test")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="HuggingFace model ID")
    parser.add_argument("--save_dir", type=str, default="./models/Llama-2-7b-hf", help="Local directory for cached weights")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--layer_idx", type=int, default=15, help="Layer index to benchmark (default: 15)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic weights instead of downloading real model")
    parser.add_argument("--n_calib", type=int, default=256, help="Number of calibration tokens (default: 256)")
    parser.add_argument("--n_test", type=int, default=128, help="Number of test tokens (default: 128)")
    parser.add_argument("--r_bridge", type=int, default=96, help="Bridge rank for Nox decomposition (default: 96)")
    parser.add_argument("--n_outliers", type=int, default=65536, help="Number of outliers for outlier shield (default: 65536)")
    parser.add_argument("--device", type=str, default=None, help="Device to execute benchmark on (default: auto-detect cuda, else cpu)")
    args = parser.parse_args()

    hf_token = args.token or HF_TOKEN
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_step(f"NVIDIA GPU Acceleration enabled: {torch.cuda.get_device_name(0)} (TF32 Tensor Cores active).")
    else:
        log_step(f"Executing on {device.upper()}.")

    if args.synthetic:
        log_step("Running in synthetic benchmark mode...")
        weights = generate_synthetic_layer_weights(d_model=4096, d_mlp=11008)
    else:
        log_step(f"Fetching real Layer {args.layer_idx} weights from '{args.model_id}'...")
        weights = fetch_layer_weights(
            repo_id=args.model_id,
            layer_idx=args.layer_idx,
            save_dir=args.save_dir,
            token=hf_token,
        )

    weights = {k: v.to(device) for k, v in weights.items()}

    X_calib, X_test = build_calibrated_layer_input(
        d_model=weights["q_proj"].shape[1],
        n_calib=args.n_calib,
        n_test=args.n_test,
    )
    X_calib = X_calib.to(device)
    X_test = X_test.to(device)

    with torch.inference_mode():
        run_day2_benchmark(
            layer_weights=weights,
            X_calib=X_calib,
            X_test=X_test,
            ranks=(1, 32, 80, 32, 1),
            r_bridge=args.r_bridge,
            n_outliers=args.n_outliers,
        )


if __name__ == "__main__":
    main()
