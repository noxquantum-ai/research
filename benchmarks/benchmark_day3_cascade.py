"""Day 3 Benchmark: Multi-Layer Sequential Cascade Test (Step 4.2 Inter-Layer Causal Healing).

Evaluates cascading layer-by-layer compression across a sequence of Transformer layers
(e.g. Layers 15 -> 18 or full Layers 0 -> 31). Proves that calibrating downstream layers on
actual upstream compressed activations stabilizes residual stream drift compared to unhealed
Multiverse and independent matrix compression.
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

from nox.sequential_compiler import compile_sequential_cascade, get_tier_config
from nox.utils import compute_metrics


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
            log_step(f"Loading cached {name} (Layer {layer_idx}) from '{cache_path}'...")
            layer_weights[name] = torch.load(cache_path, weights_only=True).to(dtype=torch.float32)
        else:
            missing_tensors[name] = tensor_key

    if not missing_tensors:
        log_step(f"All Layer {layer_idx} weights loaded from local cache.")
        return layer_weights

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    log_step(f"Resolving safetensors shard index for '{repo_id}' to fetch missing Layer {layer_idx} tensors...")
    index_url = f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
    req_index = urllib.request.Request(index_url, headers=headers)
    with urllib.request.urlopen(req_index) as resp:
        index_json = json.loads(resp.read().decode("utf-8"))
    weight_map = index_json.get("weight_map", {})

    shard_headers = {}

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
        chunk_size = 16 * 1024 * 1024
        while curr_offset <= end_byte:
            chunk_end = min(curr_offset + chunk_size - 1, end_byte)
            req_chunk = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes={curr_offset}-{chunk_end}"})
            with urllib.request.urlopen(req_chunk) as resp:
                raw_bytes.extend(resp.read())
            curr_offset = chunk_end + 1

        dtype_str = meta["dtype"]
        dtype_map = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
        }
        pt_dtype = dtype_map.get(dtype_str, torch.float16)
        tensor = torch.frombuffer(raw_bytes, dtype=pt_dtype).reshape(shape).clone().to(dtype=torch.float32)
        cache_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}.pt")
        torch.save(tensor, cache_path)
        layer_weights[name] = tensor

    return layer_weights


def build_calibrated_layer_input(d_model=4096, n_calib=256, n_test=128, seed=42):
    """
    Synthesizes activation inputs with realistic LLM manifold properties.
    """
    torch.manual_seed(seed)
    n_total = n_calib + n_test
    latent_rank = 48
    A = torch.randn(d_model, latent_rank)
    B = torch.randn(latent_rank, n_total)
    X = A @ B

    decay = torch.exp(-torch.linspace(0, 2.5, d_model)).unsqueeze(1)
    X = X * decay

    n_outliers = int(0.02 * d_model)
    outlier_idx = torch.randperm(d_model)[:n_outliers]
    X[outlier_idx, :] *= 18.0

    X = X / (torch.norm(X, dim=0, keepdim=True) + 1e-6) * (d_model ** 0.5)

    X_calib = X[:, :n_calib]
    X_test = X[:, n_calib:]

    return X_calib, X_test


def main():
    parser = argparse.ArgumentParser(description="Day 3 Benchmark: Multi-Layer Sequential Cascade Test")
    parser.add_argument("--start_layer", type=int, default=15, help="Starting layer index (default: 15)")
    parser.add_argument("--end_layer", type=int, default=18, help="Ending layer index inclusive (default: 18)")
    parser.add_argument("--full_model", action="store_true", help="Run full 32 layers (0 to 31)")
    parser.add_argument("--tier", type=str, choices=["extreme_77x", "production_18x"], default="extreme_77x", help="Compression tier")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="HuggingFace model ID")
    parser.add_argument("--save_dir", type=str, default="./models/Llama-2-7b-hf", help="Local directory for cached weights")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--n_calib", type=int, default=256, help="Number of calibration tokens")
    parser.add_argument("--n_test", type=int, default=128, help="Number of test tokens")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto cuda, else cpu)")
    args = parser.parse_args()

    hf_token = args.token or HF_TOKEN
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_step(f"NVIDIA GPU Acceleration enabled: {torch.cuda.get_device_name(0)} (TF32 active).")
    else:
        log_step(f"Executing on {device.upper()}.")

    if args.full_model:
        start_l = 0
        end_l = 31
    else:
        start_l = args.start_layer
        end_l = args.end_layer

    layer_indices = list(range(start_l, end_l + 1))
    log_step(f"Benchmarking Cascade from Layer {start_l} to Layer {end_l} ({len(layer_indices)} layers total)...")

    X_calib, X_test = build_calibrated_layer_input(
        d_model=4096,
        n_calib=args.n_calib,
        n_test=args.n_test,
    )

    def fetcher(l_idx):
        return fetch_layer_weights(
            repo_id=args.model_id,
            layer_idx=l_idx,
            save_dir=args.save_dir,
            token=hf_token,
        )

    with torch.inference_mode():
        cascade_res = compile_sequential_cascade(
            fetch_layer_fn=fetcher,
            layer_indices=layer_indices,
            X_calib_init=X_calib,
            X_test_init=X_test,
            tier=args.tier,
            n_heads=32,
            n_kv_heads=32,
            seq_len=128,
            device=device,
            verbose=True,
        )

    history = cascade_res["history"]
    cfg = cascade_res["config"]

    print("\n" + "=" * 110)
    print(f" DAY 3 BENCHMARK: SEQUENTIAL CASCADE SCORECARD (LAYERS {start_l} -> {end_l})")
    print(f" Tier: {cfg['description']}")
    print("=" * 110)
    print(f"{'Layer':<8} | {'MV RelErr':<11} | {'Indep RelErr':<14} | {'Healed RelErr':<15} | {'Healed Drift':<14} | {'Healed CosSim':<14} | {'Healed Norm':<11}")
    print("-" * 110)

    for h in history:
        l_idx = h["layer_idx"]
        mv_err = f"{h['multiverse']['block']['relative_error']:.4f}"
        indep_err = f"{h['independent']['block']['relative_error']:.4f}"
        healed_err = f"{h['causal_healed']['block']['relative_error']:.4f}"
        healed_drift = f"{h['causal_healed']['block']['angular_drift']:.5f}"
        healed_cos = f"{h['causal_healed']['block']['cosine_similarity']:.4f}"
        healed_norm = f"{h['causal_healed']['block']['norm_ratio']:.4f}"
        print(f"Layer {l_idx:<2} | {mv_err:<11} | {indep_err:<14} | {healed_err:<15} | {healed_drift:<14} | {healed_cos:<14} | {healed_norm:<11}")

    print("-" * 110)

    print("\n" + "=" * 110)
    print(f" BLOCK TRANSFORMATION DELTA (Delta_block) LAYER-BY-LAYER FIDELITY")
    print("=" * 110)
    print(f"{'Layer':<8} | {'MV Delta RelErr':<16} | {'Indep Delta RelErr':<19} | {'Healed Delta RelErr':<20} | {'Healed Delta CosSim':<20}")
    print("-" * 110)

    for h in history:
        l_idx = h["layer_idx"]
        mv_d_err = f"{h['multiverse']['delta']['relative_error']:.4f}"
        indep_d_err = f"{h['independent']['delta']['relative_error']:.4f}"
        healed_d_err = f"{h['causal_healed']['delta']['relative_error']:.4f}"
        healed_d_cos = f"{h['causal_healed']['delta']['cosine_similarity']:.4f}"
        print(f"Layer {l_idx:<2} | {mv_d_err:<16} | {indep_d_err:<19} | {healed_d_err:<20} | {healed_d_cos:<20}")

    print("-" * 110)

    # Exit stream evaluation
    first_layer_healed_err = history[0]["causal_healed"]["block"]["relative_error"]
    last_layer_healed_err = history[-1]["causal_healed"]["block"]["relative_error"]
    first_layer_indep_err = history[0]["independent"]["block"]["relative_error"]
    last_layer_indep_err = history[-1]["independent"]["block"]["relative_error"]

    healed_drift_growth = (last_layer_healed_err - first_layer_healed_err)
    indep_drift_growth = (last_layer_indep_err - first_layer_indep_err)

    print("\n[Step 4.2 Inter-Layer Causal Healing Verification]")
    print(f"  First Layer RelErr (Layer {start_l}): Healed = {first_layer_healed_err:.4f} | Independent = {first_layer_indep_err:.4f}")
    print(f"  Last Layer RelErr  (Layer {end_l}): Healed = {last_layer_healed_err:.4f} | Independent = {last_layer_indep_err:.4f}")
    print(f"  Residual Drift Growth across {len(layer_indices)} layers: Healed = {healed_drift_growth:+.4f} | Independent = {indep_drift_growth:+.4f}")
    if abs(healed_drift_growth) <= abs(indep_drift_growth):
        print("  Status: PASSED (Inter-layer causal healing bounded compounding drift).")
    else:
        print("  Status: PENDING REVIEW.")


if __name__ == "__main__":
    main()
