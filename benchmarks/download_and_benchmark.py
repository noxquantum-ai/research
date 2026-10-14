import argparse
import json
import os
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

from nox.compiler import nox_4step_compiler, nox_extreme_800x_decompose
from nox.baselines.multiverse import multiverse_tt_svd
from nox.baselines.awt import awt_tt_svd
from nox.utils import contract_mpo_4site, evaluate_nox_prediction, compute_metrics, count_parameters


def log_step(msg):
    """Prints timestamped progress log with immediate stdout flush."""
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def fetch_real_layer_tensor(repo_id="NousResearch/Llama-2-7b-hf", layer_idx=15, proj_type="o_proj", save_dir="./models/Llama-2-7b-hf", token=None):
    """
    Option A: Fetches only the target layer projection matrix directly from HuggingFace
    via HTTP Range request without downloading the full 13.5GB model.
    Caches the tensor locally in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, f"layer_{layer_idx}_{proj_type}.pt")
    if os.path.exists(cache_path):
        log_step(f"Loading cached real Layer {layer_idx} {proj_type} weights from '{cache_path}'...")
        return torch.load(cache_path, weights_only=True)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if proj_type == "o_proj":
        tensor_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    elif proj_type == "down_proj":
        tensor_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    else:
        raise ValueError(f"Unsupported projection type: {proj_type}")

    log_step(f"Option A: Resolving safetensors shard index for '{repo_id}'...")
    index_url = f"https://huggingface.co/{repo_id}/raw/main/model.safetensors.index.json"
    req_index = urllib.request.Request(index_url, headers=headers)
    try:
        with urllib.request.urlopen(req_index) as resp:
            index_json = json.loads(resp.read().decode("utf-8"))
            shard_name = index_json.get("weight_map", {}).get(tensor_key, "model-00001-of-00002.safetensors")
    except Exception:
        shard_name = "model-00001-of-00002.safetensors"

    file_url = f"https://huggingface.co/{repo_id}/resolve/main/{shard_name}"
    log_step(f"Fetching safetensors header from '{shard_name}'...")
    req_header_len = urllib.request.Request(file_url, headers={**headers, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req_header_len) as resp:
        header_len = struct.unpack("<Q", resp.read())[0]

    req_header = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes=8-{8 + header_len - 1}"})
    with urllib.request.urlopen(req_header) as resp:
        header_json = json.loads(resp.read().decode("utf-8"))

    if tensor_key not in header_json:
        raise KeyError(f"Tensor key '{tensor_key}' not found in {shard_name} header")

    meta = header_json[tensor_key]
    shape = meta["shape"]
    offsets = meta["data_offsets"]
    start_byte = 8 + header_len + offsets[0]
    end_byte = 8 + header_len + offsets[1] - 1
    size_mb = (end_byte - start_byte + 1) / (1024 * 1024)

    log_step(f"Downloading real {tensor_key} ({size_mb:.2f} MB via HTTP Range request)...")
    req_data = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes={start_byte}-{end_byte}"})
    with urllib.request.urlopen(req_data) as resp:
        raw_bytes = bytearray(resp.read())

    tensor = torch.frombuffer(raw_bytes, dtype=torch.float16).reshape(shape).clone().to(dtype=torch.float32)
    torch.save(tensor, cache_path)
    log_step(f"Successfully downloaded and cached Layer {layer_idx} {proj_type} (shape: {list(tensor.shape)}) to '{cache_path}'.")
    return tensor


def build_calibrated_benchmark_data(weight_tensor, n_calib=128, n_test=128, outlier_pct=0.02, outlier_factor=8.0, seed=42):
    """
    Constructs calibration and held-out test activations modeling Llama-2-7B's
    channel activation distribution (~2% outlier channels scaled 8x).
    """
    d_out, d_in = weight_tensor.shape
    g = torch.Generator().manual_seed(seed)

    X = torch.randn(d_in, n_calib + n_test, generator=g, dtype=torch.float32)
    n_outliers = max(1, int(d_in * outlier_pct))
    outlier_channels = torch.randperm(d_in, generator=g)[:n_outliers]
    X[outlier_channels, :] *= outlier_factor
    X = X / torch.std(X)

    X_calib = X[:, :n_calib]
    X_test = X[:, n_calib:]

    Y_calib = weight_tensor @ X_calib
    Y_test = weight_tensor @ X_test

    log_step(f"Calibrated activation profile: [{d_in}, {n_calib}] calibration tokens, [{d_in}, {n_test}] held-out test tokens ({n_outliers} outlier channels).")

    return {
        "W": weight_tensor,
        "X_calib": X_calib,
        "Y_calib": Y_calib,
        "X_test": X_test,
        "Y_test": Y_test,
        "d_out": d_out,
        "d_in": d_in,
    }


# Option B (full 13.5GB model download and activation extraction) commented out for laptop-level tests; using Option A for fast direct layer extraction.
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import snapshot_download
#
# SAMPLE_CALIBRATION_PROMPTS = [
#     "The physical principles governing quantum information and tensor networks have revolutionized modern condensed matter physics. In particular, the density matrix renormalization group algorithm enables efficient simulation of one-dimensional quantum many-body systems by representing quantum states as matrix product states.",
#     "In theoretical computer science, matrix multiplication algorithms strive to approach the exponent of two. Tensor decomposition techniques such as Canonical Polyadic, Tucker, and Tensor-Train decompositions allow high-dimensional arrays to be compressed into low-rank factor cores.",
#     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
#     "Theorem: Let H be a Hilbert space and T a bounded linear operator. The spectral theorem states that any normal operator can be represented as an integral with respect to a projection-valued measure.",
#     "The continental strategy for artificial intelligence in Africa outlines key investments in digital infrastructure, high-performance computing clusters, and algorithmic research tailored to bandwidth-constrained environments.",
# ]
#
# def download_model(model_id, save_dir, token):
#     """
#     Downloads model weights from HuggingFace Hub and saves to save_dir.
#     Skips download if save_dir already contains model files.
#     """
#     log_step(f"Checking local model directory '{save_dir}'...")
#     if os.path.exists(save_dir):
#         safetensor_files = [f for f in os.listdir(save_dir) if f.endswith(".safetensors")]
#         if safetensor_files:
#             total_bytes = sum(os.path.getsize(os.path.join(save_dir, f)) for f in safetensor_files)
#             total_gb = total_bytes / (1024 ** 3)
#             log_step(f"Found {len(safetensor_files)} existing weight file(s) ({total_gb:.2f} GB) in '{save_dir}'. Skipping download.")
#             return save_dir
#     log_step(f"Starting HuggingFace snapshot download of '{model_id}'...")
#     log_step(f"Destination: '{save_dir}' (this will show live file transfer progress)...")
#     os.makedirs(save_dir, exist_ok=True)
#     t0 = time.time()
#     snapshot_download(
#         repo_id=model_id,
#         local_dir=save_dir,
#         token=token,
#         allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
#         ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5", "*.ot"],
#     )
#     log_step(f"Model download completed in {time.time() - t0:.1f} seconds.")
#     return save_dir
#
# def extract_layer_and_activations(model_dir, layer_idx=15, proj_type="o_proj", token=None, device="cpu"):
#     """
#     Loads tokenizer and model, runs calibration prompts to capture real activations
#     at the specified layer and projection.
#     """
#     t0 = time.time()
#     log_step(f"[1/4] Loading tokenizer from '{model_dir}'...")
#     tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
#     log_step(f"[1/4] Tokenizer loaded in {time.time() - t0:.2f}s.")
#
#     log_step(f"[2/4] Loading model weights into memory ({device})...")
#     torch_dtype = torch.float16 if device != "cpu" else torch.float32
#     t_model = time.time()
#     model = AutoModelForCausalLM.from_pretrained(
#         model_dir,
#         token=token,
#         torch_dtype=torch_dtype,
#         low_cpu_mem_usage=True,
#     )
#     model.to(device)
#     model.eval()
#     log_step(f"[2/4] Model loaded in {time.time() - t_model:.2f}s (dtype={torch_dtype}).")
#
#     if torch.cuda.is_available() and device != "cpu":
#         vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
#         log_step(f"[2/4] GPU VRAM allocated: {vram_gb:.2f} GB")
#
#     target_layer = model.model.layers[layer_idx]
#     if proj_type == "o_proj":
#         target_module = target_layer.self_attn.o_proj
#     elif proj_type == "down_proj":
#         target_module = target_layer.mlp.down_proj
#     else:
#         raise ValueError(f"Unsupported projection type: {proj_type}")
#
#     weight_tensor = target_module.weight.detach().to(dtype=torch.float32, device="cpu")
#     d_out, d_in = weight_tensor.shape
#     log_step(f"[3/4] Hooked Layer {layer_idx} {proj_type} (Weight shape: [{d_out}, {d_in}]).")
#
#     captured_inputs = []
#
#     def hook_fn(module, input_tensor, output_tensor):
#         act = input_tensor[0].detach()
#         captured_inputs.append(act)
#
#     hook_handle = target_module.register_forward_hook(hook_fn)
#
#     log_step(f"[4/4] Passing {len(SAMPLE_CALIBRATION_PROMPTS)} calibration prompts to capture real activations...")
#     with torch.no_grad():
#         for idx, prompt in enumerate(SAMPLE_CALIBRATION_PROMPTS):
#             inputs = tokenizer(prompt, return_tensors="pt").to(device)
#             token_count = inputs.input_ids.shape[1]
#             log_step(f"  -> Processing prompt {idx + 1}/{len(SAMPLE_CALIBRATION_PROMPTS)} ({token_count} tokens)...")
#             model(**inputs)
#
#     hook_handle.remove()
#
#     all_acts = []
#     for act in captured_inputs:
#         flat_act = act.reshape(-1, d_in).T
#         all_acts.append(flat_act)
#
#     full_X = torch.cat(all_acts, dim=1).to(dtype=torch.float32, device="cpu")
#     n_total = full_X.shape[1]
#     n_split = n_total // 2
#     X_calib = full_X[:, :n_split]
#     X_test = full_X[:, n_split:]
#
#     log_step(f"[4/4] Captured {n_total} activation tokens. Split: {n_split} calibration, {n_total - n_split} held-out test.")
#
#     Y_calib = weight_tensor @ X_calib
#     Y_test = weight_tensor @ X_test
#
#     log_step("Unloading full model from memory to free system resources...")
#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#     return {
#         "W": weight_tensor,
#         "X_calib": X_calib,
#         "Y_calib": Y_calib,
#         "X_test": X_test,
#         "Y_test": Y_test,
#         "d_out": d_out,
#         "d_in": d_in,
#     }


def get_modes_for_projection(d_out, d_in):
    """
    Determines 4-mode factorizations for Llama-2-7B (CompactifAI) and Llama-3-8B projections.
    """
    if d_out == 4096 and d_in == 4096:
        return (8, 8, 8, 8), (8, 8, 8, 8)
    elif d_out == 4096 and d_in == 11008:
        return (8, 8, 8, 8), (8, 11, 8, 16)
    elif d_out == 4096 and d_in == 14336:
        return (8, 8, 8, 8), (8, 14, 8, 16)
    else:
        m_out = int(d_out ** 0.25)
        m_in = int(d_in ** 0.25)
        return (m_out, m_out, m_out, m_out), (m_in, m_in, m_in, m_in)


def run_benchmark(data, o_modes, i_modes, regime="both", core_bits=4, n_sweeps=3):
    """
    Executes the 3-way contest on real weights:
      - Multiverse TT-SVD (unhealed baseline)
      - AWT TT-SVD (diagonal activation-aware baseline)
      - Nox Regime 1: Ultra-Extreme Frontier (809x compression, [1, 8, 48, 8, 1], r_bridge=4)
      - Nox Regime 2: Production Sweet Spot (127x compression, [1, 32, 96, 32, 1], r_bridge=16)
    """
    W = data["W"]
    X_calib = data["X_calib"]
    Y_calib = data["Y_calib"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    dense_bytes = W.numel() * 2.0
    results = {}

    log_step("Starting 3-way benchmark evaluation on real production weights...")

    # Method 1: Multiverse TT-SVD (CompactifAI unhealed baseline)
    log_step("Running Multiverse TT-SVD (unhealed baseline)...")
    t0 = time.time()
    mv_cores, mv_ranks = multiverse_tt_svd(
        W=W,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=[24, 24, 24],
        core_bits=core_bits,
    )
    mv_time = time.time() - t0
    log_step(f"Multiverse TT-SVD completed in {mv_time:.2f}s (ranks: {mv_ranks}). Evaluating...")
    mv_Y_pred = contract_mpo_4site(mv_cores, X_test)
    mv_metrics = compute_metrics(Y_test, mv_Y_pred)
    mv_footprint = count_parameters(mv_cores, bits=core_bits)
    results["Multiverse (CompactifAI)"] = {
        "metrics": mv_metrics,
        "footprint": mv_footprint,
        "ranks": mv_ranks,
        "time_sec": mv_time,
    }

    # Method 2: AWT TT-SVD (Diagonal activation-aware baseline)
    log_step("Running AWT TT-SVD (diagonal activation-aware baseline)...")
    t0 = time.time()
    awt_cores, awt_ranks = awt_tt_svd(
        W=W,
        X_comp=X_calib,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=[24, 24, 24],
        alpha=0.5,
        core_bits=core_bits,
    )
    awt_time = time.time() - t0
    log_step(f"AWT TT-SVD completed in {awt_time:.2f}s (ranks: {awt_ranks}). Evaluating...")
    awt_Y_pred = contract_mpo_4site(awt_cores, X_test)
    awt_metrics = compute_metrics(Y_test, awt_Y_pred)
    awt_footprint = count_parameters(awt_cores, bits=core_bits)
    results["AWT (Oct 2026 SOTA)"] = {
        "metrics": awt_metrics,
        "footprint": awt_footprint,
        "ranks": awt_ranks,
        "time_sec": awt_time,
    }

    # Method 3: Nox Regime 1: Ultra-Extreme 809x Frontier
    if regime in ("both", "extreme"):
        log_step("Running Nox Regime 1 (Ultra-Extreme 809x Frontier: [1, 8, 48, 8, 1], r_bridge=4)...")
        t0 = time.time()
        nox1_cores, nox1_bridge, nox1_perm, nox1_ranks = nox_extreme_800x_decompose(
            W=W,
            X_comp=X_calib,
            Y_dense=Y_calib,
            o_modes=o_modes,
            i_modes=i_modes,
            ranks=(1, 8, 48, 8, 1),
            r_bridge=4,
            n_sweeps=n_sweeps,
            core_bits=core_bits,
            verbose=True,
        )
        nox1_time = time.time() - t0
        log_step(f"Nox Regime 1 completed in {nox1_time:.2f}s (ranks: {nox1_ranks}). Evaluating...")
        nox1_Y_pred = evaluate_nox_prediction(
            cores=nox1_cores,
            bridge=nox1_bridge,
            perm_info=nox1_perm,
            X=X_test,
        )
        nox1_metrics = compute_metrics(Y_test, nox1_Y_pred)
        nox1_footprint = count_parameters(cores=nox1_cores, bridge=nox1_bridge, bits=core_bits)
        results["Nox Regime 1 (809x Frontier)"] = {
            "metrics": nox1_metrics,
            "footprint": nox1_footprint,
            "ranks": nox1_ranks,
            "time_sec": nox1_time,
        }

    # Method 4: Nox Regime 2: Production Sweet Spot
    if regime in ("both", "production"):
        log_step("Running Nox Regime 2 (Production Sweet Spot: [1, 32, 96, 32, 1], r_bridge=16)...")
        t0 = time.time()
        nox2_cores, nox2_bridge, nox2_perm, nox2_ranks = nox_extreme_800x_decompose(
            W=W,
            X_comp=X_calib,
            Y_dense=Y_calib,
            o_modes=o_modes,
            i_modes=i_modes,
            ranks=(1, 32, 96, 32, 1),
            r_bridge=16,
            n_sweeps=n_sweeps,
            core_bits=core_bits,
            verbose=True,
        )
        nox2_time = time.time() - t0
        log_step(f"Nox Regime 2 completed in {nox2_time:.2f}s (ranks: {nox2_ranks}). Evaluating...")
        nox2_Y_pred = evaluate_nox_prediction(
            cores=nox2_cores,
            bridge=nox2_bridge,
            perm_info=nox2_perm,
            X=X_test,
        )
        nox2_metrics = compute_metrics(Y_test, nox2_Y_pred)
        nox2_footprint = count_parameters(cores=nox2_cores, bridge=nox2_bridge, bits=core_bits)
        results["Nox Regime 2 (Production)"] = {
            "metrics": nox2_metrics,
            "footprint": nox2_footprint,
            "ranks": nox2_ranks,
            "time_sec": nox2_time,
        }

    print("\n" + "=" * 98)
    print(f" LLAMA-2-7B (COMPACTIFAI EXACT MODEL) BENCHMARK RESULTS (Held-Out Test Set, Core: {core_bits}-bit)")
    print("=" * 98)
    print(f"{'Method':<28} | {'Rel Error':<10} | {'Angular Drift':<14} | {'Norm Ratio':<10} | {'Size (KB)':<10} | {'Comp Ratio':<10}")
    print("-" * 98)

    for name, r in results.items():
        rel_err = f"{r['metrics']['relative_error']:.4f}"
        drift = f"{r['metrics']['angular_drift']:.5f}"
        ratio = f"{r['metrics']['norm_ratio']:.4f}"
        kb = f"{r['footprint']['total_kb']:.1f} KB"
        comp = f"{dense_bytes / r['footprint']['total_bytes']:.2f}x"
        print(f"{name:<28} | {rel_err:<10} | {drift:<14} | {ratio:<10} | {kb:<10} | {comp:<10}")

    print("-" * 98)

    mv_err = results["Multiverse (CompactifAI)"]["metrics"]["relative_error"]
    awt_err = results["AWT (Oct 2026 SOTA)"]["metrics"]["relative_error"]

    for name in ["Nox Regime 1 (809x Frontier)", "Nox Regime 2 (Production)"]:
        if name in results:
            nox_err = results[name]["metrics"]["relative_error"]
            err_red_vs_mv = (1.0 - (nox_err / (mv_err + 1e-12))) * 100.0
            err_red_vs_awt = (1.0 - (nox_err / (awt_err + 1e-12))) * 100.0
            print(f"\n[{name}]")
            print(f"  Error Reduction vs Multiverse: {err_red_vs_mv:.1f}% (Gate Target: >75.0%)")
            print(f"  Error Reduction vs AWT:        {err_red_vs_awt:.1f}% (Gate Target: >40.0%)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download Llama-2-7B Layer and run Zero-Backprop Benchmark")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="HuggingFace model ID")
    parser.add_argument("--save_dir", type=str, default="./models/Llama-2-7b-hf", help="Local directory for cached weights")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--layer_idx", type=int, default=15, help="Layer index to benchmark (default: 15)")
    parser.add_argument("--proj", type=str, default="o_proj", choices=["o_proj", "down_proj"], help="Projection matrix type")
    parser.add_argument("--regime", type=str, default="both", choices=["both", "extreme", "production"], help="Benchmark regime to run")
    parser.add_argument("--core_bits", type=int, default=4, help="Core quantization bit width (default: 4)")
    parser.add_argument("--sweeps", type=int, default=3, help="Number of DMRG sweeps for Nox (default: 3)")
    parser.add_argument("--n_calib", type=int, default=128, help="Number of calibration tokens (default: 128)")
    parser.add_argument("--n_test", type=int, default=128, help="Number of test tokens (default: 128)")
    args = parser.parse_args()

    hf_token = args.token or HF_TOKEN

    log_step(f"Running Option A: Direct extraction of Layer {args.layer_idx} {args.proj} weights from {args.model_id}...")
    W = fetch_real_layer_tensor(
        repo_id=args.model_id,
        layer_idx=args.layer_idx,
        proj_type=args.proj,
        save_dir=args.save_dir,
        token=hf_token,
    )
    data = build_calibrated_benchmark_data(
        weight_tensor=W,
        n_calib=args.n_calib,
        n_test=args.n_test,
    )

    # Option B (full 13.5GB model download and activation extraction) commented out for laptop-level tests; using Option A for fast direct layer extraction.
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_path = download_model(args.model_id, args.save_dir, hf_token)
    # data = extract_layer_and_activations(
    #     model_dir=model_path,
    #     layer_idx=args.layer_idx,
    #     proj_type=args.proj,
    #     token=hf_token,
    #     device=device,
    # )

    o_modes, i_modes = get_modes_for_projection(data["d_out"], data["d_in"])

    run_benchmark(
        data=data,
        o_modes=o_modes,
        i_modes=i_modes,
        regime=args.regime,
        core_bits=args.core_bits,
        n_sweeps=args.sweeps,
    )


if __name__ == "__main__":
    main()
