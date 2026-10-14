"""Download Llama-3.1-8B and immediately run the Day 1 benchmark on real layer weights."""

import argparse
import os
import sys
import time
from pathlib import Path
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nox.compiler import nox_4step_compiler
from nox.baselines.multiverse import multiverse_tt_svd
from nox.baselines.awt import awt_tt_svd
from nox.utils import contract_mpo_4site, compute_metrics, count_parameters


SAMPLE_CALIBRATION_PROMPTS = [
    "The physical principles governing quantum information and tensor networks have revolutionized modern condensed matter physics. In particular, the density matrix renormalization group algorithm enables efficient simulation of one-dimensional quantum many-body systems by representing quantum states as matrix product states.",
    "In theoretical computer science, matrix multiplication algorithms strive to approach the exponent of two. Tensor decomposition techniques such as Canonical Polyadic, Tucker, and Tensor-Train decompositions allow high-dimensional arrays to be compressed into low-rank factor cores.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
    "Theorem: Let H be a Hilbert space and T a bounded linear operator. The spectral theorem states that any normal operator can be represented as an integral with respect to a projection-valued measure.",
    "The continental strategy for artificial intelligence in Africa outlines key investments in digital infrastructure, high-performance computing clusters, and algorithmic research tailored to bandwidth-constrained environments.",
]


def download_model(model_id, save_dir, token):
    """
    Downloads model weights from HuggingFace Hub and saves to save_dir.
    Skips download if save_dir already contains model files.
    """
    if os.path.exists(save_dir) and any(f.endswith(".safetensors") for f in os.listdir(save_dir)):
        print(f"Model directory '{save_dir}' already exists with weights. Skipping download.")
        return save_dir

    print(f"Starting download of '{model_id}' to '{save_dir}'...")
    os.makedirs(save_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print(f"Model successfully downloaded to '{save_dir}'.")
    return save_dir


def extract_layer_and_activations(model_dir, layer_idx=15, proj_type="o_proj", token=None, device="cpu"):
    """
    Loads tokenizer and model, runs calibration prompts to capture real activations
    at the specified layer and projection.
    """
    print(f"Loading tokenizer from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)

    print(f"Loading model on {device} (float16)...")
    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        token=token,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    target_layer = model.model.layers[layer_idx]
    if proj_type == "o_proj":
        target_module = target_layer.self_attn.o_proj
    elif proj_type == "down_proj":
        target_module = target_layer.mlp.down_proj
    else:
        raise ValueError(f"Unsupported projection type: {proj_type}")

    weight_tensor = target_module.weight.detach().to(dtype=torch.float32, device="cpu")
    d_out, d_in = weight_tensor.shape
    print(f"Extracted Layer {layer_idx} {proj_type} weight matrix: shape [{d_out}, {d_in}]")

    captured_inputs = []

    def hook_fn(module, input_tensor, output_tensor):
        act = input_tensor[0].detach()
        captured_inputs.append(act)

    hook_handle = target_module.register_forward_hook(hook_fn)

    print("Running forward pass with calibration prompts to capture real activations...")
    with torch.no_grad():
        for prompt in SAMPLE_CALIBRATION_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)

    hook_handle.remove()

    all_acts = []
    for act in captured_inputs:
        flat_act = act.reshape(-1, d_in).T
        all_acts.append(flat_act)

    full_X = torch.cat(all_acts, dim=1).to(dtype=torch.float32, device="cpu")
    n_total = full_X.shape[1]
    print(f"Captured {n_total} activation tokens.")

    n_split = n_total // 2
    X_calib = full_X[:, :n_split]
    X_test = full_X[:, n_split:]

    Y_calib = weight_tensor @ X_calib
    Y_test = weight_tensor @ X_test

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "W": weight_tensor,
        "X_calib": X_calib,
        "Y_calib": Y_calib,
        "X_test": X_test,
        "Y_test": Y_test,
        "d_out": d_out,
        "d_in": d_in,
    }


def get_modes_for_projection(d_out, d_in):
    """
    Determines 4-mode factorizations for Llama-3.1-8B projections.
    """
    if d_out == 4096 and d_in == 4096:
        return (8, 8, 8, 8), (8, 8, 8, 8)
    elif d_out == 4096 and d_in == 14336:
        return (8, 8, 8, 8), (8, 14, 8, 16)
    else:
        m_out = int(d_out ** 0.25)
        m_in = int(d_in ** 0.25)
        return (m_out, m_out, m_out, m_out), (m_in, m_in, m_in, m_in)


def run_benchmark(data, o_modes, i_modes, target_ranks, core_bits=4, n_sweeps=3, outlier_pct=0.005):
    """
    Executes the 3-way contest on real weights and real held-out activations.
    """
    W = data["W"]
    X_calib = data["X_calib"]
    Y_calib = data["Y_calib"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    dense_bytes = W.numel() * 2.0
    results = {}

    print("\nRunning Multiverse TT-SVD (unhealed)...")
    t0 = time.time()
    mv_cores, mv_ranks = multiverse_tt_svd(
        W=W,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        core_bits=core_bits,
    )
    mv_time = time.time() - t0
    mv_Y_pred = contract_mpo_4site(mv_cores, X_test)
    mv_metrics = compute_metrics(Y_test, mv_Y_pred)
    mv_footprint = count_parameters(mv_cores, S_sparse=None, bits=core_bits)
    results["Multiverse (CompactifAI)"] = {
        "metrics": mv_metrics,
        "footprint": mv_footprint,
        "ranks": mv_ranks,
        "time_sec": mv_time,
    }

    print("Running AWT TT-SVD (diagonal activation-aware)...")
    t0 = time.time()
    awt_cores, awt_ranks = awt_tt_svd(
        W=W,
        X_comp=X_calib,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        alpha=0.5,
        core_bits=core_bits,
    )
    awt_time = time.time() - t0
    awt_Y_pred = contract_mpo_4site(awt_cores, X_test)
    awt_metrics = compute_metrics(Y_test, awt_Y_pred)
    awt_footprint = count_parameters(awt_cores, S_sparse=None, bits=core_bits)
    results["AWT (Oct 2026 SOTA)"] = {
        "metrics": awt_metrics,
        "footprint": awt_footprint,
        "ranks": awt_ranks,
        "time_sec": awt_time,
    }

    print("Running Nox Entanglement Compiler (4-step closed form)...")
    t0 = time.time()
    nox_cores, nox_S, nox_ranks = nox_4step_compiler(
        W=W,
        X_comp=X_calib,
        Y_dense=Y_calib,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        outlier_pct=outlier_pct,
        r_subspace=128,
        n_sweeps=n_sweeps,
        core_bits=core_bits,
    )
    nox_time = time.time() - t0
    nox_Y_pred = contract_mpo_4site(nox_cores, X_test) + (nox_S @ X_test)
    nox_metrics = compute_metrics(Y_test, nox_Y_pred)
    nox_footprint = count_parameters(nox_cores, S_sparse=nox_S, bits=core_bits)
    results["Nox Engine (Ours)"] = {
        "metrics": nox_metrics,
        "footprint": nox_footprint,
        "ranks": nox_ranks,
        "time_sec": nox_time,
    }

    print("\n" + "=" * 94)
    print(f" LLAMA-3.1-8B LAYER 15 BENCHMARK RESULTS (Held-Out Test Set, Core Precision: {core_bits}-bit)")
    print("=" * 94)
    print(f"{'Method':<25} | {'Rel Error':<10} | {'Angular Drift':<14} | {'Norm Ratio':<10} | {'Size (KB)':<10} | {'Comp Ratio':<10}")
    print("-" * 94)

    for name, r in results.items():
        rel_err = f"{r['metrics']['relative_error']:.4f}"
        drift = f"{r['metrics']['angular_drift']:.5f}"
        ratio = f"{r['metrics']['norm_ratio']:.4f}"
        kb = f"{r['footprint']['total_kb']:.1f} KB"
        comp = f"{dense_bytes / r['footprint']['total_bytes']:.2f}x"
        print(f"{name:<25} | {rel_err:<10} | {drift:<14} | {ratio:<10} | {kb:<10} | {comp:<10}")

    print("-" * 94)

    mv_err = results["Multiverse (CompactifAI)"]["metrics"]["relative_error"]
    awt_err = results["AWT (Oct 2026 SOTA)"]["metrics"]["relative_error"]
    nox_err = results["Nox Engine (Ours)"]["metrics"]["relative_error"]

    err_red_vs_mv = (1.0 - (nox_err / (mv_err + 1e-12))) * 100.0
    err_red_vs_awt = (1.0 - (nox_err / (awt_err + 1e-12))) * 100.0

    print(f"\nNox Error Reduction vs Multiverse: {err_red_vs_mv:.1f}% (Gate Target: >75.0%)")
    print(f"Nox Error Reduction vs AWT:        {err_red_vs_awt:.1f}% (Gate Target: >40.0%)")

    pass_mv = err_red_vs_mv >= 75.0
    pass_awt = err_red_vs_awt >= 40.0

    if pass_mv and pass_awt:
        print("\n>>> DAY 1 GATE RESULT: [PASS] Real weights gate cleared with zero backprop! <<<")
    else:
        print("\n>>> DAY 1 GATE RESULT: [REVIEW] Check bond dimensions or outlier percentage. <<<")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download Llama-3.1-8B and run Day 1 Benchmark")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B", help="HuggingFace model ID")
    parser.add_argument("--save_dir", type=str, default="./models/Llama-3.1-8B", help="Local directory to save/load model")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token (defaults to HF_TOKEN env var)")
    parser.add_argument("--layer_idx", type=int, default=15, help="Layer index to benchmark (default: 15)")
    parser.add_argument("--proj", type=str, default="o_proj", choices=["o_proj", "down_proj"], help="Projection matrix type")
    parser.add_argument("--rank", type=int, default=24, help="Target bond dimension (default: 24)")
    parser.add_argument("--core_bits", type=int, default=4, help="Core quantization bit width (default: 4)")
    parser.add_argument("--sweeps", type=int, default=3, help="Number of DMRG sweeps for Nox (default: 3)")
    parser.add_argument("--outlier_pct", type=float, default=0.005, help="Sparse outlier fraction (default: 0.005, i.e. 0.5 percent)")
    args = parser.parse_args()

    hf_token = args.token or os.environ.get("HF_TOKEN")
    if not hf_token and not (os.path.exists(args.save_dir) and any(f.endswith(".safetensors") for f in os.listdir(args.save_dir))):
        print("Error: HF_TOKEN is not set in environment or passed via --token.")
        print("Please export HF_TOKEN='your_token_here' or pass --token.")
        sys.exit(1)

    # Step 1: Download model
    model_path = download_model(args.model_id, args.save_dir, hf_token)

    # Step 2: Extract target layer and real activations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = extract_layer_and_activations(
        model_dir=model_path,
        layer_idx=args.layer_idx,
        proj_type=args.proj,
        token=hf_token,
        device=device,
    )

    # Step 3: Run Day 1 benchmark immediately
    o_modes, i_modes = get_modes_for_projection(data["d_out"], data["d_in"])
    target_ranks = [args.rank, args.rank, args.rank]

    run_benchmark(
        data=data,
        o_modes=o_modes,
        i_modes=i_modes,
        target_ranks=target_ranks,
        core_bits=args.core_bits,
        n_sweeps=args.sweeps,
        outlier_pct=args.outlier_pct,
    )


if __name__ == "__main__":
    main()
