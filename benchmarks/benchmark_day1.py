"""Day 1 benchmark: 3-way contest between Multiverse, AWT, and Nox on synthetic and real weights."""

import argparse
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nox.compiler import nox_4step_compiler
from nox.baselines.multiverse import multiverse_tt_svd
from nox.baselines.awt import awt_tt_svd
from nox.utils import contract_mpo_4site, compute_metrics, count_parameters


def generate_synthetic_benchmark_data(
    d_out=256,
    d_in=256,
    n_calib=256,
    n_test=256,
    seed=42,
    device="cpu",
):
    """
    Generates synthetic weight and activation data mirroring LLM properties:
      1. Anisotropic activations: a few channels have high-magnitude energy (simulating attention sinks).
      2. Heavy-tailed weights: contains sparse outlier coordinates (simulating LLM outlier channels).
      3. Separate calibration set and held-out evaluation set.
    """
    torch.manual_seed(seed)

    # Generate anisotropic channel scales for activations
    channel_energy = torch.ones(d_in, device=device)
    n_high_energy = max(1, int(0.02 * d_in))
    channel_energy[:n_high_energy] = 25.0

    # Calibration activations
    X_calib_raw = torch.randn(d_in, n_calib, device=device)
    X_calib = X_calib_raw * channel_energy.unsqueeze(1)

    # Held-out test activations
    X_test_raw = torch.randn(d_in, n_test, device=device)
    X_test = X_test_raw * channel_energy.unsqueeze(1)

    # Base weight matrix with decaying spectrum
    U, _ = torch.linalg.qr(torch.randn(d_out, d_out, device=device))
    V, _ = torch.linalg.qr(torch.randn(d_in, d_in, device=device))
    spectrum = torch.exp(-torch.linspace(0, 3.5, min(d_out, d_in), device=device))
    S_diag = torch.diag(spectrum)
    W = (U[:, :min(d_out, d_in)] @ S_diag @ V[:, :min(d_out, d_in)].T) * 0.1

    # Inject sparse outlier coordinates (0.5% of entries)
    n_outliers = max(1, int(0.005 * W.numel()))
    outlier_indices = torch.randperm(W.numel())[:n_outliers]
    W_flat = W.flatten()
    W_flat[outlier_indices] += torch.randn(n_outliers, device=device) * 1.5
    W = W_flat.reshape(d_out, d_in)

    Y_calib = W @ X_calib
    Y_test = W @ X_test

    return {
        "W": W,
        "X_calib": X_calib,
        "Y_calib": Y_calib,
        "X_test": X_test,
        "Y_test": Y_test,
        "d_out": d_out,
        "d_in": d_in,
    }


def run_benchmark(
    data,
    o_modes,
    i_modes,
    target_ranks=None,
    eps=0.02,
    core_bits=4,
    n_sweeps=3,
    outlier_pct=0.005,
    r_subspace=128,
):
    """
    Executes the 3-way benchmark at matched byte / rank budget on held-out test data.
    """
    W = data["W"]
    X_calib = data["X_calib"]
    Y_calib = data["Y_calib"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    dense_bytes = W.numel() * 2.0
    results = {}

    # Method 1: Multiverse TT-SVD (Data-blind Frobenius SVD + post-hoc quantization)
    t0 = time.time()
    mv_cores, mv_ranks = multiverse_tt_svd(
        W=W,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        eps=eps,
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

    # Method 2: AWT (Diagonal activation-scaled TT-SVD + post-hoc quantization)
    t0 = time.time()
    awt_cores, awt_ranks = awt_tt_svd(
        W=W,
        X_comp=X_calib,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        alpha=0.5,
        eps=eps,
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

    # Method 3: Nox Entanglement Compiler (Outliers + Subspace Projection + 1-Site DMRG + In-Loop Quant)
    t0 = time.time()
    nox_cores, nox_S, nox_ranks = nox_4step_compiler(
        W=W,
        X_comp=X_calib,
        Y_dense=Y_calib,
        o_modes=o_modes,
        i_modes=i_modes,
        ranks=target_ranks,
        eps=eps,
        outlier_pct=outlier_pct,
        r_subspace=r_subspace,
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

    # Print Formatted Results Table
    print("\n" + "=" * 92)
    print(f" DAY 1 BENCHMARK RESULTS (Held-Out Test Set, Core Precision: {core_bits}-bit)")
    print("=" * 92)
    print(f"{'Method':<25} | {'Rel Error':<10} | {'Angular Drift':<14} | {'Norm Ratio':<10} | {'Size (KB)':<10} | {'Comp Ratio':<10}")
    print("-" * 92)

    for name, r in results.items():
        rel_err = f"{r['metrics']['relative_error']:.4f}"
        drift = f"{r['metrics']['angular_drift']:.5f}"
        ratio = f"{r['metrics']['norm_ratio']:.4f}"
        kb = f"{r['footprint']['total_kb']:.1f} KB"
        comp = f"{dense_bytes / r['footprint']['total_bytes']:.2f}x"
        print(f"{name:<25} | {rel_err:<10} | {drift:<14} | {ratio:<10} | {kb:<10} | {comp:<10}")

    print("-" * 92)

    # Verification of Day 1 Pass Gate
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
        print("\n>>> DAY 1 GATE RESULT: [PASS] All fidelity thresholds cleared! <<<")
    else:
        print("\n>>> DAY 1 GATE RESULT: [REVIEW] Inspect bond dimensions or outlier ratio. <<<")

    return results


def main():
    parser = argparse.ArgumentParser(description="Day 1 Single-Layer Benchmark: Multiverse vs AWT vs Nox")
    parser.add_argument("--d_out", type=int, default=256, help="Output dimension (default: 256)")
    parser.add_argument("--d_in", type=int, default=256, help="Input dimension (default: 256)")
    parser.add_argument("--rank", type=int, default=16, help="Target bond dimension (default: 16)")
    parser.add_argument("--core_bits", type=int, default=4, help="Core quantization bits (default: 4)")
    parser.add_argument("--sweeps", type=int, default=3, help="Number of DMRG sweeps for Nox (default: 3)")
    parser.add_argument("--outlier_pct", type=float, default=0.005, help="Outlier decoupling ratio (default: 0.005)")
    args = parser.parse_args()

    # Determine 4-mode factorization
    mode_size = int(args.d_out ** 0.25)
    if mode_size ** 4 != args.d_out:
        # Fallback to balanced factors if not a clean 4th power
        o_modes = (4, 4, 4, 4)
        i_modes = (4, 4, 4, 4)
    else:
        o_modes = (mode_size, mode_size, mode_size, mode_size)
        i_modes = (mode_size, mode_size, mode_size, mode_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Day 1 verification on device: {device} ({args.d_out}x{args.d_in})")

    data = generate_synthetic_benchmark_data(
        d_out=args.d_out,
        d_in=args.d_in,
        n_calib=256,
        n_test=256,
        device=device,
    )

    ranks = [args.rank, args.rank, args.rank]
    run_benchmark(
        data=data,
        o_modes=o_modes,
        i_modes=i_modes,
        target_ranks=ranks,
        core_bits=args.core_bits,
        n_sweeps=args.sweeps,
        outlier_pct=args.outlier_pct,
    )


if __name__ == "__main__":
    main()
