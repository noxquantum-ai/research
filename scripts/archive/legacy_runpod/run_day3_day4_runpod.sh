#!/usr/bin/env bash
# RunPod RTX 4090 Master Execution Script for Day 3 and Day 4
# Full 32-Layer Sequential Model Compression & Entanglement Cliff Sweep

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "=========================================================================="
echo " NoxQuantum Day 3-4 Execution: Full 32-Layer Sequential Cascade on RTX 4090"
echo "=========================================================================="

# 1. Hardware & CUDA verification
python3 -c "
import torch
if not torch.cuda.is_available():
    print('[ERROR] CUDA is not available on this machine!')
    exit(1)
dev_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f'[Nox System] Active GPU: {dev_name} ({vram_gb:.2f} GB VRAM)')
"

# 2. Dependency check
echo "[Nox Setup] Installing / verifying dependencies..."
pip install -q -r requirements.txt

# 3. Model directory setup
mkdir -p ./models/Llama-2-7b-hf
mkdir -p ./checkpoints/day3_day4_cascade

MODE="${1:-sweep_cliff}"

case "$MODE" in
  test_4_layers)
    echo "[Execution] Running 4-layer validation cascade (Layers 15 -> 18) with checkpointing..."
    python3 benchmarks/benchmark_day3_cascade.py \
      --start_layer 15 \
      --end_layer 18 \
      --tier production_18x \
      --save_checkpoint_dir ./checkpoints/day3_day4_cascade/layers_15_18_prod
    ;;

  full_32_layers)
    echo "[Execution] Running FULL 32-LAYER sequential cascade (Layers 0 -> 31) in Production 18x Tier..."
    python3 benchmarks/benchmark_day3_cascade.py \
      --full_model \
      --tier production_18x \
      --skip_independent \
      --save_checkpoint_dir ./checkpoints/day3_day4_cascade/full_32_layers_prod
    ;;

  full_32_layers_fast)
    echo "[Execution] Running FULL 32-LAYER pure compression (<6 min on RTX 4090, skipping all baselines)..."
    python3 benchmarks/benchmark_day3_cascade.py \
      --full_model \
      --tier production_18x \
      --skip_baselines \
      --save_checkpoint_dir ./checkpoints/day3_day4_cascade/full_32_layers_prod
    ;;

  sweep_cliff)
    echo "[Execution] Running full 32-layer Entanglement Cliff Sweep across all 3 tiers..."
    python3 benchmarks/benchmark_day3_cascade.py \
      --full_model \
      --sweep_cliff \
      --skip_independent \
      --save_checkpoint_dir ./checkpoints/day3_day4_cascade/entanglement_cliff_sweep
    ;;

  eval_perplexity)
    echo "[Execution] Evaluating WikiText-2 Perplexity & Autoregressive Generation on checkpoints..."
    CKPT_DIR="${2:-./checkpoints/day3_day4_cascade/full_32_layers_prod}"
    python3 benchmarks/benchmark_day4_perplexity.py \
      --checkpoint_dir "$CKPT_DIR" \
      --eval_dense \
      --run_generation
    ;;

  full_pipeline)
    echo "[Execution] Running end-to-end: Full 32-layer compression followed by Perplexity evaluation..."
    python3 benchmarks/benchmark_day3_cascade.py \
      --full_model \
      --tier production_18x \
      --skip_baselines \
      --save_checkpoint_dir ./checkpoints/day3_day4_cascade/full_32_layers_prod

    echo "[Execution] Starting Day 4-5 Perplexity & Generation Benchmark on 32-layer checkpoint..."
    python3 benchmarks/benchmark_day4_perplexity.py \
      --checkpoint_dir ./checkpoints/day3_day4_cascade/full_32_layers_prod \
      --eval_dense \
      --run_generation
    ;;

  the_big_guns)
    echo "[Execution] LAUNCHING THE BIG GUNS: Full 32-Layer 3-Tier Master Sweep + Multi-Tier Perplexity & Generation"
    echo "[Execution] Zero shortcuts, all baselines active (Multiverse Default, Multiverse Iso-Byte, Independent, Healed)..."

    # Step 1: Complete 32-layer sweep across all 3 tiers with every baseline computed
    python3 benchmarks/benchmark_day3_cascade.py \
      --full_model \
      --sweep_cliff \
      --save_checkpoint_dir ./checkpoints/day3_day4_cascade/master_cliff_sweep

    # Step 2: Perplexity & Generation on Tier 2 (Production 18x - Primary CompactifAI Killer)
    echo ""
    echo "[The Big Guns] Evaluating WikiText-2 Perplexity on Tier 2 (Production 18x)..."
    python3 benchmarks/benchmark_day4_perplexity.py \
      --checkpoint_dir ./checkpoints/day3_day4_cascade/master_cliff_sweep/production_18x \
      --eval_dense \
      --run_generation

    # Step 3: Perplexity & Generation on Tier 3 (High-Fidelity 10x - Lossless Tier)
    echo ""
    echo "[The Big Guns] Evaluating WikiText-2 Perplexity on Tier 3 (High-Fidelity 10x)..."
    python3 benchmarks/benchmark_day4_perplexity.py \
      --checkpoint_dir ./checkpoints/day3_day4_cascade/master_cliff_sweep/high_fidelity_10x \
      --run_generation

    # Step 4: Perplexity & Generation on Tier 1 (Extreme 77x - Stress Test Tier)
    echo ""
    echo "[The Big Guns] Evaluating WikiText-2 Perplexity on Tier 1 (Extreme 77x)..."
    python3 benchmarks/benchmark_day4_perplexity.py \
      --checkpoint_dir ./checkpoints/day3_day4_cascade/master_cliff_sweep/extreme_77x \
      --run_generation

    # Step 5: Automatically package checkpoints for export
    echo ""
    echo "[The Big Guns] Packaging checkpoints for export..."
    python3 scripts/export_checkpoints.py \
      --checkpoint_dir ./checkpoints/day3_day4_cascade/master_cliff_sweep \
      --method all
    ;;

  export)
    CKPT_DIR="${2:-./checkpoints/day3_day4_cascade/master_cliff_sweep}"
    HF_REPO="${3:-}"
    echo "[Export] Packaging and exporting checkpoints from '$CKPT_DIR'..."
    if [ -n "$HF_REPO" ]; then
      python3 scripts/export_checkpoints.py --checkpoint_dir "$CKPT_DIR" --method all --hf_repo "$HF_REPO"
    else
      python3 scripts/export_checkpoints.py --checkpoint_dir "$CKPT_DIR" --method all
    fi
    ;;

  *)
    echo "Usage: bash run_day3_day4_runpod.sh [test_4_layers | full_32_layers | full_32_layers_fast | sweep_cliff | eval_perplexity | full_pipeline | the_big_guns | export]"
    exit 1
    ;;
esac

echo "[Nox Complete] Day 3-4 cascade execution finished successfully."
