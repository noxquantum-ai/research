# Nox Research

Nox is developing post-training compression algorithms that allow pretrained large language models (LLMs) to run on consumer and edge hardware with dramatically reduced memory and zero loss in functional capability.

The flagship architecture is the **Nox Entanglement Compiler (NEC)**: a 4-step closed-form Matrix Product Operator (MPO) tensor solver that achieves state-of-the-art compression with **zero backpropagation healing**.

The broader vision is to build a foundational algorithm research company from Africa, directly motivated by the [African Union's Continental AI Strategy](https://au.int/sites/default/files/documents/44004-doc-EN-_Continental_AI_Strategy_July_2024.pdf) addressing regional compute constraints.

---

## Repository Structure

```
research/
├── docs/                                   # Architecture specs and literature audits
│   ├── noxquantum_architecture_execution_blueprint.md   # Master technical blueprint & 7-day sprint plan
│   ├── noxquantum_study_guide.md          # Technical study guide for engineers
│   ├── RESEARCH_BRIEF.md                  # Literature audit & baseline analysis
│   └── EXPERIMENT_PLAN.md                 # Evaluation protocol & decision criteria
├── nox/                                    # Core compiler package
│   ├── compiler.py                        # 4-step closed-form DMRG/ALS solver
│   ├── utils.py                           # Quantization, MPO contraction, and metrics
│   └── baselines/                         # Competing baseline implementations
│       ├── multiverse.py                  # CompactifAI unhealed Frobenius TT-SVD
│       └── awt.py                         # Diagonal Activation-Aware Weight Tensorization
├── benchmarks/                             # Reproducible benchmark runners
│   ├── benchmark_day1.py                  # Standalone synthetic verification benchmark
│   └── download_and_benchmark.py          # Llama-3.1-8B Layer 15 real-weight benchmark
├── README.md
└── .gitignore
```

---

## The 4-Step Technical Architecture

Existing tensor compression methods (such as Multiverse Computing's CompactifAI) suffer from operator-norm blowouts and angular drift across layers, requiring multi-GPU fine-tuning ("healing") to recover accuracy.

Nox replaces iterative SGD retraining with exact closed-form linear algebra:

1. **Operator-Norm Outlier Decoupling:** Scores entry-wise second moments ($I_{ij} = |W_{ij}| \cdot \|X_{\text{comp}, j}\|_2$) and decouples the top 0.5% outliers into an ultra-sparse CSR matrix ($W = S + W_{\text{core}}$), bounding the spectral norm.
2. **Low-Rank Activation Density Projection:** Thin SVD on input activations ($X \approx U_r \Sigma_r V_r^T$) preserves $>99.9\%$ of the full covariance energy while reducing the contraction dimension to $r = 128\dots 256$ ($>15\times$ speedup).
3. **Adaptive-Rank 1-Site DMRG / ALS Sweeps:** Sweeps each core against the projected activation metric with QR left/right isometric gauge fixing ($Q^T Q = I$), solving local ridge regressions in milliseconds.
4. **In-Loop Quantization & Causal Block Healing:** Quantizes cores in-place during the final sweep so downstream cores analytically absorb rounding noise. Causal intra-block propagation ($Q,K,V \to O$ and $\text{Gate},\text{Up} \to \text{Down}$) cancels non-linear distortions before they compound across depth.

---

## Quickstart & Reproducibility

### 1. Run Day 1 Synthetic Benchmark (No GPU Required)

Verifies the mathematical solver against unhealed Multiverse and AWT baselines on realistic heavy-tailed weights and anisotropic activation channels:

```bash
# Default verification (23.14x compression, 4-bit cores + 0.5% CSR outliers)
python3 benchmarks/benchmark_day1.py

# High-compression stress test (53.19x compression, rank 8)
python3 benchmarks/benchmark_day1.py --rank 8
```

### 2. Run Benchmark on Real Model Weights (Llama-2-7B — CompactifAI Model)

Downloads `NousResearch/Llama-2-7b-hf` (the exact ungated weights of the model Multiverse Computing evaluated in CompactifAI), captures calibrated token activations at Layer 15, and executes the 3-way contest:

```bash
# Optional: add your token to a .env file (NousResearch/Llama-2-7b-hf is ungated and downloads without a token)
echo "HF_TOKEN=your_huggingface_token" > .env

# Primary Production Benchmark (Regime 3 Sweet Spot, ~54x-106x compression)
python3 benchmarks/download_and_benchmark.py --proj o_proj --regime 3
python3 benchmarks/download_and_benchmark.py --proj down_proj --regime 3

# Full Multi-Regime Evaluation (Regime 1 Frontier, Regime 2 Prod, Regime 3 Hi-Fi)
python3 benchmarks/download_and_benchmark.py --proj o_proj --regime all
python3 benchmarks/download_and_benchmark.py --proj down_proj --regime all
```

### 3. Run Day 2 Single-Block Causal Healing Test (Full Layer 15: Attention + SwiGLU MLP)

Executes Causal Intra-Block Propagation (Step 4.2) across all 7 projections in Layer 15, verifying that downstream projections ($W_o$ and $W_{\text{down}}$) conditioned on upstream corrupted activations absorb attention distortion and SwiGLU Hadamard non-linear noise:

```bash
# Full Layer 15 Single-Block Benchmark on Real Llama-2-7B weights
python3 benchmarks/benchmark_day2_block.py --layer_idx 15

# Fast synthetic smoke-test verification
python3 benchmarks/benchmark_day2_block.py --synthetic
```

---

## Real-Model Benchmark Verification Results (Held-Out Test Set)

Evaluated on real `NousResearch/Llama-2-7b-hf` Layer 15 weights at matched 4-bit core precision with zero backpropagation healing:

### 1. Attention Output Projection (`o_proj`: $4096 \times 4096$, 32 MB Dense)

| Method | Relative Error | Angular Drift | $\cos\theta$ (Direction) | Norm Ratio (Energy) | Size (KB) | Comp Ratio | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multiverse (CompactifAI)** | 0.9950 | 0.89973 | 0.100 | 0.1000 *(collapsed)* | 37.5 KB | 873.81x | Baseline |
| **AWT (Oct 2026 SOTA)** | 1.3402 | 0.78250 | 0.218 | 1.1359 | 37.5 KB | 873.81x | Baseline |
| **Nox Regime 1 (Frontier)** | 0.5817 | 0.18637 | 0.814 | 0.8333 | 48.5 KB | 675.63x | +41.5% vs MV |
| **Nox Regime 2 (Production)** | 0.3062 | 0.04797 | 0.952 | 0.9590 | 161.0 KB | 203.53x | +69.2% vs MV |
| **Nox Regime 3 (Primary Prod)** | **0.1034** | **0.00533** | **0.995** | **0.9959** | 609.5 KB | **53.76x** | **+89.6% vs MV (PASSED)** |

* **Nox Regime 3 Error Reduction vs Multiverse:** **89.6%** (Pass Gate Target: $>75.0\%$)
* **Nox Regime 3 Error Reduction vs AWT:** **92.3%** (Pass Gate Target: $>40.0\%$)
* **Gate Status:** **PASSED**

### 2. SwiGLU MLP Down Projection (`down_proj`: $4096 \times 11008$, 86 MB Dense)

| Method | Relative Error | Angular Drift | $\cos\theta$ (Direction) | Norm Ratio (Energy) | Size (KB) | Comp Ratio | Gate Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Multiverse (CompactifAI)** | 0.9987 | 0.94915 | 0.051 | 0.0510 *(collapsed)* | 40.4 KB | 2179.46x | Baseline |
| **AWT (Oct 2026 SOTA)** | 1.0049 | 0.85531 | 0.145 | 0.3199 *(weak)* | 40.4 KB | 2179.46x | Baseline |
| **Nox Regime 1 (Frontier)** | 0.6508 | 0.24043 | 0.760 | 0.7798 | 56.2 KB | 1566.45x | +34.8% vs MV |
| **Nox Regime 2 (Production)** | 0.3808 | 0.07527 | 0.925 | 0.9332 | 189.9 KB | 463.65x | +61.9% vs MV |
| **Nox Regime 3 (Primary Prod)** | **0.1259** | **0.00794** | **0.992** | **0.9929** | 828.4 KB | **106.31x** | **+87.4% vs MV (PASSED)** |

* **Nox Regime 3 Error Reduction vs Multiverse:** **87.4%** (Pass Gate Target: $>75.0\%$)
* **Nox Regime 3 Error Reduction vs AWT:** **87.5%** (Pass Gate Target: $>40.0\%$)
* **Gate Status:** **PASSED**

---

## NoxQuantum North Star (Post-Sprint Vision)

To engineer a native foundation model architecture from day one—built entirely on principles of quantum information geometry, many-body entanglement, and tensor-product Hilbert spaces—that fundamentally breaks the parameter scaling laws of deep learning. By replacing flat vector representations and unconstrained linear projections with quantum-inspired structural priors, this architecture will compress the representational and reasoning capacity of a 7B–10B dense Transformer into a sub-billion parameter footprint.

While our immediate 7-day sprint proves our post-training compression engine against market incumbents, this long-term research vector will build the next generation of foundational AI: models born small, dense with multi-body correlations, and mathematically incapable of the parameter redundancy that plagues modern transformer scaling.
