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

### 2. Run Day 1 Benchmark on Real Llama-3.1-8B Weights

Downloads `meta-llama/Llama-3.1-8B` (requires HuggingFace token), captures real activations at Layer 15, and executes the 3-way contest:

```bash
export HF_TOKEN="your_huggingface_token"

# Benchmark Layer 15 Attention Output Projection (o_proj: 4096 x 4096)
python3 benchmarks/download_and_benchmark.py --save_dir ./models/Llama-3.1-8B --proj o_proj --rank 24

# Benchmark Layer 15 SwiGLU MLP Down Projection (down_proj: 4096 x 14336)
python3 benchmarks/download_and_benchmark.py --save_dir ./models/Llama-3.1-8B --proj down_proj --rank 24
```

---

## Day 1 Benchmark Verification Results

Held-out test set comparison at matched 4-bit core precision:

| Method | Relative Error | Angular Drift | Norm Ratio | Size (KB) | Comp Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multiverse (CompactifAI)** | 0.9231 | 0.36350 | 1.1437 | 4.2 KB | 30.12x |
| **AWT (Oct 2026 SOTA)** | 3.1637 | 0.10054 | 4.0328 | 4.2 KB | 30.12x |
| **Nox Engine (Ours)** | **0.0116** | **0.00006** | **1.0002** | 5.5 KB | 23.14x |

* **Nox Error Reduction vs Multiverse:** **98.7%** (Pass Gate Target: $>75.0\%$)
* **Nox Error Reduction vs AWT:** **99.6%** (Pass Gate Target: $>40.0\%$)
* **Gate Status:** **PASSED**
