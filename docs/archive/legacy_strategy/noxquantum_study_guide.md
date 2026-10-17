# NoxQuantum Founder & Engineer Study Guide

To fully own the mathematics, execution, and architectural defensibility of the Nox Entanglement Compiler over the next week, focus your study on these specific concepts, papers, and mathematical tools. 

---

## 1. Linear Algebra & Tensor Network Foundations

You do not need a Ph.D. in quantum physics, but you *must* be fluent in these four linear algebra concepts:

* **Singular Value Decomposition (SVD) & Low-Rank Approximations:**
  * *What to know:* How $W = U \Sigma V^T$ splits a matrix into orthogonal bases and singular values $\sigma_i$. Understand why truncating small singular values minimizes the Frobenius norm ($\|W - \hat{W}\|_F$).
* **Matrix Product Operators (MPO) / Tensor Trains:**
  * *What to know:* How a giant 2D weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is reshaped into a higher-order tensor and factored into a chain of small 4-dimensional tensor cores $\mathcal{G}_m$ connected by bond dimensions $\chi$.
* **QR Decomposition & Isometric Gauge-Fixing:**
  * *What to know:* Why orthogonalizing neighboring cores via QR ($Q^T Q = I$) prevents condition-number explosions ($\kappa = 1$) during iterative sweeps.
* **Ridge Regression / Closed-Form Least Squares:**
  * *What to know:* How the local update equation $\mathcal{G}_m^{\star} = Y Z^T (Z Z^T + \lambda I)^{-1}$ solves for an optimal tensor core in milliseconds without gradient descent.

---

## 2. Key Literature & Papers (Read in This Order)

1. **The Baseline to Beat (Multiverse Computing):**
   * *Paper:* **CompactifAI** (arXiv:2401.14109) and US Patent Application **US20250111212A1**.
   * *Takeaway:* Understand what they did (Tensor Train SVD + multi-GPU fine-tuning "healing") and why their reliance on GPU healing and un-fused contractions creates your opening.
2. **Why Plain Tensor Networks Fail on LLMs:**
   * *Paper:* **"Rethinking the Role of Tensor Decompositions in Post-Training LLM Compression"** (Zagitov et al., June 2026, `arXiv:2606.03465`).
   * *Takeaway:* Master the concepts of **Operator-Norm vs. Frobenius-Norm Mismatch** and **Progressive Angular Drift & Norm Shrinkage** across layers.
3. **The Activation-Aware SOTA (October 2026):**
   * *Paper:* **"Activation-Aware Weight Tensorization (AWT)"** (Beatini et al., Oct 7, 2026, `arXiv:2610.10085`).
   * *Takeaway:* See how preconditioning with input activation statistics massively closes perplexity gaps without retraining, and understand why their *diagonal* preconditioner left the full non-diagonal covariance problem unsolved.
4. **The Negative Result Warning:**
   * *Paper:* **"Pre-registered tests of solid-state-physics-inspired LLM compression"** (Jun-qiang Lu, Sept 2026, `arXiv:2609.34292`).
   * *Takeaway:* Learn why decorative physics metaphors fail when they don't directly minimize actual forward-pass activation distortion.

---

## 3. Core Mathematical Concepts in Nox

* **Activation Density Matrix ($\rho = XX^T$):**
  * *What to know:* Why input activations are not isotropic white noise ($\mathbb{E}[xx^T] \neq I$), and how projecting into the top $r = 128\text{--}256$ principal components captures $>99.9\%$ of the activation energy.
* **Alternating Linear Scheme (ALS) / 1-Site DMRG:**
  * *What to know:* How optimizing one tensor core at a time while holding all other cores fixed turns a hard global non-linear optimization problem into a sequence of easy local linear least-squares solves.
* **Quantization Error Absorption:**
  * *What to know:* How quantizing core $\mathcal{G}_1$ to 4-bit *before* solving core $\mathcal{G}_2$'s environment forces $\mathcal{G}_2$ to analytically compensate for $\mathcal{G}_1$'s rounding noise.

---

## 4. Systems & Execution Reality

* **Memory-Bandwidth Bound vs. Compute Bound:**
  * *What to know:* Why autoregressive decoding ($\text{batch\_size} = 1$) is entirely limited by how fast weights move from VRAM to SRAM/registers.
* **Kernel Fusion:**
  * *What to know:* Why un-fused PyTorch `einsum` loops slow down tensor networks, and how keeping intermediate bond dimensions ($\chi$) inside L1/L2 cache makes compressed MPOs faster than dense matrices.