#### The Nox Contribution
Nox implements a strictly well-posed **Intra-Subspace Causal Cascade** (`nox/block_compiler.py`, `nox/sequential_compiler.py`) where every projection heals upstream drift inside its native linear manifold:
1. **Upstream Linear Drift Absorption ($W_q, W_k, W_v$ and $W_{\text{gate}}, W_{\text{up}}$):** Conditioned on the actual corrupted residual stream arriving from preceding compressed blocks ($\text{RMSNorm}(X_{\text{comp}})$), while targeting the clean reference projections $W \cdot \text{RMSNorm}(X_{\text{dense}})$.
2. **Downstream Non-Linear Distortion Damping ($W_o$ and $W_{\text{down}}$):** $W_o$ is calibrated on the corrupted post-Softmax attention context $A_{\text{comp}} = \text{Softmax}(\hat{Q}\hat{K}^T/\sqrt{d})\hat{V}$ targeting clean $Y_{\text{attn}}^{\text{dense}}$, while $W_{\text{down}}$ is calibrated on the corrupted SwiGLU activation $H_{\text{comp}} = \text{SiLU}(\hat{W}_{\text{gate}} M_{\text{comp}}) \odot (\hat{W}_{\text{up}} M_{\text{comp}})$ targeting clean $Y_{\text{mlp}}^{\text{dense}}$.

---

## Part II: Systems & Numerical Engineering Breakthroughs

### 5. Prime-Factor Boundary Pinning: `(43, 4, 8, 8)` Mode Topology
In LLaMA architectures, the SwiGLU intermediate dimension $d_{\text{mlp}} = 11,008 = 43 \times 256$ contains a large prime factor ($43$). Naive mode orderings that place $43$ on an interior site (e.g., $i_2 = 43$ with bond ranks $r_2 = 80, r_3 = 32$) inflate the local site regression dimension to $D_2 = 80 \times 32 \times 43 = 110,080$, requiring $>48\text{ GB}$ of Gram matrix memory and $O(D_2^3) \approx 1.3 \times 10^{15}$ FLOPs.

By pinning the prime factor $43$ to the boundary site $m=0$ (`(43, 4, 8, 8)` where $r_0 = 1$) while pairing the interior cuts with small powers of two ($i_1 = 4, i_2 = 8$) and an Area-Law Schmidt bond profile (`[1, 32, 80, 32, 1]`), Nox reduces the site dimension from $110,080 \to 1,376$ (an **80x dimension reduction** and **512,000x solver FLOP reduction**), enabling full-layer compilation on a single 15 GB T4 GPU.

### 6. Dual-Regime Cholesky + Prior-Warm-Started Matrix-Free Conjugate Gradient
To eliminate the $O(D^3)$ inversion bottleneck and $O(D^2)$ Gram matrix memory spike at interior sites ($D > 4096$):
* **Boundary Sites ($D \le 4096$):** Solved via chunked FP16/FP32 `addmm_` accumulation and symmetric positive-definite Cholesky decomposition (`torch.linalg.cholesky_ex`).
* **Interior Sites ($D > 4096$):** Solved via **Prior-Warm-Started Conjugate Gradient (CG)** initialized at the AWT warm-start $\mathcal{G}_m^{(0)} = \mathcal{G}_m^{\text{prior}}$. Because $\mathcal{G}_m^{\text{prior}}$ lies inside the quadratic basin of attraction, 6 matrix-free CG iterations converge to $<5 \times 10^{-4}$ relative parity while reducing FLOPs by $>200\times$.

### 7. The "Skip-Connection Illusion" Audit & Iso-Byte Fairness Protocol
Standard Transformer evaluations evaluate relative error on the full block output $X_{\text{out}} = X_{\text{in}} + \Delta_{\text{block}}$. Because $\Vert{}X_{\text{in}}\Vert{}_F \approx 6.7 \times \Vert{}\Delta_{\text{block}}\Vert{}_F$, a collapsed compression algorithm that outputs $\Delta_{\text{block}} \approx 0$ still achieves $\sim 0.15$ relative error and $0.989$ cosine similarity on $X_{\text{out}}$ by coasting on the passive identity skip connection.

Nox establishes two rigorous evaluation standards:
1. **Isolated Block Delta Evaluation ($\Delta_{\text{block}} = Y_{\text{attn}} + Y_{\text{mlp}}$):** Directly measures whether the compressed block performs non-trivial neural computation.
2. **Iso-Byte Capacity Matching:** Evaluates baseline TT-SVD (CompactifAI) at the exact same physical byte footprint ($\ge 5.2\text{ MB/layer}$) as Nox. Empirical testing on LLaMA-2-7B Layers $15 \to 18$ proves that even with $7.5\times$ more parameters, unshielded TT-SVD remains functionally dead ($\Delta_{\text{block}}$ Relative Error $= 0.9993$, $\cos\theta = 0.012$), whereas Nox preserves active block transformations ($\cos\theta = 0.732$) and cuts compounding multi-layer residual drift without backpropagation.
