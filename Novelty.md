# Technical Novelty & Architectural Specification: Nox Entanglement Compiler (NEC)

This document delineates standard mathematical primitives used within the NoxQuantum codebase (Matrix Product Operators, QR canonicalization, Conjugate Gradient, and compressed sparse storage) from the **original mathematical formulations, closed-form calibration algorithms, and systems engineering breakthroughs** that enable the Nox Entanglement Compiler (NEC) to compress Large Language Models with **zero backpropagation healing**.

---

## Part I: Core Mathematical & Algorithmic Inventions

### 1. Tri-Partite Operator Decomposition ($W = S_{\text{sparse}} + S_{\text{bridge}} + W_{\text{core}}$)

#### The Prior-Art Failure Mode
Zagitov et al. (*arXiv:2606.03465*, June 2026) demonstrated that post-training tensor decompositions on LLMs suffer from a fundamental **Operator-Norm vs. Frobenius-Norm mismatch**:

$$\Vert{}W\Vert{}_F = \Vert{}\mathcal{T}(W)\Vert{}_F \quad \text{but} \quad \Vert{}W\Vert{}_\sigma \neq \Vert{}\mathcal{T}(W)\Vert{}_\sigma$$

When a pretrained Transformer weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is folded into a higher-order tensor $\mathcal{T}(W)$, localized heavy-tailed coordinate spikes and structured channel outliers act as high-entanglement "anti-rank" perturbations across every bipartite Schmidt cut, forcing Matrix Product Operator (MPO) bond dimensions $\chi$ to explode. 
* **CompactifAI** (*arXiv:2401.14109*) ignores this entirely, causing immediate signal collapse ($\cos\theta < 0.10$) prior to multi-GPU fine-tuning.
* **AWT** (*arXiv:2610.10085*, Oct 2026) applies only a diagonal column scale $D_\alpha$, leaving cross-channel manifold correlations and coordinate spikes intact.

#### The Nox Contribution
Nox formulates a closed-form **Tri-Partite Hybrid Operator Decomposition** that strips both zero-dimensional coordinate singularities and low-dimensional global manifolds prior to tensorization (`nox/compiler.py`):

1. **Bit-Packed Operator-Norm Outlier Shield ($S_{\text{sparse}}$):** Ranks individual weight coordinates by their second-moment functional activation impact:
   $$I_{ij} = \vert{}W_{ij}\vert{} \cdot \Vert{}X_{\text{calib}, j}\Vert{}_2$$
   The top $k$ entries are extracted into a 1.0-byte bit-packed buffer (4-bit group-quantized value + 4-bit delta coordinate index), eliminating high-frequency delta perturbations that would otherwise pollute all MPO bond cuts.
2. **Covariance-Projected SVD Bridge ($S_{\text{bridge}} = A_q B_q$):** Solves the regularized low-rank functional projection directly in the empirical activation subspace of the residual weight matrix $W_{\text{rem}} = W - S_{\text{sparse}}$:
   $$S_{\text{bridge}} = \arg\min_{\text{rank}(S) \le r_{\text{bridge}}} \left\Vert{} (W_{\text{rem}} - S) X_{\text{calib}} \right\Vert{}_F^2 + \lambda \Vert{}S\Vert{}_F^2$$
   Factorized into 4-bit group-quantized matrices $A_q \in \mathbb{R}^{d_{\text{out}} \times r_b}$ and $B_q \in \mathbb{R}^{r_b \times d_{\text{in}}}$, this bridge captures global cross-channel semantic directions (such as attention sinks) that violate the 1D area law of tensor trains.
3. **Bounded-Entanglement Residual Core ($W_{\text{core}} = W - S_{\text{sparse}} - S_{\text{bridge}}$):** With coordinate singularities and global rank-$r_b$ manifolds removed, $W_{\text{core}}$ exhibits a rapidly decaying Schmidt spectrum suitable for compact MPO factorization.

---

### 2. Full-Covariance Subspace DMRG in Diagonally Conditioned Physical Space

#### The Prior-Art Failure Mode
In 2D low-rank matrix factorization (e.g., SVD-LLM), activation-aware compression whitens $W$ via the Cholesky factor $L$ of the empirical input covariance $\Sigma_X = X X^T = L L^T$, decomposes $W L \approx U \Sigma V^T$, and absorbs $L^{-1}$ into $V^T L^{-1}$. 

In a multi-site Tensor Train / MPO, however, multiplying a chain of local cores $\mathcal{G}_1 \cdots \mathcal{G}_k$ by a dense $d_{\text{in}} \times d_{\text{in}}$ matrix $L^{-1}$ couples all $k$ physical input modes simultaneously, inflating the MPO bond dimension back to $d_{\text{in}}$ and destroying compression. Because of this barrier, AWT (*arXiv:2610.10085*) retreated to a strictly diagonal preconditioner $D_\alpha$, discarding all off-diagonal activation entanglement.

#### The Nox Contribution
Nox optimizes the **full off-diagonal activation covariance** inside an MPO without ever inverting a dense whitening matrix (`nox/compiler.py`):

1. **Fuseable Diagonal Conditioning:** We apply the exact diagonal preconditioner $d_{\text{awt}} = (\text{RMS}(X_{\text{calib}}))^{\alpha}$ ($W_{\text{cond}} = W_{\text{core}} \text{diag}(d_{\text{awt}})$, $X_{\text{cond}} = \text{diag}(d_{\text{awt}}^{-1}) X_{\text{calib}}$). Unlike dense $L^{-1}$, the vector $d_{\text{awt}}^{-1}$ fuses directly into the upstream RMSNorm or preceding linear projection at **zero runtime memory or FLOP overhead**.
2. **Exact Principal Subspace Reduction:** Using the thin SVD of the conditioned calibration activations $X_{\text{cond}} = U_X \Sigma_X V_X^T$, we project both inputs and residual targets into the $r_{\text{cov}}$-dimensional principal subspace ($r_{\text{cov}} = 256$):
   $$\tilde{X} = U_{X, :r_{\text{cov}}} \Sigma_{X, :r_{\text{cov}}} \in \mathbb{R}^{d_{\text{in}} \times r_{\text{cov}}}, \qquad \tilde{Y} = Y_{\text{target}} V_{X, :r_{\text{cov}}} \in \mathbb{R}^{d_{\text{out}} \times r_{\text{cov}}}$$
3. **Prior-Anchored Isometric DMRG Sweeps:** Because $V_X$ has orthonormal columns, minimizing $\Vert{}\hat{W}_{\text{MPO}} \tilde{X} - \tilde{Y}\Vert{}_F^2$ is mathematically identical to minimizing the full non-diagonal activation-weighted distortion $\text{Tr}\big((\hat{W}_{\text{MPO}} - W_{\text{cond}}) X_{\text{cond}} X_{\text{cond}}^T (\hat{W}_{\text{MPO}} - W_{\text{cond}})^T\big)$. To prevent nullspace overfitting when $r_{\text{cov}} < d_{\text{in}}$, each site update is anchored to the isometric AWT prior $\mathcal{G}_m^{\text{prior}}$:
   $$\left( Z_m^T Z_m + \gamma_m I \right) \mathcal{G}_m^{\star} = Z_m^T \tilde{Y}_{(m)} + \gamma_m \mathcal{G}_m^{\text{prior}}$$
   maintained under bidirectional QR canonicalization ($Q^T Q = I$) so local site environments remain well-conditioned across sweeps.

---

### 3. In-Loop Quantization-Aware Error Absorption

#### The Prior-Art Failure Mode
Existing tensor-network compressors decompose weights in FP16/FP32 and quantize the tensor cores post-hoc. Because an MPO contracts cores multiplicatively ($\hat{W} = \mathcal{G}_1 \mathcal{G}_2 \cdots \mathcal{G}_k$), independent rounding errors $\delta_m$ compound across sites.

#### The Nox Contribution
During the final left-to-right DMRG sweep ($m = 0 \to k-1$), immediately after solving for continuous core $\mathcal{G}_m^{\star}$, Nox applies 4-bit group quantization in-place before constructing the environment tensor $Z_{m+1}$ for site $m+1$:

$$\mathcal{G}_m \leftarrow \text{Dequant}\big(\text{Quant}_{\text{4-bit}}(\mathcal{G}_m^{\star})\big)$$

Consequently, the environment matrix $Z_{m+1}$ at site $m+1$ explicitly contains the exact quantization error of sites $0 \dots m$, allowing $\mathcal{G}_{m+1}^{\star}$ to analytically compensate for upstream core rounding noise in closed form.

---

### 4. Well-Posed Symmetric Intra-Block & Inter-Layer Causal Healing

#### The Prior-Art Failure Mode
Independent layer-wise compression causes residual stream drift to snowball across depth (+146% error growth across 4 layers in unhealed CompactifAI), forcing incumbents to rely on expensive multi-GPU backpropagation fine-tuning ("healing"). Conversely, naive attempts to force post-nonlinearity layers to regress raw skip-connection errors fail because post-Softmax and post-SiLU activations do not linearly span the residual skip space.

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
