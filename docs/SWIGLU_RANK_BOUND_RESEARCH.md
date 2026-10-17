# SwiGLU rank-floor and tensor-network feasibility review

**Status (2026-10-17):** mathematical diagnosis and CPU/CUDA diagnostic harness complete; six synthetic CPU tests pass. The real layer-0 rank-floor / Down-crossover diagnostic has now run on the existing A10 using the saved checkpoint and 4,096 held-out activations on both dense and compressed trajectories. No layer compilation, perplexity evaluation, new compressor, or instance power action was performed. Raw results are backed up locally in `experiment_artifacts/diagnostic_36h/swiglu_rank_bound/layer0/`.

**Superseding decision:** the final fixed-feature readout experiment at ranks 426 and 651 is complete and returned **NO-GO**. Rank 651's calibration-only readout passed the MLP and storage gates but missed the full-block gate (0.29443 versus 0.273). The shared-neuron line is closed rather than extended with another workaround. See `experiment_artifacts/diagnostic_36h/swiglu_readout_feasibility/layer0/REPORT.md`. The immediate milestone is now a quality-first structural compression baseline at a more conservative reduction rate.

## 1. The shared-neuron candidate has a hard output-rank ceiling

The failed joint candidate stores \(Z\in\mathbb R^{m\times r}\), \(G_r,U_r\in\mathbb R^{r\times d}\), and \(D_r\in\mathbb R^{d\times r}\), and represents

\[
\widehat G=ZG_r,\qquad \widehat U=ZU_r,\qquad
\widehat D=D_rZ^\top.
\]

For any fixed sample matrix \(X\in\mathbb R^{d\times n}\), let

\[
\widehat H(X)=\operatorname{SiLU}(\widehat G X)\odot(\widehat U X).
\]

The resulting sampled output matrix is

\[
\widehat Y=\widehat D\widehat H=D_r\underbrace{(Z^\top\widehat H)}_{r\times n},
\qquad \therefore\quad \operatorname{rank}(\widehat Y)\le r.
\]

The key point is that the nonlinear calculation can produce an arbitrary vector in the \(m\)-dimensional neuron space, but the final Down map projects every sample through the same \(r\)-dimensional output range — the nonlinear part cannot undo that bottleneck. Here the saved candidate used \(r=426\); the dense model dimension is \(d=4096\), so the dense sampled output can have rank up to 4096.

For dense held-out outputs (Y=[y(x_1),\dots,y(x_n)]\), with singular values (\sigma_1\ge\dots\ge\sigma_q\), Eckart–Young gives the exact lower bound

\[
\min_{\operatorname{rank}(\widetilde Y)\le r}
\frac{\|Y-\widetilde Y\|_F}{\|Y\|_F}
=\sqrt{\frac{\sum_{i>r}\sigma_i^2}{\sum_i\sigma_i^2}}.
\]

This is an optimistic floor for the shared-rank candidate: its Gate/Up construction imposes further constraints. It is **not** a lower bound for all tensor networks. In particular, a matrix product operator (MPO) can have high ordinary matrix rank while keeping low ranks in its tensor unfoldings; the exact expressivity depends on how row/column dimensions are tensorized and ordered.

The second restriction is nonlinear non-equivariance. For a general neuron-space linear map \(P\),

\[
\operatorname{SiLU}(Pg)\odot(Pu)\ne
P\big[\operatorname{SiLU}(g)\odot u\big]
\]

in general. Coordinate permutations commute with a shared elementwise activation and Hadamard product; arbitrary rotations, projections, and rescalings do not. Thus \(Z\) is not merely a harmless change of basis: it changes which quantities meet coordinatewise at the SwiGLU gate. This distinguishes a neuron-axis compression from a gauge transformation on the residual stream.

The exact candidate error decomposition remains

\[
\widehat D\widehat h-Dh
=D(\widehat h-h)+(\widehat D-D)\widehat h,
\quad h=\operatorname{SiLU}(Gx)\odot(Ux).
\]

The measured held-out compressed-trajectory values were: product error 0.7420 → 0.5877 (improved), product error propagated through dense \(D\) 0.6202 → 0.4524, Down term 0.2097 → 0.5210, and product/Down cross-energy fraction −0.0853 → +0.1542. Complete MLP error consequently worsened 0.6262 → 0.7413. The Down approximation turned partial cancellation into reinforcement. The rank floor will distinguish how much of this is unavoidable for rank 426 from how much is due to the particular \(Z\) and \(D_r\); it has not yet been measured.

## 2. Diagnostic implementation and measured results

`nox/swiglu_rank_bound.py` contains small, non-compressor routines for the spectral tail, exact SwiGLU function, and controlled Down crossovers. `scripts/diagnose_shared_swiglu_rank_bound.py` reads the saved layer checkpoint and dense weights, computes the same dense and compressed held-out transformer trajectories as the previous diagnostic, then reports:

- the full singular spectrum and relative Eckart–Young tail curve for dense MLP outputs on both held-out trajectories, including ranks 322 and 426;
- candidate Gate/Up held fixed while swapping candidate Down, saved-NEC Down, and dense-oracle Down;
- saved-NEC baseline, complete MLP error, product and Down contributions, their interference, and decomposition-identity residuals;
- hashes, coefficient count, candidate rank, runtime, CUDA peak memory, and candidate serialization size.

The dense-Down case is explicitly an oracle and not a deployable result. The script performs no optimization. Six synthetic CPU tests cover the Eckart–Young formula, curve monotonicity, the rank ceiling, non-equivariance, matrix conventions, and exact crossover decomposition; all six passed. Python compilation and CLI help passed. The CUDA run took 75.86 seconds, with 4.06 GB peak PyTorch allocation (2.30 GB incremental); `nvidia-smi` observed 5,483 MiB total GPU memory while active.

At rank 426, the Eckart–Young held-out output error floor was **0.24280** on the dense trajectory and **0.24142** on the compressed trajectory. At ranks 322 / 426 / 512 / 1,024, the errors were 0.2784 / 0.2428 / 0.2177 / 0.1148 (dense trajectory), and 0.2782 / 0.2414 / 0.2155 / 0.1088 (compressed trajectory). Thus the rank-426 output-range ceiling does **not** by itself rule out the 25%-improvement target (error ≤0.470); it is an optimistic floor, not evidence that the candidate can attain it.

Held-out MLP crossovers (relative Frobenius error, candidate Gate/Up held fixed):

| Input trajectory | Saved NEC baseline | Candidate G/U + dense Down oracle | Candidate G/U + saved NEC Down | Candidate G/U + candidate Down |
|---|---:|---:|---:|---:|
| Dense | 0.6623 | 0.4842 | 0.6262 | 0.7676 |
| Compressed | 0.6262 | 0.4524 | 0.5975 | 0.7413 |

On the compressed trajectory, replacing only candidate Down with saved NEC Down improves over the saved baseline by 4.6%; the dense-Down oracle improves by 27.8% but is not deployable; the candidate's own Down raises error by 18.4%. For the latter, the product error is 0.5877, its error after dense Down is 0.4524, and the candidate Down-error term has norm 0.5210 relative to the dense output. Product/Down cross-energy is +0.1542, so the errors reinforce. This localizes the dominant measured defect to the candidate Down representation, not the product alone. It does **not** show that a rank-426 Down factor can attain the spectral floor.

**Reproducibility caveat:** the original GPU diagnostic did not serialize the candidate factors, only its measurements. The harness accepts `--candidate-artifact` for exact crossovers if those factors are supplied. Otherwise it rebuilds the same construction from calibration with seed 0; randomized SVDs need not be bitwise identical across devices or library builds. The rank-floor calculation is independent of candidate factors. This run's held-out crossover values reproduce the prior diagnostic to the displayed precision.

The completed CUDA command (run from the repository root on the existing A10) was:

```bash
.venv/bin/python scripts/diagnose_shared_swiglu_rank_bound.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache \
  --layer 0 --device cuda --cpu-threads 8 --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --output experiment_artifacts/diagnostic_36h/swiglu_rank_bound/layer0/layer0_rank_bound_cuda_run01.json
```

No full-model inference is part of this command. The SVD is over a 4096-by-4096 held-out MLP output matrix. The full rank curve, hashes, crossover decompositions, runtime, and resource counters are in `experiment_artifacts/diagnostic_36h/swiglu_rank_bound/layer0/layer0_rank_bound_cuda_run01.json`; a concise report and run log are adjacent.

## 3. Related research: what transfers and what does not

| Work | Actual mechanism and result | Relevance / boundary for NEC |
|---|---|---|
| [MoDeGPT, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/fb7214d2fdfd84165b08539d59c92e07-Paper-Conference.pdf) | Minimizes a calibration-set module output reconstruction loss for pairs of operators. Its MLP form explicitly covers gated MLPs by concatenating gate/up operators, selecting neuron columns, then solving a compensated Down map through an activation-correlation/Nyström construction. The abstract reports 90–95% zero-shot retention at 25–30% compression; body tables vary with model, calibration set, and compression level (for example, LLaMA-2-7B's 30% WikiText2-calibrated average is 60.78 vs 69.00 dense). These results are not a guarantee at NEC's much smaller coefficient budgets. | Strongest precedent for preserving the true nonlinear module and native neuron coordinates. Selection commutes with elementwise activation; arbitrary shared projections do not. It is a width-selection/low-rank method, not an MPO, and does not show that a tensorized full-width map fits NEC's budget. Its calibration objective is a baseline to compare against, not a novelty claim. |
| [FLAP, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28960) | Uses a channel fluctuation score (activation variance weighted by the corresponding weight-column norm), adaptive structured pruning across layers/modules, and a deterministic baseline-bias term to compensate for pruned channels. Reports training-free LLaMA results, including WikiText-2 PPL 14.62 at 20% pruning and 17.62 at 30% for LLaMA-7B vs dense 12.62. | Supports activation-sensitive structural budget allocation and cheap compensation, but the compensation is an added bias approximation; it does not encode the full nonlinear SwiGLU operator as a compact TN. |
| [SliceGPT, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/316648eb8b4ffb6010f531b07848c300-Paper-Conference.pdf) | Uses an exact orthogonal invariance of RMSNorm-connected residual-stream computations to rotate hidden coordinates, then slices low-variance dimensions; adjacent blocks and skip paths are transformed consistently. Reports 25% slicing retention of 99%, 96%, and 87% of dense scores for OPT-66B, LLaMA-2-70B, and Phi-2, respectively. | Validates exploiting a real model symmetry. That symmetry is on the residual hidden axis. It does not license arbitrary rotations of the intermediate SwiGLU neuron axis, where SiLU and the Hadamard product are coordinatewise. |
| [SVD-LLM V2, NAACL 2025](https://aclanthology.org/2025.naacl-long.217/) | Allocates heterogeneous truncation ratios across matrix types/layers from truncation-loss estimates, then performs activation-covariance-aware SVD truncation. Its local objective is \(\|WX-W'X\|_F^2\); the paper uses eigendecomposition/SVD of \(XX^\top\) to avoid the positive-definiteness/numerical issues of a Cholesky whitening route. | Good precedent for loss-aware rank allocation and stable algebraic initialization. Still operates per matrix and does not directly optimize the complete Gate/Up/Down nonlinear output or error cancellation. |
| [Tensor Train, Oseledets (2011)](https://epubs.siam.org/doi/10.1137/090752286); [Tensorizing Neural Networks, NeurIPS 2015](https://arxiv.org/abs/1509.06569); [Tensorized Transformer, NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/dc960c46c38bd16e953d97cdeefdbc68-Abstract.html) | TT/MPO stores a tensor as a chain of cores; low bond ranks constrain ranks of chosen unfoldings. TT/MPO parameter counts can grow linearly in site count at fixed local dimensions and bond rank. Earlier neural-network work applied this to dense layers; tensorized-transformer work used block-term structure for attention. | Supports compact operator storage without requiring low ordinary matrix rank. It also makes tensorization/site order a first-order modeling decision: low unfolding rank is not automatic, and the NEC spectral-cut failures show the current layout can be poor. Neither citation establishes zero-healing Llama-7B quality for NEC's operator tensors. |
| [MPOP, ACL 2021](https://aclanthology.org/2021.acl-long.418/); [MPO PicoGPT study, 2026](https://arxiv.org/abs/2603.28534) | MPOP uses MPO compression with lightweight fine-tuning of auxiliary tensors. The 2026 PicoGPT study reports TT-SVD initialization followed by training/fine-tuning on a roughly 1M-parameter character model. | Both are useful implementation precedents but their reported quality relies on weight optimization or a much smaller/toy training regime; neither demonstrates zero-healing compression of a pretrained 7B LLM. |

The literature supports the *principles* that are relevant here: preserve actual module outputs, allocate budget non-uniformly, respect exact symmetries, and treat TT rank as tensorization-dependent. It does not supply evidence that an NEC-scale tensor network at roughly 7.3% of dense MLP coefficients can meet the 90% downstream-retention objective without healing.

## 4. Parameter and byte accounting

For Llama-2-7B MLP dimensions \(d=4096\), \(m=11008\), three dense matrices contain

\[
N_{\rm dense}=3dm=135{,}266{,}304
\]

coefficients (270,532,608 bytes at FP16, excluding metadata). The failed shared candidate used

\[
N_{\rm shared}=r(m+3d)=426(11008+3\cdot4096)=9{,}924{,}096
\]

floating coefficients, 0.20% below the saved NEC count of 9,943,596. Its artifacts serialized to 19,850,909 bytes versus 30,370,779 bytes for the saved NEC MLP artifacts. The byte reduction was mainly sparse-index/schema payload, not a comparable reduction in floating coefficients. These are MLP artifacts only, not layer or model totals.

For a matrix \(W\in\mathbb R^{n_o\times n_i}\), factor \(n_o=\prod_{k=1}^L p_k\), \(n_i=\prod_{k=1}^L q_k\), and represent the tensorized matrix as an MPO with bond dimensions \(r_0=r_L=1\). Its coefficient count is

\[
N_{\rm MPO}(W)=\sum_{k=1}^L r_{k-1}p_kq_kr_k.
\]

A neuron-coordinate-preserving MLP candidate would use a separate MPO for \(G,U,D\), with native intermediate coordinates represented by the full physical index (no common \(Z\in\mathbb R^{m\times r}\) projector):

\[
N_{\rm total}=N_{\rm MPO}(G)+N_{\rm MPO}(U)+N_{\rm MPO}(D)\le 9{,}943{,}596.
\]

This count is only a budget constraint; actual artifact bytes must include core dtype, shapes, sparse/index data if retained, serialization overhead, and any auxiliary tensors. No speedup follows from coefficient reduction alone; the runtime would need a direct factorized contraction benchmark.

## 5. Historical next-candidate proposal (superseded; do not implement)

**Native-neuron, whole-function output-sensitive MPO.** Keep the coordinatewise SwiGLU operation on all native neuron channels. Represent Gate, Up, and Down as separate MPOs so that no shared linear projector forces ρ(\(\widehat D\)) ≤ 426. Couple them only through a module-level calibration objective and a shared coefficient budget, not through a low-rank neuron basis.

For \(g=Gx\), \(u=Ux\), \(h=\operatorname{SiLU}(g)\odot u\), its first-order output perturbation is

\[
\delta y\approx \Delta D\,h+
D\left[(u\odot\operatorname{SiLU}'(g))\odot(\Delta Gx)
+\operatorname{SiLU}(g)\odot(\Delta Ux)\right].
\]

This gives output-sensitivity weights for selecting which MPO cuts/ranks deserve coefficients. The actual screening metric must remain the exact nonlinear module loss

\[
L_{\rm MLP}=\sum_{x\in\mathcal C}
\|D_{\rm MPO}[\operatorname{SiLU}(G_{\rm MPO}x)\odot(U_{\rm MPO}x)]-D[\operatorname{SiLU}(Gx)\odot(Ux)]\|_2^2,
\]

evaluated on held-out data separately. Candidate construction should start from algebraic TT-SVD/operator-Schmidt factors, enumerate/select bond profiles under the exact complete-artifact budget using calibration-only function errors, and avoid another calibration-only DMRG sweep at the outset. No dense residual or post-finalization weight update is part of this proposal.

This candidate is prioritized because it removes both diagnosed restrictions (common neuron subspace and independent-matrix objective) while retaining compact TN operators. It is **not yet justified for implementation**: existing MPO cuts discard substantial residual spectral energy, and prior DMRG fitting overfit calibration. The rank-tail/crossover gate is now complete; next work should remain a CPU-side algebraic coefficient/profile screen, not another GPU run.

- If the rank-426 floor is at or above 0.470 (the 25%-improvement target over saved NEC's 0.6262 MLP error), no rank-426 shared-output representation can meet that target, whatever its Gate/Up fitting; abandon that topology.
- The measured floor (about 0.241) is far below the candidate's 0.7413 error, so the output-rank ceiling alone does not explain the failure. Candidate Gate/Up plus saved NEC Down reaches 0.5975, only a 4.6% improvement on the compressed trajectory; the dense-Down oracle reaches 0.4524, while candidate Down reaches 0.7413. This localizes substantial recoverable error to the candidate Down map, while leaving open whether a different rank-426 Down factor can recover it.
- The native-neuron, output-sensitive MPO is the single prioritized research candidate because it can preserve the neuronwise nonlinearity without imposing a common rank-426 output basis. Before implementation, perform a CPU-side algebraic coefficient/profile screen and specify a held-out function objective; a measurable one-layer advantage remains mandatory.

The eventual one-layer gate remains approximately 25% relative improvement in both held-out MLP and full-block errors at matched *complete artifact bytes*, with no active-transform regression. A pass is not downstream capability evidence; after this research gate, any model-level evaluation remains a separate authorization and experiment.

## 6. Main failure modes and unresolved points

1. **Rank ceiling versus direction error.** The tail bound quantifies the best possible rank-426 sampled output approximation, while candidate crossovers reveal whether \(D_r\) specifically is the culprit. Neither alone establishes behavior outside the held-out span.
2. **Tensorization entanglement.** A native-coordinate MPO avoids ordinary rank 426, but bad site order or high operator-Schmidt spectra can still make the coefficient budget inadequate. Previous NEC cuts discarded about 80–93% of residual squared spectral energy.
3. **Function-loss fitting can overfit.** The calibration-vs-held-out DMRG result already worsened from 0.716 to 0.811 with more sweeps. Use output loss for rank screening and report the held-out gap; do not equate optimization convergence with preservation.
4. **First-order sensitivity is only a selector.** With large Gate/Up errors, second-order terms and cancellation matter. Final decisions require the exact SwiGLU product and complete MLP output.
5. **Coefficient equality is not byte equality or fast inference.** Include all factors/indexes/metadata in file size and separately benchmark direct contractions.
6. **Layer-0 MLP quality is not model capability.** Even a strong one-layer result would still need paired multi-layer perplexity and benchmark retention; no such capability result is part of this diagnostic.
