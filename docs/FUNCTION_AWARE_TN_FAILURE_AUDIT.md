# Function-aware tensor-network failure audit

## Scope and evidence rules

This audit follows the committed implementation and artifact records, not the
original architecture pitch. Relevant source includes `nox/compiler.py`,
`nox/sequential_compiler.py`, `nox/block_compiler.py`, `nox/operator_schmidt.py`,
`nox/pees.py`, `nox/residual_codec.py`, `nox/joint_swiglu.py`,
`nox/grouped_swiglu_tn.py`, and the run drivers under `scripts/` and
`benchmarks/`. The main result trail is `EXPERIMENT_LOG.md` and
`experiment_artifacts/`.

The repository does not contain the ignored historical `.pt` weights and
compressed checkpoints. Therefore the tier coefficient counts below are
recomputed from the committed Llama shapes, site layouts, and rank budgets;
they are configuration-derived counts, not a fresh decode audit of the old
checkpoint files. The arithmetic is executable in
`scripts/audit_function_aware_parameters.py`. Tensor payload bytes, serialized
bytes, and numerical coefficient counts are kept separate.

## Recalculated coefficient counts

For the pinned Llama-2-7B layout, the dense model contains 6,738,415,616
stored numerical scalars: 6,476,005,376 attention and MLP projection scalars,
262,144,000 embedding and output-head scalars, and 266,240 RMSNorm scalars
(262,144 across the 64 block norms plus 4,096 final-norm values).
The grouped-SwiGLU manifest’s 6.197B count is consistent with the saved factor
shapes:

| Representation | Coefficients | Ratio vs dense | Reduction vs dense |
|---|---:|---:|---:|
| Dense Llama-2-7B | 6,738,415,616 | 1.000× | 0% |
| Legacy high-fidelity config, formula-derived | 939,044,864 | 7.176× | 86.1% |
| Legacy production config, formula-derived | 761,323,264 | 8.851× | 88.7% |
| Legacy extreme config, formula-derived | 611,015,936 | 11.028× | 90.9% |
| Grouped SwiGLU, rank 2612 in all 32 MLPs | 6,197,252,096 | 1.087× | 8.03% |
| Grouped SwiGLU, rank 1866 in all 32 MLPs | 5,115,564,032 | 1.317× | 24.1% |
| Direct packed 3-bit weights | 6,738,415,616 | 1.000× | 0% |

The old tier formula counts each TT-core scalar, each bridge factor scalar,
each sparse residual value, and each `d_awt` scale. Sparse coordinates and
checkpoint metadata are bytes but not numerical model coefficients. It keeps
all 32 layers’ norms, embeddings, and the language-model head dense, and uses
the saved boundary-layer upgrade rules. Its ratios are much lower than the
77×/18×/10× labels because the labels described a byte estimate, not the
number of independent scalar values.

There is a second accounting issue in the old estimates. At the original
artifact-writing commit, `total_kb` counted bridge values at 0.5 byte per
coefficient and sparse values at 1 byte each while the artifact writer saved
FP16 bridge tensors plus sparse coordinates and FP16 values. Those output
files therefore cannot be read as exact serialized size or coefficient
count. The later `estimate_artifact_tensor_bytes` accounts for FP16 factors
and 10 bytes per sparse value/index pair, but that still reports bytes rather
than independent coefficients.

For the latest grouped SwiGLU point, rank 2612 stores
`3 × 2612 × (4096 + 11008) = 118,354,944` factor values per layer. This is
87.5% of the dense MLP’s 135,266,304 values. Attention, norms, embeddings, and
the output head remain dense, so the full model retains 6,197,252,096 of
6,738,415,616 scalar values. This is about 1.09× whole-model parameter
compression, despite a larger MLP artifact-byte reduction. At rank 1866, the
MLPs retain 62.5% of their values, but model PPL rises to 22.47.

## Method-by-method findings

| Method | Mathematical attempt and parameter accounting | Where error entered and measured outcome | Best-supported failure diagnosis |
|---|---|---|---|
| Original four-site MPO / NEC | Reshape each projection into four output/input mode pairs, then store a TT/MPO with three internal ranks. For a 4-site core sequence, values are `Σ r[k]·o[k]·i[k]·r[k+1]`; NEC adds two low-rank bridge factors, sparse residual values, and diagonal activation scales. | The original site order imposes three simultaneous Schmidt cuts. In the layer-15 spectral audit, the central rank-128 cut discarded roughly 80–93% of squared residual energy depending on the projection. Layer-0 layer-output error was already material; the four-layer raw-WikiText prefix PPLs were 5.546, 10.151, 61.652, and 511.624 after compressing layers 0–3, against dense PPL 5.292. | The chosen tensorization and rank budget were not shown to be compressible enough. Errors also accumulated as later blocks saw shifted residual streams. Not a decoder corruption: reconstructed tensor audits matched the saved tensors. |
| DMRG/local core fitting | Warm-start TT-SVD followed by local least-squares updates to individual cores against calibration activations, optionally quantized in-loop and regularized with a diagonal weight prior. Every core scalar is a coefficient; DMRG does not lower the count beyond its chosen ranks. | For layer-15 V, validation error increased from about 0.716 after TT-SVD to 0.779 and 0.811 after two local sweeps even as calibration error fell. The same pattern appears across projections: removing the fitted core often improves validation output error while worsening calibration fit. Larger calibration coverage and a weight prior helped little. | Calibration generalization / local objective overfit, on top of insufficient TT expressivity. More sweeps are not a remedy. The records do not point to numerical instability as the dominant cause. |
| Sparse outlier shield | Select entries by `|W[i,j]| × channel_rms[j]`, store their values/positions, and factor the remainder. Each stored residual value is one model coefficient; coordinates add index bytes. | Sparse values preserve large weighted entries but leave most of each matrix to the low-rank bridge/MPO. The legacy tier byte budget omitted or undercounted index and FP16 payload costs. In the layer-0 component ablation, the sparse-only output errors remained large (roughly 0.64–0.82 across tested MLP projections). | Shielding did not make the residual low-entanglement, and its value count consumed a large fraction of the advertised budget. It is a useful residual control, not evidence of a 77× coefficient reduction. |
| Covariance-projected low-rank bridge | Fit `A B` with rank `r`; it contributes `(d_out+d_in)r` numerical values per projection, in addition to the core and sparse values. It targets calibration output directions and is serialized as two factors. | A bridge can dominate the parameter count. The bridge-only layer-0/15 ablations do not reproduce the dense projection on held-out activations. Its large rank makes many nominal compression ratios primarily byte/precision claims. | The bridge recovered calibration directions but was not sufficiently predictive out of sample; also, its coefficient cost was underrepresented by the original byte estimate. |
| Entanglement-ordered and balanced gauges | Apply common neuron permutations and reciprocal Up/Down scalings before a fixed tensorization. These transformations are exact; they store no extra values when folded into the matrices. | The gauge can lower a local neighbor-energy proxy, but the proxy is not a Schmidt-spectrum or functional metric. A measured exact Q/K and V/O balancing gauge gave layer-15 PPL 5.615 versus 5.613 for its matched control, a slight regression. | Exactness is supported; a useful compressibility benefit is not. The objective used to choose the order did not establish lower rank at matched functional error. |
| Operator-Schmidt unfolding | Reshape a matrix into four modes, choose a two-site bipartition, and truncate the unfolding SVD. The two factors contain `r(m+n)` scalars, plus any retained bridge, sparse values, and input scale. | On layer 0 the best tested operator-Schmidt unfolding lowered a block relative error from 0.36374 to 0.34828 (about 4.25% relative), with no meaningful storage reduction at the matched factor budget. The full MLP remained badly distorted; restoring dense MLP weights mattered far more than changing the attention representation. | This was a modest cut-selection gain, not evidence that the full function has low operator-Schmidt rank. The cut changes the matrix tensorization but not the function, and the chosen rank/coordinate structure remained too restrictive. |
| Cross-layer bridge and PEES | Fit low-rank corrections to local or propagated residual defects using calibration activations, with correction values counted as additional coefficients. PEES whitens token-space modes and truncates the propagated defect. | PEES improved a two-layer prefix PPL only from about 9.44 to 9.31 while adding about 24.5% bytes. Cross-layer defect corrections also improved selected interaction tests but did not establish a whole-model quality/parameter win. | Corrections captured some propagated directions but spent a large auxiliary budget and left substantial functional error. This also departs from a no-sidecar structural compressor. |
| Packed residual codecs | Store quantized integer codes and scale/index metadata. This changes bits per coefficient, not the number of logical weight entries. A full direct packed 3-bit checkpoint is a storage baseline, not a structural parameter reduction. | The full packed bundle is 3,161,886,843 bytes against a 13,476,876,272-byte FP16 source shard (4.262× smaller). On 10,000 tokens / 9,999 labels, PPL was 6.8153 vs 5.2922, next-token accuracy 0.5707 vs 0.6105, and dense top-1 agreement was 78.998%. | Storage reduction worked; it did not reduce independent coefficient count or prove faster inference. The evaluator reconstructs dense matrices. Quantization error is a separate distortion source from tensor truncation. |
| Shared-neuron SwiGLU | Build one rank-`r` shared intermediate basis for Gate preactivations, Up preactivations, and the nonlinear product. It stores `r(m+3d)` scalars, counting the shared basis once. | At matched layer-0 artifact bytes, the method reduced MLP artifact size by about 34.6% but worsened held-out MLP relative error from 0.626 to 0.741. Down-projection error reinforced the Gate/Up error. The readout feasibility control at rank 651 improved MLP error only to 0.459 and full-block error 0.294, missing the 0.273 block gate; rank 426 MLP error was 0.478. | A single shared neuron subspace is too restrictive for all three roles, and the downstream errors interfere. This was a specific shared-basis NO-GO, not a proof that every module-level method fails. |
| Grouped native-coordinate SwiGLU TN | Preserve all 11,008 neuron coordinates in groups and factor Gate, Up, Down independently with shared within-group rank. The artifact stores exactly the sum of all group core values; no dense residual is present. | At rank 2612 (87.5% of dense MLP coefficients), full-model PPL was 6.7789 vs 4.8597 dense on a 4,096-token test cap (4,095 scored labels). Next-token accuracy was 0.5621 vs 0.6281 (89.5% ratio). At rank 1866 (62.5% of MLP coefficients), PPL was 22.4716 and accuracy 0.4076. | The high-rank point is a usable quality-first reference but gives only 8.03% fewer whole-model coefficients. The lower-rank point shows a sharp function-quality cliff. It does not meet a 25–30% whole-model reduction target with 90% capability retention. |

## What the history supports

1. The main failure was not just “bad SVD.” A restrictive coordinate tensorization, rank limits, calibration-fit mismatch, component quantization, and sequential composition each contributed at different stages.
2. One-layer error was already too large in the old NEC runs; the later prefix collapse shows accumulated functional error and distribution shift on top of it.
3. High calibration error fit did not predict held-out output quality. More DMRG sweeps, larger bridges, and a diagonal prior did not fix the observed generalization gap.
4. Serialization was audited and is not the cause of the old PPL collapse. The raw WikiText-2 source and Llama RMSNorm epsilon were corrected, which changed the reference PPL and parity but did not rescue the old compression tiers.
5. The exact transformations so far preserve functions but have not produced a measured coefficient-distortion advantage. Any new rotation must be proved against RoPE, RMSNorm, the residual path, and the nonlinear SwiGLU computation before it is treated as a free gauge.
6. The latest 6.78 PPL point is a valid reference under its recorded 4,096-token protocol; it should not be compared numerically to 9,999-label PPL runs without stating the evaluation window/count.

## Hypotheses still untested

- Whether exact RoPE-commuting headwise factors or a whole-model signed hidden-coordinate permutation improve MPO cut spectra on layers 0, 15, and 31 after their parameter overhead is included.
- Multi-stage butterfly and Kronecker-structured exact gauges were not implemented in this bounded search. A butterfly or Kronecker factor could be exact within a V/O cancellation gauge, but must be folded into the projection factors or charged as auxiliary coefficients; Q/K mixing is limited to the RoPE commutant, and residual rotations are limited by the learned diagonal RMSNorm gains.
- Whether alternative four-site orderings produce consistent coefficient/error improvements across Q, K, V, O, Gate, Up, and Down rather than one favorable layer-0 cut.
- Whether a function-aware Down readout, paired with fixed-rank Gate/Up approximations, reduces held-out complete-SwiGLU error at an exactly matched coefficient count.
- Whether that improvement survives 4K, 16K, and (if feasible) 65K calibration tokens and then 1/2/4-layer prefix propagation.
- Whether any tensor-network candidate beats ordinary SVD, activation-weighted SVD, the existing grouped TN, and a clearly labeled MoDeGPT-inspired functional-readout control at matched scalar counts.
- No current artifact demonstrates 90% retention on MMLU, HellaSwag, BoolQ, ARC-Challenge, or GSM8K. The complete dense MMLU run was interrupted, and there is no compressed MMLU score.

## Decision before running the new batch

The old NEC coefficient ratios are lower than their titles imply, but remain
far more aggressive than the 1.09× grouped-TN point and fail on quality. Stage
A must show a cross-layer coefficient-versus-distortion advantage before a
reparameterized tensor network is treated as promising. Stage B remains
independent: a direct complete-module output fit may still beat the old
matrix-first approach even if Stage A is negative.
