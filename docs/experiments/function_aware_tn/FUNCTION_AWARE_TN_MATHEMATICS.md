# Function-aware tensor-network mathematics

## Model and coefficient convention

The target is `NousResearch/Llama-2-7b-hf` at revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`, with the repository’s pinned raw
WikiText-2 source. A “coefficient” in this report means one stored numerical
scalar in a model weight or factor. This is distinct from checkpoint bytes,
quantization bits, and the unconstrained degrees of freedom of an orthogonal
factor. Shared numerical values are counted once. Sparse coordinate indices
and container metadata affect bytes but are not numerical coefficients.

For an MPO with physical dimensions `p_k = o_k i_k` and ranks
`r_0 = r_4 = 1`, the scalar count is

\[
N_{\mathrm{TT}} = \sum_{k=0}^{3} r_k p_k r_{k+1}.
\]

The old NEC representation adds `(d_out + d_in) r_bridge` bridge values,
`n_sparse` residual values, and `d_in` diagonal activation scales for each
projection. The old byte estimate did not count those same values at their
serialized dtypes; see the failure audit and
`scripts/audit_function_aware_parameters.py`.

## Exact representation changes

### Q/K under RoPE

Let a query/key head’s pre-RoPE vectors be `q` and `k`, and let `R_p` be the
RoPE rotation at position `p`. If we replace the projections so that
`q' = A q` and `k' = A^{-T} k`, then the score at positions `p,s` becomes

\[
(R_p A q)^T(R_s A^{-T}k)
= q^T A^T R_p^T R_s A^{-T} k.
\]

The score is unchanged at every pair of positions only if
`A^T R_Δ A^{-T} = R_Δ` for every relative rotation `R_Δ`, equivalently `A^T`
commutes with the RoPE family. For Llama-2’s split-half RoPE with distinct
frequencies, a permitted head-local map is block diagonal over rotary pairs,
with each 2×2 block of the form `s R(θ)` for nonzero `s`. It may not mix
different frequency pairs arbitrarily. In grouped-query attention, Q heads
sharing a KV head must share the corresponding Q/K map. The implementation
`rope_commuting_qk_gauge` enforces this form and its CPU test applies the real
causal RoPE attention computation.

The family is broader than the energy-balancing diagonal gauge already used
in `nox/gauge.py`. A noncommuting head rotation is not an exact symmetry. Any
finite-precision error from storing transformed weights in FP16 is measured
separately from the FP32 identity.

For the Stage A search, pair scales balance the aggregate query/key row
energies for each shared KV pair. Pair angles diagonalize the summed 2×2
query/key row covariance for that frequency pair. These are closed-form
coordinate choices inside the exact RoPE commutant; the angle and scale values
are folded into the compressed factors and add no deployed scalars.

### V/O under head boundaries and GQA

For each KV head, let `v'_h = B_h v_h`. Apply `B_h^{-1}` to the columns of
`o_proj` that consume this value head. When a KV head feeds multiple query
heads, use the same `B_h` for the value and its matching inverse for every
query-head output slice. Attention probabilities do not change, and the
weighted value sum followed by `o_proj` is unchanged. This does not permit
mixing unrelated heads, because their attention weights differ. The helper
`value_output_gauge` uses small independent 2×2 invertible factors within
each head and is tested with GQA.
Its scales balance V-row and O-column energies, and its angles diagonalize the
sum of their pairwise 2×2 covariances. Because no RoPE acts on V, these
head-local pair rotations are exact regardless of frequency pair.

### SwiGLU neuron symmetries

With the Llama linear convention `G,U ∈ R^{m×d}`, `D ∈ R^{d×m}`,

\[
f(x) = D\left[\operatorname{SiLU}(Gx)\odot(Ux)\right].
\]

For any neuron permutation matrix `P`,

\[
G'=PG,\quad U'=PU,\quad D'=DP^T
\]

preserves `f`. For any invertible diagonal `S`,

\[
U'=SU,\quad D'=DS^{-1}
\]

also preserves `f`, because the scaling enters linearly after the SiLU gate.
This is why reciprocal Up/Down balancing is exact. In contrast, replacing
`G` with `S G`, rotating Gate neurons, or mixing Gate and Up before SiLU is
not generally exact: `SiLU` is elementwise and is not equivariant to those
maps. A new Gate rotation is an approximation unless the complete function is
proved invariant.

### Residual-stream coordinates and RMSNorm

RMSNorm divides by a scalar derived from the input RMS and then multiplies by
a learned diagonal gain. It is equivariant to a signed permutation `P` if the
gain vector is permuted with the coordinates. A whole-model signed permutation
is exact when it is folded consistently into embeddings, all projection
inputs/outputs, every RMSNorm gain, residual branches, and the language-model
head. The helper `signed_permutation_residual_gauge` handles the local weight
transform; a complete-model compiler must apply one consistent map across
layers.

An arbitrary orthogonal residual rotation preserves the RMS denominator but
turns the learned diagonal RMSNorm gain into a generally dense matrix. Keeping
that matrix adds up to `d²` values per norm and changes the deployed
architecture. It is not a free gauge. Rotations are exact only inside
subspaces on which every relevant RMSNorm gain is constant, or if the extra
normalization matrix is stored and charged to the parameter budget. Residual
identity paths also require both branches to use the same basis.

All circuit factors used in a search must either be fused into the saved model
factors, or be counted as auxiliary coefficients. Reindexing a weight tensor
does not itself add a stored floating-point parameter; a separately stored
dense transform does.

## Schmidt-spectrum measurement

For a matrix `W[d_out,d_in]`, a two-site candidate reshapes it into four
physical indices and selects a bipartition `π`. Its operator-Schmidt matrix
is

\[
M_\pi = \operatorname{reshape}(\operatorname{permute}(W,\pi)).
\]

The rank-`r` truncation keeps the top singular triplets of `M_π` and stores
`r(m+n)` coefficients for its two factors. Its relative Frobenius residual is

\[
\epsilon_r^2 = \frac{\sum_{i>r}\sigma_i^2}{\sum_i\sigma_i^2}.
\]

For four-site NEC tensorization, the three cut unfoldings pair each output
mode with its input mode, matching the reshape in `nox/compiler.py`; each
alternative site order permutes these exact coordinates before taking its
three unfolding spectra. The matrices represent the same dense operator, but
their Schmidt spectra can differ because the index tensorization changes.

`nox/spectra.py` uses exact FP64 Gram eigenvalues when one cut dimension is at
most 256. The Kaggle diagnostic uses a randomized range SVD for larger
matrices, with a 3,072-value cap, 64 oversampling directions, two power
iterations, and an independent seed repeat for the direct matrix and original
NEC middle cut. It records captured energy, unresolved energy, and repeat-seed
capture differences. The TT rank profile uses each unfolding's exact
Frobenius norm, so energy omitted by the randomized spectrum remains in the
estimated tail instead of being renormalized away. If the sketch does not
resolve a residual target, the result is only a rank lower bound and is
excluded from the Stage A improvement gate. When the target is resolved, the
TT error remains an estimate based on randomized Ritz values, not a confidence
interval or global rank optimum.

The benchmark must compare at equal independent coefficient counts:

- Ordinary truncated SVD of the original matrix.
- Activation-weighted SVD of the matrix under calibration covariance.
- Tensor-network truncation under the original NEC tensorization and at least
  two alternative dimension-compatible site orders.
- Tensor-network truncation after each legal, fused function-preserving
  transform that passes exactness checks.

Weight reconstruction error and complete-module functional error are separate
metrics. Lower operator entropy alone is not a success criterion.

## Function-aware SwiGLU candidate

For calibration input columns `X`, define

\[
H=\operatorname{SiLU}(G X)\odot(U X),\qquad Y = D H.
\]

The current training-free candidate uses activation-weighted SVD factors for
Gate and Up under `|| (W - W_hat) X ||_F²`, then solves a rank-constrained
module-output readout for Down. Given fixed approximations `G_hat,U_hat`, set
`H_hat = SiLU(G_hat X) ⊙ (U_hat X)` and solve

\[
\min_{\operatorname{rank}(D)\le r}
\frac{1}{N}\|Y-DH_{\rm hat}\|_F^2 + \lambda\|D\|_F^2.
\]

Let `H_hat/√N = U_h S_h V_h^T` and
`T=(Y/√N)V_h`. Completing the square converts the rank-`r` problem to a
truncated SVD of

\[
T\operatorname{diag}\left(\frac{s_j}{\sqrt{s_j^2+\lambda}}\right).
\]

Its factors are mapped back through
`diag(1/sqrt(s_j²+λ)) U_h^T`. The implementation is
`fit_functional_down_readout` in `nox/function_aware_tn.py`. It is an exact
closed-form readout solution for fixed Gate/Up factors; it does not claim a
global optimum over all three matrices. The full candidate still uses the
complete SwiGLU output as its construction and selection metric, has no dense
residual, has no post-compression update, and stores only its low-rank
factors.

Its scalar count is

\[
N_{\rm candidate} = r_G(m+d)+r_U(m+d)+r_D(m+d),
\]

with every factor entry counted once. The prescribed dense coefficient
retentions are 75%, 50%, and 25% of `3md`; the equal-rank controls use the
largest integer rank that does not exceed each budget. If ranks are allocated
unequally, they must still satisfy the same total coefficient cap and be
chosen on the internal selection split, not on held-out activations.

The corresponding MoDeGPT-inspired SwiGLU control selects shared Gate/Up
neurons by a diagonal proxy for activation ridge leverage, keeps matching
Down columns, then fits the Down readout to dense module outputs. The diagonal
proxy uses product-activation energy; it is not the full covariance leverage
score in MoDeGPT. This is an explicit adaptation to three-operator SwiGLU, not
a reproduction of the paper’s full Transformer-module rules. No comparable
MoDeGPT attention baseline is claimed; the attention controls instead include
ordinary and activation-weighted projection SVD plus complete-module output
readouts.

## Gates and evaluation separation

The Stage A provisional gate is a reduction of at least 20% in required TN
coefficients at a matched residual-energy target on at least two of layers 0,
15, and 31, with transform coefficients included and no material regression
on the third. A positive spectrum result must still pass held-out module
outputs.

Stage B uses distinct fitting, internal selection, and held-out activation
sets. It reports calibration and held-out MLP error, full residual-block
error, output cosine, norm ratio, exact scalar count, and PPL/prefix behavior
only for candidates that pass the local-module gate. Post-compression gradient
updates, distillation, teacher inference, and dense correction sidecars are
excluded.
