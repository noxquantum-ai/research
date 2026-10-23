# NEC accuracy redesign: coupled construction under fixed budgets

The acceptance gate is simultaneous full-32-layer held-out PPL <= 10,
actual complete-package storage compression >= 5x, and structural coefficient
reduction >= 2x. Dense PPL (~4.86) remains the preferred quality level. A short
prefix or a projected byte count cannot satisfy this gate. No nonlinear
derivative optimization, autograd, fine-tuning, or post-export healing is used.

## Causes being addressed

1. Core ALS minimizes unfolding reconstruction. The nonlinear composition
   `D[SiLU(Gx) * Ux]` can worsen even when its individual matrix fits improve.
2. The old calibration has 4096 tokens in 128-token windows. Evaluation uses
   2048-token windows. Representative long-context construction data is needed
   before treating small-sample reconstruction as a generalization result.
3. Four-bit cores leave unused bytes at the 2x structural cap. Five-bit factors
   are mathematically compatible with the 5x package limit; code previously
   prohibited them. Precision and structural rank must be allocated jointly.

The dense cache itself passes tested layer-transition parity checks at relative
errors around 1e-6 or lower. The actual source RMSNorm epsilon is 1e-5.

## Coupled core construction

With all other stored coefficients fixed, the map from one MPO core to its
represented weight is linear. For Up, the exact linear response of the full
MLP to a core increment is

    A_U(delta) = D_hat [SiLU(G_hat x) * (delta_U x)].

For Down it is

    A_D(delta) = delta_D [SiLU(G_hat x) * (U_hat x)].

The compiler solves prior-anchored normal equations with a matrix-free
preconditioned conjugate-gradient linear solver:

    (A* A + lambda I) delta = A* (teacher_output - current_output).

For Gate, the candidate solve reconstructs clean teacher preactivations with
the linear map `delta_G x`. There is no SiLU derivative or Gauss-Newton Gate
step. Every proposed core is packed before scoring the complete nonlinear
MLP. Acceptance requires improvements on both fit and calibration-guard
samples. Those guard samples are excluded from the new core normal equations;
the warm-start codec may already have used them, so they are not an independent
external holdout. Separate validation activations and full-model held-out PPL
must confirm any claimed improvement.

The MLP target includes the attention residual-stream defect, so its output
targets the clean whole-block state rather than leaving an uncorrected
attention difference outside the objective. Core shapes, ranks, factor widths,
branch tensors and tensor payload sizes are invariant during this stage.

An initial prototype used an analytic Gate linearization. That derivative-based
path was removed before scaling because it does not satisfy the strict
zero-gradient construction requirement. It is not an accepted final method.

## Data and evaluation protocol

`capture_packed_nec_activations.py` uses pre-hooks on the original Hugging Face
decoder. It samples 32768 training tokens and 8192 separate validation tokens
in 2048-token windows, records exact sampling offsets and token hashes, and
avoids materializing the vocabulary head during capture. Calibration captures
are rebuildable scratch data; committed manifests identify their inputs.

`--initialization-output` saves matched controls before coupled construction.
They are construction artifacts outside the deployable package. Paired held-out
block errors and full-model PPL compare the same rank, precision and bytes.

## Relevant primary research

- [MoDeGPT](https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb7214d2fdfd84165b08539d59c92e07-Abstract-Conference.html)
  supports module-level decomposition without backpropagation. Its reported
  25-30% structural compression does not establish NEC's 2x target.
- [Swift-SVD](https://arxiv.org/abs/2604.01609) derives an activation-output
  covariance solution and separates local compressibility from global
  importance. Per-matrix optimality does not establish nonlinear module quality.
- [Per-Matrix Optimality Is Not Enough](https://arxiv.org/abs/2609.15838)
  explicitly studies matrix, block and full-model error scopes. Its training
  stages are outside this experiment's zero-gradient constraint.
- [CompactifAI](https://arxiv.org/abs/2401.14109) establishes the tensor-network
  compression precedent; its headline ratios are not evidence that NEC can
  reproduce those results under zero gradient-based healing.

The NEC hypothesis is the combination of packed MPO core construction,
exact nonlinear acceptance and simultaneous coefficient/serialized-byte limits.
No general novelty or successful-quality claim is made before validation.
