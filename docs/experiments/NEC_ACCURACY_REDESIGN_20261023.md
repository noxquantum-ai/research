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

## Representation-level alternative: sparse coordinate excitations

A small MPO bond constrains operator rank in the chosen tensor unfolding.
If that unfolding has a broad spectrum, increasing fitting effort cannot
recover its omitted directions. A sparse coordinate residual can retain
full algebraic rank while using few stored values. The experimental alternative
therefore stores an actual Schmidt MPO plus sparse residual amplitudes, with
one serialized support bit per original matrix coordinate. Every amplitude,
MPO core value and learned scale counts toward the structural limit; support
masks and all metadata count toward the byte limit.

This is explicitly a tensor-network/sparsity hybrid. Sparsity and quantization
are established techniques, not a new NEC contribution by relabeling.
[SparseGPT](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)
and [Wanda](https://openreview.net/pdf?id=PxoFut3dWW) provide relevant
training-free sparsification baselines. NEC must test whether its MPO backbone
and coupled nonlinear construction improve on a matched sparse-only control.
No quantum advantage or priority claim follows from this architecture.

Support is selected from source-minus-backbone coordinate defects weighted
by construction input RMS. On the fixed support the compiler solves exact
prior-anchored linear normal equations. The coupled MLP constructor can also
solve Up and Down excitation amplitudes through the full MLP function with
the other operators fixed. Gate remains a linear preactivation reconstruction.
Candidates are actually packed before fit/guard acceptance. There are no
nonlinear derivatives, extra hidden dense weights or post-export training.

Before GPU scaling, each operator must fit both hard caps including the
one-bit support mask. Local validation error, end-to-end held-out PPL and
actual final export measurements are separate gates. A better local fit alone
does not establish the full-model targets.

## Additional function-preserving construction candidates

`functional_routing.py` treats Query and Key as a coupled bilinear operator.
With one partner fixed, score reconstruction is linear in the other's MPO
core or excitation amplitudes. A fixed teacher-probability metric centers
scores to remove the irrelevant per-query constant. The compiler solves those
linear systems without softmax derivatives, then requires packed proposals
to improve actual attention-distribution KL on fit and guard windows. Query
positions are subsampled, but each includes all causal keys in its original
2048-token context. This is a screening/construction approximation, not a
substitute for complete end-to-end evaluation.

The optional `readout_projection` Gate policy alternates two explicit set
projections instead of differentiating the nonlinear MLP: solve a regularized
linear Down readout system for a hidden-feature correction; divide through
fixed Up activations with a ridge; invert SiLU by branch-aware bisection to
obtain Gate preactivation targets. Values outside SiLU's real range project
onto its mathematical minimum. Both negative-output inverse branches are
considered, selecting the closest to the current preactivation. A linear
Gate core solve then proposes a packed candidate. Exact coupled MLP fit and
guard acceptance remain mandatory. No coefficient or inference branch is
added. These are experimental candidates until held-out model results exist.
