# Streaming NEC numerical diagnosis

Date: 2026-10-21

This note diagnoses the compiler-equivalence failure measured on the real A40
layer-0 `down_proj` qualification. It does not claim improved Llama quality and
does not authorize a larger calibration or full-model run.

## Findings

The covariance bridge equation is correct. For

\[
X=U\Sigma V^\top,\quad C=XX^\top,\quad H=YX^\top,
\]

the streaming path evaluates

\[
YV=HU\Sigma^{-1}.
\]

A deterministic FP64 control with condition number \(10^7\) measured relative
differences of \(7.98\times10^{-11}\) in the singular values and
\(5.20\times10^{-10}\) in the projected target. The left singular vectors
differed by \(6.91\times10^{-5}\) after sign alignment, as expected from the
worse conditioning of a Gram eigendecomposition, but the reconstructed target
coordinates remained accurate.

The first material instability is not an algebraic error in that identity. It
is an interaction between approximate local solves and discontinuous in-sweep
factor quantization:

1. High-Fidelity interior sites use six matrix-free conjugate-gradient steps.
   In a small forced-CG control, the maximum relative normal-equation residual
   was approximately \(2.0\times10^{-2}\).
2. The historical final sweep quantizes each solved core immediately. Every
   subsequent site therefore constructs its environment from an already
   rounded core. A small backend perturbation that crosses a Q4 bin changes the
   next regression system and is amplified sequentially.
3. With Q4 disabled, the final calibration functional disagreement was only
   \(7.81\times10^{-7}\), despite the ill-conditioned input.
4. With six CG iterations and historical in-sweep Q4, it rose to
   \(1.279\times10^{-2}\) on calibration data and \(3.168\times10^{-2}\) on an
   independent synthetic validation matrix.

These controls reproduce the qualitative A40 observation: the bridge starts
with a small discrepancy, sweep 1 remains close, and the second quantized
sweep causes the largest jump.

## Experimental stability control

`quantization_schedule="post_sweep"` is now available as an explicit numerical
control. It performs both gauge-fixed sweeps in floating point and quantizes
all four final cores once, after no later local solve can consume a rounded
environment. The historical `in_sweep` behavior remains the default.

The number of CG iterations is also explicit. Under the same deterministic
fixture:

| Control | Max normal residual | Calibration disagreement | Validation disagreement | Weight difference |
|---|---:|---:|---:|---:|
| No Q4, 6 CG | 0.769* | 0.000000781 | 0.0000192 | 0.0000204 |
| Historical in-sweep Q4, 6 CG | 0.0201 | 0.01279 | 0.03168 | 0.03341 |
| Post-sweep Q4, 6 CG | 0.0201 | 0.00902 | 0.01681 | 0.01726 |
| Post-sweep Q4, 48 CG | 0.000116 | 0.00211 | 0.00361 | 0.00388 |

\*The maximum is from an intermediate forced-CG site. Despite its large normal
residual, both backends follow almost the same unquantized trajectory. It
becomes consequential when Q4 makes that trajectory discontinuous.

The post-sweep/48-CG result passes the local 1% functional-equivalence target.
It is not yet a production fix: it changes the historical optimization and may
alter compression fidelity. In the synthetic fixture, post-sweep Q4 had a
larger calibration reconstruction error than historical in-sweep Q4, so
equivalence must not be confused with approximation quality.

## Instrumentation and revalidation

The compiler can now report, for every sweep and site, the solver type, local
width, iteration count, regularization scale and relative normal-equation
residual. Qualification identities include the quantization schedule, CG
threshold and iteration count, preventing artifacts from incompatible controls
from being compared or reused.

The staged RunPod runner adds three non-automatic controls:

```bash
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-stable-reference
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-stable-fp64
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-stable-compare
```

They compare full-matrix and FP64-statistics compilers using post-sweep Q4 and
48 CG iterations on the same 4,096/1,024 real activation sets. The comparison
now requires direct functional parity in addition to weight, scalar metric,
structure and provenance checks. It can compare FP64 without requiring an
unrelated FP32 run. No stage starts another automatically.

Reproduce the local diagnostic with:

```bash
PYTHONPATH=. python3 benchmarks/diagnose_streaming_numerics.py \
  --output results/streaming_numerical_diagnostic_cpu_20261021.json
```

## Decision boundary

Do not promote the streaming backend, increase calibration tokens, or run a
complete layer until the real A40 stable control passes. Even if compiler
equivalence is restored, the separate 0.4206 held-out relative error of the
reference `down_proj` remains unresolved. Improving that approximation is a
later model-quality experiment, not evidence for or against the streaming
mathematics.
