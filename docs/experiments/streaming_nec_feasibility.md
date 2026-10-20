# Streaming NEC calibration: mathematical and implementation audit

Date: 2026-10-20

Scope: the four-site MPO High-Fidelity NEC path with independent dense
calibration activations, zero post-compression healing, ranks
`[1,64,128,64,1]`, bridge rank 128, sparse budget 524,288, two DMRG sweeps,
and simulated Q4 components. The reference full-matrix compiler remains the
default. No paid GPU run was started and the production RunPod memory gate was
not relaxed.

## Exact projection reformulation

For calibration matrices (X\in\mathbb R^{d_i\times N}) and
(Y\in\mathbb R^{d_o\times N}), accumulate by complete token chunks

\[
 C=XX^\top=\sum_bX_bX_b^\top,\qquad
 H=YX^\top=\sum_bY_bX_b^\top .
\]

The channel RMS used by the sparse shield is exactly
(c_j=\sqrt{C_{jj}/N+10^{-8}}). If
(X=U\Sigma V^\top), the bridge quantity used by the existing compiler is
recoverable without (V):

\[
 (Y-SX)V=(H-SC)U\Sigma^{-1}.
\]

After the quantized sparse-plus-bridge component (S_t) is fixed and
(D=\operatorname{diag}((c+10^{-6})^\alpha)), define

\[
 C_c=D^{-1}CD^{-1},\qquad H_c=(H-S_tC)D^{-1}.
\]

For (C_c=U_c\Lambda U_c^\top), the implementation constructs

\[
 X_r=U_c\Lambda^{1/2},\qquad
 Y_r=H_cU_c\Lambda^{-1/2}.
\]

Consequently (X_rX_r^\top=C_c) and (Y_rX_r^\top=H_c). Every active NEC
one-site update uses calibration data only through a local normal matrix and
right-hand side. Those are invariant to an orthogonal change of sample
coordinates, so replacing the SVD-derived reduced pair by ((X_r,Y_r)) gives
the same DMRG normal equations in exact arithmetic. The optional diagonal
weight prior is also exact: it adds

\[
 \operatorname{diag}(s^2)\quad\text{to }C_c,
 \qquad W_c\operatorname{diag}(s^2)\quad\text{to }H_c,
 \quad s={c\over d}\sqrt{N\lambda}.
\]

This is one global fit over all tokens, not minibatch fitting and not model
averaging. (T=YY^\top) is not required to solve the active compiler, although
the accumulator can retain it for exact loss accounting.

## Numerical qualification

The covariance identity is exact algebraically but a Gram matrix squares the
condition number. `nox/calibration.py` therefore makes the accumulation dtype
explicit, symmetrizes covariance before `eigh`, and preserves the full-matrix
implementation as the reference. It does not apply the conventional
`eps * max(N,d)` singular-value cutoff: that would silently discard modes kept
by the reference compiler and reaches 0.0078125 for FP32 and N=65,536. It
instead retains every expected positive mode, records how many lie below that
cutoff, and fails closed if Gram roundoff loses a positive mode. Float64 is the
CPU reference mode. Float32 is faster on commodity GPUs but is not certified
for real Llama activation spectra until compared against the SVD path on a real
projection. Streaming QR/TSQR is the preferred fallback if that comparison
shows corruption of small but relevant directions; it was analyzed but not
implemented in this change.

For a fixed linear dense target (Y=WX), (H=WC). The independent block path
therefore stores four input covariances (normalized q/k/v input, dense attention
context, normalized gate/up input, and dense SwiGLU/down input) and constructs
one cross-covariance at a time. q/k/v and gate/up reuse their shared Gram
factorization. This reuse does not alter the fit.

## Sequence-preserving forward streaming

Calibration contains 512 independent windows of 128 tokens at (N=65,536).
`forward_transformer_block_chunked` and
`collect_independent_block_calibration` process an integer number of complete
windows per chunk. Each chunk calls the unchanged Llama-2 block operations:
RMSNorm with model epsilon, q/k/v projections, RoPE positions 0--127, causal
attention confined to each window, residual addition, the second RMSNorm,
SwiGLU, and the down projection. No attention crosses a window boundary and the
sample selection/order is unchanged.

The statistics route is deliberately rejected for trajectory-aware,
compressed-prefix, product-aware, non-MPO4, gauge, PEES, and defect-bridge
experiments. Their targets cannot all be inferred from the four independent
dense covariances.

## Complete memory/compute audit

All byte figures below are FP32 tensor payloads and exclude library workspaces,
allocator fragmentation, model shards, and temporary copies.

| Operation | Full path at 65,536 tokens | Streaming path | Exact? | Dominant work |
|---|---:|---:|---|---:|
| 4096-wide input activation | 1.00 GiB | window chunk | yes | forward GEMMs |
| 11008-wide SwiGLU activation | 2.69 GiB | window chunk | yes | forward GEMMs |
| `svd(X)` right vectors, 4096 input | up to 1.00 GiB | absent | exact arithmetic | (O(Nd_i^2)) vs covariance `eigh` |
| `svd(X)` right vectors, 11008 input | up to 2.69 GiB | absent | exact arithmetic | same |
| (C), 4096 input | absent as an explicit persistent tensor | 64 MiB | yes | (O(Nd_i^2)) accumulation |
| (H), 11008-by-4096 gate/up | implicit | 172 MiB, one projection at a time | yes | (O(d_od_i^2)) when (H=WC) |
| (C), 11008 down input | absent | 462.25 MiB | yes | (O(Nd_i^2)) |
| sparse importance/support | weight-sized (64 or 172 MiB) | unchanged | yes | (O(d_od_i)) |
| bridge target | (d_o\times\min(N,d_i)) intermediates | (H U\Sigma^{-1}) | exact arithmetic | eig/SVD plus GEMM |
| AWT conditioning | token matrix plus weight | covariance congruence plus weight | yes | (O(d_i^2)) |
| TT-SVD warm start | weight-sized unfoldings | unchanged | yes | weight-only SVDs |
| interior DMRG design chunk | up to about 1.0 GiB in High-Fidelity | unchanged | yes | repeated contractions and 6 CG products |
| Q4 simulation | component-sized | unchanged | yes | linear in coefficients |
| artifact reconstruction | dense weight required by current HF injection | unchanged | yes | (O(d_od_i)) storage/work |
| validation propagation | full token matrices | complete-window chunks | yes | forward GEMMs |

The active contractions, covariance accumulation, Gram eigendecomposition,
TT-SVD, and DMRG solve all remain on the selected CUDA device. Complete-window
inputs are transferred to the GPU once per chunk; dense block outputs are moved
to the configured output device once so they do not accumulate in VRAM. CPU
transfers inside the compiler are limited to final artifact serialization and
metadata. The interior DMRG systems already use bounded design chunks and
matrix-free six-step CG instead of allocating 20,480--65,536-square Hessians.
The qualification path does not enable TF32, reduced precision, compilation,
or experimental fused kernels because those would complicate the numerical
parity test. One avoidable synchronization was removed: local-design Frobenius
norms now accumulate as FP64 scalars on-device rather than calling `.item()`
for every chunk.

The four persistent block-input covariances total about 654 MiB in FP32. The
largest single High-Fidelity local design chunk is approximately 1 GiB for the
interior sites (`D=65,536`, 4,096 design rows). The down-projection covariance,
two cached 11008-dimensional eigenvector bases, conditioned covariance,
reduced input/target, layer weights, and DMRG design temporaries plausibly fit
well below 48 GiB, but eigensolver workspace is backend-dependent. This is an
engineering estimate, not a measured GPU peak.

Memory removal does not make 65,536 tokens cheap. Covariance accumulation still
costs (O(Nd_i^2)), `eigh` costs (O(d_i^3)), and DMRG cost scales with
(r_{cov}=\operatorname{rank}(X)) until it saturates at (d_i). In particular,
moving from 256 to full-rank down-projection calibration can increase the
number of reduced DMRG modes from 256 to 11,008 (43 times), even though local
design matrices are chunked. At 16,384 tokens all Llama-2-7B projection input
widths can already reach full rank; 65,536 increases linguistic coverage and
covariance accumulation work but not the maximum DMRG rank.

## Verified local results

`tests/test_streaming_calibration.py` covers chunk accumulation, optional
(YY^\top), unequal chunks, rank-deficient inputs, FP32 no-silent-truncation
behavior, a complete sparse/bridge/AWT/two-sweep fit, diagonal prior,
sequence-aligned Transformer forwards, and a complete seven-projection block
comparison with two sweeps and Q4 diagnostics for every stored component. All
11 tests pass. In the current local environment, 157 runnable repository tests
passed and four CUDA tests were skipped; complete discovery had one import error
because the local interpreter lacks the declared `pyarrow` dependency.
These are small-matrix regression tests; they do not certify real-sized
High-Fidelity projections. `qualify_streaming.sh` is the required production-
shape qualification before changing the official backend.

The CPU synthetic benchmark (`torch 2.11.0+cpu`, 64-by-64 projection, two
sweeps, float64 accumulated statistics) produced:

| Tokens | Relative streamed/reference weight difference | Full elapsed | Streamed elapsed | Full RSS delta | Streamed RSS delta |
|---:|---:|---:|---:|---:|---:|
| 256 | 2.65e-6 | 0.125 s | 0.088 s | 18.4 MiB | 2.73 MiB |
| 1,024 | 3.39e-6 | 0.035 s | 0.041 s | 1.09 MiB | sampling floor |
| 4,096 | 4.57e-6 | 0.130 s | 0.039 s | 8.18 MiB | sampling floor |

The timings are affected by warm-up and the RSS values by the CPU allocator;
they establish small-case parity, not Llama runtime or a GPU capacity claim.
Artifact ranks and payload sizes were identical in every case, and calibration
and held-out errors agreed to the displayed precision.

Reproduce locally:

```bash
python3 -m unittest tests.test_streaming_calibration -v
python3 benchmarks/benchmark_streaming_nec.py --tokens 256,1024,4096
```

The compilation script exposes the new path without changing its default:

```bash
python3 scripts/compile_high_fidelity.py \
  --output /workspace/nec-streaming-smoke \
  --start-layer 0 --end-layer 0 \
  --n-calib 4096 --n-validation 1024 --seq-len 128 \
  --calibration-sampling uniform --validation-sampling uniform \
  --calibration-backend streaming_statistics \
  --calibration-windows-per-chunk 4 --statistics-dtype float64 \
  --calibration-activation-mode dense --construction-mode independent \
  --representation mpo4 --mpo-sweeps 2 --mpo-alpha-awt 0.5 \
  --attention-gauge none --mlp-gauge none --mlp-objective independent \
  --budget-allocation uniform --defect-bridge-mode none --pees-rank 0 \
  --device cuda:0
```

The production `runpod_driver.py` always passes the backend recorded in
`run_config.json`; that configuration explicitly remains `full_matrix` until
real qualification succeeds. Qualification is deliberately staged. Each
command runs one stage and stops:

```bash
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-reference
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-fp32
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-compare
```

These stages use the real layer-0 SwiGLU activation entering `down_proj`, with
the reference SVD and FP32 statistics on identical 4,096/1,024-token
calibration/validation samples. The comparison checks the reconstructed
weight, ranks, coefficient count, two-sweep Q4 diagnostics, calibration and
held-out projection errors, wall time, CUDA peaks, and RSS. FP64 is a separate
`down-fp64` stage; the runner refuses it unless the FP32 comparison failed. It has no
automatic timeout by default; `NEC_QUAL_TIMEOUT_SECONDS` enables an explicit
per-stage limit when a budget requires one. Unbuffered compiler output is appended to a main log, while a
60-second heartbeat records the current phase, GPU utilization and VRAM, host RAM,
disk space, load average, and elapsed time as durable JSONL. It records partial status and logs, reuses verified
completed stages only when their artifact and stage hashes, numerical-source
fingerprint, hardware, and complete qualification identity match the current
invocation. The Git commit is retained separately as provenance, so publishing
reports does not invalidate unchanged numerical artifacts. The runner never
starts a successor automatically. Terminating the
process does not terminate the RunPod pod or stop billing.
The qualification bootstrap omits the unused MMLU harness; invoking the later
production bootstrap installs and verifies it before any PPL/MMLU execution.

Only a passing `down-compare` unlocks the explicit `layer-reference` and
`layer-fp32` stages. This prevents the original three-full-layer qualification
from consuming the budget before a single real projection establishes runtime
and memory. None of these commands invokes the 160-GiB production preflight.

## Hardware decision and cheapest sound next experiment

No 24/48/80 GiB claim is yet verified because this workspace has no CUDA GPU
and only 2.6 GiB system RAM; the cached `down_proj` file is zero bytes. The
existing 160 GiB RunPod preflight threshold is therefore unchanged. The prior
duplicate model cleanup in `gpu_qualification.py` is already fixed on current
`main`.

The cheapest scientifically sound next step is an A40 48 GiB smoke profile,
not the 32-layer run:

1. Run the real layer-0 `down_proj` at 4,096 tokens through both backends and
   record factor/weight parity, calibration/held-out output error, wall time,
   CUDA allocated/reserved peaks, and host RSS.
2. Run one complete streaming layer at 4,096 tokens. Stop if projected 32-layer
   compression alone exceeds the approximately $20 budget.
3. If parity and headroom pass, repeat the down projection at 16,384 tokens;
   this reaches the maximum possible 11,008-dimensional calibration rank.
4. Only then test 65,536 tokens. It supplies more samples but does not increase
   the maximum DMRG rank beyond the 16,384-token case.

At $0.59/hour, one hour of A40 qualification costs $0.59 and the entire $20
budget buys 33.9 hours. A full run should not be authorized unless measured
one-layer time plus PPL/MMLU time fits that envelope with checkpoint/resume
headroom. If it does not, 4,096 or 8,192 tokens is the defensible smaller
profile to test next; which one is affordable cannot be selected honestly
until the real-projection and one-layer timings exist.
