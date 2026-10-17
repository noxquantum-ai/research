# Joint SwiGLU representation and layer-0 diagnostic plan

Status: CPU implementation and synthetic CUDA validation are complete. No new
Llama weights were compiled, no full-model inference was run, and no real-model
result for this candidate exists. The A10 remains running; only a small
synthetic unit test was run there. The real-model diagnostic below is a proposal
and has not been executed.

## Evidence and diagnosis

The saved layer-0 high-fidelity NEC checkpoint has 0.36374 relative full-block
output error on the matched 4,096-token validation input. Dense restoration
ablations in `experiment_artifacts/diagnostic_36h/operator_schmidt/layer0/REPORT.md`
show that restoring Gate, Up, and Down together reduces that error to 0.18759;
restoring all attention projections reduces it only to 0.34305. No single MLP
projection restoration explains most of the gain: restoring Gate, Up, or Down
alone yields 0.35553, 0.34916, and 0.36415 respectively. Restoring Gate+Up gives
0.34402. The previously tested independent Gate+Up operator-Schmidt candidate
gives 0.34828, only a 4.25% relative reduction from the saved block error.

These controlled restorations point to the Gate/Up/Down composition as the
main layer-0 defect, but do not yet distinguish a bad Gate approximation from
Up, Down, compensating errors, or a representation that fails to preserve
their joint structure. Projection-wise reconstruction error cannot answer
that. The next diagnostic therefore compares all eight dense/compressed
projection combinations on identical inputs and reports the exact nonlinear
product and Down contributions, including their alignment.

### What the existing construction already tries

`nox/block_compiler.py::ridge_compensated_up_target` computes an elementwise
ridge target for Up after Gate has been approximated. For intermediate channel
`j`, it minimizes a scalar objective of the form

\[
 w_j\bigl(\operatorname{SiLU}(\hat g_j)u_j-
 \operatorname{SiLU}(g_j^{\rm dense})u_j^{\rm dense}\bigr)^2
 +\lambda(u_j-u_j^{\rm dense})^2,
\qquad w_j\approx\|D_{:,j}\|_2^2.
\]

This is a useful finite closed-form preconditioning of the elementwise product.
It is not a jointly fitted three-operator representation: the Down metric is
diagonal, off-diagonal coupling in `D.T @ D` is omitted, Up is subsequently
approximated through its existing per-matrix compressor, and the exact
held-out output decomposition is not its objective. The experiment metadata
available in the repository does not contain a matched held-out result for
`mlp_product_aware`; the recorded PEES manifests identify the MLP objective as
`independent` and `mlp_product_aware` as null. Therefore this is an unmeasured
method in the tracked results, not a method we can honestly call a measured
failure.

`tests/test_trajectory_aware_compiler.py` checks scalar-objective minimization
and routes the compressed contexts/budgets to the expected projections; it is
unit coverage, not a held-out model-quality experiment.

The trajectory flags in `compile_transformer_block` are also distinct from a
joint Gate/Up/Down representation. `trajectory_mode="mlp"` and `"all"` make
Down's initial fit use the compressed SwiGLU activation while keeping its
coefficient budgets fixed. `"all"` also conditions the attention output fit on
compressed Q/K/V attention; `"product"` conditions that attention fit but does
not set `compile_mlp_block(... conditional_context=True)`. The flags can
therefore alter the inputs seen by a per-matrix fit without sharing a tensor
factor across the three MLP operators. No matched held-out record for these
trajectory modes was found in the experiment artifacts; again, lack of a
tracked test is not proof of failure. The historical `Hadamard co-healing`
implementation was later reverted (`abc1d44` / `d80d6ec`) and is not being
reintroduced as a differently named sidecar.

## Exact diagnostic quantities

For dense Gate/Up/Down matrices `G`, `U`, `D`, a fixed MLP input matrix `X`,

\[
g=GX,\quad u=UX,\quad h=\operatorname{SiLU}(g)\odot u,\quad y=Dh.
\]

For candidate matrices and states with hats, the exact output error is

\[
\hat D\hat h-Dh
=D(\hat h-h)+(\hat D-D)\hat h.
\]

The two output-space terms are reported separately, along with their cosine and
cross-energy fraction

\[
\rho=\frac{2\langle e_{\rm product},e_{\rm down}\rangle}
{\|e_{\rm product}\|_2^2+\|e_{\rm down}\|_2^2}.
\]

Positive `rho` means reinforcement; negative `rho` means cancellation. A
separate exact expansion isolates Gate, Up, and their bilinear interaction:

\[
\hat h-h=
 [\operatorname{SiLU}(\hat g)-\operatorname{SiLU}(g)]\odot u
 +\operatorname{SiLU}(g)\odot(\hat u-u)
 +[\operatorname{SiLU}(\hat g)-\operatorname{SiLU}(g)]\odot(\hat u-u).
\]

The harness evaluates all eight dense/compressed choices for `(G,U,D)` on both
the dense and compressed transformer trajectories. This distinguishes (a)
operator error on a shared input from (b) error inherited through the
compressed attention/residual path. It also records held-out preactivation,
SwiGLU product, complete MLP output, active transform, and full-block output
errors. These remain representation diagnostics, not downstream capability
measurements.

## Selected candidate: shared-neuron activation TTN

The opt-in candidate replaces the three independent MLP artifacts with a
single three-arm factorization tied to the actual shared intermediate-neuron
index. It stores one shared basis `Z` and three leaves:

\[
\hat G=ZG_r,\qquad \hat U=ZU_r,\qquad \hat D=D_rZ^T,
\]

and executes the complete MLP without reconstructing these dense matrices:

\[
\hat y=D_r Z^T\left[
\operatorname{SiLU}(Z(G_r x))\odot Z(U_r x)
\right].
\]

This is a shared-bond tree/tensor-network contraction over the physical SwiGLU
neuron axis: Gate and Up meet at the nonlinearity, and Down contracts through
the same learned subspace. It does not store a dense or quantized residual.

For calibration inputs `X`, form the dense state stack

\[
S=[GX,\ UX,\ \operatorname{SiLU}(GX)\odot UX].
\]

Let `Z_r` contain the leading `r` left singular vectors of `S` and
`P_r=Z_r Z_r^T`. By the Eckart–Young theorem, the exact top-r subspace is the
rank-`r` orthogonal projector minimizing

\[
\|(I-P_r)GX\|_F^2+
\|(I-P_r)UX\|_F^2+
\|(I-P_r)(\operatorname{SiLU}(GX)\odot UX)\|_F^2.
\]

Factors are then `G_r=Z_r^T G`, `U_r=Z_r^T U`, and `D_r=D Z_r`. This is an
algebraic activation-subspace initializer; for large stacks the implementation
uses a seeded randomized truncated SVD to approximate this subspace. It is not
a proof that the nonlinear MLP output is optimal. The held-out output metric
is decisive; there are no gradient steps, DMRG sweeps, validation-informed
choices, or post-finalization updates.

The floating coefficient count is

\[
N_{\rm joint}=r(m+3d),
\]

where `m=11,008` and `d=4,096` for Llama-2-7B. The current saved layer-0 Gate,
Up, and Down artifacts contain 7,505,964 floating tensor scalars. Matching
that budget gives `r=322`, 7,501,312 stored floating scalars, versus
135,266,304 dense MLP weights (5.55%). Integer sparse indices are not counted
as floating coefficients, but are included in the serialized-byte comparison.

CPU-side schema accounting gives 30,370,779 bytes for separately serialized
saved MLP artifacts and 15,005,277 bytes for the candidate artifact schema at
FP16 factor storage (0.494x). The candidate estimate uses empty tensors with
the exact candidate shapes/dtypes and exact metadata schema; the actual GPU
run will serialize the fitted candidate and record its bytes. This is an
MLP-artifact estimate only, not a full layer or model storage result. The
current implementation and direct apply live in `nox/joint_swiglu.py`; it is
not yet wired into the default NEC compiler or checkpoint loader pending
evidence.

### Main mathematical risks

- A single shared subspace may be too restrictive: Gate and Up can require
  different directions, while the product can occupy directions absent from
  either preactivation.
- The unweighted state SVD minimizes internal activation residuals, not the
  full output loss under `D.T @ D`; error cancellation or amplification may
  cause poor output fidelity even when the internal subspace fit is good.
- The SiLU and Hadamard product are nonlinear, so the linear rank-r projection
  is a surrogate and is not guaranteed to preserve the product.
- FP16 factors and a rank chosen solely by count may be insufficient; the
  fixed-budget held-out result will show whether this topology is competitive.
- The factorized apply is executable, but no speed claim is made. Kernel
  efficiency and actual end-to-end runtime need a separate benchmark.

## Local validation completed

- `python3 -m unittest tests/test_joint_swiglu.py -v`: 9 CPU tests pass and
  the CUDA-gated test skips locally.
- Full available test set excluding `tests/test_wikitext_source.py`: 92 tests
  run, 88 pass and 4 CUDA-gated tests skip on the local CPU-only environment.
- Unfiltered discovery cannot import `tests/test_wikitext_source.py` because
  local Python lacks `pyarrow`; the
  unavailable module is excluded from the 92-test available suite. No
  dependency was installed or environment changed.
- A synthetic CUDA round-trip/apply test ran on the already-running A10:
  10/10 joint-SwiGLU tests passed in 0.307 seconds. It used only small random
  matrices and did not access model weights or checkpoints. The A10 remained
  running and idle afterward.

The isolated CUDA test command was:

```bash
cd /tmp
PYTHONPATH=/tmp/nox_joint_swiglu_cuda \
  /home/shadeform/nec-crosslayer-research/.venv/bin/python \
  -m unittest discover -s /tmp/nox_joint_swiglu_cuda/tests -v
```

## Proposed real-model GPU experiment — not yet run

This is the one-layer held-out diagnostic requested for review. It reuses the
saved layer-0 checkpoint and its 4,096-token train/validation activations plus
the pinned dense layer-0 weights. It does not compile a layer, write a new
checkpoint, evaluate perplexity/MMLU, or touch layers 1–31.

1. Confirm the existing A10 is reachable, idle, and has at least 14 GiB free;
   do not start, stop, or provision an instance. Check the checkout and Python
   environment without changing or cleaning its existing diagnostic outputs.
2. Transfer only `nox/joint_swiglu.py` and
   `scripts/diagnose_swiglu_joint.py` into the existing checkout after checking
   for local edits. Run one diagnostic with 4,096 calibration and 4,096 held-out
   tokens, sequence length 128, and the pinned Llama-2-7B layer-0 dense weights.
3. Save JSON under
   `experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/`. Copy that output
   and stdout back to local storage immediately; do not rely on the A10 as the
   only copy.
4. Stop after this run. Do not scale to a prefix, full model, PPL, or MMLU until
   reviewing the results together.

After confirming that the destination files do not exist or contain user
changes, transfer the two new source files into the existing main checkout:

```bash
scp -i A10-mvc1exyr_key.pem nox/joint_swiglu.py \
  shadeform@129.213.125.12:/home/shadeform/nec-crosslayer-research/nox/joint_swiglu.py
scp -i A10-mvc1exyr_key.pem scripts/diagnose_swiglu_joint.py \
  shadeform@129.213.125.12:/home/shadeform/nec-crosslayer-research/scripts/diagnose_swiglu_joint.py
```

From the remote repository root, assuming the prior checkpoint and
dense-weight cache paths remain present:

```bash
mkdir -p experiment_artifacts/diagnostic_36h/joint_swiglu/layer0
test ! -e experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/layer0_joint_swiglu.json
test ! -e experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/run.log
set -o pipefail
.venv/bin/python -u scripts/diagnose_swiglu_joint.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache \
  --layer 0 --device cuda --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --output experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/layer0_joint_swiglu.json \
  2>&1 | tee experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/run.log
```

After the run, copy both files to local storage outside the A10 before doing
anything to the remote checkout or instance:

```bash
mkdir -p /home/fluxx/Workspace/research_backups/nox_joint_swiglu_layer0_20261017_run01
scp -i A10-mvc1exyr_key.pem \
  shadeform@129.213.125.12:/home/shadeform/nec-crosslayer-research/experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/layer0_joint_swiglu.json \
  /home/fluxx/Workspace/research_backups/nox_joint_swiglu_layer0_20261017_run01/
scp -i A10-mvc1exyr_key.pem \
  shadeform@129.213.125.12:/home/shadeform/nec-crosslayer-research/experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/run.log \
  /home/fluxx/Workspace/research_backups/nox_joint_swiglu_layer0_20261017_run01/
sha256sum /home/fluxx/Workspace/research_backups/nox_joint_swiglu_layer0_20261017_run01/*
```

The planned JSON records all eight Gate/Up/Down combinations on both dense and
compressed trajectories; product-versus-Down error magnitude and interference;
saved NEC and candidate MLP/active-transform/full-block errors; shared rank and
coefficient count; exact serialized artifact bytes; runtime; and peak CUDA
allocation. The candidate is built once from train states at the saved MLP
floating-coefficient budget and tested only on held-out states.

Estimated run time is 2–5 minutes, not yet measured for this harness. Memory
planning is conservative at up to 12 GiB; the A10 has 23,028 MiB total. Abort
before execution if less than 14 GiB is free or the process would compete with
another user. The previous small candidate screen peaked at 7.83 GiB, but this
run also keeps dense/compressed layer weights and both activation splits
resident, so that earlier peak is not a direct estimate.

The predeclared advancement criterion is: at the matched MLP coefficient
budget, the candidate must reduce held-out complete-MLP function relative
error by at least 5% relative versus saved NEC on the compressed trajectory,
and must not worsen held-out active-transform or full-block error. The
serialized MLP artifact must be smaller than the saved three-artifact payload.
If not, do not run a prefix compile; use the eight-way error/interference
results to decide whether the limitation is Gate, Up, Down, their cancellation,
or the single shared-subspace topology itself.
