# NEC projection-aware operator-Schmidt redesign

## Status

CPU implementation and tests are complete. The GPU diagnostic and
compile/evaluation commands below are prepared but have **not** been run. No
GPU instance was started, contacted, or provisioned. The redesign has not yet
passed a real-model or downstream-accuracy gate.

## Diagnosis from the saved evidence

The failure is cumulative approximation error, not a checkpoint decode bug.
The saved four-layer high-fidelity cascade was evaluated against the pinned
Llama-2-7B dense reference with the corrected raw WikiText-2 protocol:

| Layers compressed | Dense PPL | NEC PPL |
|---|---:|---:|
| none | 5.292 | 5.292 |
| 0 | 5.292 | 5.546 |
| 0–1 | 5.292 | 10.151 |
| 0–2 | 5.292 | 61.652 |
| 0–3 | 5.292 | 511.624 |

The source records are under
`experiment_artifacts/diagnostic_36h/remote_records/high_fidelity_no_healing_4layer_prefix/`.
Layer 0 alone is already damaging, while the later isolated layers are much
less damaging than their cascade counterparts. That points to both initial
operator distortion and sensitivity to the corrupted prefix distribution.

The layer-15 projection experiments provide a mechanism-level clue. In the
matched calibration/validation projection-fit record, the existing MPO's
held-out relative output errors were 0.804 for V, 0.796 for O, 0.792 for Up,
and 1.022 for Down. Removing the fitted MPO core improved held-out error for
all seven tested projections, although it worsened calibration error. In the
unquantized V staged fit, validation error rose from about 0.716 after the
TT-SVD warm start to 0.779 and 0.811 after two local fitting sweeps, while
calibration error improved. This is direct evidence of a mismatch between the
existing local fitting objective and held-out operator action; adding more
sweeps is not the proposed remedy. See `projection_fit.json` and
`v_proj_stages.json` in `remote_records/corrected_raw_layer15/`.

The existing four-site MPO has a second, structural limitation: at its central
rank-128 cut, the supplied spectral audit reports roughly 80–93% discarded
squared energy in the preconditioned residual, depending on projection. A
single site ordering and three simultaneous TT cuts can therefore discard
important cross-coordinate interactions before local fitting begins. The
current calibration least-squares updates can then overfit directions seen in
the small calibration span. Better data coverage and diagonal weight priors
were already tested and did not close this gap.

The local projection re-fit checks also showed that artifact decoding matches
the saved tensors closely. The severe PPL growth is present in the decoded
weights themselves, so codec round-trip precision is not the cause.

## Selected representation

The experiment replaces only the four-site tensor core, not NEC as a whole.
Sparse outlier shielding and the covariance-projected low-rank bridge remain
available. For a matrix W with dimensions d_o × d_i, choose exact
factorizations d_o = o_0 o_1 and d_i = i_0 i_1, reshape W to
W[o_0,o_1,i_0,i_1], and make a candidate operator-Schmidt unfolding according
to a permutation π of those four axes:

\[
M_\pi=\operatorname{reshape}\!\left(\operatorname{permute}(W,\pi)\right).
\]

Its truncated SVD is \(M_\pi\approx U_r\Sigma_rV_r^\top\). The stored
two-site MPO factors are \(L=\Sigma_r^{1/2}U_r^\top\) and
\(R=\Sigma_r^{1/2}V_r^\top\), so
\(\widehat W_\pi=\operatorname{unpermute}(L^\top R)\). Equivalently, each
Schmidt term is a Kronecker product. The implementation also applies these
factors directly with two small contractions; it need not form the dense
matrix for a single linear operator application.

Candidate cuts preserve attention head coordinates and SwiGLU neuron groups,
but explicitly compare head-aligned, crossed, and balanced alternatives. The
partition is selected by the construction objective rather than assuming that
semantic grouping is automatically compressible. The MLP candidate includes
the exact 11,008 = 43×256 neuron grouping used by Llama-2-7B, as well as a
balanced 86×128 alternative.

For fixed sparse and bridge components \(S,B\), let \(E=W-S-B\), calibration
inputs \(X\), and dense targets \(Y=WX\). The target objective is

\[
\mathcal L(\widehat W)=
\lambda_w\frac{\|W-\widehat W\|_F^2}{\|W\|_F^2}
 +\lambda_a\frac{\|(W-\widehat W)X\|_F^2}{\|WX\|_F^2}.
\]

Before factorization, a diagonal metric approximates the activation covariance:

\[
D_{jj}^2=\frac{\lambda_w}{\|W\|_F^2}
 +\frac{\lambda_a\|X_{j,:}\|_2^2}{\|WX\|_F^2},
\qquad M_\pi=\operatorname{unfold}_\pi(E D).
\]

This gives an algebraic SVD warm start that uses both weight energy and
calibration-channel energy. For each candidate, the code then scores the
actual normalized mixed objective (including the current quantized sparse,
bridge, and core tensors) on construction data. A small ridge-stabilized
least-squares problem jointly sets the three component amplitudes. This is a
closed-form amplitude solve, not gradient training or post-compression
healing. The sparse and bridge subspaces are still constructed sequentially;
the amplitude solve does not claim a fully joint optimum over all their
coordinates.

The objective used to choose a cut is measured on construction data, not
validation data. Held-out data is reserved for the diagnostic report. The
diagonal metric ignores off-diagonal activation covariance, so it is only a
preconditioner; the real test is the held-out output error and then the
downstream cascade.

## Prior-art boundary

Kronecker/low-rank fully connected compression, spectral tensor-train
factorization, and MPO-based transformer compression are established ideas;
the operator-Schmidt/Kronecker primitive and head-aware layouts are not
claimed as novel. Relevant prior work includes [Kronecker Fully-Connected
Layers](https://arxiv.org/abs/1507.05775), [Spectral Tensor-Train
Compression](https://proceedings.mlr.press/v130/obukhov21a.html), and a
[transformer MPO compression study](https://arxiv.org/abs/2603.28534). The
activation-aware diagonal metric also has related work in [AWT](https://arxiv.org/abs/2610.10085).
The engineering hypothesis here is narrower: choose among exact transformer-
specific unfoldings at NEC's existing core budget, score them before local
fitting with a normalized weight-plus-activation objective, and measure how
that interacts with NEC's existing sparse and bridge terms. This is an
NEC-specific empirical redesign, not a priority or novelty claim.

### Budget and preliminary evidence

For the 4096×4096 attention projections and high-fidelity ranks
`[1, 64, 128, 64, 1]`, the old four-site core has 1,056,768 coefficients.
At this same core coefficient budget, a crossed 4096×4096 operator-Schmidt
cut uses rank 129 and 1,056,768 coefficients; a head/head 1024×16384 cut uses
rank 60 and 1,044,480 coefficients. The unequal ranks are expected: the
factor count is \(r(m+n)\), where \(m,n\) are the unfolding dimensions.

At the current artifact dtype (FP16 tensors), both this new core and the
legacy four-site core store those 1,056,768 coefficients in 2,113,536 bytes.
The new input scale is 4,096 FP16 values (8,192 bytes), comparable to the old
diagonal d_awt; matched sparse and bridge budgets have the same leading
payload. Thus the predicted tensor payload is similar, not smaller, before
small metadata differences. The saved layer-15 artifact was 70,110,452
serialized bytes against 404,766,720 bytes for its seven dense FP16
projections (5.77×); that is the existing NEC baseline, not a measured
operator-Schmidt result. The GPU harness measures the candidate's complete
serialized layer checkpoint before making any storage claim.

A preliminary, weight-only CPU screen on saved layer-15 dense projections
held the old sparse-plus-bridge baseline fixed and replaced only the MPO
core. Relative dense-weight error changed as follows: Q 0.949→0.821, K
0.943→0.815, V 0.982→0.840, O 0.959→0.899. For V, sparse-plus-bridge alone
was 0.884. A naive head/head cut was weaker than the selected crossed cut on
V (0.850 vs 0.840). These numbers motivate a matched real-model screen; they
are not activation-held-out results, a transformer-block result, or evidence
of retained model capability. The detailed GPU harness below records all
candidate scores so this preliminary screen can be reproduced against the
saved prefix checkpoints.

The new core's coefficient count is matched to or below the old core budget,
but this is not by itself a storage claim. The on-disk comparison must count
the factors, input scale, sparse indices and values, bridge, per-layer
metadata, and actual `torch.save` bytes. The diagnostic records both tensor
payload bytes, per-projection artifact bytes, and the assembled candidate
layer checkpoint bytes against the saved layer. Whole-model size must also
include unchanged embeddings, output head, and other components.

## Code and CPU verification

- `nox/operator_schmidt.py`: layout enumeration, deterministic exact/randomized
  truncated SVD, coefficient accounting, mixed-objective construction,
  artifact serialization/reconstruction, and direct factorized application.
- `nox/block_compiler.py`: opt-in `representation="operator_schmidt"`; the
  existing `mpo4` path remains the default and is preserved.
- `nox/compiler.py`: codec dispatch for reconstruction.
- `scripts/compile_high_fidelity.py`: opt-in `--representation
  operator_schmidt`; output identity records the method and objective weights.
- `tests/test_operator_schmidt.py`: both contraction layouts, full-rank
  reconstruction, screened synthetic topology, joint gain solve,
  serialization/byte count, numerical stability, and all seven block
  projections.
- `scripts/diagnose_nec_components.py`: compares saved core/sparse/bridge
  ablations and candidate cuts per projection, reports the old MPO cut
  spectra, and tests dense restoration of individual Q/K/V/O/Gate/Up/Down
  operators plus attention/MLP groups at the block output.

The complete CPU suite ran 86 tests: 83 passed and 3 existing CUDA-only PEES
tests were skipped. The synthetic tests verify algebra, shape, serialization,
coefficient count, and the direct-vs-dense apply path. They do not demonstrate
that a transformer can already run end-to-end without materializing every
projection: the existing evaluation injector still reconstructs weights. No GPU run has yet
validated runtime, numerical error at Llama scale, or model quality.

## GPU plan — do not execute without explicit authorization

The saved 4-layer source checkpoint and dense projection cache are expected
on the prior experimental host at
`/home/shadeform/diag36h-research/checkpoints/diagnostic_36h/high_fidelity_no_healing_4layer_prefix`
and `models/diagnostic_36h_cache`. Verify both after authorization; do not
assume that the stopped instance, address, or environment persisted. Use the
existing host only; do not provision a replacement.

After the user authorizes starting/using the GPU, verify the machine and
available disk/memory, then copy only the changed NEC source, harness, and
tests to the experimental checkout. The previous host and key path supplied
for this project were `shadeform@129.80.249.133` and
`/home/fluxx/Workspace/research/A10-mvafve1s_key.pem`; the host must be
re-confirmed after restart. Once its address is confirmed, these are the
read-only preflight and targeted source-sync commands (do not sync unrelated
or untracked files):

```bash
GPU_HOST=shadeform@129.80.249.133  # replace only if the restarted host has a new address
ssh -i /home/fluxx/Workspace/research/A10-mvafve1s_key.pem \
  -o BatchMode=yes -o ConnectTimeout=15 "$GPU_HOST" \
  nvidia-smi
ssh -i /home/fluxx/Workspace/research/A10-mvafve1s_key.pem \
  -o BatchMode=yes -o ConnectTimeout=15 "$GPU_HOST" \
  test -x /home/shadeform/diag36h-research/.venv/bin/python
ssh -i /home/fluxx/Workspace/research/A10-mvafve1s_key.pem \
  -o BatchMode=yes -o ConnectTimeout=15 "$GPU_HOST" \
  test -d /home/shadeform/diag36h-research/checkpoints/diagnostic_36h/high_fidelity_no_healing_4layer_prefix
ssh -i /home/fluxx/Workspace/research/A10-mvafve1s_key.pem \
  -o BatchMode=yes -o ConnectTimeout=15 "$GPU_HOST" \
  test -d /home/shadeform/diag36h-research/models/diagnostic_36h_cache
rsync -avR -e 'ssh -i /home/fluxx/Workspace/research/A10-mvafve1s_key.pem -o BatchMode=yes' \
  ./nox/__init__.py ./nox/block_compiler.py ./nox/compiler.py \
  ./nox/operator_schmidt.py ./scripts/compile_high_fidelity.py \
  ./scripts/diagnose_nec_components.py ./tests/test_operator_schmidt.py \
  "$GPU_HOST:/home/shadeform/diag36h-research/"
```

The private key is only read by SSH and must not be copied, printed, or
committed. From
`/home/shadeform/diag36h-research`, use the existing
`.venv/bin/python`. The source changes should first pass the focused tests on
that host. Then run these stages in order:

```bash
# 1. Original component ablation + matched candidate screen on damaging layer 0.
.venv/bin/python -u scripts/diagnose_nec_components.py \
  --checkpoint-dir checkpoints/diagnostic_36h/high_fidelity_no_healing_4layer_prefix \
  --dense-weights-dir models/diagnostic_36h_cache \
  --layer 0 --device cuda --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --output checkpoints/diagnostic_36h/operator_schmidt/layer0_all_projections.json

# 2. Compile the new representation for layers 0–1, using the matched pinned
#    model revision, raw WikiText train/validation samples, and high-fidelity
#    rank/bridge/sparse budgets. Construction remains zero-healing.
.venv/bin/python -u scripts/compile_high_fidelity.py \
  --output checkpoints/diagnostic_36h/operator_schmidt/prefix2 \
  --start-layer 0 --end-layer 1 --n-calib 1024 --n-validation 1024 \
  --seq-len 128 --calibration-sampling uniform --validation-sampling uniform \
  --representation operator_schmidt --operator-schmidt-weight-mix 1.0 \
  --operator-schmidt-activation-mix 1.0 \
  --model-id NousResearch/Llama-2-7b-hf \
  --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --device cuda

# 3. Same raw WikiText-2 test protocol and pinned model; include the dense
#    baseline and record actual serialized bytes, next-token scores, and PPL.
.venv/bin/python -u benchmarks/benchmark_day4_perplexity.py \
  --checkpoint_dir checkpoints/diagnostic_36h/operator_schmidt/prefix2 \
  --model_id NousResearch/Llama-2-7b-hf \
  --model_revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --device cuda --eval_dense --seq_len 2048 --stride 512 --max_eval_tokens 10000

# 4. Only if the 2-layer result materially beats the saved NEC prefix and
#    runtime/remaining credit allow, resume to layers 0–3 and re-evaluate.
.venv/bin/python -u scripts/compile_high_fidelity.py \
  --output checkpoints/diagnostic_36h/operator_schmidt/prefix2 \
  --start-layer 0 --end-layer 3 --n-calib 1024 --n-validation 1024 \
  --seq-len 128 --calibration-sampling uniform --validation-sampling uniform \
  --representation operator_schmidt --operator-schmidt-weight-mix 1.0 \
  --operator-schmidt-activation-mix 1.0 --resume \
  --model-id NousResearch/Llama-2-7b-hf \
  --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d --device cuda

.venv/bin/python -u benchmarks/benchmark_day4_perplexity.py \
  --checkpoint_dir checkpoints/diagnostic_36h/operator_schmidt/prefix2 \
  --model_id NousResearch/Llama-2-7b-hf \
  --model_revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --device cuda --eval_dense --seq_len 2048 --stride 512 --max_eval_tokens 10000
```

The 4-layer resume command above intentionally extends the 2-layer output
directory so it reuses the exact already-written layer-0/1 artifacts. If the
GPU screen finds a strong single-layer candidate but layer-0 restoration
points to a different projection, first compile/evaluate only that projection
group with a dedicated follow-up harness; do not launch a full cascade by
default. No MMLU score is planned in this four-hour diagnostic budget.

### Time and memory budget

The matched old high-fidelity prefix record reports 766.9–774.0 seconds per
layer (about 12.8 minutes; 51.4 minutes for four layers). The new method omits
the old repeated DMRG fitting sweeps but evaluates multiple randomized-SVD
cuts. Its runtime has not been measured, so a realistic initial allowance is
15–25 minutes per new layer plus 20–40 minutes for the component screen and
model/data startup. A 2-layer run, matched PPL pass, and component diagnostic
are estimated at roughly 1–2.5 hours; extending to four layers and rerunning
PPL could consume about 2–4 hours. These are planning estimates, not timing
claims. Stop if the first stage is materially slower or fails the held-out
gate. A complete 32-layer run is not feasible to assume within four hours:
the previous baseline alone implies about 6.8 hours, before evaluation.

The previous host was an A10-class 24-GiB GPU, but that must be rechecked.
Projection compilation should need substantially less memory than full-model
evaluation; budget conservatively for up to 8 GiB compiler memory until
measured. The 7B FP16 model plus activations and logits likely requires
roughly 16–22 GiB for the 2,048-token PPL run. Confirm `nvidia-smi` first and
watch peak allocated/reserved memory. If the verified GPU is smaller, do not
attempt the PPL command without an approved offload plan.

## Risks and decision gates

1. A better weight/activation matrix fit may still not preserve attention
   softmax or SwiGLU block behavior; restoration ablations and held-out block
   metrics are required.
2. The diagonal covariance preconditioner may select a cut that is good for
   the sampled channel marginals but poor for correlated held-out inputs.
3. Sequential sparse/bridge/core construction plus scalar gain fitting may
   retain destructive component interference; gains are not a joint
   factor-coordinate optimum.
4. The new core may have a similar coefficient count but slightly different
   serialized overhead. Compare complete artifact and checkpoint bytes.
5. Direct factorized linear application is implemented and CPU-tested, but
   attention/MLP factorized inference is not wired into the existing Hugging
   Face model injector, and no speedup is claimed.
6. Operator-Schmidt/Kronecker decompositions and head-/group-aware layouts
   have prior art. This is an NEC-specific representation-screening and
   fitting improvement, not a claim of a new mathematical primitive.

Proceed to four layers only if the matched 2-layer PPL and held-out block
metrics show a substantial improvement over the original catastrophic prefix
and storage remains materially below dense. PPL is only a proxy; it does not
certify the 90% downstream-accuracy target. Do not launch full-model
compilation until the 4-layer gate is strong and remaining credit is verified.
