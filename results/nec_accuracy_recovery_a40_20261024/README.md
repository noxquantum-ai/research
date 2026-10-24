# NEC accuracy recovery at fixed or better compression

The original published NEC hybrid remains the protected baseline. Work uses the
full Llama-2-7B architecture and NEC for all seven projections in all 32 layers,
with no gradient-based fine-tuning or healing. Remaining capability benchmarks
were stopped at the user's request and are not being resumed.

## Acceptance policy

| Metric | Current measured baseline | New requirement |
| --- | ---: | ---: |
| Complete published serialized bytes | 2,490,273,288 | At most this many actual bytes |
| Actual whole-model storage compression | 5.412758341× | At least this ratio |
| Total logical coefficients, including scales | 3,323,032,892 | At most this many coefficients |
| Structural reduction | 2.027791443× | At least this ratio |
| Dense transformer layers retained | 0 | 0 |
| Held-out WikiText-2 PPL | NEC 8.348960; dense 4.867285 | Aim ≤ dense; current fallback ≤ dense + 1.0 (updated 2026-10-24) |
| Next-token accuracy | NEC 53.9990%; dense 62.8199% | 54–62% (updated by user) |
| Dense top-1 agreement | 71.0259% | At least 85% (updated by user) |
| Teacher KL | 0.543950 nats/token | At most 0.5 nats/token |

The current 1.0 PPL allowance is an absolute difference; it supersedes the earlier 0.5 allowance at the user’s explicit request. The full-test ceiling is 5.867285. The original focused window was 08:52:23–09:22:23 UTC; the user subsequently specified another 15 minutes, updating the reporting deadline to approximately 09:34:42 UTC. This is a work budget, not a guarantee of reaching the quality target. The token accuracy tolerance is
interpreted as two percentage points. Teacher agreement is measured against the
dense model's 100% self-agreement. The latest user gates are explicitly PPL ≤5.867, KL ≤0.5, next-token accuracy 54–62%, and agreement ≥85%. Earlier 98%/two-point policies in historical reports are retained as history. No claim is made that these targets have been
achieved or are guaranteed to be achievable at this budget.

Use matched input IDs, context 2,048, stride 512 and dataset provenance. Prior
confirmation used the 308,700 scored WikiText-2 test tokens after offset 32,768.
That range is now a known regression set, not a fresh untouched test. Candidate
selection uses training construction/guard windows; validation is reported
separately, and full regression evaluation is reserved for promising candidates.
A further untouched corpus is needed for a new generalization claim.

Passing the token metrics does not establish that BoolQ or HellaSwag remain
unchanged; those claims require subsequent capability evaluation. Existing full
scores remain recorded: BoolQ NEC 48.50% vs matched dense 79.36%; NEC HellaSwag
50.39% length-normalized accuracy. No candidate replaces the published baseline
on the strength of local reconstruction errors alone.

## First construction experiment

The existing hybrid retains its Schmidt/MPO backbone and the exact same sparse
amplitude and scale counts. Compare:

1. Channel-relative normal-equation reconstruction on fixed support, with
   analytically fitted scalar scales.
2. An eight-level nonzero amplitude grid, exploiting all three-bit codes for
   selected nonzero residual amplitudes without adding scale coefficients.
3. Covariance-aware support and sequential error feedback for both pruning and
   rounding, in a temporary unit-RMS input gauge. No covariance or inverse matrix
   is retained for inference.

Fit on 24,576 training tokens, select on a disjoint 8,192-token training guard,
and report separate 8,192-token validation errors. The initial layer-0 probe
holds the other operators and all later layers fixed. Candidate tensor payload
and logical coefficient counts must match exactly. Complete serialized gzip
bytes must also be measured before promotion: equal tensor bytes do not imply
equal lossless-container bytes.

## Research grounding

[SparseGPT (ICML 2023)](https://proceedings.mlr.press/v202/frantar23a.html)
provides analytical second-order sparse reconstruction and error feedback.
[SLiM (ICML 2025)](https://proceedings.mlr.press/v267/mozaffari25a.html)
combines one-shot sparsity, quantization and analytical low-rank compensation.
[EoRA](https://arxiv.org/abs/2410.21271) emphasizes activation-eigenspace residual
approximation without gradient training.
[QuIP# (ICML 2024)](https://proceedings.mlr.press/v235/tseng24a.html) motivates
attention to covariance conditioning and low-bit rounding error. Its full method
also includes fine-tuning; that component is excluded here.

These ideas are established prior art. Applying them to NEC's existing MPO plus
coordinate-excitation representation is an experimental design, not a claim of
novelty or a substitution of conventional quantization for NEC.

## Measured initial controls

Layer-0 constructor selection used training fit/guard tokens only, then reported
separate validation activations. Exact tensor payloads and coefficient counts
were unchanged:

| Projection | Baseline validation relative MSE | Selected constructor | New validation relative MSE |
| --- | ---: | --- | ---: |
| Query | 0.00554723 | Covariance feedback, nonzero grid | 0.000389999 |
| Down | 0.0641991 | Relative-anchored PCG, fitted scales | 0.0616066 |

See [local raw probe](layer0_constructor_probe.json). These are local operator
errors, not full-model quality measurements.

A full 32-layer Q/K-only reconstruction, keeping every other NEC operator fixed,
failed the no-regression policy on 8,191 matched validation tokens:

| Metric | Current NEC baseline | Q/K control |
| --- | ---: | ---: |
| PPL | 6.697377 | 6.671261 |
| Teacher KL | 0.535163 | 0.529597 |
| Next-token accuracy | 57.3678% | 57.1847% |
| Dense top-1 agreement | 72.0181% | 71.9937% |

The control is **not promoted**. Better independent reconstruction did not
preserve all model-level metrics. See [baseline](baseline_validation_8k.json),
[control](qk_control_validation_8k.json) and [acceptance record](qk_control_status.json).

The full sequential reconstruction of all seven operators completed. It used
actual compressed-prefix inputs, clean teacher targets, and fixed existing
Schmidt/branch shapes. It failed every matched validation quality comparison:

| Validation metric | Protected hybrid | Sequential reconstruction | Teacher-logit selected mixture |
| --- | ---: | ---: | ---: |
| PPL | 6.697377 | 6.811285 | 6.655036 |
| Teacher KL | 0.535163 | 0.549769 | 0.530365 |
| Next-token accuracy | 57.3678% | 56.7086% | 57.3678% |
| Dense top-1 agreement | 72.0181% | 71.3954% | 72.1646% |

The selected mixture used two conditional passes over complete attention and
MLP groups, accepting eight switches in 130 student evaluations on training
guard windows only. It retained the exact logical coefficient and tensor
payload budgets. It was then evaluated on the separate full test regression:

| Full test metric (308,700 scored tokens) | Protected hybrid | Selected mixture |
| --- | ---: | ---: |
| PPL | 8.348960 | 8.268307 |
| Teacher KL | 0.543950 | 0.535477 |
| Next-token accuracy | 53.9990% | 54.1108% |
| Dense top-1 agreement | 71.0259% | 71.1775% |
| Actual complete serialized bytes, NEC gzip only | 2,490,273,288 | 2,492,350,788 |
| Actual storage ratio | 5.412758341× | 5.408246534× |
| Logical coefficients | 3,323,032,892 | 3,323,032,892 |

**Neither trial is promoted.** The selected mixture improves all four quality
metrics slightly but exceeds the byte limit by 2,077,500 bytes. Its exported
runtime passed bitwise logit parity. It remains far from the dense-quality and
agreement requirements even if lossless packaging fixes its storage.
See [full sequential evaluation](coupled_stage32_validation_8k.json),
[selected validation](selected_construction_validation_8k.json),
[selected full test](selected_construction_heldout_test.json), and
[actual package measurement](selected_construction_package_measurement.json).

Causal controls, using the original dense counterparts for the other operators,
localize the larger contribution to the MLP:

| Original hybrid replacement control | Validation PPL | Teacher KL | Teacher agreement |
| --- | ---: | ---: | ---: |
| Dense | 3.984860 | 0 | 100% |
| Attention only compressed | 4.531894 | 0.138856 | 85.3376% |
| MLP only compressed | 5.701762 | 0.366502 | 77.0968% |
| All seven compressed | 6.697377 | 0.535163 | 72.0181% |

These controls are diagnostic, not storage-feasible deployment candidates or
lower bounds on what joint compression can achieve. Both components need
improvement; independent output MSE is not sufficient to establish quality.
Raw [attention](baseline_attention_only_validation_8k.json) and
[MLP](baseline_mlp_only_validation_8k.json) controls are archived.

A lossless scalar bitplane packing prototype was also rejected: on the first
reconstructed layer its gzip size increased from 52,427,541 to 52,571,543 bytes.
Integer levels round-tripped exactly, but it did not satisfy the storage
objective. The prototype codec was not retained. Its [measurement record](bitplane_layer0_rejected.json)
is archived to avoid repeating the unsuccessful direction.

## Shared input basis for Schmidt excitations

A version-2 sparse excitation artifact can store amplitudes in the existing
MPO input basis. If the backbone already stores input scales `s[j]`, the
physical sparse entry is `amplitude[i,k] / s[index[i,k]]`. No second scale
vector is stored. The exact amplitude-to-weight map, adjoint and curvature
preconditioner all use this same factor. This lets the existing calibration
preconditioning influence residual rounding as well as the MPO backbone.

The new mode uses the same amplitude count, scale count, support bitmap and
bit width. It supports both fresh compilation with
`--sparse-amplitude-basis backbone_input_scale` and guard-selected reconstruction
methods `pcg_canonical` and `feedback_canonical`. Existing version-1 artifacts
retain their original decoding. Forty-six targeted CPU checks passed, including
budget accounting, linear adjoints, constructor guards and exact plain/gzip
package parameter round trips for the new artifact type. These are engineering
checks, not evidence of model quality.

This is an input-coordinate preconditioner shared with NEC's tensor-network
representation, not a claim of a quantum speedup or a newly discovered
mathematical technique. Full-model quality and complete serialized bytes remain
required before acceptance.

The layer-0 MLP probe selected the shared basis for Gate, Up and Down on
training guards and reduced separate validation projection MSE. Nevertheless,
replacing that complete MLP in the full hybrid worsened validation PPL to
6.717929, KL to 0.538201, accuracy to 57.1603%, and agreement to 71.9326%.
This candidate is rejected. See [projection errors](canonical_layer0_probe.json)
and [model evaluation](canonical_layer0_model_validation_8k.json).

The full 32-layer shared-basis/channel-relative trial completed and failed all
four validation comparisons: PPL 6.761066, KL 0.535584, accuracy 57.0993%, and
agreement 71.9448%. Conditional operator selection improved PPL to 6.618507,
but accuracy declined to 57.2458%; it was rejected. See
[construction](canonical_relative_validation_8k.json),
[selected validation](canonical_selected_validation_8k.json) and
[rejection](canonical_selected_status.json).

## Confirmed analytical readout construction

A prior-anchored closed-form solve reconstructs the existing FP16 output head
from paired dense/NEC final hidden features. It explicitly changes the readout;
it is analytical distillation with no gradient-based fitting, not a claim that
the head stays untouched. Fit uses twelve construction-training windows and
selection uses the remaining four training guards. No delta matrix or extra
inference coefficients are retained. Matched validation and full test follow
selection; test tokens never choose the head.

| Matched full-test metric | Published hybrid | Selected body, original readout | Selected body, constructed readout |
| --- | ---: | ---: | ---: |
| PPL (dense 4.867285) | 8.348960 | 8.268307 | **8.187282** |
| Teacher KL | 0.543950 | 0.535477 | **0.520827** |
| Next-token accuracy | 53.9990% | 54.1108% | **54.4033%** |
| Teacher agreement | 71.0259% | 71.1775% | **71.9210%** |
| Complete package bytes | 2,490,273,288 | 2,382,168,906 | **2,420,734,087** |
| Storage ratio | 5.412758341× | 5.658392852× | **5.568247906×** |
| Logical coefficients | 3,323,032,892 | 3,323,032,892 | **3,323,032,892** |
| Structural reduction | 2.027791443× | 2.027791443× | **2.027791443×** |

The two selected-body points form a trade-off: changing the readout improves
quality but makes its FP16 bytes less losslessly compressible. Both satisfy
the published hybrid’s byte ceiling. The constructed-readout point improves
all measured token-quality metrics over the published hybrid, but **still
fails the dense-quality, accuracy and agreement targets**. The remaining
capability benchmarks have not been resumed or remeasured.

The probe took about 50 seconds. Its short validation result is PPL 6.580348
on 8,191 scored tokens, with dense PPL 3.984860. This is the same full 32-layer
architecture as the longer test; its PPL must not be substituted for the
308,700-token test result. The original evaluator simultaneously measures old
and new heads from the same NEC hidden states, avoiding duplicate body passes.
See [matched validation](readout_validation_matched_8k.json),
[full test](readout_heldout_test.json), [complete bytes](readout_package_measurement.json),
[GPU bitwise parity](readout_runtime_parity.json) and
[persistent copy verification](readout_persistent_copy_verification.json).

Lossless gzip now also covers retained FP16 safetensors, with exact byte
round trips and temporary decompression during loading. The complete candidate
is saved on the network volume at
`/workspace/runs/nec-accuracy-recovery-20261024/readout_candidate/deployable_model`.
Its file hashes and total size match the GPU-verified scratch package. The
published HF revision remains unchanged. No model weights were downloaded to
the workstation. CPU logit parity differed under its execution path; CUDA
fresh-interpreter parity passed exactly. Tiny package tests check exact
parameters for both plain/gzip retained containers and reconstructed heads.

## Bounded representation screens

The layer-0 Down precision control spends some freed container bytes on four-bit
NEC excitation amplitudes, retaining its MPO and exact coefficient count. It
worsened validation PPL to 6.661906 versus the selected body’s 6.655036 and
also worsened KL, accuracy and agreement, so no 32-layer run was launched.
[Raw control](down_precision_layer0_validation_8k.json) and
[storage preflight](down_precision_preflight.json) are recorded.

A training-only neuron-function Gram matrix drives a shared permutation of
Gate/Up rows and Down columns. This preserves the original dense SwiGLU
function to floating-point roundoff. At existing bond rank 98, the Down
weighted cut error fell from 0.946212 to 0.935080, while random permutation
stayed at 0.946020. This is a rank screen, not a model quality claim.

Fresh, matched identity/neuron-order/signed-Hadamard MLP constructions took
about 16.5 seconds together. The Hadamard arm was worse locally. Joint
construction reduced errors further, and the neuron permutation remained
better than its fresh identity control, but four resulting candidates failed
actual training-logit acceptance. None is promoted. The temporary materialized
weights are measurement files for saved NEC factor artifacts; they are not
reported as deployable compressed packages. See
[cut screen](swiglu_ordering_layer0_screen.json),
[MLP screen](mlp_basis_screen.json),
[readout-projected Gate screen](mlp_order_readout_screen.json), and
[end-to-end training guard](mlp_order_model_guard.json).

A further 65-second full-32-layer affine construction freed 131,072 coordinate
amplitudes/scales before introducing 131,072 bias coefficients. Sequential
mean corrections at full and half strength failed the actual training-logit
quality gate; no package was exported. This ruled out simple accumulated mean
shift as a sufficient repair. See [the rejected trial](affine_reconstruction_rejected.json).

The current full-32-layer Down trial uses a closed-form regularized asymmetric
target motivated by [regularized calibration](https://arxiv.org/html/2602.05902v1).
Its bounded operator search selected five groups and passed validation at PPL
6.538154; full test is running. Those validation tokens are a development
regression set, not proof that full-test PPL is near dense.

An explicit experimental allocator also redistributes existing coordinate
excitations across Gate/Up neurons using their nonlinear Down-weighted output
sensitivities. A global amplitude stream counts every value and scale,
retains the original MPO, and stays within each original coefficient ceiling.
It addresses the old uniform-per-row allocation rather than increasing the
model size. Its full-32-layer quality remains unverified.

## Latest full test and failed global allocation

The regularized Down construction plus training-logit selection measured
full-test PPL **8.128802**, KL **0.512765**, next-token accuracy **54.4775%**,
and teacher agreement **72.0246%**. It improves all four token metrics over
the saved 8.187282 candidate, but actual complete-package bytes are still
unmeasured. It is not yet a fully verified replacement. See
[full test](regularized_down_heldout_test.json) and
[validation](regularized_down_validation_8k.json).

The global output-sensitive Gate/Up allocation failed: validation PPL
**45.208806**, KL **2.455690**, accuracy **38.2493%**, and agreement
**44.4146%**. It was rejected before full-test evaluation or export. Per-row
feature coverage cannot be discarded on the strength of local weighted MSE.
This does not prove a universal impossibility bound; it rejects this allocator
and reinforces the mismatch between its proxy and end-to-end behavior.
[Raw evaluation](global_sectors_validation_8k.json).

The user’s requested time window did not produce PPL ≤5.867 / KL ≤0.5 /
accuracy 54–62% / agreement ≥85% together. These targets remain unmet.
The [Strata review](STRATA_REVIEW.md) records primary sources and boundaries
for applying its ideas to NEC.
