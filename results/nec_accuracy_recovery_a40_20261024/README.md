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
| Held-out WikiText-2 PPL | NEC 8.348960; dense 4.867285 | Aim ≤ dense; fallback ≤ dense + 0.5 |
| Next-token accuracy | NEC 53.9990%; dense 62.8199% | Within 2 percentage points of matched dense |
| Dense top-1 agreement | 71.0259% | At least 98% |
| Teacher KL | 0.543950 nats/token | Must decrease |

The 0.5 PPL allowance is an absolute difference. The token accuracy tolerance is
interpreted as two percentage points. Teacher agreement is measured against the
dense model's 100% self-agreement. No claim is made that these targets have been
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
98% agreement requirements even if lossless packaging fixes its storage.
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

A separate full 32-layer trial is now running with the shared-basis choices,
training-only inverse channel-RMS MLP weighting, and curvature-relative
coefficient anchoring. All seven operators, existing coefficient counts,
three-bit excitation amplitudes and five-bit MPO cores remain fixed. Source is
frozen separately from the earlier trial. Its persistent status is
`/workspace/runs/nec-accuracy-recovery-20261024/canonical_relative_status.json`.
No result from this trial is claimed before its model-level evaluation.

An optional lossless container for retained FP16 safetensors is also being
measured. It preserves every coefficient and uses temporary disk space while
loading. A single real retained shard fell from 262,551,408 to 204,062,652 bytes
with verified byte-identical recovery. Complete-package bytes and runtime
verification are still required; this is a packaging improvement, not an
accuracy improvement.
