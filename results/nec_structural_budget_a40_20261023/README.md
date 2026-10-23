# NEC dual storage/structural budget study

Date: 2026-10-23  
Hardware: NVIDIA A40 48 GB  
Model: NousResearch Llama-2-7b-hf, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`

## Decision

The strict-budget 32-layer follow-up has completed and been exported as a
deployable package. It passes the 5x byte and 2x coefficient gates but fails
the quality gate:

| Full-model measurement | Result |
| --- | ---: |
| Source serialized bytes | 13,479,247,511 |
| Complete exported package bytes | **2,144,321,837** |
| Actual whole-model serialized compression | **6.2860189x** |
| Total logical model coefficients | **3,331,706,294** |
| Structural coefficient reduction | **2.0225125x** |
| Dense transformer layers retained | 0 of 32 |
| Validation-slice PPL | **320.2343**, dense 4.0829 |
| Teacher KL | 4.36274 nats/token |
| Next-token accuracy / dense top-1 agreement | 17.36% / 21.03% |
| Joint acceptance | **INFEASIBLE for this checkpoint: quality gate fails** |

The complete package contains 91 serialized files, including all dense
embeddings/head/norms, NEC core and branch artifacts, metadata, tokenizer files,
and copied reconstruction code. The retained tensor payload is 524,828,672
bytes, and the NEC checkpoint files total 1,615,658,230 bytes. The exact
coefficient count is 262,412,288 retained scalars plus 3,069,294,006 NEC
coefficients. Runtime still reconstructs dense matrices.

Layers 0–7 reuse the independently calibrated 20%-residual checkpoints. The
remaining 24 layers compiled in **106.30 seconds** on A40 with unchanged global
budgets. The package is on the persistent network volume at
`/workspace/runs/nec-structural2-column05-residual20-full32/export/deployable_model`.
Compact records are in [full32_and_controls.json](full32_and_controls.json).
The exported package produces **bitwise-identical logits** to injection into
the original dense architecture on a 128-token deterministic parity probe:
maximum absolute difference 0, relative logit error 0. Bundled NEC code matches
the verification source and serialized-file byte totals remain unchanged.
See [export_parity.json](export_parity.json). This verifies serialization;
the random parity inputs do not provide a capability score.
The 5x and 2x requirements are met together; 10x, 3x structural reduction,
and acceptable quality have not been achieved by this model.

NEC now treats serialized storage and logical coefficient count as independent
constraints. The existing byte-only packed Schmidt model fails the new
structural requirement: its completed 32-layer artifacts represent
5,614,065,208 total model coefficients, only **1.2003x** fewer than the dense
model's 6,738,417,664 coefficients.

A new dual-budget compiler limits every projection by both serialized bytes
and logical coefficients. It counts all factor values, quantization scales,
input/output scales, calibration branches, and activation-sensitive defect
values. Tensor padding and integer coordinates are storage overhead, not
learned coefficients, so they are excluded from structural count but included
in serialized bytes.

The 2x and 3x structural targets are mathematically compatible with the 5x
whole-model byte cap:

| Structural target | Total coefficient cap | NEC artifact coefficient budget | Projected package ratio with 4-bit factors* |
| --- | ---: | ---: | ---: |
| 2x | 3,369,208,832 | 3,106,796,544 | 6.21x |
| 3x | 2,246,139,221 | 1,983,726,933 | 8.51x |

\*Analytic tensor-payload projection with a uniform allocation and 1.25%
calibration branch; quality is not implied. The compiler's partial-run package
ratio is reported conservatively because missing layers are charged their full
assigned byte budgets.

## Matched eight-layer results

All rows replace every projection in layers 0–7, leave layers 8–31 dense, use
the same 4,095-token WikiText-2 test slice, and enforce a projected complete
model coefficient count at or below the 2x cap. This test slice has been used
repeatedly for development and selection; it no longer provides an untouched
generalization estimate. A final candidate requires fresh independent tokens.

| Representation | Calibration trajectory | PPL | Teacher KL | Accuracy | Dense agreement | Projected full-run structural lower bound | Projected full-run package ratio lower bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-site semantic cut | Dense | 481.72 | 4.6156 | 14.46% | 17.66% | 2.0005x | 5.2567x |
| Balanced Schmidt cut | Dense | 254.05 | 3.9545 | 21.44% | 25.20% | 2.0003x | 5.2552x |
| Balanced cut + 5% activation-sensitive column defect | Dense | **67.13** | **2.6523** | **31.11%** | **37.63%** | **2.0036x** | **5.2566x** |
| Same defect model | Compressed prefix | 220.05 | 3.8013 | 21.47% | 27.45% | 2.0036x | 5.2566x |
| Defect + 10% residual path | Dense | 54.78 | 2.4472 | 33.21% | 40.02% | 2.0035x | 5.2541x |
| Defect + 20% residual path | Dense | **51.64** | **2.3908** | **33.77%** | **41.07%** | **2.0034x** | **5.2540x** |
| Defect + 30% residual path | Dense | 59.88 | 2.5334 | 32.33% | 38.97% | 2.0035x | 5.2540x |
| Defect + 20% residual + 20% complementary tensor cut | Dense | 69.57 | 2.6737 | 30.18% | 36.87% | 2.0039x | 5.2531x |
| Dense teacher | — | 4.859 | 0 | 62.83% | 100% | 1x | 1x |

The best result still fails the PPL ≤ 10 quality gate. This is a
representation-quality failure. The projections above are conditional on
future compilation of the remaining 24 layers and are not the achieved
compression of the evaluated assembly.

For the 20% residual pilot, the compressed eight-layer region stores
770,903,430 coefficients versus 1,619,001,344 dense coefficients: **2.1001x**
structural reduction. Its 406,590,696 serialized checkpoint bytes versus
3,238,002,688 original dense FP16 weight bytes give **7.9638x** region storage
compression. Including the unchanged remainder gives 5,890,319,750 total
coefficients (**1.1440x**) and an estimated 10,647,835,519 package bytes
(**1.2659x**). The latter is assembled byte accounting, not a measured exported
package directory.

## Error attribution

Matched replacement ablations on the same development slice use the best
20% residual checkpoint:

| NEC projections replaced | Layers 0–7 PPL | Layers 0–31 PPL |
| --- | ---: | ---: |
| None | 4.859 | 4.859 |
| Q/K | 4.973 | 5.787 |
| V/O | 6.414 | 21.377 |
| All attention | 6.792 | 27.927 |
| Gate/Up | 13.957 | 52.886 |
| Down | 8.941 | 26.792 |
| All MLP | 32.731 | 226.105 |
| All projections | 51.637 | 314.123 |

These interventions localize the largest error to the MLP and demonstrate
additional attention/MLP interactions. Single-group replacements are
diagnostics, not candidate compression results with matched total storage.
The 32-layer test-slice ablation took 118.86 seconds including loading and
reconstruction; its raw record is
[full32_projection_ablation.json](full32_projection_ablation.json). The full
model fails on both the test development slice and the matched validation
slice; these are separate protocols and their PPL values must not be mixed.

## Construction and accounting corrections

Full raw evaluation records, model/dataset hashes and compile settings for
the residual and complementary-cut experiments are archived in
[remote_evaluations.json](remote_evaluations.json).

## Matched validation and covariance metric

The separable covariance experiment uses partial traces of the empirical input
covariance to construct two small mode metrics. NEC runs quantized ALS in
that product metric, while absorbing its inverse into the physical cores
before packing. The metric matrices are construction workspace only; they
add no persistent learned coefficients. This is an approximate product
covariance, not exact full-covariance whitening.

The algebra and artifact round-trip tests pass, but the matched 4,095-token
validation result regresses:

| 20% residual / 5% defect construction | Validation PPL | Teacher KL | Accuracy | Dense agreement |
| --- | ---: | ---: | ---: | ---: |
| Diagonal activation metric | **60.468** | **2.7047** | **33.36%** | **40.00%** |
| Product covariance metric | 71.438 | 2.8656 | 31.18% | 37.70% |
| Diagonal metric, MLP-weighted allocation | 63.378 | 2.7573 | 32.04% | 38.61% |
| Native low-rank MLP control, full input covariance | 57.762 | 2.6588 | 34.92% | 42.15% |
| Dense teacher | 4.083 | 0 | 66.37% | 100% |

The product metric remains experimental and is not the default. Activation
whitening and adaptive singular-value allocation have established precedent
in [SVD-LLM](https://arxiv.org/abs/2403.07378) and
[SVD-LLM V2](https://aclanthology.org/2025.naacl-long.217/).
The NEC hypothesis here concerns mode-separable metrics absorbed into MPO
cores under simultaneous coefficient and byte limits; no general novelty
claim is made.

The MLP-weighted allocation uses role multipliers 0.4 for Q/K, 0.7 for V/O,
and 1.3 for Gate/Up/Down, renormalized under the unchanged global budgets.
It regresses slightly against uniform role allocation. This rejects that
particular allocation; it does not establish that all nonuniform allocations
are worse.

The native low-rank control uses the established output/input SVD bipartition
for MLPs and full covariance on its single input mode. Attention retains the
NEC MPO representation. It improves validation PPL only modestly and remains
a separately labeled diagnostic, not a new tensor-network novelty claim or
the representation used in the exported full32 package.

### Accounting fixes

The residual branch now propagates the FP16 codec scales actually stored in
the artifact. Previously construction used simulated FP32 scales, producing
a small difference between its returned matrix and artifact reconstruction.
The regression test checks the sum of complementary tensor, coordinate and
activation residual branches against the decoded artifact.

Resume rejects artifacts exceeding their assigned byte or coefficient limits.
Manifests report the actual evaluated assembly separately from a full-run
projection. Completing every layer checkpoint does not by itself mark a full
deployable package as materialized.

## Activation-sensitive defect ablation

The defect branch selects input coordinates using the Schmidt residual norm
times calibration activation RMS, then solves a ridge-anchored closed-form
correction against clean teacher outputs. It is part of compilation, uses no
gradient fine-tuning, and is charged to both budgets.

On layer 0 alone, 5% generalized better than larger branches even though the
larger branches reduced calibration reconstruction error further:

| Defect allocation | Test PPL | Teacher KL | Dense top-1 agreement |
| ---: | ---: | ---: | ---: |
| 5% | **4.925** | **0.01353** | **95.63%** |
| 10% | 4.960 | 0.01848 | 95.21% |
| 20% | 4.977 | 0.01915 | 95.09% |

The eight-layer improvement from 254.05 to 67.13 shows that a coordinate
defect can preserve activation-sensitive structure more efficiently than bond
rank alone. The remaining gap shows that independent local corrections still
do not compose through depth. Sequential asymmetric calibration worsened the
result and is rejected in its current form.

## Next representation gate

The full32 follow-up establishes actual package size and the failure of the
combined quality target. The next candidate must
beat the defect model at the same exact coefficient and byte budgets on held-out
one-layer and eight-layer tests. Candidate search should jointly choose tensor
layout, bond profile, precision, and defect allocation under a module-level
attention/MLP objective. A multi-site MPO is justified only if its measured
function error per stored coefficient beats the balanced two-site baseline;
its tensor-network identity alone is insufficient.

Automatic tensor-rank and structure selection are established research areas;
NEC's experimental question is narrower: whether calibration-only,
function-aware selection can preserve a pretrained LLM under simultaneous
structural and serialized budgets without healing. Relevant baselines include
[MARS](https://proceedings.mlr.press/v206/kodryan23a.html),
[Permutation Search of Tensor Network Structures](https://proceedings.mlr.press/v162/li22y.html),
and [Approximately Optimal Core Shapes](https://proceedings.mlr.press/v202/ghadiri23a.html).

