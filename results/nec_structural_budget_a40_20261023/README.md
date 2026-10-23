# NEC dual storage/structural budget study

Date: 2026-10-23  
Hardware: NVIDIA A40 48 GB  
Model: NousResearch Llama-2-7b-hf, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`

## Decision

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
model coefficient count at or below the 2x cap.

| Representation | Calibration trajectory | PPL | Teacher KL | Accuracy | Dense agreement | Structural reduction lower bound | Package ratio lower bound |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two-site semantic cut | Dense | 481.72 | 4.6156 | 14.46% | 17.66% | 2.0005x | 5.2567x |
| Balanced Schmidt cut | Dense | 254.05 | 3.9545 | 21.44% | 25.20% | 2.0003x | 5.2552x |
| Balanced cut + 5% activation-sensitive column defect | Dense | **67.13** | **2.6523** | **31.11%** | **37.63%** | **2.0036x** | **5.2566x** |
| Same defect model | Compressed prefix | 220.05 | 3.8013 | 21.47% | 27.45% | 2.0036x | 5.2566x |
| Dense teacher | — | 4.859 | 0 | 62.83% | 100% | 1x | 1x |

The best result still fails the PPL ≤ 10 quality gate. This is a
representation-quality failure, not an allocation success.

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

Do not extend the present two-site codec to 32 layers. The next candidate must
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

