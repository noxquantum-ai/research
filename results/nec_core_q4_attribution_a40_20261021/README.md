# NEC core-wise Q4 attribution on A40

This is one High-Fidelity NEC qualification of Llama-2-7B layer 0 `down_proj`
using the pinned model and WikiText-2 calibration windows. It was run after
adding a diagnostic callback that saves the four FP32 MPO cores immediately
before post-sweep Q4 quantization. The production compiler defaults and fit
were unchanged.

## Configuration and provenance

- Model: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
- Source commit: `e5f69f1ec3ddb6a433f8aa9cb4b856d619c47063`
- GPU: NVIDIA A40, 46,068 MiB reported VRAM
- Projection: layer 0 `down_proj`, shape `4096 x 11008`
- Data: pinned `Salesforce/wikitext` raw WikiText-2 revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`; 4,096 uniform train tokens for
  calibration, 1,024 disjoint uniform validation tokens, sequence length 128,
  seeds 42 and 43
- NEC: ranks `[1,64,128,64,1]`, bridge rank 128, sparse budget 524,288, two
  DMRG sweeps, 48 CG iterations maximum, post-sweep Q4, group size 64
- Compilation runtime: 479.9 s; wrapper wall time: 511 s; estimated GPU charge:
  `$0.08375` at `$0.59/hour`
- PyTorch peak allocated VRAM: 4.41 GiB; sampled device use: 5.10 GiB

## Findings

All errors below are relative Frobenius output errors. The absolute error is
normalized by the dense projection output; the Q4 transition is normalized by
the floating-point MPO output immediately before quantization.

| Stage / intervention | Calibration error vs. dense | Validation error vs. dense | Q4 output change vs. float MPO |
| --- | ---: | ---: | ---: |
| TT warm start | 36.31% | 41.49% | — |
| DMRG sweep 1 | 34.10% | 41.79% | — |
| DMRG sweep 2, before MPO Q4 | 33.46% | 42.02% | — |
| All four MPO cores Q4 | 33.70% | 42.24% | 4.675% |

The DMRG sweeps lower calibration error by 2.85 percentage points from TT
initialization, while held-out error rises by 0.53 points before Q4. Q4 then
raises the absolute held-out error by a further 0.216 points. Thus, Q4 causes a
measurable operator change, but it does not explain the roughly 42% absolute
projection error.

| MPO core quantized alone | Validation output change vs. float MPO | Absolute dense-target validation error |
| --- | ---: | ---: |
| Core 0 | 2.269% | 42.072% |
| Core 1 | 2.231% | 42.069% |
| Core 2 | 2.236% | 42.071% |
| Core 3 | 2.507% | 42.083% |

The cumulative Q4 output change after cores 0, 1, 2, and 3 is 2.269%, 3.189%,
3.913%, and 4.675%, respectively. These changes are not additive. Core 3 is the
largest single-core contributor in this run, though its isolated absolute
dense-target error impact remains small.

The simulated all-core Q4 operator and the serialized artifact differ by
0.021% on validation outputs. That final comparison includes FP16 artifact and
stage-snapshot serialization effects as well as reconstruction arithmetic.

## Interpretation and scope

This is one real projection, not a full transformer or language-model quality
evaluation. It supports two separate conclusions: the current DMRG fitting
overfits this calibration/validation split at projection level, and the MPO
Q4 boundary changes the operator by 4.67% on validation inputs. The latter
raises absolute dense-target error only modestly because most error already
exists in the float MPO approximation. These measurements do not establish
WikiText perplexity or MMLU retention.

The per-core report was computed after fitting, using validation only for
measurement. No parameters or gauges were selected with validation data.

The qualification report, wrapper logs, heartbeat records, final compressed
artifact, and 3.30 MB pre-Q4 core snapshot are preserved in the corresponding
RunPod network-volume directory:
`/workspace/runs/nec-core-q4-attribution-a40_20261021/`. The repository copy
contains the reports and logs; bulky tensor snapshots remain on that persistent
volume.
