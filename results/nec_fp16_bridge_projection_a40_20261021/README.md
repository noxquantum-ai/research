# NEC FP16-bridge full-projection control on A40 (2026-10-21)

This experiment tests whether keeping only the rank-128 bridge at FP16 removes
the direct-SVD versus FP64 streaming disagreement for the complete real
Llama-2-7B layer-0 `down_proj`. The sparse shield and final MPO cores remain
Q4-simulated. Both compilers use two DMRG sweeps, post-sweep MPO quantization,
and at most 48 matrix-free CG iterations. This is a projection-level numerical
control, not a language-model quality result.

## Result

The formal qualification failed its 1% reconstructed-weight and functional
parity thresholds.

| Measurement | Full-matrix reference | FP64 streaming |
|---|---:|---:|
| Runtime | 435.68 s | 467.73 s |
| Peak PyTorch allocation | 4.40 GiB | 10.10 GiB |
| Calibration reconstruction error | 33.5886% | 33.5903% |
| Held-out reconstruction error | 42.1636% | 42.1662% |
| Independent coefficients | 3,280,994 | 3,280,994 |

The similar scalar reconstruction errors conceal a material operator
difference: reconstructed-weight disagreement is 18.45%, calibration output
disagreement is 4.04%, and held-out output disagreement is 4.34%.

## First divergence and Q4 amplification

| Compiler stage | Held-out direct/streaming disagreement |
|---|---:|
| Sparse shield | 0.0000% |
| Sparse + FP16 bridge | 0.0202% |
| TT warm start | 0.0682% |
| DMRG sweep 1 | 0.1909% |
| DMRG sweep 2, before final MPO Q4 | 0.2319% |
| Serialized artifact, after final MPO Q4 | **4.3367%** |

This falsifies the hypothesis that bridge Q4 alone explains the complete
projection mismatch. FP16 keeps the bridge nearly equivalent and reduces the
previous stable control's final held-out disagreement from 6.85% to 4.34%, but
the final independent Q4 of the MPO cores amplifies a 0.232% pre-quantization
difference by approximately 18.7 times. The next numerical target is therefore
the MPO gauge/quantization boundary, not another bridge factorization.

The reference NEC projection itself still has 42.16% held-out output error.
Consequently, improved compiler parity would not by itself establish adequate
compression fidelity or justify a 32-layer run.

## Reproducibility and cost

- Model: `NousResearch/Llama-2-7b-hf`
- Revision: `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
- Dataset: WikiText-2 raw, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`
- Calibration/validation: 4,096/1,024 tokens, uniform disjoint windows
- Source commit: `23b781b6e7a4ffdbad11a1758ba7236f72e4b298`
- GPU: NVIDIA A40, 46,068 MiB
- Wrapper GPU-cost estimate: approximately $0.161 total at $0.59/hour,
  including the comparison stage
- Qualification status: `fp64_passes: false`

The large artifacts and stage matrices remain on the persistent volume under
`/workspace/runs/nec-streaming-qualification/down_projection_bridge_fp16`.
This directory contains both reports, progress records, comparison output,
heartbeats, and complete logs.
