# A40 functional-equivalence diagnostic (2026-10-20)

This directory records a controlled real-model comparison of the original
full-matrix NEC compiler with the FP32 and FP64 streaming-statistics backends.
It is a projection-level qualification, not an end-to-end language-model
benchmark.

## Scope and provenance

- GPU: NVIDIA A40, 46,068 MiB reported VRAM.
- Model: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Source commit: `534ebda89d4a70db65685e5ec6efbd94ebbbea89` (clean tree).
- Numerical-source fingerprint:
  `3e294a4d8955f3a440c783b1b51e5aec55a6870ae8b7204a7ef615522a16bb72`.
- Operator: decoder layer 0 `down_proj`, shape `4096 x 11008`.
- Calibration: 4,096 uniformly sampled WikiText-2 raw training tokens.
- Validation: 1,024 uniformly sampled, disjoint WikiText-2 raw validation
  tokens.
- Production structure: ranks `[1,64,128,64,1]`, bridge rank 128, sparse
  budget 524,288, two DMRG sweeps, Q4 simulation enabled.
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`.

The direct functional metric is

```text
D_X = ||(W_stream - W_reference) X||_F / ||W_reference X||_F.
```

All variants used identical token IDs and hidden states, verified by hashes in
the reports.

## Final results

| Metric | Full matrix | Streaming FP32 | Streaming FP64 |
|---|---:|---:|---:|
| Core qualification time (s) | 215.51 | 167.91 | 192.11 |
| Peak PyTorch allocation (GiB) | 4.41 | 6.15 | 10.10 |
| Peak sampled device use (GiB) | 4.84 | 6.90 | 11.12 |
| Peak process RSS (GiB) | 8.50 | 2.36 | 2.36 |
| Calibration relative error | 0.336159 | 0.336175 | 0.336161 |
| Validation relative error | 0.420631 | 0.420573 | 0.420617 |
| Reconstructed-weight difference vs. reference | -- | 0.325240 | 0.319185 |
| Calibration direct functional disagreement | -- | 0.061333 | 0.060339 |
| Validation direct functional disagreement | -- | 0.069755 | 0.068494 |
| Strict 1% qualification | reference | **FAIL** | **FAIL** |

The nearly identical scalar reconstruction errors do not imply operator
equivalence. Both streaming results differ functionally from the reference by
about 6% on calibration activations and 7% on held-out activations. FP64 only
slightly reduces this disagreement and consumes substantially more VRAM.

## Stage localization

Values are `weight difference / calibration D_X / validation D_X` relative to
the corresponding full-matrix stage.

| Stage | FP32 | FP64 |
|---|---:|---:|
| Sparse shield | 0 / 0 / 0 | 0 / 0 / 0 |
| Sparse + bridge | 0.02593 / 0.01130 / 0.00973 | 0.02427 / 0.01076 / 0.00925 |
| TT warm start | 0.05690 / 0.01268 / 0.01164 | 0.05238 / 0.01197 / 0.01094 |
| DMRG sweep 1 | 0.12040 / 0.02005 / 0.02311 | 0.11187 / 0.01866 / 0.02151 |
| DMRG sweep 2 | 0.32524 / 0.06133 / 0.06975 | 0.31919 / 0.06034 / 0.06849 |
| Serialized artifact | 0.32524 / 0.06133 / 0.06975 | 0.31919 / 0.06034 / 0.06849 |

The sparse stage is identical. Detectable divergence first appears in the
covariance-derived bridge, grows through TT initialization and sweep 1, and is
amplified most strongly by the second in-loop-Q4 sweep. Serialization adds no
material divergence. This localizes the disagreement but does not by itself
prove a single root cause.

## Decision

Neither streaming backend is promoted to the production runner. Increasing
calibration to 8k/16k or running a complete layer would confound the unresolved
backend mismatch with scale. The next diagnostic should isolate the bridge,
TT-SVD and second-sweep inputs/solves against the direct-SVD reference before
any larger paid run. Independently, the reference compiler's 0.4206 held-out
relative error remains a serious fidelity warning that streaming does not fix.

Binary reconstructed matrices and artifacts are intentionally not committed;
their SHA-256 hashes, sizes, numerical results, full logs and heartbeats are in
the reports. The transfer archive used to preserve these records had SHA-256
`fce801da33d0bfde7b206b62f9e3f59498a926a402669ecd808d0c18cbb3d893`.
