# Live hybrid NEC trial reports

Snapshot phase: `compiling_stage24`; UTC: 2026-10-23T20:15:46.140144+00:00.

These are measured checkpoints from the new pod. Partial checkpoints retain
all uncompiled layers dense and do not demonstrate whole-model compression.

| Compressed layers | Scored tokens | Dense PPL | NEC PPL | KL (nats/token) | Next-token accuracy | Teacher top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–7 | 4095 | 4.08287 | 4.37879 | 0.08902 | 65.08% | 89.04% |
| 0–15 | 4095 | 4.08287 | 5.05345 | 0.23211 | 62.15% | 81.51% |

**All targets demonstrated together: NOT YET.**

The gate requires all 32 layers, independent held-out PPL ≤10, actual
complete-package compression ≥5×, structural reduction ≥2×, and exported
runtime parity. No gradient-based healing is used. Runtime reconstructs
dense matrices, so storage reduction does not imply runtime memory savings.
