# Live hybrid NEC trial reports

Snapshot phase: `finished`; UTC: 2026-10-23T20:56:20.549992+00:00.

These are measured checkpoints from the new pod. Partial checkpoints retain
all uncompiled layers dense and do not demonstrate whole-model compression.

| Compressed layers | Scored tokens | Dense PPL | NEC PPL | KL (nats/token) | Next-token accuracy | Teacher top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–7 | 4095 | 4.08287 | 4.37879 | 0.08902 | 65.08% | 89.04% |
| 0–15 | 4095 | 4.08287 | 5.05345 | 0.23211 | 62.15% | 81.51% |
| 0–23 | 4095 | 4.08287 | 5.99404 | 0.39975 | 59.46% | 76.02% |
| 0–31 | 4095 | 4.08287 | 6.84527 | 0.53076 | 57.97% | 73.02% |

Fresh held-out test range: 308,700 scored tokens; dense PPL 4.86728; NEC PPL 8.34896; KL 0.54395 nats/token.

Actual complete package: 2,641,936,119 bytes; 5.10203× storage compression; 2.02779× structural reduction.

**All targets demonstrated together: YES.**

The gate requires all 32 layers, independent held-out PPL ≤10, actual
complete-package compression ≥5×, structural reduction ≥2×, and exported
runtime parity. No gradient-based healing is used. Runtime reconstructs
dense matrices, so storage reduction does not imply runtime memory savings.
