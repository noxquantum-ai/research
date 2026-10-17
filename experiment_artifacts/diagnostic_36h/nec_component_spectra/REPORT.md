# NEC component Schmidt-spectrum diagnostic

## Scope

This is a read-only test of the historical High-Fidelity layer-15 Q artifact.
It asks whether NEC's sparse shield, low-rank bridge, and diagonal scale make
the residual easier to represent at the existing four-site MPO ranks.  It does
not fit a new model, use validation data, or claim that one projection explains
the full-model failure.

The dense weight and saved artifact are available locally and are identified
by SHA-256 in `legacy_high_fidelity_layer15_q.json`.  The audit uses the actual
NEC layout `(8,8,8,8)` on both axes, site order `(0,1,2,3)`, and ranks
`(1,64,128,64,1)`.  The two 64-dimensional boundary spectra are exact FP64
Gram eigenspectra.  The 4096-by-4096 middle spectrum uses a randomized
range-SVD with 256 reported values, oversampling 32, two power iterations, and
seed 42; unresolved energy remains in the tail rather than being renormalized
away.

## Result

| Matrix supplied to the matched cuts | Middle-cut discarded energy at rank 128 |
|---|---:|
| `W` | 0.884217 |
| `W - S` | 0.890661 |
| `W - S - AB` | 0.890568 |
| `(W - S - AB) D` | 0.890634 |

Subtracting the sparse shield increases the measured rank-128 tail by 0.006444
(0.644 percentage points).  The bridge then changes it by -0.000093 and the
diagonal scale by +0.000066.  Repeating the randomized middle-cut calculation
with seeds 43 and 44 gave sparse deltas of +0.006445 and +0.006475, bridge
deltas of -0.000077 and -0.000072, and scale deltas of +0.000078 and +0.000048.

For this projection, the sparse/bridge/scale sequence therefore does not reduce
the central residual tail at NEC's configured rank; the sparse step makes it
slightly worse and the other changes are negligible at the resolution of this
audit.  The result does not establish that subtraction must increase Schmidt
rank in general, nor does it establish the same behavior for other projections
or layers.  Gate, Up, and Down require their distinct `(43,4,8,8)` boundary
layout and must be measured separately.

## Reproduction

```bash
python3 -u scripts/audit_nec_schmidt_components.py \
  --checkpoint compressed_models/tiers/high_fidelity_10x/compressed_layer_15.pt \
  --dense-weight models/Llama-2-7b-hf/layer_15_q_proj.pt \
  --projection q_proj \
  --output experiment_artifacts/diagnostic_36h/nec_component_spectra/legacy_high_fidelity_layer15_q.json \
  --device cpu --seed 42 --max-rank 256 --oversample 32 --power-iterations 2
```

The output path is create-only to prevent accidental replacement of an
existing measurement.
