# NEC full-model storage preflight

## Scope and constraint

This preflight covers **all 32 transformer layers** in the Llama-2-7B
checkpoint. No layer outside that set is treated as an allocation candidate.
At most **4 layers may remain dense**; NEC may compress more than the required
28 layers. The quality evaluation uses the complete original Llama-2-7B.

The source model directory contains **13,479,247,511 serialized bytes**. The
32 candidate layers contain **12,952,543,232 dense tensor-payload bytes**;
the remaining model files, embeddings/head, other tensors, and source metadata
account for about **526,704,279 bytes**. The whole-model estimate preserves
all source bytes except the dense payloads replaced by NEC and adds the actual
or estimated serialized `.pt` checkpoints. It is an optimistic lower bound
because the mixed package has not been rebuilt and its new shard/index
overhead has not been measured.

Only checkpoints for layers 0–3 currently exist. Their serialized sizes range
from **69,815,746 to 70,098,370 bytes**, averaging **69,989,650 bytes**. The
other 28 artifact sizes below use this measured min/mean/max as estimates.

## Estimated whole-model frontier

| Compressed layers | Dense layers | Estimated whole-model ratio range | Estimated mixed bytes at mean artifact size | Meets 3× diagnostic? | Meets 10×? |
|---:|---:|---:|---:|:---:|:---:|
| 28 | 4 | 3.281–3.287× | 4,105,489,551 | Yes | No |
| 29 | 3 | 3.572–3.580× | 3,770,712,481 | Yes | No |
| 30 | 2 | 3.919–3.929× | 3,435,935,411 | Yes | No |
| 31 | 1 | 4.342–4.354× | 3,101,158,341 | Yes | No |
| 32 | 0 | 4.866–4.882× | 2,766,381,271 | Yes | No |

All ratios are preflight upper bounds, not measured package ratios or quality
results. The 3× column is diagnostic only; 3× is insufficient as a success
target. The approximately 10× ambition remains visible.

## Feasibility finding

- **10× is not feasible with the current NEC artifact sizes.** For a 10× ratio
  against the complete 13,479,247,511-byte model, the final package may contain
  at most 1,347,924,751 bytes. Even if all 32 layer projections are
  compressed, that leaves an average checkpoint budget of about **25,662,884
  bytes per layer** before rebuilt-package overhead. The existing mean is
  69,989,650 bytes. The best estimated ratio with today's artifacts is about
  **4.87×**, even with no dense transformer layers.
- The **at-most-four-dense-layer constraint and a ratio above 3× are
  storage-feasible** in the optimistic preflight estimates. Keeping exactly
  four dense layers yields about 3.28×; retaining fewer improves the estimate
  up to about 4.87× if all 32 checkpoints have the observed mean size.
- Whether any feasible allocation meets PPL `<= 10` is **unknown**. Quality,
  teacher KL, and prediction accuracy require full-model evaluation after the
  remaining checkpoints are compiled.
- Only four checkpoints exist today. The estimates are useful for the GPU
  cost gate but cannot substitute for measured files or a serialized mixed
  model package.

The earlier 16-layer-only plan is superseded. The A40 is idle, and the
interrupted attempt at layer 4 produced no new checkpoint. Resuming through
layer 31 is estimated at roughly **57 additional GPU-hours** using the prior
12-layer, 24.5-hour projection. This run can test whether the quality target
overlaps the current 3.28–4.87× frontier; it cannot meet 10× without a smaller
NEC representation.

## Reproduction

```bash
python scripts/preflight_nec_compression_targets.py \
  --candidate-layers 0-31 \
  --scope-layers 0-31 \
  --max-dense-layers 4 \
  --whole-model-storage \
  --compression-targets 3 10
```
