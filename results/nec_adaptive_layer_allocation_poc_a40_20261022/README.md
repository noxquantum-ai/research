# NEC adaptive layer allocation proof of concept

## Result

The bounded allocator recovered the previously measured best pair from the
four available NEC checkpoints. It compressed layers 0 and 3, matching the
rank-1 allocation in the prior 4,095-token WikiText-2 validation sweep. The
search never loaded that sweep or its scores.

| Stage / allocation | Data | PPL | Teacher KL | Use |
| --- | --- | ---: | ---: | --- |
| Proxy only: layers 0, 1 | WikiText-2 test, 4,095 scored tokens | 8.1204 | 0.5076 | Failed to recover known best |
| Exact refine: layers 0, 1 | WikiText-2 train, separate 4,095 scored tokens | 8.5873 | 0.4900 | Proxy shortlist finalist |
| Exact refine: layers 0, 3 | WikiText-2 train, separate 4,095 scored tokens | 6.8938 | 0.2607 | Selected finalist |
| Refined choice: layers 0, 3 | WikiText-2 test, 4,095 scored tokens | 6.3184 | 0.2592 | Independent check |

The old validation answer key gives layers 0+3 PPL 5.079 (rank 1) and layers
0+1 PPL 7.497 (rank 4). Against that local answer key, the refined selection
has zero PPL regret; the proxy-only selection has 2.418 PPL regret. On the new
held-out test slice, 0+3 improves on 0+1 by 1.802 PPL. The test slice is the
same in both runs.

Full answer key from the prior four-layer sweep:

| Compressed layers | Validation PPL | Rank |
| --- | ---: | ---: |
| 0, 3 | 5.079 | 1 |
| 0, 2 | 5.908 | 2 |
| 1, 3 | 7.016 | 3 |
| 0, 1 | 7.497 | 4 |
| 1, 2 | 7.543 | 5 |
| 2, 3 | 7.805 | 6 |

## Search cost and compression

There were six possible two-layer allocations. Proxy search scored all six
using its quadratic Fisher model, which requires no multi-layer model
forwards. Exact refinement evaluated only the top two proxy candidates, 0+1
and 0+3. The remaining four pairs were not run. Refinement measured both
candidates on 4,095 tokens and chose by measured PPL under the PPL 10 limit,
using KL as a tie-breaker.

The run performed five model configurations for the dense and four singleton
probes on 1,023 scored tokens. Refinement used a dense reference and two
candidate configurations on 4,095 scored tokens. Held-out evaluation used a
dense teacher pass and one selected-student pass on 4,095 tokens. Total wall
time was 94.8 seconds on an NVIDIA A40, including 52.2 seconds to load the
model. Proxy probes took 7.1 seconds; exact refinement took 14.7 seconds.

The two compressed checkpoints save 669,619,324 serialized weight bytes
against the four candidate layers' dense projection and norm tensors (41.4%
of that region). The other 28 layers remain dense. This is storage savings for
the four-layer candidate region, not a claim of whole-model 41.4% compression
or an inference speedup.

## Interpretation

The singleton proxy alone did not pick the best pair. It placed 0+1 just
ahead of 0+3 because their estimated KL values were close. Measuring only the
top two pairs on a separate selection slice corrected the choice, and the
selected pair remained below PPL 10 on a separate WikiText-2 test slice.

This validates the shortlist-plus-refinement approach on this four-checkpoint
case. It does not validate 32-layer allocation or prove that two exact
candidate evaluations will be enough in larger search spaces. It also does
not prove that the answer-key ranking generalizes beyond the measured data.

## Artifacts

- `proxy_only_report.json`: first run, including proxy ranking, singleton
  costs, and held-out results.
- `proxy_only_run.log`: first run console log.
- `refined_report.json`: refinement run with token hashes, exact metrics,
  measured candidate list, checkpoint hashes, timings, and held-out metrics.
- `refined_run.log`: refinement run console log.

The proxy-only run used source commit `e933d195`; refinement used
`d49bd141`. Both used the saved checkpoints in
`/workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg`.
