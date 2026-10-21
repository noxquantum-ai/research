# One-projection WikiText-2 language-model evaluation

## Result and scope

This run evaluates the saved layer-0 `mlp.down_proj` compression stages inside
one pinned, otherwise dense Llama-2-7B model. It is a causal language-model
measurement, not a full-model NEC evaluation: only this one projection is
replaced at a time. The experiment does not establish whole-model perplexity,
MMLU retention, or inference-speed/storage benefits.

The delivered serialized NEC stage raises full-test PPL from **4.85948** for
the matched dense model to **4.94839**. Mean NLL rises by **0.018131 nats per
target**; the paired PPL ratio is **1.01830** (95% contiguous-block bootstrap
interval **[1.01715, 1.01958]**). Top-1 next-token accuracy changes from
**62.7687%** to **62.4258%** (−0.3429 percentage points).

The `down_proj` layer-output error previously measured at about 42% therefore
does not translate into a comparably large language-model loss change for this
single layer. Conversely, the small LM-loss change here does not show that
errors will remain small when many compressed layers are composed.

## Matched stage results

All rows score the same **341,468** target tokens from the pinned raw WikiText-2
test split. “10k prefix” scores the first 9,999 targets (10,000 input tokens)
from the full-test run. Causality makes those prefix logits independent of
future tokens; the prefix/full causal-window equivalence is covered by a
synthetic regression test. The full-test paired interval is shown for each NEC
stage.

| Layer-0 `down_proj` stage | Full-test PPL | 10k-prefix PPL | Top-1 accuracy | PPL ratio vs. dense (95% block-bootstrap CI) |
|---|---:|---:|---:|---:|
| Dense reference | 4.85948 | 5.29195 | 62.7687% | 1.00000 |
| Sparse shield | 4.98941 | 5.51274 | 62.2410% | 1.02674 [1.02526, 1.02829] |
| Sparse + bridge | 4.97296 | 5.49151 | 62.3189% | 1.02335 [1.02194, 1.02490] |
| TT warm start | 4.95392 | 5.45917 | 62.4003% | 1.01943 [1.01820, 1.02077] |
| DMRG sweep 1 | 4.94991 | 5.44514 | 62.4120% | 1.01861 [1.01743, 1.01991] |
| DMRG sweep 2, pre-Q4 | 4.94800 | 5.43880 | 62.4228% | 1.01821 [1.01706, 1.01950] |
| Serialized NEC artifact | 4.94839 | 5.43759 | 62.4258% | 1.01830 [1.01715, 1.01958] |

NLL and PPL both improve across the DMRG stages relative to the TT warm start
on this language-model metric, though the final stage remains worse than the
dense reference. The final Q4 transition changes full-test PPL from 4.947995
to 4.948392 (about **0.0080%**); most of the measured PPL gap is already
present before final MPO Q4. This is consistent with the separate projection
waterfall, where the float MPO already has large held-out output error.

The bootstrap resamples paired, contiguous 512-target blocks (667 blocks for
the full test split; 5,000 replicates; seed 20261021) and reports a percentile
95% interval for mean NLL difference and its exponentiated PPL ratio. This is a
descriptive local-dependence adjustment on one fixed corpus, not an IID or
cross-dataset generalization guarantee. The 10k-prefix interval uses 20 blocks
and is correspondingly less stable. Per-token NLL and top-1 prediction sidecars
are included next to `single_projection_ppl/result.json`; their SHA-256 hashes
are recorded in that JSON and were verified after transfer.

## Reproduction and provenance

- Model and tokenizer: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Tested projection: layer 0 MLP `down_proj`, matrix shape `[4096, 11008]`.
- Dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, `test` split; source SHA-256
  `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
- Text construction: raw split rows joined with two newlines; pinned Llama
  tokenizer with special tokens enabled; input begins with BOS.
- Evaluation: context 2,048, stride 512, batch size 1, cache disabled,
  FP16 model/SDPA, FP32 shifted-token cross entropy; every target after the
  initial BOS is scored exactly once. Each stage's token-ID hash is identical.
- Primary result: first 10,000 input tokens / 9,999 scored labels. Full test:
  341,469 input tokens / 341,468 scored labels.
- Calibration artifact provenance: full-matrix High-Fidelity qualification,
  compiler source `e5f69f1ec3ddb6a433f8aa9cb4b856d619c47063`; exact NEC
  settings and stage-file hashes are in
  [`single_projection_ppl/result.json`](single_projection_ppl/result.json)
  and the original qualification report at
  [`../nec_core_q4_attribution_a40_20261021/qualification/report.json`](../nec_core_q4_attribution_a40_20261021/qualification/report.json).
- Evaluation code commit: `79774d01e52c2c3c566feca144b5391b82dc96ed`.
- Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1,
  Accelerate 1.14.0; NVIDIA A40 (46,068 MiB). Peak PyTorch allocated VRAM
  was 14,599,479,808 bytes (about 13.6 GiB); peak reserved was
  15,133,048,832 bytes (about 14.1 GiB).
- The seven-stage command took 2,075 seconds wall time, including model load
  and finalization; RunPod's recorded GPU-only estimate was **$0.3401** at
  $0.59/hour. No timeout was set.

On the RunPod pod with the saved network-volume stages and the pinned
qualification environment, rerun with:

```bash
cd /workspace/research
export NEC_STREAMING_QUAL_ROOT=/workspace/runs/nec-core-q4-attribution-a40_20261021
export NEC_STAGE_PPL_ROOT=/workspace/runs/nec-core-q4-attribution-a40_20261021
export NEC_EVAL_GPU_MEMORY_GIB=32 NEC_EVAL_CPU_MEMORY_GIB=24
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh projection-stage-ppl
```

The verbose main log, 70-record heartbeat log, wrapper exit/cost record, full
result JSON, and 14 per-token sidecars are preserved in this directory.
`single_projection_ppl/progress.json` is the final resumable stage checkpoint
captured immediately before finalization; `result.json` and
`single_projection_ppl/last_execution.json` are the authoritative COMPLETE/PASS
records.
