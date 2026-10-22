# NEC full-model adaptive allocation result (A40, 2026-10-22)

## Outcome

The complete 32-layer Llama-2-7B model was evaluated with the selected mixed allocation on the full WikiText-2 test split. The run completed all 341,468 scored tokens (341,469 input tokens), using context 2,048 and stride 512.

**The allocation is infeasible for the requested PPL limit of 10.** It meets the maximum of four non-NEC transformer layers, but its held-out PPL is 712.182. The dense teacher scored PPL 4.859 on the same test tokens.

| Metric | Measured result |
| --- | ---: |
| Selected allocation | NEC layers 0, 4–30; FP16 dense layers 1–3; INT8 layer 31 |
| NEC-compressed layers | 28 / 32 |
| Retained dense layers | 4 / 32 (3 FP16, 1 INT8) |
| Dense teacher PPL | 4.85948 |
| Mixed model PPL | 712.18235 |
| Mean NLL | 6.56833 |
| Teacher-student KL (nats/token) | 4.99486 |
| Next-token accuracy | 0.12791 |
| Dense top-1 agreement | 0.15586 |
| PPL <= 10 | **Failed** |
| Actual deployable package size | 3,907,403,530 bytes |
| Dense FP16 source package size | 13,479,247,511 bytes |
| Actual whole-package ratio | 3.44967x |
| NEC-region weight-storage ratio | 3.83200x |
| Runtime | 2,675.75 seconds |

The whole-package ratio counts the serialized deployable package, including dense tensors, NEC and INT8 artifacts, metadata, tokenizer files, and loader code. The 10x aspiration was not met. A ratio above 3x does not make this a successful quality-compression result.

## Selection versus held-out evaluation

The allocator selected the configuration after exact refinement on 4,095 tokens, where it measured PPL 666.170 and teacher KL 4.84802. The independent full test evaluation measured PPL 712.182 and KL 4.99486. Neither result meets the PPL constraint.

The allocation search used a single-layer Fisher interaction proxy and refined a bounded shortlist; it did not exhaustively search all allocations. See `report.json` for search protocol, hashes, per-layer probes, candidate metrics, storage accounting, and feasibility status. `run.log` preserves the evaluator's completion trace.

## Accuracy spot check and public benchmarks

A short MMLU 5-shot diagnostic scored the first test question from each of the 57 subjects:

| Evaluation | Result | Status |
| --- | ---: | --- |
| MMLU 5-shot spot check | 14/57 correct (**24.56%**) | Diagnostic only; one example per subject |
| MMLU full test | No score | Stopped at 97/56,168 likelihood requests; harness estimated 9 h 32 min remaining, above the 30-minute limit |
| HellaSwag full validation | No score | Only an 8-example adapter smoke check was run |

The MMLU sample is too small to represent the benchmark and is not a valid comparison to the published Llama 2 7B full-set score of 45.3% (5-shot, [Llama 2 paper](https://arxiv.org/abs/2307.09288)). The standard harness run used the pinned MMLU dataset and 5-shot protocol; it was stopped before producing a full result. `mmlu_sample.json` and `mmlu_sample.log` preserve the diagnostic; `mmlu_full_attempt.log` and `public_benchmark_runner.py` preserve the aborted full-run attempt. No full HellaSwag score was produced.

The selected package is 3.7 GiB on the GPU host. It was not copied to the local workspace because that filesystem had only 1.5 GiB free at the time of transfer; the model package is intentionally not included in Git.

## Reproducibility

- Model: `NousResearch/Llama-2-7b-hf`
- Source code commit used: `c6968cd53e21b48782629ba60afa00e995d77e64`
- GPU: NVIDIA A40
- Report status: `COMPLETE`
