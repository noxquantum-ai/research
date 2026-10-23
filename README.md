NEC (Nox Entanglement Compiler) is our post-training, quantum-inspired structural compression system for pretrained language models. It uses tensor networks and related mathematical techniques to reduce the number of independent model parameters without post-compression training.

We have demonstrated aggressive structural reductions, including configuration-derived estimates of roughly 7–11×, but those representations severely degrade model capability. Preserving accuracy and functionality at substantial parameter reductions remains NEC’s central unresolved challenge.

## Latest dual-budget NEC result

A newer sparse-excitation hybrid exactly reproduced its eight-layer result
on a replacement A40 pod and has now reached **16 compressed layers at PPL
5.053 versus dense 4.083**, KL **0.232**. These are matched development-slice
measurements through the complete original architecture, with uncompiled
layers still dense. Extension to all 32 layers is in progress; actual
whole-model compression and final held-out quality for this candidate are
not yet established. See the [live measured checkpoints](results/nec_sparse_excitation_a40_20261023/live/README.md)
and [trial protocol](results/nec_sparse_excitation_a40_20261023/README.md).

NEC now tracks actual serialized storage and logical coefficient reduction as
separate constraints. This corrected an important weakness in the packed
Schmidt path: its completed 5× byte-target model stores 5.614B total logical
coefficients and therefore achieves only **1.20× structural reduction**.

The revised 32-layer compiler now produces a complete package satisfying
**6.286× actual serialized compression and 2.023× structural reduction**.
It compresses attention and MLP in every layer, retains zero transformer
layers dense, and uses no gradient fine-tuning or healing. Quality still
fails decisively, so the combined objective is **not achieved**.

| Full 32-layer result, 2026-10-23 | Dense reference | NEC |
| --- | ---: | ---: |
| Complete serialized model | 13,479,247,511 bytes | **2,144,321,837 bytes** |
| Total stored coefficients, including all factors and scales | 6,738,417,664 | **3,331,706,294** |
| Whole-model storage compression | 1× | **6.286×** |
| Whole-model structural reduction | 1× | **2.023×** |
| Validation-slice PPL, 4,095 scored tokens | 4.083 | **320.234** |
| Teacher KL, nats/token | 0 | 4.363 |
| Next-token accuracy | 66.37% | 17.36% |
| Dense top-1 agreement | 100% | 21.03% |
| Combined quality / storage / structural acceptance | Reference | **INFEASIBLE: quality fails** |

The complete deployable directory includes retained weights, NEC artifacts,
metadata, tokenizer files and loader code. Its runtime reconstructs dense
weights; these storage ratios do not imply equivalent runtime memory savings.
The remaining 24 layers compiled in 106 seconds with eight checkpoints reused.

Eight-layer development improved PPL from 254.05 to 51.64 through an
activation-sensitive coordinate defect and residual capacity allocation, but
the full model still accumulates severe errors. In matched 32-layer test-slice
ablations, attention alone scores PPL 27.93 and MLP alone 226.11. Exported
logits match checkpoint injection exactly. Full-model
accuracy preservation at 5–10× storage and 2–3× structural compression remains
NEC's unresolved objective.

See the [dual storage/structural budget report](results/nec_structural_budget_a40_20261023/README.md).

## Latest full-model allocation run

The 2026-10-22 A40 run applied the adaptive allocator to all 32 Llama-2-7B transformer layers. It compressed 28 layers and retained four dense layers, but the resulting model failed the quality target. Ratios below are measured from serialized files, not estimated coefficient counts.

| Evaluation | Result | Notes |
| --- | ---: | --- |
| Allocation | NEC 0, 4–30; FP16 dense 1–3; INT8 dense 31 | 28 NEC-compressed, 4 dense |
| Full WikiText-2 test PPL | 712.18 | 341,468 scored tokens; dense teacher 4.85948 |
| Teacher-student KL | 4.99486 nats/token | Full WikiText-2 test split |
| Next-token accuracy / teacher top-1 agreement | 12.79% / 15.59% | Same full test evaluation |
| Whole deployable package | 3.907 GB vs 13.479 GB dense; **3.45×** | Includes weights, metadata, tokenizer and loader files |
| Transformer-layer region | **3.83×** | Actual serialized allocation vs dense FP16 bytes for layers 0–31 |
| MMLU 5-shot diagnostic | 14/57 (**24.56%**) | One test question per subject; sample only, not the full benchmark |
| PPL ≤ 10 / 10× compression goal | **Failed / not met** | Allocation marked infeasible; this is not a successful quality-compression result |

The full MMLU run was stopped at 97/56,168 likelihood requests; the harness then estimated 9 h 32 min remaining, far above the requested 30-minute limit. No full MMLU score was produced. HellaSwag was not run beyond an adapter smoke check. The one-question-per-subject MMLU result is only a quick diagnostic and must not be compared as if it were the published full benchmark. The detailed report, run logs, allocation, and benchmark attempt are in [the experiment archive](results/nec_full32_adaptive_allocation_a40_20261022/README.md).
