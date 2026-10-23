NEC (Nox Entanglement Compiler) is our post-training, quantum-inspired structural compression system for pretrained language models. It uses tensor networks and related mathematical techniques to reduce the number of independent model parameters without post-compression training.

We have demonstrated aggressive structural reductions, including configuration-derived estimates of roughly 7–11×, but those representations severely degrade model capability. Preserving accuracy and functionality at substantial parameter reductions remains NEC’s central unresolved challenge.

## Latest dual-budget NEC result

NEC now tracks actual serialized storage and logical coefficient reduction as
separate constraints. This corrected an important weakness in the packed
Schmidt path: its completed 5× byte-target model stores 5.614B total logical
coefficients and therefore achieves only **1.20× structural reduction**.

The revised compiler enforces at least 2× structural reduction while retaining
the 5–10× storage objective. On an eight-layer Llama-2-7B prefix, a balanced
Schmidt cut plus a 5% activation-sensitive defect branch improved PPL from
254.05 to 67.13 at a conservative **2.0036× structural / 5.2566× serialized**
lower bound. This remains a failed quality result, not a successful compressed
model.

| NEC dual-budget checkpoint | Compressed layers | PPL | KL | Structural reduction | Serialized compression | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Byte-only packed Schmidt | 32 | 95.27 | 2.981 | 1.200× | 5.002× projected | Fails quality and structural gates |
| 2× semantic Schmidt | 8 | 481.72 | 4.616 | ≥2.0005× | ≥5.2567× | Fails quality gate |
| 2× balanced Schmidt | 8 | 254.05 | 3.954 | ≥2.0003× | ≥5.2552× | Fails quality gate |
| 2× balanced + 5% defect | 8 | **67.13** | **2.652** | **≥2.0036×** | **≥5.2566×** | Best structural run; still fails PPL ≤ 10 |

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
