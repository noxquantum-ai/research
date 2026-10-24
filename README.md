NEC (Nox Entanglement Compiler) is our post-training, quantum-inspired structural compression system for pretrained language models. It uses tensor networks and related mathematical techniques to reduce the number of independent model parameters without post-compression training.

The latest measured research package improves the full Llama-2-7B result to **5.568× actual whole-model storage compression, 2.028× structural coefficient reduction, and WikiText-2 test-regression PPL 8.187**. The published baseline remains unchanged at 5.413× and PPL 8.349. The research candidate uses training-guarded NEC operator selection and an explicitly reconstructed FP16 output head from a closed-form solve; it adds no inference coefficients. NEC compresses attention and MLP in every transformer layer without gradient-based fine-tuning or healing. Capability scores show substantial accuracy loss. The current objective is full-model PPL ≤5.867 (updated user gate, October 24), next-token accuracy of 54–62% and at least 85% teacher top-1 agreement (updated user gates), while maintaining or improving the current storage and structural compression. These tighter quality targets are not met; the aspirational 10× storage target also remains open.

## Latest complete NEC result

The representation combines a Schmidt/MPO tensor-network backbone, sparse residual coordinate amplitudes, and analytical activation-aware construction. It is a hybrid representation; tensor networks, sparsity and reduced-precision serialization all contribute. Every stored core, amplitude and learned scale counts toward structural size. Complete package storage includes retained embeddings, output head, norms, support masks, metadata, tokenizer and decoder code.

| Full 32-layer result | Dense reference | Published NEC | Research candidate, 2026-10-24 |
| --- | ---: | ---: | ---: |
| Complete serialized model | 13,479,247,511 bytes | 2,490,273,288 bytes | **2,420,734,087 bytes** |
| Total stored weight/factor coefficients | 6,738,417,664 | 3,323,032,892 | **3,323,032,892** |
| Actual whole-model storage compression | 1× | 5.413× | **5.568×** |
| Structural coefficient reduction | 1× | 2.028× | **2.028×** |
| WikiText-2 test-regression PPL | 4.8673 | 8.3490 | **8.1873** |
| Teacher KL, nats/token | 0 | 0.54395 | **0.52083** |
| Next-token accuracy | 62.82% | 54.00% | **54.40%** |
| Dense top-1 agreement | 100% | 71.03% | **71.92%** |
| Dense transformer layers retained | 32 | 0 | **0** |
| Current dense-quality, accuracy and agreement targets | Reference | Not met | **Not met** |

Quality uses 308,700 scored test tokens after offset 32,768, context 2,048 and stride 512. This is the remaining test range, not the entire test split. It is now a known regression range; candidate selection uses construction-training guards, not these test tokens. Dense and NEC use identical token IDs. The package loader reproduces checkpoint-injection logits bit for bit. This establishes the stated PPL/storage/structural gates, not dense-equivalent capability or deployment speed.

Before lossless container compression, the complete package is 2,641,936,119 bytes (**5.102×**). Lossless gzip containers preserve decoded tensors and coefficients exactly; the published repository is about **151.7 MB smaller** after including publication files. Model weights remain on the GPU pod and the [private Hugging Face repository](https://huggingface.co/Fluxx08/NEC-Llama-2-7B-Hybrid), revision `3b6f1f379048e52bb982754cd7e8e144c54cfbe7`.

The current loader reconstructs dense matrices for inference. Storage compression therefore does not establish comparable GPU-memory or throughput improvements. Pure layer construction took 52.43 minutes on an A40; the held-out evaluation took 8.09 minutes. These timings exclude preceding research, data preparation and packaging.

See the [complete experiment report](results/nec_sparse_excitation_a40_20261023/README.md), [measured final reports](results/nec_sparse_excitation_a40_20261023/final/summary.json), and [staged raw evaluations](results/nec_sparse_excitation_a40_20261023/live/README.md).

## Current accuracy work

[Fixed-budget accuracy reconstruction](results/nec_accuracy_recovery_a40_20261024/README.md) is testing covariance-aware construction of NEC's existing Schmidt/MPO and sparse excitation representation, without gradients or additional stored coefficients. A Q/K-only full-model control slightly improved PPL but reduced next-token accuracy and agreement, so it was rejected under the no-regression rule. Two full sequential reconstruction variants failed their quality gates. The latest operator-selected body plus analytical readout reconstruction improves every measured token-quality metric over the published baseline while meeting its storage and coefficient ceilings. Its complete package passed bitwise GPU logit parity and is stored on the network volume. Capability benchmarks have not been rerun, so the older capability scores below do not measure this new candidate. See the linked report for rejected trials and ongoing representation screens.

## Full capability benchmarks

Benchmarking was stopped at the user’s request after full NEC BoolQ and HellaSwag evaluations and a matched dense BoolQ run. Remaining benchmarks are stopped while NEC accuracy work continues. Reference scores below were supplied by the user and appear in [CompactifAI's published comparison](https://arxiv.org/html/2401.14109v2); matching prompts, few-shot settings and scoring protocols have not been established, so these columns are external references rather than matched baselines.

| Benchmark | Original reference | INT4 reference | CompactifAI 88% | CompactifAI 93% | Dense, matched harness | NEC 5.413× |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMLU | 46.41 | 45.53 | 45.32 | 44.16 | Not run | Stopped |
| HellaSwag | 80.55 | 79.25 | 77.87 | 76.54 | Not run | **50.39** |
| BoolQ | 79.76 | 78.19 | 77.90 | 76.77 | **79.36** | **48.50** |
| TriviaQA | 19.03 | 19.00 | 18.33 | 18.10 | Not run | Stopped |
| GSM8K | 23.05 | 22.44 | 22.58 | 17.74 | Not run | Stopped |

All entries are percentages; higher is better. Dense and NEC scores will use the same harness for the matched comparison. CompactifAI columns are external references, since the paper does not establish that its prompts and few-shot settings match ours. Full protocols, sample counts and status are in the [benchmark report](results/nec_full_five_benchmarks_a40_20261023/README.md).

The matched BoolQ run scored 79.36% for dense and 48.50% for NEC on all 3,270 validation questions, a 30.86-point gap. This exposes a substantial capability loss on this task despite the held-out WikiText-2 PPL result. See the [paired full-split records](results/nec_full_five_benchmarks_a40_20261023/progress/).

NEC HellaSwag scored 50.39% length-normalized accuracy on all 10,042 validation questions. Dense HellaSwag was not run, as requested.

## Earlier dual-budget result

The preceding complete model achieved 6.286× storage and 2.023× structural reduction but failed quality with validation-slice PPL 320.234 (dense 4.083). The newer result above supersedes it; [its report](results/nec_structural_budget_a40_20261023/README.md) remains available as a failed historical experiment.

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
