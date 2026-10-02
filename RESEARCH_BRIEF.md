# Research brief — 2026-10-02

## The question

Can a compact representation of a **pretrained** language model preserve behavior across several tasks better than strong quantization and matrix methods at the **same measured memory and latency**? If so, what contribution comes specifically from quantum information ideas?

The first study should answer a narrower version: on one open model and one target device, can a tensor-network method improve the quality–memory–latency frontier? A representation that only reduces parameter count, checkpoint size, or reconstruction error does not establish that result.

## What already exists

| Work | What it establishes | Implication for Nox |
| --- | --- | --- |
| [CompactifAI paper](https://arxiv.org/abs/2401.14109) and [Multiverse US application US20250111212A1](https://patents.google.com/patent/US20250111212A1/en) | Multiverse has published tensor-network compression of pretrained LLM weights. The application describes decomposing layer matrices into tensor networks; dependent claims include tensor-star structure and a random search over permutations. | “Tensor networks for LLM compression” is already occupied as a research description. Study the actual claims before any public invention claim. A patent application is not proof that every tensor method is covered or that a patent has been granted. |
| [TensorGPT](https://arxiv.org/abs/2307.00526) | Tensor-train compression of language-model embeddings, with evaluation including a Raspberry Pi. | Embedding tensorization alone is existing work. |
| [HEAT](https://arxiv.org/abs/2211.16749) | Hardware-aware tensor decomposition, contraction scheduling, and distillation for Transformers. | Hardware-aware selection and fast contractions are existing research directions. |
| [Minima preprint](https://arxiv.org/abs/2602.01613) | Reports sensitivity-based selection among Tucker, tensor-train, and tensor-ring decompositions with custom kernels and model-level latency measurements. | “Choose tensor type by layer sensitivity, then write kernels” is insufficient as a novelty claim. Treat the reported results as author claims until reproduced. |
| [ASVD](https://arxiv.org/abs/2312.05821) and [SVD-LLM](https://arxiv.org/abs/2403.07378) | Activation-aware and truncation-aware matrix factorization for LLMs. | Compare against a strong factorization baseline, not plain truncated SVD alone. |
| [QuIP#](https://arxiv.org/abs/2402.04396) and [AQLM](https://arxiv.org/abs/2401.06118) | Strong low-bit quantization research, with fast inference implementations. | A 4-bit GPTQ/AWQ baseline is practical, but at 2–3 bits the comparison should include a credible stronger method that runs on the target hardware. |

**Patent note.** The published [US application](https://patents.google.com/patent/US20250111212A1/en) lists a 2023-10-02 priority date and, in the patent database checked on 2026-10-02, a pending status. Multiverse has other related publications, including a [2026 hybrid quantum-classical training application](https://patents.google.com/patent/EP4733989A1/en). Filing, publication, grant, jurisdiction, and claim scope are different questions. This document is a research map, not a freedom-to-operate opinion; obtain patent counsel before relying on a legal boundary.

## Candidate mechanism to test

**Working hypothesis:** the useful compression budget is determined by how a weight operator acts on *actual activations* across several domains, and by how strongly chosen groups of its input/output indices couple. Tensor-network tools can expose that coupling through **Schmidt spectra** at candidate partitions. The spectrum may help choose index ordering and bond dimensions within a memory and latency budget.

For a linear layer with weight matrix `W`, input `x`, and compressed operator `W_hat`, define the calibration covariance `C = E[xxᵀ]`. Then

`E ||(W − W_hat)x||² = ||(W − W_hat) C^(1/2)||_F²`.

This equality gives a defensible layer-local objective. Build `C` from a balanced mixture of calibration domains, or use a diagonal/low-rank approximation when a full covariance is too expensive. Tensorize the activation-weighted operator under candidate index orderings. At each cut, singular values form a Schmidt spectrum; discarded squared values estimate approximation error for that cut. Spend bond dimension where the spectrum decays slowly and the layer is behaviorally sensitive. Directly contract the factors with activations at inference.

This is **a proposed combination to test, not an established new algorithm**. Activation-aware compression, tensor-train decompositions, permutation search, and hardware-aware rank selection each have prior art. The research contribution would have to be a specific method and measured result that existing approaches do not obtain.

Two important caveats:

1. “Entanglement” here is an analogy for a classical operator reshaped as a tensor. Its spectrum depends on index grouping and basis; it is not an intrinsic quantum property of the model or evidence of quantum advantage.
2. Low local error does not guarantee retained generation quality. The method must survive model-level, task-level, and latency evaluation. Many small contractions can be slower than one optimized dense or quantized matrix multiplication.

## The strongest near-term claim Nox could earn

> On **[named model]** running on **[named device]**, our **[specified operator]** produced **[quality result on held-out tasks]** at **[measured resident/peak memory]** and **[measured latency/throughput]**. We compared it with **[named quantization]** and **[named matrix method]** under the same runtime and calibration budget. The improvement came from **[ablation showing the quantum-information-inspired choice matters]**.

Until the bracketed fields are filled from reproducible runs, this is a hypothesis. A narrower result is still useful: for example, showing that activation-weighted Schmidt spectra predict which layers can be compressed, even if the executable representation fails the latency gate. That would support a new direction, but not the full efficiency claim.

## Sources to revisit during implementation

- [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct): candidate open model, Apache 2.0 license listed on the model card.
- [lm-evaluation-harness configuration guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md): reproducible task configuration and seeds.
- [AQLM paper](https://arxiv.org/abs/2401.06118) and [QuIP# paper](https://arxiv.org/abs/2402.04396): low-bit comparisons and implementation constraints.
- [African Union Continental AI Strategy](https://au.int/sites/default/files/documents/44004-doc-EN-_Continental_AI_Strategy_July_2024.pdf), section 2.4.3.1: regional compute motivation.
