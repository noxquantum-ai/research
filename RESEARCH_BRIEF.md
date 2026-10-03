# Research brief — 2026-10-02

## The question

Can a compact representation of a **pretrained** language model preserve behavior across several tasks better than strong quantization and matrix methods at the **same measured memory and latency**? If so, what contribution comes specifically from quantum information ideas?

The first study should answer a narrower version: on one open model and one target device, which executable representation, if any, improves the quality–memory–latency frontier? A representation that only reduces parameter count, checkpoint size, or reconstruction error does not establish that result.

## What already exists

| Work | What it establishes | Implication for Nox |
| --- | --- | --- |
| [CompactifAI paper](https://arxiv.org/abs/2401.14109) and [Multiverse US application US20250111212A1](https://patents.google.com/patent/US20250111212A1/en) | The paper reports up to 93% model-memory reduction on Llama 2 7B, 70% fewer parameters, and roughly 2–3% accuracy drop across five tasks. Its tensorized checkpoints were “healed” with less than one epoch of additional training on eight A10G GPUs; the 93% point also quantizes non-tensorized layers to 4-bit. Its inference speed table is a distributed model-parallel forward-pass benchmark, not a single-device end-to-end token-generation comparison. The application describes decomposing pretrained layer matrices into tensor networks; dependent claims include tensor-star structure and a random search over permutations. | Tensor-network compression is established and has substantial reported results. The comparability gap leaves room for a stricter Nox study: same device, current executable quantization and matrix baselines, matched measured memory and latency, and held-out multi-task quality. Study actual patent claims before any public invention claim; an application does not prove that every tensor method is covered or that a patent has been granted. |
| [TensorGPT](https://arxiv.org/abs/2307.00526) | Tensor-train compression of language-model embeddings, with evaluation including a Raspberry Pi. | Embedding tensorization alone is existing work. |
| [HEAT](https://arxiv.org/abs/2211.16749) | Hardware-aware tensor decomposition, contraction scheduling, and distillation for Transformers. | Hardware-aware selection and fast contractions are existing research directions. |
| [Minima preprint](https://arxiv.org/abs/2602.01613) | Reports sensitivity-based selection among Tucker, tensor-train, and tensor-ring decompositions with custom kernels and model-level latency measurements; its Qwen3-32B results report 64→40 GiB peak VRAM and 40→50 tokens/s at 8K context on an A100, relative to a dense baseline. | “Choose tensor type by layer sensitivity, then write kernels” is insufficient as a novelty claim. The public abstract/results do not establish a matched Pareto comparison with strong 3–4-bit quantization, so Nox must run that comparison. Treat preprint results as author claims until reproduced. |
| [ASVD](https://arxiv.org/abs/2312.05821) and [SVD-LLM](https://arxiv.org/abs/2403.07378) | Activation-aware and truncation-aware matrix factorization for LLMs. | Compare against a strong factorization baseline, not plain truncated SVD alone. |
| [QuIP#](https://arxiv.org/abs/2402.04396) and [AQLM](https://arxiv.org/abs/2401.06118) | Strong low-bit quantization research, with fast inference implementations. | A 4-bit GPTQ/AWQ baseline is practical, but at 2–3 bits the comparison should include a credible stronger method that runs on the target hardware. |
| [SLIM](https://deepmind.google/research/publications/148040/), [LQER](https://proceedings.mlr.press/v235/zhang24j.html), and [BALANCER](https://www.ijcai.org/proceedings/2026/475) | Combine quantization, sparsity, and low-rank error correction; BALANCER allocates precision/rank under a memory budget using calibration and curvature signals. | These are central non-TN baselines and they already occupy the broad idea of sensitivity/information-guided budget allocation. |
| [RDQ](https://arxiv.org/abs/2607.10137) and [Cross-Layer Error Compensation](https://arxiv.org/abs/2607.14630) | Recent 2026 work models residual-stream drift or accumulated feature error across layers, rather than treating every layer independently. | “Correct accumulated quantization error across depth” is no longer an open claim. Reproduce and compare before extending it. |
| [OnPTQ](https://arxiv.org/abs/2609.36828) | Submitted 2026-09-29, it calibrates on quantized-model trajectories, finds eroded token-decision margins, estimates consequence with short counterfactual branches, and refreshes trajectories after calibration. It reports results on Qwen-based multimodal models. | “Protect important token decisions” and “calibrate on the model’s own prefixes” are now directly occupied directions. Its stated limitations are Qwen-only multimodal evaluation and short-horizon, greedy consequence estimates. |
| [SPEAR](https://arxiv.org/abs/2606.11244) | Adds token-dependent error compensators to selected layers, with fused serving support. | Input-adaptive side corrections are also active work; a generic gated residual adapter is not a novelty claim. |
| [Sigma-Delta Quantization for LLMs](https://journals.sagepub.com/doi/full/10.3233/FAIA251342) | Applies sigma-delta/noise-shaping ideas to very low-bit LLM weights. | “Borrow noise shaping from signal processing” alone is not new. |
| [Quantum Parameter Adaptation (QPA)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7a9d48d13056b6b6a7f5bce1c3ba8f2d-Abstract-Conference.html) | Uses quantum circuits during training to generate classical PEFT/LoRA parameters, then decouples inference from quantum hardware. | Adjacent path for compact adapters, not a direct compact representation of a frozen pretrained model. |

**Patent note.** The published [US application](https://patents.google.com/patent/US20250111212A1/en) lists a 2023-10-02 priority date and, in the patent database checked on 2026-10-02, a pending status. Multiverse has other related publications, including a [2026 hybrid quantum-classical training application](https://patents.google.com/patent/EP4733989A1/en). Filing, publication, grant, jurisdiction, and claim scope are different questions. This document is a research map, not a freedom-to-operate opinion; obtain patent counsel before relying on a legal boundary.

## Candidate directions to screen

The research question does not imply tensor networks. Screen mechanisms by whether they can preserve the model's behavior **and** execute efficiently on the chosen hardware.

| Direction | Quantum-information connection | What to test | Current assessment |
| --- | --- | --- | --- |
| Behavior-level rate–distortion | Quantum fidelity / distinguishability between output states; for ordinary token distributions represented as diagonal density matrices, fidelity reduces to a classical Bhattacharyya/Hellinger measure. | Does output-distribution fidelity on multi-domain prompts allocate a compression budget better than weight error, KL, or curvature signals? | Promising as an evaluation/objective layer, not a representation by itself. The quantum notation may add no advantage over its classical equivalent; test that explicitly. |
| Quantum Fisher / information geometry | Sensitivity of output distributions to parameter changes; quantum Fisher geometry generalizes distinguishability metrics. | Can its practical classical approximation rank layer or channel perturbations better than activation/Hessian methods at equal calibration cost? | Useful hypothesis, but calibration-signal-based global allocation is already active work, including BALANCER. Needs a sharply different estimator or result. |
| Correlation factorization (tensor networks) | Schmidt decomposition and entanglement spectra expose correlations across chosen partitions. | Compare tensorization layouts and bond allocation against quantization, low-rank, and direct serving latency. | Strongest direct link to quantum information and a credible candidate, but crowded and exposed to patent overlap. |
| Quantum-inspired parameter generation | Parameterized circuits generate classical parameters during training. | Could it shrink task-specific adapters while keeping inference classical? | QPA is existing work on PEFT/LoRA adapters, not a frozen base-model representation. Requires training or quantum-simulation resources. |
| Error-correcting / subspace codes | Redundancy can make information robust to corruption; code-space ideas might preserve task-relevant directions under aggressive quantization. | Is there a lightweight, executable code for model subspaces or quantization residuals that improves the accuracy–memory frontier? | Speculative for pretrained LLM weights; no directly comparable result surfaced in this initial search. Defer unless a founder has a concrete code construction. |
| Hybrid quantization + sparse / low-rank correction | No unique quantum link; related to redundancy removal and residual coding. | Reproduce strong baselines such as SLIM/LQER, then test whether a different residual structure improves the frontier. | Essential comparison family. It may be the best practical route even if the eventual method is not quantum-inspired. |

One useful common diagnostic is behavior distortion. Given original and compressed next-token distributions `p` and `q`, compare task quality and KL with `F(p,q) = (Σᵢ √(pᵢqᵢ))²`. For diagonal density matrices this is the usual classical fidelity, so its value would come from better compression decisions or prediction of task regressions—not from calling a classical distribution quantum.

For any linear layer with weight `W`, input `x`, and approximation `W_hat`, activation-weighted local error is `E ||(W − W_hat)x||² = ||(W − W_hat) C^(1/2)||_F²`, where `C = E[xxᵀ]`. That objective can pilot a tensor-network factorization, but it can also evaluate quantized, low-rank, or sparse candidates. Do not let the initial candidate dictate the search space.

Quantum hardware is not assumed. A proposed structure must have a classical executable form on the target device and beat or complement strong baselines there. “Quantum-inspired” describes the source of an idea; it does not establish quantum advantage.

## One novelty candidate to test

### Allocate the compression budget by verified capability recovery

**Hypothesis.** A small calibration set with automatically checkable outcomes can reveal which quantization changes actually damage a capability. Under a fixed size and latency budget, a static mixed-precision model chosen by measured recovery in task success may preserve held-out capability better than allocating bits from teacher-logit similarity, curvature, or weight sensitivity alone.

The distinction is narrow but important. A compressed model can disagree with its dense teacher and still solve the task; it can also closely match teacher logits and fail after a small decision change compounds through a multi-step solution. OnPTQ directly addresses branch consequence using short-horizon behavioral divergence, but its 2026-09-29 preprint evaluates Qwen-based multimodal models and says that its local consequence surrogate may miss longer-horizon or stochastic effects. Nox's testable extension is to use an **external task outcome**—for example, exact-answer checks for math or unit-test results for code—to score the marginal value of increasing precision in each layer or layer group. The chosen representation remains a static, hardware-supported mixed-precision checkpoint; no dynamic inference gate is required.

This is a candidate, not a novelty claim. Task-aware calibration, knowledge distillation, reward-guided optimization, and recent OnPTQ are adjacent prior art. Before claiming novelty, inspect those methods and any newer work for verifier- or reward-weighted post-training quantization. A small, verifiable task suite is only a first test; it cannot establish broad preservation across open-ended tasks.

### Minimal test

1. Freeze one language model, one runtime, and one device. Establish BF16 and strong executable low-bit baselines, including at least one current high-quality quantizer.
2. On a calibration split with verifiable math/code examples, measure each layer group's marginal task-success recovery when moving from a lower-bit to a higher-bit executable candidate. Compare that signal with teacher-logit KL, local reconstruction error, and curvature/sensitivity scores. Repeat selected upgrades to measure interaction effects.
3. Select a static precision assignment under a byte budget and a measured latency ceiling, using verifier-based recovery per cost as the allocation signal. Retain a fixed general-text behavior anchor to limit specialization. Keep to kernels supported by the selected runtime.
4. Evaluate on unseen prompts and at least one task family not used to allocate bits. Compare quality, correct-to-wrong and wrong-to-correct flips, calibration cost, resident/peak memory, and end-to-end latency against the same quantizer's uniform-bit version and the best executable baseline.

**Kill condition:** if verifier-based marginal recovery does not improve held-out capability over established sensitivity-based bit allocation, or the mixed-precision schedule loses its benefit in measured latency, discard this direction. If ordinary behavioral losses already predict task failures well enough to choose the same layers, this hypothesis is not differentiated enough.

### Quantum-information connection

There is a loose connection to protecting logical observables while allowing internal representations to change: task success is the observable Nox would protect, and the compressed weights are one implementation of that behavior. This is an analogy, not quantum error correction. The first implementation is classical, and the task-outcome objective should be compared with its ordinary classical equivalent. Quantum information belongs in the method only if a specific coding, geometry, or tensor structure improves the measured frontier beyond the classical version.

## The strongest near-term claim Nox could earn

> On **[named model]** running on **[named device]**, our **[specified operator]** produced **[quality result on held-out tasks]** at **[measured resident/peak memory]** and **[measured latency/throughput]**. We compared it with **[named quantization]** and **[named matrix method]** under the same runtime and calibration budget. The improvement came from **[ablation showing the quantum-information-inspired choice matters]**.

Until the bracketed fields are filled from reproducible runs, this is a hypothesis. A useful early result may be a negative one: a careful screen could show that a candidate structure fails the executable latency or behavior test, and rule it out before Nox commits to a research identity around it.

## Sources to revisit during implementation

- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507): candidate open model, Apache 2.0 license listed on the model card; the Qwen3 documentation identifies the 2507 Instruct line as non-thinking mode.
- [lm-evaluation-harness configuration guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/config_files.md): reproducible task configuration and seeds.
- [AQLM paper](https://arxiv.org/abs/2401.06118) and [QuIP# paper](https://arxiv.org/abs/2402.04396): low-bit comparisons and implementation constraints.
- [African Union Continental AI Strategy](https://au.int/sites/default/files/documents/44004-doc-EN-_Continental_AI_Strategy_July_2024.pdf), section 2.4.3.1: regional compute motivation.
