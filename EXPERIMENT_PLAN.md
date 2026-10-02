# Three-week experiment plan — 2026-10-02

## Scope and decision rule

Provisional target: **Qwen2.5-1.5B-Instruct** on one reproducibly available **16–24 GB NVIDIA GPU**. Fix the exact GPU, driver, runtime, and model revision before the first benchmark. If the team only has CPU access, use one named CPU and its optimized inference runtime throughout; do not compare GPU and CPU results. The model choice can change once the actual device is known, but freeze it after the first baseline.

**Primary success criterion:** at one or more comparable operating points, the candidate has no more measured peak inference memory and no worse end-to-end latency than a strong baseline, while improving held-out quality beyond evaluation uncertainty on a predeclared task aggregate and avoiding a material regression on any task. Report the full tradeoff curve, even if no point passes. Do not use parameter count or compressed file size as a proxy for peak memory or speed.

This is deliberately strict. A three-week project may only reach a credible component-level result. Record that as such.

## Comparison set

1. **Uncompressed reference:** BF16/FP16 model on the same device and runtime when feasible.
2. **Deployable quantization:** at least one optimized 4-bit weight method available in the selected runtime (for example GPTQ/AWQ with a compatible kernel), plus a 3-bit or other lower-memory point if supported. Include all scales, codebooks, and metadata in stored size.
3. **Strong matrix method:** ASVD or SVD-LLM where reproducible, with factor operations executed directly. Match measured memory and show its latency.
4. **Simple tensor control:** fixed tensorization/order and uniform bond dimension, at the candidate's approximate memory budget.
5. **Nox candidate and ablations:** activation-weighted spectra and rank allocation; remove activation weighting, ordering search, and domain balancing one at a time. Compare at matched memory and measure speed for every variant retained.

For research-grade 2–3 bit claims, add an executable AQLM or QuIP# baseline if it supports the model and device. If it does not, state that gap and restrict the claim to the baselines actually run. Do not compare to paper numbers obtained with other models, prompts, or hardware.

## Evaluation protocol

- Freeze a calibration set, a development set, and a held-out test set before tuning. Calibration should cover several text domains and never contain test examples. Record exact dataset snapshots, sample IDs, tokenizer, seeds, and prompt templates.
- Use a broad, small fixed battery at first: language-model perplexity on held-out text; HellaSwag, ARC Challenge, and a math or reasoning set such as GSM8K; add at least one instruction-following or multilingual slice if the chosen model can score reliably. Fix shot counts and chat template. A task with floor-level base accuracy is uninformative for preservation; replace it before candidate tuning.
- Report each task's raw score, uncompressed score, and score retention. For an aggregate, predeclare a macro average of normalized score retention; also show the worst task regression. Use paired bootstrap confidence intervals where task items permit them. Never hide a loss in one capability inside a single mean.
- Bench prefill and decode separately at fixed prompt/output lengths and concurrency 1, then one realistic longer-context or small-batch case. Record median and p95 latency, tokens/sec, resident GPU memory, peak GPU memory, CPU RAM, and load time. Warm the model, repeat runs, and report the exact benchmark script and runtime versions.
- Measure direct factorized execution. If factors are materialized into a dense matrix at load or per token, count that dense allocation and speed honestly. Use the same KV-cache format and context length across methods.

## Week 1: establish reality

**Days 1–2:** lock hardware and model; make the BF16 and optimized quantized variants run; save exact commands and hardware metadata. Confirm all quality tasks have useful signal on this model.

**Days 3–5:** benchmark quality, memory, and latency for the reference and at least two quantized operating points. Implement or reproduce one strong matrix baseline. Produce the first Pareto plot. If a baseline cannot run, document its failure and choose the next strongest executable alternative.

**Deliverable:** immutable baseline table, raw results, benchmark scripts, and environment manifest. Without this, a new compression result is hard to interpret.

## Week 2: test the mechanism cheaply

**Days 6–7:** capture input activations for representative attention and MLP projections across calibration domains. Estimate diagonal or low-rank `C`; log the approximation used. For selected matrices, compare raw vs activation-weighted Schmidt spectra under several index orderings and bond budgets.

**Days 8–10:** implement direct factorized forward passes for two or three selected matrices. Compare local output error, calibration/development task effects, actual factor size, and isolated layer latency against fixed tensorization and ASVD/SVD-LLM. Search only a small predeclared set of orderings and ranks; record all trials.

**Gate:** if direct factorized execution cannot plausibly match the quantized operator's latency at any useful fidelity and size, stop scaling the tensor operator and preserve the diagnostic result. If it can, proceed to model-level compression.

## Week 3: model-level evidence

**Days 11–15:** allocate the budget across selected model layers using measured sensitivity and bond spectra. Keep the full model executable without reconstructing dense weights. Compare against baseline variants at matched measured memory, then optimize the most expensive contraction only if profiling identifies a clear target.

**Days 16–18:** freeze the candidate; run the held-out battery and full latency/memory protocol. Run the minimum ablations that reveal whether activation weighting, tensorization choice, and spectrum-based budget allocation each matter.

**Days 19–21:** produce a concise technical note with the method, exact configuration, all points on the quality–memory–latency plot, error bars, failures, and reproducible scripts. Prepare a short demo only if the candidate actually clears the gate.

## What goes into the YC application

The strongest three-week evidence is a reproducible artifact and a precise sentence about what it does on one device. Include the baseline frontier and one ablation that isolates the proposed mechanism. If the method does not beat the frontier, present the measured finding and the next technical question; do not claim a breakthrough from compression ratio alone. The founding team's background can support execution credibility, but the experiment should carry the technical claim.

## Open decisions

- The exact hardware and available budget.
- Whether 1.5B is large enough to show a meaningful capability-preservation signal on the chosen tasks; move to a 3B model only if the device and schedule permit it before freezing baselines.
- Which optimized 3-bit/4-bit and matrix implementations run reliably on that exact model and hardware.
