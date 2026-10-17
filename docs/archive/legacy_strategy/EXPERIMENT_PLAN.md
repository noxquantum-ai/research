# Three-week experiment plan — 2026-10-02

## Scope and decision rule

Provisional target: **Qwen3-4B-Instruct-2507** on one reproducibly available **16–24 GB NVIDIA GPU**. Its 4B parameters make a multi-task capability check more informative than a sub-2B model while remaining practical on a single such GPU at moderate context lengths. It is Apache 2.0, and the Instruct-2507 checkpoint uses non-thinking mode, which avoids generation-mode variation in the benchmark. Fix the exact GPU, driver, runtime, and model revision before the first benchmark. If the team only has CPU access, use one named CPU and its optimized inference runtime throughout; do not compare GPU and CPU results. The model choice can change once the actual device is known, but freeze it after the first baseline.

**Primary success criterion:** at one or more comparable operating points, the candidate has no more measured peak inference memory and no worse end-to-end latency than a strong baseline, while improving held-out quality beyond evaluation uncertainty on a predeclared task aggregate and avoiding a material regression on any task. Report the full tradeoff curve, even if no point passes. Do not use parameter count or compressed file size as a proxy for peak memory or speed.

This is deliberately strict. A three-week project may only reach a credible component-level result. Record that as such.

## Comparison set

1. **Uncompressed reference:** BF16/FP16 model on the same device and runtime when feasible.
2. **Deployable quantization:** at least one optimized 4-bit weight method available in the selected runtime (for example GPTQ/AWQ with a compatible kernel), plus a 3-bit or other lower-memory point if supported. Include all scales, codebooks, and metadata in stored size.
3. **Strong matrix method:** ASVD or SVD-LLM where reproducible, with factor operations executed directly. Match measured memory and show its latency.
4. **Mixed-method frontier:** add one established quantization-plus-correction method, such as SLIM or LQER, where it runs on the chosen runtime and model. Include a structured sparse point only when the target hardware has an efficient kernel.
5. **Novelty candidate:** test whether verifier-measured task-success recovery is a better signal for static mixed-precision allocation than reconstruction, curvature, or sensitivity scores. Use verifiable math/code tasks plus a fixed general-text behavior anchor. Compare against the same quantizer with uniform precision and established sensitivity-based allocation; do not call this novel until the literature check is complete.
6. **Representation screening is conditional:** tensor networks, behavior-distortion objectives, low-rank corrections, and other structures are options only if the diagnostic points to them. Tensor-network compression, generic information-guided allocation, cross-layer error compensation, on-policy decision preservation, and adaptive token correction already have substantial or very recent prior art.

For research-grade 2–3 bit claims, add an executable AQLM or QuIP# baseline if it supports the model and device. If it does not, state that gap and restrict the claim to the baselines actually run. Do not compare to paper numbers obtained with other models, prompts, or hardware. Include only mechanisms that fit the three-week schedule; the screen is meant to choose one deep dive, not build every family.

## Evaluation protocol

- Freeze a calibration set, a development set, and a held-out test set before tuning. Calibration should cover several text domains and never contain test examples. Record exact dataset snapshots, sample IDs, tokenizer, seeds, and prompt templates.
- Use a broad, small fixed battery at first: language-model perplexity on held-out text; HellaSwag, ARC Challenge, and a math or reasoning set such as GSM8K; add at least one instruction-following or multilingual slice if the chosen model can score reliably. Fix shot counts and chat template. A task with floor-level base accuracy is uninformative for preservation; replace it before candidate tuning.
- Report each task's raw score, uncompressed score, and score retention. For an aggregate, predeclare a macro average of normalized score retention; also show the worst task regression. Use paired bootstrap confidence intervals where task items permit them. Never hide a loss in one capability inside a single mean.
- Bench prefill and decode separately at fixed prompt/output lengths and concurrency 1, then one realistic longer-context or small-batch case. Record median and p95 latency, tokens/sec, resident GPU memory, peak GPU memory, CPU RAM, and load time. Warm the model, repeat runs, and report the exact benchmark script and runtime versions.
- Measure direct factorized execution. If factors are materialized into a dense matrix at load or per token, count that dense allocation and speed honestly. Use the same KV-cache format and context length across methods.

## Week 1: establish reality

**Days 1–2:** lock hardware and model; make the BF16 and optimized quantized variants run; save exact commands and hardware metadata. Confirm all quality tasks have useful signal on this model.

**Days 3–5:** benchmark quality, memory, and latency for the reference and at least two quantized operating points. Implement or reproduce one strong matrix baseline. On held-out math/code development examples, log paired dense-versus-quantized correctness and test whether teacher-logit KL, local reconstruction error, and short-branch divergence identify correct-to-wrong flips. Produce the first Pareto plot. If a baseline cannot run, document its failure and choose the next strongest executable alternative.

**Deliverable:** immutable baseline table, raw results, benchmark scripts, environment manifest, and a diagnostic answering whether behavioral reconstruction misses capability failures. Without this, a new compression result is hard to interpret.

## Week 2: compare mechanisms cheaply

**Days 6–7:** complete the prior-art check around verifier- or reward-weighted PTQ, OnPTQ, and task-aware bit allocation. If the Week 1 diagnostic shows that ordinary sensitivity scores miss meaningful correctness failures, measure the marginal task-success recovery from selected layer-group precision upgrades. Do not add a dynamic runtime operator.

**Days 8–10:** choose a static precision schedule under a memory budget and measured latency ceiling using the verifier-based marginal recovery signal. Compare it with uniform precision and sensitivity-based allocation on held-out examples. Check a task family not used for allocation to detect specialization. If the initial diagnostic fails to show a distinct gap, or the verifier-based schedule does not improve held-out success, stop this candidate and use the time to reproduce the most informative baseline result instead.

**Gate:** proceed only if the diagnostic establishes that current allocation signals systematically miss task failures and the verifier-based schedule improves held-out capability without violating the measured latency ceiling. Confirm all of this against the literature before making a novelty claim. A theoretical compression ratio alone does not pass. If no direction passes, document the result and narrow the next question.

## Week 3: model-level evidence

**Days 11–15:** scale the selected calibration change across the complete model. Keep the compressed form executable without reconstructing dense weights. Compare against baseline variants at matched measured memory, then profile calibration overhead and inference bottlenecks.

**Days 16–18:** freeze the candidate; run the held-out battery and full latency/memory protocol. Ablate the chosen method's defining components and compare against the best executable baseline.

**Days 19–21:** produce a concise technical note with the method, exact configuration, all points on the quality–memory–latency plot, error bars, failures, and reproducible scripts. Prepare a short demo only if the candidate actually clears the gate.

## What goes into the YC application

The strongest three-week evidence is a reproducible artifact and a precise sentence about what it does on one device. Include the baseline frontier and one ablation that isolates the proposed mechanism. If the method does not beat the frontier, present the measured finding and the next technical question; do not claim a breakthrough from compression ratio alone. The founding team's background can support execution credibility, but the experiment should carry the technical claim.

## Open decisions

- The exact hardware and available budget.
- Whether the team's device can run the 4B model comfortably at the selected context length; if not, select a smaller fixed model before freezing baselines.
- Which optimized 3-bit/4-bit, sparse, low-rank, and correction implementations run reliably on that exact model and hardware.
