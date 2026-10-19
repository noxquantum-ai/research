# RunPod: High-Fidelity NEC, full Llama-2-7B

This folder compiles the existing High-Fidelity NEC configuration and then performs a matched, in-run comparison against the pinned dense model. It does not change the compression method or fine-tune/heal either model. Dense and NEC scores are produced by the same evaluator source, model revision, tokenizer, dataset bytes, prompts, precision, and scoring code in separate processes.

## Pod and persistence

Use a RunPod GPU pod with at least one CUDA 12.8-compatible GPU with 24 GiB VRAM, at least 32 GiB system RAM, and a 100 GB persistent network volume mounted at `/workspace`. Preflight also requires 20 GiB free VRAM by default (`NEC_MIN_GPU_FREE_GIB` can lower this only when intentionally accepting less headroom). The verified NEC layer-0 smoke succeeded on a 15 GiB T4; the larger minimum is for full compression and evaluation headroom. An A100 80 GB is a practical single-GPU choice if available. Keep the model cache, environment, checkpoints, logs, and reports under `/workspace`; the scripts refuse to proceed if `/workspace` is not mounted or if fewer than 50 GiB are free. The RunPod environment must be able to download the pinned model/tokenizer and datasets; provide Hugging Face credentials through the pod's private environment/secret mechanism if your account requires them, never by committing a token.

RunPod documents network volumes as persistent across pod replacement and requires the consuming pod to be in the volume's data center. Select the GPU and volume in the same data center. Avoid multiple writers on one checkpoint volume.

## Install and run

Place a clean checkout of the committed `main` repository at `/workspace/research` (or set `NEC_REPO_ROOT`), then run exactly one command from that checkout:

```bash
bash runpod_execution/nec_high_fidelity/run.sh
```

`run.sh` creates the pinned persistent runtime if it is absent, runs the fail-closed preflight, compiles or resumes all 32 NEC layers, and evaluates dense and NEC WikiText-2 followed by dense and NEC MMLU. For a detached job, run that same command inside `tmux` or use the pod provider's persistent terminal. `bootstrap.sh` installs Python 3.13.15, PyTorch 2.11.0+cu128, and the locked requirements. Preflight loads only the pinned model configuration and tokenizer—not the 7B weight tensors—and checks Llama-2 dimensions, architecture, vocabulary, RMSNorm epsilon, vocabulary/special IDs, a fixed tokenizer probe, GPU/RAM/disk, package versions, mounted persistence, and a clean committed source tree.

The runner compiles all 32 layers and seven projections using the existing `run_config.json`. It writes directly to `/workspace/runs/nec-high-fidelity-20261017/checkpoint`, uses the compiler's `--resume` validation when the manifest exists, and reuses completed evaluation JSON only when both the checkpoint-manifest SHA-256 and evaluator source commit match. Before accepting an already-complete checkpoint, it re-audits the manifest, all layer hashes, all persisted layer inputs, and the full recorded calibration/configuration against `run_config.json`. Every invocation has a separate log/attempt directory; it will not overwrite completed PPL/MMLU reports. If the pod or SSH session stops, reattach the same network volume and run the same command again. The compiler skips integrity-checked completed layers and resumes at the first missing layer.

Monitor from another shell:

```bash
tail -f /workspace/runs/nec-high-fidelity-20261017/logs/resources.jsonl
cat /workspace/runs/nec-high-fidelity-20261017/compile_progress.json
cat /workspace/runs/nec-high-fidelity-20261017/evaluation/evaluation_progress.json
```

The final matched result is written under `evaluation/results/<checkpoint-manifest-sha256>/<source-commit>/final_results.json`. Four immutable reports are retained beside it: dense and NEC WikiText-2, and dense and NEC MMLU. Keying by both artifact and source commit prevents reuse across checkpoint or evaluator changes. The final report is not emitted unless the protocol fingerprints match, WikiText token IDs/windows/counts match, all independent NLL checks pass, the MMLU subject sets and per-subject counts match, and both 57-question adapter smokes plus both complete 57-subject/14,042-question runs pass.

## Evaluated protocols

- Model: `NousResearch/Llama-2-7b-hf`, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Compression: existing High-Fidelity NEC only; all 32 layers; seven projections/layer; no healing.
- Calibration: pinned WikiText-2 raw v1 revision `b08601e04326c79dfdd32d625aee71d232d685c3`, 256 train tokens, 128 validation tokens, prefix sampling, length 128, fixed seeds recorded in `run_config.json`.
- Comparison model: the unmodified dense `NousResearch/Llama-2-7b-hf` at the same pinned revision; this is measured in the same run, not copied from a publication.
- PPL: pinned WikiText-2 raw test revision, 2048 context, stride 512, 10,000 input tokens and full split, each shifted target scored once; float32 token NLL is summed, divided by the exact number of scored labels, then exponentiated. Dense and NEC token-stream SHA-256 values and window specifications must match.
- MMLU: pinned `cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe`, lm-evaluation-harness commit `ddd67220430a2470529f25fd5c05a576ca1057a0`, standard 5-shot, all 57 subjects, batch 1, deterministic seeds, no chat template. Dense and NEC use the identical harness adapter and task definitions.

The runner derives evaluator placement caps as at most 75% of the least-free GPU VRAM and 75% of effective container RAM, capped at 60 GiB/GPU and 48 GiB CPU, leaving space for runtime allocations. With an idle 24 GiB GPU / 32 GiB RAM pod this is about 18 GiB / 24 GiB; an idle 80 GiB / 64 GiB pod gets at most 60 GiB / 48 GiB. Override `NEC_EVAL_GPU_MEMORY_GIB` or `NEC_EVAL_CPU_MEMORY_GIB` only to fit the actual pod; this changes placement only, not precision or scoring. Resource telemetry continues through compilation and evaluation; `compile_progress.json` tracks the last compiled layer, while `evaluation_progress.json` records PPL/MMLU stage state and failures.
