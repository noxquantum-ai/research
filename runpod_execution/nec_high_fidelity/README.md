# RunPod High-Fidelity NEC evaluation

This run compiles all 32 layers and seven supported projections of the pinned Llama-2-7B model using the existing High-Fidelity NEC ranks and zero-healing independent construction. It then measures dense and NEC models with the same WikiText-2 and MMLU evaluators. NEC artifacts are reconstructed to dense FP16 matrices for evaluation; this run does not establish native compressed inference speed or packed checkpoint savings.

## RunPod resources

The quality-max profile uses 65,536 uniformly sampled WikiText-2 train tokens and a disjoint 16,384-token validation cache, in 128-token windows. This is a new calibration protocol, not an exact replay of the historical 256/128 prefix-token experiment. It raises the sample count above each supported projection's input width (including 11,008 for `down_proj`) but cannot guarantee generalization or a target retention percentage.

Because the compiler repeatedly factors and fits large calibration matrices, the runner requires a B200/B300-class GPU: at least 160 GiB total VRAM, 150 GiB free VRAM, 128 GiB effective RAM, and 150 GiB free on the persistent volume mounted at `/workspace`. These are conservative qualification gates, not a measured runtime estimate. The script refuses to start compilation on smaller resources. Use a persistent network volume in the same data center as the pod.

## Install and run

Use a clean committed checkout at `/workspace/research` (or set `NEC_REPO_ROOT`) and run one command:

```bash
bash runpod_execution/nec_high_fidelity/run.sh
```

Do not use the production command for the initial streaming-calibration
qualification. The budget-controlled qualification is staged and never
progresses automatically:

```bash
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-reference
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-fp32
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-compare
```

There is no automatic timeout by default. Set `NEC_QUAL_TIMEOUT_SECONDS` to a
positive integer to impose a per-stage budget limit. Unbuffered compiler logs
and a 60-second resource/progress heartbeat are persisted under the qualification log directory.
Terminate the pod manually to stop RunPod billing. FP64 and all-seven-projection layer stages
are explicit follow-up commands. An unnecessary FP64 run is refused, and each
layer backend remains locked until its single-projection comparison passes.
Completed stages are reusable only when their SHA-256, clean Git commit,
hardware identity, and full qualification settings match the current run.

For the first stage, follow the compiler and resource streams from a second
terminal with:

```bash
tail -F /workspace/runs/nec-streaming-qualification/logs/reference-full-matrix.log
tail -F /workspace/runs/nec-streaming-qualification/logs/reference-full-matrix.heartbeat.jsonl
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh status
```

The projection qualification also stores FP16 diagnostic weight snapshots after
the sparse shield, sparse-plus-bridge construction, TT-SVD warm start, both DMRG
sweeps, and final artifact serialization. Streaming variants directly measure
`||(W_stream-W_reference)X||_F / ||W_reference X||_F` on the identical
calibration and validation activations at every stage. The main log identifies every projection, DMRG sweep, local site, solver and
elapsed site time. The heartbeat records GPU utilization/VRAM, host RAM, disk,
load average and the latest durable phase. Logging does not impose a wall-time
limit or automatically advance to another paid stage.

Artifact reuse is keyed by a deterministic fingerprint of the numerical source
files plus the complete qualification identity. The producing Git commit remains
in every report for provenance, while later report-only or documentation commits
do not invalidate unchanged compiler artifacts.

The script creates or repairs the Python 3.13.15 environment, installs the pinned PyTorch/CUDA and requirements plus the pinned `lm-evaluation-harness` commit, and runs `pip check` before model compilation. Preflight verifies the model architecture/tokenizer, package versions, RunPod persistence, resources, and clean source commit. It then loads the pinned dense model and qualifies PPL batch equivalence at batch size 1 versus the selected batch, scans all 14,042 MMLU prompts for five-shot fit, tests the longest prompt and 4,096-token context at the selected MMLU batch, and initializes the pinned harness on one question per subject. Finally it compiles one actual pinned `down_proj` matrix with a synthetic 65,536-token activation matrix at the full High-Fidelity rank and Q4 settings, then checks artifact decoding against the compiler reconstruction. This last stage validates compiler memory and artifact mechanics, not model quality. No 32-layer compilation starts unless every qualification passes.

Run the command inside `tmux` or a persistent pod terminal. Resume with the same command after interruption. Checkpoints, intermediate calibration states, configuration, logs, resource telemetry, preflight qualification, and reports live below:

```text
/workspace/runs/nec-high-fidelity-quality-max-20261019/
```

The compiler resumes only after verifying checkpoint hashes and run configuration. Its inputs include a 65,536-token train calibration cache and 16,384-token validation cache. These add substantial persistent storage; preflight requires 150 GiB free. Run progress is recorded in `compile_progress.json`; GPU/RAM/disk telemetry is appended to `logs/resources.jsonl`. Evaluation stage status is in `evaluation/evaluation_progress.json`.

The output includes dense and NEC WikiText-2 raw PPL on the 10,000-token primary prefix and full test split. Each report records mean token NLL, next-token accuracy, and exact top-1 predictions in SHA-256-verified sidecars; the final matched audit calculates dense/NEC prediction agreement. Full 5-shot MMLU uses all 57 subjects and all 14,042 questions. Both variants use the pinned model revision, tokenizer, dataset revisions, FP16 weights, SDPA, identical scoring code, and preflight-qualified shared batch sizes.

The final result distinguishes benchmark completion from quality acceptance. The strict quality gate requires full-test PPL ratio at most 1.7, primary WikiText next-token accuracy retention at least 90%, MMLU accuracy retention at least 90%, and full-test WikiText-2 dense top-1 agreement at least 90%. It also reports 70%, 80%, 90%, and 95% retention thresholds. A completed benchmark may fail the scientific gate without being mislabeled as an execution failure.

## Reproducibility and interpretation

- Model: `NousResearch/Llama-2-7b-hf`, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- NEC: ranks `[1,64,128,64,1]`, bridge rank 128, sparse budget 524,288, two DMRG sweeps, Q4 simulation with group size 64, independent construction, dense activation mode, no post-compression healing.
- Calibration: `Salesforce/wikitext` raw v1 revision `b08601e04326c79dfdd32d625aee71d232d685c3`, train split; 65,536 tokens, uniform distinct 128-token windows, seed 42. Validation: validation split, 16,384 tokens, same sampling policy, seed 43.
- WikiText-2 test: raw test split at the same dataset revision, pinned model tokenizer, 2,048 context, stride 512, float32 shifted cross-entropy, each target scored once.
- MMLU: `cais/mmlu` revision `c30699e8356da336a370243923dbaf21066bb9fe`; `lm-evaluation-harness` version 0.4.13 at commit `ddd67220430a2470529f25fd5c05a576ca1057a0`; five-shot, 57 subjects, deterministic seeds, no chat template.
- Comparison: dense model measured in this run under the same evaluation procedures. The experiment is a new independent zero-healing configuration due to its expanded, uniformly sampled calibration and validation caches; results should not be described as the historical trajectory-conditioned rerun.

The sole configuration for this RunPod experiment is [run_config.json](run_config.json). The final JSON is keyed by checkpoint manifest hash and source commit under `evaluation/results/`; it records structural coefficients separately from serialized bytes and explicitly marks packed-byte savings and native compressed inference as unevaluated.
