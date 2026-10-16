# Compression pipeline investigation (36-hour window)

Started 2026-10-16. This log records the local repository snapshot, prior evidence, and the reproducible remote A10 runs. The remote staging tree is `/home/shadeform/diag36h-research`; checkpoints and run logs are retained there under `checkpoints/diagnostic_36h/`.

## Current scope and corrections (2026-10-16)

The user narrowed the target to one high-fidelity engine with zero healing, and explicitly authorized incremental pushes directly to `main`. The agreed quality checks are raw WikiText PPL within 5% of matched dense, at least 95% dense top-1 agreement, and at least 95% of the published five-shot Llama 2 7B MMLU score on the complete test set. MMLU acceptance uses one aggregate score across all 57 subjects. The published result is an external reference, not a same-run dense comparison. Full-model serialized size must include all tensors needed for inference. The high-fidelity name denotes a compression target, not an achieved ratio.

Two measured bugs invalidate the earlier claim that the baseline protocol was fully verified:

1. The PyTorch WikiText files used below are **processed word-level text**, despite the loader calling them `wikitext-2-raw-v1`. They contain 54,625 / 11,718 / 15,218 literal `<unk>` replacements in train / validation / test. The canonical raw dataset is now loaded from `Salesforce/wikitext` at revision `b08601e04326c79dfdd32d625aee71d232d685c3`, using its `wikitext-2-raw-v1` Parquet files with an explicit double-newline row join. There is no processed-text fallback. Source-file and evaluated-text hashes are recorded separately.
2. The custom RMSNorm default was `1e-6`, but the pinned LLaMA-2 configuration requires `1e-5`. On 128 raw validation tokens, the cached dense weights match Hugging Face exactly. With identical HF layer inputs, layer-0 output relative error is **0.23142913** at the old epsilon and **0.00042425** at the configured epsilon. Layer-15 errors are **0.00018317** and **0.00018305** respectively. This isolates a normalization implementation mismatch before compression, particularly at small embedding magnitudes. The remaining small difference includes FP32 custom versus FP16 HF arithmetic.

Controlled reevaluation of the *same existing four high-fidelity no-healing layer checkpoints* (15–18; still fitted using the old calibration path):

| Evaluation text | Dense PPL | Compressed PPL | Dense top-1 agreement |
|---|---:|---:|---:|
| Canonical raw WikiText | 5.2922 | 7.9964 | 73.53% |
| Legacy processed text | 20.7908 | 30.6696 | 68.84% |

Each run uses 10,000 input tokens, 9,999 scored labels, context 2,048 and stride 512. Raw scores still fail the quality target. Correcting the dataset label does not eliminate compression damage. The raw comparison is an evaluation-source control, not a newly calibrated compressed model.

Commands (GPU host, repository root):

```bash
.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir checkpoints/diagnostic_36h/high_fidelity_10x/no_healing \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/reference_v1/results.json
# Epsilon control; this explicitly compares legacy 1e-6 with model.config.rms_norm_eps.
.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir checkpoints/diagnostic_36h/high_fidelity_10x/no_healing \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/reference_v1/epsilon_control.json --parity-only
```

Raw logs and JSON are under `experiment_artifacts/diagnostic_36h/reference_v1/`. The new `scripts/compile_high_fidelity.py` runs only independent high-fidelity compression, reads weights from the pinned safetensors snapshot, uses configuration-derived normalization, saves train/validation inputs, and evaluates reconstructed saved components. A corrected one-layer compile is the next check; full-model PPL/MMLU and the 95% target remain unverified.

After the source and normalization fixes, `.venv/bin/python -m unittest discover -s tests -v` on the GPU host passed **9 tests** in 0.055 seconds. The new tests compare a tiny real HF decoder against the custom decoder at small embedding magnitudes (including grouped-query attention), require the pinned raw Parquet variant, and ensure download failures cannot silently select processed data.

The next command was launched on the GPU host at base repository commit `f0415af` plus the source changes identified in its run fingerprint:

```bash
.venv/bin/python -u scripts/compile_high_fidelity.py \
  --output checkpoints/diagnostic_36h/corrected_raw_layer15 \
  --start-layer 15 --end-layer 15 --n-calib 256 --n-validation 128 --seq-len 128
```

Its compilation completed in 206.217 seconds. This command has no healing or alternate regime option. The saved layer occupies 70,110,452 bytes versus 404,766,720 dense FP16 tensor bytes (5.77× for this layer, excluding unchanged model tensors). Reconstructed block relative error was 0.013957 on calibration and 0.046512 on held-out validation.

### Corrected single-layer and fitting controls

Source/data/normalization fixes were committed and pushed directly to `main` as `a130d3c`. Subsequent diagnostic changes are recorded by the next commit; all execution remains on the remote A10.

The corrected layer-15 checkpoint was evaluated with `scripts/diagnose_reference_pipeline.py --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 --weights-dir models/diagnostic_36h_cache --output checkpoints/diagnostic_36h/corrected_raw_layer15/evaluation.json` using the same raw-test protocol above. Dense PPL was **5.2922001**, compressed PPL **5.6438818**, dense top-1 agreement **89.57896%**. Dense next-token accuracy was 61.04610%, compressed 60.37604%. The previous layer-15 checkpoint scored PPL 5.6445222 and agreement 89.40894% on the same raw text. Correcting data and normalization is necessary but does not explain most remaining layer-15 compression loss. Both single-layer results fail the agreed PPL/agreement gates.

Commands (remote repository root):

```bash
.venv/bin/python -u scripts/audit_projection_fit.py \
  --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
  --dense-weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/corrected_raw_layer15/projection_fit.json
.venv/bin/python -u scripts/diagnose_projection_stages.py \
  --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
  --dense-weights-dir models/diagnostic_36h_cache --projection v_proj \
  --output checkpoints/diagnostic_36h/corrected_raw_layer15/v_proj_stages.json
```

The seven-projection audit uses the saved 256 raw train tokens and 128 distinct raw validation tokens, clean dense projection inputs, and actual saved components. All seven projections improve calibration error but worsen held-out error when the fitted TT component is added. V-projection train/validation errors are 0.155965/0.803680; removing its TT component gives 0.27637/0.76176. Down-projection held-out error is 1.02235, versus 0.89497 without the TT component.

The V-projection staged control reproduced the existing quantized checkpoint within the enforced 0.001 relative matrix tolerance. Disabling quantization yields train/validation errors 0.137952/0.810554 after FP16 serialization: it does not repair generalization. In that unquantized control, TT warm-start validation error was 0.716343, worsening to 0.778723 after sweep 1 and 0.810556 after sweep 2 while calibration error decreases. This supports a fitting/generalization failure under the current calibration coverage; it does not prove a universal budget limit or a serialization fault.

Next controlled experiment (base `a130d3c` plus diagnostic/sampling edits): prepare 1,024 train tokens as either eight contiguous 128-token windows or eight nonoverlapping corpus-spread windows, seed 42, keeping the same 128 validation tokens and compression budget. No held-out labels enter fitting.

```bash
.venv/bin/python -u scripts/prepare_calibration_control.py \
  --output checkpoints/diagnostic_36h/calibration_coverage1024
for sampling in prefix uniform; do
  .venv/bin/python -u scripts/diagnose_projection_stages.py \
    --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
    --dense-weights-dir models/diagnostic_36h_cache --projection v_proj \
    --calibration-input checkpoints/diagnostic_36h/calibration_coverage1024/$sampling.pt \
    --only-quantized \
    --output checkpoints/diagnostic_36h/calibration_coverage1024/${sampling}_v_proj.json
done
```

Both coverage controls completed. V-projection relative errors:

| Calibration | Tokens | Calibration error | Held-out error | Held-out at TT warm-start |
|---|---:|---:|---:|---:|
| Prefix | 256 | 0.155965 | 0.803680 | See staged JSON |
| Prefix | 1,024 | 0.374522 | 0.767132 | 0.712595 |
| Corpus-spread | 1,024 | 0.408913 | 0.724945 | 0.678979 |

Coverage helps at fixed budget, but the subsequent calibration fitting still worsens held-out error. These are projection errors, not full-model accuracy or PPL. Dense-FP16 round-trip controls give exactly zero error for all seven projections. Direct groupwise 4-bit quantization of the full dense matrices gives held-out errors 0.053181 / 0.047386 / 0.113340 / 0.104873 / 0.070968 / 0.092284 / 0.107434 for q/k/v/o/gate/up/down respectively. This control retains all matrix entries and is not the target compressed representation or a packed checkpoint; no size claim is made for it. Command:

```bash
.venv/bin/python -u scripts/audit_projection_fit.py \
  --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
  --dense-weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/corrected_raw_layer15/projection_budget_controls.json
```

The next diagnostic adds a fixed dense-weight prior only to the TT fitting objective, leaving the sparse/bridge initialization and storage budget unchanged. In original coordinates its additional squared-error term is `N * strength * ||(W_hat - W) diag(channel_rms)||_F^2`. Deterministic basis columns implement that term exactly. Strength 1 is a single energy-matched control, not a sweep. Validation data is never used in this objective; the normal engine default remains strength 0 pending measurement.

```bash
.venv/bin/python -u scripts/diagnose_projection_stages.py \
  --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
  --dense-weights-dir models/diagnostic_36h_cache --projection v_proj \
  --calibration-input checkpoints/diagnostic_36h/calibration_coverage1024/uniform.pt \
  --only-quantized --diagonal-prior-strength 1 \
  --output checkpoints/diagnostic_36h/calibration_coverage1024/uniform_weight_prior_v_proj.json
```

The prior control completed: calibration error **0.449208**, held-out error **0.673847**, compared with **0.408913 / 0.724945** without the prior. The first sweep improves held-out error to 0.673074 instead of worsening it to 0.706886. This supports the regularizer as a partial remedy, not an adequate-quality engine. It remains opt-in for diagnostics. These early diagnostic results reconstruct the FP16 artifact components in memory; baseline reconstruction was separately checked against the existing disk checkpoint. Subsequent projection diagnostics now save and reload their own `.pt` artifacts and record their actual file sizes.

A single larger-budget control completed with the original zero-strength fit and bridge rank 512 instead of 128, retaining identical 1,024 corpus-spread tokens and TT ranks:

```bash
.venv/bin/python -u scripts/diagnose_projection_stages.py \
  --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
  --dense-weights-dir models/diagnostic_36h_cache --projection v_proj \
  --calibration-input checkpoints/diagnostic_36h/calibration_coverage1024/uniform.pt \
  --only-quantized --bridge-rank-control 512 \
  --output checkpoints/diagnostic_36h/calibration_coverage1024/uniform_bridge512_v_proj.json
```

Its calibration/held-out errors are **0.207751 / 0.616448**, versus **0.408913 / 0.724945** at bridge rank 128. It still worsens held-out error relative to its own TT warm-start (0.598875). The saved and reloaded projection artifact occupies **15,742,968 bytes**; reconstruction differs from the pre-save FP32 fit by **0.0002437404** relative norm. Fit/audit duration was **59.394 seconds**. More bridge capacity helps, but neither this result nor the fixed-budget regularizer establishes adequate full-model quality. This budget control is not a new production regime.

Diagnostic source, completed coverage/prior results, and 12 passing tests were committed and pushed to `main` as `f8265e6`. The larger-budget result is recorded in the following results-only increment. The next concrete check is to retain the regularized projection artifact and evaluate both it and the baseline on multiple disjoint validation windows before extending the candidate to all seven projections of a single layer. There is no GPU/data blocker. Full 32-layer compilation remains gated on the failing one-layer PPL/agreement checks.

Direct tensor accounting of the corrected layer checkpoint: TT cores 13,329,408 bytes; bridge A/B 10,878,976 / 9,109,504; channel scales 71,168; sparse indices 29,334,944; sparse values 7,333,736. Sparse payload alone is 36,668,680 bytes. Packing/index representation must improve to meet the intended full-model size target; simulated quantization does not provide packed storage.

The updated remote suite passed **12 tests** in 0.086 seconds, including the augmented objective identity and unchanged zero-strength path. These tests are engineering checks, not evidence of full-model quality. Full-model corrected PPL, MMLU, 95% agreement, and the intended compression ratio remain unverified.

Correction to an intermediate status message: historical raw logs **are present** at the paths below. Git's `*.log` ignore rule hid them from the default file search. `nox_big_guns.log` lines 2860–2900 show the failed canonical dataset load, processed fallback, dense PPL 19.75, and compressed PPL 10,916.44. Those logs substantiate the recorded number, but do not by themselves isolate how much each compression defect contributed.

## Repository and execution snapshot

### Broader validation and next single-layer control

At `2805641` plus the subsequent sampling/plumbing changes, prepared eight disjoint raw validation windows (1,024 tokens total, sequence length 128, seed 43). Both fits reuse the exact existing 1,024-token corpus-spread train activation file. Input file hashes and diagnostic source fingerprints are now stored in results.

```bash
.venv/bin/python -u scripts/prepare_calibration_control.py \
  --output checkpoints/diagnostic_36h/broad_validation1024 \
  --n-validation 1024 --validation-sampling uniform
for strength in 0 1; do
  .venv/bin/python -u scripts/diagnose_projection_stages.py \
    --checkpoint-dir checkpoints/diagnostic_36h/corrected_raw_layer15 \
    --dense-weights-dir models/diagnostic_36h_cache --projection v_proj \
    --calibration-input checkpoints/diagnostic_36h/calibration_coverage1024/uniform.pt \
    --validation-input checkpoints/diagnostic_36h/broad_validation1024/validation.pt \
    --only-quantized --diagonal-prior-strength "$strength" \
    --output checkpoints/diagnostic_36h/broad_validation1024/prior${strength}_v_proj.json
done
```

Saved/reloaded V-projection held-out error is **0.749948** without the prior and **0.696569** with it. The regularizer improves every one of the eight windows: baseline range 0.733554–0.769630; regularized range 0.682120–0.713550. Calibration errors reproduce the earlier fits. This is a reproducible partial improvement, not a sufficient quality result. The test suite now passes **14 tests**, including held-out sampling isolation and a mock-compression test proving all seven projections receive the prior while retaining dense, unhealed fitting inputs.

Next command: `bash scripts/run_weight_prior_layer_control.sh` on the GPU host. It refuses an existing output directory, checks the eight-window result first, then compiles only layer 15 with strengths 0 and 1 at identical high-fidelity ranks, calibration, validation, and zero-healing settings. It evaluates each saved layer in the pinned full dense model using the same raw WikiText PPL/agreement protocol. Outputs go to `checkpoints/diagnostic_36h/layer15_uniform_prior_control/`. The full 32-layer candidate remains gated; projection improvements do not waive its PPL/agreement/MMLU requirements.

Research context: [Activation-Aware Weight Tensorization](https://arxiv.org/html/2610.10085v1) distinguishes calibration-weighted objectives from held-out fidelity and reports that richer covariance objectives do not always generalize better. Its experiments concern other models and selected operator/suffix replacements; they do not establish our target. The regularizer here is an ordinary fixed weighted Frobenius penalty implemented by basis augmentation; no novelty or guaranteed accuracy claim is made.

The matched single-layer driver started on the A10 at **2026-10-16 10:04:45 UTC**, commit **9af8ac2**, after verifying improvement on all eight windows. Its durable log is `checkpoints/diagnostic_36h/layer15_uniform_prior_control_driver.log`. Strength-0 compilation is in progress; strength-1 compilation and matched PPL/agreement evaluation are queued by the fail-fast driver.

### Storage control and MMLU data readiness

Measured a separate packed encoding against the existing corrected 256-token layer-15 checkpoint. Groupwise factors use packed nibbles and FP32 scales; sparse values stay FP16, while indices become CSR row pointers plus int16 columns. This reduces disk overhead; factor requantization is not lossless and is explicitly gated at 0.001 relative error per component. The running fitting controls are unchanged.

```bash
.venv/bin/python -m unittest discover -s tests -p test_packing.py
.venv/bin/python -u scripts/pack_checkpoint_control.py \
  --input checkpoints/diagnostic_36h/corrected_raw_layer15/compressed_layer_15.pt \
  --output-dir checkpoints/diagnostic_36h/packing_layer15
```

All **3 packing tests passed**, and the full remote suite subsequently passed **17 tests** in 2.484 seconds. Actual disk bytes decreased from **70,110,452 to 24,346,251** (2.87972× smaller than the existing artifact). Factor relative errors range roughly 0.000264–0.000369; sparse coordinates, sparse values, and channel scales round-trip exactly. This is **component-round-trip validation only**. The packed format is not yet integrated into inference, and no full-model storage ratio or inference memory saving is claimed. A decoded checkpoint was exported for an independent matched-quality check:

```bash
.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir checkpoints/diagnostic_36h/packing_layer15/roundtrip \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/packing_layer15/evaluation.json
```

This GPU quality check is pending completion of the running single-layer comparison.

Separately validated and cached full MMLU data on the remote host, using `cais/mmlu` revision **c30699e8356da336a370243923dbaf21066bb9fe**, with no fallback:

```bash
.venv/bin/python -u scripts/audit_mmlu_source.py \
  --output checkpoints/diagnostic_36h/mmlu_source_preflight.json
```

The Parquet files contain **285 dev examples (5 per subject) and 14,042 test examples across 57 subjects**. Their file hashes and subject counts are in the output JSON. The pinned LLaMA tokenizer appends A/B/C/D as token IDs 319/350/315/360 after the answer prefix in the probe. This prepares data/protocol checks, not a model accuracy result. The [dataset specification](https://huggingface.co/datasets/cais/mmlu/raw/main/README.md) and [reference evaluator](https://raw.githubusercontent.com/hendrycks/test/master/evaluate.py) support using subject-specific demonstrations and answer-letter scoring; the earlier subset/fallback evaluator remains unsuitable for the final acceptance claim.

- Local repository: `/home/fluxx/Workspace/research`.
- Initial branch/HEAD: `main` / `f0415af` (`chore: organize compressed model tiers into nested tiers directory`). This is not the suggested `debug/36h-pipeline` / `42a50aa`.
- The worktree was clean at the start. Initial investigation edits were local and uncommitted. Later user messages explicitly authorize commits and pushes to `main`; publication will be recorded with commit IDs. No reset or clean was performed.
- GPU inference and validation ran on the user-provided A10 host (`shadeform@129.80.249.133`), not on the local machine. The host reported an NVIDIA A10 with 23,028 MiB, Torch 2.7.0 CUDA, and an isolated venv. Model weights and WikiText inputs were fetched there.
- Initial remote environment had Transformers 5.20.0; model load failed because it required `torch.distributed.checkpoint.hf_storage`, absent from Torch 2.7.0. The remote venv was pinned to Transformers 4.51.3, after which dense and compressed inference succeeded. This was an environment compatibility failure, not a pipeline quality result.

## Artifacts and prior run evidence

The repository contains actual historical checkpoints under `compressed_models/tiers/`, manifests under `compressed_models/tiers/*/model_manifest.json`, and logs `compressed_models/nox_big_guns.log`, `compressed_models/noxquantum-day-3-cascade.log`, and `compressed_models/day5_benchmarks.log`.

- The historical full-model WikiText run recorded dense PPL 19.75 and production PPL 10,916.44; a high-fidelity run recorded PPL 16,518.61 and an extreme run 8,135.61. These are real log values, but they are from an older, unpinned run/configuration and do not substitute for this corrected four-layer run. No full-model “No Healing” PPL was recorded.
- An older Day 3 value labeled “PPL” compared compressed logits against dense argmax targets (`exp(cross_entropy)`); it is a teacher-agreement diagnostic, not next-token language-model PPL. Day 5 also used a hard-coded external dense reference in one comparison.
- Historical manifests reported 0.3054 first-layer and 0.8117 last-layer relative output errors for a production full cascade. Those figures are consistent with the archived logs, but the current run below is the reproducible measurement used for this investigation.
- Historical serialized production checkpoints occupied 1,446,940,090 bytes across 32 layers (1,379.91 MiB), while the old estimate was 263.21 MB. The old estimate omitted actual FP16 tensor payloads and sparse indices/values; the current estimator records tensor payload separately from serialized file bytes. The format stores floating tensors and is not an int4 bit-packed checkpoint.

## Model, data, and protocol

- Model and tokenizer: `NousResearch/Llama-2-7b-hf`, pinned revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`; tokenizer vocabulary 32,000.
- Dataset for the historical controls below: processed WikiText-2 files from `https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/`, incorrectly labeled raw by the old loader.
- Legacy processed split provenance:

  | Split | Bytes | SHA-256 | Full token count |
  |---|---:|---|---:|
  | train | 10,797,148 | `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f` | 2,780,248 |
  | valid | 1,121,681 | `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8` | 286,348 |
  | test | 1,256,449 | `d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0` | 324,152 |

- Compression calibration used the first 256 tokenized WikiText train tokens; held-out layer validation used the first 128 tokenized WikiText valid tokens. The code generated real token embeddings and propagated them through dense layers 0–14 to form layer 15 inputs. No synthetic or fallback data was used.
- PPL used the same pinned model/tokenizer, raw WikiText test split, `seq_len=2048`, `stride=512`, and first 10,000 test tokens for dense and compressed models. It reports `exp(mean next-token cross-entropy)` over exactly 9,999 scored shifted labels (the first label is ignored). Dense evaluation is included in each run.
- Diagnostic gate: each sequential held-out full-residual layer error at or below 0.10 and compressed PPL at or below 1.25× the matched dense baseline. This gate is a stop condition for scaling to a full-model run, not a claim that all models should meet a universal threshold.

## Commands and results

All commands below ran on the remote host from `/home/shadeform/diag36h-research` with its `.venv`. The compiler run wrote to a separate retry directory after an interrupted foreground SSH session left partial layer 15 artifacts; those partial artifacts were preserved.

### Production four-layer run

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/benchmark_day3_cascade.py \
  --start_layer 15 --end_layer 18 --tier production_18x \
  --model_id NousResearch/Llama-2-7b-hf \
  --model_revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --device cuda --n_calib 256 --n_test 128 \
  --save_dir /home/shadeform/diag36h-research/models/diagnostic_36h_cache \
  --save_checkpoint_dir /home/shadeform/diag36h-research/checkpoints/diagnostic_36h/retry/healed \
  --save_no_healing_checkpoint_dir /home/shadeform/diag36h-research/checkpoints/diagnostic_36h/retry/no_healing \
  --save_validation_activations_dir /home/shadeform/diag36h-research/checkpoints/diagnostic_36h/retry/validation_inputs
```

The dense-held-out sequential scorecard (`retry.log`) gave healed full-residual output relative errors 0.0447, 0.0686, 0.0878, 0.1067 for layers 15–18. Independent/no-healing errors were 0.0449, 0.0688, 0.0874, 0.1060. The endpoint error first exceeds the 0.10 gate at layer 18. Healing did not reduce four-layer residual drift growth: +0.0621 healed vs +0.0611 independent. The block-transform delta errors were 0.9567, 0.9513, 0.9717, 0.9586 for healed, with cosine similarities 0.2960, 0.3185, 0.2510, 0.2937.

Serialized checkpoint sizes: healed 169,023,120 bytes (161.19 MiB; four files); no-healing 174,849,744 bytes (166.75 MiB; four files). The healed manifest estimated 160.99 MiB of tensor payload and recorded the same serialized byte count.

### Matched next-token PPL

Command used for each checkpoint directory (the command was run separately for each of the four paths below):

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/benchmark_day4_perplexity.py \
  --checkpoint_dir CHECKPOINT_DIR \
  --model_id NousResearch/Llama-2-7b-hf \
  --model_revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d \
  --device cuda --eval_dense --seq_len 2048 --stride 512 --max_eval_tokens 10000
```

| Tier/path | Dense PPL | Compressed PPL | Change | Checkpoint bytes |
|---|---:|---:|---:|---:|
| production healed | 20.79 | 30.75 | +47.9% | 169,023,120 (161.19 MiB) |
| production no-healing | 20.79 | 30.63 | +47.3% | 174,849,744 (166.75 MiB) |

Both scored 9,999 next-token labels from the same test SHA above. Neither meets the 1.25× gate. The 0.12 PPL advantage for production no-healing is small relative to the roughly 10-point degradation; healing does not solve the quality problem.

### High-fidelity budget control

The same four-layer command was run with `--tier high_fidelity_10x` and output paths under `/home/shadeform/diag36h-research/checkpoints/diagnostic_36h/high_fidelity_10x/`. It used the same model revision and data/token counts.

| Tier/path | Dense PPL | Compressed PPL | Change | Checkpoint bytes |
|---|---:|---:|---:|---:|
| high-fidelity healed | 20.79 | 30.97 | +48.9% | 264,178,320 (251.94 MiB) |
| high-fidelity no-healing | 20.79 | 30.67 | +47.5% | 280,489,040 (267.50 MiB) |

High fidelity occupies about 56% more bytes than production for the healed artifacts. Its healed sequential output errors were 0.0448, 0.0687, 0.0877, 0.1064; drift growth was +0.0616. The output and PPL barely changed despite the extra bytes. Both tiers use an effective causal bridge rank of 153 at 256 calibration tokens; high fidelity raises the TT core ranks and doubles the configured outlier count. This evidence argues against total serialized size as the sole cause, but does not isolate the fitting algorithm from calibration coverage.

### Saved-checkpoint audit

The audit script was run on both production paths for layers 15–18 with their saved held-out dense-input activations, and on both high-fidelity paths for layer 18:

```bash
.venv/bin/python scripts/audit_checkpoint_reconstruction.py \
  --checkpoint-dir CHECKPOINT_DIR \
  --dense-weights-dir /home/shadeform/diag36h-research/models/diagnostic_36h_cache \
  --layer LAYER --device cuda \
  --activation-file VALIDATION_INPUT_FILE
```

All audited files reported `audit_status=COMPLETE`. Production projection weight relative Frobenius errors were roughly 0.95–1.04 across audited tensors (cosines as low as about 0.08); for example, production healed layer 15 q-projection error was 0.9515 and down-projection error was 1.0359. On the same dense held-out layer inputs, reconstructed full-block output errors were:

| Layer | Production healed | Production no-healing | High-fidelity healed | High-fidelity no-healing |
|---:|---:|---:|---:|---:|
| 15 | 0.044689 | 0.044896 | — | — |
| 16 | 0.052783 | 0.052867 | — | — |
| 17 | 0.055808 | 0.055756 | — | — |
| 18 | 0.058152 | 0.058379 | 0.057957 | 0.058211 |

The production layer 15 checkpoint audit reproduces the compile-time 0.0447 output error to rounding. This gives no evidence that serialization corrupts the compressed operator: loaded checkpoints reconstruct and run, their byte counts match the manifests, and the saved layer output matches compile-time output. Individual matrices are very approximate; the full residual hides some one-layer output error, while replay through prior compressed layers raises the sequential endpoint error to about 0.106. This locates a real cascade effect but does not yet identify the internal solver defect, if any.

### Focused tests and remote environment repair

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result after pinning the remote venv: 6 tests passed. They cover serialized checkpoint size, compiler tensor payload accounting, dense-argmax naming, empty-cache behavior, shifted PPL token counts, and run fingerprints. Remote versions: Torch 2.7.0 with CUDA available, Transformers 4.51.3.

The environment repair command was:

```bash
.venv/bin/pip install transformers==4.51.3
```

## Conclusions and next run

- Legacy measurements show highly approximate weight reconstructions and a four-layer PPL degradation of about 47–49% on processed text. The wrong processed-vs-raw WikiText source and RMSNorm epsilon are both corrected. Dense custom-vs-HF layer-0 parity is now 0.000424 relative error at epsilon 1e-5 (the old 1e-6 path was 0.231429). On corrected raw data, one zero-healing layer-15 replacement still gives PPL 5.6132 vs 5.2922 dense and 89.72% top-1 agreement. The first remaining failing stage is therefore the compressed operator fit/budget, before any full-model cascade can be assessed. Current controls rule out packed serialization as the dominant source; they do not yet distinguish approximation budget from the fitting objective as the main cause.
- Healing provides no meaningful recovery. At both budgets no-healing is marginally better on PPL and endpoint output error.
- Raising from production to high fidelity increases serialized bytes substantially but leaves matrix/output/PPL quality essentially unchanged. Do not scale either configuration to all 32 layers yet.
- Full-model corrected PPL and the intended full-model “No Healing” result remain unverified. The historical full-model PPL values are archived but use the previous run/config and are not a substitute.
- The legacy 1,024-token calibration control completed. One-layer processed-text PPL was 22.43 / 22.40 (healed / no-healing, 256 calibration tokens) versus 22.36 / 22.31 (1,024 tokens), with dense 20.79. No-healing layer checkpoint bytes were 70,112,564 and 70,114,548. Saved same-input output errors were 0.044818 and 0.045021 under the legacy epsilon. Larger calibration had a small PPL benefit but did not repair the shared normalization or dataset bug. These results are retained under `experiment_artifacts/diagnostic_36h/calibration_control/`.
- Hardware/data blocker: none for the four-layer tests on the provided A10. The full 32-layer run is intentionally not started because the four-layer production and high-fidelity controls both fail the quality gate. Continue with small controlled solver/calibration experiments, then rerun full-model PPL only after a validated layer-stage improvement.

## Investigation files changed

- `benchmarks/benchmark_day3_cascade.py`
- `benchmarks/benchmark_day4_perplexity.py`
- `benchmarks/benchmark_day5_multitask.py`
- `kaggle_execution/run_cascade.py`
- `nox/artifacts.py` (new)
- `nox/compiler.py`
- `nox/data.py` (new)
- `nox/sequential_compiler.py`
- `scripts/audit_checkpoint_reconstruction.py` (new)
- `scripts/autonomous_nightly_orchestrator.py`
- `scripts/update_docs_with_results.py`
- `tests/test_pipeline_accounting.py` (new)

## Matched 1,024-token layer-15 prior control (2026-10-16)

The A10 is running a paired no-healing comparison with the same high-fidelity compiler, model, uniform train/validation windows, and ranks. The only setting changed is a fixed dense-weight prior (`diagonal_prior_strength=0` versus `1`). Both runs compile layer 15 only. They are evaluated in the full dense model with all other layers dense, so this is a one-layer intervention, not a full-model result.

The driver ran the following sequence for strengths 0 and 1, serially:

```bash
for strength in 0 1; do
  out="checkpoints/diagnostic_36h/layer15_uniform_prior_control/prior${strength}"
  .venv/bin/python -u scripts/compile_high_fidelity.py \
    --output "$out" --start-layer 15 --end-layer 15 \
    --n-calib 1024 --n-validation 1024 --seq-len 128 \
    --calibration-sampling uniform --validation-sampling uniform \
    --diagonal-prior-strength "$strength"
  .venv/bin/python -u scripts/diagnose_reference_pipeline.py \
    --checkpoint-dir "$out" --weights-dir models/diagnostic_36h_cache \
    --output "$out/evaluation.json"
done
```

The first arm completed in 767.0 seconds. It used 1,024 uniform calibration tokens from raw WikiText train and 1,024 uniform held-out tokens from raw WikiText validation; each consists of eight 128-token windows sampled from the separate split. The pinned WikiText revision is `b08601e04326c79dfdd32d625aee71d232d685c3`, with no fallback or literal `<unk>` in train/validation. The train parquet SHA-256 is `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`; validation is `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`; raw test is `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`. The tokenizer is the same pinned Llama-2-7B revision as the weights. The compiler used high-fidelity ranks `[1,64,128,64,1]`, bridge rank 128, 524,288 configured outliers and epsilon 0.005. The same model/tokenizer scored 10,000 raw WikiText test input tokens, 9,999 shifted labels, context 2,048 and stride 512. This run records the dense baseline alongside its compressed result:

| Measure | Dense | One-layer high-fidelity, no healing, prior 0 |
|---|---:|---:|
| Raw WikiText PPL | 5.2922001 | 5.6134000 |
| Next-token accuracy | 61.0461% | 60.2460% |
| Dense top-1 agreement | — | 89.6990% |
| Serialized layer checkpoint | — | 70,116,148 bytes |
| Same-input layer-15 held-out residual error | — | 0.0517685 |

The 0.0517685 layer-output error passes the project's 0.10 diagnostic threshold. The global quality goals still fail: PPL is 6.07% above dense, while dense top-1 agreement is below 95%. The layer error is not sufficient to predict final-model agreement. The checkpoint is 5.77× smaller than the seven dense FP16 projection weights for this one layer, not evidence of 10× model storage reduction. No healing is enabled (`causal_healing=False`); only layer 15 is replaced.

The strength-1 arm completed in 3,414.0 seconds (56m54s), 4.45× the strength-0 compile time. It produced the same serialized checkpoint size. Its raw PPL changed by only -0.000177 and dense top-1 agreement by +0.020 percentage points; dense next-token accuracy fell slightly. The 4,096-column prior improves matrix reconstruction and local held-out block error modestly, but has no meaningful language-model quality benefit at this setting:

| Measure | Prior 0 | Prior 1 |
|---|---:|---:|
| Raw WikiText PPL (dense 5.2922001) | 5.6134000 | 5.6132231 |
| Raw next-token accuracy (dense 61.0461%) | 60.2460% | 60.2160% |
| Dense top-1 agreement | 89.6990% | 89.7190% |
| Processed-text PPL | 22.0973 | 22.0665 |
| Serialized layer checkpoint | 70,116,148 bytes | 70,116,148 bytes |
| Compile runtime | 767.0 seconds | 3,414.0 seconds |
| Direct saved-checkpoint held-out block error | 0.051808 | 0.051034 |

Both arms score the same 10,000 raw test input tokens and 9,999 shifted labels with the other 31 layers dense. Both fail the 95% dense-agreement goal and exceed the dense PPL by about 6.07%. The prior-1 result does not justify its additional compute.

The saved-checkpoint audit compared all seven reconstructed matrices directly against their dense FP16 references and replayed the 1,024 held-out dense layer inputs. Relative Frobenius errors for prior-0 → prior-1 were: q 0.9151→0.8327, k 0.9105→0.8271, v 0.9361→0.8494, o 0.9721→0.8744, gate 0.9402→0.9171, up 0.9486→0.9261, down 1.0521→0.9703. The local block errors agree with the compiler manifest to rounding (prior-0 0.05177; prior-1 0.05100). Thus the serialized factors reconstruct the saved approximation, but that approximation remains far from dense weights. The dense-weight prior narrows the weight gap without restoring model agreement.

### Packed checkpoint quality check

The decoded packed layer-15 checkpoint was evaluated in the same full dense model and WikiText protocol. Dense PPL was 5.2922001; packed-round-trip PPL was 5.6438975, with 89.5890% dense top-1 agreement. This matches the original 256-token checkpoint's raw result (PPL 5.6438818, agreement 89.5790%) to rounding. The FP32 direct reconstruction audit found 0.0003946–0.0006112 relative weight error per projection and 0.00001023 relative held-out block-output difference across 128 tokens. The original artifact used 70,110,452 bytes; the packed file used 24,346,251 bytes (2.87972× smaller). This rules out pack/unpack serialization as the source of the large quality loss. It is not yet integrated into model inference or a full-model storage measurement.

## Audit metric correction

An FP16 direct packed-checkpoint audit produced impossible cosine values above 1.0. The cause was FP16 reduction/rounding in the generic weight comparison. `scripts/audit_checkpoint_reconstruction.py::compare_weights` now promotes both tensors to FP32 for subtraction, norms and cosine, clamps cosine to its mathematical range, and uses an epsilon floor for a zero reference norm. A 100,000-element FP16 regression test catches reduction overflow. The first test attempt caught a residual FP32 rounding result of 1.00000012; cosine clamping fixed it, and the final remote suite passed **21 tests**:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The repeated CPU audit of the existing packed layer-15 artifact reports per-projection weight relative errors from 0.0003946 (gate) to 0.0006112 (down), and a same-input held-out block reconstruction difference of 0.00001023 (128 tokens). All cosine results are now in [-1, 1]. This establishes round-trip fidelity against the original compressed checkpoint only; it does not establish fidelity to dense weights or full-model quality. Raw outputs are preserved in `experiment_artifacts/diagnostic_36h/remote_records/packing_layer15/direct_audit_fp32.json` and its log.

The full MMLU protocol has been prepared: 14,042 test questions across 57 subjects, 285 five-shot demonstrations, and 11,305,511 total prompt tokens at the 4,096-token cap. The full dense evaluation was started and then stopped after 2,976 predictions to release the A10. Its partial accuracy is not used as a score; the log and prediction prefix remain preserved under `checkpoints/diagnostic_36h/mmlu/`.

For the external reference, the Llama 2 paper reports **45.3%** for base Llama 2 7B on five-shot MMLU (Table 19, [paper](https://arxiv.org/abs/2307.09288)). A 95% threshold against that published result is 43.035%. The harness accepts the score explicitly and marks it as an external, protocol-unmatched reference; any compressed score must be produced by our pinned full-test harness over all subjects. The acceptance number is the single aggregate score; subject scores are diagnostic only. Do not describe the external comparison as matched dense retention. The interrupted partial run is retained only as an execution record.

The MMLU reference-handling tests and full remote suite passed **28 tests** on the A10 host:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Once a full 32-layer high-fidelity checkpoint exists and passes the WikiText gate, its complete MMLU score command is:

```bash
.venv/bin/python -u benchmarks/benchmark_full_mmlu.py \
  --checkpoint-dir FULL_HIGH_FIDELITY_ZERO_HEALING_CHECKPOINT \
  --reference-accuracy 0.453 \
  --reference-source "Llama 2 paper Table 19: base 7B, five-shot MMLU, arXiv:2307.09288" \
  --output checkpoints/diagnostic_36h/mmlu/full_compressed.json \
  --batch-size 1 --max-context 4096
```

This command scores the complete test set and reports the overall score, its ratio to 45.3%, and whether it reaches 43.035%. No compressed MMLU score exists yet.

The exploratory post-compression factor fine-tuning control updated the seven saved layer-15 projection factors jointly for 8 epochs (64 Adam steps) against dense full-block outputs on 1,024 raw WikiText train tokens. It selected by 1,024 separate raw WikiText validation tokens. This changes an already written checkpoint, so it is **healing** and is excluded from the zero-healing engine/results. The initial report's `zero_healing=true` marker was incorrect; the source code now labels this diagnostic as healing, and this note supersedes that field in the retained artifact. Its ranks, bridge, sparse capacity, and serialized size were unchanged.

The control started from the prior-0 layer-15 checkpoint at commit `51c748d`. Held-out block output error changed from **0.05176836 to 0.05111379**; the saved reconstruction audit reproduced the latter (cosine 0.998693), and all seven matrix comparisons completed. The saved artifact is **70,116,852 bytes**, 704 bytes larger than its 70,116,148-byte parent. On the same 10,000 raw WikiText test tokens / 9,999 shifted labels, dense PPL/next-token accuracy were **5.2922001 / 61.0461%**; the original zero-healing parent scored **5.6134000 / 60.2460%**, with **89.6990%** dense top-1 agreement; this post-compression fine-tuned diagnostic scored **5.6080208 / 60.2560%**, with **89.8190%** agreement. It improves the parent only marginally and still has PPL 5.97% above dense and agreement 5.18 percentage points below 95%. This is not a qualifying zero-healing result.

Exact executed sequence on the A10:

```bash
bash scripts/run_block_distillation_control.sh
# The driver called these three commands serially:
.venv/bin/python -u scripts/refine_checkpoint_block.py \
  --checkpoint-dir checkpoints/diagnostic_36h/layer15_uniform_prior_control/prior0 \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/layer15_block_distillation \
  --epochs 8 --learning-rate 1e-4 --patience 3
.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir checkpoints/diagnostic_36h/layer15_block_distillation \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/layer15_block_distillation/evaluation.json
.venv/bin/python -u scripts/audit_checkpoint_reconstruction.py \
  --checkpoint-dir checkpoints/diagnostic_36h/layer15_block_distillation \
  --dense-weights-dir models/diagnostic_36h_cache --layer 15 --device cuda \
  --activation-file checkpoints/diagnostic_36h/layer15_block_distillation/dense_input_layer_15.pt
```

The run fingerprint was `a4ae289663e4c46b8092f4f43478b2dacf5717e93b7c7693494b8d1a8a954d`; optimization took 8.361 seconds. Logs and small reports are copied locally under `experiment_artifacts/diagnostic_36h/remote_records/layer15_block_distillation/`; the 70 MB checkpoint remains on the A10. The 28-test suite also ran there and passed in 2.511 seconds.

After correcting the healing classification in commit `a34fe6d`, the remote full suite passed **29 tests** in 2.469 seconds. `classification_correction.json` beside the retained remote checkpoint explicitly supersedes its original incorrect zero-healing metadata.

No four-layer or full 32-layer run follows from this failed one-layer gate. The next engine experiment must change the compressor's initial construction/fitting procedure and write a final checkpoint directly, with no optimizer applied afterward. Any eventual 32-layer sweep will use only the high-fidelity configuration with zero healing.

## Exact attention gauge experiment

**Question:** can an exact reparameterization of attention move weight structure into a form the existing high-fidelity codec stores more accurately, without changing the dense block function?

The candidate applies reciprocal per-channel scaling to Q/K and V/O. A first Q/K version used independent scales for each channel; its dense block parity test failed by 4.2%, because those scales did not commute with cross-position RoPE rotations. That version was not compressed. The corrected transform ties each Q/K scale across the two coordinates mixed by Llama RoPE, which is required for the exact symmetry. On the A10, the corrected dense parity errors were `4.9848571e-8` on calibration inputs and `6.0965924e-8` on held-out inputs.

The A10 one-layer comparison started from source commit `a34fe6d`; the only compressed layer was 15. It used pinned `NousResearch/Llama-2-7b-hf` revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`, high-fidelity ranks `[1,64,128,64,1]`, bridge rank 128, 524,288 configured sparse outliers, diagonal prior 0, and epsilon 0.005. Calibration used 1,024 uniformly sampled tokens from raw WikiText train; validation used 1,024 disjoint uniformly sampled tokens from raw WikiText validation; each used eight 128-token windows. The dataset revision was `b08601e04326c79dfdd32d625aee71d232d685c3`; train and validation Parquet SHA-256 values were `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7` and `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`. There was no fallback data or literal `<unk>`.

Exact commands on the remote A10:

```bash
.venv/bin/python -m unittest discover -s tests -p test_attention_gauge.py -v
.venv/bin/python -u scripts/compile_high_fidelity.py \
  --output checkpoints/diagnostic_36h/gauge_balanced_layer15 \
  --start-layer 15 --end-layer 15 --n-calib 1024 --n-validation 1024 \
  --seq-len 128 --calibration-sampling uniform \
  --validation-sampling uniform --attention-gauge balanced
.venv/bin/python -u scripts/audit_checkpoint_reconstruction.py \
  --checkpoint-dir checkpoints/diagnostic_36h/gauge_balanced_layer15 \
  --dense-weights-dir models/diagnostic_36h_cache --layer 15 --device cuda \
  --activation-file checkpoints/diagnostic_36h/gauge_balanced_layer15/dense_input_layer_15.pt
.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir checkpoints/diagnostic_36h/gauge_balanced_layer15 \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/gauge_balanced_layer15/evaluation.json
.venv/bin/python -m unittest discover -s tests -v
```

The gauge run took 766.724 seconds and wrote a 70,116,468-byte checkpoint, 320 bytes larger than the matched no-gauge prior-0 control (70,116,148 bytes). Both are 5.77x smaller than the seven dense FP16 projection payloads for this layer (404,766,720 bytes). The saved-checkpoint audit completed: held-out layer-output relative error was 0.051782 (control 0.0517685); matrix relative errors for Q/K/V/O/gate/up/down were 0.918029/0.907921/0.935408/0.973156/0.940183/0.948645/1.051585. The artifact faithfully reconstructs its own compressed approximation, but its weights remain far from the dense matrices.

On the same 10,000 raw WikiText test input tokens and 9,999 shifted labels (context 2,048, stride 512), dense PPL/next-token accuracy were 5.2922001/61.0461%. The gauge checkpoint scored PPL 5.6152387, next-token accuracy 60.1760%, and dense top-1 agreement 89.63897%; PPL was 6.1041% above dense. The matched no-gauge control scored PPL 5.6134000, accuracy 60.2460%, and agreement 89.69897% (6.0693% PPL increase). The gauge therefore worsened PPL by 0.0328% relative to the control and reduced agreement by 0.0600 percentage points. It did not strengthen the engine; these differences are negligible and slightly adverse.

The focused gauge tests passed (2 tests), and the full remote suite passed **31 tests** in 2.545 seconds. The test and evaluation records are copied under `experiment_artifacts/diagnostic_36h/remote_records/gauge_balanced_layer15/`; the checkpoint and raw calibration inputs remain on the A10. No base-model fine-tuning or post-compression healing ran. The compressor did perform its normal DMRG/local least-squares factor construction on the calibration activations as part of writing the compressed artifact; no optimizer updated the checkpoint after that artifact was written.

This experiment covers one replaced layer with the other 31 layers dense. It does not verify full-model PPL, the 32-layer cascade, or MMLU. Do not scale this gauge transform to the full model on the strength of this result. The next useful step is a different, evidence-led initial encoding change or retaining the current high-fidelity codec if the approximately 90% agreement / 6% PPL gap is acceptable.
