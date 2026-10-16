# Compression pipeline investigation (36-hour window)

Started 2026-10-16. This log records the local repository snapshot, prior evidence, and the reproducible remote A10 runs. The remote staging tree is `/home/shadeform/diag36h-research`; checkpoints and run logs are retained there under `checkpoints/diagnostic_36h/`.

## Current scope and corrections (2026-10-16)

The user narrowed the target to one high-fidelity engine with zero healing, and explicitly authorized incremental pushes directly to `main`. The agreed quality checks are raw WikiText PPL within 5% of matched dense, at least 95% dense top-1 agreement, and MMLU accuracy at least 95% of the matched dense score. Full-model serialized size must include all tensors needed for inference. The high-fidelity name denotes a compression target, not an achieved ratio.

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

Its compilation is in progress. This command has no healing or alternate regime option.

Correction to an intermediate status message: historical raw logs **are present** at the paths below. Git's `*.log` ignore rule hid them from the default file search. `nox_big_guns.log` lines 2860–2900 show the failed canonical dataset load, processed fallback, dense PPL 19.75, and compressed PPL 10,916.44. Those logs substantiate the recorded number, but do not by themselves isolate how much each compression defect contributed.

## Repository and execution snapshot

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

- Legacy measurements show highly approximate weight reconstructions and a four-layer PPL degradation of about 47–49% on processed text. The later dense-parity check found a wrong RMSNorm epsilon, so the custom calibration/reference stream was not faithful to HF inference. The first failing stage now established is dense layer-0 normalization. Corrected compression experiments are needed to separate its contribution from fitting, quantization and cascade loss.
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
