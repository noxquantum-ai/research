# Full NEC capability benchmark protocol

Evaluate the completed 32-layer NEC hybrid and the original dense model with the same harness, prompts and full splits. Both are run sequentially on the same A40.
Model: `Fluxx08/NEC-Llama-2-7B-Hybrid`, private HF revision
`3b6f1f379048e52bb982754cd7e8e144c54cfbe7`.
Actual published storage: 2,490,273,288 bytes (5.412758× vs dense FP16).
Logical coefficients: 3,323,032,892 (2.027791× structural reduction).

## Frozen scoring choices

LM Evaluation Harness **0.4.11**, original base-model prompts without a chat
template; seeds random=0, numpy/torch/fewshot=1234; maximum context 4,096.
Generation uses the task's greedy generation defaults and stopping strings.
No evaluation `limit` is used. Retain secondary metrics as well as these
prespecified primary metrics:

| Task | Full public labeled split | Few-shot | Primary metric | Expected questions |
| --- | --- | ---: | --- | ---: |
| BoolQ | Validation | 0 | Accuracy | 3,270 |
| HellaSwag | Validation | 0 | Length-normalized accuracy | 10,042 |
| MMLU | Test, all 57 subjects | 5 | Question-weighted accuracy | 14,042 |
| TriviaQA | `rc.nocontext` validation | 0 | Alias exact match, ignore case/punctuation | 17,944 |
| GSM8K | Test | 5 | Strict `####` answer exact match | 1,319 |

Validation is used where official test labels are unavailable. Dataset revisions
are resolved to immutable hashes before loading; split row counts/fingerprints
and the harness's complete task configurations are saved with results. The
runner checks full effective sample counts and saves per-example responses.
No measured score will be published until its complete split has finished.

The exported package's bundled decoder reconstructs NEC weights in FP16; the dense arm loads the original FP16 checkpoint. The A40 uses automatic batch sizing (ceiling 32), shared-context choice-logit caching and local-scratch SQLite response caching, with consistent snapshots to the network volume after each task. The arms run sequentially and save independent results. Evaluation measures capability; it does not measure compressed-kernel speed.

Runner: [evaluate_nec_full_suite.py](../../benchmarks/evaluate_nec_full_suite.py).
Persistent remote directory:
`/workspace/runs/nec-full-five-benchmarks-20261023` for NEC, and
`/workspace/runs/dense-full-five-benchmarks-20261023` for dense. Inspect each
`status.json`, `protocol.json`, `datasets.json`, `run.log` and
`<task>_summary.json`. Full raw results use gzip JSON containers. The runner
accepts `--model-kind dense --dense-model-dir <local-model-snapshot>` for the
matched dense arm.

## External reference comparison

The user-supplied scores also appear in
[CompactifAI, table 2](https://arxiv.org/html/2401.14109v2). The paper names the
LM Evaluation Harness but does not provide sufficient prompt/few-shot details
to establish a matched protocol. Reference columns are not our dense results
and must not be treated as a controlled accuracy-retention comparison.

| Benchmark | Original reference | INT4 reference | CompactifAI 88% | CompactifAI 93% | Dense, matched harness | NEC 5.413× |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMLU | 46.41 | 45.53 | 45.32 | 44.16 | Pending | Pending |
| HellaSwag | 80.55 | 79.25 | 77.87 | 76.54 | Pending | Pending |
| BoolQ | 79.76 | 78.19 | 77.90 | 76.77 | **79.36** | **48.50** |
| TriviaQA | 19.03 | 19.00 | 18.33 | 18.10 | Pending | Pending |
| GSM8K | 23.05 | 22.44 | 22.58 | 17.74 | Pending | Pending |

Matched BoolQ used the complete 3,270-question validation split: dense 79.36%,
NEC 48.50%, a 30.86 percentage-point gap. Raw reports are in
[`progress/`](progress/).
