# Full NEC capability benchmark protocol

Evaluate only the completed 32-layer NEC hybrid. No new dense run is requested.
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
| TriviaQA | `rc.nocontext` validation | 0 | Alias exact match, ignore case/punctuation | Verify against dataset |
| GSM8K | Test | 5 | Strict `####` answer exact match | 1,319 |

Validation is used where official test labels are unavailable. Dataset revisions
are resolved to immutable hashes before loading; split row counts/fingerprints
and the harness's complete task configurations are saved with results. The
runner checks full effective sample counts and saves per-example responses.
No measured score will be published until its complete split has finished.

The exported package's bundled decoder reconstructs NEC weights in FP16.
The A40 uses automatic batch sizing (ceiling 32), shared-context choice-logit
caching and local-scratch SQLite response caching, with consistent snapshots
to the network volume after each task. One loaded model serves batched requests;
five concurrent model copies would compete for the same GPU memory. Results
are saved separately per task, permitting completed tasks to survive restarts.
Evaluation measures capability; it does not measure compressed-kernel speed.

Runner: [evaluate_nec_full_suite.py](../../benchmarks/evaluate_nec_full_suite.py).
Persistent remote directory:
`/workspace/runs/nec-full-five-benchmarks-20261023`.
Inspect `status.json`, `protocol.json`, `datasets.json`, `run.log` and
`<task>_summary.json`. Full raw results use gzip JSON containers.

## External reference comparison

The user-supplied scores also appear in
[CompactifAI, table 2](https://arxiv.org/html/2401.14109v2). The paper names the
LM Evaluation Harness but does not provide sufficient prompt/few-shot details
to establish a matched protocol. Reference columns are not our dense results
and must not be treated as a controlled accuracy-retention comparison.

| Benchmark | Original reference | INT4 reference | CompactifAI 88% | CompactifAI 93% | NEC 5.413× |
| --- | ---: | ---: | ---: | ---: | ---: |
| MMLU | 46.41 | 45.53 | 45.32 | 44.16 | Pending |
| HellaSwag | 80.55 | 79.25 | 77.87 | 76.54 | Pending |
| BoolQ | 79.76 | 78.19 | 77.90 | 76.77 | Pending |
| TriviaQA | 19.03 | 19.00 | 18.33 | 18.10 | Pending |
| GSM8K | 23.05 | 22.44 | 22.58 | 17.74 | Pending |
