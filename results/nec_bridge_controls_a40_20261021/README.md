# NEC real-activation bridge controls on A40 (2026-10-21)

This is a bridge-only diagnostic on the real Llama-2-7B layer-0 `down_proj`.
It compares the direct-SVD reference and FP64 Gram-statistics paths before
TT-SVD or DMRG. It is not a full projection, perplexity, or MMLU result.

## Reproducibility

- Model: `NousResearch/Llama-2-7b-hf`
- Revision: `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
- Weight shape: `4096 x 11008` (`down_proj`, layer 0)
- Bridge rank: 128; active calibration rank: 256
- Calibration: 4,096 uniformly sampled WikiText-2 raw training tokens
- Validation: 1,024 uniformly sampled WikiText-2 raw validation tokens
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Source commit: `b412f50c8fd655a7112b5d6c1cc698848e708954`
- GPU: NVIDIA A40, 46,068 MiB
- Successful run: 68 seconds; estimated GPU cost: $0.0111 at $0.59/hour
- Maximum heartbeat-observed VRAM: 9,767 MiB (sampled, not allocator peak)

The first invocation completed its calculation but could not save because the
wrapper had not created the output directory. Commit `b412f50` fixed that
deterministic wrapper error; the clean rerun passed. The append-only log keeps
both attempts.

## Direct/streaming bridge disagreement

| Bridge representation | Weight | Calibration output | Held-out output |
|---|---:|---:|---:|
| Unquantized FP32 | 0.0733% | 0.0407% | 0.0348% |
| FP16 factors | 0.0840% | 0.0508% | **0.0463%** |
| Independent factorwise Q4 | 4.701% | 2.626% | 2.480% |
| Operator-refit Q4 | 13.220% | 9.361% | 9.517% |

FP16 preserves the near-equivalence of the unquantized direct and Gram bridge
operators. Independent Q4 amplifies held-out disagreement by about 71 times
relative to unquantized factors. The operator-refit experiment is a negative
result: it reduces each backend's own bridge quantization error (approximately
21.91% to 20.47%) but makes the two backends diverge substantially more. It
therefore must not replace the historical Q4 bridge.

This establishes a precise bridge-level result only. The deterministic CPU
stress test showed that later MPO initialization and DMRG can amplify even an
FP16 bridge difference, so a matched full-projection FP16-bridge control is
required before considering any compiler change.

Files:

- `report.json`: complete machine-readable scientific result.
- `last_execution.json`: successful wrapper status, time, and cost.
- `logs/bridge-diagnostic.log`: append-only stdout/stderr for both invocations.
- `logs/bridge-diagnostic.heartbeat.jsonl`: resource telemetry.
