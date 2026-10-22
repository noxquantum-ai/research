# NEC prefix KL and dense-boundary controls (A40)

This diagnostic reassembled the saved four-layer NEC checkpoint in several
configurations and compared each model's next-token distribution with the
pinned dense Llama-2-7B teacher. It did not compile, fit, or save model weights.
Layers 4–31 remained dense in every configuration.

## Matched short-slice results

The PPL and KL metrics use the same 4,095 causal targets from the first 4,096
WikiText-2 raw validation tokens (five windows, context 2,048, stride 512).
KL is the mean tokenwise `KL(P_dense || P_student)` in nats, with the teacher
and student distributions computed from their full-model logits. Both logits
come from the FP16 model; softmax and KL accumulation use FP32.

| Configuration | Compressed layers | PPL | Mean KL | Dense top-1 agreement | Bytes saved across layers 0–3 |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense reference | none | 4.083 | — | 100.00% | 0 |
| NEC prefix | 0–3 | 233.499 | 4.069 | 28.45% | 1,339,108,280 |
| Keep layer 0 dense | 1–3 | 44.867 | 2.409 | 43.39% | 1,004,157,306 |
| Keep layer 3 dense | 0–2 | 30.384 | 2.037 | 48.55% | 1,004,439,930 |
| Keep layers 0 and 3 dense | 1–2 | 7.543 | 0.630 | 72.94% | 669,488,956 |

The attention-only control had PPL 4.856; the MLP-only control had PPL 106.443.
These controls do not have KL entries in this run.

Keeping both prefix endpoints dense materially improves this four-layer
checkpoint: mean KL falls by 84.5% and the short-slice PPL falls from 233.50 to
7.54. Keeping either endpoint dense alone does not produce the same recovery.
The result supports boundary controls for this checkpoint and budget; it does
not establish a universal layer-selection rule.

## Storage accounting

The all-dense baseline is the exact FP16 payload of the seven projections and
two normalization tensors in layers 0–3: **1,619,066,880 bytes**. Compressed
storage uses the actual serialized `compressed_layer_N.pt` file sizes, including
serialization and checkpoint metadata. The four compressed checkpoints total
**279,958,600 bytes**. Keeping layers 0 and 3 dense stores **949,577,924 bytes**
for these four layers and saves **669,488,956 bytes** against the all-dense
baseline. These figures exclude the common dense suffix (layers 4–31) and other
model files.

## Reproduction details

- Model: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`; WikiText-2 raw validation at
  dataset revision `b08601e04326c79dfdd32d625aee71d232d685c3`.
- GPU: NVIDIA A40; PyTorch `2.11.0+cu128`.
- Source commit: `74d2960674a3ebfbfffa3f6f9073efaa2bd0afa4`.
- Checkpoint source on the persistent RunPod volume:
  `/workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg/`.
- Full machine-readable report, including checkpoint hashes, token hashes,
  per-variant metrics, storage accounting, and activation traces:
  [`report.json`](report.json).

This is a 4,096-token development diagnostic, not a full-validation evaluation
or a WikiText-2 test result. The 7.54 PPL value should not be read as the
full-split quality of the boundary-protected model; that variant still needs a
matched full-validation run before making a quality claim.
