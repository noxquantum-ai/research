# Full-validation NEC boundary control (A40)

This run evaluates the saved four-layer NEC checkpoint with layers 0 and 3 kept
dense and layers 1 and 2 compressed. Layers 4–31 remain dense. No model was
recompiled or refit.

## Results

| Configuration | PPL | Mean teacher KL | Next-token accuracy | Dense top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Fully dense reference | 4.94691 | — | 62.306% | 100.000% |
| Compress layers 1–2 | 9.56570 | 0.67471 nats/token | 53.119% | 70.481% |

Both variants score the same **299,963 next-token targets** from 299,964
validation input tokens. KL is `KL(P_dense || P_student)`, averaged over those
targets. The teacher and student distributions use FP16 model logits with FP32
log-softmax and KL accumulation. The split uses 583 windows (context 2,048,
stride 512).

The boundary-protected configuration is below the project's PPL 10 tolerance
on the full validation split. It retains **669,488,956 bytes** of savings across
layers 0–3 against the dense FP16 projection-and-norm payload, storing
949,577,924 bytes for that region instead of 1,619,066,880. It still has a
higher PPL and lower next-token accuracy than the dense reference, so this is a
local mixed-allocation result, not a full-model compression result.

## Reproduction details

- Model: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Dataset: WikiText-2 raw validation, pinned dataset revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`; source Parquet SHA-256
  `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`.
- GPU: NVIDIA A40; PyTorch `2.11.0+cu128`.
- Source commit: `86b246463b0d64e7230aa150286ae15bdf8b7921`.
- Checkpoints: persistent RunPod volume at
  `/workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg/`.
- [`report.json`](report.json) contains token and checkpoint hashes, protocol,
  per-variant metrics, exact KL quantiles, and storage accounting.
