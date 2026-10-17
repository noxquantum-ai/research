# High-Fidelity NEC layer-0 smoke record

Status: passed. This is a one-layer execution/correctness smoke only; it is not a full-model benchmark and provides no PPL or MMLU result.

- Kaggle kernel: `toheebogunade/nec-high-fidelity-compile-stage-00`, version 3
- Model: `NousResearch/Llama-2-7b-hf`, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
- NEC source bundle commit: `425cd41770855b48da7ff0bf45ee5bb46a5c0ff6`; source manifest SHA-256 `98b8255d4dc1e7abd60b84d6cf69fe5fec83b4a87fe30bab3262a560021a302b`
- Kaggle launcher SHA-256: `9bc11ab20d6263f865155982319dda844f3c5c565fa531090da77e6118af4450`
- Calibration: pinned WikiText-2 raw train/validation revision `b08601e04326c79dfdd32d625aee71d232d685c3`, 256 train and 128 validation tokens, sequence length 128, prefix sampling, seed 42/43; Llama-2 RMSNorm epsilon `1e-5`.
- Configuration: High-Fidelity `(1,64,128,64,1)` MPO bonds, bridge rank 128, sparse budget 524,288, two MPO sweeps, groupwise Q4 size 64, FP16 serialization, no healing.
- Runtime: Python 3.13.15; PyTorch 2.11.0+cu128; CUDA 12.8; Transformers 5.16.1; datasets 4.8.5; two Tesla T4 GPUs (15,360 MiB each), 33.66 GB RAM, 20.96 GB Kaggle working disk. This layer used GPU 0 (peak observed 13,885/15,360 MiB); GPU 1 was idle. Minimum observed available RAM was about 28.65 GB; working-disk free space stayed above 20.93 GB.
- Time: 701.36 s compiler time; 797.84 s stage time; wall clock 21:50:42–22:04:00 UTC.
- Result: layer 0 contains all seven projections, with expected shapes and bonds. Independent serialized-vs-in-memory reconstruction checks passed; per-projection relative Frobenius discrepancies were `0.000238`–`0.000437`. All seven MPO artifacts passed shape, bond, dtype, finiteness, sparse-index, bridge, and positive-scale validation.
- Layer artifact: 69,813,506 bytes; SHA-256 `a801cb2884a9ea8cc5ccdad4cf019c3dcf35cab74f6163d8efc841b1f499c656`. The persistent binary checkpoint is retained in the private Kaggle kernel output and is attached to the next stage; the manifest and logs are committed alongside this note.
- Precision audit across 49 stored components: zero Q4 clipping events; 3,360,227 nonzero values rounded to zero by Q4; zero FP16 underflows-to-zero, subnormals, overflows, or nonfinite values. Q4 rounding-to-zero is recorded separately from FP16 underflow and is not represented as lossless.
- Projection structural coefficient counts from serialized tensors: q 2,628,244; k 2,629,132; v 2,612,699; o 2,633,028; gate 3,273,812; up 3,273,152; down 3,280,963 (20,331,030 total for this layer’s seven projections only).

Files here preserve the stage progress record, model manifest, compiler log, and full Kaggle kernel log. The model revision and run configuration are also recorded in `kaggle_execution/nec_high_fidelity/run_config.json`.
