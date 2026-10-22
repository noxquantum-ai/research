# Four-layer NEC prefix evaluation (A40)

This record covers a four-layer prefix of Llama-2-7B: all seven supported
projections in layers 0–3 use High-Fidelity NEC; layers 4–31 remain dense. The
language-modeling result is a full raw WikiText-2 **validation** split
development measurement, not the WikiText-2 test benchmark. MMLU was not run.

## Paired PPL result

| Variant | PPL | Mean NLL (nats/token) | Next-token accuracy |
|---|---:|---:|---:|
| Dense reference | 4.94691 | 1.59876325 | 62.3064% |
| NEC, layers 0–3 | 294.21950 | 5.68432610 | 21.3746% |

Both models scored the same 299,963 next-token targets from 299,964 tokenizer
input tokens (583 windows; context 2,048, stride 512, batch size 1). The paired
mean-NLL increase, NEC minus dense, is 4.08556285 nats/token, with a 95%
contiguous-block bootstrap interval [4.04948561, 4.12093126] (512-token blocks,
5,000 repetitions, seed 20261021). The PPL ratio is 59.4754×; next-token
accuracy decreases by 40.9317 percentage points, and paired top-1 agreement is
25.9355%. NEC does **not** meet the PPL < 6 target.

The split is `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, validation. The source Parquet
SHA-256 is
`204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`; its raw
joined-text SHA-256 is
`b44fb967f92b525731216a507d1fbc44b97446b7ef0a2b4d1a893f81f1bbc29e`. The
scored token ID stream and per-token NLL / top-1 sidecars are preserved under
`evaluation/` for independent paired analysis.

## Compilation and artifact details

The pinned model and tokenizer are
`NousResearch/Llama-2-7b-hf@8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
Compilation used the existing High-Fidelity NEC configuration: MPO ranks
`[1,64,128,64,1]`, bridge rank 128, sparse budget 524,288 per projection, two
DMRG sweeps, AWT exponent 0.5, post-sweep groupwise Q4 simulation, and 48
matrix-free CG iterations above width 4,096. The construction was independent
with dense calibration activations, zero diagonal prior, and no gradient
healing. RMSNorm epsilon was `1e-5`.

Calibration used 4,096 uniformly selected WikiText-2 raw training tokens in
128-token sequences (seed 42); compiler-held-out validation used 1,024 tokens
(seed 43). Dataset provenance, selected window offsets, token hashes, run
fingerprint, and per-file checkpoint hashes are recorded in
`checkpoint_metadata/model_manifest.json`. The manifest records source
fingerprint `8016bbab30c8753a9ecd70420984eacc94b9616eeb8cc10ecba4a28acc8f3d42`;
the compile execution record does not contain a Git commit SHA. The PPL
evaluation reports were generated at source commit
`cd1583e33620fc64c8be5011191f55c54689a99f`.

The audited model contains 6,010,309,113 independent numerical coefficients,
versus 6,738,415,616 dense coefficients: 10.8053% whole-model structural
reduction (1.1211×). The four serialized layer artifacts total 279,958,600
bytes. They are intentionally not committed as `.pt` files, consistent with
the repository's artifact policy; their byte counts and SHA-256 hashes are in
the manifest, and the original files remain on the persistent RunPod volume at
`/workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg/`.
Q4 is simulated and factors are serialized as FP16; this is not packed-int4
native inference, and the PPL evaluator reconstructs dense layer weights.

| Layer | Calibration relative error | Same-input validation error | Cascade validation error |
|---:|---:|---:|---:|
| 0 | 0.23435 | 0.25768 | 0.25768 |
| 1 | 0.04987 | 0.03007 | 0.15115 |
| 2 | 0.00727 | 0.00868 | 0.15112 |
| 3 | 0.01016 | 0.01192 | 0.15177 |

These are compiler activation-reconstruction diagnostics from the manifest,
not language-model metrics. The prefix extension stage took 29,394 s (8 h 9 m
54 s; estimated GPU-only cost $4.817 at $0.59/hour). The matched dense and NEC
PPL passes plus comparison took 493 s total (estimated GPU-only cost $0.0808).
These estimates exclude the earlier layer-0 compilation, idle time, storage,
and other RunPod charges. The GPU was an NVIDIA A40 with 46,068 MiB reported
VRAM; maximum per-layer PyTorch allocation in the compiler history was about
15.31 GiB (16,440,082,944 bytes).

## Archived files

- `evaluation/dense.json`, `evaluation/nec.json`: complete evaluator records.
- `evaluation/comparison/report.json`: paired metrics and bootstrap interval.
- `evaluation/nec.checkpoint_audit.json`: checkpoint integrity and coefficient
  audit.
- `evaluation/*.npy`, `evaluation/*.i32`: per-token NLL and top-1 sidecars.
- `evaluation/*/last_execution.json`: dense, NEC, and comparison execution
  records.
- `checkpoint_metadata/`: compiler manifest and successful prefix-extension
  execution record.
- `logs/`: compile, full-split PPL, comparison, and heartbeat logs. The dense
  PPL log also retains the earlier interrupted attempt followed by the
  completed restart.

The PPL reports, token sidecars, comparison, and checkpoint audit were
re-verified after transfer. No MMLU or WikiText-2 test-split result is included.
