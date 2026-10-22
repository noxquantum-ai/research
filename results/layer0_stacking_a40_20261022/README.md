# Layer-0 NEC stacking qualification (A40)

This record evaluates all seven projections in Llama-2-7B layer 0, then measures
the resulting one-layer prefix against the pinned dense model on the complete
WikiText-2 **validation** split. It is a development result, not a full-model
evaluation; the WikiText-2 test split and MMLU were not run.

## Paired validation result

| Variant | PPL | Mean NLL | Next-token accuracy |
|---|---:|---:|---:|
| Dense reference | 4.94691 | 1.59876325 | 62.3064% |
| NEC, layer 0 (seven projections) | 5.05998 | 1.62136206 | 61.9093% |

Both variants scored the same 299,963 next-token targets from 299,964 tokenizer
input tokens (583 sliding windows, context 2,048, stride 512, batch 1). The NEC
to dense PPL ratio is 1.02286. Paired top-1 agreement is 94.0256%; NEC top-1
accuracy is 0.3970 percentage points lower. The paired mean-NLL increase is
0.0225988 nats/token, with a 95% contiguous-block bootstrap interval
[0.0209021, 0.0245549] using 512-token blocks, 5,000 repetitions and seed
20261021. This is encouraging for one compressed layer only and does not
establish stacked-model or task performance.

The exact dataset is `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, `validation` split. Its downloaded
Parquet SHA-256 is
`204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`; the
tokenized text SHA-256 is
`b44fb967f92b525731216a507d1fbc44b97446b7ef0a2b4d1a893f81f1bbc29e`.
The pinned model and tokenizer are
`NousResearch/Llama-2-7b-hf@8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
Text was joined with two newlines; no chat template was applied. Causal-shifted
cross-entropy scores each target once. Full per-token NLL and top-1 sidecars are
included in `evaluation/` for independent paired analysis.

## Compression and execution

The run used the existing High-Fidelity configuration: four-site MPO ranks
`[1,64,128,64,1]`, bridge rank 128, requested sparse budget 524,288, two DMRG
sweeps, AWT exponent 0.5, post-sweep groupwise Q4 simulation, 48 matrix-free CG
iterations above width 4,096, zero diagonal prior, independent/dense
construction, and no gradient healing. Calibration used 4,096 uniformly
sampled WikiText-2 raw training tokens in 128-token sequences (seed 42); the
held-out compiler validation set used 1,024 tokens (seed 43). The complete
calibration and held-out window offsets and hashes are in
`checkpoint_metadata/model_manifest.json`. The Q4-simulated factors are stored
as FP16 in the checkpoint; this is not a packed-int4 inference result.

The artifact audit counted 20,331,152 independent projection coefficients in
the compressed layer versus 202,375,168 dense layer-0 projection coefficients:
89.9537% structural reduction (9.954x). Across the whole model, because only
one of 32 layers is compressed, the verified count is 6,556,371,600 versus
6,738,415,616 coefficients, or 2.7016% structural reduction (1.0278x). The
serialized layer checkpoint is 69,815,746 bytes versus 404,766,720 bytes for
the layer's FP16 dense tensors (5.798x byte ratio, 82.7516% reduction). These
are distinct structural and serialized-byte measures.

Compiler execution took 11,289 s wall time (11,246 s recorded by the compiler),
with 15,978,709,504 bytes peak PyTorch CUDA allocation and 16,181,624,832 bytes
reserved. Dense and NEC PPL passes took 244.3 s and 238.2 s respectively, with
peak allocated CUDA memory 14,599,479,808 and 14,747,842,048 bytes. The host was
an NVIDIA A40 (46,068 MiB reported VRAM); GPU-only estimated cost for the
successful compile and evaluation invocations was approximately $1.93 at
$0.59/hour, excluding idle, storage and pod overhead.

The compiler ran from source commit `122077bc900b906c186e4603dbd359a52ea1bb48`;
the paired evaluation reports source commit
`7ac14f00d020a2f89c7325ab5341a13679368395`. The latter includes the CUDA
allocator-context fix required by the pod's PyTorch 2.11 build. Its targeted
production-evaluation test suite passed (23 tests). The complete original
compressed checkpoint is not committed here; it remains on the persistent
RunPod volume at
`/workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg/compressed_layer_0.pt`,
SHA-256 `d9ca8e0b8264b806ce7677044e2d4df49c0232dde2f1fcacd3f6c8fb76ee0b09`.

## Files

- `evaluation/comparison.json`: paired metrics and bootstrap interval.
- `evaluation/dense.json`, `evaluation/nec.json`: full protocol and runtime records.
- `evaluation/nec.checkpoint_audit.json`: verified coefficient counts,
  checkpoint integrity and reconstruction audit.
- `evaluation/*.npy`, `evaluation/*.i32`: per-token NLL and top-1 sidecars.
- `checkpoint_metadata/`: original layer manifest and successful compile record.
- `logs/`: compiler, dense/NEC evaluation, paired-comparison, heartbeat and
  launch logs, including the two evaluator startup failures before the API fix.

Reproduce on the same persistent RunPod workspace after pulling the recorded
source revision and bootstrapping its pinned environment:

```bash
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh layer-reference
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh layer0-dev-ppl
```

The second command is safe to rerun only when its result files are absent or
pass the runner's fingerprint validation; do not overwrite the archived
results. The current repository includes the later allocator-API fix in
`benchmarks/evaluate_high_fidelity_nec.py`.
