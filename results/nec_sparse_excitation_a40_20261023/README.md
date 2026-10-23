# Sparse-excitation NEC: staged full-32-layer trial

Target: all 32 transformer layers compressed, held-out PPL <= 10, actual
complete deployable package compression >= 5x, and structural coefficient
reduction >= 2x, simultaneously, with no gradient-based healing. **All three
minimum gates have now been demonstrated together for the complete model.**


## Completed full-model result

| Metric | Dense | NEC |
| --- | ---: | ---: |
| Complete published bytes | 13,479,247,511 | 2,490,273,288 |
| Logical weight/factor coefficients | 6,738,417,664 | 3,323,032,892 |
| Storage / structural reduction | 1× / 1× | **5.412758× / 2.027791×** |
| Held-out test PPL | 4.867285 | **8.348960** |
| Teacher KL, nats/token | 0 | 0.543950 |
| Next-token accuracy | 62.8199% | 53.9990% |
| Dense top-1 agreement | 100% | 71.0259% |

All 32 layers and all seven projections are NEC-compressed. No transformer
layers remain dense. The held-out range contains 308,700 scored tokens after
WikiText-2 test offset 32,768; context 2,048, stride 512. It is not the entire
test split. The final assembly meets PPL ≤10, actual storage ≥5× and structural
reduction ≥2× simultaneously. Dense-equivalent accuracy and 10× compression
remain unachieved; downstream capability benchmarks are pending.

The complete plain package is 2,641,936,119 bytes (5.102034×). Lossless gzip
containers reduce the final published repository to 2,490,273,288 bytes,
including the model card and publication files. Both exported loaders reproduce
original checkpoint-injection logits bit for bit. Logical coefficients are
unchanged by the lossless containers. The runtime reconstructs dense matrices;
no compressed GPU execution or throughput improvement is established.

The private model repository is
[Fluxx08/NEC-Llama-2-7B-Hybrid](https://huggingface.co/Fluxx08/NEC-Llama-2-7B-Hybrid),
immutable revision `3b6f1f379048e52bb982754cd7e8e144c54cfbe7`.
Its LFS payload sizes and hashes were checked against the actual uploaded
files. Model weights were not downloaded to the local workstation.

See [final measured reports](final/summary.json),
[repository bytes and hashes](final/hf_repository_measurement.json),
[component accounting](final/representation_breakdown.json),
[lossless runtime parity](final/gzip_runtime_parity.json), and
[paired qualitative generations](final/paired_greedy_examples.json).
Those examples include arithmetic failures; they are not a capability benchmark.
A zero-main-MPO ablation causes PPL 103,570 on the short development slice.
That checks contribution within this representation, not a matched reoptimized
sparse baseline or evidence of a quantum advantage.

Pure layer construction totals 3,145.8 seconds (52.43 minutes) on an A40;
held-out evaluation took 485.6 seconds. Setup, research and packaging are excluded.
The original eight-layer checkpoint was reproduced exactly before extension.

The representation combines a Schmidt MPO backbone with sparse residual
coordinate amplitudes. All amplitudes, cores and learned scales count as
stored coefficients. Every support-mask bit counts toward serialized bytes.
This is explicitly a tensor-network/sparsity hybrid; neither conventional
pruning nor quantization is claimed as a new NEC contribution.

| Terminated pod, 8-layer checkpoint | Dense teacher | Hybrid NEC |
| --- | ---: | ---: |
| Validation-slice PPL | 4.08287 | 4.37879 |
| Teacher KL, nats/token | 0 | 0.08902 |
| Next-token accuracy | 66.37% | 65.08% |
| Dense top-1 agreement | 100% | 89.04% |

This record was supplied by the user after terminating the old pod; see
[its explicit provenance](predecessor_8layer_user_report.json). The evaluated
architecture was the complete original model with layers 0-7 replaced and
layers 8-31 dense. It scored 4095 validation tokens, context 2048, stride 512.
It is not a full-model compression or untouched final quality result. The old
hybrid checkpoint files were under `/tmp` and did not survive pod termination.

The new pod reproduces this checkpoint, then continues to 16, 24 and 32
compressed layers with the same representation, budgets, construction data
and solver settings. Optional paired-routing and inverse-SiLU Gate experiments
are disabled. The frozen command is in [compiler_argv.json](compiler_argv.json).
Source and model/calibration provenance are recorded in the run directory.

## Inspect the current run after connecting to the pod

All checkpoints, logs and results are now on the persistent network volume:

```bash
python -m json.tool /workspace/runs/nec-sparse10-coupled-sequential-full32/status.json
```

The status file contains completed layer IDs, the current phase/log, measured
checkpoint metrics and any failure. Evaluation files appear after each stage:

```bash
cat /workspace/runs/nec-sparse10-coupled-sequential-full32/evaluation_stage8_validation_4k.json
cat /workspace/runs/nec-sparse10-coupled-sequential-full32/evaluation_stage32_validation_4k.json
```

At the end, actual complete-model byte and coefficient measurements are in
`export/package_measurement.json`. If the 32-layer development checkpoint
passes PPL 10, the runner evaluates the remaining WikiText-2 test tokens after
offset 32768 as a fresh confirmation range. It also checks the exported
package's bundled decoder in a separate interpreter against checkpoint
injection. `success_demonstrated` requires held-out quality, both measured
compression gates and exported-runtime parity together.

The original dense activation cache can be regenerated from seed 314159,
32768 training tokens and 8192 validation tokens in 2048-token windows with
`scripts/capture_packed_nec_activations.py`. Only this rebuildable cache uses
scratch `/tmp`; model artifacts and results use `/workspace/runs`.

## Lossless serialization option

The exporter also supports `--checkpoint-container gzip`. This changes only
the serialized checkpoint container, preserves the original checkpoint bytes
exactly after decompression, and leaves every logical coefficient unchanged.
The package loader handles decompression itself, so its complete on-disk
directory remains deployable. Every compressed checkpoint records matching
source/round-trip SHA-256 hashes. Complete-package bytes are recounted after
writing the containers, loader, metadata and quality report.

This option is separate from the frozen accuracy experiment and earns no
structural-compression credit. CPU package tests verify exact equality of all
model parameters for plain and gzip packages. A real exported model must also
pass the separate-process bundled-loader parity check before release. The
outer download archive size is not used as the model compression measurement.
