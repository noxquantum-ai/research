# Sparse-excitation NEC: staged full-32-layer trial

Target: all 32 transformer layers compressed, held-out PPL <= 10, actual
complete deployable package compression >= 5x, and structural coefficient
reduction >= 2x, simultaneously, with no gradient-based healing. No success
has yet been demonstrated for the complete model.

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
