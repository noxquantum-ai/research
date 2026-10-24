# NEC accuracy recovery at fixed or better compression

The original published NEC hybrid remains the protected baseline. Work uses the
full Llama-2-7B architecture and NEC for all seven projections in all 32 layers,
with no gradient-based fine-tuning or healing. Remaining capability benchmarks
were stopped at the user's request and are not being resumed.

## Acceptance policy

| Metric | Current measured baseline | New requirement |
| --- | ---: | ---: |
| Complete published serialized bytes | 2,490,273,288 | At most this many actual bytes |
| Actual whole-model storage compression | 5.412758341× | At least this ratio |
| Total logical coefficients, including scales | 3,323,032,892 | At most this many coefficients |
| Structural reduction | 2.027791443× | At least this ratio |
| Dense transformer layers retained | 0 | 0 |
| Held-out WikiText-2 PPL | NEC 8.348960; dense 4.867285 | Aim ≤ dense; fallback ≤ dense + 0.5 |
| Next-token accuracy | NEC 53.9990%; dense 62.8199% | Within 2 percentage points of matched dense |
| Dense top-1 agreement | 71.0259% | At least 98% |
| Teacher KL | 0.543950 nats/token | Must decrease |

The 0.5 PPL allowance is an absolute difference. The token accuracy tolerance is
interpreted as two percentage points. Teacher agreement is measured against the
dense model's 100% self-agreement. No claim is made that these targets have been
achieved or are guaranteed to be achievable at this budget.

Use matched input IDs, context 2,048, stride 512 and dataset provenance. Prior
confirmation used the 308,700 scored WikiText-2 test tokens after offset 32,768.
That range is now a known regression set, not a fresh untouched test. Candidate
selection uses training construction/guard windows; validation is reported
separately, and full regression evaluation is reserved for promising candidates.
A further untouched corpus is needed for a new generalization claim.

Passing the token metrics does not establish that BoolQ or HellaSwag remain
unchanged; those claims require subsequent capability evaluation. Existing full
scores remain recorded: BoolQ NEC 48.50% vs matched dense 79.36%; NEC HellaSwag
50.39% length-normalized accuracy. No candidate replaces the published baseline
on the strength of local reconstruction errors alone.

## First construction experiment

The existing hybrid retains its Schmidt/MPO backbone and the exact same sparse
amplitude and scale counts. Compare:

1. Channel-relative normal-equation reconstruction on fixed support, with
   analytically fitted scalar scales.
2. An eight-level nonzero amplitude grid, exploiting all three-bit codes for
   selected nonzero residual amplitudes without adding scale coefficients.
3. Covariance-aware support and sequential error feedback for both pruning and
   rounding, in a temporary unit-RMS input gauge. No covariance or inverse matrix
   is retained for inference.

Fit on 24,576 training tokens, select on a disjoint 8,192-token training guard,
and report separate 8,192-token validation errors. The initial layer-0 probe
holds the other operators and all later layers fixed. Candidate tensor payload
and logical coefficient counts must match exactly. Complete serialized gzip
bytes must also be measured before promotion: equal tensor bytes do not imply
equal lossless-container bytes.

## Research grounding

[SparseGPT (ICML 2023)](https://proceedings.mlr.press/v202/frantar23a.html)
provides analytical second-order sparse reconstruction and error feedback.
[SLiM (ICML 2025)](https://proceedings.mlr.press/v267/mozaffari25a.html)
combines one-shot sparsity, quantization and analytical low-rank compensation.
[EoRA](https://arxiv.org/abs/2410.21271) emphasizes activation-eigenspace residual
approximation without gradient training.
[QuIP# (ICML 2024)](https://proceedings.mlr.press/v235/tseng24a.html) motivates
attention to covariance conditioning and low-bit rounding error. Its full method
also includes fine-tuning; that component is excluded here.

These ideas are established prior art. Applying them to NEC's existing MPO plus
coordinate-excitation representation is an experimental design, not a claim of
novelty or a substitution of conventional quantization for NEC.

## Measured initial controls

Layer-0 constructor selection used training fit/guard tokens only, then reported
separate validation activations. Exact tensor payloads and coefficient counts
were unchanged:

| Projection | Baseline validation relative MSE | Selected constructor | New validation relative MSE |
| --- | ---: | --- | ---: |
| Query | 0.00554723 | Covariance feedback, nonzero grid | 0.000389999 |
| Down | 0.0641991 | Relative-anchored PCG, fitted scales | 0.0616066 |

See [local raw probe](layer0_constructor_probe.json). These are local operator
errors, not full-model quality measurements.

A full 32-layer Q/K-only reconstruction, keeping every other NEC operator fixed,
failed the no-regression policy on 8,191 matched validation tokens:

| Metric | Current NEC baseline | Q/K control |
| --- | ---: | ---: |
| PPL | 6.697377 | 6.671261 |
| Teacher KL | 0.535163 | 0.529597 |
| Next-token accuracy | 57.3678% | 57.1847% |
| Dense top-1 agreement | 72.0181% | 71.9937% |

The control is **not promoted**. Better independent reconstruction did not
preserve all model-level metrics. See [baseline](baseline_validation_8k.json),
[control](qk_control_validation_8k.json) and [acceptance record](qk_control_status.json).

The next trial reconstructs all seven operators sequentially against the actual
mixed-prefix activations and clean teacher targets. It reuses existing MPO
backbones and all branch shapes, selects excitation construction on training
guards and applies the existing coupled attention/MLP normal-equation solves.
The complete model is measured after replacing eight candidate layers and again
after all 32; other layers in the stage-eight assembly use the previous NEC
checkpoints. Full test regression, complete-package measurement and bundled
runtime parity are conditional on validation improvement. None of these local
controls establishes the final dense-level quality target.

Persistent status:
`/workspace/runs/nec-accuracy-recovery-20261024/coupled_status.json`.
Runner: [run_nec_accuracy_reconstruction.py](../../scripts/run_nec_accuracy_reconstruction.py).

An optional `--joint-mlp-output-metric channel_relative` objective is prepared
for a separate comparison. It derives inverse channel-RMS weights from training
fit targets only, and applies the same weights to each linear map and its
adjoint. This tests whether output-energy dominance is suppressing useful
smaller channels. The current sequential trial uses the original `energy`
objective. Metric weights are construction statistics, not stored parameters.

A lossless scalar bitplane packing prototype was also rejected: on the first
reconstructed layer its gzip size increased from 52,427,541 to 52,571,543 bytes.
Integer levels round-tripped exactly, but it did not satisfy the storage
objective. The prototype codec was not retained. Its [measurement record](bitplane_layer0_rejected.json)
is archived to avoid repeating the unsuccessful direction.
