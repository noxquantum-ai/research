# Four-layer NEC failure diagnosis (A40)

## Finding

The four-layer failure is real and is not explained by a wrong layer index,
checkpoint decode, tokenizer, or custom block-forward implementation. The
strongest localization is the MLP path: separate Gate/Up/Down projection fits
do not preserve their composed SwiGLU function, and their errors compound
when the four compressed blocks are stacked. Dense-trajectory calibration
adds an input-context mismatch, but it is not a complete explanation: layers
2 and 3 already have large MLP-branch errors when evaluated on the same dense
inputs used as the reference.

This is a causal diagnosis of the saved four-layer artifact, not proof that
one specific compiler option alone caused the failure. No compression,
training, or checkpoint was changed in this investigation.

## Controlled PPL ablation

The A40 run applied each intervention to the indicated projection(s) in **all
four compressed layers**, leaving layers 4--31 dense. Scores use the same
first 4,096 tokens of the pinned WikiText-2 raw validation stream, context
2,048, stride 512, 4,095 scored labels, and the same model/tokenizer revision.
This short diagnostic subset is not the complete validation split and should
not be substituted for the full-split result in
`../layer0_3_prefix_stacking_a40_20261022/README.md`.

| Variant | PPL | Mean NLL | Next-token accuracy |
|---|---:|---:|---:|
| Dense | 4.0829 | 1.4068 | 66.37% |
| NEC attention projections only | 4.8556 | 1.5801 | 63.93% |
| NEC MLP projections only | 106.4428 | 4.6676 | 28.62% |
| NEC all seven projections | 233.4995 | 5.4532 | 24.03% |
| Gate only | 10.0542 | 2.3080 | 53.26% |
| Up only | 7.2491 | 1.9809 | 57.39% |
| Down only | 12.2676 | 2.5070 | 51.31% |
| Gate + Up | 69.7933 | 4.2455 | 31.89% |
| Gate + Down | 31.7072 | 3.4565 | 41.17% |
| Up + Down | 46.2822 | 3.8348 | 36.80% |

The factorial result localizes most of the damage to the MLP operators and
their interactions, rather than attention alone. Gate+Up is substantially
worse than either isolated replacement, as expected for errors combined
through a nonlinear product. The single-projection PPLs are not additive
contributions and do not imply that every individual operator is harmless.

## Mechanism supported by source and measurements

For dense inputs `x`, the block compiler forms

\[
g=W_gx,\qquad u=W_ux,\qquad p=\operatorname{SiLU}(g)\odot u,
\qquad y=W_dp.
\]

In the independent construction, it separately fits `gate_proj` and
`up_proj` to their dense linear targets, then forms the actual inference
product from the two compressed outputs. It fits `down_proj` on the dense
product `p`, not that changed product. In code this is the
`H_swiglu_dense` versus `H_swiglu_comp` branch in `nox/block_compiler.py`
(lines 477--478, 549--579). With `\hat g=g+\delta_g`,
`\hat u=u+\delta_u`, and `\hat D=D+\delta_D`, the exact output discrepancy
is

\[
\begin{aligned}
\hat p-p={}&[\operatorname{SiLU}(g+\delta_g)-\operatorname{SiLU}(g)]\odot u
 +\operatorname{SiLU}(g)\odot\delta_u\\
&+[\operatorname{SiLU}(g+\delta_g)-\operatorname{SiLU}(g)]\odot\delta_u,\\
\hat y-y={}&D(\hat p-p)+\delta_Dp+\delta_D(\hat p-p).
\end{aligned}
\]

The product cross-term is absent from three independent linear projection
objectives. On the saved dense-context validation inputs, the measured
relative MLP-branch errors for layers 0--3 were respectively 0.499, 0.030,
0.850, and 0.902. The measured Gate/Up product discrepancy passed through
the dense Down matrix was 0.482, 0.0078, 0.785, and 0.852; the Down operator's
error on the dense product was 0.422, 0.0055, 0.800, and 0.841. Thus a
compressed-prefix mismatch is especially visible at layer 1, while layers 2
and 3 also exhibit severe same-dense-input MLP approximation error.

The compiler's full-block relative error can hide this failure. For layers
2 and 3 on saved validation inputs, same-input full-block error is only
0.0087 and 0.0119, while error in the attention-plus-MLP update itself is
0.825 and 0.886 (cosine 0.566 and 0.464). The full-block comparison includes
the residual stream, which is much larger than the update in these layers;
it is therefore a poor stand-alone proxy for whether the learned sub-block
function was preserved.

The trajectory evidence is also concrete. Normalized MLP-input discrepancy
between dense and four-layer-compressed execution over the saved 128-token
validation windows grows from 0.100 at layer 0 to 0.480, 0.781, and 0.969 at
layers 1--3 (cosines 0.995, 0.880, 0.662, 0.500). The independent run used
`calibration_activation_mode=dense` and `construction_mode=independent`, so
later layers are fitted on clean dense trajectories rather than the inputs
created by the compressed prefix. This is a measured distribution/context
mismatch, but the dense-context errors above show it is not the only failure
mode.

The compressed-prefix trace also contains extreme hidden-state outliers: at
the input to layer 1 in the saved validation-window trace, token-norm p99 is
about 851 and the maximum is about 3,131. Two channels account for about
73.6% and 25.6% of squared activation energy in that trace. This supports a
strongly nonuniform cascade rather than a smooth small perturbation. It does
not establish that those channels alone cause the perplexity failure, and
the report checks finiteness (not clipping) rather than attributing the
outliers to an overflow event.

## Excluded implementation/evaluation explanations

- The pinned Llama-2-7B config and tokenizer were checked; custom dense and
  compressed block-forward implementations matched Hugging Face with global
  relative errors below `5e-4` on saved windows and a 2,048-token sequence.
- All 28 injected projection matrices were exactly equal to their
  corresponding FP16 artifact decodes, with shape and finite-value checks.
- Reconstructing from FP32 factors and then casting to FP16 changed the
  short-subset PPL only from 233.4995 to 233.7518. Factor decoding/storage
  precision therefore does not explain the collapse.
- The independent NLL formula cross-check passed for every PPL intervention.
  The 4,095-token diagnostic subset is intentionally reported separately
  from the full 299,963-token validation evaluation (PPL 294.2195).

## Provenance and limitations

Model: `NousResearch/Llama-2-7b-hf`, revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`. Dataset: `Salesforce/wikitext`,
`wikitext-2-raw-v1`, validation, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. The four checkpoint hashes and
dataset hashes are embedded in the JSON reports. Both A40 analyses loaded
saved artifacts; the factorial run reused and hash-verified its completed
PPL control report, and performed no fitting. The four-layer PPL result
remains catastrophic; MMLU was not run.

The diagnostic scripts and machine-readable outputs are
`scripts/diagnose_prefix_failure.py`,
`scripts/diagnose_prefix_mlp_interactions.py`,
`report.json`, and `../nec_prefix_failure_a40_20261022/report.json`.
The MLP report binds to the current script/helper hashes. The earlier
prefix-trace report records a script hash that differs from the current
working copy, so that exact first diagnostic script revision was not
preserved; treat its measurements as archived evidence rather than claiming
bit-for-bit rerun reproducibility. No full-model PPL or MMLU claim is made.
