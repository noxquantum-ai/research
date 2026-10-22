# Layer-2 Down-only composition refit (A40)

## Result

Refitting only `down_proj` against the saved NEC Gate/Up product gives a
small held-out improvement, but does not rescue the layer-2 MLP branch. On
the same disjoint 1,024-token dense-context validation activations, relative
MLP-output error falls from **0.84174 to 0.83578**. The training error falls
more, from **0.76407 to 0.71386**. The Gate/Up product error itself remains
**0.80203** on held-out inputs.

| Held-out quantity | Relative error vs. dense MLP output |
|---|---:|
| Dense Down applied to compressed Gate/Up product | 0.78457 |
| Saved NEC Down applied to compressed product | 0.84174 |
| Refit NEC Down applied to compressed product | 0.83578 |
| Saved NEC Down applied to dense product (context control) | 0.79976 |

The fit preserves the High-Fidelity MPO ranks, bridge rank, requested sparse
budget, two sweeps, post-sweep Q4 simulation, prior, and 48 CG iterations.
The original layer-2 Down artifact has 3,280,935 independent numerical
coefficients; the refit has 3,280,907 (28 fewer, due to the realized number
of nonzero sparse values). No compression capacity was added. Refit
compilation took 429.2 s, with 5.49 GiB peak allocated VRAM. It made zero
gradient updates and did not modify any original checkpoint.

Interpretation: the existing Down fit is not the main recoverable bottleneck
in this controlled setting. A budget-matched Down refit learns the training
objective better but transfers only a small fraction of that gain to held-out
examples. More importantly, even the uncompressed dense Down matrix applied
to the compressed Gate/Up product has 78.5% held-out error against the dense
MLP target. This indicates that a large discrepancy is already present in
the Gate/Up product; a single linear Down map has limited ability to undo it.
This does not prove that no Down parameterization at this budget could do
better, nor does this projection-level test establish a language-model PPL
change.

## Objective and protocol

For each saved dense block input `x`, the script reconstructs the dense
post-attention RMSNorm MLP input, then forms

\[
p=\operatorname{SiLU}(Gx)\odot Ux,\qquad
\hat p=\operatorname{SiLU}(\widehat Gx)\odot\widehat Ux,
\]

using the exact dense weights and the saved, artifact-decoded Gate/Up
operators. It fits only a new NEC Down artifact to

\[
\min_{\widehat D\in\mathcal C_{\mathrm{HF}}}
\|\widehat D\hat p_{\mathrm{train}}-Dp_{\mathrm{train}}\|_F^2,
\]

where \(\mathcal C_{\mathrm{HF}}\) is the existing High-Fidelity NEC
Down parameterization and budget. Calibration is the saved 4,096-token
WikiText-2 raw training activation set (32 independent 128-token windows);
validation is the saved, uniformly sampled, disjoint 1,024-token
WikiText-2 raw validation set (8 windows). For both dense teacher and
compressed Gate/Up, `x` is the same dense-context MLP input. This isolates
Gate/Up composition and Down's ability to compensate it; it does not apply
compressed-prefix trajectory correction.

Model: `NousResearch/Llama-2-7b-hf`, revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`. Compiler source commit:
`cd1583e33620fc64c8be5011191f55c54689a99f` (the relevant production source
files are unchanged through the diagnosis commit). A40 report, full settings,
all parent checkpoint and activation hashes, coefficient audit, artifact
hash, and per-split metrics are in `report.json`. The verbose compiler log is
`execution.log`.

The new, standalone Down artifact is preserved on the RunPod network volume
at `/workspace/runs/nec-layer2-down-refit-a40_20261022-v1/layer2_down_refit.pt`;
it is intentionally not committed as a model binary. To reproduce, run
`scripts/refit_layer2_down_comp_context.py` from the activated NEC environment
with a new, empty output directory:

```bash
python -u scripts/refit_layer2_down_comp_context.py \\
  --checkpoint-dir /workspace/runs/nec-streaming-qualification/layer0/reference-post-sweep-q4-48cg \\
  --output /workspace/runs/nec-layer2-down-refit-a40_20261022-reproduction
```

The script refuses to overwrite results and verifies the original checkpoint
hash before and after fitting.

No PPL, MMLU, or full-model evaluation was run in this experiment.
