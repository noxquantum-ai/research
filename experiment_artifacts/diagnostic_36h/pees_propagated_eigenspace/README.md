# PEES: propagated eigenspace error shield

## Hypothesis

The MPO+sparse+bridge artifacts capture the bulk projection, while the most
damaging functional defects are concentrated in a smaller activation subspace.
PEES fits an additive low-rank sidecar to

\[
R = W X_{\mathrm{dense}} - \widehat W X_{\mathrm{compressed}},
\]

using only training calibration activations. For compressed context
\(X=USV^T\), it computes the ridge-whitened target

\[
M = R V S (S^2 + \lambda I)^{-1/2},
\]

truncates \(M\) to rank \(r\), and maps those factors back through
\((S^2+\lambda I)^{-1/2}U^T\). Thus the target includes both local operator
error and inherited trajectory error, and the rank is spent according to
compressed activation energy. The implementation uses the token Gram matrix,
so the 11,008-wide Down context does not require a feature-width covariance.

PEES is applied in sequence to V, O, Gate, and Down. After each sidecar is
added to the existing bridge factors, compressed contexts are recomputed before
fitting the next projection. The sidecars are constructed before checkpoint
serialization; there are no gradient updates or post-compression checkpoint
edits. The serialized checkpoint is capped at 1.25 times the matched NEC
artifact, with rank reduced automatically to stay under the measured byte cap.

The underlying ingredients relate to propagated-error compensation (QEP) and
activation-eigenspace low-rank compensation (EoRA); this experiment does not
claim those ingredients as new. The NEC-specific question is whether sequential
within-block defect fitting can use a small sidecar under a fixed byte ceiling
to improve held-out composed behavior. [QEP](https://arxiv.org/abs/2504.09629),
[EoRA](https://arxiv.org/abs/2410.21271), and the perturbation-robustness
motivation in [QDrop](https://arxiv.org/abs/2203.05740) are relevant prior work.

## Matched control

Baseline artifact directory on the remote GPU host:

`/home/shadeform/diag36h-research/checkpoints/diagnostic_36h/high_fidelity_no_healing_compressed_prefix_2layer`

It uses `NousResearch/Llama-2-7b-hf` revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`, WikiText-2 raw revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, 1,024 uniform train tokens,
1,024 distinct uniform validation tokens, sequence length 128, high-fidelity
ranks `[1,64,128,64,1]`, bridge rank 128, 524,288 sparse outliers, and no
healing.

The saved raw WikiText test result is PPL 9.44237, mean NLL 2.245207,
next-token accuracy 0.528853, and dense top-1 agreement 0.722572 over 9,999
scored labels. Each NEC layer artifact is about 69.8–70.0 MB.

## Candidate command

```bash
python -u scripts/compile_high_fidelity.py \
  --output checkpoints/diagnostic_36h/pees_prop_eigenspace_2layer_2609812 \
  --start-layer 0 --end-layer 1 \
  --n-calib 1024 --n-validation 1024 --seq-len 128 \
  --calibration-sampling uniform --validation-sampling uniform \
  --calibration-activation-mode compressed_prefix \
  --construction-mode independent \
  --pees-rank 192 --pees-tokens 4096 --pees-ridge-ratio 0.01 \
  --max-checkpoint-multiplier 1.25
```

The run was executed on the authorized remote A10 from source commit
`2609812`. The per-layer checkpoints remain on that host. The compile log,
evaluation log, manifest, and evaluation JSON are copied into this local
artifact directory.

## Results

The compile completed in 1,539.34 seconds for layers 0 and 1. It used no
post-compression updates. The matched 10,000-input-token, 9,999-label raw
WikiText evaluation used context length 2,048 and stride 512.

| Model | PPL | Mean NLL | Next-token accuracy | Dense top-1 agreement | Two-layer artifact bytes |
|---|---:|---:|---:|---:|---:|
| Dense reference | 5.2922 | 1.666234 | 61.0461% | — | — |
| Existing compressed-prefix control | 9.4424 | 2.245207 | 52.8853% | 72.2572% | 139,846,298 |
| PEES, 1.25× per-layer byte cap | 9.3072 | 2.230791 | 53.1953% | 72.8073% | 174,054,490 |

PEES improved PPL by 0.1351 and next-token accuracy by 0.310 percentage
points over the matched compressed-prefix control. Next-token accuracy was
87.15% of dense, up from 86.63%; the result remains below the 90% capability
retention objective. Dense top-1 agreement rose by 0.55 percentage points.
This is a small recovery, not a pass or a basis for extending PEES to the
four-layer or full-model run.

The candidate added 34,208,192 bytes (+24.46%) to the two-layer artifact.
Per-layer baseline/candidate serialized sizes were 69,810,336/86,865,901 bytes
and 70,039,648/87,188,589 bytes. The resulting 4.65× figure compares only
the seven projection weights in these two layers with their compressed layer
artifacts; it is not a full-model compression ratio.

The held-out block diagnostics support inherited error propagation: layer 1
had 0.0882 relative error when evaluated on its own dense input, versus 0.2363
on the compressed-prefix cascade. Layer 0 held-out block error was 0.2562
after PEES (the prior matched report was approximately 0.2748). Calibration
projection errors improved, for example Gate 0.2991→0.1953 and Down
0.3342→0.2208, but these changes did not transfer to a meaningful composed
quality recovery. The sampled-context row-space energy metric is not a
held-out representability result; the rank-limited defect residuals remained
0.55–0.71 across the selected projections.

Conclusion: the current PEES allocation spends nearly the full authorized
25% sidecar overhead yet recovers only a small part of the lost language-model
quality. Do not scale it as configured. The A10 is idle after this evaluation;
no four-layer or full-model run was started.

### Reproducibility files

- `compile.log` — complete per-layer metrics and PEES factor records.
- `evaluation.log` and `evaluation_wikitext.json` — matched dense/compressed
  evaluation and dataset fingerprint.
- `model_manifest.json` — pinned model revision, data provenance, source
  fingerprint, settings, checkpoint sizes, and compilation times.
- Remote checkpoint directory:
  `/home/shadeform/nec-crosslayer-research/checkpoints/diagnostic_36h/pees_prop_eigenspace_2layer_2609812`
