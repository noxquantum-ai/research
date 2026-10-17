# Function-aware tensor-network experiment report

## Scope and execution status

This report covers the repository audit, exact-symmetry implementation, CPU spectrum/function tests, and the first bounded Kaggle module batch. It does not claim a Llama result until the pinned-model job completes.

The target remains `NousResearch/Llama-2-7b-hf` at revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`; calibration and held-out data remain the pinned raw WikiText-2 revision in the Kaggle driver. The private T4 job is capped at 21,600 seconds, streams one layer at a time, and tests layers 0, 15, and 31. The initial job stopped before model or dataset access because its dependency bootstrap failed in the Kaggle Python 3.13 image. Its JSON and logs are preserved in `experiment_artifacts/function_aware_tn/kaggle_batch_1_dependency_failure/`. The second and final initial submission has been pushed and is queued; no Stage A/B Llama measurements are available yet.

The retry reuses compatible image packages, records their versions, and includes the pip output tail if installation is still required. The local bundle manifest verifies all 38 source/config files. The job is private and uses `NvidiaTeslaT4`. No token value is stored in the source or logs.

## Local implementation and verification

The CPU implementation includes exact RoPE-commuting Q/K gauges, GQA-aware V/O gauges, legal SwiGLU neuron permutation and Up/Down reciprocal scaling, signed residual-coordinate permutation, TT-spectrum accounting with unresolved energy retained, matched-budget MLP and attention module comparisons, and model coefficient accounting.

The function-aware SwiGLU candidate uses activation-weighted Gate/Up factors and a closed-form, ridge-regularized Down readout fit to dense complete-module outputs. It does not optimize Gate/Up jointly, so it is a stagewise module-level candidate rather than a global minimizer. Attention is evaluated by complete attention-output error and the downstream residual-block diagnostic. The MoDeGPT comparison is explicitly a diagonal-proxy-inspired shared-neuron baseline, not a faithful reproduction.

Local verification completed before the Kaggle retry:

- 27 focused CPU tests passed, including exact FP32/FP16 symmetry checks, GQA, full residual blocks, embeddings/head basis folding, unresolved-spectrum accounting, coefficient-budget enforcement, and Kaggle-driver stream helpers.
- Python compilation passed for all new experiment modules and drivers.
- The deterministic synthetic CPU comparison and legacy coefficient audit ran successfully.

## Synthetic CPU comparison

This is a small synthetic SwiGLU (`d_model=32`, `intermediate=48`, seed `20261017`), not Llama evidence. Candidate selection used the internal-selection split; the table reports untouched synthetic held-out MLP relative error.

| Dense coefficient fraction | Ordinary SVD | Activation-weighted SVD | Function-aware Down readout | MoDeGPT-inspired shared-neuron selection |
|---:|---:|---:|---:|---:|
| 75% | 0.7551 | 0.2929 | 0.3048 | **0.2439** |
| 50% | 0.8877 | 0.5256 | 0.5405 | **0.4598** |
| 25% | 0.9881 | 0.8576 | 0.8527 | **0.6929** |

The function-aware factor candidate used 3,440, 2,240, and 1,120 coefficients against exact caps of 3,456, 2,304, and 1,152. The shared-neuron baseline used each full cap. In this one synthetic draw, the function-aware candidate did not beat the strongest matched control at any point. This is a CPU smoke comparison only; it cannot establish behavior on pretrained Llama activations.

## Recalculated historical and reference counts

The pinned dense model has 6,738,415,616 numerical coefficients under the repository’s untied embedding/head layout. Recomputed legacy NEC configuration counts are 939,044,864 (7.176×), 761,323,264 (8.851×), and 611,015,936 (11.028×), respectively. These are configuration-derived counts because ignored historical tensor checkpoints are absent; byte ratios and sparse index storage are reported separately in the failure audit.

The best measured complete-model reference remains the prior grouped SwiGLU model: 6,197,252,096 coefficients, 1.087× compression (8.03% reduction), WikiText-2 PPL 6.7789 vs 4.8597 dense on its 4,096-token/4,095-label protocol. The rank-1866 point stored 5,115,564,032 coefficients (1.317×) but reached PPL 22.4716. These are historical measurements, not results from the new function-aware candidate.

## Stage A and B results

**Pending the second Kaggle run.** The first run accessed neither the pinned checkpoint nor WikiText and therefore provides no spectra, equivalence error, or module-output measurements. The first-run preflight failure is an environment/install issue, not evidence for or against the research hypothesis.

The retry writes exact gauge errors and empirical FP16 tolerances; direct and four-site spectra for the seven projection matrices; original and two alternative NEC site orders; coefficient counts at 1%, 5%, 10%, and 20% residual-energy targets; matched ordinary-SVD coefficients; and 4K/16K fitting, internal-selection, held-out, residual-block, cosine, and norm metrics at 75/50/25% module budgets. A Stage A pass requires a coherent legal gauge group or a coherent site-order result to save at least 20% coefficients at the 10% residual-energy target in at least two representative layers. A Stage B pass requires at least 15% lower median held-out module-output error than the strongest matched structural baseline across all three layers, with no layer more than 10% worse.

The 65,536-token calibration point, prefix tests, and full-model Stage C are not part of this first successful diagnostic run. Prefix tests should only be prepared if a candidate passes the module gate. The full-model benchmark has not been run for this new method.

## Final decision

**1. What was wrong with the old NEC representation?**

Pending no new Llama batch is needed to answer the historical part: the old coefficient ratios were far less dramatic than the byte labels, the fixed tensorization/ranks did not preserve individual module functions well enough, calibration-fit DMRG overfit, and residual corrections/bridges consumed significant coefficient budget. Sequential functional error compounded sharply in 1–4-layer prefixes.

**2. Did we discover a better function-preserving representation?**

The mathematical candidates are implemented and pass the local synthetic equivalence suite; pinned-model real-activation equivalence, randomized Schmidt spectra, and coefficient counts remain pending Kaggle batch 2.

**3. Did function-aware compression beat ordinary structural baselines?**

No pretrained-model conclusion yet. On the single synthetic CPU draw, the function-aware factor candidate lost to the matched MoDeGPT-inspired shared-neuron baseline at all three budgets.

**4. What is the highest full-model parameter reduction actually demonstrated?**

For the new method, none. The highest full-model coefficient reduction in the repository with a corresponding full-model WikiText-2 measurement is the rank-1866 grouped SwiGLU point at 1.317× coefficient compression (24.1% reduction), with PPL 22.4716. The quality-first grouped reference is 1.087× at PPL 6.7789. Legacy NEC configs imply larger coefficient ratios (up to 11.028×), but there is no corresponding audited full-model coefficient artifact and full-model PPL result for those configs in the retained evidence. Neither measured grouped point meets the requested quality/compression target.

**5. Is there a credible path toward 50–70% whole-model parameter reduction without healing?**

**NOT YET ESTABLISHED.** Existing high-fidelity evidence is only 8.03% whole-model reduction, and the synthetic function-aware candidate lost to a simple matched structural control. Stage A/B on the pinned model are still required before making a stronger negative or positive claim.
