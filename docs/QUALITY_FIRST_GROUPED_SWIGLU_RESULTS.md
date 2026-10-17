# Quality-first grouped SwiGLU experiment results

Status: first quality-first full-model result recorded. The full-32 grouped SwiGLU candidate at approximately 87.5% of dense MLP coefficients scored WikiText-2 PPL 6.7789. The user has confirmed this is acceptable for the current iteration; no additional GPU run is underway.

## Evaluation setup

- Dense reference: `NousResearch/Llama-2-7b-hf`, revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Dataset: WikiText-2 raw test split, `Salesforce/wikitext` revision `b08601e04326c79dfdd32d625aee71d232d685c3`, source SHA-256 `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
- Protocol: 4,096 maximum evaluation tokens, sequence length 2,048, stride 512; 4,095 next-token labels were scored.
- Attention, embeddings, normalization, and output head remain dense. The grouped SwiGLU artifacts replace only MLPs. No gradients, healing, dense residual, or post-compression optimization are used.
- All PPL values below use the same dense model, tokenizer revision, test text, and evaluation protocol.

## Results at 62.5% of dense MLP coefficients

The grouped tensor-network keeps the original SwiGLU neuron coordinates and factorizes Gate, Up, and Down. With one group and rank 1,866, each MLP artifact stores 84,552,192 floating coefficients versus 135,266,304 dense coefficients. The full set has 5,411,484,712 serialized bytes, including checkpoint metadata, versus 8,657,043,456 bytes for FP16 dense MLP matrices.

| Compressed MLP layers | WikiText-2 PPL | Next-token accuracy | Candidate artifact bytes |
| --- | ---: | ---: | ---: |
| Dense baseline | 4.8597 | 0.62808 | 0 |
| 0 only | 4.8705 | 0.62833 | 169,108,889 |
| 0–1 | 4.8993 | 0.62882 | 338,217,778 |
| 0–3 | 5.1800 | 0.62271 | 676,435,556 |
| 0–7 | 6.1821 | 0.59219 | 1,352,871,112 |
| 0–11 | 7.6452 | 0.55092 | 2,029,306,692 |
| 0–15 | 10.8513 | 0.50183 | 2,705,742,296 |
| 0–19 | 15.7397 | 0.45397 | 3,382,177,900 |
| 0–23 | 19.7954 | 0.42589 | 4,058,613,504 |
| 0–27 | 21.4583 | 0.41245 | 4,735,049,108 |
| 0–29; layers 30–31 dense | 21.8574 | 0.41001 | 5,073,266,910 |
| 0–31 | 22.4716 | 0.40757 | 5,411,484,712 |

The 0–1 and 0–3 settings are practically useful local controls under the user's stated PPL 5–6 tolerance, but they compress only a small portion of the model. The uniform 32-layer setting preserves structural parameter reduction but is not usable at this coefficient budget. Keeping the final two MLPs dense barely changes the collapse, so it is not only a last-layer problem. The prefix curve worsens steadily as approximation error accumulates.

The 32-layer 62.5% compilation took 572.85 seconds and peaked at 10,976,272,896 allocated CUDA bytes. Its full 4,096-token PPL run took 18.97 seconds and peaked at 16,437,170,176 allocated CUDA bytes.

## 87.5% full-model result

The full 32-layer A10 run used the same native-neuron grouped two-site tensor-network construction and sequential calibration path, with one group and rank 2,612 per Gate/Up/Down operator. The artifacts contain 118,354,944 floating coefficients per MLP (3,787,358,208 across 32 layers), approximately 87.5% of the dense MLP coefficient count. There are no gradients, dense residuals, or post-compression optimization. Attention, embeddings, normalization, and output head remain dense.

| Model | WikiText-2 raw-test PPL | Next-token accuracy | Mean NLL |
| --- | ---: | ---: | ---: |
| Dense Llama-2-7B | 4.8597298 | 0.6280830 | 1.5809828 |
| 32 grouped-TN MLPs | 6.7788787 | 0.5621490 | 1.9138117 |

The compressed model retains 89.50% of dense next-token accuracy on this evaluation (accuracy ratio, not a downstream benchmark suite). The user considers PPL 6.779 acceptable for this iteration. This is a meaningful quality-first result, though it remains above the initially stated 5–6 PPL band and does not establish 90% retention across downstream benchmarks.

The 32 serialized MLP artifacts total **7,574,860,840 bytes** (about 7.06 GiB), versus 8,657,043,456 bytes for dense FP16 MLP matrices, a 12.50% MLP storage reduction. This excludes the unchanged dense model components, so it is not a 12.5% whole-model reduction. Each artifact is 236,714,393–236,714,405 bytes. The completed compile took **586.18 s** and peaked at **11,090,858,496 CUDA-allocated bytes**; evaluation took **22.08 s** and peaked at **16,437,170,176 CUDA-allocated bytes**.

### Parameter and storage reduction

For Llama-2-7B dimensions (32 layers, hidden size 4,096, intermediate size 11,008, vocabulary 32,000), the three dense MLP matrices contain `32 × 3 × 4,096 × 11,008 = 4,328,521,728` scalar parameters. The grouped-TN factors contain `32 × 118,354,944 = 3,787,358,208` coefficients, saving **541,163,520 parameters**, or **12.50% of MLP parameters** (about 1.14× compression of the MLP portion). The unchanged attention, embeddings, norms, and output head account for another 2,409,893,888 parameters. Thus the analytic total falls from 6,738,415,616 to 6,197,252,096 parameters: about **8.03% fewer parameters overall**, or only about **1.09× whole-model parameter compression**. These counts describe stored factor coefficients, not runtime speed or memory.

At FP16 for unchanged components, substituting the measured MLP artifact bytes gives an estimated whole-model serialization reduction of about 8.03% (approximately 13.48 GB to 12.39 GB decimal). This is a projection, not a measured unified checkpoint: the unchanged components were not repacked into one artifact, and serialization metadata/container overhead may differ.

The matched benchmark used Llama-2-7B-HF revision `8efe6c9b93655b934e27bd9981e3ec13e55aee9d` and the `Salesforce/wikitext` raw WikiText-2 test split at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. It scored 4,095 next-token labels from a 4,096-token cap, with sequence length 2,048 and stride 512. The dataset source SHA-256 is `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.

Reproducibility artifacts: [manifest](../experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0875/manifest.json), [evaluation JSON](../experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0875/ppl4096/evaluation.json), and [evaluation log](../experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0875/ppl4096/run.log). The manifest records compilation inputs, layerwise records, runtime, memory, and total bytes. The 32 large `.pt` weight artifacts remain on the A10; they are not committed to Git.

**Closeout:** the user accepts PPL 6.7789 for this iteration. The result and all compact reproducibility outputs are committed on `main`; the A10 was verified idle after the run. No further GPU experiment is planned as part of this session.

## Reproduction

From the repository root on the A10, with the existing `.venv` and downloaded model cache:

```bash
.venv/bin/python scripts/compile_native_swiglu_prefix.py \
  --initial-input-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --output-dir experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0625 \
  --layers 32 --representation grouped_tn --groups 1 --retain-fraction 0.625

.venv/bin/python benchmarks/benchmark_native_swiglu_perplexity.py \
  --artifact experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0625/native_mlp_layer_{0..31}.pt \
  --output experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0625/ppl4096/evaluation.json \
  --max-eval-tokens 4096 --eval-dense

# Completed quality-first setting: same protocol, retain fraction 0.875
# (rank 2,612), using the matching full32_groups1_0875 output directory.

.venv/bin/python benchmarks/benchmark_grouped_swiglu_prefix_sweep.py \
  --artifact-dir experiment_artifacts/quality_first/grouped_swiglu_tn/full32_groups1_0625 \
  --prefix-lengths 4 8 12 16 20 24 28 30 32 \
  --output experiment_artifacts/quality_first/grouped_swiglu_tn/prefix_sweep/ppl4096.json \
  --max-eval-tokens 4096
```

Raw outputs and layer-wise measurements are under `experiment_artifacts/quality_first/grouped_swiglu_tn/`. The 62.5% large artifacts are backed up locally outside the repository, split across the workspace filesystem and ChromeOS Downloads because the workspace filesystem alone has insufficient free space. The 87.5% full-32 artifacts remain on the A10; the small manifest/evaluation JSON/log are stored in this repository.
