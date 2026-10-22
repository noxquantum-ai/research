# NEC two- and three-layer allocation sweep (A40)

This sweep reassembled the same saved four-layer NEC checkpoint under every
choice of two compressed layers and every choice of three compressed layers.
There was no recompilation or fitting. Layers 4–31 stayed dense.

## Matched short-slice results

All metrics use the same first 4,096 WikiText-2 raw validation tokens: 4,095
non-duplicated causal targets, five windows, context 2,048, stride 512. KL is
`KL(P_dense || P_student)` in nats/token, computed from full-model logits on
the same targets. Softmax and KL accumulation use FP32.

| Compressed layers | Dense layers among 0–3 | PPL | Mean KL | Dense top-1 agreement | Bytes saved |
| --- | --- | ---: | ---: | ---: | ---: |
| none (dense reference) | 0–3 | 4.083 | — | 100.00% | 0 |
| 0, 3 | 1, 2 | **5.079** | **0.244** | 83.10% | 669,619,324 |
| 0, 2 | 1, 3 | 5.908 | 0.392 | 79.07% | 669,708,924 |
| 1, 3 | 0, 2 | 7.016 | 0.562 | 73.63% | 669,399,356 |
| 0, 1 | 2, 3 | 7.497 | 0.615 | 72.84% | 669,681,980 |
| 1, 2 | 0, 3 | 7.543 | 0.630 | 72.94% | 669,488,956 |
| 2, 3 | 0, 1 | 7.805 | 0.662 | 72.92% | 669,426,300 |
| 0, 2, 3 | 1 | 23.093 | 1.730 | 52.06% | 1,004,377,274 |
| 0, 1, 3 | 2 | 25.147 | 1.825 | 52.09% | 1,004,350,330 |
| 0, 1, 2 | 3 | 30.384 | 2.037 | 48.55% | 1,004,439,930 |
| 1, 2, 3 | 0 | 44.867 | 2.409 | 43.39% | 1,004,157,306 |
| 0–3 | none | 233.499 | 4.069 | 28.45% | 1,339,108,280 |

The best short-slice allocation compresses layers **0 and 3** and keeps layers
**1 and 2** dense. This directly shows that retaining the two ends of this
four-layer prefix is not the uniquely best two-layer choice. All six pairs
score below PPL 10 on this short slice; each of the four three-layer
allocations scores above 23, so this evidence is about allocation at a fixed
two-layer compression count.

The result complements the full-validation result for compressing layers 1
and 2: that boundary-protected configuration scored PPL 9.566 and mean KL
0.675 across all 299,963 targets. The short-slice winner (compress layers 0
and 3) still needs full-validation PPL and KL before it can be compared on the
same basis.

## Reproduction details

- Model: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Dataset: WikiText-2 raw validation, revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`; token ID hash
  `be82d08cb9eb67dd4e9fb0c5110091f4af26a9b9f84b136a36dec3c91553945c`.
- GPU: NVIDIA A40; PyTorch `2.11.0+cu128`.
- Source commit: `86b246463b0d64e7230aa150286ae15bdf8b7921`.
- [`report.json`](report.json) records per-variant PPL, KL quantiles, selected
  layers, exact storage estimates, checkpoint hashes, and evaluation protocol.
- Matched full-validation result for the layers-1–2 configuration:
  [`full-validation boundary control`](../nec_prefix_fullval_boundary_a40_20261022/README.md).
