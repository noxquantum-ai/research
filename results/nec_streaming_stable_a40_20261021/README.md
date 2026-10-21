# Stabilized streaming NEC qualification (A40, 2026-10-21)

This directory records a real-model, layer-0 `down_proj` comparison between
the full-matrix NEC compiler and FP64 streaming sufficient statistics.  It is
a projection-level numerical qualification, not a language-model benchmark.
Streaming is **not** qualified for production by this result.

## Controlled configuration

- Model: `NousResearch/Llama-2-7b-hf`
- Revision: `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
- Device: NVIDIA A40, 47,699,722,240 bytes VRAM
- Projection: decoder layer 0 `down_proj`, shape `4096 x 11008`
- Calibration/validation: 4,096/1,024 real WikiText-2 tokens, uniform disjoint
  128-token windows, seeds 42/43
- High-Fidelity ranks: `[1,64,128,64,1]`; bridge rank 128; sparse budget
  524,288; two DMRG sweeps; Q4 simulation
- Numerical control: Q4 after both sweeps and at most 48 CG iterations
- Compiler source for the matched runs: `51f738f4234e6888570e66199b5887bc88b7632a`
- Bridge diagnostic source: `7b4afe23d66df328c32bfc759262c0421d995912`

Both compilers used identical token IDs and hidden states.  All 4,096 expected
calibration modes remained positive in the FP64 Gram factorization.

## Matched projection result

| Metric | Full matrix | Streaming FP64 |
|---|---:|---:|
| Core runtime | 442.85 s | 468.20 s |
| Peak PyTorch allocation | 4.40 GiB | 10.10 GiB |
| Calibration relative error | 0.337040 | 0.337012 |
| Held-out relative error | 0.422366 | 0.422235 |
| Largest CG normal-equation residual | 7.17e-6 | 7.59e-6 |

Scalar errors agree, but the operators do not:

| Direct comparison | Result | Gate |
|---|---:|---:|
| Reconstructed-weight relative difference | 0.294785 | <=0.01 |
| Calibration functional disagreement | 0.062775 | <=0.01 |
| Held-out functional disagreement | 0.068492 | <=0.01 |

The formal comparison therefore reports `fp64_passes: false`.  Logged wrapper
cost for the reference, streaming run, and comparison was about $0.162 at
$0.59/GPU-hour.  The later bridge-only diagnostics were not timed by that
wrapper and should not be treated as provider billing records.

## First-divergence diagnosis

The real-activation bridge diagnostic directly compared SVD of `X` with FP64
eigendecomposition of `XX^T` before TT-SVD or DMRG:

- top-256 singular-value relative difference: `1.751e-4`;
- largest top-256 principal angle: `2.827` degrees;
- relative gap at the rank-256 cutoff: `0.002039`;
- unquantized bridge held-out disagreement: `0.0003485`;
- separately Q4-quantized bridge held-out disagreement: `0.0247977`.

Thus the Gram bridge is close before quantization, but Q4 of its two factors
amplifies the small basis change by roughly two orders of magnitude.  Three
gauge controls did not improve this:

| Bridge factor gauge | Held-out Q4 disagreement |
|---|---:|
| Existing balanced columns | 0.024798 |
| Balanced SVD | 0.041249 |
| Fixed square random anchor (condition ~456) | 0.219238 |
| Pivoted operator anchor (same pivots; condition ~58) | 0.063942 |

The alternatives are negative diagnostic results and were not integrated into
the NEC compiler.  Post-sweep Q4 and better-converged CG also failed to make
the full compilers equivalent.  The production runner remains unchanged.

Separately, the full-matrix reference itself has `0.422366` held-out relative
output error on this projection.  Fixing streaming equivalence would not by
itself establish model fidelity.

## Reproduction

The matched runs were executed with:

```bash
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-stable-reference
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-stable-fp64
bash runpod_execution/nec_high_fidelity/qualify_streaming.sh down-stable-compare
```

The bridge-only diagnostic is:

```bash
python benchmarks/diagnose_real_bridge.py \
  --output bridge-diagnostic-final.json \
  --reference-output /workspace/runs/nec-streaming-qualification/down_projection_stable/reference-full-matrix \
  --device cuda:0
```

JSON reports and complete wrapper/heartbeat logs are committed here.  The
large `.pt` artifacts and stage matrices remain on the persistent RunPod
volume under `/workspace/runs/nec-streaming-qualification/down_projection_stable`.
The transferred report archive had SHA-256
`4af84cb05b31708fd6b1fac0c9fed4742395088d146e0538a3b87f1794830df5`.
