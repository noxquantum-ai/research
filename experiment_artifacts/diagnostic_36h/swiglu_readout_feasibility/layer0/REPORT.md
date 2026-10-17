# Final shared-neuron SwiGLU readout feasibility test

**Decision: NO-GO.** Neither authorized rank met all advancement gates. The shared-neuron architecture is unsuccessful at this milestone; no two-layer PPL evaluation is justified.

## Exact held-out results

All MLP errors below use the saved NEC compressed attention trajectory and the dense MLP function on that same input. Full-block and active-transform errors compare against the dense layer-0 path.

| Configuration | MLP error | Active-transform error | Full-block error | MLP ≤0.470 | Block ≤0.273 | Active no regression | Budget |
|---|---:|---:|---:|:---:|:---:|:---:|:---:|
| Saved NEC | 0.626175 | 0.430836 | 0.363740 | — | — | baseline | — |
| Rank 426 held-out oracle | 0.393936 | 0.316519 | 0.267226 | yes | diagnostic only | diagnostic only | diagnostic only |
| Rank 426 calibration-only | 0.477592 | 0.357949 | 0.302205 | **no** | **no** | yes | yes |
| Rank 651 held-out oracle | 0.348466 | 0.298555 | 0.252060 | yes | diagnostic only | diagnostic only | diagnostic only |
| Rank 651 calibration-only | 0.458930 | 0.348740 | 0.294430 | yes | **no** | yes | yes |

The oracle fits use held-out targets and are unattainable diagnostics, never deployable artifacts. They prove that the fixed rank-426 and rank-651 feature matrices can support the requested MLP error with perfect hindsight. The calibration-only construction does not generalize well enough to meet all gates. Rank 651 passes the MLP threshold by 0.01107 but misses the full-block threshold by 0.02143. Per the predeclared rule, this is a NO-GO; no corrective sidecar or further shared-neuron workaround will be attempted.

Both legitimate constructions selected ridge ratio 0.01 on a disjoint internal calibration split made from whole 128-token sequences: 3,072 fitting tokens and 1,024 selection tokens. The held-out 4,096 tokens were not used for ridge selection or artifact construction. The final Down factors were refitted once on all calibration tokens, stored in FP16, frozen, and then evaluated. FP16 storage changed rank-651 MLP error from 0.45893013 to 0.45892963, so readout quantization was not the limiting factor.

## Budget accounting

| Artifact | Floating coefficients | Payload bytes | Actual serialized bytes | Declared comparison |
|---|---:|---:|---:|---|
| Saved NEC MLP | 9,943,596 | — | 30,370,779 | reference |
| Rank 426 shared TTN | 9,924,096 | 19,848,192 | 19,851,421 | coefficient-matched; passes |
| Rank 651 shared TTN | 15,165,696 | 30,331,392 | 30,334,621 | byte-matched; passes by 36,158 bytes |

Rank 651 uses more floating coefficients than saved NEC because it replaces NEC's sparse-index and schema overhead with contiguous FP16 factors. It remains below the complete saved MLP artifact byte budget, as required.

## Interpretation

The feature-specific oracles close the gap from saved NEC's 0.6262 MLP error to 0.3939 (rank 426) and 0.3485 (rank 651). Therefore the shared nonlinear features are not intrinsically incapable on the observed held-out set. The failure is generalization of the calibration-only linear readout: internal calibration selection errors were 0.4871 and 0.4711, and held-out errors were 0.4776 and 0.4589. The rank-651 MLP improvement is real, but upstream attention/residual error leaves the block at 0.2944 rather than the required 0.273.

This result falsifies the shared-neuron method under the declared advancement rule, not all tensor-network compression. The research milestone should now change from near-5× structural compression to a quality-first structurally compressed Llama baseline at a more conservative reduction rate, followed by a measured tensor-network improvement over that baseline.

## Reproducibility

Run date: 2026-10-17. Device: NVIDIA A10. Runtime: 2.6802 seconds. Peak PyTorch CUDA allocation: 6,285,204,480 bytes. The GPU was idle at 0 MiB / 0% after the process exited and its power state was not changed.

```bash
timeout 660 .venv/bin/python scripts/diagnose_swiglu_readout_feasibility.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache \
  --output experiment_artifacts/diagnostic_36h/swiglu_readout_feasibility/layer0/layer0_readout_feasibility.json \
  --layer 0 --device cuda --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --max-runtime-seconds 600
```

The raw JSON contains all input/weight hashes, the complete ridge grids, oracle and legitimate measurements, budgets, runtimes, and pass/fail flags. JSON SHA-256: `aa35982ce9daeda65d9941acee487358f8a5102c1b310f2ae62b4587b7501286`. Run-log SHA-256: `36d8908f4cc402ed4b866c4a6e51147561edfa8850311b08cc16fc4ac17e05bb`.

Results and every required layer-0 dense projection/norm cache tensor are backed up outside the GPU instance under `/home/fluxx/Workspace/research_backups/nox_swiglu_readout_feasibility_20261017/`; all nine remote/local cache hashes matched. The saved checkpoint and calibration/held-out input tensors were already backed up under `/home/fluxx/Workspace/research_backups/nox_operator_schmidt_layer0_20261017/layer0_initial/`.
