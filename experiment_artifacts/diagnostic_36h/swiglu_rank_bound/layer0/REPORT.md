# Layer-0 SwiGLU rank-floor and Down-crossover diagnostic

Run: 2026-10-17, existing NVIDIA A10, repository checkout `786409e3ead2432ffc13c425fccd57a2ceb3e67a`.

## Result

The shared-neuron rank-426 output ceiling is not by itself enough to explain the failed candidate. On 4,096 held-out token activations, the optimal rank-426 approximation error of the dense MLP output matrix was 0.24280 for the dense trajectory and 0.24142 for the saved-NEC trajectory. This is the Eckart–Young lower bound for that rank-constrained output matrix only, not a bound for arbitrary tensor networks and not evidence that NEC can attain the floor.

Held-out complete MLP relative errors with candidate Gate/Up factors fixed:

| Input trajectory | Saved NEC baseline | Candidate G/U + dense Down oracle | Candidate G/U + saved NEC Down | Candidate G/U + candidate Down |
|---|---:|---:|---:|---:|
| Dense | 0.66228 | 0.48417 | 0.62617 | 0.76765 |
| Compressed | 0.62618 | 0.45238 | 0.59745 | 0.74129 |

For the compressed trajectory, candidate Gate/Up with the saved NEC Down improves error by only 4.6% relative to the saved baseline. The dense-Down oracle gives a 27.8% improvement but is not deployable. The candidate's own Down makes error 18.4% worse than baseline. The candidate's SwiGLU product error is 0.58769; propagated through dense Down it gives 0.45238 error, whereas the candidate Down term has relative norm 0.52101 and positively reinforces the product error (cross-energy fraction +0.15421). This confirms that the measured failure is dominated by the candidate Down representation and its error alignment, not by the product error alone.

At the matched floating-coefficient budget, the candidate stores 9,924,096 coefficients versus 9,943,596 for saved NEC. Its serialized MLP artifact is 19,850,909 bytes versus 30,370,779 bytes for saved NEC. That storage win does not offset the function-error regression.

Rank curve, relative spectral tail error:

| Rank | Dense trajectory | Compressed trajectory |
|---:|---:|---:|
| 64 | 0.43338 | 0.43504 |
| 128 | 0.37561 | 0.37702 |
| 256 | 0.30539 | 0.30575 |
| 322 | 0.27843 | 0.27816 |
| 426 | 0.24280 | 0.24142 |
| 512 | 0.21770 | 0.21549 |
| 768 | 0.15846 | 0.15398 |
| 1,024 | 0.11479 | 0.10877 |
| 1,536 | 0.05752 | 0.05027 |
| 2,048 | 0.02726 | 0.02179 |
| 3,072 | 0.00457 | 0.00321 |

## Reproduction and integrity

The candidate factor tensors from the previous run were not saved, so this run rebuilt the same construction on CUDA from the saved calibration trajectory, using seed 0. Its held-out crossover metrics reproduce the prior experiment to the shown precision. Candidate SHA-256: `f46ce772aed84348275b068ab0876abef5d5f80818fc97435fc6d905bd8e22a3`.

Command, from the remote repository root:

```bash
.venv/bin/python scripts/diagnose_shared_swiglu_rank_bound.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache \
  --output experiment_artifacts/diagnostic_36h/swiglu_rank_bound/layer0/layer0_rank_bound_cuda_run01.json \
  --layer 0 --seq-len 128 --n-heads 32 --n-kv-heads 32 --device cuda --cpu-threads 8
```

The saved checkpoint SHA-256 is `81d44f6b676853cbd218f0c2b22bd0a388eed7195b88153045f5c00ead2de6e9`. Calibration input SHA-256 is `8e9f16b50016b9460607fe116129f62d66702b5484de3fd60b96327c9bd1bc73`; held-out input SHA-256 is `f240af3942d70fc411f0615c65921a8bfecd78c1a70b4256eb790495d076d6a4`. Exact hashes for all seven dense projection files are stored in the JSON.

Runtime was 75.86 seconds. PyTorch peak allocated memory was 4,059,201,536 bytes (incremental peak 2,297,430,016 bytes); `nvidia-smi` observed 5,483 MiB total GPU use while active. No other experiment was running. The A10 was idle at 0 MiB / 0% after completion and was left powered on.

The raw full-precision singular values, rank curve, crossover decomposition, input/weight hashes, candidate coefficient count, serialized bytes, runtime, and memory counters are in `layer0_rank_bound_cuda_run01.json`; terminal output is in `run_cuda01.log`.

## Interpretation / next gate

The 24.1% rank floor is below the 47.0% target corresponding to a 25% relative improvement over the saved NEC compressed-trajectory MLP error. Thus rank 426 is not mathematically ruled out by the sampled output range. But the actual candidate reaches only 74.1%; its own Down factor is substantially worse than the saved NEC Down crossover. A future representation needs to approximate the full nonlinear function with a stronger Down operator under the same complete artifact budget. The dense-Down oracle is diagnostic only. This result does not justify a two-layer compile, full-model run, or capability claim.
