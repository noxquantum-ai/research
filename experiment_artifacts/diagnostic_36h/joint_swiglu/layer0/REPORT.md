# Layer-0 joint SwiGLU diagnostic

**Result: the shared-neuron TTN fails the predeclared advancement gate.** At a
matched MLP coefficient budget, its held-out MLP function error is 0.7413,
versus 0.6262 for saved NEC on the compressed transformer trajectory (18.38%
worse). Its active-transform error rises from 0.4308 to 0.4900 and full-block
error from 0.3637 to 0.4137. It does use 34.64% fewer serialized MLP bytes.
Because both quality checks regress, storage savings do not qualify it for a
prefix or larger run.

## Reproducibility and scope

- Model: `NousResearch/Llama-2-7b-hf`, revision
  `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`.
- Data: WikiText-2 raw train and validation splits at dataset revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`; 4,096 calibration tokens
  (seed 42) and 4,096 disjoint held-out validation tokens (seed 43), each
  represented as 32 windows of 128 tokens. The JSON records token-ID and
  activation hashes and the exact sample offsets.
- Checkpoint: existing high-fidelity layer-0 NEC artifact; SHA-256
  `81d44f6b676853cbd218f0c2b22bd0a388eed7195b88153045f5c00ead2de6e9`.
- Hardware/runtime: NVIDIA A10, PyTorch 2.7.0, CUDA available. GPU was idle
  before the run with 22,599 MiB free. No competing process was observed.
- The remote clone was on `main` at commit
  `786409e3ead2432ffc13c425fccd57a2ceb3e67a`; only the diagnostic source
  files were transferred there. Their SHA-256 hashes were
  `bfbb0fdefbb9a0305e695f59d9f458561acab34ae4c703d42772996da6e8f8af`
  (`nox/joint_swiglu.py`) and
  `667cb4a9c5377cdb977f837ebb9dd17422ca86c3a761bd853d3199c74386a965`
  (`scripts/diagnose_swiglu_joint.py`). Existing remote edits and artifacts
  were preserved.
- The diagnostic reads saved NEC and dense weights, evaluates all eight
  Gate/Up/Down choices on both dense and compressed transformer trajectories,
  then builds one factorized candidate from calibration activations. No
  checkpoint was compiled or written; no gradients, healing, PPL, MMLU,
  prefix, or full-model evaluation ran.

Exact remote command:

```bash
.venv/bin/python -u scripts/diagnose_swiglu_joint.py \
  --checkpoint-dir checkpoints/diagnostic_36h/operator_schmidt/layer0_initial \
  --dense-weights-dir models/diagnostic_36h_cache \
  --layer 0 --device cuda --seq-len 128 --n-heads 32 --n-kv-heads 32 \
  --output experiment_artifacts/diagnostic_36h/joint_swiglu/layer0/layer0_joint_swiglu.json
```

The process exited successfully. Script-measured runtime after loading the
checkpoint, weights, and activations was 10.073 s; candidate construction took
0.213 s. Peak PyTorch CUDA allocated memory over the full diagnostic was
8,366,678,016 bytes (7.79 GiB).
The candidate-construction peak was 5,762,088,960 bytes (5.37 GiB), or
2,028,994,560 bytes (1.89 GiB) above its CUDA allocation baseline. These are
PyTorch allocated-memory counters, not a claim about reserved memory or
inference speed.

## Matched-budget candidate and quality gate

The actual recursive count from the saved artifacts is **9,943,596 floating
tensor coefficients**, not the 7,505,964 pre-run CPU estimate previously
recorded in the design note. The earlier estimate is superseded: the run
counts the actual saved Gate/Up/Down payloads, including their floating
components. At the measured budget the shared candidate uses rank 426 and
9,924,096 floating coefficients (19,500 fewer, 0.20% below NEC).

| Held-out metric | Saved NEC | Shared-neuron candidate | Relative change | Gate |
|---|---:|---:|---:|---|
| MLP output relative error, compressed trajectory | 0.626175 | 0.741295 | 18.38% worse | Fail: required ≥5% improvement |
| Active-transform relative error | 0.430836 | 0.489986 | 13.73% worse | Fail: no regression |
| Full-block output relative error | 0.363740 | 0.413679 | 13.73% worse | Fail: no regression |
| Serialized MLP artifact bytes | 30,370,779 | 19,850,909 | 34.64% smaller | Pass |
| Floating coefficients | 9,943,596 | 9,924,096 | 0.20% fewer | Matched |

Candidate bytes are the actual `torch.save` size of the fitted factor artifact;
NEC bytes are the sum of the three separately serialized saved MLP artifacts.
The candidate tensor payload is 19,848,192 bytes; saved NEC tensor payload is
30,361,016 bytes. Sparse integer-index payloads are included in serialized
bytes and separately counted in the JSON, not treated as floating
coefficients. This is MLP-artifact storage only, not full-layer or full-model
size.

The candidate does improve the held-out SwiGLU product: on the compressed
trajectory its product relative error is 0.587686 versus 0.742007 for saved
NEC; after applying the **dense** Down operator, the product-error term is
0.452381 versus 0.620233. But its Down-error term is 0.521009 versus 0.209682
for saved NEC, and the product/Down cross-energy fraction flips from -0.0853
(partial cancellation) to +0.1542 (reinforcement). Thus the better internal
product does not survive the candidate's Down approximation.

## Eight-way projection ablation

The tables below report the complete nonlinear MLP error on held-out inputs
and, separately, calibration inputs. `GUD` uses `0` for dense and `1` for
saved NEC-compressed Gate, Up, and Down in that order. `MLP` is
`||D̂ĥ-Dh||/||Dh||`; `product` is `||ĥ-h||/||h||`, where
`h=SiLU(Gx)⊙(Ux)`; `product→D` is
`||D(ĥ-h)||/||Dh||`; and `Down term` is
`||(D̂-D)ĥ||/||Dh||`. `D amplification` is the product→D relative error
divided by product relative error, showing how Down's operator geometry scales
the internal product error. `rho=2⟨e_product,e_down⟩/(||e_product||²+
||e_down||²)`; negative means cancellation, positive means reinforcement.
`—` means the corresponding two-term interaction is absent.

### Held-out, dense transformer trajectory

| GUD | MLP | Product | Product→D | Down term | D amplification | rho |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |
| 001 | 0.614 | 0.000 | 0.000 | 0.614 | — | — |
| 010 | 0.570 | 0.679 | 0.570 | 0.000 | 0.839 | — |
| 011 | 0.640 | 0.679 | 0.570 | 0.252 | 0.839 | +0.054 |
| 100 | 0.552 | 0.661 | 0.552 | 0.000 | 0.836 | — |
| 101 | 0.622 | 0.661 | 0.552 | 0.276 | 0.836 | +0.014 |
| 110 | 0.656 | 0.773 | 0.656 | 0.000 | 0.848 | — |
| 111 | 0.662 | 0.773 | 0.656 | 0.186 | 0.848 | -0.055 |

### Held-out, compressed transformer trajectory

| GUD | MLP | Product | Product→D | Down term | D amplification | rho |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |
| 001 | 0.596 | 0.000 | 0.000 | 0.596 | — | — |
| 010 | 0.545 | 0.660 | 0.545 | 0.000 | 0.826 | — |
| 011 | 0.608 | 0.660 | 0.545 | 0.268 | 0.826 | +0.002 |
| 100 | 0.539 | 0.650 | 0.539 | 0.000 | 0.829 | — |
| 101 | 0.596 | 0.650 | 0.539 | 0.287 | 0.829 | -0.047 |
| 110 | 0.620 | 0.742 | 0.620 | 0.000 | 0.836 | — |
| 111 | 0.626 | 0.742 | 0.620 | 0.210 | 0.836 | -0.085 |

### Calibration, dense transformer trajectory

| GUD | MLP | Product | Product→D | Down term | D amplification | rho |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |
| 001 | 0.617 | 0.000 | 0.000 | 0.617 | — | — |
| 010 | 0.578 | 0.688 | 0.578 | 0.000 | 0.841 | — |
| 011 | 0.651 | 0.688 | 0.578 | 0.260 | 0.841 | +0.055 |
| 100 | 0.555 | 0.664 | 0.555 | 0.000 | 0.836 | — |
| 101 | 0.629 | 0.664 | 0.555 | 0.285 | 0.836 | +0.015 |
| 110 | 0.665 | 0.784 | 0.665 | 0.000 | 0.848 | — |
| 111 | 0.673 | 0.784 | 0.665 | 0.189 | 0.848 | -0.053 |

### Calibration, compressed transformer trajectory

| GUD | MLP | Product | Product→D | Down term | D amplification | rho |
|---|---:|---:|---:|---:|---:|---:|
| 000 | 0.000 | 0.000 | 0.000 | 0.000 | — | — |
| 001 | 0.607 | 0.000 | 0.000 | 0.607 | — | — |
| 010 | 0.548 | 0.660 | 0.548 | 0.000 | 0.830 | — |
| 011 | 0.620 | 0.660 | 0.548 | 0.280 | 0.830 | +0.013 |
| 100 | 0.543 | 0.651 | 0.543 | 0.000 | 0.834 | — |
| 101 | 0.608 | 0.651 | 0.543 | 0.301 | 0.834 | -0.040 |
| 110 | 0.627 | 0.748 | 0.627 | 0.000 | 0.838 | — |
| 111 | 0.637 | 0.748 | 0.627 | 0.216 | 0.838 | -0.079 |

On the held-out compressed trajectory, restoring only Up changes total MLP
error from 0.626 to 0.596 (4.8% relative reduction); restoring only Gate gives
0.608 (2.9%); restoring only Down gives 0.620 (0.95%). The Gate/Up product
error is much larger when both are compressed (0.742) than for Gate-only
(0.650) or Up-only (0.660), revealing a nonlinear interaction rather than an
isolated bad matrix. In the exact Gate/Up product expansion, the two
first-order error terms reinforce (`rho=+0.924`), while each largely cancels
with the bilinear interaction term (`rho=-0.949` and `-0.926`). The full
precision pairwise values are in the JSON.

The saved NEC Down error partially cancels the product error on the full
compressed path (`rho=-0.085`), though it still increases the final error
slightly from 0.620 to 0.626. This means its Down matrix is not accurate in
isolation—the Down-only row is 0.596—but its error direction happens to offset
part of the Gate/Up product defect. Independent component error magnitudes
alone would miss that interaction.

## Mathematical diagnosis and limits

The candidate's common neuron projector `P=ZZᵀ` gives
`Ĝ=PG`, `Û=PU`, and `D̂=DP`. Its calibration SVD chooses `P` to retain
unweighted energy in a stack of dense Gate, Up, and SwiGLU activation states.
That construction lowers the held-out product error, but it does not optimize
the downstream operator-weighted loss `||D(ĥ-h)||²` or the full coupled
quantity `||D̂ĥ-Dh||²`. At this rank, the candidate's Down contribution is
2.48× saved NEC's and changes from partial cancellation to reinforcement. The
measured failure is therefore the shared subspace's poor preservation of the
Down action (and its interaction with the product), not a failure to make the
internal product closer.

This one experiment cannot distinguish whether a better output-weighted
subspace would fix the issue or whether the common linear projector is
intrinsically too restrictive around the coordinatewise SiLU/Hadamard
nonlinearity. The evidence does establish that this candidate/objective is
not ready for wider evaluation. No workaround or alternate fitting objective
was tried after the failure.

## Artifacts and stop state

- Full precision measurements: `layer0_joint_swiglu.json`.
- Exact stdout and run fingerprint: `run.log`.
- Both files were copied to
  `/home/fluxx/Workspace/research_backups/nox_joint_swiglu_layer0_20261017_run01/`
  before report edits. Backup SHA-256 values: JSON
  `81feec7c6723c036742fe25ba6e5e599d34747be605eb325f6cf700ea98454b2`;
  log `52acd9ca867b50fee2952d1c2ff8ee47e2b995bfb06381e06f985d46a401be27`.
- No experimental process remains on the A10. The instance was left running;
  no power action was taken. No two-layer/four-layer compile, PPL, MMLU, or
  full-model job was started.
