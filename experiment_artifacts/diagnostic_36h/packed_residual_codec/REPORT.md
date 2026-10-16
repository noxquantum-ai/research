# Packed codec rate–distortion control

## Status

This record covers the matched two-layer representation experiment, the four-layer
extension, and the qualifying 32-layer zero-healing construction. Full matched MMLU
evaluation is recorded separately after completion.

The conventional control representation is direct, packed, symmetric 3-bit groupwise
quantization with 64-weight row-local groups and calibration-only channel scaling

\[
s_j=\operatorname{clamp}(\operatorname{rms}(X_j)^{0.5},0.1,10),\qquad
\widehat W_{:,j}=Q_3(W_{:,j}s_j)/s_j.
\]

It is constructed once from 1,024 uniformly sampled raw WikiText-2 training tokens.
There are no gradients, post-serialization updates, validation targets, or test
targets in construction.

## Root cause supported by the measurements

The first failure is representational generalization, followed by cross-layer
amplification.

- At the central MPO cut and configured rank 128, discarded preconditioned-residual
  energy is about 0.80–0.87 for layer-0/1 attention projections and about 0.92–0.93
  for Gate, Up, and Down. More DMRG sweeps cannot cross this fixed-rank approximation
  bound.
- The four-layer NEC checkpoint has small full-residual same-input errors for layers
  2 and 3 (0.00905 and 0.01246), while their active-transformation errors are 0.8594
  and 0.9261. Residual skips therefore conceal severe operator error.
- Four NEC layers compose to 0.23815 held-out hidden relative error and PPL 511.62.
  Replacing the same seven projections with the direct packed codec reduces composed
  error to 0.10224 and PPL to 5.35404. This rules out transformer depth alone as the
  cause.
- The packed NEC-plus-2-bit-residual candidate is worse than direct 3-bit in both
  serialized bytes and held-out language quality. Retaining the current MPO consumes
  bytes without preserving enough useful operator information.

The measurements support an MPO rank/coordinate mismatch as the dominant cause for
these layers. Correlated nonlinear error propagation still explains why the poor
operators become catastrophic when composed, but cross-layer fitting cannot recover
information that the representation has already discarded.

## Matched two-layer comparison

All rows use the pinned Llama-2-7B revision, 1,024 raw training calibration tokens,
separate validation activations, and the 10,000-input-token raw WikiText-2 test
protocol with 9,999 labels. Bytes are actual `torch.save` file sizes for both layers.

| Representation | Bytes | PPL | NLL | Next-token accuracy | Dense top-1 agreement |
|---|---:|---:|---:|---:|---:|
| Dense | — | 5.29220 | 1.66623 | 61.0461% | 100% |
| NEC MPO baseline | 139,846,298 | 9.44237 | 2.24521 | 52.8853% | 72.2573% |
| Direct 3-bit, unscaled | 164,509,386 | 5.36666 | 1.68021 | 60.9361% | 95.0595% |
| **Direct 3-bit, alpha 0.5** | **164,655,290** | **5.32334** | **1.67210** | **60.9661%** | **97.3897%** |
| Direct 4-bit, alpha 0.5 | 215,249,082 | 5.30060 | 1.66782 | 61.0161% | 98.8999% |
| NEC + packed 2-bit residual, alpha 0.5 | 253,839,466 | 5.66524 | 1.73435 | 60.0160% | 90.3990% |

Direct 3-bit alpha 0.5 is the conventional Pareto control. It costs 17.74% more serialized
bytes than NEC for the pair and recovers nearly all two-layer language quality. The
hybrid is rejected by the preregistered dominance rule.

## Four-layer extension

| Representation | Bytes | PPL | NLL | Next-token accuracy | Dense top-1 agreement | Composed hidden error |
|---|---:|---:|---:|---:|---:|---:|
| Dense | — | 5.29220 | 1.66623 | 61.0461% | 100% | 0 |
| NEC MPO baseline | 279,960,564 | 511.62442 | 6.23759 | 16.9517% | 20.6221% | 0.23815 |
| **Direct 3-bit, alpha 0.5** | **329,310,580** | **5.35404** | **1.67785** | **60.9161%** | **94.7795%** | **0.10224** |

The candidate removes the catastrophic four-layer cliff and generalizes beyond the
two-layer selection case.

## Complete 32-layer control construction and storage audit

The full artifact was constructed atomically by `scripts/compile_direct_packed.py`.
Every layer is fitted during initial construction from the dense training trajectory.
The complete operation took 30.94 seconds on the A10.

| Stored component | Serialized bytes |
|---|---:|
| 32 packed projection/layer-norm files | 2,633,959,038 |
| Embedding, final norm, and LM head file | 524,298,655 |
| Tokenizer and config metadata | 3,621,208 |
| Manifest | 7,942 |
| **Complete delivered bundle** | **3,161,886,843** |
| Original pinned safetensor weight shards | 13,476,876,272 |
| **Honest serialized weight-bundle ratio** | **4.2623x** |

The source tensor payload is 13,476,839,424 bytes. The candidate tensor payload,
including packed codes, every scale, all layer norms, embeddings, final norm, and LM
head, is 3,157,975,040 bytes (4.2675x). Container and metadata overhead accounts for
the small difference between payload and file ratios.

This is a storage result. The current evaluator reconstructs FP16 matrices into the
dense Hugging Face model and therefore does not demonstrate a runtime-memory or speed
improvement.

On the matched 32-layer WikiText protocol:

| Model | PPL | NLL | Next-token accuracy | Relative accuracy retention | Dense top-1 agreement |
|---|---:|---:|---:|---:|---:|
| Dense | 5.29220 | 1.66623 | 61.0461% | 100% | 100% |
| Direct packed 3-bit | 6.8153 | 1.91917 | 57.0657% | 93.48% | 78.9979% |

## Scientific assessment

Packed groupwise quantization and activation scaling are established techniques. This
artifact does not use quantum-computing principles and is not a successful NEC result.
The substantive research
result is the falsification of the current all-MPO representation at its measured
rate: tensor-cut spectral tails predict a severe capacity limit, and a simpler packed
operator preserves composed behavior far better at only 17.6–17.7% more layer bytes.

The proposed entropy gate remains a useful system policy: use an MPO only when its
activation-preconditioned cut spectra and held-out rate–distortion result beat the
alternative codec. In the measured early layers, the gate selects direct packed
quantization. The data does not support adding PEES, a packed residual, larger bridges,
or further cross-layer fitting to these MPO artifacts.

## Recommendation

Keep the complete direct 3-bit artifact only as the accuracy/storage control. Do not
present it as Nox or NEC. A qualifying Nox candidate must retain a material tensor-network
component and beat this control on a held-out matched-byte comparison. Work on packed
matrix kernels is also required before claiming inference-memory compression.

The simulated-4-bit-free NEC recompile was not run: removing rounding from an FP16
container would isolate one error source but would require another expensive MPO
compile, while the fixed-rank cut tails and direct-codec dominance already answer the
go/no-go representation question. This remains an unmeasured implementation control,
not an assumed result.
