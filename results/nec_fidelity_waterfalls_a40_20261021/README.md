# NEC absolute-fidelity waterfalls on A40 (2026-10-21)

These read-only analyses reuse the saved stage snapshots from the real
Llama-2-7B layer-0 `down_proj` qualifications. They regenerate and hash-check
the exact 4,096 calibration and 1,024 held-out WikiText-2 activations, then
measure every stage against the original dense projection. No fitting or
selection uses the held-out activations.

## Held-out absolute output error

The metric is
`||(W_stage-W_dense)X_val||_F / ||W_dense X_val||_F`.

| Stage | Historical Q4 bridge | FP16 bridge |
|---|---:|---:|
| Sparse shield | 53.5308% | 53.5308% |
| Sparse + bridge | 42.2429% | 42.0698% |
| TT-SVD initialization | **41.4880%** | **41.3249%** |
| DMRG sweep 1 | 41.7905% | 41.6963% |
| DMRG sweep 2, before final MPO Q4 | 42.0206% | 41.9360% |
| Serialized artifact, after MPO Q4 | 42.2366% | 42.1636% |

The representation is already inaccurate before final quantization. TT-SVD
initialization has approximately 41.3--41.5% held-out error, and final MPO Q4
adds only about 0.22 percentage points to the absolute dense-target error.
Therefore Q4 instability is important for compiler reproducibility but is not
the primary cause of NEC's approximately 42% absolute projection error.

## Calibration versus held-out behavior

For the historical bridge, DMRG lowers calibration error from 36.31% at the TT
warm start to 33.46% after sweep 2, while held-out error worsens from 41.49% to
42.02%. The FP16 bridge shows the same pattern: calibration improves from
36.07% to 33.34%, while held-out error worsens from 41.32% to 41.94%.

This is direct evidence that the current local optimization improves its
calibration objective while degrading the untouched activation set. It does
not by itself identify whether the cause is sample coverage, regularization,
the multilinear parameterization, or their interaction.

## Quantization boundary

Relative to the pre-Q4 operator itself, final MPO Q4 changes held-out outputs
by 4.6749% for the historical bridge and 4.6504% for the FP16 bridge. This is
large enough to matter, but much smaller than the pre-existing dense-target
error. The stage snapshots are stored as dense FP16 tensors, so this boundary
also includes their negligible snapshot-rounding difference from the internal
FP32 operator.

The artifact inventory confirms that historical files retain only final Q4
cores; pre-Q4 cores were not serialized. Per-core historical quantization
ablations therefore cannot be reconstructed honestly from these artifacts.
The separate deterministic MPO diagnostic supplies per-core and gauge tests
without pretending to recover missing state.

## Reproducibility

- Model revision: `8efe6c9b93655b934e27bd9981e3ec13e55aee9d`
- WikiText-2 revision: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Analysis source: `e61a164b305a8012bc5af5a3fe66cdd8adaaa6eb`
- GPU: NVIDIA A40
- Stable/FP16 waterfall wall time: 53/49 seconds
- Combined estimated GPU cost: approximately $0.0167 at $0.59/hour

Machine-readable reports, wrapper records, heartbeats, and complete logs are
stored in this directory. The original 90 MB stage tensors remain on the
persistent RunPod volume and are not duplicated in Git.
