# Layer-0 smoke: initial launch failure

- Kernel: `toheebogunade/nec-high-fidelity-compile-stage-00`, run 1
- Outcome: `ERROR` before model download or compression began
- Failure: the launcher expected `nec_source_bundle.tar.gz`, but Kaggle had automatically unpacked the dataset; `/kaggle/input` therefore contained zero files with that archive name.
- Exact exception: `FileNotFoundError: Expected one source bundle in Kaggle inputs; found 0`
- Impact: no model weights were downloaded, no layers/projections were compiled, and no PPL or MMLU evaluation was run.
- Recovery: the uploaded dataset was independently downloaded; its extracted manifest and all 45 source-file size/SHA-256 records validated against source commit `425cd41770855b48da7ff0bf45ee5bb46a5c0ff6`. The launcher now accepts either a packed archive or the manifest-verified extracted tree. The retry is the same layer-0 smoke gate.

## Retry 2

- Outcome: `ERROR` before model download or compression began
- Failure: `UnboundLocalError` in the extracted-tree branch because its selected manifest path was not assigned to the shared return variable.
- Impact: still no model download, compiled layer, or benchmark execution.
- Recovery: the launch functions now accept an injectable input root and have end-to-end tests that exercise source discovery, manifest/hash validation, and copying for both compile and evaluation launchers. The full suite is rerun before retry 3.
