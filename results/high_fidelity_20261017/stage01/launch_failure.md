# Stage-1 launch failure record

- Kernel: `toheebogunade/nec-high-fidelity-compile-stage-01`, run 1
- Outcome: `ERROR` before model download or compression began.
- Exact failure: `FileNotFoundError: Expected exactly one packed bundle or extracted bundle manifest in Kaggle inputs; found 0 archive(s), 2 manifest(s)`.
- Cause: Kaggle attached the same extracted source bundle from both the private source dataset and the previous stage output.
- Recovery: both source manifests and every declared file are now independently verified. Duplicate inputs are accepted only if the exact manifest SHA-256 matches; conflicting bundles fail closed. Dataset inputs are preferred deterministically. Tests cover both launchers and both identical/conflicting duplicate cases.
- Impact: no new layers were compiled; the only valid checkpoint remains stage 0, layers `[0]`.
