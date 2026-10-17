# Remote GPU run records

These are text logs and manifests copied from `/home/shadeform/diag36h-research/checkpoints/diagnostic_36h/` on the user-provided A10 host. Checkpoint tensors and model weights remain on that host; no large model artifacts are duplicated in this repository. Summary, commands, split provenance, and measured results are in the repository-root `docs/archive/EXPERIMENT_LOG.md`.

`remote_records/` preserves remote directory names and is the authoritative mapping of logs to runs. Earlier flat copies remain for continuity. `reference_v1/` records the raw-versus-processed data control and the normalization parity diagnosis. Earlier files that label PyTorch text as raw have incorrect dataset labels; see the correction in `docs/archive/EXPERIMENT_LOG.md`.
