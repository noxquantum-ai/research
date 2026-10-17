# Repository map

NEC is the repository’s sole active research focus. The root README states the current evidence and the unresolved accuracy problem. Historical reports remain available under the archive and experiment directories; old claims are preserved as historical records, not current project claims.

## Current implementation and execution

- `nox/` contains the NEC compiler, tensor-network construction, calibration, quantization baselines, and supporting mathematical utilities.
- `scripts/` contains current analysis, compilation, evaluation, and audit entry points. Superseded orchestration and provider-specific launchers are under `scripts/archive/`.
- `kaggle_execution/function_aware_tn/` contains the current private Kaggle diagnostic bundle. `kaggle_execution/archive/` preserves the earlier cascade runner.

## Experiments and evidence

- `docs/experiments/` contains focused research reports, grouped by experiment family.
- `benchmarks/` contains evaluation programs; `tests/` contains CPU and module-level checks.
- `experiment_artifacts/` stores run outputs, logs, manifests, and machine-readable measurements. Do not replace prior results when adding new runs.
- `checkpoints/` and `compressed_models/` retain historical manifests and model artifacts.

## Historical archive

- `docs/archive/EXPERIMENT_LOG.md` preserves the chronological experiment log.
- `docs/archive/legacy/` preserves the former README, broad strategy and novelty documents, and historical scorecards.
- `docs/archive/legacy_strategy/` preserves superseded plans and speculative analyses.
- `experiment_artifacts/archive/` holds archived logs; code and execution records remain in their existing versioned paths or the script archives.

See `docs/archive/README.md` for how to interpret superseded documents and `docs/experiments/README.md` for the research report index.
