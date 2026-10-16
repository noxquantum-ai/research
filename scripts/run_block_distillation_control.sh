#!/usr/bin/env bash
# Refine and evaluate exactly one high-fidelity, zero-healing layer-15 checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

source_dir=checkpoints/diagnostic_36h/layer15_uniform_prior_control/prior0
output=checkpoints/diagnostic_36h/layer15_block_distillation
weights=models/diagnostic_36h_cache
mkdir -p "$(dirname "$output")"
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite $output" >&2
  exit 1
fi
[[ -s "$source_dir/compressed_layer_15.pt" && -s "$source_dir/model_manifest.json" ]]
[[ -s "$weights/layer_15_q_proj.pt" ]]

date -u
if [[ -f SOURCE_COMMIT ]]; then cat SOURCE_COMMIT; fi
sha256sum scripts/refine_checkpoint_block.py nox/block_distillation.py \
  scripts/diagnose_reference_pipeline.py scripts/audit_checkpoint_reconstruction.py

.venv/bin/python -u scripts/refine_checkpoint_block.py \
  --checkpoint-dir "$source_dir" --weights-dir "$weights" --output "$output" \
  --epochs 8 --learning-rate 1e-4 --patience 3 \
  > "${output}_refinement.log" 2>&1

.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir "$output" --weights-dir "$weights" \
  --output "$output/evaluation.json" \
  > "${output}_evaluation.log" 2>&1

.venv/bin/python -u scripts/audit_checkpoint_reconstruction.py \
  --checkpoint-dir "$output" --dense-weights-dir "$weights" --layer 15 \
  --device cuda --activation-file "$output/dense_input_layer_15.pt" \
  > "${output}_audit.log" 2>&1

echo "COMPLETE: layer-15 high-fidelity zero-healing distillation control saved to $output"
