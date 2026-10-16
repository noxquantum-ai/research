#!/usr/bin/env bash
# Matched single-layer diagnostic; no healing and no full-model compilation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output=checkpoints/diagnostic_36h/layer15_uniform_prior_control
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite $output" >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('checkpoints/diagnostic_36h/broad_validation1024')
baseline = json.loads((root / 'prior0_v_proj.json').read_text())
candidate = json.loads((root / 'prior1_v_proj.json').read_text())
assert baseline['input_file_sha256'] == candidate['input_file_sha256']
b = baseline['controls']['int4_simulation']['stages'][-1]
c = candidate['controls']['int4_simulation']['stages'][-1]
assert len(b['validation_window_relative_errors']) == 8
assert len(c['validation_window_relative_errors']) == 8
assert all(y < x for x, y in zip(b['validation_window_relative_errors'], c['validation_window_relative_errors']))
print('Projection improvement reproduced on all eight windows; proceeding to single-layer control.')
PY
mkdir -p "$output"
date -u
if [[ -f SOURCE_COMMIT ]]; then cat SOURCE_COMMIT; fi
sha256sum scripts/compile_high_fidelity.py scripts/diagnose_reference_pipeline.py nox/compiler.py nox/block_compiler.py
for strength in 0 1; do
  checkpoint="$output/prior$strength"
  .venv/bin/python -u scripts/compile_high_fidelity.py \
    --output "$checkpoint" --start-layer 15 --end-layer 15 \
    --n-calib 1024 --n-validation 1024 --seq-len 128 \
    --calibration-sampling uniform --validation-sampling uniform \
    --diagonal-prior-strength "$strength" \
    > "$output/prior${strength}_compile.log" 2>&1
  .venv/bin/python -u scripts/diagnose_reference_pipeline.py \
    --checkpoint-dir "$checkpoint" --weights-dir models/diagnostic_36h_cache \
    --output "$checkpoint/evaluation.json" \
    > "$output/prior${strength}_evaluation.log" 2>&1
done
echo "COMPLETE: matched single-layer controls saved under $output"
