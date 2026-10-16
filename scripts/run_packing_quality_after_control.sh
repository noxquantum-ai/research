#!/usr/bin/env bash
# Serialize GPU evaluation after the already running matched layer experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while pgrep -f '^bash scripts/run_weight_prior_layer_control.sh$' > /dev/null; do
  sleep 15
done
if ! grep -q '^COMPLETE: matched single-layer controls saved' checkpoints/diagnostic_36h/layer15_uniform_prior_control_driver.log; then
  echo "Layer control did not finish successfully; refusing to continue the queue." >&2
  exit 1
fi
.venv/bin/python -u scripts/diagnose_reference_pipeline.py \
  --checkpoint-dir checkpoints/diagnostic_36h/packing_layer15/roundtrip \
  --weights-dir models/diagnostic_36h_cache \
  --output checkpoints/diagnostic_36h/packing_layer15/evaluation.json
