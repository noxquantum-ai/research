#!/usr/bin/env bash
# Run the full dense MMLU reference after current A10 jobs finish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

while pgrep -f '^bash scripts/run_weight_prior_layer_control.sh$' > /dev/null \
   || pgrep -f '^bash scripts/run_packing_quality_after_control.sh$' > /dev/null; do
  sleep 30
done

prior1=checkpoints/diagnostic_36h/layer15_uniform_prior_control/prior1/evaluation.json
packed=checkpoints/diagnostic_36h/packing_layer15/evaluation.json
[[ -s "$prior1" ]] || echo "Prior-1 layer result is missing; running independent dense reference anyway."
[[ -s "$packed" ]] || echo "Packed-checkpoint quality result is missing; running independent dense reference anyway."

mkdir -p checkpoints/diagnostic_36h/mmlu
output=checkpoints/diagnostic_36h/mmlu/full_dense_baseline.json
[[ ! -e "$output" && ! -e "${output%.json}.predictions.jsonl" ]] \
  || { echo "Refusing to overwrite existing dense MMLU result" >&2; exit 1; }
date -u
if [[ -f SOURCE_COMMIT ]]; then cat SOURCE_COMMIT; fi
.venv/bin/python -u benchmarks/benchmark_full_mmlu.py \
  --output "$output" --batch-size 1 --max-context 4096 \
  > checkpoints/diagnostic_36h/mmlu/full_dense_baseline.log 2>&1
echo "COMPLETE: dense full-test MMLU baseline saved to $output"
