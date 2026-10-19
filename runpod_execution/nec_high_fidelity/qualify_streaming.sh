#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEC_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${NEC_WORKSPACE_ROOT:-/workspace}"
VENV_DIR="${NEC_VENV_DIR:-$WORKSPACE_ROOT/venvs/nec-hf}"
QUAL_ROOT="${NEC_STREAMING_QUAL_ROOT:-$WORKSPACE_ROOT/runs/nec-streaming-qualification}"
CALIBRATION_TOKENS=4096
VALIDATION_TOKENS=1024
WINDOWS_PER_CHUNK="${NEC_STREAMING_WINDOWS_PER_CHUNK:-4}"

export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  bash "$SCRIPT_DIR/bootstrap.sh"
fi
mkdir -p "$QUAL_ROOT/logs"
cd "$REPO_ROOT"

common=(
  --start-layer 0 --end-layer 0
  --n-calib "$CALIBRATION_TOKENS" --n-validation "$VALIDATION_TOKENS"
  --seq-len 128 --calibration-sampling uniform --validation-sampling uniform
  --calibration-activation-mode dense --construction-mode independent
  --diagonal-prior-strength 0 --mpo-sweeps 2 --mpo-alpha-awt 0.5
  --representation mpo4 --attention-gauge none --mlp-gauge none
  --mlp-objective independent --budget-allocation uniform
  --defect-bridge-mode none --pees-rank 0
  --model-id NousResearch/Llama-2-7b-hf
  --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d
  --device cuda:0
)

run_variant() {
  local name="$1"
  local backend="$2"
  local statistics_dtype="$3"
  local output="$QUAL_ROOT/$name"
  if [[ -f "$output/compressed_layer_0.pt" ]]; then
    echo "Reusing completed qualification variant: $output"
    return
  fi
  if [[ -e "$output/model_manifest.json" ]]; then
    echo "ERROR: incomplete qualification output exists at $output; inspect it before retrying." >&2
    exit 2
  fi
  "$VENV_DIR/bin/python" scripts/compile_high_fidelity.py \
    --output "$output" \
    --calibration-backend "$backend" \
    --calibration-windows-per-chunk "$WINDOWS_PER_CHUNK" \
    --statistics-dtype "$statistics_dtype" \
    "${common[@]}" 2>&1 | tee "$QUAL_ROOT/logs/$name.log"
}

# The official run.sh remains on the full_matrix backend. This separate command
# is the required real-layer qualification before that configuration changes.
run_variant reference-full-matrix full_matrix float64
run_variant streaming-fp32 streaming_statistics float32
run_variant streaming-fp64 streaming_statistics float64

"$VENV_DIR/bin/python" benchmarks/compare_streaming_nec_runs.py \
  --reference "$QUAL_ROOT/reference-full-matrix" \
  --fp32 "$QUAL_ROOT/streaming-fp32" \
  --fp64 "$QUAL_ROOT/streaming-fp64" \
  --output "$QUAL_ROOT/streaming_qualification_report.json"

echo "Qualification complete: $QUAL_ROOT/streaming_qualification_report.json"
