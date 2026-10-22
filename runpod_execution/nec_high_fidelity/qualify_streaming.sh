#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEC_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${NEC_WORKSPACE_ROOT:-/workspace}"
VENV_DIR="${NEC_VENV_DIR:-$WORKSPACE_ROOT/venvs/nec-hf}"
QUAL_ROOT="${NEC_STREAMING_QUAL_ROOT:-$WORKSPACE_ROOT/runs/nec-streaming-qualification}"
WINDOWS_PER_CHUNK="${NEC_STREAMING_WINDOWS_PER_CHUNK:-4}"
HEARTBEAT_SECONDS="${NEC_QUAL_HEARTBEAT_SECONDS:-60}"
TIMEOUT_SECONDS="${NEC_QUAL_TIMEOUT_SECONDS:-}"
GPU_HOURLY_USD="${NEC_GPU_HOURLY_USD:-0.59}"
STAGE="${1:-help}"

export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<EOF
Usage: $0 STAGE

Each invocation runs exactly one stage and then stops:
  down-reference     Real layer-0 down_proj, full-matrix reference
  down-fp32          Same projection, FP32 streaming statistics
  down-compare       Compare reference and FP32; writes the layer-stage gate
  down-fp64          Optional FP64 streaming diagnostic (never automatic)
  down-compare-fp64  Compare reference, FP32, and FP64
  down-stable-reference  Post-sweep-Q4/48-CG full-matrix numerical control
  down-stable-fp64       Matching FP64 streaming numerical control
  down-stable-compare    Compare the two stable-control variants
  bridge-diagnostic      Bridge-only real-activation Q4/FP16/refit diagnostic
  waterfall-stable-reference  Absolute fidelity of saved stable-reference stages
  waterfall-fp16-reference    Absolute fidelity of saved FP16-bridge stages
  projection-stage-ppl        Dense-model WikiText PPL waterfall using saved down_proj stages
  down-bridge-fp16-reference  Stable control with only the bridge kept FP16
  down-bridge-fp16-fp64       Matching FP64 streaming bridge-without-Q4 control
  down-bridge-fp16-compare    Compare the bridge-without-Q4 variants
  down-bridge-refit-reference Stable control with operator-refit bridge Q4
  down-bridge-refit-fp64      Matching FP64 streaming operator-refit control
  down-bridge-refit-compare   Compare the operator-refit variants
  layer-reference    All seven layer-0 projections, full-matrix reference
  layer-fp32         All seven layer-0 projections, FP32 streaming statistics
  layer-fp64         Optional all-seven FP64 streaming diagnostic
  layer-compare      Compare completed layer variants
  layer0-dev-ppl     Matched dense and one-layer NEC PPL on raw WikiText-2 validation
  layer0-3-reference Extend the verified layer-0 artifact to layers 0-3 (same compiler settings)
  layer0-3-dev-ppl   Matched dense and four-layer NEC PPL on raw WikiText-2 validation
  layer0-pack-audit  Pack round-trip and held-out single-block output check
  status             Print saved status without running computation

There is no timeout by default. Set NEC_QUAL_TIMEOUT_SECONDS to a positive
number to enforce an optional per-stage limit. NEC_QUAL_HEARTBEAT_SECONDS controls
periodic progress/resource logging (default 60 seconds). No stage starts its successor.
Streaming layer stages require a passing down-compare report. The full-matrix
reference layer stage is independent of streaming parity. Terminate the RunPod
pod manually when finished to stop billing.
EOF
}

if [[ "$STAGE" == "help" || "$STAGE" == "--help" || "$STAGE" == "-h" ]]; then
  usage
  exit 0
fi
if [[ ! "$HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NEC_QUAL_HEARTBEAT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ -n "$TIMEOUT_SECONDS" && ! "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NEC_QUAL_TIMEOUT_SECONDS must be empty or a positive integer." >&2
  exit 2
fi
if ! [[ "$GPU_HOURLY_USD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "ERROR: NEC_GPU_HOURLY_USD must be a nonnegative number." >&2
  exit 2
fi

if [[ "$STAGE" == "status" ]]; then
  if [[ ! -d "$QUAL_ROOT" ]]; then
    echo "No qualification state exists at $QUAL_ROOT"
    exit 0
  fi
  find "$QUAL_ROOT" -maxdepth 4 -type f \
    \( -name 'progress.json' -o -name 'last_execution.json' -o -name 'comparison*.json' \) \
    -print -exec python3 -m json.tool {} \;
  if [[ -d "$QUAL_ROOT/logs" ]]; then
    find "$QUAL_ROOT/logs" -maxdepth 1 -type f -name '*.heartbeat.jsonl' \
      -print -exec tail -n 1 {} \;
    find "$QUAL_ROOT/logs" -maxdepth 1 -type f -name '*.log' \
      -print -exec tail -n 25 {} \;
  fi
  exit 0
fi

mkdir -p "$QUAL_ROOT/logs"
cd "$REPO_ROOT"
INVOCATION_STARTED="$(date +%s)"

record_exit() {
  local name="$1"
  local output_dir="$2"
  local exit_code="$3"
  local elapsed="$4"
  local invocation_elapsed="$(( $(date +%s) - INVOCATION_STARTED ))"
  python3 - "$name" "$output_dir" "$exit_code" "$elapsed" "$invocation_elapsed" "$GPU_HOURLY_USD" "$QUAL_ROOT/logs/$name.log" "$QUAL_ROOT/logs/$name.heartbeat.jsonl" "$TIMEOUT_SECONDS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

name, output, exit_code, elapsed, invocation_elapsed, hourly, main_log, heartbeat_log, timeout = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
elapsed = float(elapsed)
invocation_elapsed = float(invocation_elapsed)
record = {
    "stage": name,
    "command_exit_code": int(exit_code),
    "status": "PASS" if int(exit_code) == 0 else "TIMEOUT" if int(exit_code) == 124 else "FAIL",
    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    "elapsed_wall_seconds": elapsed,
    "total_invocation_elapsed_wall_seconds": invocation_elapsed,
    "automatic_timeout": bool(timeout),
    "configured_timeout_seconds": int(timeout) if timeout else None,
    "main_log": main_log,
    "heartbeat_log": heartbeat_log,
    "gpu_hourly_usd": float(hourly),
    "estimated_gpu_cost_usd": elapsed / 3600.0 * float(hourly),
    "estimated_total_invocation_gpu_cost_usd": invocation_elapsed / 3600.0 * float(hourly),
    "billing_warning": "Stopping this process does not stop RunPod billing; terminate the pod manually.",
}
path = output / "last_execution.json"
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

run_logged() {
  local name="$1"
  local output_dir="$2"
  shift 2
  local started elapsed exit_code heartbeat_pid
  started="$(date +%s)"
  python3 benchmarks/qualification_heartbeat.py \
    --stage "$name" \
    --progress "$output_dir/progress.json" \
    --workspace "$WORKSPACE_ROOT" \
    --output "$QUAL_ROOT/logs/$name.heartbeat.jsonl" \
    --interval-seconds "$HEARTBEAT_SECONDS" &
  heartbeat_pid="$!"
  trap 'kill -TERM '"$heartbeat_pid"' 2>/dev/null || true' INT TERM EXIT
  set +e
  if [[ -n "$TIMEOUT_SECONDS" ]]; then
    timeout --foreground --signal=TERM --kill-after=60s "$TIMEOUT_SECONDS" "$@" \
      2>&1 | tee -a "$QUAL_ROOT/logs/$name.log"
  else
    "$@" 2>&1 | tee -a "$QUAL_ROOT/logs/$name.log"
  fi
  exit_code="${PIPESTATUS[0]}"
  set -e
  kill -TERM "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  trap - INT TERM EXIT
  elapsed="$(( $(date +%s) - started ))"
  record_exit "$name" "$output_dir" "$exit_code" "$elapsed"
  echo "IMPORTANT: this process stopped, but RunPod billing continues until you terminate the pod." >&2
  return "$exit_code"
}

run_ppl_eval_or_reuse() {
  local name="$1" output_dir="$2" result_path="$3" variant="$4" checkpoint_dir="$5"
  shift 5
  if [[ -f "$result_path" ]]; then
    local validation_args=(--result "$result_path" --variant "$variant")
    if [[ -n "$checkpoint_dir" ]]; then
      validation_args+=(--checkpoint-dir "$checkpoint_dir")
    fi
    echo "Verifying completed PPL result before reuse: $result_path" | tee -a "$QUAL_ROOT/logs/$name.log"
    "$VENV_DIR/bin/python" benchmarks/validate_ppl_result.py "${validation_args[@]}" \
      2>&1 | tee -a "$QUAL_ROOT/logs/$name.log"
    echo "Reusing verified completed PPL result; no model inference repeated." | tee -a "$QUAL_ROOT/logs/$name.log"
    return 0
  fi
  run_logged "$name" "$output_dir" "$@"
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  run_logged bootstrap "$QUAL_ROOT/bootstrap" \
    env NEC_BOOTSTRAP_PROFILE=qualification bash "$SCRIPT_DIR/bootstrap.sh"
fi

down_stage() {
  local name="$1"
  local backend="$2"
  local dtype="$3"
  local output="$QUAL_ROOT/down_projection/$name"
  local reference_argument=()
  if [[ "$backend" == "streaming_statistics" ]]; then
    reference_argument=(--reference-output "$QUAL_ROOT/down_projection/reference-full-matrix")
  fi
  run_logged "$name" "$output" \
    "$VENV_DIR/bin/python" benchmarks/qualify_streaming_projection.py \
      --output "$output" \
      --backend "$backend" \
      --statistics-dtype "$dtype" \
      "${reference_argument[@]}" \
      --n-calib 4096 --n-validation 1024 --seq-len 128 \
      --windows-per-chunk "$WINDOWS_PER_CHUNK" \
      --device cuda:0 \
      --model-id NousResearch/Llama-2-7b-hf \
      --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d
}

down_stable_stage() {
  local name="$1"
  local backend="$2"
  local output="$QUAL_ROOT/down_projection_stable/$name"
  local reference_argument=()
  if [[ "$backend" == "streaming_statistics" ]]; then
    reference_argument=(--reference-output "$QUAL_ROOT/down_projection_stable/reference-full-matrix")
  fi
  run_logged "stable-$name" "$output" \
    "$VENV_DIR/bin/python" benchmarks/qualify_streaming_projection.py \
      --output "$output" \
      --backend "$backend" \
      --statistics-dtype float64 \
      --quantization-schedule post_sweep \
      --cg-threshold 4096 --cg-iterations 48 \
      "${reference_argument[@]}" \
      --n-calib 4096 --n-validation 1024 --seq-len 128 \
      --windows-per-chunk "$WINDOWS_PER_CHUNK" \
      --device cuda:0 \
      --model-id NousResearch/Llama-2-7b-hf \
      --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d
}

down_bridge_control_stage() {
  local family="$1"
  local mode="$2"
  local name="$3"
  local backend="$4"
  local root="$QUAL_ROOT/down_projection_bridge_$family"
  local output="$root/$name"
  local reference_argument=()
  if [[ "$backend" == "streaming_statistics" ]]; then
    reference_argument=(--reference-output "$root/reference-full-matrix")
  fi
  run_logged "bridge-$family-$name" "$output" \
    "$VENV_DIR/bin/python" benchmarks/qualify_streaming_projection.py \
      --output "$output" \
      --backend "$backend" \
      --statistics-dtype float64 \
      --quantization-schedule post_sweep \
      --bridge-quantization-mode "$mode" \
      --bridge-refit-iterations 4 \
      --cg-threshold 4096 --cg-iterations 48 \
      "${reference_argument[@]}" \
      --n-calib 4096 --n-validation 1024 --seq-len 128 \
      --windows-per-chunk "$WINDOWS_PER_CHUNK" \
      --device cuda:0 \
      --model-id NousResearch/Llama-2-7b-hf \
      --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d
}

compare_bridge_control() {
  local family="$1"
  local root="$QUAL_ROOT/down_projection_bridge_$family"
  run_logged "bridge-$family-compare" "$root" \
    "$VENV_DIR/bin/python" benchmarks/compare_streaming_projection_runs.py \
      --reference "$root/reference-full-matrix" \
      --fp64 "$root/streaming-fp64" \
      --output "$root/comparison-fp64.json" \
      --gpu-hourly-usd "$GPU_HOURLY_USD"
}

waterfall_stage() {
  local family="$1"
  local run_output="$2"
  local output="$run_output/absolute-fidelity-waterfall"
  if [[ ! -f "$run_output/report.json" || ! -d "$run_output/stage_weights" ]]; then
    echo "ERROR: waterfall requires a completed qualification with stage snapshots: $run_output" >&2
    exit 2
  fi
  mkdir -p "$output"
  run_logged "waterfall-$family" "$output" \
    "$VENV_DIR/bin/python" benchmarks/analyze_projection_waterfall.py \
      --run-output "$run_output" \
      --output "$output/report.json" \
      --device cuda:0 \
      --windows-per-chunk "$WINDOWS_PER_CHUNK"
}

projection_stage_ppl() {
  local root="${NEC_STAGE_PPL_ROOT:-$WORKSPACE_ROOT/runs/nec-core-q4-attribution-a40_20261021}"
  local run_output="$root/down_projection_stable/reference-full-matrix"
  local output="$root/single_projection_ppl"
  if [[ ! -f "$run_output/report.json" || ! -d "$run_output/stage_weights" ]]; then
    echo "ERROR: projection-stage-ppl requires saved stable-reference stage weights at $run_output" >&2
    exit 2
  fi
  export NEC_EVAL_GPU_MEMORY_GIB="${NEC_EVAL_GPU_MEMORY_GIB:-32}"
  export NEC_EVAL_CPU_MEMORY_GIB="${NEC_EVAL_CPU_MEMORY_GIB:-24}"
  run_logged projection-stage-ppl "$output" \
    "$VENV_DIR/bin/python" benchmarks/evaluate_single_projection_stages.py \
      --run-output "$run_output" \
      --output-dir "$output" \
      --batch-size 1 \
      --context-length 2048 \
      --stride 512 \
      --block-tokens 512 \
      --bootstrap-repetitions 5000 \
      --offload-folder "$root/evaluation_offload"
}

require_down_gate() {
  local variant="${1:-any}"
  local gate="$QUAL_ROOT/down_projection/comparison.json"
  if [[ "$variant" == "fp64" ]]; then
    gate="$QUAL_ROOT/down_projection/comparison-fp64.json"
  elif [[ "$variant" == "any" && -f "$QUAL_ROOT/down_projection/comparison-fp64.json" ]]; then
    gate="$QUAL_ROOT/down_projection/comparison-fp64.json"
  fi
  if [[ ! -f "$gate" ]]; then
    echo "ERROR: layer stages require a completed down-compare at $gate" >&2
    exit 2
  fi
  "$VENV_DIR/bin/python" - "$gate" "$variant" <<'PY'
import json
import subprocess
import sys
report = json.load(open(sys.argv[1]))
variant = sys.argv[2]
from benchmarks.qualify_streaming_projection import algorithm_identity
if report.get("algorithm_identity") != algorithm_identity():
    raise SystemExit("down_proj gate is stale for the current numerical source fingerprint")
passed = (
    report.get("fp32_passes") if variant == "fp32" else
    report.get("fp64_passes") if variant == "fp64" else
    report.get("fp32_passes") or report.get("fp64_passes")
)
if not passed:
    raise SystemExit(f"{variant} down_proj qualification did not pass; do not start a full layer")
PY
}

require_fp64_needed() {
  local gate="$QUAL_ROOT/down_projection/comparison.json"
  if [[ ! -f "$gate" ]]; then
    echo "ERROR: run down-compare before the optional FP64 diagnostic." >&2
    exit 2
  fi
  "$VENV_DIR/bin/python" - "$gate" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
from benchmarks.qualify_streaming_projection import algorithm_identity
if report.get("algorithm_identity") != algorithm_identity():
    raise SystemExit("down_proj comparison is stale for the current numerical source fingerprint")
if report.get("fp32_passes"):
    raise SystemExit("FP32 already passed; refusing an unnecessary FP64 stage")
PY
}

layer_common=(
  --start-layer 0
  --n-calib 4096 --n-validation 1024
  --seq-len 128 --calibration-sampling uniform --validation-sampling uniform
  --calibration-activation-mode dense --construction-mode independent
  --diagonal-prior-strength 0 --mpo-sweeps 2 --mpo-alpha-awt 0.5
  --mpo-quantization-schedule post_sweep --mpo-cg-threshold 4096 --mpo-cg-iterations 48
  --representation mpo4 --attention-gauge none --mlp-gauge none
  --mlp-objective independent --budget-allocation uniform
  --defect-bridge-mode none --pees-rank 0
  --model-id NousResearch/Llama-2-7b-hf
  --model-revision 8efe6c9b93655b934e27bd9981e3ec13e55aee9d
  --device cuda:0
)

layer_stage() {
  local name="$1"
  local backend="$2"
  local dtype="$3"
  local gate_variant="$4"
  local output="$QUAL_ROOT/layer0/$name"
  local resume_argument=()
  if [[ "$backend" == "streaming_statistics" ]]; then
    require_down_gate "$gate_variant"
  fi
  if [[ -f "$output/model_manifest.json" ]]; then
    resume_argument=(--resume)
  fi
  run_logged "$name" "$output" \
    "$VENV_DIR/bin/python" scripts/compile_high_fidelity.py \
      --output "$output" \
      --calibration-backend "$backend" \
      --calibration-windows-per-chunk "$WINDOWS_PER_CHUNK" \
      --statistics-dtype "$dtype" \
      "${resume_argument[@]}" \
      "${layer_common[@]}" --end-layer 0
}

case "$STAGE" in
  down-reference)
    down_stage reference-full-matrix full_matrix float64
    ;;
  down-fp32)
    down_stage streaming-fp32 streaming_statistics float32
    ;;
  down-fp64)
    require_fp64_needed
    down_stage streaming-fp64 streaming_statistics float64
    ;;
  down-compare)
    run_logged down-compare "$QUAL_ROOT/down_projection" \
      "$VENV_DIR/bin/python" benchmarks/compare_streaming_projection_runs.py \
        --reference "$QUAL_ROOT/down_projection/reference-full-matrix" \
        --fp32 "$QUAL_ROOT/down_projection/streaming-fp32" \
        --output "$QUAL_ROOT/down_projection/comparison.json" \
        --gpu-hourly-usd "$GPU_HOURLY_USD"
    ;;
  down-compare-fp64)
    run_logged down-compare-fp64 "$QUAL_ROOT/down_projection" \
      "$VENV_DIR/bin/python" benchmarks/compare_streaming_projection_runs.py \
        --reference "$QUAL_ROOT/down_projection/reference-full-matrix" \
        --fp32 "$QUAL_ROOT/down_projection/streaming-fp32" \
        --fp64 "$QUAL_ROOT/down_projection/streaming-fp64" \
        --output "$QUAL_ROOT/down_projection/comparison-fp64.json" \
        --gpu-hourly-usd "$GPU_HOURLY_USD"
    ;;
  down-stable-reference)
    down_stable_stage reference-full-matrix full_matrix
    ;;
  down-stable-fp64)
    down_stable_stage streaming-fp64 streaming_statistics
    ;;
  down-stable-compare)
    run_logged stable-down-compare "$QUAL_ROOT/down_projection_stable" \
      "$VENV_DIR/bin/python" benchmarks/compare_streaming_projection_runs.py \
        --reference "$QUAL_ROOT/down_projection_stable/reference-full-matrix" \
        --fp64 "$QUAL_ROOT/down_projection_stable/streaming-fp64" \
        --output "$QUAL_ROOT/down_projection_stable/comparison-fp64.json" \
        --gpu-hourly-usd "$GPU_HOURLY_USD"
    ;;
  bridge-diagnostic)
    stable_reference="$QUAL_ROOT/down_projection_stable/reference-full-matrix"
    if [[ ! -f "$stable_reference/stage_weights/sparse.pt" ]]; then
      echo "ERROR: bridge-diagnostic requires the completed down-stable-reference artifact." >&2
      exit 2
    fi
    bridge_output="$QUAL_ROOT/down_projection_stable/bridge-diagnostic-controls"
    mkdir -p "$bridge_output"
    run_logged bridge-diagnostic "$bridge_output" \
      "$VENV_DIR/bin/python" benchmarks/diagnose_real_bridge.py \
        --output "$bridge_output/report.json" \
        --reference-output "$stable_reference" \
        --device cuda:0 \
        --n-calib 4096 --n-validation 1024 --seq-len 128 \
        --windows-per-chunk "$WINDOWS_PER_CHUNK"
    ;;
  waterfall-stable-reference)
    waterfall_stage stable-reference \
      "$QUAL_ROOT/down_projection_stable/reference-full-matrix"
    ;;
  waterfall-fp16-reference)
    waterfall_stage fp16-reference \
      "$QUAL_ROOT/down_projection_bridge_fp16/reference-full-matrix"
    ;;
  projection-stage-ppl)
    projection_stage_ppl
    ;;
  down-bridge-fp16-reference)
    down_bridge_control_stage fp16 fp16_control reference-full-matrix full_matrix
    ;;
  down-bridge-fp16-fp64)
    down_bridge_control_stage fp16 fp16_control streaming-fp64 streaming_statistics
    ;;
  down-bridge-fp16-compare)
    compare_bridge_control fp16
    ;;
  down-bridge-refit-reference)
    down_bridge_control_stage refit operator_refit_q4 reference-full-matrix full_matrix
    ;;
  down-bridge-refit-fp64)
    down_bridge_control_stage refit operator_refit_q4 streaming-fp64 streaming_statistics
    ;;
  down-bridge-refit-compare)
    compare_bridge_control refit
    ;;
  layer-reference)
    layer_stage reference-post-sweep-q4-48cg full_matrix float64 any
    ;;
  layer-fp32)
    layer_stage streaming-fp32 streaming_statistics float32 fp32
    ;;
  layer-fp64)
    layer_stage streaming-fp64 streaming_statistics float64 fp64
    ;;
  layer-compare)
    require_down_gate any
    variant_arguments=()
    if [[ -f "$QUAL_ROOT/layer0/streaming-fp32/compressed_layer_0.pt" ]]; then
      variant_arguments+=(--fp32 "$QUAL_ROOT/layer0/streaming-fp32")
    fi
    if [[ -f "$QUAL_ROOT/layer0/streaming-fp64/compressed_layer_0.pt" ]]; then
      variant_arguments+=(--fp64 "$QUAL_ROOT/layer0/streaming-fp64")
    fi
    if [[ "${#variant_arguments[@]}" -eq 0 ]]; then
      echo "ERROR: no completed streaming layer variant exists." >&2
      exit 2
    fi
    run_logged layer-compare "$QUAL_ROOT/layer0" \
      "$VENV_DIR/bin/python" benchmarks/compare_streaming_nec_runs.py \
        --reference "$QUAL_ROOT/layer0/reference-post-sweep-q4-48cg" \
        "${variant_arguments[@]}" \
        --output "$QUAL_ROOT/layer0/comparison.json" \
        --gpu-hourly-usd "$GPU_HOURLY_USD"
    ;;
  layer0-dev-ppl)
    layer_dir="$QUAL_ROOT/layer0/reference-post-sweep-q4-48cg"
    eval_dir="$layer_dir/evaluation-dev"
    if [[ ! -f "$layer_dir/model_manifest.json" || ! -f "$layer_dir/compressed_layer_0.pt" ]]; then
      echo "ERROR: layer0-dev-ppl requires the completed all-seven-projection layer-0 reference checkpoint." >&2
      exit 2
    fi
    export NEC_EVAL_GPU_MEMORY_GIB="${NEC_EVAL_GPU_MEMORY_GIB:-32}"
    export NEC_EVAL_CPU_MEMORY_GIB="${NEC_EVAL_CPU_MEMORY_GIB:-48}"
    run_ppl_eval_or_reuse layer0-dense-dev-ppl "$eval_dir/dense" "$eval_dir/dense.json" dense "" \
      "$VENV_DIR/bin/python" benchmarks/evaluate_high_fidelity_nec.py \
        --variant dense \
        --output "$eval_dir/dense.json" \
        --dataset-split validation \
        --context-length 2048 --stride 512 --batch-size 1 \
        --max-eval-tokens 0 \
        --offload-folder "$eval_dir/dense-offload"
    run_ppl_eval_or_reuse layer0-nec-dev-ppl "$eval_dir/nec" "$eval_dir/nec.json" high_fidelity_nec "$layer_dir" \
      "$VENV_DIR/bin/python" benchmarks/evaluate_high_fidelity_nec.py \
        --variant high_fidelity_nec \
        --checkpoint-dir "$layer_dir" \
        --output "$eval_dir/nec.json" \
        --dataset-split validation \
        --allow-partial-prefix \
        --expected-calibration-tokens 4096 \
        --expected-heldout-tokens 1024 \
        --expected-quantization-schedule post_sweep \
        --expected-cg-threshold 4096 --expected-cg-iterations 48 \
        --context-length 2048 --stride 512 --batch-size 1 \
        --max-eval-tokens 0 \
        --offload-folder "$eval_dir/nec-offload"
    run_logged layer0-dev-ppl-compare "$eval_dir/comparison" \
      "$VENV_DIR/bin/python" benchmarks/compare_ppl_results.py \
        --dense "$eval_dir/dense.json" \
        --nec "$eval_dir/nec.json" \
        --output "$eval_dir/comparison/report.json" \
        --block-tokens 512 --bootstrap-repetitions 5000 --seed 20261021
    ;;
  layer0-3-reference)
    layer_dir="$QUAL_ROOT/layer0/reference-post-sweep-q4-48cg"
    if [[ ! -f "$layer_dir/model_manifest.json" || ! -f "$layer_dir/compressed_layer_0.pt" ]]; then
      echo "ERROR: layer0-3-reference requires the completed layer-0 reference artifact." >&2
      exit 2
    fi
    run_logged layer0-3-prefix-reference "$layer_dir" \
      "$VENV_DIR/bin/python" scripts/compile_high_fidelity.py \
        --output "$layer_dir" \
        --calibration-backend full_matrix \
        --calibration-windows-per-chunk "$WINDOWS_PER_CHUNK" \
        --statistics-dtype float64 \
        --resume \
        "${layer_common[@]}" --end-layer 3
    ;;
  layer0-3-dev-ppl)
    layer_dir="$QUAL_ROOT/layer0/reference-post-sweep-q4-48cg"
    eval_dir="$layer_dir/evaluation-dev-prefix4"
    if [[ ! -f "$layer_dir/model_manifest.json" || ! -f "$layer_dir/compressed_layer_3.pt" ]]; then
      echo "ERROR: layer0-3-dev-ppl requires completed NEC artifacts for layers 0-3." >&2
      exit 2
    fi
    export NEC_EVAL_GPU_MEMORY_GIB="${NEC_EVAL_GPU_MEMORY_GIB:-32}"
    export NEC_EVAL_CPU_MEMORY_GIB="${NEC_EVAL_CPU_MEMORY_GIB:-48}"
    run_ppl_eval_or_reuse layer0-3-dense-dev-ppl "$eval_dir/dense" "$eval_dir/dense.json" dense "" \
      "$VENV_DIR/bin/python" benchmarks/evaluate_high_fidelity_nec.py \
        --variant dense \
        --output "$eval_dir/dense.json" \
        --dataset-split validation \
        --context-length 2048 --stride 512 --batch-size 1 \
        --max-eval-tokens 0 \
        --offload-folder "$eval_dir/dense-offload"
    run_ppl_eval_or_reuse layer0-3-nec-dev-ppl "$eval_dir/nec" "$eval_dir/nec.json" high_fidelity_nec "$layer_dir" \
      "$VENV_DIR/bin/python" benchmarks/evaluate_high_fidelity_nec.py \
        --variant high_fidelity_nec \
        --checkpoint-dir "$layer_dir" \
        --output "$eval_dir/nec.json" \
        --dataset-split validation \
        --allow-partial-prefix \
        --expected-calibration-tokens 4096 \
        --expected-heldout-tokens 1024 \
        --expected-quantization-schedule post_sweep \
        --expected-cg-threshold 4096 --expected-cg-iterations 48 \
        --context-length 2048 --stride 512 --batch-size 1 \
        --max-eval-tokens 0 \
        --offload-folder "$eval_dir/nec-offload"
    run_logged layer0-3-dev-ppl-compare "$eval_dir/comparison" \
      "$VENV_DIR/bin/python" benchmarks/compare_ppl_results.py \
        --dense "$eval_dir/dense.json" \
        --nec "$eval_dir/nec.json" \
        --output "$eval_dir/comparison/report.json" \
        --block-tokens 512 --bootstrap-repetitions 5000 --seed 20261021
    ;;
  layer0-pack-audit)
    layer_dir="$QUAL_ROOT/layer0/reference-post-sweep-q4-48cg"
    pack_dir="$layer_dir/packed-control"
    packed_artifacts="$pack_dir/artifacts"
    if [[ ! -f "$layer_dir/compressed_layer_0.pt" || ! -f "$layer_dir/layer_input_validation_layer_0.pt" ]]; then
      echo "ERROR: layer0-pack-audit requires the completed layer-0 checkpoint and saved held-out activations." >&2
      exit 2
    fi
    run_logged layer0-pack-control "$pack_dir/pack-stage" \
      "$VENV_DIR/bin/python" scripts/pack_checkpoint_control.py \
        --input "$layer_dir/compressed_layer_0.pt" \
        --output-dir "$packed_artifacts"
    run_logged layer0-pack-forward-audit "$pack_dir/forward-audit-stage" \
      "$VENV_DIR/bin/python" scripts/audit_packed_checkpoint.py \
        --original "$layer_dir/compressed_layer_0.pt" \
        --packed "$packed_artifacts/packed_checkpoint.pt" \
        --activation-file "$layer_dir/layer_input_validation_layer_0.pt" \
        --output "$packed_artifacts/block_roundtrip.json"
    ;;
  *)
    echo "ERROR: unknown qualification stage: $STAGE" >&2
    usage >&2
    exit 2
    ;;
esac
