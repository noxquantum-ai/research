#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEC_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${NEC_WORKSPACE_ROOT:-/workspace}"
RUN_ROOT="${NEC_RUN_ROOT:-$WORKSPACE_ROOT/runs/nec-high-fidelity-quality-max-20261019}"
VENV_DIR="${NEC_VENV_DIR:-$WORKSPACE_ROOT/venvs/nec-hf}"

export NEC_WORKSPACE_ROOT="$WORKSPACE_ROOT"
export NEC_RUN_ROOT="$RUN_ROOT"
export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$RUN_ROOT"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: pinned runtime missing; run bootstrap.sh first." >&2
  exit 2
fi
cd "$REPO_ROOT"
"$VENV_DIR/bin/python" "$SCRIPT_DIR/preflight.py"
