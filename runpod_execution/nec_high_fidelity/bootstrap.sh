#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEC_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${NEC_WORKSPACE_ROOT:-/workspace}"
VENV_DIR="${NEC_VENV_DIR:-$WORKSPACE_ROOT/venvs/nec-hf}"

if ! mountpoint -q "$WORKSPACE_ROOT"; then
  echo "ERROR: $WORKSPACE_ROOT is not a mounted persistent RunPod volume." >&2
  echo "Attach a RunPod network volume at $WORKSPACE_ROOT before bootstrapping." >&2
  exit 2
fi
if [[ ! -f "$REPO_ROOT/kaggle_execution/nec_high_fidelity/run_config.json" ]]; then
  echo "ERROR: NEC repository not found at $REPO_ROOT" >&2
  exit 2
fi

export UV_CACHE_DIR="$WORKSPACE_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE_ROOT/cache/uv/python"
export PIP_CACHE_DIR="$WORKSPACE_ROOT/cache/pip"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$(dirname "$VENV_DIR")"

python3 -m pip install --user --disable-pip-version-check "uv==0.8.17"
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13.15
uv venv --python 3.13.15 "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0"
uv pip install --python "$VENV_DIR/bin/python" \
  -r "$SCRIPT_DIR/requirements.txt"
"$VENV_DIR/bin/python" -m pip check

echo "Environment ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run preflight: $SCRIPT_DIR/preflight.sh"
