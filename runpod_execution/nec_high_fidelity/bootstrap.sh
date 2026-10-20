#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEC_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${NEC_WORKSPACE_ROOT:-/workspace}"
VENV_DIR="${NEC_VENV_DIR:-$WORKSPACE_ROOT/venvs/nec-hf}"
BOOTSTRAP_PROFILE="${NEC_BOOTSTRAP_PROFILE:-production}"

if [[ "$BOOTSTRAP_PROFILE" != "production" && "$BOOTSTRAP_PROFILE" != "qualification" ]]; then
  echo "ERROR: NEC_BOOTSTRAP_PROFILE must be production or qualification." >&2
  exit 2
fi

if ! mountpoint -q "$WORKSPACE_ROOT"; then
  echo "ERROR: $WORKSPACE_ROOT is not a mounted persistent RunPod volume." >&2
  echo "Attach a RunPod network volume at $WORKSPACE_ROOT before bootstrapping." >&2
  exit 2
fi
if [[ ! -f "$SCRIPT_DIR/run_config.json" ]]; then
  echo "ERROR: NEC repository not found at $REPO_ROOT" >&2
  exit 2
fi

export UV_CACHE_DIR="$WORKSPACE_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE_ROOT/cache/uv/python"
export PIP_CACHE_DIR="$WORKSPACE_ROOT/cache/pip"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$(dirname "$VENV_DIR")"

# uv's managed-Python catalog is versioned with uv. 0.8.17 predates the
# CPython 3.13.15 standalone build and reports it as unavailable even though
# that build exists. Keep uv pinned, but use a release that knows this exact
# interpreter target.
python3 -m pip install --user --disable-pip-version-check "uv==0.13.1"
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.13.15
uv venv --python 3.13.15 --seed --allow-existing "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" "pip==25.2"
uv pip install --python "$VENV_DIR/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0"
uv pip install --python "$VENV_DIR/bin/python" \
  -r "$SCRIPT_DIR/requirements.txt"
if [[ "$BOOTSTRAP_PROFILE" == "production" ]]; then
  uv pip install --python "$VENV_DIR/bin/python" \
    "lm_eval[hf] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@ddd67220430a2470529f25fd5c05a576ca1057a0"
  # Reassert the audited numerical/runtime pins after resolving harness extras.
  uv pip install --python "$VENV_DIR/bin/python" \
    -r "$SCRIPT_DIR/requirements.txt"
fi
"$VENV_DIR/bin/python" -m pip check
if [[ "$BOOTSTRAP_PROFILE" == "production" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
assert importlib.metadata.version("lm_eval") == "0.4.13"
PY
fi

echo "Environment ready: $VENV_DIR (profile=$BOOTSTRAP_PROFILE)"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Run preflight: $SCRIPT_DIR/preflight.sh"
