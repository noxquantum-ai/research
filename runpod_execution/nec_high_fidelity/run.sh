#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NEC_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE_ROOT="${NEC_WORKSPACE_ROOT:-/workspace}"
RUN_ROOT="${NEC_RUN_ROOT:-$WORKSPACE_ROOT/runs/nec-high-fidelity-quality-max-20261019}"
VENV_DIR="${NEC_VENV_DIR:-$WORKSPACE_ROOT/venvs/nec-hf}"

export NEC_REPO_ROOT="$REPO_ROOT"
export NEC_WORKSPACE_ROOT="$WORKSPACE_ROOT"
export NEC_RUN_ROOT="$RUN_ROOT"
export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$VENV_DIR/bin/python" ]] || \
   ! "$VENV_DIR/bin/python" -c 'import importlib.metadata, pip, lm_eval; assert importlib.metadata.version("lm_eval") == "0.4.13"' >/dev/null 2>&1 || \
   ! "$VENV_DIR/bin/python" -m pip check >/dev/null 2>&1; then
  echo "Pinned runtime not found; bootstrapping it on the persistent volume." >&2
  bash "$SCRIPT_DIR/bootstrap.sh"
fi
bash "$SCRIPT_DIR/preflight.sh"
cd "$REPO_ROOT"
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/runpod_driver.py" --mode all
