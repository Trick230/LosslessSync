#!/usr/bin/env bash
# GPU job on a cloud provider: ./gpu/run_gpu.sh separate SRC.flac OUT.flac [--model M]
# Modal -> Kaggle -> Colab (LONG=1: Kaggle first). The logic is in run_gpu.py, which
# Windows runs directly.
set -euo pipefail
PY="$HOME/.lls/venv/bin/python"
[ -x "$PY" ] || PY=python3
exec "$PY" "$(dirname "$0")/run_gpu.py" "$@"
