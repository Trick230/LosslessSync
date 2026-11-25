#!/usr/bin/env bash
# GPU job on a cloud provider: ./gpu/run_gpu.sh separate SRC.flac OUT.flac [--model M]
# Modal -> Kaggle -> Colab (LONG=1: Kaggle first). The logic is in run_gpu.py, run with the
# project's Python (`lls py gpu/run_gpu.py ...` on Windows).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$HERE/../lls" py "$HERE/run_gpu.py" "$@"
