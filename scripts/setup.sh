#!/usr/bin/env bash
# LosslessSync setup for Linux (CachyOS PC or cloud). Idempotent; same steps as setup.ps1.
#   scripts/setup.sh [--check] [--env pc|cloud] [--no-stems] [--no-rust]
# Everything installed lives in ~/.lls (venv, venv-sep, cargo-target), outside the repo.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLS="$HOME/.lls"
CHECK=0 STEMS=1 RUST=1 ENV="${LLS_ENV:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --env) ENV="$2"; shift ;;
    --no-stems) STEMS=0 ;;
    --no-rust) RUST=0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
if [ -z "$ENV" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then ENV=pc; else ENV=cloud; fi
fi
export CARGO_TARGET_DIR="$LLS/cargo-target"
export PATH="$HOME/.cargo/bin:$PATH"
VENV="$LLS/venv" SEP="$LLS/venv-sep"
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

have() { command -v "$1" >/dev/null 2>&1; }
yn() { if eval "$1" >/dev/null 2>&1; then echo yes; else echo no; fi; }

check() {
  echo "environment : $ENV ($(uname -s) $(uname -m))"
  echo "repo        : $REPO"
  echo "venv        : $([ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import losslesssync,sys;print(sys.version.split()[0], "losslesssync ok")' 2>/dev/null || echo missing)"
  echo "ffmpeg      : $(have ffmpeg && echo yes || echo no), rubberband: $(yn 'ffmpeg -hide_banner -filters | grep -q rubberband')"
  echo "cargo       : $(have cargo && cargo --version || echo no)"
  echo "lls-dsp     : $([ -x "$VENV/bin/python" ] && (cd "$REPO" && "$VENV/bin/python" -c 'from losslesssync.stretch import native;print(native.dsp() or "not built")') || echo '?')"
  echo "gpu         : $(have nvidia-smi && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
  echo "stems python: ${LOSSLESSSYNC_EXT_PYTHON:-unset} ($([ -x "$SEP/bin/python" ] && "$SEP/bin/python" -c 'import torch;print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>/dev/null || echo 'no venv-sep'))"
  echo "kaggle/modal: $(yn "[ -x '$VENV/bin/kaggle' ]")/$(yn "[ -x '$VENV/bin/modal' ]"), rclone: $(have rclone && echo yes || echo no)"
  echo "media       : LLS_MEDIA_DIR=${LLS_MEDIA_DIR:-unset} LLS_OUT_DIR=${LLS_OUT_DIR:-unset}"
}
[ "$CHECK" = 1 ] && { check; exit 0; }

persist() {   # persist NAME VALUE: in ~/.bashrc (and fish, CachyOS's default shell); only "# lls" lines are touched
  local f="$HOME/.bashrc" fl
  touch "$f"; sed -i "/^export $1=.*# lls$/d" "$f"; echo "export $1=\"$2\" # lls" >> "$f"
  if have fish; then
    mkdir -p "$HOME/.config/fish/conf.d"; fl="$HOME/.config/fish/conf.d/lls.fish"
    touch "$fl"; sed -i "/# $1$/d" "$fl"
    if [ "$1" = PATH ]; then echo "fish_add_path -g \"${2%%:*}\" # PATH" >> "$fl"
    else echo "set -gx $1 \"$2\" # $1" >> "$fl"; fi
  fi
}

pyfind() { for p in "$@"; do have "$p" && { echo "$p"; return; }; done; echo python3; }
mkdir -p "$LLS"

echo "== system packages"
if have apt-get; then
  need=""
  have ffmpeg || need="$need ffmpeg"
  python3 -c 'import ensurepip' 2>/dev/null || need="$need python3-venv"
  [ "$ENV" = cloud ] && { have rclone || need="$need rclone"; }
  [ "$RUST" = 1 ] && { have c++ || need="$need g++"; have clang || need="$need clang libclang-dev"; }
  [ -n "$need" ] && { $SUDO apt-get update -qq; $SUDO apt-get install -y -qq $need; }
elif have pacman; then
  have ffmpeg || echo "ffmpeg missing: sudo pacman -S ffmpeg rubberband"
  [ "$ENV" = cloud ] && ! have rclone && echo "rclone missing: sudo pacman -S rclone"
fi
ffmpeg -hide_banner -filters 2>/dev/null | grep -q rubberband || echo "WARNING: ffmpeg has no rubberband filter (needed for --pitch keep with rubberband)"

echo "== venv $VENV"
[ -x "$VENV/bin/python" ] || "$(pyfind python3.13 python3.12 python3.11 python3)" -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q -e "$REPO"
[ "$ENV" = cloud ] && "$VENV/bin/python" -m pip install -q kaggle modal
persist PATH "$VENV/bin:\$PATH"

if [ "$RUST" = 1 ]; then
  echo "== lls-dsp (Rust)"
  have cargo || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  persist CARGO_TARGET_DIR "$CARGO_TARGET_DIR"
  (cd "$REPO/dsp" && { cargo build -q --release --features signalsmith 2>/dev/null ||
     { echo "signalsmith feature failed (C++ compiler / libclang?): building without it"; cargo build -q --release; }; })
fi

if [ "$ENV" = pc ] && [ "$STEMS" = 1 ]; then
  echo "== stems venv $SEP (torch CUDA)"
  [ -x "$SEP/bin/python" ] || "$(pyfind python3.12 python3.11 python3.13 python3)" -m venv "$SEP"
  if ! "$SEP/bin/python" -c 'import torch, audio_separator, python_stretch, soundfile' 2>/dev/null; then
    "$SEP/bin/python" -m pip install -q --upgrade pip
    "$SEP/bin/python" -c 'import torch' 2>/dev/null ||
      "$SEP/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126
    "$SEP/bin/python" -m pip install -q "audio-separator[gpu]" python-stretch soundfile
  fi
  persist LOSSLESSSYNC_EXT_PYTHON "$SEP/bin/python"
fi

echo "== done (open a new shell for the environment variables)"
LOSSLESSSYNC_EXT_PYTHON="${LOSSLESSSYNC_EXT_PYTHON:-$([ -x "$SEP/bin/python" ] && echo "$SEP/bin/python")}" check
