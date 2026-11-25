"""Locates the optional native kernels (dsp/, written in Rust, binary lls-dsp).

Looked up in $LOSSLESSSYNC_DSP, then the newest build (release or dev-fast profile) in
dsp/target next to the package or in the shared $CARGO_TARGET_DIR (default ~/.lls/cargo-target),
then PATH.
Set LOSSLESSSYNC_DSP=off to force the pure-Python code.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache

_EXE = "lls-dsp.exe" if os.name == "nt" else "lls-dsp"


@lru_cache(maxsize=1)
def dsp() -> str | None:
    env = os.environ.get("LOSSLESSSYNC_DSP", "")
    if env.lower() in ("off", "0", "no"):
        return None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if env and os.path.isfile(env):
        return env
    shared = os.environ.get("CARGO_TARGET_DIR") or os.path.join(os.path.expanduser("~"), ".lls", "cargo-target")
    builds = [os.path.join(t, prof, _EXE) for t in (os.path.join(root, "dsp", "target"), shared)
              for prof in ("release", "dev-fast")]
    builds = [b for b in builds if os.path.isfile(b)]
    if builds:
        return max(builds, key=os.path.getmtime)
    return shutil.which(_EXE)


@lru_cache(maxsize=None)
def has(sub: str, word: str = "") -> bool:
    """Whether the kernel has this subcommand (its usage text names `word`, if given)."""
    exe = dsp()
    if not exe:
        return False
    r = subprocess.run([exe, sub], stdin=subprocess.DEVNULL, capture_output=True)
    return (sub.encode() in r.stderr) and (word.encode() in r.stderr)


def run(*args) -> str:
    exe = dsp()
    r = subprocess.run([exe, *map(str, args)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"{exe} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-2000:]}")
    return r.stdout.decode()
