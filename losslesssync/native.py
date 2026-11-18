"""Locates the optional native kernels (rust/, binary lls-dsp).

Looked up in $LOSSLESSSYNC_DSP, then rust/target/release next to the package, then PATH.
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
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (env, os.path.join(root, "rust", "target", "release", _EXE), shutil.which(_EXE)):
        if cand and os.path.isfile(cand):
            return cand
    return None


def run(*args) -> str:
    exe = dsp()
    r = subprocess.run([exe, *map(str, args)], stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError(f"{exe} failed ({r.returncode}): {r.stderr.decode(errors='replace')[-2000:]}")
    return r.stdout.decode()
