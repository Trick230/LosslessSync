"""Signalsmith Stretch of the whole mix, through lls-dsp stretch (built with --features
signalsmith). A frequency-domain stretcher: no splices, so none of WSOLA's doubled attacks or
phasing on dense material, and the least added roughness of the engines scored by
tools/ground_truth.py. lls-dsp compensates the stretcher's latency; the caller still
measures what is left.
"""
from __future__ import annotations

import subprocess

from ..audio.wavio import read_info
from . import native

BLOCK_MS, INTERVAL_MS = 120.0, 30.0     # lls-dsp stretch's default preset


def available() -> bool:
    return native.has("stretch", "usage: lls-dsp stretch")


def stretch(in_wav: str, out_wav: str, k: float, chunk: int = 1 << 16) -> dict:
    """Write in_wav (float WAV) stretched in time by k (k > 1 lengthens) to out_wav."""
    exe = native.dsp()
    if not exe or not available():
        raise RuntimeError("this lls-dsp has no Signalsmith stretch (lls setup)")
    info = read_info(in_wav)
    x = info.memmap()
    p = subprocess.Popen([exe, "stretch", out_wav, str(info.channels), str(info.sr), repr(k)],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    assert p.stdin is not None and p.stderr is not None
    try:
        for a in range(0, info.frames, chunk):
            p.stdin.write(x[a:a + chunk].tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError(f"lls-dsp stretch failed ({p.returncode}): "
                           f"{err.decode(errors='replace')[-2000:]}")
    return {"engine": "signalsmith", "native": True, "block_ms": BLOCK_MS,
            "interval_ms": INTERVAL_MS}
