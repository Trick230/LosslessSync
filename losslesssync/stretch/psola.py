"""Pitch-synchronous time stretch for the vocals stem (TD-PSOLA, dsp/src/psola.rs).

Whole pitch periods are repeated (or dropped) where a period matches its neighbour best, and
the input is copied exactly in between, so a voice keeps its waveform: no phasiness and none
of the doubled-period roughness WSOLA can leave on a voice. The timing error stays within
t_max_ms of the exact stretched timeline (a period is forced past that).
"""
from __future__ import annotations

from ..audio.wavio import WavWriter, read_info
from . import native


def available() -> bool:
    return native.has("psola")


def stretch(in_wav: str, out_wav: str, k: float, t_min_ms: float = 2.0,
            t_max_ms: float = 10.0) -> dict:
    info = read_info(in_wav)
    n_out = int(round(info.frames * k))
    w = WavWriter(out_wav, n_out, info.channels, info.sr, info.mask)
    w.f.close()                           # header only: the kernel appends the samples
    voiced, rep, forced, med, mx = native.run(
        "psola", in_wav, info.offset, info.frames, info.channels, out_wav, n_out, repr(k),
        info.sr, t_min_ms, t_max_ms).split()
    return {"engine": "psola", "native": True, "t_min_ms": t_min_ms, "t_max_ms": t_max_ms,
            "voiced_pct": float(voiced) * 100, "periods": int(rep), "forced": int(forced),
            "err_median_ms": float(med), "err_max_ms": float(mx)}
