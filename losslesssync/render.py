"""Sample-accurate assembly of the synced track from the speed-corrected source."""
from __future__ import annotations

import numpy as np

from .wavio import WavInfo, WavWriter

BLOCK = 1 << 18


def _fade(n: int, rising: bool) -> np.ndarray:
    w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return (w if rising else w[::-1]).astype(np.float32)[:, None]


def _copy(w: WavWriter, src: np.ndarray, a: int, n: int, fade_in: int, fade_out: int):
    """Write n frames of src starting at frame a (zeros outside src) with edge fades."""
    pos = 0
    while pos < n:
        m = min(BLOCK, n - pos)
        blk = np.zeros((m, src.shape[1]), np.float32)
        lo, hi = a + pos, a + pos + m
        s0, s1 = max(lo, 0), min(hi, len(src))
        if s1 > s0:
            blk[s0 - lo:s1 - lo] = src[s0:s1]
        if fade_in and pos < fade_in:
            k = min(fade_in - pos, m)
            blk[:k] *= _fade(fade_in, True)[pos:pos + k]
        if fade_out and pos + m > n - fade_out:
            start = max(pos, n - fade_out)
            ramp = _fade(fade_out, False)[start - (n - fade_out):]
            blk[start - pos:] *= ramp[:m - (start - pos)]
        w.write(blk)
        pos += m


def render(inter: WavInfo, pieces: list[dict], n_out: int, out_wav: str,
           fill: WavInfo | None = None, fade_ms: float = 4.0):
    """pieces: sorted, non-overlapping {'start','end' (ref s), 'offset' (s)} on the ref timeline."""
    sr = inter.sr
    src = inter.memmap()
    ref = fill.memmap() if fill is not None else None
    F = max(1, int(round(fade_ms * 1e-3 * sr)))
    w = WavWriter(out_wav, n_out, inter.channels, sr, inter.mask)
    cur = 0
    for p in pieces:
        n0 = min(max(int(round(p["start"] * sr)), cur), n_out)
        n1 = min(max(int(round(p["end"] * sr)), n0), n_out)
        if n0 > cur:
            _gap(w, ref, cur, n0 - cur, F, n_out)
        shift = int(round(p["offset"] * sr))
        n = n1 - n0
        if n > 0:
            _copy(w, src, n0 - shift, n, F if n0 > 0 else 0, F if n1 < n_out else 0)
        cur = n1
    if cur < n_out:
        _gap(w, ref, cur, n_out - cur, F, n_out)
    w.close()


def _gap(w: WavWriter, ref, start: int, n: int, F: int, n_out: int):
    if ref is None:
        w.write_silence(n)
    else:
        _copy(w, ref, start, n, F if start > 0 else 0, F if start + n < n_out else 0)
