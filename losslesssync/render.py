"""Sample-accurate assembly of the synced track from the speed-corrected source."""
from __future__ import annotations

import numpy as np

from .wavio import WavInfo, WavWriter

BLOCK = 1 << 18


def _fade(n: int, rising: bool) -> np.ndarray:
    w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return (w if rising else w[::-1]).astype(np.float32)[:, None]


def _copy(w: WavWriter, src: np.ndarray, a: int, n: int, fade_in: int, fade_out: int,
          gain: float = 1.0):
    """Write n frames of src starting at frame a (zeros outside src) with edge fades,
    scaled by gain."""
    pos = 0
    while pos < n:
        m = min(BLOCK, n - pos)
        blk = np.zeros((m, src.shape[1]), np.float32)
        lo, hi = a + pos, a + pos + m
        s0, s1 = max(lo, 0), min(hi, len(src))
        if s1 > s0:
            blk[s0 - lo:s1 - lo] = src[s0:s1]
            if gain != 1.0:
                blk[s0 - lo:s1 - lo] *= np.float32(gain)
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
           fill: WavInfo | None = None, fade_ms: float = 300.0, splice_ms: float = 4.0,
           gain: float = 1.0):
    """pieces: sorted, non-overlapping {'start','end' (ref s), 'offset' (s)} on the ref timeline.
    The source is scaled by gain (the reference fill is not).

    Edges next to a gap fade over fade_ms (inside the piece, so no extra source is used);
    pieces that butt against each other only get a splice_ms click guard."""
    sr = inter.sr
    src = inter.memmap()
    ref = fill.memmap() if fill is not None else None
    def ms(v):
        return max(1, int(round(v * 1e-3 * sr)))

    F, S = ms(fade_ms), ms(splice_ms)
    w = WavWriter(out_wav, n_out, inter.channels, sr, inter.mask)
    cur = 0
    for k, p in enumerate(pieces):
        n0 = min(max(int(round(p["start"] * sr)), cur), n_out)
        n1 = min(max(int(round(p["end"] * sr)), n0), n_out)
        if n0 > cur:
            _gap(w, ref, cur, n0 - cur, S, n_out)
        shift = int(round(p["offset"] * sr))
        n = n1 - n0
        if n > 0:
            nxt = int(round(pieces[k + 1]["start"] * sr)) if k + 1 < len(pieces) else n_out
            Fi = ms(p["fade_in_ms"]) if "fade_in_ms" in p else F
            Fo = ms(p["fade_out_ms"]) if "fade_out_ms" in p else F
            fi = 0 if n0 == 0 else (Fi if n0 > cur and p.get("fade_in", True) else S)
            fo = 0 if n1 >= n_out else (Fo if nxt > n1 else S)
            _copy(w, src, n0 - shift, n, min(fi, n // 2), min(fo, n // 2), gain)
        cur = n1
    if cur < n_out:
        _gap(w, ref, cur, n_out - cur, S, n_out)
    w.close()


def _gap(w: WavWriter, ref, start: int, n: int, F: int, n_out: int):
    if ref is None:
        w.write_silence(n)
    else:
        _copy(w, ref, start, n, F if start > 0 else 0, F if start + n < n_out else 0)
