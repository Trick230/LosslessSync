"""Sample-accurate assembly of the synced track from the speed-corrected source."""
from __future__ import annotations

import numpy as np

from .wavio import WavInfo, WavWriter

BLOCK = 1 << 18


def _fade(n: int, rising: bool) -> np.ndarray:
    w = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
    return (w if rising else w[::-1]).astype(np.float32)[:, None]


def _copy(w: WavWriter, src: np.ndarray, a: int, n: int, fade_in: int, fade_out: int,
          gain: float = 1.0, under_in: np.ndarray | None = None,
          under_out: np.ndarray | None = None, at: int = 0):
    """Write n frames of src starting at frame a (zeros outside src) with edge fades,
    scaled by gain. under_in / under_out (indexed by output frame, the first written being
    at) rise under the fade in / fade out with the complementary ramp: a cross-fade."""
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
            r = _fade(fade_in, True)[pos:pos + k]
            blk[:k] *= r
            if under_in is not None:
                blk[:k] += _read(under_in, at + pos, k) * (1 - r)
        if fade_out and pos + m > n - fade_out:
            start = max(pos, n - fade_out)
            ramp = _fade(fade_out, False)[start - (n - fade_out):]
            ramp = ramp[:m - (start - pos)]
            blk[start - pos:] *= ramp
            if under_out is not None:
                blk[start - pos:] += _read(under_out, at + start, m - (start - pos)) * (1 - ramp)
        w.write(blk)
        pos += m


def render(inter: WavInfo, pieces: list[dict], n_out: int, out_wav: str,
           fill: WavInfo | None = None, fade_ms: float = 300.0, splice_ms: float = 4.0,
           gain: float = 1.0):
    """pieces: sorted, non-overlapping {'start','end' (ref s), 'offset' (s)} on the ref timeline,
    read from inter, or from their own 'src' (a WavInfo of the same layout) when they have one.
    The source is scaled by gain (the reference fill is not).

    Edges next to a gap fade over fade_ms (inside the piece, so no extra source is used);
    with a fill, the fill rises under that fade (a cross-fade) and is not faded itself.
    Pieces that butt against each other cross-fade over splice_ms centred on the splice, each
    read splice_ms / 2 past its edge, so continuous sound does not dip to silence there;
    a piece too short for that gets a splice_ms fade to silence instead."""
    sr = inter.sr
    src = inter.memmap()
    ref = fill.memmap() if fill is not None else None
    def ms(v):
        return max(1, int(round(v * 1e-3 * sr)))

    F, S = ms(fade_ms), ms(splice_ms)
    H = S // 2
    spans, cur = [], 0
    for p in pieces:
        n0 = min(max(int(round(p["start"] * sr)), cur), n_out)
        n1 = min(max(int(round(p["end"] * sr)), n0), n_out)
        spans.append((n0, n1))
        cur = n1
    # butt[k]: piece k cross-fades into piece k + 1
    butt = [H > 0 and k + 1 < len(spans) and spans[k][1] == spans[k + 1][0] < n_out
            and spans[k][1] - spans[k][0] >= 2 * S and spans[k + 1][1] - spans[k + 1][0] >= 2 * S
            for k in range(len(spans))]
    w = WavWriter(out_wav, n_out, inter.channels, sr, inter.mask)
    cur = 0          # end of the last piece
    for k, p in enumerate(pieces):
        n0, n1 = spans[k]
        if n0 > cur:
            _gap(w, ref, cur, n0 - cur)
        n = n1 - n0
        if n > 0:
            nxt = spans[k + 1][0] if k + 1 < len(spans) else n_out
            Fi = ms(p["fade_in_ms"]) if "fade_in_ms" in p else F
            Fo = ms(p["fade_out_ms"]) if "fade_out_ms" in p else F
            fi = 0 if n0 == 0 else (Fi if n0 > cur and p.get("fade_in", True) else S)
            fo = 0 if n1 >= n_out else (Fo if nxt > n1 else S)
            bi, bo = k > 0 and butt[k - 1], butt[k]
            a, b = n0 + (H if bi else 0), n1 - (H if bo else 0)
            fi, fo = (0 if bi else fi), (0 if bo else fo)
            arr, shift = _source(p, src, sr)
            _copy(w, arr, a - shift, b - a, min(fi, (b - a) // 2), min(fo, (b - a) // 2), gain,
                  under_in=ref if n0 > cur else None,
                  under_out=ref if nxt > n1 and fo else None, at=a)
            if bo:
                arr2, shift2 = _source(pieces[k + 1], src, sr)
                _xfade(w, arr, b - shift, arr2, b - shift2, 2 * H, gain)
        cur = n1
    if cur < n_out:
        _gap(w, ref, cur, n_out - cur)
    w.close()


def _source(p: dict, src: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    return (p["src"].memmap() if "src" in p else src), int(round(p["offset"] * sr))


def _read(src: np.ndarray, a: int, n: int) -> np.ndarray:
    """n frames of src from frame a, zeros outside it."""
    blk = np.zeros((n, src.shape[1]), np.float32)
    s0, s1 = max(a, 0), min(a + n, len(src))
    if s1 > s0:
        blk[s0 - a:s1 - a] = src[s0:s1]
    return blk


def _xfade(w: WavWriter, a_src: np.ndarray, a: int, b_src: np.ndarray, b: int, n: int,
           gain: float):
    """n frames: a_src from frame a fading out plus b_src from frame b fading in (the raised
    cosines sum to 1, so a splice inside continuous sound keeps its level)."""
    blk = _read(a_src, a, n) * _fade(n, False) + _read(b_src, b, n) * _fade(n, True)
    if gain != 1.0:
        blk *= np.float32(gain)
    w.write(blk)


def _gap(w: WavWriter, ref, start: int, n: int):
    if ref is None:
        w.write_silence(n)
    else:
        _copy(w, ref, start, n, 0, 0)       # the pieces' fades cross-fade into it
