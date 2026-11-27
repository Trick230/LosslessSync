"""Segments: building, fine offsets, splitting and merging, residual drift."""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy.signal import fftconvolve

from .features import ANA_SR, _nw, pmap
from .speed import Anchor, gcc_phat, is_confident, parabolic

# ---------------------------------------------------------------- segments

@dataclass
class Segment:
    anchors: list[Anchor]
    offset: float = 0.0              # ref = src' + offset (s), src' = speed-corrected source
    fine: list = field(default_factory=list)   # (src'_t, offset, conf, method)
    spread_ms: float | None = None
    method: str = "coarse"
    ref_start: float = 0.0
    ref_end: float = 0.0

    def coarse_offset(self, k):
        return float(np.median([a.off(k) for a in self.anchors]))

    def core(self, k, chunk_s):
        """Trusted ref interval: from first to last anchor chunk centre."""
        c = [a.ref + chunk_s / 2 for a in self.anchors]
        return min(c), max(c)

    def src_core(self, k, chunk_s):
        c = [a.src_raw * k + chunk_s / 2 for a in self.anchors]
        return min(c), max(c)


def build_segments(anchors: list[Anchor], k: float, chunk_s: float, tol: float = 0.035,
                   min_anchors: int = 2) -> list[Segment]:
    conf = sorted((a for a in anchors if is_confident(a)), key=lambda a: a.src_raw)
    groups: list[list[Anchor]] = []
    for a in conf:
        o = a.off(k)
        for g in reversed(groups[-4:]):
            if abs(o - np.median([b.off(k) for b in g[-8:]])) < tol:
                g.append(a)
                break
        else:
            groups.append([a])
    groups = [g for g in groups if len(g) >= min_anchors]
    # merge src-adjacent groups that share an offset
    merged: list[list[Anchor]] = []
    for g in groups:
        if merged and abs(np.median([a.off(k) for a in merged[-1]]) -
                          np.median([a.off(k) for a in g])) < tol:
            merged[-1] = sorted(merged[-1] + g, key=lambda a: a.src_raw)
        else:
            merged.append(g)
    segs = [Segment(anchors=g) for g in merged]
    # drop segments nested inside a larger one in source time (repeated content, false hits)
    keep = []
    for s in segs:
        lo, hi = s.src_core(k, chunk_s)
        nested = any(o is not s and len(o.anchors) > len(s.anchors) and
                     o.src_core(k, chunk_s)[0] <= lo and hi <= o.src_core(k, chunk_s)[1]
                     for o in segs)
        if not nested:
            keep.append(s)
    for s in keep:
        s.offset = s.coarse_offset(k)
    return keep


# ---------------------------------------------------------------- fine offsets

def fine_measure(ref16: np.ndarray, src16: np.ndarray, r0: float, off: float, length: float,
                 max_lag: float = 0.08, phat_thr: float = 7.0):
    """Refine offset at one window. Returns (offset, confidence, method) or None."""
    sr = ANA_SR
    a0 = int(round(r0 * sr))
    b0 = int(round((r0 - off) * sr))
    L = int(length * sr)
    if a0 < 0 or b0 < 0 or a0 + L > len(ref16) or b0 + L > len(src16):
        return None
    a, b = ref16[a0:a0 + L], src16[b0:b0 + L]
    if np.sqrt(np.mean(a * a)) < 1e-4 or np.sqrt(np.mean(b * b)) < 1e-4:
        return None
    d, conf = gcc_phat(a, b, sr, max_lag)
    if conf >= phat_thr:
        return off + d, conf, "phat"
    # fallback: multi-band onset flux at 1 ms (works when phase is not coherent,
    # e.g. a time-stretched source)
    m = int(max_lag * sr)
    if b0 - m < 0 or b0 + L + m > len(src16):
        return None
    # Flux frames come from a per-signal cache on a global 1 ms grid (overlapping windows
    # share them): each window starts on the grid frame nearest to it, and `shift` puts
    # the measured lag back on the exact sample positions.
    fa, fb = round(a0 / FLUX_HOP), round((b0 - m) / FLUX_HOP)
    Fa = flux_slice(ref16, fa, (L - FLUX_N_FFT) // FLUX_HOP + 1)
    Fb = flux_slice(src16, fb, (L + 2 * m - FLUX_N_FFT) // FLUX_HOP + 1)
    if Fa is None or Fb is None:
        return None
    shift = (fa * FLUX_HOP - a0) - (fb * FLUX_HOP - (b0 - m))
    n = Fa.shape[1]
    nl = Fb.shape[1] - n + 1
    if nl < 3:
        return None
    # all lags at once, summed over bands: correlation through the FFT in float64 (a
    # windowed einsum copied bands x n x lags floats and summed in float32)
    vals = fftconvolve(Fb.astype(np.float64), Fa[:, ::-1].astype(np.float64),
                       mode="valid", axes=1).sum(0) / Fa.size
    j = int(np.argmax(vals))
    if vals[j] < 0.12 or j == 0 or j == nl - 1:
        return None
    lag = j + parabolic(vals, j)
    M = (nl - 1) / 2
    return off + ((M - lag) * FLUX_HOP + shift) / sr, float(vals[j]), "flux"


FLUX_HOP = 16  # samples at 16 kHz = 1 ms
FLUX_N_FFT = 256
FLUX_BLOCK = 8192  # frames per cached block (~8 s)

# id(signal) -> (weakref to it, {block: log band energies}, {block: lock while computed})
_BAND_CACHE: dict[int, tuple[weakref.ref, dict[int, np.ndarray], dict[int, threading.Lock]]] = {}
_BAND_LOCK = threading.Lock()


def _band_block(x: np.ndarray, blk: int) -> np.ndarray:
    """Log band energies (bands x frames) of frames [blk * FLUX_BLOCK, ...) of x, frame f
    starting at sample f * FLUX_HOP; computed once per signal and block. Pool threads
    measure neighbouring windows, so a block being computed is waited for, not redone."""
    key = id(x)
    with _BAND_LOCK:
        ent = _BAND_CACHE.get(key)
        if ent is None or ent[0]() is not x:
            for k in [k for k, e in _BAND_CACHE.items() if e[0]() is None]:
                del _BAND_CACHE[k]
            ent = (weakref.ref(x), {}, {})
            _BAND_CACHE[key] = ent
        B = ent[1].get(blk)
        if B is not None:
            return B
        lock = ent[2].setdefault(blk, threading.Lock())
    with lock:
        B = ent[1].get(blk)
        if B is None:
            win, W = _flux_bands(FLUX_N_FFT, 8)
            s0 = blk * FLUX_BLOCK * FLUX_HOP
            seg = np.asarray(x[s0:s0 + (FLUX_BLOCK - 1) * FLUX_HOP + FLUX_N_FFT], np.float32)
            n = max(0, (len(seg) - FLUX_N_FFT) // FLUX_HOP + 1)
            frames = np.lib.stride_tricks.sliding_window_view(seg, FLUX_N_FFT)[::FLUX_HOP][:n]
            mag = np.abs(sfft.rfft(frames * win, axis=1, workers=_nw()))
            B = np.log(mag @ W + 1e-5).T
            ent[1][blk] = B
    return B


def flux_slice(x: np.ndarray, f0: int, n: int) -> np.ndarray | None:
    """Half-wave rectified log-magnitude flux in log-spaced bands, z-scored, over the n
    frames of x from grid frame f0 (frame f at sample f * FLUX_HOP, 1 ms hop), from the
    per-signal cache; None if they do not fit in x."""
    if n < 2 or f0 < 0 or (f0 + n - 1) * FLUX_HOP + FLUX_N_FFT > len(x):
        return None if n >= 2 else np.zeros((8, 0), np.float32)
    b0, b1 = f0 // FLUX_BLOCK, (f0 + n - 1) // FLUX_BLOCK
    B = np.concatenate([_band_block(x, b) for b in range(b0, b1 + 1)], axis=1)
    B = B[:, f0 - b0 * FLUX_BLOCK:f0 - b0 * FLUX_BLOCK + n]
    F = np.maximum(np.diff(B, axis=1), 0)
    F -= F.mean(1, keepdims=True)
    F /= F.std(1, keepdims=True) + 1e-9
    return F.astype(np.float32)


_FLUX_CACHE: dict = {}


def _flux_bands(n_fft: int, n_bands: int):
    """Hann window and the (bins x bands) matrix that averages magnitudes per band."""
    key = (n_fft, n_bands)
    if key not in _FLUX_CACHE:
        f = np.fft.rfftfreq(n_fft, 1.0 / ANA_SR)
        edges = np.geomspace(150, 7000, n_bands + 1)
        W = np.stack([((f >= lo) & (f < hi)) / max(1, ((f >= lo) & (f < hi)).sum())
                      for lo, hi in zip(edges[:-1], edges[1:])], axis=1).astype(np.float32)
        _FLUX_CACHE[key] = (np.hanning(n_fft).astype(np.float32), W)
    return _FLUX_CACHE[key]


def fine_segments(segs: list[Segment], ref16, src16, k: float, chunk_s: float, progress=None):
    total = sum(len(s.anchors) for s in segs)
    done = 0
    for s in segs:
        res_all = pmap(lambda a: fine_measure(ref16, src16, a.ref, a.off(k), chunk_s),
                       s.anchors)
        done += len(s.anchors)
        if progress:
            progress(done / max(1, total))
        s.fine = [(a.src_raw * k + chunk_s / 2, res[0], res[1], res[2])
                  for a, res in zip(s.anchors, res_all) if res]
        summarize_fine(s)


def phase_coherent(s: Segment) -> bool:
    """Most windows phase-coherent: isolated GCC-PHAT hits in an incoherent source are
    chance peaks and must not set the offset."""
    ph = sum(f[3] == "phat" for f in s.fine)
    return ph >= 2 and ph >= 0.5 * len(s.fine)


def summarize_fine(s: Segment):
    ph = [f for f in s.fine if f[3] == "phat"]
    coherent = phase_coherent(s)
    use = ph if coherent else s.fine
    if use:
        vals = np.array([f[1] for f in use])
        s.offset = float(np.median(vals))
        s.spread_ms = float(1.4826 * np.median(np.abs(vals - s.offset)) * 1000)
        s.method = "phat" if coherent else "flux"
    else:
        s.method = "coarse"
        s.spread_ms = None


def split_on_jumps(segs: list[Segment], k: float, chunk_s: float,
                   min_jump: float = 0.002) -> list[Segment]:
    """Split a segment where its fine offset jumps (e.g. dropped packets in a recording)."""
    out = []
    stack = list(segs)
    while stack:
        s = stack.pop(0)
        ph = sorted([f for f in s.fine if f[3] == "phat"], key=lambda f: f[0])
        if not phase_coherent(s):
            ph = []
        best = None
        for i in range(3, len(ph) - 2):
            L = np.array([f[1] for f in ph[:i]])
            Rr = np.array([f[1] for f in ph[i:]])
            jump = abs(np.median(L) - np.median(Rr))
            mad = max(np.median(np.abs(L - np.median(L))), np.median(np.abs(Rr - np.median(Rr))))
            if jump > min_jump and mad < jump / 4 and (best is None or jump > best[0]):
                best = (jump, (ph[i - 1][0] + ph[i][0]) / 2)
        if best is None:
            out.append(s)
            continue
        cut = best[1]
        left_f = [f for f in s.fine if f[0] < cut]
        right_f = [f for f in s.fine if f[0] >= cut]
        centre = lambda a: a.src_raw * k + chunk_s / 2
        la = [a for a in s.anchors if centre(a) < cut]
        ra = [a for a in s.anchors if centre(a) >= cut]
        if not la or not ra:
            out.append(s)
            continue
        a, b = Segment(anchors=la, fine=left_f), Segment(anchors=ra, fine=right_f)
        summarize_fine(a)
        summarize_fine(b)
        stack[:0] = [a, b]
    return out


def drop_ref_nested(segs: list[Segment], k: float, chunk_s: float) -> list[Segment]:
    """Drop segments lying inside a larger segment on the ref timeline: the ref is filled
    once, so a second match there is repeated content in the source (a recap, a preview)."""
    def core(s):
        return s.core(k, chunk_s)
    return [s for s in segs
            if not any(o is not s and len(o.anchors) > len(s.anchors) and o.method != "coarse"
                       and core(o)[0] <= core(s)[0] and core(s)[1] <= core(o)[1] for o in segs)]


def merge_equal(segs: list[Segment], k: float, chunk_s: float,
                tol: float = 0.002) -> list[Segment]:
    """Join segments that are neighbours on the ref timeline and share the same offset
    (one continuous piece that was split around a weak or rejected passage)."""
    segs = sorted(segs, key=lambda s: s.core(k, chunk_s)[0])
    out: list[Segment] = []
    for s in segs:
        if (out and out[-1].method != "coarse" and s.method != "coarse"
                and abs(out[-1].offset - s.offset) < tol):
            m = Segment(anchors=out[-1].anchors + s.anchors, fine=out[-1].fine + s.fine)
            summarize_fine(m)
            out[-1] = m
        else:
            out.append(s)
    return out


def residual_drift(segs: list[Segment]) -> tuple[float, int, float]:
    """Pooled slope of fine offset vs time inside segments (0 when speed is exact)."""
    num = den = 0.0
    n = 0
    span = 0.0
    for s in segs:
        ph = [f for f in s.fine if f[3] == "phat"]
        if len(ph) < 3 or not phase_coherent(s):
            continue
        t = np.array([f[0] for f in ph])
        o = np.array([f[1] for f in ph])
        med = np.median(o)
        keep = np.abs(o - med) < 0.005 + 1e-4 * np.abs(t - np.median(t))
        t, o = t[keep], o[keep]
        if len(t) < 3:
            continue
        num += float(((t - t.mean()) * (o - o.mean())).sum())
        den += float(((t - t.mean()) ** 2).sum())
        n += len(t)
        span += t.max() - t.min()
    return (num / den if den > 0 else 0.0), n, span
