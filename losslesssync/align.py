"""Alignment engine: speed detection, anchors, segmentation, fine offsets, cut points."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d

from .features import ANA_SR, features, frame_rms_db, pool

HOP = 0.01          # coarse feature hop (100 fps)
POOL = 4            # global search runs at 25 fps
NCC_REG = 0.05      # per-element regulariser: penalises near-silent windows

SPEED_CANDIDATES = {
    "1": 1.0,
    "PAL 25 <- 23.976": 1001 / 960,
    "PAL 25 <- 24": 25 / 24,
    "23.976 <- 25": 960 / 1001,
    "24 <- 25": 24 / 25,
    "24 <- 23.976": 1001 / 1000,
    "23.976 <- 24": 1000 / 1001,
}


# ---------------------------------------------------------------- correlation helpers

def parabolic(y: np.ndarray, m: int) -> float:
    if 0 < m < len(y) - 1:
        d = y[m - 1] - 2 * y[m] + y[m + 1]
        if d < 0:
            return float(np.clip(0.5 * (y[m - 1] - y[m + 1]) / d, -0.5, 0.5))
    return 0.0


class GlobalMatcher:
    """Normalised cross-correlation of a fixed-length feature chunk against the whole ref."""

    def __init__(self, R: np.ndarray, T: int):
        self.N, self.B = R.shape
        self.T = T
        self.n = sfft.next_fast_len(self.N + T, real=True)
        self.Rf = sfft.rfft(R.astype(np.float32), self.n, axis=0, workers=-1)
        c = np.concatenate([[0.0], np.cumsum((R.astype(np.float64) ** 2).sum(axis=1))])
        self.wnorm = np.sqrt(np.maximum(c[T:] - c[:-T], 0) + NCC_REG * T * self.B)

    def ncc(self, C: np.ndarray) -> np.ndarray:
        Cf = sfft.rfft(C.astype(np.float32), self.n, axis=0, workers=-1)
        num = sfft.irfft((self.Rf * np.conj(Cf)).sum(axis=1), self.n, workers=-1)
        num = num[: self.N - self.T + 1]
        cn = np.sqrt((C.astype(np.float64) ** 2).sum()) + 1e-9
        return num / (self.wnorm * cn)


def local_ncc(C: np.ndarray, R: np.ndarray, lags: np.ndarray) -> np.ndarray:
    T, B = C.shape
    cn = np.sqrt(float((C * C).sum())) + 1e-9
    out = np.full(len(lags), -1.0)
    for i, l in enumerate(lags):
        if 0 <= l <= len(R) - T:
            W = R[l:l + T]
            out[i] = float((W * C).sum()) / (np.sqrt(float((W * W).sum()) + NCC_REG * T * B) * cn)
    return out


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_lag: float,
             fmin: float = 150.0, fmax: float = 7000.0) -> tuple[float, float]:
    """Delay d (s) such that a(t) ~ b(t - d), and a peak-to-noise confidence."""
    n = sfft.next_fast_len(len(a) + len(b), real=True)
    wa = np.hanning(len(a)).astype(np.float32)
    wb = np.hanning(len(b)).astype(np.float32)
    A = sfft.rfft(a * wa, n, workers=-1)
    Bf = sfft.rfft(b * wb, n, workers=-1)
    G = A * np.conj(Bf)
    mag = np.abs(G)
    G /= mag + 1e-9 * (mag.max() + 1e-30)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    G[(f < fmin) | (f > fmax)] = 0
    cc = sfft.irfft(G, n, workers=-1)
    L = int(max_lag * sr)
    cc = np.concatenate([cc[-L:], cc[:L + 1]])
    m = int(np.argmax(cc))
    lag = (m - L + parabolic(cc, m)) / sr
    excl = max(2, int(0.002 * sr))
    mask = np.ones(len(cc), bool)
    mask[max(0, m - excl):m + excl + 1] = False
    noise = np.sqrt(np.mean(cc[mask] ** 2)) + 1e-12
    return lag, float(cc[m] / noise)


# ---------------------------------------------------------------- anchors

@dataclass
class Anchor:
    src_raw: float   # chunk start, raw source seconds
    ref: float       # matched ref time of chunk start (s)
    ncc: float
    ratio: float     # best / second-best coarse peak

    def off(self, k: float) -> float:
        return self.ref - self.src_raw * k


def active_mask(x16: np.ndarray, n_frames: int, k: float, thr_db: float = -60.0) -> np.ndarray:
    rms = frame_rms_db(x16, ANA_SR, HOP)
    if len(rms) == 0:
        return np.zeros(n_frames, bool)
    idx = np.minimum((np.arange(n_frames) / k).astype(np.int64), len(rms) - 1)
    return rms[idx] > thr_db


def find_anchors(S: np.ndarray, R: np.ndarray, k: float, active: np.ndarray,
                 chunk_s: float = 6.0, step_s: float = 1.5, starts=None,
                 progress=None) -> list[Anchor]:
    T = int(round(chunk_s / HOP)) // POOL * POOL
    T4 = T // POOL
    step = max(POOL, int(round(step_s / HOP / POOL)) * POOL)
    if len(R) <= T + POOL or len(S) <= T:
        return []
    R4, S4 = pool(R, POOL), pool(S, POOL)
    gm = GlobalMatcher(R4, T4)
    excl = int(round(1.0 / (HOP * POOL)))
    if starts is None:
        starts = list(range(0, len(S) - T, step))
    anchors = []
    for n, i in enumerate(starts):
        if progress and n % 50 == 0:
            progress(n / max(1, len(starts)))
        if active[i:i + T].mean() < 0.3:
            continue
        c4 = S4[i // POOL:i // POOL + T4]
        if len(c4) < T4:
            continue
        v = gm.ncc(c4)
        b = int(np.argmax(v))
        v2 = v.copy()
        cands = [b]
        second = None
        for _ in range(2):
            lo = max(0, cands[-1] - excl)
            v2[lo:cands[-1] + excl + 1] = -np.inf
            c = int(np.argmax(v2))
            if not np.isfinite(v2[c]):
                break
            if second is None:
                second = float(v2[c])
            cands.append(c)
        ratio = float(v[b]) / max(second if second is not None else 0.0, 1e-3)
        C = S[i:i + T]
        best = None
        for c in cands:
            lags = np.arange(c * POOL - 6, c * POOL + 7)
            vals = local_ncc(C, R, lags)
            m = int(np.argmax(vals))
            if best is None or vals[m] > best[1]:
                best = (lags[m] + parabolic(vals, m), float(vals[m]))
        anchors.append(Anchor(src_raw=i * HOP / k, ref=best[0] * HOP, ncc=best[1],
                              ratio=max(ratio, 0.0)))
    return anchors


def is_confident(a: Anchor, ncc_thr: float = 0.15, ratio_thr: float = 1.2) -> bool:
    return a.ncc >= ncc_thr and a.ratio >= ratio_thr


# ---------------------------------------------------------------- speed

def detect_speed(src16: np.ndarray, R: np.ndarray, log, n_probe: int = 30) -> tuple[float, dict]:
    scores = {}
    T = int(round(6.0 / HOP)) // POOL * POOL
    for name, k in SPEED_CANDIDATES.items():
        S = features(src16, speed=k)
        act = active_mask(src16, len(S), k)
        hi = max(1, len(S) - T - 1)
        starts = sorted({int(x) // POOL * POOL for x in np.linspace(0, hi, n_probe)})
        an = find_anchors(S, R, k, act, starts=starts)
        vals = sorted((a.ncc for a in an), reverse=True)
        top = vals[:max(3, len(vals) // 2)]
        scores[name] = float(np.mean(top)) if top else 0.0
        log(f"    speed {name:<18} x{k:.6f}  score {scores[name]:.3f}")
    best = max(scores, key=scores.get)
    return SPEED_CANDIDATES[best], scores


def chains(anchors: list[Anchor], k: float, base_tol: float = 0.05,
           drift: float = 2.5e-3) -> list[list[Anchor]]:
    """Group confident anchors into runs of continuous mapping (tolerating small drift)."""
    conf = sorted((a for a in anchors if is_confident(a)), key=lambda a: a.src_raw)
    out: list[list[Anchor]] = []
    for a in conf:
        placed = False
        for ch in reversed(out[-4:]):
            last = ch[-1]
            ds = (a.src_raw - last.src_raw) * k
            if abs(a.off(k) - last.off(k)) < base_tol + drift * ds:
                ch.append(a)
                placed = True
                break
        if not placed:
            out.append([a])
    return out


def refine_speed(anchors: list[Anchor], k: float, snap_tol: float = 6e-5) -> tuple[float, bool]:
    num = den = 0.0
    for ch in chains(anchors, k):
        if len(ch) < 4:
            continue
        s = np.array([a.src_raw * k for a in ch])
        r = np.array([a.ref for a in ch])
        if s[-1] - s[0] < 30:
            continue
        sm, rm = s.mean(), r.mean()
        num += float(((s - sm) * (r - rm)).sum())
        den += float(((s - sm) ** 2).sum())
    if den <= 0:
        return k, False
    k_new = k * num / den
    for kc in SPEED_CANDIDATES.values():
        if abs(k_new / kc - 1) < snap_tol:
            return kc, True
    return k_new, False


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
    Fa = onset_flux(a)
    Fb = onset_flux(src16[b0 - m:b0 + L + m])
    n = Fa.shape[1]
    nl = Fb.shape[1] - n + 1
    if nl < 3:
        return None
    vals = np.array([(Fa * Fb[:, j:j + n]).sum() for j in range(nl)]) / Fa.size
    j = int(np.argmax(vals))
    if vals[j] < 0.12 or j == 0 or j == nl - 1:
        return None
    lag = j + parabolic(vals, j)
    M = (nl - 1) / 2
    return off + (M - lag) * FLUX_HOP / sr, float(vals[j]), "flux"


FLUX_HOP = 16  # samples at 16 kHz = 1 ms


def onset_flux(x: np.ndarray, n_fft: int = 256, n_bands: int = 8) -> np.ndarray:
    """Half-wave rectified log-magnitude flux in log-spaced bands, 1 ms hop, z-scored."""
    sr = ANA_SR
    n = (len(x) - n_fft) // FLUX_HOP + 1
    if n < 2:
        return np.zeros((n_bands, 0), np.float32)
    idx = np.arange(n)[:, None] * FLUX_HOP + np.arange(n_fft)
    win = np.hanning(n_fft).astype(np.float32)
    mag = np.abs(sfft.rfft(x[idx] * win, axis=1, workers=-1))
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(150, 7000, n_bands + 1)
    B = np.stack([np.log(mag[:, (f >= lo) & (f < hi)].mean(1) + 1e-5)
                  for lo, hi in zip(edges[:-1], edges[1:])])
    F = np.maximum(np.diff(B, axis=1), 0)
    F -= F.mean(1, keepdims=True)
    F /= F.std(1, keepdims=True) + 1e-9
    return F.astype(np.float32)


def fine_segments(segs: list[Segment], ref16, src16, k: float, chunk_s: float, progress=None):
    total = sum(len(s.anchors) for s in segs)
    done = 0
    for s in segs:
        s.fine = []
        for a in s.anchors:
            done += 1
            if progress and done % 20 == 0:
                progress(done / max(1, total))
            res = fine_measure(ref16, src16, a.ref, a.off(k), chunk_s)
            if res:
                s.fine.append((a.src_raw * k + chunk_s / 2, res[0], res[1], res[2]))
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


def merge_equal(segs: list[Segment], k: float, chunk_s: float,
                tol: float = 0.002) -> list[Segment]:
    """Join segments that are neighbours on the ref timeline and share the same offset
    (one continuous piece that was split around a weak or rejected passage)."""
    segs = sorted(segs, key=lambda s: s.core(k, chunk_s)[0])
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].method != "coarse" and s.method != "coarse"                 and abs(out[-1].offset - s.offset) < tol:
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


# ---------------------------------------------------------------- cut points

def local_sim(R: np.ndarray, S: np.ndarray, off_frames: int, f0: int, f1: int,
              win: int = 31) -> np.ndarray:
    f = np.arange(f0, f1)
    g = f - off_frames
    valid = (g >= 0) & (g < len(S)) & (f >= 0) & (f < len(R))
    B = R.shape[1]
    Rf = np.zeros((len(f), B), np.float32)
    Sg = np.zeros((len(f), B), np.float32)
    okr = (f >= 0) & (f < len(R))
    Rf[okr] = R[f[okr]]
    Sg[valid] = S[g[valid]]
    dot = uniform_filter1d((Rf * Sg).sum(1), win, mode="nearest")
    er = uniform_filter1d((Rf * Rf).sum(1), win, mode="nearest")
    es = uniform_filter1d((Sg * Sg).sum(1), win, mode="nearest")
    sim = dot / np.sqrt(er * es + (NCC_REG * B) ** 2)
    sim[~valid] = -1e3
    return sim


def coherence_sim(ref16: np.ndarray, src16: np.ndarray, offset: float, f0: int, f1: int,
                  n_fft: int = 1024, fmin: float = 150.0, fmax: float = 6000.0,
                  smooth: int = 5, beta: float = 0.25, block: int = 2048) -> np.ndarray:
    """Per-frame (HOP) phase coherence between ref and source at the exact offset.

    The shared M&E is sample-coherent, so at the right lag coherence is high; unrelated
    audio gives ~0. Bins are weighted by |cross-spectrum|^beta (partial PHAT) so that a
    few coincident musical tones cannot fake a match. Much sharper than magnitude
    features for locating edit points.
    """
    sr = ANA_SR
    f = np.arange(f0, f1)
    p = np.round(f * HOP * sr).astype(np.int64)
    q_exact = p - offset * sr
    qi = np.floor(q_exact).astype(np.int64)
    frac = float(q_exact[0] - qi[0]) if len(f) else 0.0
    h = n_fft // 2
    valid = (p >= 0) & (p < len(ref16)) & (qi >= 0) & (qi < len(src16))
    rp = np.pad(ref16, h)
    sp = np.pad(src16, h)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    rot = np.exp(2j * np.pi * freqs[band] * frac / sr)
    win = np.hanning(n_fft).astype(np.float32)
    base = np.arange(n_fft)          # padded index: sample p - h + j  ->  p + j
    out = np.full(len(f), -1e3)
    vi = np.nonzero(valid)[0]
    for s in range(0, len(vi), block):
        ii = vi[s:s + block]
        X = sfft.rfft(rp[p[ii, None] + base] * win, axis=1, workers=-1)[:, band]
        Y = sfft.rfft(sp[qi[ii, None] + base] * win, axis=1, workers=-1)[:, band] * rot
        G = X * np.conj(Y)
        m = np.abs(G) + 1e-20
        w = m ** beta
        out[ii] = (w * (G.real / m)).sum(axis=1) / w.sum(axis=1)
    if smooth > 1 and valid.any():
        sm = uniform_filter1d(np.where(valid, out, 0.0), smooth, mode="nearest")
        out = np.where(valid, sm, -1e3)
    return out


def best_cut(sA, sB, tau: float = 0.08, gap_pen: float = 3.0, expected_gap: int | None = None,
             prior_margin: float = 4.0):
    """Ordered states A -> gap -> B over n frames. Returns (endA, startB) frame indices.

    expected_gap: gap length implied by a source that is continuous at the edit
    (max(0, offB - offA)). A linear recording never plays the same source samples twice,
    so the gap is at least that long; exactly that length is preferred unless a longer
    gap scores clearly better.
    """
    n = len(sA) if sA is not None else len(sB)
    PA = np.concatenate([[0.0], np.cumsum(sA)]) if sA is not None else None
    PB = np.concatenate([[0.0], np.cumsum(sB)]) if sB is not None else None
    PG = tau * np.arange(n + 1)
    if PA is None:   # gap -> B
        val = PG - PB + PB[-1] - gap_pen * (np.arange(n + 1) > 0)
        return 0, int(np.argmax(val))
    if PB is None:   # A -> gap
        val = PA + PG[-1] - PG - gap_pen * (np.arange(n + 1) < n)
        return int(np.argmax(val)), n
    g = max(0, int(expected_gap or 0))
    if g >= n:
        return 0, n

    def score(c1, c2):
        return PA[c1] + PG[c2] - PG[c1] - gap_pen * (c2 > c1) + PB[-1] - PB[c2]

    # free solution with gap >= g (gap of 0 allowed only when g == 0)
    cands = []
    if g == 0:
        c = int(np.argmax(PA + PB[-1] - PB))
        cands.append((c, c))
    mg = max(g, 1)
    if mg <= n:
        pre = PA - PG
        run_arg = np.zeros(n + 1, np.int64)
        a = 0
        for i in range(n + 1):
            if pre[i] >= pre[a]:
                a = i
            run_arg[i] = a
        c2s = np.arange(mg, n + 1)
        vals = pre[run_arg[c2s - mg]] + PG[c2s] - PB[c2s]
        j = int(np.argmax(vals))
        cands.append((int(run_arg[c2s[j] - mg]), int(c2s[j])))
    free = max(cands, key=lambda c: score(*c))
    if expected_gap is None:
        return free
    c1 = np.arange(0, n - g + 1)
    st = PA[c1] + tau * g - gap_pen * (g > 0) + PB[-1] - PB[c1 + g]
    j = int(np.argmax(st))
    if st[j] + prior_margin >= score(*free):
        return int(c1[j]), int(c1[j] + g)
    return free


def place_segments(segs: list[Segment], R: np.ndarray, S: np.ndarray, ref_dur: float,
                   chunk_s: float, k: float, ref16: np.ndarray, src16: np.ndarray,
                   margin: float = 0.75) -> list[Segment]:
    """Decide where each segment starts/ends on the ref timeline.

    R/S: short-high-pass features (fallback when the source is not phase coherent).
    """
    fps = 1.0 / HOP
    segs = sorted(segs, key=lambda s: s.core(k, chunk_s)[0])
    n_ref = len(R)

    def sim(s, f0, f1, phase):
        if phase:
            return coherence_sim(ref16, src16, s.offset, f0, f1)
        return local_sim(R, S, int(round(s.offset * fps)), f0, f1, win=15)

    # typical similarity inside each segment, to scale the gap threshold
    level = {}
    for s in segs:
        lo, hi = s.core(k, chunk_s)
        vals = []
        for phase in (True, False):
            if phase and s.method != "phat":
                continue
            v = []
            for t in np.linspace(lo, max(lo, hi - 1.0), 12):
                x = sim(s, int(t * fps), int((t + 1.0) * fps), phase)
                v.append(x[x > -1e2])
            v = np.concatenate(v) if v else np.zeros(0)
            level[(id(s), phase)] = float(np.median(v)) if len(v) else 0.3

    def tau_for(pair, phase):
        lv = min(level.get((id(s), phase), 0.3) for s in pair)
        if phase:
            return float(np.clip(0.3 * lv, 0.02, 0.1))
        # magnitude similarity is noisy per frame and has a small positive chance level
        return float(np.clip(0.45 * lv, 0.05, 0.2))

    while True:
        bounds = []
        ok = True
        for i in range(len(segs) + 1):
            A = segs[i - 1] if i > 0 else None
            Bs = segs[i] if i < len(segs) else None
            if A is None and Bs is None:
                break
            if A is None:
                f0, f1 = 0, int(Bs.core(k, chunk_s)[0] * fps)
            elif Bs is None:
                f0, f1 = int(A.core(k, chunk_s)[1] * fps), n_ref
            else:
                ea, sb = A.core(k, chunk_s)[1], Bs.core(k, chunk_s)[0]
                f0 = int(max(0.0, min(ea, sb) - margin) * fps)
                f1 = int(min(n_ref * HOP, max(ea, sb) + margin) * fps)
            f0, f1 = max(0, min(f0, n_ref)), max(0, min(f1, n_ref))
            if f1 <= f0:
                bounds.append((f0, f0))
                continue
            pair = [s for s in (A, Bs) if s is not None]
            # short-term phase coherence only helps if the source kept its phase
            # (plain resampling); a time-stretched source falls back to magnitudes
            phase = all(s.method == "phat" and level.get((id(s), True), 0) >= 0.15
                        for s in pair)
            sA = sim(A, f0, f1, phase) if A is not None else None
            sB = sim(Bs, f0, f1, phase) if Bs is not None else None
            eg = None
            if A is not None and Bs is not None:
                eg = int(round(max(0.0, Bs.offset - A.offset) * fps))
            c1, c2 = best_cut(sA, sB, tau=tau_for(pair, phase), expected_gap=eg)
            bounds.append((f0 + c1, f0 + c2))
        for i, s in enumerate(segs):
            s.ref_start = bounds[i][1] * HOP
            s.ref_end = bounds[i + 1][0] * HOP
            if s.ref_end - s.ref_start < 0.5:
                segs.pop(i)
                ok = False
                break
        if ok:
            break
    for s in segs:
        s.ref_end = min(s.ref_end, ref_dur)
        # never read before the start of the source
        s.ref_start = max(s.ref_start, s.offset, 0.0)
    return segs


def _energy_db(x: np.ndarray, t0: float, t1: float, hop: int = 32, win: int = 64):
    """Short-term level (dB) of x between t0 and t1 (s): one value per `hop` samples."""
    sr = ANA_SR
    i0, i1 = int(round(t0 * sr)), int(round(t1 * sr))
    if i0 < 0 or i1 + win > len(x) or i1 <= i0:
        return None
    n = (i1 - i0) // hop
    idx = i0 + np.arange(n)[:, None] * hop + np.arange(win)
    return 10 * np.log10((x[idx] ** 2).mean(1) + 1e-12)


def _snap(src16, s_t: float, kind: str, search: float = 0.08, ctx: float = 0.03,
          drop_db: float = 15.0, floor_db: float = -55.0):
    """Source time of the edit point near s_t: where the level falls (kind='end') or rises
    (kind='start') by drop_db against the surrounding ctx. TV splices show up as such a
    dip or step; returns None when there is none."""
    hop = 32
    step = hop / ANA_SR
    e = _energy_db(src16, s_t - search - ctx, s_t + search + ctx, hop)
    if e is None:
        return None
    c = int(round(ctx / step))
    best = None
    for i in range(c, len(e) - c):
        if kind == "end":
            ref_lvl = np.median(e[i - c:i])
            ok = ref_lvl > floor_db and e[i] <= ref_lvl - drop_db
        else:
            ref_lvl = np.median(e[i:i + c])
            ok = ref_lvl > floor_db and e[i - 1] <= ref_lvl - drop_db
        if ok:
            t = s_t - search - ctx + i * step
            if best is None or abs(t - s_t) < abs(best - s_t):
                best = t
    return best


def _splice_point(src16, s_t: float, search: float = 0.5, smooth: float = 0.01,
                  drop_db: float = 20.0):
    """Source time of the deepest level dip within +-search of s_t (a TV splice inside
    continuous source), or None when nothing dips drop_db below the window's median."""
    hop = 32
    step = hop / ANA_SR
    e = _energy_db(src16, s_t - search, s_t + search, hop)
    if e is None or len(e) < 3:
        return None
    e = uniform_filter1d(e, max(1, int(round(smooth / step))), mode="nearest")
    i = int(np.argmin(e))
    if e[i] > np.median(e) - drop_db:
        return None
    return s_t - search + i * step


def refine_edges(segs: list[Segment], src16: np.ndarray, ref_dur: float) -> list[Segment]:
    """Move edges that border a gap onto the splice point visible in the source level.

    place_segments works on 10 ms frames with smoothed similarity; this pins the cut to the
    exact place the TV edit left in the source (a short dip or a step to silence), then
    re-imposes that no source sample is used twice.
    """
    segs = sorted(segs, key=lambda s: s.ref_start)
    split = set()
    for i, (a, b) in enumerate(zip(segs, segs[1:])):
        # Source runs on across the cut: the TV edit is somewhere in it, and the ref gap
        # does not say where, so look wider for the splice and cut both edges there.
        s_a, s_b = a.ref_end - a.offset, b.ref_start - b.offset
        if b.offset > a.offset and abs(s_b - s_a) < 0.05:
            t = _splice_point(src16, 0.5 * (s_a + s_b))
            if t is not None:
                a.ref_end, b.ref_start = t + a.offset, t + b.offset
                split.update((("end", i), ("start", i + 1)))
    for i, s in enumerate(segs):
        prev = segs[i - 1] if i > 0 else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        if ("end", i) not in split and s.ref_end < ref_dur - 0.01 and (nxt is None or nxt.ref_start - s.ref_end > 0.01):
            t = _snap(src16, s.ref_end - s.offset, "end")
            if t is not None:
                s.ref_end = min(ref_dur, t + s.offset)
        if ("start", i) not in split and s.ref_start > 0.01 and (prev is None or s.ref_start - prev.ref_end > 0.01):
            t = _snap(src16, s.ref_start - s.offset, "start")
            if t is not None:
                s.ref_start = max(0.0, t + s.offset)
    for a, b in zip(segs, segs[1:]):
        b.ref_start = max(b.ref_start, a.ref_end)                    # no overlap on the ref
        if b.offset > a.offset:                                      # no source reuse
            b.ref_start = max(b.ref_start, a.ref_end - a.offset + b.offset)
    return [s for s in segs if s.ref_end - s.ref_start > 0.1]


# ---------------------------------------------------------------- pitch mode

def detect_pitch_mode(ref16: np.ndarray, src16: np.ndarray, anchors: list[Anchor], k: float,
                      chunk_s: float, bpo: int = 96) -> tuple[str, dict]:
    expected = np.log2(k) * bpo
    sr, n_fft = ANA_SR, 4096
    grid = 120.0 * 2 ** (np.arange(int(5.0 * bpo)) / bpo)
    lin = np.fft.rfftfreq(n_fft, 1.0 / sr)
    win = np.hanning(n_fft)
    D = 14
    acc = np.zeros(2 * D + 1)

    def logspec(x, pos):
        if pos < 0 or pos + n_fft > len(x):
            return None
        m = np.abs(np.fft.rfft(x[pos:pos + n_fft] * win))
        L = np.log(np.interp(grid, lin, m) + 1e-6)
        L -= uniform_filter1d(L, bpo // 3, mode="nearest")
        return L

    conf = sorted((a for a in anchors if is_confident(a)), key=lambda a: -a.ncc)[:40]
    used = 0
    for a in conf:
        for dt in np.linspace(0.5, chunk_s - 0.5, 6):
            t = a.ref + dt
            s_raw = (t - a.ref) / k + a.src_raw
            Lr = logspec(ref16, int(t * sr))
            Ls = logspec(src16, int(s_raw * sr))
            if Lr is None or Ls is None:
                continue
            n = len(Lr)
            for j, d in enumerate(range(-D, D + 1)):
                if d >= 0:
                    acc[j] += float(np.dot(Lr[:n - d], Ls[d:]))
                else:
                    acc[j] += float(np.dot(Lr[-d:], Ls[:n + d]))
            used += 1
    if used == 0:
        return "resample", {"frames": 0}
    best = int(np.argmax(acc)) - D
    at0 = acc[D]
    e = int(round(expected))
    at_exp = acc[D + e] if abs(e) <= D else -np.inf
    # no speed change: nothing to choose, but the measured shift is still reported
    mode = "resample" if abs(expected) < 1.5 or at_exp >= at0 else "tempo"
    return mode, {"frames": used, "best_shift_bins": best, "expected_shift_bins": float(expected),
                  "score_shifted": float(at_exp), "score_unshifted": float(at0)}
