"""Pieces the source plays at another speed than the rest: end credits the broadcaster
sped up, an intro taken from another release. The main alignment leaves them as holes in
the reference; each hole is searched again over a range of relative speeds against the
part of the source that the neighbouring segments leave unused."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .cuts import HOP, _refine_coherent, best_cut, coherence_sim, local_sim
from .features import ANA_SR, features
from .segments import Segment
from .speed import find_anchors, gcc_phat, is_confident, active_mask

MIN_HOLE = 4.0            # shortest ref hole searched (s)
MAX_SPAN = 150.0          # longest ref span searched (s)
SPEED_RANGE = (0.75, 1.34)
COARSE_STEP = 0.012       # relative speed grid: a 3 s chunk drifts < 20 ms (half a pooled frame)
CHUNK, STEP = 3.0, 0.75
MAG_LEVEL = 0.4           # band-magnitude similarity a matched stretched piece reaches
MAIN_BAND = 0.01          # relative speeds this close to 1 are left to the main pass


def _anchors(x, Rh, c):
    """Confident anchors of x played c times faster; src_raw is on x's clock. A resampled
    speed-up also shifts the pitch, so x is resampled, not just read faster."""
    fr = Fraction(c).limit_denominator(200)
    y = signal.resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)
    S = features(y)
    act = active_mask(y, len(S), 1.0)
    out = [a for a in find_anchors(S, Rh, 1.0, act, CHUNK, STEP) if is_confident(a)]
    for a in out:
        a.src_raw /= float(fr)
    return out


def _score(anchors, c):
    """Confident anchors that agree on one line (ref = o + c * src)."""
    if len(anchors) < 2:
        return 0.0, None
    o = np.array([a.ref - a.src_raw * c for a in anchors])
    m = float(np.median(o))
    good = [a for a, v in zip(anchors, o) if abs(v - m) < 0.03]
    return sum(a.ncc for a in good), good


def _resample(x, c):
    fr = Fraction(c).limit_denominator(2000)
    return signal.resample_poly(x, fr.numerator, fr.denominator).astype(np.float32), float(fr)


def _fit(ref16, x, c, off, t0, t1, win=2.0):
    """GCC-PHAT on the source resampled by c: returns (c, offset, windows) refined from the
    drift of the measured offsets, or None when too few windows lock."""
    y, c = _resample(x, c)
    ts, offs = [], []
    t = t0
    while t + win <= t1:
        i0 = int(round(t * ANA_SR))
        j0 = int(round((t - off) * ANA_SR))
        if j0 >= 0 and j0 + int(win * ANA_SR) <= len(y):
            d, conf = gcc_phat(ref16[i0:i0 + int(win * ANA_SR)], y[j0:j0 + int(win * ANA_SR)],
                               ANA_SR, 0.02)
            if conf > 7:
                ts.append(t + win / 2)
                offs.append(off + d)
        t += win / 2
    if len(ts) < 3:
        return None
    ts, offs = np.array(ts), np.array(offs)
    slope, icept = np.polyfit(ts - ts[0], offs, 1)
    # y runs on the ref clock: a drift of slope means its speed is off by (1 - slope)
    c_new = c / (1 - slope)
    return c_new, _reoffset(c, icept, c_new, ts[0]), len(ts)


def _reoffset(c, off, c_new, t):
    """The offset that keeps ref time t on the same source sample when x is resampled by
    c_new instead of c (y time = x time * c, ref = y time + offset)."""
    return float(t - (t - off) * c_new / c)


def find(segs: list[Segment], ref16: np.ndarray, inter16: np.ndarray, ref_dur: float,
         log=None, stretched: bool = False) -> list[dict]:
    """Returns pieces {'ref_start','ref_end','speed' (relative to the main speed),
    'src_start' (s on the speed-corrected source, where ref_start plays), 'offset' (on the
    source resampled by speed)} for the holes that match at another speed. The edge of a
    neighbour segment a piece butts on can move (segs are updated in place).
    stretched: the source is time-stretched (no phase coherence), so a piece may be matched
    on band magnitudes; in a resampled source a real piece is phase coherent."""
    segs = sorted(segs, key=lambda s: s.ref_start)
    src_dur = len(inter16) / ANA_SR
    out = []
    for i in range(len(segs) + 1):
        A = segs[i - 1] if i else None
        B = segs[i] if i < len(segs) else None
        h0, h1 = (A.ref_end if A else 0.0), (B.ref_start if B else ref_dur)
        u0 = A.ref_end - A.offset if A else 0.0
        u1 = B.ref_start - B.offset if B else src_dur
        u0, u1 = max(0.0, u0), min(src_dur, u1)
        # a hole with one open side (before the first or after the last segment) can face
        # minutes of other programme: search next to the neighbour only; the cost grows
        # with both spans
        if h1 - h0 > MAX_SPAN:
            if A is not None and B is not None:
                continue
            if A is not None:
                h1 = h0 + MAX_SPAN
            else:
                h0 = h1 - MAX_SPAN
        cap = (h1 - h0) / SPEED_RANGE[0] + 2.0
        if A is None:
            u0 = max(u0, u1 - cap)
        if B is None:
            u1 = min(u1, u0 + cap)
        # x starts on the 16 kHz grid
        u0 = int(u0 * ANA_SR) / ANA_SR
        if h1 - h0 < MIN_HOLE or u1 - u0 < MIN_HOLE / SPEED_RANGE[1]:
            continue
        if u1 - u0 > 2 * cap:
            # an inner hole facing a long unused span (an ad break): the piece runs on from
            # one of the neighbours, so only the source next to each is searched
            v1 = int((u1 - cap) * ANA_SR) / ANA_SR
            found = [q for q in (_search(ref16, inter16, h0, h1, u0, u0 + cap, stretched),
                                 _search(ref16, inter16, h0, h1, v1, u1, stretched)) if q is not None]
            p = max(found, key=lambda q: q["ref_end"] - q["ref_start"]) if found else None
        else:
            p = _search(ref16, inter16, h0, h1, u0, u1, stretched)
        if p is not None:
            if not p["coherent"]:
                # the main pass ended the neighbour with nothing matched beyond it
                if A is not None and p["ref_start"] - h0 < 0.05:
                    _shared_cut(ref16, inter16, A, p, True)
                if B is not None and h1 - p["ref_end"] < 0.05:
                    _shared_cut(ref16, inter16, B, p, False)
            if log:
                log(f"  {p['ref_start']:.3f}-{p['ref_end']:.3f} s plays x{p['speed']:.5f} "
                    f"faster than the rest ({p['windows']} windows)")
            out.append(p)
    return out


def _search(ref16, inter16, h0, h1, u0, u1, stretched=False):
    sr = ANA_SR
    pad = 0.5
    r0 = max(0.0, h0 - pad)
    Rh = features(ref16[int(r0 * sr):int(min(len(ref16) / sr, h1 + pad) * sr)])
    x = inter16[int(u0 * sr):int(u1 * sr)]
    grid = np.exp(np.arange(np.log(SPEED_RANGE[0]), np.log(SPEED_RANGE[1]), COARSE_STEP))
    best = (0.0, None, None)
    for c in grid:
        if abs(c - 1) < MAIN_BAND:       # the main speed: the main pass had its chance
            continue
        sc, good = _score(_anchors(x, Rh, c), c)
        if sc > best[0]:
            best = (sc, c, good)
    sc, c, good = best
    if c is None or len(good) < 3:
        return None
    for cc in np.linspace(c - COARSE_STEP * c, c + COARSE_STEP * c, 9):
        if abs(cc - 1) < MAIN_BAND:
            continue
        s2, g2 = _score(_anchors(x, Rh, cc), cc)
        if s2 > sc and len(g2) >= 3:
            sc, c, good = s2, cc, g2
    # ref = r0 + a.ref ; y time (x resampled by c) = a.src_raw * c
    off = float(np.median([r0 + a.ref - a.src_raw * c for a in good]))
    lo = min(r0 + a.ref for a in good)
    hi = max(r0 + a.ref for a in good) + CHUNK
    fit = None
    for _ in range(2):
        f = _fit(ref16, x, c, off, lo, hi)
        if f is None:
            break
        c, off, n = f
        fit = n
    tol, den = 2e-5, 1000
    if fit is None and not stretched:
        return None
    if fit is None:
        # no phase coherence (a time-stretched source): the line through the anchors,
        # good to a frame, and a simple ratio if one is that close
        ref_t = np.array([r0 + a.ref for a in good])
        src_t = np.array([a.src_raw for a in good])
        (slope, icept), cov = np.polyfit(src_t, ref_t, 1, cov=True)
        # a simple ratio only when it is within the fit's own uncertainty
        se = float(np.sqrt(max(cov[0, 0], 0.0)))
        c, off, fit, den = float(slope), float(icept), len(good), 30
        tol = min(1e-3, 2 * se) / c
    if abs(c - 1) < MAIN_BAND:                   # converged to the main speed: not ours
        return None
    fr = Fraction(c).limit_denominator(den)
    if abs(float(fr) - c) < tol * c:
        off, c = _reoffset(c, off, float(fr), 0.5 * (lo + hi)), float(fr)
    y, c2 = _resample(x, c)
    off = _reoffset(c, off, c2, 0.5 * (lo + hi))
    c = c2
    # matched span: coherence (10 ms) above a share of its level over the anchors' span
    f0, f1 = int(h0 / HOP), int(h1 / HOP)
    a0, a1 = int((lo + CHUNK / 2) / HOP) - f0, int((hi - CHUNK / 2) / HOP) - f0
    coh = coherence_sim(ref16, y, off, f0, f1)
    lv = float(np.median(coh[a0:a1]))
    coherent = lv >= 0.25
    if not coherent and not stretched:
        return None
    if coherent:
        # unrelated audio stays within ~0.1 of 0; the matched level varies with the content
        on = coh > max(0.15, 0.3 * lv)
        br = 30
    else:
        # a time-stretched source has no phase coherence: band magnitudes instead, which
        # have a small positive chance level (as cuts.place_segments)
        Rw = features(ref16[int(f0 * HOP * sr):int(f1 * HOP * sr)])
        sim = local_sim(Rw, features(y), int(round(off / HOP)) - f0, 0, f1 - f0, win=15)
        lv = float(np.median(sim[a0:a1]))
        if lv < MAG_LEVEL:
            return None
        # noisy per frame and with dips inside the match: judged over 0.5 s, bridged over
        # 1.5 s (frames without source count as 0)
        sim = uniform_filter1d(np.where(sim > -1e2, sim, 0.0), 50, mode="nearest")
        on = sim > float(np.clip(0.45 * lv, 0.05, 0.2))
        br = 150
    s, e = a0, a1
    while s > 0 and (on[s - 1] or on[max(0, s - br):s].mean() > 0.5):
        s -= 1
    while e < len(on) and (on[e] or on[e:e + br].mean() > 0.5):
        e += 1
    while e > a1 and not on[e - 1]:          # back to the last matched frame
        e -= 1
    while s < a0 and not on[s]:
        s += 1
    seg = Segment(anchors=[], offset=off, method="phat" if coherent else "flux",
                  ref_start=max(h0, (f0 + s) * HOP), ref_end=min(h1, (f0 + e) * HOP))
    if coherent:
        _refine_coherent([seg], ref16, y, len(ref16) / sr)
    seg.ref_start, seg.ref_end = max(h0, seg.ref_start), min(h1, seg.ref_end)
    # where the source runs on from the neighbour into this piece, the cut is the neighbour's
    if seg.ref_start - h0 < 0.05 and abs((h0 - off) / c) < 0.05:
        seg.ref_start = h0
    if h1 - seg.ref_end < 0.05 and abs(u0 + (h1 - off) / c - u1) < 0.05:
        seg.ref_end = h1
    if seg.ref_end - seg.ref_start < MIN_HOLE / 2:
        return None
    return {"ref_start": seg.ref_start, "ref_end": seg.ref_end, "speed": c,
            "src_start": u0 + (seg.ref_start - off) / c, "src_offset": u0, "offset": off,
            "windows": fit, "coherent": coherent}


def _shared_cut(ref16, inter16, seg: Segment, p: dict, before: bool, span: float = 2.0):
    """Place the cut between a neighbour segment and a time-stretched piece butting on it
    (before: the segment comes first) on band magnitudes, as cuts.place_segments does
    between two segments."""
    sr = ANA_SR
    t = p["ref_start"] if before else p["ref_end"]
    w0, w1 = max(0.0, t - span), min(len(ref16) / sr, t + span)
    Rw = features(ref16[int(w0 * sr):int(w1 * sr)])
    n = len(Rw)
    # the segment: ref = source + offset
    s0 = int(max(0.0, w0 - seg.offset - 0.5) * sr) / sr
    S = features(inter16[int(s0 * sr):max(int(s0 * sr), int((w1 - seg.offset + 0.5) * sr))])
    s_seg = local_sim(Rw, S, int(round((s0 + seg.offset - w0) / HOP)), 0, n, win=15)
    ok_seg = s_seg > -1e2
    # the piece: ref = offset + (source - src_offset) * speed
    c, off, u = p["speed"], p["offset"], p["src_offset"]
    x0 = int(max(0.0, u + (w0 - off) / c - 0.5) * sr) / sr
    x1 = u + (w1 - off) / c + 0.5
    y, _ = _resample(inter16[int(x0 * sr):max(int(x0 * sr), int(x1 * sr))], c)
    s_p = local_sim(Rw, features(y), int(round((off + (x0 - u) * c - w0) / HOP)), 0, n, win=15)
    ok_p = s_p > -1e2
    if not ok_seg.any() or not ok_p.any():
        return
    lv = [float(np.median(s_seg[ok_seg])), float(np.median(s_p[ok_p]))]
    # frames without source (window past a source end) count as no match
    s_seg, s_p = np.where(ok_seg, s_seg, 0.0), np.where(ok_p, s_p, 0.0)
    tau = float(np.clip(0.45 * min(lv), 0.05, 0.2))
    a, b = (s_seg, s_p) if before else (s_p, s_seg)
    c1, c2 = best_cut(a, b, tau=tau, expected_gap=0)
    cut = w0 + 0.5 * (c1 + c2) * HOP
    if before:
        cut = min(max(cut, seg.ref_start + 0.5), p["ref_end"] - MIN_HOLE / 2)
        seg.ref_end = p["ref_start"] = cut
        p["src_start"] = u + (cut - off) / c
    else:
        cut = max(min(cut, seg.ref_end - 0.5), p["ref_start"] + MIN_HOLE / 2)
        seg.ref_start = p["ref_end"] = cut
