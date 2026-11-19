"""Cut points between segments: placement, edge refinement, opening."""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import binary_dilation, uniform_filter1d

from .features import ANA_SR, _nw
from .speed import DROPOUT_MAX_JUMP, HOP, NCC_REG
from .segments import FLUX_HOP, Segment, _flux_bands
from .gaps import _energy_db, dropout_gaps

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
        X = sfft.rfft(rp[p[ii, None] + base] * win, axis=1, workers=_nw())[:, band]
        Y = sfft.rfft(sp[qi[ii, None] + base] * win, axis=1, workers=_nw())[:, band] * rot
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
            margin_prior = 4.0
            if A is not None and Bs is not None:
                eg = int(round(max(0.0, Bs.offset - A.offset) * fps))
                # a small forward jump is lost audio (signal dropout): the source runs on
                # across it, so the gap is exactly the jump, wherever it falls
                if 0.0 < Bs.offset - A.offset <= DROPOUT_MAX_JUMP:
                    margin_prior = np.inf
            c1, c2 = best_cut(sA, sB, tau=tau_for(pair, phase), expected_gap=eg,
                              prior_margin=margin_prior)
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
                  drop_db: float = 20.0, floor_db: float = -50.0):
    """Source time of the deepest level dip within +-search of s_t (a TV splice inside
    continuous source), or None when nothing dips drop_db below the window's median."""
    hop = 32
    step = hop / ANA_SR
    e = _energy_db(src16, s_t - search, s_t + search, hop)
    if e is None or len(e) < 3:
        return None
    e = uniform_filter1d(e, max(1, int(round(smooth / step))), mode="nearest")
    i = int(np.argmin(e))
    if e[i] > np.median(e) - drop_db or e[i] > floor_db:
        return None
    return s_t - search + i * step


def _level_feat(x: np.ndarray, n_fft: int = 1024, n_bands: int = 32, hp: int = 50,
                dead_db: float = -70.0):
    """Log band levels at 1 ms hop, high-passed in time (hp frames) and z-scored per band;
    frames near digital silence (muted audio) are zeroed so they carry no evidence."""
    n = (len(x) - n_fft) // FLUX_HOP + 1
    if n < hp * 2:
        return None
    win, W = _flux_bands(n_fft, n_bands)
    frames = np.lib.stride_tricks.sliding_window_view(np.asarray(x, np.float32),
                                                      n_fft)[::FLUX_HOP][:n]
    mag = np.abs(sfft.rfft(frames * win, axis=1, workers=_nw()))
    e = 10 * np.log10((mag ** 2).sum(1) / ((win ** 2).sum() * n_fft) + 1e-20)
    dead = binary_dilation(e < dead_db, np.ones(hp + 1, bool))
    B = np.log(mag @ W + 1e-4).T
    B -= uniform_filter1d(B, hp, axis=1, mode="nearest")
    B[:, dead] = 0
    if dead.all():
        return B
    B /= B[:, ~dead].std(1, keepdims=True) + 1e-9
    return np.clip(B, -3, 3)


def _locate_switch(ref16, src16, A: Segment, B: Segment, g: float, span: float = 1.5,
                   ctx: float = 4.0) -> bool:
    """Place the switch from A to B at 1 ms, with a ref gap of g s between them (a lost-audio
    skip: g = B.offset - A.offset; extra source material such as an ad: g = 0): band levels
    of the ref against the source at A's offset before the cut and at B's offset after it.
    Works without phase coherence (time-stretched sources). Bands are weighted by how well ref and source agree in the context on both sides, which mutes
    the bands where only the dialogue (different language) lives."""
    sr = ANA_SR
    lo = max(A.ref_start + 0.1, A.ref_end - span)
    hi = min(B.ref_end - 0.1, B.ref_start + span)
    if hi - lo < g + 0.2:
        return False
    # with phase coherence the placement cut (coherence similarity) is already exact
    f0, f1 = int(round(lo / HOP)), int(round(hi / HOP))
    h = (f1 - f0) // 2
    coh = np.concatenate([coherence_sim(ref16, src16, A.offset, f0, f0 + h),
                          coherence_sim(ref16, src16, B.offset, f1 - h, f1)])
    if np.median(coh) >= 0.15:
        return False
    L0 = max(A.ref_start, lo - ctx, A.offset, B.offset, 0.0)
    L1 = min(B.ref_end, hi + ctx, len(ref16) / sr, len(src16) / sr + min(A.offset, B.offset))
    if L0 > lo or L1 < hi:
        return False
    n = int((L1 - L0) * sr)
    i_r = int(round(L0 * sr))
    i_a = int(round((L0 - A.offset) * sr))
    i_b = int(round((L0 - B.offset) * sr))
    if min(i_r, i_a, i_b) < 0 or i_r + n > len(ref16) or max(i_a, i_b) + n > len(src16):
        return False
    Fr, Fa, Fb = (_level_feat(v[i:i + n]) for v, i in ((ref16, i_r), (src16, i_a), (src16, i_b)))
    if Fr is None or Fa is None or Fb is None:
        return False
    k0 = int(round((lo - L0) * sr / FLUX_HOP))
    k1 = int(round((hi - L0) * sr / FLUX_HOP))
    w = np.ones(len(Fr), np.float32)
    if k0 > 500 or Fr.shape[1] - k1 > 500:   # >= 0.5 s of context
        agree = (Fr[:, :k0] * Fa[:, :k0]).sum(1) + (Fr[:, k1:] * Fb[:, k1:]).sum(1)
        if agree.max() > 0:
            w = np.maximum(agree, 0)
    w /= w.sum()
    pa = w @ (Fr[:, k0:k1] * Fa[:, k0:k1])
    pb = w @ (Fr[:, k0:k1] * Fb[:, k0:k1])
    PA = np.concatenate([[0.0], np.cumsum(pa)])
    PB = np.concatenate([[0.0], np.cumsum(pb)])
    gf = int(round(g * sr / FLUX_HOP))
    c = np.arange(0, len(pa) - gf + 1)
    if len(c) == 0:
        return False
    score = PA[c] + PB[-1] - PB[c + gf]
    j = int(c[np.argmax(score)])
    t = L0 + ((k0 + j) * FLUX_HOP + 512) / sr   # frame centre (n_fft 1024)
    A.ref_end, B.ref_start = t, t + g
    return True


def refine_edges(segs: list[Segment], src16: np.ndarray, ref_dur: float,
                 ref16: np.ndarray | None = None) -> list[Segment]:
    """Move edges that border a gap onto the splice point visible in the source level.

    place_segments works on 10 ms frames with smoothed similarity; this pins the cut to the
    exact place the TV edit left in the source (a short dip or a step to silence), then
    re-imposes that no source sample is used twice.
    """
    segs = sorted(segs, key=lambda s: s.ref_start)
    drops = dropout_gaps(segs, ref16) if ref16 is not None else set()
    split = {("end", i) for i in drops} | {("start", i + 1) for i in drops}
    for i in sorted(drops):
        _locate_switch(ref16, src16, segs[i], segs[i + 1], segs[i + 1].offset - segs[i].offset)
    for i, (a, b) in enumerate(zip(segs, segs[1:])):
        if i in drops:
            continue
        # Extra source material between two ref-adjacent segments (ad, inserted scene)
        if (ref16 is not None and b.offset < a.offset and b.ref_start - a.ref_end < 0.05
                and _locate_switch(ref16, src16, a, b, 0.0)):
            # the matched evidence can be blind for a few hundred ms; a level step in the
            # source (silence around the inserted material) pins the edit exactly
            t = a.ref_end
            steps = [(_snap(src16, t - b.offset, "start", search=0.5), b.offset),
                     (_snap(src16, t - a.offset, "end", search=0.5), a.offset)]
            steps = [x + o for x, o in steps if x is not None]
            if steps:
                a.ref_end = b.ref_start = min(steps, key=lambda u: abs(u - t))
            split.update((("end", i), ("start", i + 1)))
            continue
        # Source runs on across the cut: the TV edit is somewhere in it, and the ref gap
        # does not say where, so look wider for the splice and cut both edges there.
        s_a, s_b = a.ref_end - a.offset, b.ref_start - b.offset
        if b.offset > a.offset and abs(s_b - s_a) < 0.05:
            t = _splice_point(src16, 0.5 * (s_a + s_b))
            if t is not None:
                a.ref_end, b.ref_start = t + a.offset, t + b.offset
                split.update((("end", i), ("start", i + 1)))
            elif ref16 is not None and _locate_switch(ref16, src16, a, b, b.offset - a.offset):
                # hard splice with no dip (time-stretched source): matched band levels
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


def _spec_similarity(ref16, src16, a: float, b: float, offset: float, win: float = 2.0):
    """Per-window correlation of log spectra between ref [a, b) and src at ref - offset."""
    sr, n, h = ANA_SR, 1024, 512
    hann = np.hanning(n)

    def spec(x, t):
        i = int(round(t * sr))
        seg = x[max(i, 0):i + int(win * sr)]
        if i < 0 or len(seg) < n:
            return None
        fr = np.lib.stride_tricks.sliding_window_view(seg, n)[::h] * hann
        return np.log(np.abs(np.fft.rfft(fr)) + 1e-4)

    out = []
    for t in np.arange(a, b - win, win):
        R, S = spec(ref16, t), spec(src16, t - offset)
        if R is None or S is None:
            continue
        m = min(len(R), len(S))
        R = R[:m].ravel() - R[:m].mean()
        S = S[:m].ravel() - S[:m].mean()
        d = np.linalg.norm(R) * np.linalg.norm(S)
        if d > 0:
            out.append(float(R @ S / d))
    return np.array(out)


def find_opening(segs: list[Segment], ref16, src16, ref_dur: float,
                 min_len: float = 20.0, max_len: float = 120.0, min_sim: float = 0.7,
                 margin: float = 0.1) -> int | None:
    """Index of the segment holding the opening titles, or None.

    The opening is music only, so both languages carry the same audio all through it: its
    worst windows still match closely, unlike dialogue scenes. It follows a gap, lies in
    the first 40% of the episode and lasts 20-120 s (the end credits look alike but are
    late)."""
    p10 = []
    for s in segs:
        sim = _spec_similarity(ref16, src16, s.ref_start, s.ref_end, s.offset)
        p10.append(float(np.percentile(sim, 10)) if len(sim) >= 5 else -1.0)
    best = None
    for i, s in enumerate(segs):
        gap_before = s.ref_start - (segs[i - 1].ref_end if i else 0.0)
        if not (gap_before > 0.5 and s.ref_start < 0.4 * ref_dur
                and min_len <= s.ref_end - s.ref_start <= max_len and p10[i] >= min_sim):
            continue
        others = [v for j, v in enumerate(p10) if j != i and v > -1.0]
        if others and p10[i] < np.median(others) + margin:
            continue
        if best is None or p10[i] > p10[best]:
            best = i
    return best
