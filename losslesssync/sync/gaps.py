"""Signal dropouts and muted spans."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .features import ANA_SR, pmap
from .segments import Segment, fine_measure, phase_coherent, summarize_fine
from .speed import DROPOUT_MAX_JUMP, HOP, NCC_REG, Anchor, parabolic

# ---------------------------------------------------------------- signal dropouts

def _energy_db(x: np.ndarray, t0: float, t1: float, hop: int = 32, win: int = 64):
    """Short-term level (dB) of x between t0 and t1 (s): one value per `hop` samples."""
    sr = ANA_SR
    i0, i1 = int(round(t0 * sr)), int(round(t1 * sr))
    if i0 < 0 or i1 + win > len(x) or i1 <= i0:
        return None
    n = (i1 - i0) // hop
    idx = i0 + np.arange(n)[:, None] * hop + np.arange(win)
    return 10 * np.log10((x[idx] ** 2).mean(1) + 1e-12)




def track_offsets(ref16, src16, Rb, Sb, base_off: float, t0: float, t1: float,
                  win: float = 1.5, step: float = 0.25, max_jump: float = 1.5,
                  min_ncc: float = 0.25, min_ratio: float = 1.1, coarse: int = 4,
                  edge: int = 8, tol: float = 0.002) -> list[tuple]:
    """Offset measured in short sliding ref windows over [t0, t1), searching +-max_jump
    around base_off: (window centre, offset, confidence, method) where it is reliable."""
    fps = 1.0 / HOP
    T = int(round(win * fps))
    L = int(round(max_jump * fps))
    excl = 5
    starts = []
    t = max(0.0, t0)
    while t + win <= t1:
        starts.append(t)
        t += step

    def one(t):
        f = int(round(t * fps))
        g = f - int(round(base_off * fps))
        if f + T > len(Rb) or g - L < 0 or g + T + L > len(Sb):
            return None
        C = Rb[f:f + T]
        Sseg = Sb[g - L:g + T + L]
        num = fftconvolve(Sseg, C[::-1], mode="valid", axes=0).sum(1)
        e = np.concatenate([[0.0], np.cumsum((Sseg * Sseg).sum(1))])
        es = e[T:] - e[:-T]
        B = C.shape[1]
        ncc = num / (np.sqrt(es + NCC_REG * T * B) * (np.sqrt(float((C * C).sum())) + 1e-9))
        j = int(np.argmax(ncc))
        v2 = ncc.copy()
        v2[max(0, j - excl):j + excl + 1] = -np.inf
        second = float(v2.max()) if np.isfinite(v2).any() else 0.0
        if ncc[j] >= min_ncc and ncc[j] >= min_ratio * max(second, 1e-3):
            off = base_off + (L - (j + parabolic(ncc, j))) / fps
            res = fine_measure(ref16, src16, t, off, win, max_lag=0.03)
            if res is not None:
                return (t + win / 2, res[0], res[1], res[2])
        return None

    # Measure on a coarse grid first and fill in the full grid only between neighbours
    # that disagree (or where a window failed). Dropouts only move the offset forward and
    # edits leave windows unmatched, so two coarse windows at one offset cannot have a
    # change between them. The dense grid is used at both ends of the range and for
    # `edge` windows on both sides of every disagreement, so a short run next to a change
    # still gets the consecutive measurements offset_runs needs to confirm it.
    n = len(starts)
    first = sorted(set(range(0, n, coarse)) | set(range(min(n, edge)))
                   | set(range(max(0, n - edge), n)))
    res = dict(zip(first, pmap(lambda i: one(starts[i]), first)))
    fill: set = set()
    for i, j in zip(first, first[1:]):
        a, b = res[i], res[j]
        if j - i > 1 and (a is None or b is None or abs(a[1] - b[1]) >= tol):
            fill.update(range(max(0, i - edge), min(n, j + edge + 1)))
    redo = sorted(fill - res.keys())
    res.update(zip(redo, pmap(lambda i: one(starts[i]), redo)))
    return [res[i] for i in sorted(res) if res[i] is not None]


def offset_runs(obs: list[tuple], tol: float = 0.004, min_n: int = 4) -> list[list[tuple]]:
    """Split time-ordered offset measurements into runs of one constant offset.

    A new run starts only once min_n consecutive measurements agree on a new value, so a
    single bad window (straddling a dropout, weak passage) cannot start or end a run."""
    runs: list[list[tuple]] = []
    cur: list[tuple] = []
    pend: list[tuple] = []

    def level(r):
        return float(np.median([o[1] for o in r[-8:]]))

    for o in obs:
        if cur and abs(o[1] - level(cur)) < tol:
            cur.append(o)
            pend = []
            continue
        if pend and abs(o[1] - level(pend)) >= tol:
            pend = []
        pend.append(o)
        if len(pend) >= min_n:
            if cur:
                runs.append(cur)
            cur, pend = pend, []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= min_n]
    merged: list[list[tuple]] = []
    for r in runs:
        if merged and abs(level(merged[-1]) - level(r)) < tol:
            merged[-1] = merged[-1] + r
        else:
            merged.append(r)
    return merged


def _monotonic_runs(runs: list[list[tuple]], max_jump: float,
                    tol: float = 0.004) -> list[list[tuple]]:
    """Keep the runs that a recording with dropouts can produce: lost audio only ever moves
    the offset forward (by at most max_jump), so a run that steps back, or one that sits
    between two runs at the same offset, is a false match (repeated music, weak passage).
    The shorter run of each offending pair is dropped; equal neighbours are merged."""
    runs = [list(r) for r in runs]

    def level(r):
        return float(np.median([o[1] for o in r]))

    while True:
        merged: list[list[tuple]] = []
        for r in runs:
            if merged and abs(level(merged[-1]) - level(r)) < tol:
                merged[-1] = merged[-1] + r
            else:
                merged.append(r)
        runs = merged
        bad = next((i for i in range(len(runs) - 1)
                    if not 0.0 < level(runs[i + 1]) - level(runs[i]) <= max_jump), None)
        if bad is None:
            return runs
        if bad > 0 and abs(level(runs[bad + 1]) - level(runs[bad - 1])) < tol:
            del runs[bad]   # a detour between two runs at one offset
        else:
            del runs[bad if len(runs[bad]) < len(runs[bad + 1]) else bad + 1]


def split_on_dropouts(segs: list[Segment], ref16, src16, Rb, Sb, k: float, chunk_s: float,
                      ref_dur: float, max_jump: float = DROPOUT_MAX_JUMP, progress=None) -> list[Segment]:
    """Re-segment on a fine time grid so that signal dropouts in the recording (lost audio
    shifts everything after it a little) become their own cuts, even several in a row.

    Each segment is tracked from the end of its predecessor to the start of its successor,
    so short pieces between dropouts that the coarse 6 s anchors could not confirm are
    recovered."""
    segs = sorted(segs, key=lambda s: s.core(k, chunk_s)[0])
    found: list[Segment] = []
    for i, s in enumerate(segs):
        if progress:
            progress(i / max(1, len(segs)))
        lo = segs[i - 1].core(k, chunk_s)[1] if i > 0 else 0.0
        hi = segs[i + 1].core(k, chunk_s)[0] if i + 1 < len(segs) else ref_dur
        obs = track_offsets(ref16, src16, Rb, Sb, s.offset, lo, hi, max_jump=max_jump)
        runs = offset_runs(obs)
        if phase_coherent(s):
            # lost audio does not change what the source is: in a phase-coherent recording
            # a run matched only by levels is unrelated audio (another programme after
            # the episode), not the episode after a dropout
            runs = [r for r in runs if sum(o[3] == "phat" for o in r) >= max(2, 0.5 * len(r))]
        runs = _monotonic_runs(runs, max_jump)
        if not runs:
            found.append(s)
            continue
        for r in runs:
            anchors = [Anchor(src_raw=(c - chunk_s / 2 - o) / k, ref=c - chunk_s / 2,
                              ncc=1.0, ratio=10.0) for c, o, _, _ in r]
            n = Segment(anchors=anchors, fine=[(c - o, o, cf, m) for c, o, cf, m in r])
            summarize_fine(n)
            found.append(n)
    # the same stretch can be found from two neighbouring searches: keep the better one
    found.sort(key=lambda s: -len(s.fine))
    keep: list[Segment] = []
    for s in found:
        a0, a1 = s.core(k, chunk_s)
        if not any(min(a1, b.core(k, chunk_s)[1]) - max(a0, b.core(k, chunk_s)[0]) > 0.5
                   for b in keep):
            keep.append(s)
    return sorted(keep, key=lambda s: s.core(k, chunk_s)[0])


def dropout_gaps(segs: list[Segment], ref16, max_len: float = DROPOUT_MAX_JUMP + 0.05,
                 active_db: float = -45.0) -> set[int]:
    """Indices i whose gap to segs[i + 1] is a signal dropout rather than an edit: the
    source runs on across it (the lost audio just is not there) and the reference has
    programme in it (edits on the Blu-ray side fall in black/silence)."""
    out = set()
    for i, (a, b) in enumerate(zip(segs, segs[1:])):
        gap = b.ref_start - a.ref_end
        s_a, s_b = a.ref_end - a.offset, b.ref_start - b.offset
        if not (0.0 < gap <= max_len and abs(s_b - s_a) < 0.05):
            continue
        m = 0.5 * (a.ref_end + b.ref_start)
        h = max(0.5 * gap, 0.01)
        e = _energy_db(ref16, m - h, m + h, hop=32, win=160)
        if e is not None and len(e) and float(np.median(e)) > active_db:
            out.add(i)
    return out


def muted_spans(segs: list[Segment], ref16, src16, min_len: float = 0.02,
                silent_db: float = -70.0, active_db: float = -45.0,
                edge: float = 0.25) -> list[tuple]:
    """Ref spans where the recording kept its timing but the sound dropped to digital
    silence while the reference has programme (muted dropouts)."""
    hop = 32
    step = hop / ANA_SR
    out = []
    for s in segs:
        e = _energy_db(src16, s.ref_start - s.offset, s.ref_end - s.offset, hop, hop)
        if e is None:
            continue
        q = np.concatenate([[False], e < silent_db, [False]])
        d = np.diff(q.astype(np.int8))
        for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            if (b - a) * step < min_len:
                continue
            r0 = s.ref_start + a * step
            r1 = s.ref_start + b * step
            if r0 - s.ref_start < edge or s.ref_end - r1 < edge:
                continue   # silence at a cut belongs to the edit, not to a dropout
            er = _energy_db(ref16, r0, r1, hop=160, win=160)
            if er is not None and len(er) and float(np.median(er)) > active_db:
                out.append((r0, r1))
    return out
