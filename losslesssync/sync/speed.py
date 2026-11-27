"""Speed detection, anchors, correlation helpers and pitch-mode detection."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d

from .features import ANA_SR, _nw, features, frame_rms_db, pmap, pool

HOP = 0.01          # coarse feature hop (100 fps)
POOL = 4            # global search runs at 25 fps
NCC_REG = 0.05      # per-element regulariser: penalises near-silent windows
DROPOUT_MAX_JUMP = 1.5  # longest lost-audio stretch tracked as a signal dropout (s)

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
        self.Rf = sfft.rfft(R.astype(np.float32), self.n, axis=0, workers=_nw())
        c = np.concatenate([[0.0], np.cumsum((R.astype(np.float64) ** 2).sum(axis=1))])
        self.wnorm = np.sqrt(np.maximum(c[T:] - c[:-T], 0) + NCC_REG * T * self.B)

    def ncc(self, C: np.ndarray) -> np.ndarray:
        Cf = sfft.rfft(C.astype(np.float32), self.n, axis=0, workers=_nw())
        num = sfft.irfft((self.Rf * np.conj(Cf)).sum(axis=1), self.n, workers=_nw())
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


@lru_cache(maxsize=16)
def _hann32(n: int) -> np.ndarray:
    """Read-only float32 Hann window: gcc_phat runs thousands of times on equal lengths."""
    w = np.hanning(n).astype(np.float32)
    w.flags.writeable = False
    return w


@lru_cache(maxsize=16)
def _out_of_band(n: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Read-only mask of the rfft bins of an n-point FFT outside [fmin, fmax]."""
    f = np.fft.rfftfreq(n, 1.0 / sr)
    m = (f < fmin) | (f > fmax)
    m.flags.writeable = False
    return m


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, max_lag: float,
             fmin: float = 150.0, fmax: float = 7000.0) -> tuple[float, float]:
    """Delay d (s) such that a(t) ~ b(t - d), and a peak-to-noise confidence."""
    n = sfft.next_fast_len(len(a) + len(b), real=True)
    A = sfft.rfft(a * _hann32(len(a)), n, workers=_nw())
    Bf = sfft.rfft(b * _hann32(len(b)), n, workers=_nw())
    G = A * np.conj(Bf)
    mag = np.abs(G)
    G /= mag + 1e-9 * (mag.max() + 1e-30)
    G[_out_of_band(n, sr, fmin, fmax)] = 0
    cc = sfft.irfft(G, n, workers=_nw())
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
    done = [0]

    def one(i):
        done[0] += 1
        if progress and done[0] % 50 == 0:
            progress(done[0] / max(1, len(starts)))
        if active[i:i + T].mean() < 0.3:
            return None
        c4 = S4[i // POOL:i // POOL + T4]
        if len(c4) < T4:
            return None
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
        return Anchor(src_raw=i * HOP / k, ref=best[0] * HOP, ncc=best[1],
                      ratio=max(ratio, 0.0))

    return [a for a in pmap(one, starts) if a is not None]


def is_confident(a: Anchor, ncc_thr: float = 0.15, ratio_thr: float = 1.2) -> bool:
    return a.ncc >= ncc_thr and a.ratio >= ratio_thr


# ---------------------------------------------------------------- speed

def detect_speed(src16: np.ndarray, R: np.ndarray, log, n_probe: int = 30) -> tuple[float, dict]:
    scores = {}
    T = int(round(6.0 / HOP)) // POOL * POOL

    def score(k):
        S = features(src16, speed=k)
        act = active_mask(src16, len(S), k)
        hi = max(1, len(S) - T - 1)
        starts = sorted({int(x) // POOL * POOL for x in np.linspace(0, hi, n_probe)})
        an = find_anchors(S, R, k, act, starts=starts)
        vals = sorted((a.ncc for a in an), reverse=True)
        top = vals[:max(3, len(vals) // 2)]
        return float(np.mean(top)) if top else 0.0

    # candidates in parallel (find_anchors runs serially inside a pool task)
    for (name, k), v in zip(SPEED_CANDIDATES.items(), pmap(score, SPEED_CANDIDATES.values())):
        scores[name] = v
        log(f"    speed {name:<18} x{k:.6f}  score {scores[name]:.3f}")
    best = max(scores, key=lambda name: scores[name])
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
