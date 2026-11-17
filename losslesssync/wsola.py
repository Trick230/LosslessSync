"""Time stretch keeping pitch with WSOLA and a bounded search tolerance.

Each output frame is read from the input at its nominal position plus a shift of at most
tol_ms, chosen so the frame continues the waveform of the previous one (the overlap that
WSOLA cross-fades). The shift is shared by all channels, so the stereo image holds, and it
is bounded, so no event lands further than tol_ms + (k - 1) * frame / 2 from its exact
place on the stretched timeline (about tol_ms + 1 ms here). Unlike a phase vocoder there
is no spectral smearing, which is what makes voices sound robotic.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .wavio import WavWriter, read_info


def stretch(in_wav: str, out_wav: str, k: float, tol_ms: float = 5.0, frame_ms: float = 42.0,
            progress=None) -> dict:
    """Write in_wav stretched in time by k (k > 1 lengthens) to out_wav (float WAV)."""
    info = read_info(in_wav)
    x = info.memmap()
    sr, ch, n_in = info.sr, info.channels, info.frames
    N = 2 * int(round(frame_ms * 1e-3 * sr / 2))
    H = N // 2
    D = max(0, int(round(tol_ms * 1e-3 * sr)))
    n_out = int(round(n_in * k))
    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)).astype(np.float32)  # sums to 1 at hop N/2
    wout = WavWriter(out_wav, n_out, ch, sr, info.mask)

    def mid(a: int, n: int) -> np.ndarray:
        """Mono input [a, a + n), zero outside the file."""
        out = np.zeros(n, np.float64)
        lo, hi = max(a, 0), min(a + n, n_in)
        if hi > lo:
            out[lo - a:hi - a] = x[lo:hi].mean(axis=1)
        return out

    def frame(a: int) -> np.ndarray:
        out = np.zeros((N, ch), np.float32)
        lo, hi = max(a, 0), min(a + N, n_in)
        if hi > lo:
            out[lo - a:hi - a] = x[lo:hi]
        return out

    acc = np.zeros((N, ch), np.float32)       # output samples [m*H, m*H + N) in progress
    pending: list[np.ndarray] = []
    pend_n = 0
    written = 0
    shifts = []
    prev = None
    m = 0
    n_frames = (n_out + H - 1) // H
    while written < n_out:
        # nominal input position: the frame centre maps exactly onto the stretched timeline
        a = int(round((m * H + H) / k)) - H
        if prev is not None and D > 0:
            tpl = mid(prev + H, H)               # how the previous frame would carry on
            e_t = float(tpl @ tpl)
            if e_t > 1e-10:
                seg = mid(a - D, H + 2 * D)
                c = fftconvolve(seg, tpl[::-1], mode="valid")
                cs = np.concatenate([[0.0], np.cumsum(seg * seg)])
                e = cs[H:] - cs[:-H]
                d = int(np.argmax(c / np.sqrt(e + 1e-9 * H))) - D
                a += d
                shifts.append(d)
        acc += frame(a) * w[:, None]
        prev = a
        n = min(H, n_out - written)
        pending.append(acc[:n].copy())
        pend_n += n
        written += n
        acc[:H] = acc[H:]
        acc[H:] = 0.0
        if pend_n >= 1 << 16 or written >= n_out:
            wout.write(np.concatenate(pending))
            pending, pend_n = [], 0
        m += 1
        if progress and m % 2000 == 0:
            progress(m / n_frames)
    wout.close()
    s = np.abs(np.array(shifts, float)) * 1000 / sr if shifts else np.zeros(1)
    return {"engine": "wsola", "tol_ms": tol_ms, "frame_ms": N * 1000 / sr,
            "shift_median_ms": float(np.median(s)), "shift_max_ms": float(s.max())}
