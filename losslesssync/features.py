"""Language-independent spectral features for audio-to-audio alignment.

Dialogue differs between dubs but music & effects are shared, so we use log-mel band
energies, high-passed in time (removes EQ/gain/mix-level differences) and z-scored per
band. Correlation of these sequences is dominated by the common M&E content.
"""
from __future__ import annotations

import os
import threading

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d

ANA_SR = 16000

_TLS = threading.local()


def n_threads() -> int:
    """Cores this process may use; LOSSLESSSYNC_THREADS caps it (batch runs in parallel)."""
    try:
        return max(1, int(os.environ.get("LOSSLESSSYNC_THREADS", "")))
    except ValueError:
        return os.cpu_count() or 4


def _nw() -> int:
    """FFT worker count: all allowed cores, or one inside a pool task (already parallel)."""
    return 1 if getattr(_TLS, "in_pool", False) else n_threads()


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    hz = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - lo) / (c - lo), (hi - bins) / (hi - c)))
        if fb[i].sum() == 0:
            fb[i, int(np.argmin(np.abs(bins - c)))] = 1.0
    return fb


def mel_energies(x: np.ndarray, positions: np.ndarray, n_fft: int, fb: np.ndarray,
                 block: int = 4096) -> np.ndarray:
    pad = n_fft
    xp = np.concatenate([np.zeros(pad, np.float32), np.asarray(x, np.float32),
                         np.zeros(pad, np.float32)])
    win = np.hanning(n_fft).astype(np.float32)
    base = np.arange(n_fft) - n_fft // 2 + pad
    out = np.empty((len(positions), fb.shape[0]), np.float32)
    fbt = fb.T.copy()
    for s in range(0, len(positions), block):
        idx = positions[s:s + block, None] + base[None, :]
        np.clip(idx, 0, len(xp) - 1, out=idx)
        spec = sfft.rfft(xp[idx] * win, axis=1, workers=_nw())
        pw = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
        out[s:s + block] = pw @ fbt
    return out


def frame_positions(n_samples: int, hop: float, speed: float = 1.0, start: float = 0.0,
                    count: int | None = None) -> np.ndarray:
    """Sample positions of frames spaced `hop` samples on a timeline stretched by `speed`."""
    if count is None:
        count = int((n_samples - start) * speed / hop)
    return np.round(start + np.arange(max(count, 0)) * hop / speed).astype(np.int64)


def features(x: np.ndarray, sr: int = ANA_SR, hop_s: float = 0.01, speed: float = 1.0,
             n_fft: int = 1024, n_mels: int = 32, fmin: float = 80.0, fmax: float = 7000.0,
             hp_s: float = 1.0, positions: np.ndarray | None = None,
             floor_ref: float | None = None) -> np.ndarray:
    hop = sr * hop_s
    if positions is None:
        positions = frame_positions(len(x), hop, speed)
    fb = mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    e = mel_energies(x, positions, n_fft, fb)
    return postprocess(e, hop_s, hp_s, floor_ref)


def postprocess(e: np.ndarray, hop_s: float, hp_s: float, floor_ref: float | None = None):
    if floor_ref is None:
        floor_ref = float(np.percentile(e, 99.5)) if e.size else 1.0
    floor = max(floor_ref * 1e-6, 1e-12)
    L = np.log(e + floor)
    size = max(3, int(round(hp_s / hop_s)) | 1)
    L -= uniform_filter1d(L, size=size, axis=0, mode="nearest")
    sd = L.std(axis=0) + 1e-6
    L /= sd
    np.clip(L, -5.0, 5.0, out=L)
    return L.astype(np.float32)


def pool(F: np.ndarray, factor: int) -> np.ndarray:
    n = len(F) // factor
    return F[:n * factor].reshape(n, factor, -1).mean(axis=1)


def frame_rms_db(x: np.ndarray, sr: int, hop_s: float) -> np.ndarray:
    hop = int(round(sr * hop_s))
    n = len(x) // hop
    if n == 0:
        return np.zeros(0)
    p = (x[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(axis=1)
    return 10 * np.log10(p + 1e-20)
