"""Objective artefact score for pitch-keeping time stretchers (no ground truth needed).

Each excerpt is stretched by k and back by 1/k with the same engine; the round trip is
compared with the original on a log-mel spectrogram (5 ms frames), each frame against
the best match within +-10 ms so that timing jitter alone is not counted. A perfect
stretcher scores 0. 'mean' tracks overall colouring, 'p99' the worst frames (doubled attacks,
stutters, clicks). Lower is better; only differences between engines on the same
material mean anything.

    python tools/stretch_quality.py SOURCE [engine ...]
    engines: wsola[:tol[:frame_ms[:key=value...]]]  rubberband  signalsmith  atempo
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.signal import stft

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from losslesssync.stretch import wsola  # noqa: E402
from losslesssync.audio.media import speed_filter  # noqa: E402
from losslesssync.wavio import read_info  # noqa: E402

K = 1001 / 960
SR = 48000
STARTS = (60.0, 330.0, 700.0, 1000.0)
LEN = 45.0


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def run_engine(name: str, src: str, dst: str, k: float):
    kind, *par = name.split(":")
    if kind == "wsola":
        kw = {}
        if par:
            kw["tol_ms"] = float(par[0])
        if len(par) > 1 and par[1]:
            kw["frame_ms"] = float(par[1])
        for opt in par[2:]:
            key, _, val = opt.partition("=")
            kw[key] = float(val) if val else True
        wsola.stretch(src, dst, k, **kw)
    elif kind == "rubberband":
        ffmpeg(["-i", src, "-af", speed_filter(k, SR, "keep"), "-c:a", "pcm_f32le", dst])
    elif kind == "signalsmith":  # needs python-stretch in $LOSSLESSSYNC_EXT_PYTHON
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        subprocess.run([os.environ.get("LOSSLESSSYNC_EXT_PYTHON", sys.executable), "-m",
                        "losslesssync.stretch.ext", "stretch", src, dst, repr(k)], check=True,
                       capture_output=True, env=dict(os.environ, PYTHONPATH=root))
    elif kind == "atempo":
        ffmpeg(["-i", src, "-af", f"atempo={1 / k:.12f}", "-c:a", "pcm_f32le", dst])
    else:
        raise SystemExit(f"unknown engine {name}")


def load(path):
    return np.array(read_info(path).memmap())


def mel_fb(n_fft, sr, n=64, fmin=50.0, fmax=16000.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    pts = imel(np.linspace(mel(fmin), mel(fmax), n + 2))
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n, len(f)))
    for i in range(n):
        lo, c, hi = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    return fb


FB = mel_fb(2048, SR)


HOP = 240   # 5 ms
SLACK = 2   # frames of displacement forgiven (+-10 ms): timing jitter is not an artefact


def logmel(x):
    _, _, Z = stft(x, SR, nperseg=2048, noverlap=2048 - HOP, boundary=None, padded=False)
    return 10 * np.log10(FB @ (np.abs(Z) ** 2) + 1e-10)


def align(ref, out, max_lag=2400):
    a, b = ref.mean(1), out.mean(1)
    n = min(len(a), len(b))
    c = np.fft.irfft(np.fft.rfft(a[:n], 2 * n) * np.conj(np.fft.rfft(b[:n], 2 * n)))
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]])
    lag = int(np.argmax(c)) - max_lag          # out delayed by -lag relative to ref
    if lag < 0:
        out = out[-lag:]
    else:
        out = np.concatenate([np.zeros((lag, out.shape[1]), out.dtype), out])
    n = min(len(ref), len(out))
    return ref[:n], out[:n], lag


def score(ref, out):
    ref, out, lag = align(ref, out)
    d = []
    for c in range(ref.shape[1]):
        R, O = logmel(ref[:, c]), logmel(out[:, c])
        floor = R.max() - 80
        R, O = np.maximum(R, floor), np.maximum(O, floor)
        n = R.shape[1] - 2 * SLACK
        e = np.min([np.abs(R[:, SLACK:SLACK + n] - O[:, j:j + n]).mean(0)
                    for j in range(2 * SLACK + 1)], axis=0)
        loud = R[:, SLACK:SLACK + n].mean(0) > floor + 25
        d.append(e[loud])
    d = np.concatenate(d)
    return float(d.mean()), float(np.percentile(d, 99)), lag * 1000 / SR


def main():
    src = sys.argv[1]
    engines = sys.argv[2:] or ["wsola:2", "wsola:5", "rubberband", "atempo"]
    tmp = tempfile.mkdtemp(prefix="sq_")
    exc = []
    for i, t in enumerate(STARTS):
        p = os.path.join(tmp, f"x{i}.wav")
        ffmpeg(["-ss", str(t), "-t", str(LEN), "-i", src, "-ac", "2", "-ar", str(SR),
                "-c:a", "pcm_f32le", p])
        exc.append(p)
    print(f"{'engine':<16}" + "".join(f"  x{i} mean/p99" for i in range(len(exc)))
          + "   ALL mean/p99")
    for eng in engines:
        cols, ms, ps = [], [], []
        for p in exc:
            a, b = p[:-4] + "_a.wav", p[:-4] + "_b.wav"
            run_engine(eng, p, a, K)
            run_engine(eng, a, b, 1 / K)
            m, q, _ = score(load(p), load(b))
            cols.append(f"  {m:5.2f}/{q:5.2f}")
            ms.append(m)
            ps.append(q)
        print(f"{eng:<16}" + "".join(cols) + f"   {np.mean(ms):5.2f}/{np.mean(ps):5.2f}",
              flush=True)


if __name__ == "__main__":
    main()
