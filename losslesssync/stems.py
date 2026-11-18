"""Pitch-keeping stretch on separated stems: the voice with WSOLA, the rest with a phase vocoder.

A single WSOLA splice cannot fit voice and music at once (their periods differ), and a
phase vocoder makes voices sound robotic. So the source is split into a vocals stem (a
RoFormer model through audio-separator, run in the helper Python, see ext.py) and the
residual (source - vocals, so the two always sum back to the source exactly). The vocals
go through the built-in WSOLA, the residual through Signalsmith Stretch, and they are
summed after aligning their latencies.
"""
from __future__ import annotations

import os
import subprocess

import numpy as np

from . import media, wsola
from .wavio import WavWriter, read_info

BLOCK = 1 << 20


def ext(opt, *args, log=None):
    cmd = [opt.ext_python, "-m", "losslesssync.ext", *args]
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env)
    if r.returncode != 0:
        raise media.MediaError(f"helper failed ({r.returncode}): {' '.join(cmd)}\n"
                               + r.stderr.decode(errors="replace")[-3000:])


def split(raw_wav: str, voc_wav: str, rest_wav: str, tmp: str, opt):
    """vocals stem at the source rate, and the residual source - vocals."""
    info = read_info(raw_wav)
    if info.channels > 2:
        raise media.MediaError("--keep-engine stems needs a mono or stereo source")
    sep_wav = os.path.join(tmp, "sep_vocals.wav")
    ext(opt, "separate", raw_wav, sep_wav, opt.sep_model, opt.sep_model_dir)
    media.run([media.tool("ffmpeg"), "-nostdin", "-v", "error", "-y", "-i", sep_wav,
               "-af", f"aresample={info.sr}:resampler=soxr:precision=28", "-ac", str(info.channels),
               "-c:a", "pcm_f32le", "-rf64", "auto", voc_wav], capture=False)
    if not opt.keep_temp:
        os.remove(sep_wav)
    x = info.memmap()
    v_in = read_info(voc_wav).memmap()
    n = info.frames
    vw = WavWriter(voc_wav + ".tmp", n, info.channels, info.sr, info.mask)
    rw = WavWriter(rest_wav, n, info.channels, info.sr, info.mask)
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        v = np.zeros((b - a, info.channels), np.float32)
        hi = min(b, len(v_in))
        if hi > a:
            v[:hi - a] = v_in[a:hi]
        vw.write(v)
        rw.write(np.asarray(x[a:b], np.float32) - v)
    vw.close()
    rw.close()
    del v_in, x
    os.replace(voc_wav + ".tmp", voc_wav)


def stretch(raw_wav: str, tmp: str, k: float, opt, log) -> tuple[str, str, dict]:
    """Returns the stretched vocals and residual WAVs and the engine report."""
    voc, rest = os.path.join(tmp, "vocals.wav"), os.path.join(tmp, "rest.wav")
    log(f"  separating vocals ({opt.sep_model})...")
    split(raw_wav, voc, rest, tmp, opt)
    voc_k, rest_k = os.path.join(tmp, "vocals_k.wav"), os.path.join(tmp, "rest_k.wav")
    log(f"  vocals: WSOLA +-{opt.wsola_tol_ms:g} ms up to +-{opt.wsola_max_ms:g} ms; "
        "rest: Signalsmith Stretch...")
    eng = wsola.stretch(voc, voc_k, k, opt.wsola_tol_ms, opt.wsola_max_ms)
    ext(opt, "stretch", rest, rest_k, repr(k))
    if not opt.keep_temp:
        os.remove(voc)
        os.remove(rest)
    return voc_k, rest_k, {"engine": "stems", "sep_model": opt.sep_model,
                           "vocals": eng, "rest": {"engine": "signalsmith"}}


def mix(voc_k: str, rest_k: str, out_wav: str, rest_shift: int, n_out: int):
    """out = vocals + residual delayed by rest_shift samples (negative advances it)."""
    vi, ri = read_info(voc_k), read_info(rest_k)
    v, r = vi.memmap(), ri.memmap()
    w = WavWriter(out_wav, n_out, vi.channels, vi.sr, vi.mask)
    for a in range(0, n_out, BLOCK):
        b = min(a + BLOCK, n_out)
        y = np.zeros((b - a, vi.channels), np.float32)
        hi = min(b, len(v))
        if hi > a:
            y[:hi - a] += v[a:hi]
        lo2, hi2 = max(a - rest_shift, 0), min(b - rest_shift, len(r))
        if hi2 > lo2:
            y[lo2 + rest_shift - a:hi2 + rest_shift - a] += r[lo2:hi2]
        w.write(y)
    w.close()
