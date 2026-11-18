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


def _cmd(opt, *args):
    return [opt.ext_python, "-m", "losslesssync.ext", *args]


def _env():
    return dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Helper:
    """A helper-Python job running in the background (stderr kept in a file; stdin and
    stdout are pipes for a line protocol)."""

    def __init__(self, opt, tmp, name, *args):
        self.cmd = _cmd(opt, *args)
        self.err_path = os.path.join(tmp, name + ".err")
        self.err = open(self.err_path, "wb")
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, env=_env(), text=True)

    def send(self, line: str):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def expect(self, line: str):
        """Waits for the helper to print this line (raises if it exits first)."""
        got = self.p.stdout.readline().strip()
        if got != line:
            self.wait()
            raise media.MediaError(f"helper said {got!r}, expected {line!r}: {' '.join(self.cmd)}")

    def wait(self):
        rc = self.p.wait()
        self.err.close()
        if rc != 0:
            with open(self.err_path, "rb") as f:
                msg = f.read()[-3000:].decode(errors="replace")
            raise media.MediaError(f"helper failed ({rc}): {' '.join(self.cmd)}\n{msg}")

    def kill(self):
        if self.p.poll() is None:
            self.p.kill()
            self.p.wait()
        self.err.close()


def start_split(src_path, src_st, tmp, opt, log) -> tuple[Helper, str, str, str]:
    """Decode the source and start separating it in the background: it runs on the GPU
    while the alignment runs on the CPU; the same job then stretches the rest (stems.stretch
    sends it the ratio). Returns (job, raw, vocals, stretched rest) WAV paths."""
    if src_st.channels > 2:
        raise media.MediaError("--keep-engine stems needs a mono or stereo source")
    raw = os.path.join(tmp, "raw.wav")
    voc, rest_k = os.path.join(tmp, "vocals.wav"), os.path.join(tmp, "rest_k.wav")
    media.decode_to_wav(src_path, src_st, raw, src_st.channels, src_st.sample_rate)
    log(f"separating vocals in the background ({opt.sep_model})...")
    job = Helper(opt, tmp, "separate", "separate", raw, voc, rest_k, opt.sep_model,
                 opt.sep_model_dir)
    return job, raw, voc, rest_k


def stretch(split, tmp: str, k: float, opt, log, measure):
    """Waits for the separation, then stretches the vocals (WSOLA, here) while the helper
    stretches the rest (Signalsmith). measure(wav, step_s) gives a stretch latency.
    Returns the two WAVs, the engine report and the vocals and rest latencies."""
    job, raw, voc, rest_k = split
    job.send(repr(k))                     # the helper reads it once the vocals are out
    log("  waiting for the vocals separation...")
    job.expect("vocals")
    voc_k = os.path.join(tmp, "vocals_k.wav")
    log(f"  vocals: WSOLA +-{opt.wsola_tol_ms:g} ms up to +-{opt.wsola_max_ms:g} ms; "
        "rest: Signalsmith Stretch...")
    eng = wsola.stretch(voc, voc_k, k, opt.wsola_tol_ms, opt.wsola_max_ms)
    lat_v = measure(voc_k, 5.0)           # while the rest is still stretching
    job.wait()
    if not opt.keep_temp:
        os.remove(raw)
    # Signalsmith's latency hardly varies: a window every 15 s is plenty
    lat_r = measure(rest_k, 15.0)
    if not opt.keep_temp:
        os.remove(voc)
    return voc_k, rest_k, {"engine": "stems", "sep_model": opt.sep_model,
                           "vocals": eng, "rest": {"engine": "signalsmith"}}, lat_v, lat_r


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
