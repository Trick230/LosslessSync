"""Pitch-keeping stretch on separated stems, each with the stretcher that suits it.

A single splice cannot fit voice and music at once (their periods differ), and a phase
vocoder makes voices sound robotic and smears attacks. So the source is split into a vocals
stem (a RoFormer model through audio-separator, run in the helper Python, see ext.py) and the
residual (source - vocals, so they always sum back to the source exactly). The vocals go
through PSOLA (whole pitch periods, psola.py) or WSOLA; the residual is split again by
lls-dsp into its tonal part, stretched with Signalsmith Stretch, and its attacks (effects,
percussion), stretched with short-frame WSOLA. The stems are summed after aligning their
latencies.
"""
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..audio import media
from ..audio.wavio import WavWriter, read_info
from . import native, psola, wsola

BLOCK = 1 << 20


def _cmd(opt, *args):
    return [opt.ext_python, "-m", "losslesssync.stretch.ext", *args]


def _env():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return dict(os.environ, PYTHONPATH=root)


class Helper:
    """A helper-Python job running in the background (stderr kept in a file; stdin and
    stdout are pipes for a line protocol)."""

    def __init__(self, opt, tmp, name, *args):
        self.cmd = _cmd(opt, *args)
        self.on_gpu_free = None           # called once the separation is done
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
    # lls-dsp stretch options for the rest
    opts = []
    if opt.rest_block_ms > 0:
        opts += ["--block", str(opt.rest_block_ms)]
    if opt.rest_interval_ms > 0:
        opts += ["--interval", str(opt.rest_interval_ms)]
    if opt.hpss:
        opts += ["--perc", os.path.join(tmp, "perc_k.wav")]
    job = Helper(opt, tmp, "separate", "separate", raw, voc, rest_k, opt.sep_model,
                 opt.sep_model_dir, *opts)
    return job, raw, voc, rest_k


# the attacks' WSOLA in lls-dsp stretch --perc (rust/src/main.rs), for the report
PERC_FRAME_MS, PERC_TOL_MS, PERC_MAX_MS = 12.0, 3.0, 5.0


def stretch(split, tmp: str, k: float, opt, log, measure):
    """Waits for the separation, then stretches the vocals (here) while the helper stretches
    the rest (Signalsmith, after splitting off its attacks when opt.hpss). measure(wav,
    step_s) gives a stretch latency. Returns [(wav, latency)] with the vocals first, and the
    engine report (its "order" names the stems)."""
    job, raw, voc, rest_k = split
    # the helper's lls-dsp writes it next to rest_k (opt.hpss); in a batch the split ran
    # in the prefetch's own dir, not in tmp
    perc_k = os.path.join(os.path.dirname(rest_k), "perc_k.wav")
    job.send(repr(k))
    log("  waiting for the vocals separation...")
    job.expect("vocals")
    if job.on_gpu_free:
        job.on_gpu_free()
    voc_k = os.path.join(tmp, "vocals_k.wav")
    use_psola = opt.vocal_engine == "psola" and psola.available()
    rest_eng = "Signalsmith" + (" (tonal) + WSOLA (attacks)" if opt.hpss else "")
    if use_psola:
        log(f"  vocals: PSOLA (within {opt.wsola_max_ms:g} ms); rest: {rest_eng}...")
        eng = psola.stretch(voc, voc_k, k, min(2.0, opt.wsola_max_ms / 2), opt.wsola_max_ms)
        log(f"  vocals: {eng['voiced_pct']:.0f}% voiced, {eng['periods']} periods "
            f"({eng['forced']} forced), timing error median {eng['err_median_ms']:.2f} ms, "
            f"max {eng['err_max_ms']:.2f} ms")
    else:
        log(f"  vocals: WSOLA +-{opt.wsola_tol_ms:g} ms up to +-{opt.wsola_max_ms:g} ms; "
            f"rest: {rest_eng}...")
        eng = wsola.stretch(voc, voc_k, k, opt.wsola_tol_ms, opt.wsola_max_ms)
    lat_v = measure(voc_k, 5.0)           # while the rest is still stretching
    job.wait()
    if not opt.keep_temp:
        os.remove(raw)
        os.remove(voc)
    report = {"engine": "stems", "sep_model": opt.sep_model, "vocals": eng,
              "rest": {"engine": "signalsmith", "block_ms": opt.rest_block_ms or 120.0,
                       "interval_ms": opt.rest_interval_ms or (opt.rest_block_ms or 120.0) / 4}}
    order = ["vocals", "rest"]
    if not opt.hpss:
        perc_k = None
    elif not os.path.exists(perc_k):
        log("  (this lls-dsp cannot split the attacks off; rest stretched whole)")
        perc_k = None
    with ThreadPoolExecutor(1) as ex:
        # these latencies hardly vary: a window every 15 s is plenty
        def measure_p():
            try:
                return measure(perc_k, 15.0)
            except media.MediaError:
                return None
        lat_p = ex.submit(measure_p) if perc_k else None
        lat_r = measure(rest_k, 15.0)
        lat_p = lat_p.result() if lat_p else None
    parts = [(voc_k, lat_v), (rest_k, lat_r)]
    if perc_k:
        report["attacks"] = {"engine": "wsola", "native": True, "frame_ms": PERC_FRAME_MS,
                             "tol_ms": PERC_TOL_MS, "tol_max_ms": PERC_MAX_MS}
        # sparse attacks can lock the measurement onto the tonal part; WSOLA's own latency
        # is a few ms at most, so anything further off is a bad measurement
        if lat_p is None or abs(lat_p - lat_r) > 0.005:
            log("  attacks: latency not measurable, taken as the rest's")
            lat_p = lat_r
        parts.append((perc_k, lat_p))
        order.append("attacks")
    report["order"] = order
    return parts, report


def mix(voc_k: str, others: list[tuple[str, int]], out_wav: str, n_out: int):
    """out = vocals + each other stem delayed by its shift in samples (negative advances it)."""
    vi = read_info(voc_k)
    v = vi.memmap()
    rs = [(read_info(p).memmap(), s) for p, s in others]
    w = WavWriter(out_wav, n_out, vi.channels, vi.sr, vi.mask)
    for a in range(0, n_out, BLOCK):
        b = min(a + BLOCK, n_out)
        y = np.zeros((b - a, vi.channels), np.float32)
        hi = min(b, len(v))
        if hi > a:
            y[:hi - a] += v[a:hi]
        for r, sh in rs:
            lo2, hi2 = max(a - sh, 0), min(b - sh, len(r))
            if hi2 > lo2:
                y[lo2 + sh - a:hi2 + sh - a] += r[lo2:hi2]
        w.write(y)
    w.close()
