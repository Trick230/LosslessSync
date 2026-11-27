"""Ground-truth quality score for the pitch-keeping stretch (roadmap 2.1).

A real film-timing recording O is turned into what a broadcaster airs: sped up by
K = 25/23.976 with the pitch corrected, by several "broadcast" engines. Each tested
engine then slows that back down by K keeping the pitch, which is exactly the keep-pitch
render; the result is compared with O itself, not with a round trip. The summary leaves
out each engine's score on its own broadcast, where the artefacts would cancel.

Scores (lower is better, 0 = identical to the original; each frame is compared with
the best match within +-10 ms, so timing jitter alone is not counted):
  mel   mean |dB| on a 64-band log-mel spectrogram, 5 ms frames: colouring, smeared attacks
  p99   its 99th percentile: the worst frames (doubled attacks, stutters, clicks)
  tonal mean |dB| on a 8192-point spectrum (5.9 Hz bins, 100 Hz-8 kHz): harmonics smeared
        or split, phasiness; the 5 ms mel frames cannot resolve them
  rough added amplitude modulation at 30-150 Hz per mel band (dB over the original):
        the "rough", buzzing texture a phase vocoder leaves on sustained tones

    python tools/ground_truth.py SOURCE [--engines wsola,rubberband,...] [--broadcast E1,E2,...]
           [--starts 60,330,700,1000] [--len 45] [--vocals VOCALS.wav [--stems stems,stems:psola,...]]

Engines as in tools/stretch_quality.py (wsola[:tol...], rubberband, signalsmith, atempo),
plus rb-tempo (Rubber Band R2, as the 'tempo' mode renders) and, with --vocals (the
separated vocals of SOURCE, e.g. from gpu/separate.py), stems: the vocals' stretcher plus
Signalsmith on the rest, each part forwarded through the broadcast engine on its own.
--stems picks the vocals' stretcher: stems (WSOLA with the pipeline's defaults, +-5 ms up to
+-10 ms), stems:wsola:TOL:MAX, stems:psola:MIN:MAX (whole pitch periods).
Only differences between engines on the same material mean anything.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stretch_quality as sq

from losslesssync.audio import media
from losslesssync.stretch import psola, wsola

K = 25 / (24000 / 1001)
SR = sq.SR
SLACK = sq.SLACK


def write_wav(path: str, x: np.ndarray, sr: int):
    wavfile.write(path, sr, np.ascontiguousarray(x, np.float32))


def ffmpeg(args):
    subprocess.run([media.tool("ffmpeg"), "-nostdin", "-v", "error", "-y", *args], check=True)


def run_engine(name: str, src: str, dst: str, k: float):
    if name == "rb-tempo":
        ffmpeg(["-i", src, "-af", media.speed_filter(k, SR, "tempo"), "-c:a", "pcm_f32le", dst])
    else:
        sq.run_engine(name, src, dst, k)


def stems(name: str, voc: str, rest: str, voc0: str, rest0: str, dst: str, k: float):
    """The keep-pitch 'stems' render on given parts: the vocals' stretcher (see --stems) +
    Signalsmith rest. Each part is aligned to its own original (voc0, rest0), as the pipeline
    compensates each part's measured latency."""
    v, r = dst[:-4] + "_v.wav", dst[:-4] + "_r.wav"
    _, *spec = name.split(":")
    kind = spec[0] if spec else "wsola"
    lo, hi = (float(spec[1]), float(spec[2])) if len(spec) > 2 else \
        ((5.0, 10.0) if kind == "wsola" else (2.0, 10.0))     # the pipeline's defaults
    if kind == "wsola":
        wsola.stretch(voc, v, k, lo, hi)
    elif kind == "psola":
        if not psola.available():
            raise SystemExit("stems:psola needs an lls-dsp with psola")
        psola.stretch(voc, v, k, lo, hi)
    else:
        raise SystemExit(f"unknown stems variant {name}")
    sq.run_engine("signalsmith", rest, r, k)
    _, a, _ = sq.align(sq.load(voc0), sq.load(v))
    _, b, _ = sq.align(sq.load(rest0), sq.load(r))
    n = min(len(a), len(b))               # the two stretchers round the length differently
    write_wav(dst, (a[:n] + b[:n]).astype(np.float32), SR)


# ---------------------------------------------------------------- scores

_FB_T = None


def _slack_min(R, O, slack):
    n = R.shape[1] - 2 * slack
    return np.min([np.abs(R[:, slack:slack + n] - O[:, j:j + n]).mean(0)
                   for j in range(2 * slack + 1)], axis=0)


def tonal(ref, out, n_fft=8192, hop=1024):
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    band = (f >= 100) & (f <= 8000)
    _, _, Z = stft(ref, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    _, _, W = stft(out, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
    R = 20 * np.log10(np.abs(Z[band]) + 1e-9)
    O = 20 * np.log10(np.abs(W[band]) + 1e-9)
    floor = R.max() - 80
    R, O = np.maximum(R, floor), np.maximum(O, floor)
    e = _slack_min(R, O, 1)
    loud = R[:, 1:1 + len(e)].mean(0) > floor + 25
    return e[loud]


def rough(ref, out, n_fft=1024, hop=48, block=512, lo=30.0, hi=150.0):
    """Modulation energy at lo-hi Hz in each mel band, out minus ref (dB, >= 0)."""
    fb = sq.mel_fb(n_fft, SR, 32, 100.0, 8000.0)
    env = []
    for x in (ref, out):
        _, _, Z = stft(x, SR, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False)
        env.append(np.log(fb @ (np.abs(Z) ** 2) + 1e-10))
    n = min(env[0].shape[1], env[1].shape[1])
    fm = np.fft.rfftfreq(block, hop / SR)
    sel = (fm >= lo) & (fm <= hi)
    win = np.hanning(block)
    out_d = []
    for s in range(0, n - block, block // 2):
        er, eo = env[0][:, s:s + block], env[1][:, s:s + block]
        if er.mean() < env[0].max() - 12:          # quiet block: nothing to judge
            continue
        mr = (np.abs(np.fft.rfft((er - er.mean(1, keepdims=True)) * win, axis=1))[:, sel] ** 2).sum(1)
        mo = (np.abs(np.fft.rfft((eo - eo.mean(1, keepdims=True)) * win, axis=1))[:, sel] ** 2).sum(1)
        out_d.append(np.maximum(0.0, 10 * np.log10((mo + 1e-9) / (mr + 1e-9))).mean())
    return np.array(out_d)


def score(ref, out):
    ref, out, _ = sq.align(ref, out)
    mel, ton, rgh = [], [], []
    for c in range(ref.shape[1]):
        R, O = sq.logmel(ref[:, c]), sq.logmel(out[:, c])
        floor = R.max() - 80
        R, O = np.maximum(R, floor), np.maximum(O, floor)
        e = _slack_min(R, O, SLACK)
        mel.append(e[R[:, SLACK:SLACK + len(e)].mean(0) > floor + 25])
        ton.append(tonal(ref[:, c], out[:, c]))
        rgh.append(rough(ref[:, c], out[:, c]))
    mel_a = np.concatenate(mel)
    return {"mel": float(mel_a.mean()), "p99": float(np.percentile(mel_a, 99)),
            "tonal": float(np.concatenate(ton).mean()), "rough": float(np.concatenate(rgh).mean())}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source")
    ap.add_argument("--engines", default="wsola:2,rubberband,rb-tempo,signalsmith,atempo")
    ap.add_argument("--broadcast", default="atempo,rb-tempo,signalsmith",
                    help="engines that make the broadcast (pitch-corrected speed-up); "
                         "the summary leaves out each engine's score on its own broadcast")
    ap.add_argument("--starts", default=",".join(str(s) for s in sq.STARTS))
    ap.add_argument("--len", type=float, default=sq.LEN)
    ap.add_argument("--vocals", help="separated vocals of SOURCE: adds the 'stems' engines")
    ap.add_argument("--stems", default="stems", help="the stems variants (with --vocals)")
    a = ap.parse_args()
    engines = [e for e in a.engines.split(",") if e] + \
        ([e for e in a.stems.split(",") if e] if a.vocals else [])
    casts = [b for b in a.broadcast.split(",") if b]
    tmp = tempfile.mkdtemp(prefix="gt_")
    origs = []
    for i, t in enumerate(float(s) for s in a.starts.split(",")):
        o = os.path.join(tmp, f"o{i}.wav")
        ffmpeg(["-ss", str(t), "-t", str(a.len), "-i", a.source, "-ac", "2", "-ar", str(SR),
                "-c:a", "pcm_f32le", o])
        v = r = None
        if a.vocals:
            v, r = os.path.join(tmp, f"v{i}.wav"), os.path.join(tmp, f"r{i}.wav")
            ffmpeg(["-ss", str(t), "-t", str(a.len), "-i", a.vocals, "-ac", "2", "-ar", str(SR),
                    "-c:a", "pcm_f32le", v])
            write_wav(r, sq.load(o) - sq.load(v), SR)
        origs.append((o, v, r))
    keys = ("mel", "p99", "tonal", "rough")
    table: dict = {}
    for cast in casts:
        print(f"broadcast {cast}, {len(origs)} x {a.len:g} s")
        print(f"  {'engine':<14}" + "".join(f"{k:>7}" for k in keys))
        tvs = []
        for i, (o, v, r) in enumerate(origs):
            tv = os.path.join(tmp, f"tv{i}.wav")
            run_engine(cast, o, tv, 1 / K)
            parts = None
            if v:
                parts = (v[:-4] + "_tv.wav", r[:-4] + "_tv.wav")
                run_engine(cast, v, parts[0], 1 / K)
                run_engine(cast, r, parts[1], 1 / K)
            tvs.append((o, tv, parts and (*parts, v, r)))
        for eng in engines:
            sc = []
            for i, (o, tv, parts) in enumerate(tvs):
                out = os.path.join(tmp, f"out{i}.wav")
                if eng.split(":")[0] == "stems":
                    stems(eng, *parts, out, K)
                else:
                    run_engine(eng, tv, out, K)
                sc.append(score(sq.load(o), sq.load(out)))
                os.remove(out)
            m = {k: float(np.mean([s[k] for s in sc])) for k in keys}
            table[cast, eng] = m
            own = " (own broadcast)" if eng.split(":")[0] == cast else ""
            print(f"  {eng:<14}" + "".join(f"{m[k]:7.2f}" for k in keys) + own, flush=True)
    print("summary (mean over the other engines' broadcasts)")
    print(f"  {'engine':<14}" + "".join(f"{k:>7}" for k in keys))
    for eng in engines:
        rows = [table[c, eng] for c in casts if eng.split(":")[0] != c]
        if rows:
            print(f"  {eng:<14}" + "".join(f"{np.mean([r[k] for r in rows]):7.2f}" for k in keys))


if __name__ == "__main__":
    main()
