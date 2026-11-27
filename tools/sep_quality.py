"""Vocals-separation quality on a real episode, without ground-truth stems (roadmap 2.3).

The reference and a dub synced onto it (an `lls sync` output) carry the same music and
effects under different dialogue. A good vocals model leaves two rests (mix - vocals) that
agree where the dialogue differs, and leaves the mix alone where there is no dialogue.
Each model separates one file: the reference's excerpts followed by the dub's, so one GPU
job per model (through the stems helper: local GPU, or `dc gpu` with LLS_ENV=cloud).

    python tools/sep_quality.py REF OUT --report REPORT.json [--models M1,M2,...]
           [--n 6] [--len 45] [--pitch F] [--keep DIR]

Scores per model, mean |dB| on a 64-band log-mel spectrogram, 20 ms frames, each band floored
at 40 dB under the reference frame's loudest band (lower is better):
  agree   rest of the reference against rest of the dub, on the dialogue frames (the 40%
          where the two mixes differ most): dialogue left in the rests, or damage that
          differs between the languages
  keep    rest against its own mix, on the frames without dialogue (the 20% where the two
          mixes agree best): music and effects taken away with the vocals
  mix     the 'agree' frames scored on the mixes themselves (no separation): the baseline
Only differences between models on the same material mean anything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from losslesssync.app.cli import venv_python  # noqa: E402
from losslesssync.audio import media  # noqa: E402
from losslesssync.audio.wavio import read_info  # noqa: E402
from losslesssync.sync.pipeline import Options  # noqa: E402

SR = 44100
MODELS = "vocals_mel_band_roformer.ckpt,bs_roformer_vocals_resurrection_unwa.ckpt," \
         "melband_roformer_big_beta6x.ckpt,mel_band_roformer_kim_ft2_bleedless_unwa.ckpt"


def ffmpeg(args):
    subprocess.run([media.tool("ffmpeg"), "-nostdin", "-v", "error", "-y", *args], check=True)


def excerpt(path: str, t: float, n: float, dst: str, pitch: float = 1.0) -> np.ndarray:
    af = ["-af", f"rubberband=pitch={pitch!r}:pitchq=quality"] if pitch != 1.0 else []
    ffmpeg(["-ss", f"{t:.3f}", "-t", f"{n:.3f}", "-i", path, "-ac", "2", "-ar", str(SR), *af,
            "-c:a", "pcm_f32le", dst])
    x = wavfile.read(dst)[1].astype(np.float32)
    os.remove(dst)
    return x[:int(n * SR)]


def pick_starts(report: dict, n: int, length: float) -> list[float]:
    """n windows spread over the episode, each inside one synced segment (not near a cut)."""
    segs = [(s["ref_start"] + 2, s["ref_end"] - 2 - length) for s in report["segments"]
            if s["ref_end"] - s["ref_start"] > length + 4]
    if not segs:
        raise SystemExit("no segment long enough")
    total = sum(b - a for a, b in segs)
    out = []
    for i in range(n):
        u = (i + 0.5) / n * total
        for a, b in segs:
            if u <= b - a:
                out.append(a + u)
                break
            u -= b - a
    return out


def logmel(x: np.ndarray, fb: np.ndarray) -> np.ndarray:
    _, _, Z = stft(x.mean(1), SR, nperseg=2048, noverlap=2048 - 882, boundary=None, padded=False)
    return 10 * np.log10(fb @ (np.abs(Z) ** 2) + 1e-10)


def mel_fb(n_fft=2048, n_mels=64, lo=50.0, hi=16000.0):
    m = lambda f: 2595 * np.log10(1 + f / 700)
    im = lambda v: 700 * (10 ** (v / 2595) - 1)
    pts = im(np.linspace(m(lo), m(hi), n_mels + 2))
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mels, len(f)))
    for i in range(n_mels):
        a, c, b = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((f - a) / (c - a), (b - f) / (b - c)), 0, None)
    return fb


def dist(A: np.ndarray, B: np.ndarray, floor: float) -> np.ndarray:
    A, B = np.maximum(A, floor), np.maximum(B, floor)
    d = A - B
    return np.abs(d - np.median(d)).mean(0)        # a level offset between the two is no error


def separate(ext_py: str, model: str, model_dir: str, src: str, voc: str):
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([ext_py, "-m", "losslesssync.stretch.ext", "separate", src, voc, "-",
                        model, model_dir], stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, env=env, check=False)
    if r.returncode != 0 or not os.path.isfile(voc):
        raise RuntimeError(f"{model}: separation failed ({r.returncode}): {r.stderr[-1500:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("out", help="a dub synced onto REF (lls sync output)")
    ap.add_argument("--report", required=True, help="the sync's report (picks the windows)")
    ap.add_argument("--models", default=MODELS)
    ap.add_argument("--n", type=int, default=6, help="windows")
    ap.add_argument("--len", type=float, default=45.0, help="seconds per window")
    ap.add_argument("--pitch", type=float, default=1.0,
                    help="pitch factor applied to OUT's excerpts, e.g. 0.959 for a dub sped "
                         "up to PAL with its pitch kept (sync's WARNING): the two soundtracks "
                         "must match for the scores to discriminate")
    ap.add_argument("--keep", help="keep the excerpts and stems in this folder")
    a = ap.parse_args()
    with open(a.report, encoding="utf-8") as f:
        starts = pick_starts(json.load(f), a.n, a.len)
    tmp = a.keep or tempfile.mkdtemp(prefix="sepq_")
    os.makedirs(tmp, exist_ok=True)
    w = os.path.join(tmp, "x.wav")
    R = np.concatenate([excerpt(a.ref, t, a.len, w) for t in starts])
    O = np.concatenate([excerpt(a.out, t, a.len, w, a.pitch) for t in starts])
    n = min(len(R), len(O))
    R, O = R[:n], O[:n]
    pair = os.path.join(tmp, "pair.wav")
    wavfile.write(pair, SR, np.concatenate([R, O]))
    ext_py = os.environ.get("LOSSLESSSYNC_EXT_PYTHON", "")
    if not ext_py and os.path.isfile(venv_python("venv-sep")):
        ext_py = venv_python("venv-sep")
    ext_py = ext_py or sys.executable
    model_dir = Options.sep_model_dir
    fb = mel_fb()
    LR, LO = logmel(R, fb), logmel(O, fb)
    floor = LR.max(0, keepdims=True) - 40      # per frame: quiet bands would swamp the score
    loud = LR.max(0) > LR.max() - 55
    dmix = dist(LR, LO, floor)
    order = np.argsort(dmix[loud])
    idx = np.flatnonzero(loud)
    dial = idx[order[-int(0.4 * len(order)):]]
    clean = idx[order[:int(0.2 * len(order))]]
    print(f"{len(starts)} x {a.len:g} s at {', '.join(f'{t:.0f}' for t in starts)} s; "
          f"{len(dial)} dialogue / {len(clean)} clean frames")
    print(f"  {'model':<48}{'agree':>7}{'keep':>7}{'mix':>7}")
    for model in [m for m in a.models.split(",") if m]:
        voc = os.path.join(tmp, f"voc_{model}.wav")
        try:
            if not os.path.isfile(voc):
                separate(ext_py, model, model_dir, pair, voc)
        except RuntimeError as e:
            print(f"  {model:<48} failed: {str(e)[:300]}", flush=True)
            continue
        V = np.array(read_info(voc).memmap())       # the helper writes float RF64
        rr, ro = R - V[:n], O - V[n:2 * n]
        LRr, LOr = logmel(rr, fb), logmel(ro, fb)
        agree = dist(LRr, LOr, floor)[dial].mean()
        keep = 0.5 * (dist(LRr, LR, floor)[clean].mean() + dist(LOr, LO, floor)[clean].mean())
        print(f"  {model:<48}{agree:7.2f}{keep:7.2f}{dmix[dial].mean():7.2f}", flush=True)
        if not a.keep:
            os.remove(voc)
    if not a.keep:
        os.remove(pair)
        os.rmdir(tmp)


if __name__ == "__main__":
    main()
