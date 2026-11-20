"""Generate a synthetic Blu-ray reference + TV dub pair with known ground truth.

usage: python tools/make_synth.py OUT_DIR [--variant pal|tempo|none] [--seed N] [--dropouts]
                                  [--ad] [--recap] [--intro] [--channels 1|2|6]

--dropouts adds TV signal failures: lost audio (time skips) and muted stretches, several
in a row, on top of the usual edits.
--ad inserts a 30 s ad break into the TV at ref 100 s (the ref runs on across it).
--recap starts the TV with a 20 s recap made of later programme (repeated content).
--intro replaces the first 20 s of the programme with a different 25 s TV intro.
--credits plays the last 15 s of the programme (the end credits) 25% faster on the TV.
--channels sets the TV layout: mono, stereo (default) or 5.1 (AC-3 instead of MP2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from fractions import Fraction

import numpy as np
from scipy import signal

SR = 48000


def note_track(rng, dur, density=1.0):
    out = np.zeros(int(dur * SR), np.float32)
    t = 0.0
    scale = np.array([0, 2, 3, 5, 7, 8, 10])
    while t < dur - 1:
        nd = rng.uniform(0.15, 0.9) / density
        midi = 48 + rng.choice(scale) + 12 * rng.integers(0, 3)
        f0 = 440 * 2 ** ((midi - 69) / 12)
        n = int(nd * SR)
        tt = np.arange(n) / SR
        env = np.minimum(1, tt / 0.01) * np.exp(-tt * rng.uniform(1.5, 6))
        w = sum(np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 6.28)) / h ** 1.3
                for h in range(1, 6))
        i0 = int(t * SR)
        out[i0:i0 + n] += (0.12 * env * w)[:len(out) - i0]
        t += nd * rng.uniform(0.5, 1.0)
    return out


def effects_track(rng, dur, rate=0.6):
    out = np.zeros(int(dur * SR), np.float32)
    t = rng.uniform(0, 1)
    while t < dur - 2:
        n = int(rng.uniform(0.05, 1.2) * SR)
        x = rng.standard_normal(n).astype(np.float32)
        lo, hi = sorted(rng.uniform(80, 9000, 2))
        hi = max(hi, lo * 1.5)
        b, a = signal.butter(2, [lo / (SR / 2), min(hi, 20000) / (SR / 2)], "band")
        x = signal.lfilter(b, a, x).astype(np.float32)
        tt = np.arange(n) / SR
        env = np.minimum(1, tt / 0.003) * np.exp(-tt * rng.uniform(2, 25))
        i0 = int(t * SR)
        out[i0:i0 + n] += (rng.uniform(0.05, 0.4) * env * x)[:len(out) - i0]
        t += rng.exponential(1 / rate)
    return out


def ambience(rng, dur, level=0.01):
    x = rng.standard_normal(int(dur * SR)).astype(np.float32)
    b, a = signal.butter(1, 400 / (SR / 2))
    return (level * signal.lfilter(b, a, x) * 4).astype(np.float32)


def speech(rng, dur, f0_base):
    out = np.zeros(int(dur * SR), np.float32)
    t = rng.uniform(0.2, 1.0)
    while t < dur - 4:
        phrase = rng.uniform(1.0, 4.0)
        n = int(phrase * SR)
        tt = np.arange(n) / SR
        f0 = f0_base * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * tt)) * \
            rng.uniform(0.85, 1.2)
        ph = np.cumsum(f0) / SR
        src = (2 * (ph % 1) - 1).astype(np.float32)
        y = np.zeros(n, np.float32)
        s = 0
        while s < n:
            L = int(rng.uniform(0.12, 0.3) * SR)
            seg = src[s:s + L]
            if rng.random() < 0.25:
                seg = rng.standard_normal(len(seg)).astype(np.float32) * 0.3
            acc = np.zeros(len(seg), np.float32)
            for fc, bw in ((rng.uniform(300, 900), 120), (rng.uniform(900, 2400), 200),
                           (rng.uniform(2400, 3500), 300)):
                b, a = signal.iirpeak(fc / (SR / 2), fc / bw)
                acc += signal.lfilter(b, a, seg).astype(np.float32)
            e = np.sin(np.pi * np.arange(len(seg)) / len(seg)) ** 0.7
            y[s:s + len(seg)] = acc * e
            s += L
        i0 = int(t * SR)
        out[i0:i0 + n] += (0.35 * y)[:len(out) - i0]
        t += phrase + rng.uniform(0.3, 1.8)
    return out


def write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(path, SR, np.clip(x, -1, 1).astype(np.float32))


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variant", default="pal", choices=["pal", "tempo", "none"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dur", type=float, default=420.0)
    ap.add_argument("--dropouts", action="store_true")
    ap.add_argument("--ad", action="store_true")
    ap.add_argument("--recap", action="store_true")
    ap.add_argument("--intro", action="store_true")
    ap.add_argument("--credits", action="store_true")
    ap.add_argument("--channels", type=int, default=2, choices=[1, 2, 6])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    D = a.dur

    music = note_track(rng, D)
    # music only in some scenes; a dialogue-only scene at 300-330 s
    mask = np.zeros(int(D * SR), np.float32)
    for s, e in ((0, 70), (95, 180), (210, 290), (340, D)):
        mask[int(s * SR):int(e * SR)] = 1
    mask = signal.lfilter([1 / 4800] * 1, [1, -(1 - 1 / 4800)], mask).astype(np.float32)
    music *= mask
    fx = effects_track(rng, D)
    fx[int(300 * SR):int(330 * SR)] *= 0.05
    amb = ambience(rng, D)
    me_l = music + fx * 0.8 + amb
    me_r = music * 0.9 + fx + amb * 0.9
    en = speech(np.random.default_rng(a.seed + 100), D, 120)
    pt = speech(np.random.default_rng(a.seed + 200), D, 170)

    # Reference: 5.1 English (FL FR FC LFE BL BR)
    b, aa = signal.butter(2, 120 / (SR / 2))
    lfe = signal.lfilter(b, aa, fx).astype(np.float32)
    ref = np.stack([me_l, me_r, en + 0.3 * (me_l + me_r), lfe, amb * 0.7, amb * 0.7], 1)
    ref_wav = os.path.join(a.out, "ref.wav")
    write_wav(ref_wav, ref * 0.7)
    ff("-i", ref_wav, "-c:a", "flac", "-metadata:s:a:0", "language=eng",
       os.path.join(a.out, "ref.mka"))
    os.remove(ref_wav)

    # PT programme on the reference timeline (different mix balance & EQ)
    b, aa = signal.butter(2, 11000 / (SR / 2))
    pt_l = signal.lfilter(b, aa, 0.85 * me_l).astype(np.float32) + 0.8 * pt
    pt_r = signal.lfilter(b, aa, 0.85 * me_r).astype(np.float32) + 0.8 * pt
    prog = np.stack([pt_l, pt_r], 1)

    # TV signal failures (ref time, length s, kind): "skip" loses the audio and the time,
    # "mute" keeps the time but the audio is gone
    drops = []
    if a.dropouts:
        drops = [(40.0, 0.08, "skip"), (43.5, 0.3, "skip"), (46.0, 0.25, "mute"),
                 (47.2, 0.04, "skip"), (52.0, 1.2, "skip"), (360.0, 0.5, "skip"),
                 (362.0, 0.15, "mute")]
    prog_tv = prog.copy()
    for t, d, kind in drops:
        if kind == "mute":
            prog_tv[int(t * SR):int((t + d) * SR)] = 0

    # ideal result = programme with removed parts silenced
    edits = [(0, 150), (150, 250), (254, 400)]
    lead = {0: 8.0, 150: 75.0, 254: 0.0}
    if a.ad:
        edits[0:1] = [(0, 100), (100, 150)]
        lead[100] = 30.0
    if a.intro:
        edits[0] = (20, edits[0][1])
        lead[20] = 25.0
    fast = {385.0: 1.25} if a.credits else {}
    ideal = np.zeros_like(prog)
    for s, e in edits:
        ideal[int(s * SR):int(e * SR)] = prog[int(s * SR):int(e * SR)]
    for t, d, kind in drops:
        ideal[int(t * SR):int((t + d) * SR)] = 0
    write_wav(os.path.join(a.out, "ideal.wav"), ideal)

    r2 = np.random.default_rng(a.seed + 300)
    junk = lambda d: np.stack([note_track(r2, d, 2) + effects_track(r2, d, 2)] * 2, 1) * 0.8
    parts, segments = [], []
    if a.recap:
        for s, e in ((300.0, 310.0), (210.0, 220.0)):
            parts.append(prog_tv[int(s * SR):int(e * SR)])
    for s, e in edits:
        if lead[s]:
            parts.append(junk(lead[s]))
        cur, rate = s, 1.0
        for t, d, kind in sorted(drops + [(t, 0.0, "fast") for t in fast]) + [(e, 0.0, "end")]:
            if kind not in ("skip", "end", "fast") or not (s <= t <= e):
                continue
            piece = prog_tv[int(cur * SR):int(t * SR)]
            seg = {"ref_start": cur, "ref_end": t}
            if rate != 1.0:
                fr = Fraction(rate).limit_denominator(100)
                piece = signal.resample_poly(piece, fr.denominator, fr.numerator, axis=0)
                seg["speed"] = rate
            parts.append(piece.astype(np.float32))
            segments.append(seg)
            cur = t + d
            if kind == "fast":
                rate = fast[t]
    parts.append(junk(5))
    tv = np.concatenate(parts)
    truth = {"segments": segments, "variant": a.variant,
             "dropouts": [{"ref_start": t, "ref_end": t + d, "kind": k} for t, d, k in drops]}
    if a.channels == 1:
        tv = tv.mean(1)
    elif a.channels == 6:
        # 5.1 (FL FR FC LFE BL BR): the programme in front, a little of it in the surrounds
        L, R = tv[:, 0], tv[:, 1]
        tv = np.stack([L, R, 0.5 * (L + R), np.zeros_like(L), 0.3 * L, 0.3 * R], 1) * 0.7
    tv_wav = os.path.join(a.out, "tv_pre.wav")
    if a.variant == "pal":
        tv = signal.resample_poly(tv, 960, 1001, axis=0).astype(np.float32)
        truth["speed"] = 1001 / 960
        write_wav(tv_wav, tv)
    elif a.variant == "tempo":
        write_wav(tv_wav, tv)
        tmp = tv_wav + ".t.wav"
        os.replace(tv_wav, tmp)
        ff("-i", tmp, "-af", f"rubberband=tempo={1001 / 960}:pitchq=quality", "-c:a",
           "pcm_f32le", tv_wav)
        os.remove(tmp)
        truth["speed"] = 1001 / 960
    else:
        truth["speed"] = 1.0
        write_wav(tv_wav, tv)
    codec = ["ac3", "-b:a", "448k"] if a.channels == 6 else ["mp2", "-b:a", "256k"]
    ff("-i", tv_wav, "-c:a", *codec, "-metadata:s:a:0", "language=por",
       "-f", "mpegts", os.path.join(a.out, "tv.ts"))
    os.remove(tv_wav)
    with open(os.path.join(a.out, "truth.json"), "w") as f:
        json.dump(truth, f, indent=2)
    print("written to", a.out)


if __name__ == "__main__":
    sys.exit(main())
