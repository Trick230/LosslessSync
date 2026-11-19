"""Compare a sync result with the synthetic ground truth.

usage: python tools/check.py CASE_DIR  (expects truth.json, report.json, ideal.wav, out.flac)
exit code 0 when cut points are within 10 ms and timing error is under 0.1 ms
(tempo variant: time-stretching is not sample-exact, so 150 ms / median 0.5 ms / max 5 ms).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from losslesssync.audio import media
from losslesssync.sync import speed  # noqa: E402


def main():
    d = sys.argv[1]
    truth = json.load(open(os.path.join(d, "truth.json")))
    rep = json.load(open(os.path.join(d, "report.json")))
    ok = True
    tempo = truth.get("variant") == "tempo"
    cut_tol = 0.15 if tempo else 0.01
    print(f"speed: found {rep['speed']:.7f} truth {truth['speed']:.7f}  mode {rep['stretch_mode']}")
    t_cuts = sorted({x for s in truth["segments"] for x in (s["ref_start"], s["ref_end"])})
    f_cuts = sorted({round(x, 3) for s in rep["segments"] for x in (s["ref_start"], s["ref_end"])})
    print("truth cuts:", t_cuts)
    print("found cuts:", f_cuts)
    for t in t_cuts:
        e = min(abs(t - f) for f in f_cuts)
        flag = "" if e <= cut_tol else "   <-- off"
        ok &= e <= cut_tol
        print(f"  cut {t:8.3f}: error {e * 1000:7.1f} ms{flag}")
    if len(rep["segments"]) != len(truth["segments"]):
        print(f"  segment count {len(rep['segments'])} != {len(truth['segments'])}")
        ok = False

    ideal = media.decode_mono(os.path.join(d, "ideal.wav"), media.audio_streams(os.path.join(d, "ideal.wav"))[0])
    out = media.decode_mono(os.path.join(d, "out.flac"), media.audio_streams(os.path.join(d, "out.flac"))[0])
    errs = []
    for s in truth["segments"]:
        t = s["ref_start"] + 1
        while t + 4 < s["ref_end"] - 1:
            d_, conf = speed.gcc_phat(ideal[int(t * 16000):int((t + 4) * 16000)],
                                      out[int(t * 16000):int((t + 4) * 16000)], 16000, 0.05)
            if conf > 7:
                errs.append(abs(d_) * 1000)
            t += 5
    e = np.array(errs)
    print(f"timing vs ideal: {len(e)} windows, median {np.median(e):.4f} ms, max {e.max():.4f} ms")
    if tempo:
        ok &= len(e) > 0 and np.median(e) < 0.5 and e.max() < 5.0
    else:
        ok &= len(e) > 0 and e.max() < 0.1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
