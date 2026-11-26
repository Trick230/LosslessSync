"""Regression check on a real episode: sync it and compare the report with a stored baseline.

usage: python tools/regress.py [--update] [--keep]
The episode is test_out/batch2/Ep S01E01 (EN reference + PT TV recording). When it is not on
this machine it comes from the project's Drive (devcloud `dc drive get`) into
.lls/media-cache/regress and is deleted afterwards (--keep leaves it). The baseline holds only
numbers (speed, cut points, verify stats): tools/regress.json, written by --update.
Exit code 0 when the cuts, speed and verify stats match the baseline within tolerance.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from losslesssync import dc_launcher  # noqa: E402

DRIVE_DIR = "test_out/batch2"
REF, SRC = "Ep S01E01 EN.wav", "Ep S01E01 PT.aac"
BASELINE = os.path.join(ROOT, "tools", "regress.json")
CACHE = os.path.join(ROOT, ".lls", "media-cache", "regress")
OUT = os.path.join(ROOT, "test_out", "regress")

CUT_TOL_S = 0.005          # a cut that moves more than this is a regression
MEDIAN_SLACK_MS = 0.05     # verify median may grow by this much
MAX_SLACK_MS = 0.2         # verify max likewise
MEASURED_SLACK = 5         # verify windows that may be lost
COVERAGE_SLACK = 0.2       # coverage percentage points that may be lost


def fetch() -> tuple[str, str, bool]:
    local = os.path.join(ROOT, DRIVE_DIR)
    if all(os.path.isfile(os.path.join(local, f)) for f in (REF, SRC)):
        return os.path.join(local, REF), os.path.join(local, SRC), False
    dc = dc_launcher()
    if not dc:
        sys.exit("regress: the episode is not here and devcloud's dc was not found")
    os.makedirs(CACHE, exist_ok=True)
    for f in (REF, SRC):
        if not os.path.isfile(os.path.join(CACHE, f)):
            subprocess.run([dc, "drive", "get", f"{DRIVE_DIR}/{f}", CACHE], check=True, cwd=ROOT)
    return os.path.join(CACHE, REF), os.path.join(CACHE, SRC), True


def summary(rep: dict) -> dict:
    v = rep["verify"]
    return {
        "speed_fraction": rep["speed_fraction"],
        "segments": [[round(s["ref_start"], 4), round(s["ref_end"], 4), round(s["src_start"], 4)]
                     for s in rep["segments"]],
        "dropouts": len(rep["dropouts"]),
        "coverage_pct": round(rep["coverage_pct"], 3),
        "verify": {"measured": v["measured"], "median_abs_ms": round(v["median_abs_ms"], 4),
                   "max_abs_ms": round(v["max_abs_ms"], 4)},
    }


def compare(base: dict, got: dict) -> bool:
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  FAIL {msg}")

    if got["speed_fraction"] != base["speed_fraction"]:
        fail(f"speed {got['speed_fraction']} != {base['speed_fraction']}")
    if len(got["segments"]) != len(base["segments"]):
        fail(f"segment count {len(got['segments'])} != {len(base['segments'])}")
    else:
        for i, (g, b) in enumerate(zip(got["segments"], base["segments"])):
            err = max(abs(x - y) for x, y in zip(g, b))
            print(f"  segment {i}: ref {g[0]:9.3f}-{g[1]:9.3f}  max move {err * 1000:6.2f} ms")
            if err > CUT_TOL_S:
                fail(f"segment {i} moved {err * 1000:.1f} ms (was {b})")
    if got["dropouts"] != base["dropouts"]:
        fail(f"dropouts {got['dropouts']} != {base['dropouts']}")
    if got["coverage_pct"] < base["coverage_pct"] - COVERAGE_SLACK:
        fail(f"coverage {got['coverage_pct']} < {base['coverage_pct']}")
    gv, bv = got["verify"], base["verify"]
    print(f"  verify: measured {gv['measured']} (base {bv['measured']}), median {gv['median_abs_ms']} ms "
          f"(base {bv['median_abs_ms']}), max {gv['max_abs_ms']} ms (base {bv['max_abs_ms']})")
    if gv["measured"] < bv["measured"] - MEASURED_SLACK:
        fail("fewer verify windows measured")
    if gv["median_abs_ms"] > bv["median_abs_ms"] + MEDIAN_SLACK_MS:
        fail("verify median grew")
    if gv["max_abs_ms"] > bv["max_abs_ms"] + MAX_SLACK_MS:
        fail("verify max grew")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--update", action="store_true", help="write the baseline from this run")
    ap.add_argument("--keep", action="store_true", help="keep the episode fetched from Drive")
    a = ap.parse_args()
    ref, src, fetched = fetch()
    try:
        os.makedirs(OUT, exist_ok=True)
        report = os.path.join(OUT, "report.json")
        out = os.path.join(OUT, "out.flac")
        cmd = [sys.executable, "-m", "losslesssync", "sync", "--ref", ref, "--src", src,
               "--pitch", "resample", "-o", out, "--report", report]
        print("regress: syncing S01E01 (resample)", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        got = summary(json.load(open(report, encoding="utf-8")))
    finally:
        if fetched and not a.keep:
            shutil.rmtree(CACHE, ignore_errors=True)
        if os.path.exists(os.path.join(OUT, "out.flac")):
            os.remove(os.path.join(OUT, "out.flac"))   # real audio: never left behind
    if a.update:
        with open(BASELINE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(got, f, indent=1)
            f.write("\n")
        print(f"regress: baseline written to {BASELINE}")
        return 0
    ok = compare(json.load(open(BASELINE, encoding="utf-8")), got)
    print("regress: PASS" if ok else "regress: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
