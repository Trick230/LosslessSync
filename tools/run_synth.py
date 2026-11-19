"""Synthetic end-to-end check: build each case, sync it, compare with the ground truth.

usage: python tools/run_synth.py [OUT_ROOT] [--cases pal,tempo,none,drop_pal,...]

Cases: pal, tempo, none and drop_<variant> (the same with TV signal dropouts). Each runs
make_synth.py, `losslesssync sync` and check.py; exit code 0 when every case passes.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES = ["pal", "tempo", "none", "drop_pal", "drop_tempo", "drop_none"]


def run_case(out_root: str, case: str) -> bool:
    d = os.path.join(out_root, case)
    variant = case.removeprefix("drop_")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "make_synth.py"), d, "--variant", variant]
                   + (["--dropouts"] if case.startswith("drop_") else []), check=True, env=env)
    log = os.path.join(d, "sync.log")
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([py, "-m", "losslesssync", "sync", "--ref", os.path.join(d, "ref.mka"),
                            "--src", os.path.join(d, "tv.ts"), "-o", os.path.join(d, "out.flac"),
                            "--report", os.path.join(d, "report.json")],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode:
        print(f"{case}: sync failed, see {log}")
        return False
    ok = subprocess.run([py, os.path.join(HERE, "check.py"), d], env=env).returncode == 0
    print(f"{case}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_root", nargs="?", default=os.path.join(ROOT, "test_out"))
    ap.add_argument("--cases", default=",".join(CASES))
    a = ap.parse_args()
    cases = [c for c in a.cases.split(",") if c]
    bad = [c for c in cases if c.removeprefix("drop_") not in ("pal", "tempo", "none")]
    if bad:
        ap.error(f"unknown case(s): {', '.join(bad)}")
    results = [run_case(a.out_root, c) for c in cases]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
