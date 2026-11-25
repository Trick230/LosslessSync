"""Sync every episode where the media lives.

    python tools/media_sync.py [--cloud] [--dry-run] [-- BATCH_OPTIONS...]

PC (default): `losslesssync batch` over LLS_MEDIA_DIR (its refs/ and srcs/ folders when it
has them, else the folder itself, EN/PT in the names) into LLS_OUT_DIR; episodes already
done are skipped.

Cloud (--cloud or LLS_ENV=cloud): the rclone remote gdrive: (limited by root_folder_id to
the Drive folder LosslessSync) with refs/, srcs/ and synced/. One episode at a time: copy
its two files to .lls/media-cache, sync, upload the flac/json/html/log to
gdrive:synced/<episode>/, delete that episode's cache. Episodes with their report in
synced/<episode>/ are skipped. The remote comes from RCLONE_CONFIG_GDRIVE_* (see .env.example).

BATCH_OPTIONS go to `losslesssync batch` as they are (e.g. --pitch keep --verify-deep).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gpu"))
from run_gpu import load_env  # noqa: E402  (.env loader, standard library only)

from losslesssync import LLS  # noqa: E402
from losslesssync.app.batch import MEDIA_EXT, episode_key  # noqa: E402

REMOTE = "gdrive:"


def batch(ref_dir: str, src_dir: str, out_dir: str, extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "losslesssync", "batch", "--ref-dir", ref_dir,
           "--src-dir", src_dir, "--out-dir", out_dir, *extra]
    return subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT))


def pc(extra: list[str], dry: bool) -> int:
    media, out = os.environ.get("LLS_MEDIA_DIR"), os.environ.get("LLS_OUT_DIR")
    if not media or not out:
        sys.exit("set LLS_MEDIA_DIR and LLS_OUT_DIR (environment or .env)")
    refs, srcs = os.path.join(media, "refs"), os.path.join(media, "srcs")
    if not (os.path.isdir(refs) and os.path.isdir(srcs)):
        refs = srcs = media
    return batch(refs, srcs, out, extra + (["--dry-run"] if dry else []))


def rclone(*args: str, capture: bool = False) -> str:
    r = subprocess.run(["rclone", *args], text=True, stdin=subprocess.DEVNULL,
                       capture_output=capture)
    if r.returncode != 0:
        raise SystemExit(f"rclone {' '.join(args)} failed ({r.returncode}) {r.stderr or ''}")
    return r.stdout or ""


def media_files(folder: str) -> dict[str, str]:
    """{episode key: file name} of a remote folder."""
    names = [n for n in rclone("lsf", "--files-only", REMOTE + folder, capture=True).splitlines()
             if os.path.splitext(n)[1].lower() in MEDIA_EXT]
    return {episode_key(n): n for n in sorted(names)}


def cloud(extra: list[str], dry: bool) -> int:
    if not shutil.which("rclone"):
        sys.exit("rclone missing (lls setup installs it)")
    refs, srcs = media_files("refs"), media_files("srcs")
    done = {d.rstrip("/") for d in rclone("lsf", "--dirs-only", REMOTE + "synced", capture=True).splitlines()}
    cache_root = os.path.join(LLS, "media-cache")
    failed = 0
    for key in sorted(set(refs) & set(srcs)):
        ref, src = refs[key], srcs[key]
        ep = os.path.splitext(src)[0]
        if ep in done and f"{ep}_synced.json" in rclone(
                "lsf", f"{REMOTE}synced/{ep}", capture=True).splitlines():
            print(f"{key}: done, skipped")
            continue
        print(f"{key}: {ref} <- {src}")
        if dry:
            continue
        cache = os.path.join(cache_root, ep)
        try:
            rclone("copy", f"{REMOTE}refs/{ref}", os.path.join(cache, "refs"))
            rclone("copy", f"{REMOTE}srcs/{src}", os.path.join(cache, "srcs"))
            out = os.path.join(cache, "out")
            rc = batch(os.path.join(cache, "refs"), os.path.join(cache, "srcs"), out, extra)
            if os.path.isdir(out):        # failed episodes upload their log and report too
                rclone("copy", out, f"{REMOTE}synced/{ep}", "--include", "*.{flac,json,html,log}")
            failed += rc != 0
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    for key in sorted(set(refs) ^ set(srcs)):
        print(f"{key}: only in {'refs' if key in refs else 'srcs'}, skipped")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    load_env()
    sys.stdout.reconfigure(line_buffering=True)   # our lines between batch's
    is_cloud = "--cloud" in own or os.environ.get("LLS_ENV") == "cloud"
    return (cloud if is_cloud else pc)(extra, "--dry-run" in own)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
