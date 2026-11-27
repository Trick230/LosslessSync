"""Sync every episode where the media lives.

    python tools/media_sync.py [--cloud [--dir DIR] [--here] [--only ID,...] [--redo]] [--dry-run] [-- BATCH_OPTIONS...]

PC (default): `losslesssync batch` over LLS_MEDIA_DIR (its refs/ and srcs/ folders when it
has them, else the folder itself, EN/PT in the names) into LLS_OUT_DIR; episodes already
done are skipped.

Cloud (--cloud or LLS_ENV=cloud): a folder of the project's Drive through devcloud (`dc drive`,
https://github.com/Trick230/devcloud): --dir DIR (default samples), the same rule as the PC:
its refs/ and srcs/ when it has them, else DIR itself, EN/PT in the names. Results go to
synced/; episodes with their report in synced/<episode>/ are skipped. Each pending episode
becomes a GitHub Actions job of its own (`lls remote`, all in parallel); --here syncs them
on this machine instead, one at a time: copy its two files to .lls/media-cache, sync,
upload the flac/json/html/log to synced/<episode>/, delete that episode's cache. --redo
syncs done episodes again. --only picks episodes by ID (a hash of the episode key: Actions logs are public, so a runner never prints names and batch's own output
goes only into the uploaded log). dc is found through DEVCLOUD (environment or .env), else
.devcloud/ (Actions) or ~/devcloud.

BATCH_OPTIONS go to `losslesssync batch` as they are (e.g. --pitch keep --verify-deep).
"""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from losslesssync import LLS, dotenv  # noqa: E402
from losslesssync import dc_launcher as _dc  # noqa: E402
from losslesssync.app.batch import MEDIA_EXT, _lang, episode_key  # noqa: E402

QUIET = os.environ.get("GITHUB_ACTIONS") == "true"   # public logs: no media names
REDO = False                                          # --redo: done episodes too


def ep_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def batch(ref_dir: str, src_dir: str, out_dir: str, extra: list[str]) -> int:
    cmd = [sys.executable, "-m", "losslesssync", "batch", "--ref-dir", ref_dir,
           "--src-dir", src_dir, "--out-dir", out_dir, *extra]
    quiet = subprocess.DEVNULL if QUIET else None
    return subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH=ROOT),
                           stdout=quiet, stderr=quiet)


def pc(extra: list[str], dry: bool) -> int:
    media, out = os.environ.get("LLS_MEDIA_DIR"), os.environ.get("LLS_OUT_DIR")
    if not media or not out:
        sys.exit("set LLS_MEDIA_DIR and LLS_OUT_DIR (environment or .env)")
    refs, srcs = os.path.join(media, "refs"), os.path.join(media, "srcs")
    if not (os.path.isdir(refs) and os.path.isdir(srcs)):
        refs = srcs = media
    return batch(refs, srcs, out, extra + (["--dry-run"] if dry else []))


def dc_launcher() -> str:
    return _dc() or sys.exit("devcloud not found: clone https://github.com/Trick230/devcloud or set DEVCLOUD")


def load_env() -> None:
    """KEY=VALUE lines of the repo's .env (devcloud: dc secrets pull), without overriding the
    environment."""
    for k, v in dotenv().items():
        os.environ.setdefault(k, v)


def drive(*args: str, capture: bool = False, missing_ok: bool = False) -> str:
    """`dc drive ARGS` in this project (paths are inside its Drive folder); missing_ok: a
    folder that does not exist yet lists as empty."""
    r = subprocess.run([dc_launcher(), "drive", *args], text=True, stdin=subprocess.DEVNULL,
                       capture_output=capture or QUIET, cwd=ROOT, check=False)
    if r.returncode != 0 and missing_ok and "directory not found" in (r.stderr or ""):
        return ""
    if r.returncode != 0:
        raise SystemExit(f"dc drive {args[0]} failed ({r.returncode})" if QUIET else
                         f"dc drive {' '.join(args)} failed ({r.returncode}) {r.stderr or ''}")
    return r.stdout or ""


def media_files(folder: str, lang: str | None = None) -> dict[str, str]:
    """{episode key: file name} of a Drive folder; lang ("ref"/"src"): only the files whose
    name says so (EN / PT ...), for a folder holding both."""
    names = [n for n in drive("ls", "--files", folder, capture=True).splitlines()
             if os.path.splitext(n)[1].lower() in MEDIA_EXT
             and not os.path.splitext(n)[0].lower().endswith("_synced")
             and (lang is None or _lang(n) == lang)]
    return {episode_key(n): n for n in sorted(names)}


def media_dirs(base: str) -> tuple[str, str]:
    """(refs folder, srcs folder) on the Drive: base/refs and base/srcs when both exist, else base."""
    dirs = {d.rstrip("/") for d in drive("ls", "--dirs", base, capture=True).splitlines()}
    if {"refs", "srcs"} <= dirs:
        return f"{base}/refs", f"{base}/srcs"
    return base, base


def pending(base: str, only: set[str] | None) -> list[tuple[str, str, str, str]]:
    """(key, ref, src, episode) of the episodes not synced yet (only the IDs in `only`);
    ref and src are Drive paths."""
    ref_dir, src_dir = media_dirs(base)
    shared = ref_dir == src_dir
    refs = media_files(ref_dir, "ref" if shared else None)
    srcs = media_files(src_dir, "src" if shared else None)
    done = {d.rstrip("/") for d in drive("ls", "--dirs", "synced", capture=True, missing_ok=True).splitlines()}
    out = []
    for key in sorted(set(refs) & set(srcs)):
        if only is not None and ep_id(key) not in only:
            continue
        ref, src = refs[key], srcs[key]
        ep = os.path.splitext(src)[0]
        name = ep_id(key) if QUIET else f"{key} [{ep_id(key)}]"
        if not REDO and ep in done and f"{ep}_synced.json" in drive(
                "ls", "--files", f"synced/{ep}", capture=True).splitlines():
            print(f"{name}: done, skipped")
            continue
        print(f"{name}: pending" if QUIET else f"{name}: {ref} <- {src}")
        out.append((key, f"{ref_dir}/{ref}", f"{src_dir}/{src}", ep))
    if not QUIET:
        for key in sorted(set(refs) ^ set(srcs)):
            print(f"{key}: only in {'refs' if key in refs else 'srcs'}, skipped")
    return out


def remote(base: str, extra: list[str], dry: bool) -> int:
    """One GitHub Actions job per pending episode (`lls remote`), all in parallel."""
    ids = [ep_id(key) for key, *_ in pending(base, None)]
    if dry or not ids:
        return 0
    tail = f" -- {shlex.join(extra)}" if extra else ""
    redo = " --redo" if REDO else ""
    cmds = [f"./lls py tools/media_sync.py --cloud --dir {shlex.quote(base)} --here --only {i}{redo}{tail}"
            for i in ids]
    return subprocess.call([sys.executable, os.path.join(ROOT, "tools", "lls.py"), "remote", "--", *cmds],
                           cwd=ROOT, stdin=subprocess.DEVNULL)


def cloud(base: str, extra: list[str], dry: bool, only: set[str] | None = None) -> int:
    cache_root = os.path.join(LLS, "media-cache")
    failed = 0
    for key, ref, src, ep in pending(base, only):
        if dry:
            continue
        cache = os.path.join(cache_root, ep)
        try:
            drive("get", ref, os.path.join(cache, "refs"))
            drive("get", src, os.path.join(cache, "srcs"))
            out = os.path.join(cache, "out")
            rc = batch(os.path.join(cache, "refs"), os.path.join(cache, "srcs"), out, extra)
            if os.path.isdir(out):        # failed episodes upload their log and report too
                drive("put", out, f"synced/{ep}", "--include", "*.{flac,json,html,log}")
            failed += rc != 0
            print(f"{ep_id(key)}: {'ok' if rc == 0 else f'failed ({rc}), log in synced/'}")
        finally:
            shutil.rmtree(cache, ignore_errors=True)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    load_env()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]  # our lines between batch's
    is_cloud = "--cloud" in own or os.environ.get("LLS_ENV") == "cloud"
    dry = "--dry-run" in own
    global REDO
    REDO = "--redo" in own
    if not is_cloud:
        return pc(extra, dry)
    only = set(own[own.index("--only") + 1].split(",")) if "--only" in own else None
    base = own[own.index("--dir") + 1].strip("/") if "--dir" in own else "samples"
    if "--here" in own or QUIET or only is not None:
        return cloud(base, extra, dry, only)
    return remote(base, extra, dry)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
