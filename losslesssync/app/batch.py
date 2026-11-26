"""Batch mode: sync every episode in a folder, pairing reference and source files by name."""
from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..audio import media
from ..stretch import stems
from ..sync.pipeline import Options, sync, uses_stems
from . import report_html

MEDIA_EXT = {".mkv", ".m2ts", ".mts", ".ts", ".mp4", ".m4v", ".mka", ".wav", ".w64", ".flac",
             ".ac3", ".eac3", ".ec3", ".dts", ".thd", ".m4a", ".aac", ".opus", ".mp2", ".mpg"}
REF_TOKENS = {"en", "eng", "english", "ingles"}
SRC_TOKENS = {"pt", "por", "ptpt", "portugues", "portuguese", "dub", "dobrado", "tv"}
_EP_RES = [re.compile(r"(?i)(?<![a-z0-9])[st](\d{1,2})[ ._-]?e(\d{1,3})(?!\d)"),
           re.compile(r"(?i)(?<![a-z0-9])(\d{1,2})x(\d{2,3})(?!\d)")]


def _tokens(name: str) -> list[str]:
    s = unicodedata.normalize("NFKD", os.path.splitext(name)[0].lower())
    s = "".join(c for c in s if not unicodedata.combining(c))   # Português -> portugues
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]


def episode_key(name: str) -> str:
    """S01E02 / T01E02 / 1x02 -> 's1e2'; otherwise the name without language words."""
    for rx in _EP_RES:
        m = rx.search(name)
        if m:
            return f"s{int(m.group(1))}e{int(m.group(2))}"
    return " ".join(t for t in _tokens(name) if t not in REF_TOKENS | SRC_TOKENS)


def _list(folder: str, pattern: str | None) -> list[str]:
    out = []
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if (os.path.isfile(p) and os.path.splitext(n)[1].lower() in MEDIA_EXT
                and not os.path.splitext(n)[0].lower().endswith("_synced")):  # our outputs
            if pattern is None or fnmatch.fnmatch(n.lower(), pattern.lower()):
                out.append(p)
    return out


def _lang(path: str) -> str | None:
    t = set(_tokens(os.path.basename(path)))
    r, s = bool(t & REF_TOKENS), bool(t & SRC_TOKENS)
    return "ref" if r and not s else "src" if s and not r else None


def pair_files(ref_dir: str, src_dir: str, ref_glob: str | None = None,
               src_glob: str | None = None) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Returns ([(key, ref, src)], warnings). In one shared folder without globs the
    language word in the name (EN / PT ...) tells reference from source."""
    same = os.path.normcase(os.path.abspath(ref_dir)) == os.path.normcase(os.path.abspath(src_dir))
    refs, srcs = _list(ref_dir, ref_glob), _list(src_dir, src_glob)
    if same:
        if ref_glob is None:
            refs = [p for p in refs if _lang(p) == "ref" or (src_glob and p not in srcs)]
        if src_glob is None:
            srcs = [p for p in srcs if _lang(p) == "src" or (ref_glob and p not in refs)]
        refs = [p for p in refs if p not in srcs]
    warn = []
    if same and (ref_glob is None or src_glob is None):
        unk = [p for p in _list(ref_dir, None) if p not in refs and p not in srcs]
        warn += [f"cannot tell if {os.path.basename(p)} is reference or source "
                 f"(no EN/PT in the name; use --ref-glob/--src-glob)" for p in unk]
    by_key: dict[str, list[str]] = {}
    for p in srcs:
        by_key.setdefault(episode_key(os.path.basename(p)), []).append(p)
    refs_by_key: dict[str, list[str]] = {}
    for r in refs:
        refs_by_key.setdefault(episode_key(os.path.basename(r)), []).append(r)
    pairs = []
    for k, rs in refs_by_key.items():
        r = rs[0]
        cand = by_key.get(k, [])
        if len(rs) > 1:
            warn.append(f"several references for {k}: " + ", ".join(os.path.basename(x) for x in rs))
        elif len(cand) == 1:
            pairs.append((k, r, cand[0]))
        elif not cand:
            warn.append(f"no source for {os.path.basename(r)}")
        else:
            warn.append(f"several sources for {os.path.basename(r)}: "
                        + ", ".join(os.path.basename(c) for c in cand))
    for k, ps in by_key.items():
        if k not in refs_by_key:
            warn += [f"no reference for {os.path.basename(p)}" for p in ps]
    return pairs, warn


def summarize(report: dict) -> dict:
    v = report.get("verify") or {}
    return {"coverage_pct": report.get("coverage_pct"), "speed": report.get("speed_fraction"),
            "stretch_mode": report.get("stretch_mode"), "gaps": len(report.get("gaps", [])),
            "dropouts": len(report.get("dropouts") or []),
            "median_ms": v.get("median_abs_ms"), "p95_ms": v.get("p95_abs_ms"),
            "max_ms": v.get("max_abs_ms"),
            "within_1ms_pct": v.get("within_1ms_pct")}


def run_episode(ref: str, src: str, out: str, rep: str, log_path: str, opt: Options,
                threads: int | None, split=None) -> dict:
    """One episode; all its log lines go to log_path. Never raises."""
    if threads:
        os.environ["LOSSLESSSYNC_THREADS"] = str(threads)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        def log(msg):
            lf.write(msg + "\n")
            lf.flush()
        try:
            report = sync(ref, src, out, dataclasses.replace(opt, report=rep), log_fn=log,
                          split=split)
            return {"status": "ok", "seconds": time.time() - t0, **summarize(report)}
        except Exception as e:  # keep going with the next episode
            lf.write(traceback.format_exc())
            return {"status": "error", "seconds": time.time() - t0, "error": f"{type(e).__name__}: {e}"}


def episode_html(rep: str, p95_ms: float) -> str | None:
    """Write the episode's HTML report next to its JSON one; None if there is no report."""
    try:
        with open(rep, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None
    page = os.path.splitext(rep)[0] + ".html"
    with open(page, "w", encoding="utf-8") as f:
        f.write(report_html.episode_page(report, p95_ms))
    return page


def needs_check(r: dict, p95_ms: float) -> list[str]:
    if r.get("status") == "error":
        return ["failed, see log"]
    why = []
    if r.get("dropouts"):
        why.append(f"{r['dropouts']} dropout(s)")
    if r.get("p95_ms") is not None and r["p95_ms"] > p95_ms:
        why.append(f"p95 {r['p95_ms']:.2f} ms")
    if r.get("median_ms") is None and r.get("status") == "ok":
        why.append("not verified")
    return why


class _Prefetch:
    """An episode's vocals separation, started in a thread (the source decode takes a few
    seconds) into its own temp folder."""

    def __init__(self, src: str, opt: Options, out_dir: str):
        self.dir = tempfile.mkdtemp(prefix="losslesssync_sep_", dir=opt.tmp_dir or out_dir)
        self.keep = opt.keep_temp
        self.split = None
        self.thread = threading.Thread(target=self._start, args=(src, opt), daemon=True)
        self.thread.start()

    def _start(self, src, opt):
        try:
            st = media.pick_stream(media.audio_streams(src), opt.src_track, ("por",))
            self.split = stems.start_split(src, st, self.dir, opt, lambda m: None)
        except Exception:   # the episode's own sync starts it again and reports the error
            self.split = None

    def get(self):
        self.thread.join()
        return self.split

    def close(self):
        self.thread.join()
        if self.split:
            self.split[0].kill()
        if not self.keep:
            shutil.rmtree(self.dir, ignore_errors=True)


def _quiet(fn, arg, log_path):
    """Starting the next episode's separation must not fail the current episode."""
    try:
        fn(arg)
    except Exception as e:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"could not start the next episode's separation: {e}\n")


def _run_stems_pipelined(todo, done, log_fn):
    """One episode at a time, but the next episode's separation (GPU) starts as soon as the
    current one's is done, so it runs while this one finishes on the CPU."""
    pre: dict[int, _Prefetch] = {}

    def start(i):
        if i < len(todo) and i not in pre:
            t = todo[i]
            pre[i] = _Prefetch(t[2], t[6], os.path.dirname(os.path.abspath(t[3])))

    try:
        for i, t in enumerate(todo):
            log_fn(f"syncing {t[0]} ...")
            start(i)
            split = pre[i].get()
            if split:
                split[0].on_gpu_free = lambda i=i, t=t: _quiet(start, i + 1, t[5])
            done(t, run_episode(*t[1:], None, split))
            start(i + 1)              # if this episode never reached its stretch
            pre.pop(i).close()
    finally:
        for p in pre.values():
            p.close()


def batch(ref_dir: str, src_dir: str, out_dir: str, opt: Options, fmt: str = "flac",
          ref_glob: str | None = None, src_glob: str | None = None, jobs: int = 1,
          overwrite: bool = False, mux: bool = False, check_p95_ms: float = 1.0,
          dry_run: bool = False, log_fn=print) -> dict:
    pairs, warn = pair_files(ref_dir, src_dir, ref_glob, src_glob)
    for w in warn:
        log_fn(f"warning: {w}")
    log_fn(f"{len(pairs)} episode(s)")
    tasks = []
    for k, ref, src in pairs:
        stem = os.path.splitext(os.path.basename(src))[0] + "_synced"
        base = os.path.join(out_dir, stem)
        out, rep = f"{base}.{fmt}", f"{base}.json"
        o = dataclasses.replace(opt, mux=f"{base}.mkv" if mux else None)
        log_fn(f"  {k:<8} {os.path.basename(ref)}  <-  {os.path.basename(src)}")
        tasks.append((k, ref, src, out, rep, f"{base}.log", o))
    if dry_run:
        return {"episodes": [], "warnings": warn}
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, dict] = {}
    todo = []
    for t in tasks:
        k, ref, src, out, rep = t[:5]
        if not overwrite and os.path.exists(out) and os.path.exists(rep):
            try:
                with open(rep, encoding="utf-8") as f:
                    old = summarize(json.load(f))
                results[k] = {"status": "skipped (exists)", "seconds": 0.0, **old}
                continue
            except (OSError, ValueError):
                log_fn(f"warning: unreadable report {rep}, redoing {k}")
        todo.append(t)
    jobs = max(1, min(jobs, len(todo) or 1))
    threads = max(1, (os.cpu_count() or 4) // jobs) if jobs > 1 else None
    t0 = time.time()

    def done(t, r):
        results[t[0]] = r
        msg = (f"coverage {r['coverage_pct']:.1f}%, median {r['median_ms'] or 0:.3f} ms, "
               f"p95 {r['p95_ms'] or 0:.3f} ms" if r["status"] == "ok"
               else r.get("error", "").splitlines()[0][:200])
        log_fn(f"[{len(results)}/{len(tasks)}] {t[0]} {r['status']} ({r['seconds']:.0f} s) {msg}")

    if jobs == 1 and todo and uses_stems(todo[0][3], todo[0][6]):
        _run_stems_pipelined(todo, done, log_fn)
    elif jobs == 1:
        for t in todo:
            log_fn(f"syncing {t[0]} ...")
            done(t, run_episode(*t[1:], None))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_episode, *t[1:], threads): t for t in todo}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # worker died (e.g. out of memory)
                    r = {"status": "error", "seconds": 0.0, "error": f"{type(e).__name__}: {e}"}
                done(t, r)

    episodes = []
    for k, ref, src, out, rep, lp, _ in tasks:
        r = results[k]
        episodes.append({"episode": k, "reference": ref, "source": src, "output": out,
                         "report": rep, "log": lp, **r, "html": episode_html(rep, check_p95_ms),
                         "check": needs_check(r, check_p95_ms)})
    summary = {"seconds": time.time() - t0, "jobs": jobs, "warnings": warn, "episodes": episodes}
    with open(os.path.join(out_dir, "batch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(report_html.batch_index(summary, out_dir))

    log_fn("")
    log_fn("episode   status   coverage  median ms  p95 ms  dropouts  check")
    for ep in episodes:
        cov = f"{ep['coverage_pct']:.1f}%" if ep.get("coverage_pct") is not None else "-"
        med = f"{ep['median_ms']:.3f}" if ep.get("median_ms") is not None else "-"
        mx = f"{ep['p95_ms']:.3f}" if ep.get("p95_ms") is not None else "-"
        st = "ok" if ep["status"] == "ok" else "skip" if ep["status"].startswith("skip") else "ERROR"
        log_fn(f"{ep['episode']:<9} {st:<8} {cov:>8}  {med:>9}  {mx:>6}  {ep.get('dropouts', 0) or 0:>8}  "
               + ("; ".join(ep["check"]) or "-"))
    n_err = sum(e["status"] == "error" for e in episodes)
    log_fn(f"{len(episodes) - n_err}/{len(episodes)} ok, {n_err} error(s), "
           f"{sum(bool(e['check']) for e in episodes)} to check; "
           f"total {summary['seconds']:.0f} s; report in {os.path.join(out_dir, 'index.html')}")
    return summary
