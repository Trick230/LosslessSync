"""End-to-end sync: analyse, correct speed, segment, render, verify."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import align, media, report_html, stems, wavio, wsola
from .features import ANA_SR, features
from .render import render
from .wavio import read_info

SPEED_ALIASES = {"pal": 1001 / 960, "pal24": 25 / 24, "ntsc": 960 / 1001, "1": 1.0, "none": 1.0}


@dataclass
class Options:
    ref_track: int | None = None
    src_track: int | None = None
    ref_langs: tuple[str, ...] = ("eng", "jpn")
    speed: str = "auto"          # auto | pal | pal24 | ntsc | 1 | <float> | <a/b>
    pitch: str = "auto"          # auto | resample | tempo | keep
    keep_engine: str = "wsola"   # stretcher for pitch=keep: wsola | rubberband
    wsola_tol_ms: float = 5.0    # WSOLA search tolerance (bounds the timing error)...
    wsola_max_ms: float = 10.0
    ext_python: str = ""          # Python with audio-separator and python-stretch (stems)
    sep_model: str = "vocals_mel_band_roformer.ckpt"
    sep_model_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "losslesssync", "models")   # ...widened up to this for a clearly better splice
    fill: str = "silence"        # silence | ref
    fade_ms: float = 300.0        # fade at edges next to a gap
    fade_opening: bool = False   # also fade into the opening titles
    dropout_fade_ms: float = 10.0  # fade at the edges of a signal dropout
    codec: str | None = None
    bitrate: str | None = None
    language: str = "por"
    mux: str | None = None
    report: str | None = None
    keep_temp: bool = False
    tmp_dir: str | None = None
    analyze_only: bool = False
    chunk_s: float = 6.0
    step_s: float = 1.5


def parse_speed(s: str) -> float | None:
    s = s.strip().lower()
    if s == "auto":
        return None
    if s in SPEED_ALIASES:
        return SPEED_ALIASES[s]
    return float(Fraction(s))


def fmt_t(t: float) -> str:
    sign = "-" if t < 0 else ""
    t = abs(t)
    return f"{sign}{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"


class Log:
    def __init__(self, fn):
        self.fn = fn
        self.t0 = time.time()

    def __call__(self, msg: str):
        self.fn(f"[{time.time() - self.t0:6.1f}s] {msg}")


def verify(ref16: np.ndarray, out16: np.ndarray, pieces: list[dict], win: float = 6.0,
           step: float = 10.0) -> dict:
    errs, methods, points = [], [], []
    total = 0
    for p in pieces:
        t = p["start"] + 1.0
        while t + win <= p["end"] - 1.0:
            total += 1
            r = align.fine_measure(ref16, out16, t, 0.0, win, max_lag=0.1)
            if r is not None:
                errs.append(r[0])
                methods.append(r[2])
            # window centre and signed error (ms), None where nothing could be measured
            points.append([round(t + win / 2, 3), None if r is None else round(r[0] * 1000, 4)])
            t += step
    if not errs:
        return {"windows": total, "measured": 0, "points": points}
    e = np.abs(np.array(errs)) * 1000
    return {"windows": total, "measured": len(errs), "points": points,
            "phat": sum(m == "phat" for m in methods),
            "median_abs_ms": float(np.median(e)), "p95_abs_ms": float(np.percentile(e, 95)),
            "max_abs_ms": float(e.max()), "within_1ms_pct": float((e <= 1.0).mean() * 100)}


def sync(ref_path: str, src_path: str, out_path: str | None, opt: Options,
         log_fn=print, progress=None) -> dict:
    log = Log(log_fn)
    prog = progress or (lambda stage, frac: None)
    ref_streams = media.audio_streams(ref_path)
    src_streams = media.audio_streams(src_path)
    ref_st = media.pick_stream(ref_streams, opt.ref_track, opt.ref_langs, avoid=("por",))
    src_st = media.pick_stream(src_streams, opt.src_track, ("por",))
    log(f"reference: {ref_st.describe()}")
    log(f"source   : {src_st.describe()}")

    out_dir = os.path.dirname(os.path.abspath(out_path or opt.report or src_path))
    tmp = tempfile.mkdtemp(prefix="losslesssync_", dir=opt.tmp_dir or out_dir)
    split = None
    try:
        if (out_path and not opt.analyze_only and opt.pitch == "keep"
                and opt.keep_engine == "stems"):
            split = stems.start_split(src_path, src_st, tmp, opt, log)
        return _sync(ref_path, src_path, out_path, opt, log, prog, ref_streams, ref_st,
                     src_st, tmp, split)
    finally:
        if split:
            split[0].kill()
        if opt.keep_temp:
            log(f"temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _sync(ref_path, src_path, out_path, opt, log, prog, ref_streams, ref_st, src_st, tmp,
          split=None):
    log("decoding analysis audio (mono 16 kHz)...")
    prog("decode", 0.0)
    ref16 = media.decode_mono(ref_path, ref_st, ANA_SR)
    src16 = media.decode_mono(src_path, src_st, ANA_SR)
    ref_dur = len(ref16) / ANA_SR
    log(f"  ref {fmt_t(ref_dur)}  src {fmt_t(len(src16) / ANA_SR)}")
    R = features(ref16)

    forced = parse_speed(opt.speed)
    if forced is None:
        log("detecting speed ratio...")
        prog("speed", 0.0)
        k, speed_scores = align.detect_speed(src16, R, log)
    else:
        k, speed_scores = forced, {}
    log(f"  using x{k:.7f} for anchor search")

    log("matching anchors over the whole reference...")
    S = features(src16, speed=k)
    act = align.active_mask(src16, len(S), k)
    anchors = align.find_anchors(S, R, k, act, opt.chunk_s, opt.step_s,
                                 progress=lambda f: prog("anchors", f))
    n_conf = sum(align.is_confident(a) for a in anchors)
    log(f"  {len(anchors)} anchors, {n_conf} confident")
    if n_conf < 2:
        raise media.MediaError("could not find matching content between the two tracks")
    del S

    if forced is None:
        k_ref, snapped = align.refine_speed(anchors, k)
        log(f"  speed refined to x{k_ref:.7f}" + (" (exact ratio)" if snapped else ""))
        k = k_ref

    if opt.pitch == "auto":
        mode, pitch_info = align.detect_pitch_mode(ref16, src16, anchors, k, opt.chunk_s)
        if abs(k - 1) > 1e-4:
            log(f"  pitch: {'speed-up with pitch shift -> resample' if mode == 'resample' else 'pitch already corrected -> time-stretch'}"
                f" (shift {pitch_info.get('best_shift_bins')} bins, "
                f"expected {pitch_info.get('expected_shift_bins', 0):.1f})")
        elif abs(pitch_info.get("best_shift_bins") or 0) >= 4:
            sh = pitch_info["best_shift_bins"]
            log(f"  WARNING: source pitch is {sh:+d}/96 octave ({(2 ** (sh / 96) - 1) * 100:+.1f}%) "
                f"off the reference: it was probably time-stretched from PAL with the pitch "
                f"kept; that removes phase coherence, so timing is limited to ~1 ms. "
                f"The original (unstretched) recording would sync sample-exact.")
    elif opt.pitch == "keep":
        # analyse on the plain resampled source (sample-exact); the stretch comes at render
        mode, pitch_info = "resample", {}
    else:
        mode, pitch_info = opt.pitch, {}

    inter_wav = os.path.join(tmp, "inter.wav")
    for it in range(3):
        log(f"speed-correcting source (x{k:.7f}, {mode})...")
        prog("correct", 0.0)
        inter16 = media.make_intermediate(src_path, src_st, k, mode, inter_wav, ANA_SR)
        segs = align.build_segments(anchors, k, opt.chunk_s)
        log(f"  {len(segs)} segments; refining offsets (GCC-PHAT)...")
        align.fine_segments(segs, ref16, inter16, k, opt.chunk_s,
                            progress=lambda f: prog("fine", f))
        slope, n, span = align.residual_drift(segs)
        log(f"  residual drift {slope * 1e6:+.2f} ppm over {n} windows")
        if forced is None and abs(slope) > 1e-6 and n >= 6 and span >= 60 and it < 2:
            k *= 1 + slope
            continue
        break
    segs = align.split_on_jumps(segs, k, opt.chunk_s)
    dropped = [s for s in segs if s.method == "coarse" and len(s.anchors) < 4]
    # an unconfirmed segment whose offset is close to a confirmed neighbour's is that
    # neighbour mis-measured in a weak passage (e.g. dialogue only); let it extend instead
    conf = sorted((s for s in segs if s.method != "coarse"), key=lambda s: s.core(k, opt.chunk_s)[0])
    for s in segs:
        if s.method != "coarse" or s in dropped:
            continue
        c = s.core(k, opt.chunk_s)[0]
        before = [x for x in conf if x.core(k, opt.chunk_s)[0] < c]
        after = [x for x in conf if x.core(k, opt.chunk_s)[0] > c]
        near = ([before[-1]] if before else []) + ([after[0]] if after else [])
        if any(abs(s.offset - x.offset) < 0.5 for x in near):
            dropped.append(s)
    segs = [s for s in segs if s not in dropped]
    if dropped:
        log(f"  dropped {len(dropped)} unconfirmed segment(s)")
    if not segs:
        raise media.MediaError("no segment could be confirmed by fine alignment")
    segs = align.merge_equal(segs, k, opt.chunk_s)

    log("placing cut points...")
    Rb = features(ref16, hp_s=0.15)
    Sb = features(inter16, hp_s=0.15)
    segs = align.split_on_dropouts(segs, ref16, inter16, Rb, Sb, k, opt.chunk_s, ref_dur,
                                   progress=lambda f: prog("dropouts", f))
    segs = align.merge_equal(segs, k, opt.chunk_s)
    segs = align.place_segments(segs, Rb, Sb, ref_dur, opt.chunk_s, k, ref16, inter16)
    del Rb, Sb
    segs = align.refine_edges(segs, inter16, ref_dur, ref16)
    pieces = [{"start": s.ref_start, "end": s.ref_end, "offset": s.offset} for s in segs]
    drops = sorted(align.dropout_gaps(segs, ref16))
    for i in drops:
        pieces[i]["fade_out_ms"] = opt.dropout_fade_ms
        pieces[i + 1]["fade_in_ms"] = opt.dropout_fade_ms
    dropouts = [{"start": segs[i].ref_end, "end": segs[i + 1].ref_start, "kind": "lost"}
                for i in drops]
    dropouts += [{"start": a, "end": b, "kind": "muted"}
                 for a, b in align.muted_spans(segs, ref16, inter16)]
    dropouts.sort(key=lambda d: d["start"])
    if dropouts:
        log(f"  {len(dropouts)} signal dropout(s) in the recording:")
        for d in dropouts:
            what = "audio lost" if d["kind"] == "lost" else "muted"
            log(f"    {fmt_t(d['start'])} - {fmt_t(d['end'])}  "
                f"({(d['end'] - d['start']) * 1000:.0f} ms, {what})")
    opening = None if opt.fade_opening else align.find_opening(segs, ref16, inter16, ref_dur)
    if opening is not None:
        pieces[opening]["fade_in"] = False
        log(f"  opening titles at {fmt_t(segs[opening].ref_start)}: no fade-in")

    report = {
        "reference": ref_path, "reference_track": ref_st.aidx, "source": src_path,
        "source_track": src_st.aidx, "speed": k, "speed_fraction": str(Fraction(k).limit_denominator(100000)),
        "speed_scores": speed_scores, "stretch_mode": mode, "pitch": pitch_info,
        "reference_duration": ref_dur, "reference_audio_delay_s": ref_st.start_time,
        "segments": [], "gaps": [], "opening_segment": opening, "dropouts": dropouts,
    }
    covered = 0.0
    prev = 0.0
    for s in segs:
        if s.ref_start - prev > 0.01:
            report["gaps"].append({"start": prev, "end": s.ref_start})
        prev = s.ref_end
        covered += s.ref_end - s.ref_start
        report["segments"].append({
            "ref_start": s.ref_start, "ref_end": s.ref_end,
            "src_start": (s.ref_start - s.offset) / k, "src_end": (s.ref_end - s.offset) / k,
            "offset_s": s.offset, "method": s.method, "spread_ms": s.spread_ms,
            "anchors": len(s.anchors), "fine_windows": len(s.fine)})
    if ref_dur - prev > 0.01:
        report["gaps"].append({"start": prev, "end": ref_dur})
    report["coverage_pct"] = covered / ref_dur * 100

    log_segments(log, report)

    if not opt.analyze_only and out_path:
        if opt.pitch == "keep" and abs(k - 1) > 1e-4:
            inter_wav = keep_pitch(src_path, src_st, k, inter16, pieces, tmp, log, report, opt,
                                   split)
        info = read_info(inter_wav)
        n_out = int(round(ref_dur * info.sr))
        fill = None
        if opt.fill == "ref":
            fill_wav = os.path.join(tmp, "fill.wav")
            media.decode_to_wav(ref_path, ref_st, fill_wav, info.channels, info.sr)
            fill = read_info(fill_wav)
        log("rendering...")
        prog("render", 0.0)
        out_wav = os.path.join(tmp, "out.wav")
        render(info, pieces, n_out, out_wav, fill, fade_ms=opt.fade_ms)
        log("verifying result against the reference...")
        prog("verify", 0.0)
        out_st = media.audio_streams(out_wav)[0]
        out16 = media.decode_mono(out_wav, out_st, ANA_SR)
        report["verify"] = verify(ref16, out16, pieces)
        v = report["verify"]
        if v.get("measured"):
            log(f"  {v['measured']}/{v['windows']} windows measured: median {v['median_abs_ms']:.3f} ms, "
                f"p95 {v['p95_abs_ms']:.3f} ms, max {v['max_abs_ms']:.3f} ms, "
                f"{v['within_1ms_pct']:.1f}% within 1 ms")
        codec = opt.codec or media.EXT_CODEC.get(os.path.splitext(out_path)[1].lower(), "flac")
        log(f"encoding {codec} -> {out_path}")
        prog("encode", 0.0)
        media.encode(out_wav, out_path, codec, opt.bitrate, opt.language)
        report["output"] = out_path
        if opt.mux:
            log(f"muxing into {opt.mux}")
            media.mux(ref_path, out_path, opt.mux, ref_st.start_time, len(ref_streams),
                      opt.language)
            report["mux"] = opt.mux
        if abs(ref_st.start_time) > 1e-4:
            log(f"  NOTE: the reference audio track starts at {ref_st.start_time * 1000:+.1f} ms; "
                f"give the synced track the same delay when muxing by hand")
    if opt.report:
        write_report(report, opt.report)
    prog("done", 1.0)
    log("done")
    return report


def write_report(report: dict, path: str):
    """JSON report, or a self-contained HTML page when the path ends in .htm/.html."""
    if path.lower().endswith((".html", ".htm")):
        text = report_html.episode_page(report)
    else:
        text = json.dumps(report, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def stretch_latency(inter16, x16, step=5.0):
    """Median offset of a pitch-kept stretch against the resampled source, with the windows."""
    dur = min(len(inter16), len(x16)) / ANA_SR
    ts = np.arange(10.0, dur - 16.0, step)
    res = align.pmap(lambda t: align.fine_measure(inter16, x16, t, 0.0, 6.0, max_lag=0.2), ts)
    ok = [(t, r[0]) for t, r in zip(ts, res) if r is not None]
    if len(ok) < 10:
        raise media.MediaError("could not measure the time-stretch latency")
    t, off = (np.array(x) for x in zip(*ok))
    return float(np.median(off)), t, off


def keep_pitch(src_path, src_st, k, inter16, pieces, tmp, log, report, opt, split=None) -> str:
    """Stretch the source to the film timing keeping its pitch (WSOLA or Rubber Band R3) and
    shift the pieces by the stretcher's latency, measured against the resampled source."""
    keep_wav = os.path.join(tmp, "keep.wav")
    if opt.keep_engine == "stems":
        log("time-stretching source keeping its pitch (vocals WSOLA + rest Signalsmith)...")
        split = split or stems.start_split(src_path, src_st, tmp, opt, log)
        n_out = int(round(wavio.read_info(split[1]).frames * k))
        def measure(path, step):
            return stretch_latency(inter16, media.decode_mono(path, media.audio_streams(path)[0],
                                                              ANA_SR), step)[0]
        voc_k, rest_k, engine, lat_v, lat_r = stems.stretch(split, tmp, k, opt, log, measure)
        shift = int(round((lat_r - lat_v) * src_st.sample_rate))  # latency > 0: content early
        log(f"  stem latencies: vocals {lat_v * 1000:+.3f} ms, rest {lat_r * 1000:+.3f} ms")
        engine["vocals"]["latency_ms"], engine["rest"]["latency_ms"] = lat_v * 1000, lat_r * 1000
        stems.mix(voc_k, rest_k, keep_wav, shift, n_out)
        if not opt.keep_temp:
            os.remove(voc_k)
            os.remove(rest_k)
        keep16 = media.decode_mono(keep_wav, media.audio_streams(keep_wav)[0], ANA_SR)
    elif opt.keep_engine == "wsola":
        log(f"time-stretching source keeping its pitch (WSOLA, +-{opt.wsola_tol_ms:g} ms, "
            f"up to +-{opt.wsola_max_ms:g} ms)...")
        raw_wav = os.path.join(tmp, "raw.wav")
        media.decode_to_wav(src_path, src_st, raw_wav, src_st.channels, src_st.sample_rate)
        engine = wsola.stretch(raw_wav, keep_wav, k, opt.wsola_tol_ms, opt.wsola_max_ms)
        if not opt.keep_temp:
            os.remove(raw_wav)
        keep16 = media.decode_mono(keep_wav, media.audio_streams(keep_wav)[0], ANA_SR)
    else:
        log("time-stretching source keeping its pitch (Rubber Band R3)...")
        engine = {"engine": "rubberband R3"}
        keep16 = media.make_intermediate(src_path, src_st, k, "keep", keep_wav, ANA_SR)
    lat, t, off = stretch_latency(inter16, keep16)
    dev = np.abs(off - lat) * 1000
    slope = float(np.polyfit(t, off, 1)[0]) if len(t) > 2 else 0.0
    log(f"  stretch latency {lat * 1000:+.3f} ms over {len(off)} windows: "
        f"median deviation {np.median(dev):.3f} ms, p95 {np.percentile(dev, 95):.3f} ms, "
        f"drift {slope * 1e6:+.2f} ppm")
    report["keep_pitch"] = {**engine, "latency_ms": lat * 1000,
                            "windows": len(off), "median_dev_ms": float(np.median(dev)),
                            "p95_dev_ms": float(np.percentile(dev, 95)), "drift_ppm": slope * 1e6}
    report["stretch_mode"] = "keep"
    for p in pieces:
        p["offset"] += lat
    return keep_wav


def log_segments(log, report):
    k = report["speed"]
    log(f"speed x{k:.7f} ({report['speed_fraction']}), mode {report['stretch_mode']}, "
        f"coverage {report['coverage_pct']:.1f}%")
    log("  #  ref start      ref end        src start      src end        offset(ms)   method  spread")
    for i, s in enumerate(report["segments"]):
        sp = f"{s['spread_ms']:.3f}" if s["spread_ms"] is not None else "-"
        log(f"  {i:<2} {fmt_t(s['ref_start'])}  {fmt_t(s['ref_end'])}  {fmt_t(s['src_start'])}  "
            f"{fmt_t(s['src_end'])}  {s['offset_s'] * 1000:11.3f}  {s['method']:<6}  {sp}")
    for g in report["gaps"]:
        log(f"  gap {fmt_t(g['start'])} - {fmt_t(g['end'])} ({g['end'] - g['start']:.2f} s, no source)")
