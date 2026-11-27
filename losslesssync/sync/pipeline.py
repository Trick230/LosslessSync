"""End-to-end sync: analyse, correct speed, segment, render, verify."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import fft as sp_fft

from .. import MODELS
from ..app import report_html
from ..audio import media, wavio
from ..audio.render import render
from ..audio.wavio import read_info
from ..stretch import signalsmith, stems, wsola
from . import cuts, gaps, respeed, segments, speed
from .features import ANA_SR, features, pmap

SPEED_ALIASES = {"pal": 1001 / 960, "pal24": 25 / 24, "ntsc": 960 / 1001, "1": 1.0, "none": 1.0}


@dataclass
class Options:
    ref_track: int | None = None
    src_track: int | None = None
    ref_langs: tuple[str, ...] = ("eng", "jpn")
    speed: str = "auto"          # auto | pal | pal24 | ntsc | 1 | <float> | <a/b>
    pitch: str = "auto"          # auto | resample | tempo | keep
    keep_engine: str = "auto"    # pitch=keep: auto (signalsmith if lls-dsp has it, else
                                 # wsola) | signalsmith | wsola | rubberband | stems
    wsola_tol_ms: float = 5.0    # WSOLA search tolerance (bounds the timing error)...
    wsola_max_ms: float = 10.0
    ext_python: str = ""          # Python with audio-separator and python-stretch (stems)
    sep_model: str = "vocals_mel_band_roformer.ckpt"
    vocal_engine: str = "wsola"  # stems: vocals stretcher, psola | wsola
    hpss: bool = False           # stems: split the rest into tonal (Signalsmith) and attacks (WSOLA)
    rest_block_ms: float = 0.0   # stems: Signalsmith block and interval (0: its default 120/30)
    rest_interval_ms: float = 0.0
    sep_model_dir: str = MODELS   # ...widened up to this for a clearly better splice
    fill: str = "silence"        # silence | ref | me (the reference without its vocals)
    fade_ms: float = 300.0        # fade at edges next to a gap
    fade_opening: bool = False   # also fade into the opening titles
    dropout_fade_ms: float = 10.0  # fade at the edges of a signal dropout
    match_loudness: bool = True  # one constant gain so the track is as loud as the reference
    verify_deep: bool = False    # stems: verify also on the reference without its vocals
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


def verify_window(ref16: np.ndarray, out16: np.ndarray, t: float, win: float = 6.0,
                  sub: float = 3.0, off: float = 0.0) -> float | None:
    """Timing error (s) of out16 against ref16 over [t, t + win], or None. One correlation
    can lock onto a wrong peak (periodic music, the other language's dialogue: up to ~100 ms
    on a perfectly placed track), so the whole window and three overlapping halves are
    measured and only an error that at least three of them agree on is kept. out16 is
    placed at ref time - off."""
    oq = round(off * ANA_SR) / ANA_SR     # on the sample grid: the windows are cut exactly
    rs = [segments.fine_measure(ref16, out16, t, oq, win, max_lag=0.1)]
    rs += [segments.fine_measure(ref16, out16, u, oq, sub, max_lag=0.1)
           for u in np.linspace(t, t + win - sub, 3)]
    v = [r[0] - off for r in rs if r is not None]
    best: list = []
    for c in v:
        g = [x for x in v if abs(x - c) <= max(0.001, 0.1 * abs(c))]
        if len(g) > len(best):
            best = g
    return float(np.median(best)) if len(best) >= 3 else None


def verify(ref16: np.ndarray, out16: np.ndarray, pieces: list[dict], win: float = 6.0,
           step: float = 5.0, rest16: np.ndarray | None = None,
           ref_rest16: np.ndarray | None = None) -> dict:
    """rest16: the source without its vocals, in the pieces' source timing (offset). Where
    the reference's dialogue (another language) hides the shared sound, the windows the
    mix could not measure are measured again on it, and then (ref_rest16, the reference
    without its vocals) on both tracks without their vocals."""
    tp = [(t, p) for p in pieces
          for t in np.arange(p["start"] + 1.0, p["end"] - 1.0 - win + 1e-9, step)]
    ts = [t for t, _ in tp]
    res = pmap(lambda t: verify_window(ref16, out16, t, win), ts)
    n_rest = 0
    if rest16 is not None:
        # (a piece at its own speed has no rest16 timing)
        miss = [i for i, r in enumerate(res) if r is None and "speed" not in tp[i][1]]
        alt = pmap(lambda i: verify_window(ref16, rest16, tp[i][0], win,
                                           off=tp[i][1]["offset"]), miss)
        for i, r in zip(miss, alt):
            if r is not None:
                res[i] = r
                n_rest += 1
        if ref_rest16 is not None:
            miss = [i for i in miss if res[i] is None]
            alt = pmap(lambda i: verify_window(ref_rest16, rest16, tp[i][0], win,
                                               off=tp[i][1]["offset"]), miss)
            for i, r in zip(miss, alt):
                if r is not None:
                    res[i] = r
                    n_rest += 1
    # window centre and signed error (ms), None where nothing could be measured
    points = [[round(float(t) + win / 2, 3), None if r is None else round(r * 1000, 4)]
              for t, r in zip(ts, res)]
    errs = [r for r in res if r is not None]
    if not errs:
        return {"windows": len(ts), "measured": 0, "points": points}
    e = np.abs(np.array(errs)) * 1000
    extra = {"measured_on_rest": n_rest} if rest16 is not None else {}
    return {"windows": len(ts), "measured": len(errs), "points": points, **extra,
            "median_abs_ms": float(np.median(e)), "p95_abs_ms": float(np.percentile(e, 95)),
            "max_abs_ms": float(e.max()), "within_1ms_pct": float((e <= 1.0).mean() * 100)}


def unmeasured_spans(points: list, win: float = 6.0) -> list[tuple[float, float]]:
    """Runs of verify windows that could not be measured, as (start, end) in seconds."""
    spans: list = []
    for c, e in points:
        if e is not None:
            continue
        a, b = c - win / 2, c + win / 2
        if spans and a <= spans[-1][1]:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


NO_MATCH = ("the source shares no audio with the reference: wrong file, episode or audio "
            "track (see --src-track)?")
# below this much of the reference covered AND of the source used, the pair is refused (a
# truncated recording uses most of its source; another episode shares only its titles)
LOW_MATCH_PCT = 40.0


def uses_stems(out_path, opt: Options) -> bool:
    """Whether a sync separates the source (so the separation can start right away)."""
    return bool(out_path and not opt.analyze_only and opt.pitch == "keep"
                and opt.keep_engine == "stems")


def sync(ref_path: str, src_path: str, out_path: str | None, opt: Options,
         log_fn=print, progress=None, split=None) -> dict:
    """split: a separation already started with stems.start_split (batch mode starts the
    next episode's while this one finishes); it is killed when the sync ends."""
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
    ref_sep = None
    try:
        if split:
            log("vocals separation already started in the background")
        elif uses_stems(out_path, opt):
            split = stems.start_split(src_path, src_st, tmp, opt, log)
        if split and opt.verify_deep:
            ref_sep = stems.RefSeparation(ref_path, ref_st, tmp, opt, split[0])
        return _sync(ref_path, src_path, out_path, opt, log, prog, ref_streams, ref_st,
                     src_st, tmp, split, ref_sep)
    finally:
        if ref_sep:
            ref_sep.close()
        if split:
            split[0].kill()
        if opt.keep_temp:
            log(f"temp files kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def _sync(ref_path, src_path, out_path, opt, log, prog, ref_streams, ref_st, src_st, tmp,
          split=None, ref_sep=None):
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
        k, speed_scores = speed.detect_speed(src16, R, log)
    else:
        k, speed_scores = forced, {}
    log(f"  using x{k:.7f} for anchor search")

    log("matching anchors over the whole reference...")
    S = features(src16, speed=k)
    act = speed.active_mask(src16, len(S), k)
    anchors = speed.find_anchors(S, R, k, act, opt.chunk_s, opt.step_s,
                                 progress=lambda f: prog("anchors", f))
    n_conf = sum(speed.is_confident(a) for a in anchors)
    log(f"  {len(anchors)} anchors, {n_conf} confident")
    if n_conf < 2:
        raise media.MediaError(NO_MATCH)
    del S

    if forced is None:
        k_ref, snapped = speed.refine_speed(anchors, k)
        log(f"  speed refined to x{k_ref:.7f}" + (" (exact ratio)" if snapped else ""))
        k = k_ref

    if opt.pitch == "auto":
        mode, pitch_info = speed.detect_pitch_mode(ref16, src16, anchors, k, opt.chunk_s)
        if abs(k - 1) > 1e-4:
            what = ("speed-up with pitch shift -> resample" if mode == "resample"
                    else "pitch already corrected -> time-stretch")
            log(f"  pitch: {what}"
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
        segs = segments.build_segments(anchors, k, opt.chunk_s)
        log(f"  {len(segs)} segments; refining offsets (GCC-PHAT)...")
        segments.fine_segments(segs, ref16, inter16, k, opt.chunk_s,
                               progress=lambda f: prog("fine", f))
        slope, n, span = segments.residual_drift(segs)
        log(f"  residual drift {slope * 1e6:+.2f} ppm over {n} windows")
        if forced is None and abs(slope) > 1e-6 and n >= 6 and span >= 60 and it < 2:
            k *= 1 + slope
            continue
        break
    segs = segments.split_on_jumps(segs, k, opt.chunk_s)
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
    segs = segments.drop_ref_nested(segs, k, opt.chunk_s)
    if dropped:
        log(f"  dropped {len(dropped)} unconfirmed segment(s)")
    if not segs:
        raise media.MediaError(NO_MATCH)
    segs = segments.merge_equal(segs, k, opt.chunk_s)

    log("placing cut points...")
    Rb = features(ref16, hp_s=0.15)
    Sb = features(inter16, hp_s=0.15)
    segs = gaps.split_on_dropouts(segs, ref16, inter16, Rb, Sb, k, opt.chunk_s, ref_dur,
                                  progress=lambda f: prog("dropouts", f))
    segs = segments.merge_equal(segs, k, opt.chunk_s)
    segs = cuts.place_segments(segs, Rb, Sb, ref_dur, opt.chunk_s, k, ref16, inter16)
    del Rb, Sb
    segs = cuts.refine_edges(segs, inter16, ref_dur, ref16)
    log("looking for parts at another speed...")
    extra = respeed.find(segs, ref16, inter16, ref_dur, log, stretched=mode != "resample")
    # pieces, dropouts and the opening after respeed: an edge next to such a part may move
    pieces = [{"start": s.ref_start, "end": s.ref_end, "offset": s.offset} for s in segs]
    drops = sorted(gaps.dropout_gaps(segs, ref16))
    for i in drops:
        pieces[i]["fade_out_ms"] = opt.dropout_fade_ms
        pieces[i + 1]["fade_in_ms"] = opt.dropout_fade_ms
    dropouts = [{"start": segs[i].ref_end, "end": segs[i + 1].ref_start, "kind": "lost"}
                for i in drops]
    dropouts += [{"start": a, "end": b, "kind": "muted"}
                 for a, b in gaps.muted_spans(segs, ref16, inter16)]
    dropouts.sort(key=lambda d: d["start"])
    if dropouts:
        log(f"  {len(dropouts)} signal dropout(s) in the recording:")
        for d in dropouts:
            what = "audio lost" if d["kind"] == "lost" else "muted"
            log(f"    {fmt_t(d['start'])} - {fmt_t(d['end'])}  "
                f"({(d['end'] - d['start']) * 1000:.0f} ms, {what})")
    opening = None if opt.fade_opening else cuts.find_opening(segs, ref16, inter16, ref_dur)
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
    rows = [{"ref_start": s.ref_start, "ref_end": s.ref_end,
             "src_start": (s.ref_start - s.offset) / k, "src_end": (s.ref_end - s.offset) / k,
             "offset_s": s.offset, "method": s.method, "spread_ms": s.spread_ms,
             "anchors": len(s.anchors), "fine_windows": len(s.fine)} for s in segs]
    rows += [{"ref_start": x["ref_start"], "ref_end": x["ref_end"],
              "src_start": x["src_start"] / k,
              "src_end": (x["src_start"] + (x["ref_end"] - x["ref_start"]) / x["speed"]) / k,
              "offset_s": x["ref_start"] - x["src_start"], "method": "speed",
              "speed": x["speed"], "spread_ms": None, "anchors": 0,
              "fine_windows": x["windows"]} for x in extra]
    for r in sorted(rows, key=lambda r: r["ref_start"]):
        if r["ref_start"] - prev > 0.01:
            report["gaps"].append({"start": prev, "end": r["ref_start"]})
        prev = r["ref_end"]
        covered += r["ref_end"] - r["ref_start"]
        report["segments"].append(r)
    if ref_dur - prev > 0.01:
        report["gaps"].append({"start": prev, "end": ref_dur})
    report["coverage_pct"] = covered / ref_dur * 100
    src_dur = len(src16) / ANA_SR
    report["source_used_pct"] = covered / k / src_dur * 100

    log_segments(log, report)
    if report["coverage_pct"] < LOW_MATCH_PCT and report["source_used_pct"] < LOW_MATCH_PCT:
        raise media.MediaError(
            f"the source matches only {report['coverage_pct']:.0f}% of the reference and only "
            f"{report['source_used_pct']:.0f}% of the source is used: probably another episode "
            f"(shared titles or credits) or the wrong file")

    if not opt.analyze_only and out_path:
        pool = ThreadPoolExecutor(1)
        ref_loud = (pool.submit(media.loudness, ref_path, ref_st,
                                [(p["start"], p["end"]) for p in pieces]
                                + [(x["ref_start"], x["ref_end"]) for x in extra])
                    if opt.match_loudness else None)     # while the source is stretched
        pool.shutdown(wait=False)
        try:
            rest16 = None
            if opt.pitch == "keep" and abs(k - 1) > 1e-4:
                inter_wav, rest16 = keep_pitch(src_path, src_st, k, inter16, pieces, tmp, log,
                                               report, opt, split)
            info = read_info(inter_wav)
        except BaseException:
            if ref_loud:                  # do not leave its ffmpeg running behind the error
                futures_wait([ref_loud])
            raise
        codec = opt.codec or media.EXT_CODEC.get(os.path.splitext(out_path)[1].lower(), "flac")
        gain = (match_loudness(ref_loud, inter_wav, info, pieces, extra, codec, log, report)
                if ref_loud else 1.0)
        n_out = int(round(ref_dur * info.sr))
        out_pieces = pieces + extra_pieces(src_path, src_st, k, extra, mode, opt, tmp, info, log)
        out_pieces.sort(key=lambda p: p["start"])
        fill = None
        if opt.fill in ("ref", "me"):
            fill_wav = os.path.join(tmp, "fill.wav")
            media.decode_to_wav(ref_path, ref_st, fill_wav, info.channels, info.sr)
            if opt.fill == "me" and not stems.remove_vocals(
                    fill_wav, fill_regions(out_pieces, n_out, info.sr, opt.fade_ms), tmp, opt, log):
                log("  --fill me: no gaps, or more than 2 channels (cannot separate): "
                    "gaps left silent")
                fill_wav = None
            fill = read_info(fill_wav) if fill_wav else None
        log("rendering...")
        prog("render", 0.0)
        out_wav = os.path.join(tmp, "out.wav")
        render(info, out_pieces, n_out, out_wav, fill, fade_ms=opt.fade_ms, gain=gain)
        # the encoder runs on one core: it works while verify measures the same file
        enc_pool = ThreadPoolExecutor(1)
        enc = enc_pool.submit(media.encode, out_wav, out_path, codec, opt.bitrate, opt.language)
        enc_pool.shutdown(wait=False)
        try:
            log("verifying result against the reference...")
            prog("verify", 0.0)
            out_st = media.audio_streams(out_wav)[0]
            out16 = media.decode_mono(out_wav, out_st, ANA_SR)
            ref_rest16 = None
            if opt.verify_deep and rest16 is not None:
                got = ref_sep.result() if ref_sep else None
                if got:
                    log("  the reference's vocals were separated while the source was stretched")
                    raw, voc = got
                else:
                    log("separating the reference's vocals for verify...")
                    raw, voc = stems.separate_vocals(ref_path, ref_st, tmp, opt, "ref_sep")
                ref_rest16 = (media.decode_mono(raw, media.audio_streams(raw)[0], ANA_SR)
                              - media.decode_mono(voc, media.audio_streams(voc)[0], ANA_SR))
            elif opt.verify_deep:
                log("  --verify-deep needs --pitch keep with --keep-engine stems: skipped")
            report["verify"] = verify(ref16, out16, out_pieces, rest16=rest16,
                                      ref_rest16=ref_rest16)
            del rest16, ref_rest16
            v = report["verify"]
            if v.get("measured"):
                on_rest = (f" ({v['measured_on_rest']} on the source without its vocals)"
                           if v.get("measured_on_rest") else "")
                log(f"  {v['measured']}/{v['windows']} windows measured{on_rest}: median "
                    f"{v['median_abs_ms']:.3f} ms, p95 {v['p95_abs_ms']:.3f} ms, "
                    f"max {v['max_abs_ms']:.3f} ms, {v['within_1ms_pct']:.1f}% within 1 ms")
            um = unmeasured_spans(v["points"])
            if um:
                log(f"  not measurable (too little shared sound): {len(um)} stretch(es), "
                    f"{sum(b - a for a, b in um):.0f} s in all")
        except BaseException:
            futures_wait([enc])     # no encoder left writing behind the error
            if enc.exception() is None and os.path.exists(out_path):
                os.remove(out_path)
            raise
        log(f"encoding {codec} -> {out_path}")
        prog("encode", 0.0)
        enc.result()
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


# the gain never takes the source's sample peak above this; lossy codecs overshoot it
# between samples, so they get more room
HEADROOM_DB = {"flac": 0.1, "wav": 0.1}
HEADROOM_LOSSY_DB = 1.0


def _peak_dbfs(info) -> float:
    """Sample peak of a whole WAV (the pieces plus the few samples around them that the
    render splices in)."""
    m, peak = info.memmap(), 0.0
    for a in range(0, len(m), 1 << 20):
        peak = max(peak, float(np.abs(m[a:a + (1 << 20)]).max(initial=0.0)))
    return 20 * np.log10(peak) if peak > 0 else -200.0


def extra_pieces(src_path, src_st, k, extra, mode, opt, tmp, info, log) -> list[dict]:
    """Render pieces for the parts at their own speed (respeed.find), each from its own WAV
    of the source corrected by k in the main mode, then resampled by speed (in keep mode
    with Rubber Band, its latency measured against the resampled piece)."""
    out = []
    keep = opt.pitch == "keep" and abs(k - 1) > 1e-4
    sr = src_st.sample_rate
    for i, x in enumerate(extra):
        c = x["speed"]
        ia = x["src_start"] - 1.0
        ib = x["src_start"] + (x["ref_end"] - x["ref_start"]) / c + 1.0
        n0, n1 = max(0, int(np.floor(ia / k * sr))), int(np.ceil(ib / k * sr))
        wav = os.path.join(tmp, f"piece{i}.wav")
        media.make_piece(src_path, src_st, n0, n1, k, c, "keep" if keep else mode, wav)
        pi = read_info(wav)
        if pi.sr != info.sr or pi.channels != info.channels:
            raise media.MediaError(f"piece {i}: layout {pi.channels}ch/{pi.sr} Hz differs")
        lat = 0.0
        if keep:
            res_wav = os.path.join(tmp, f"piece{i}_res.wav")
            media.make_piece(src_path, src_st, n0, n1, k, c, "resample", res_wav)
            a16 = media.decode_mono(res_wav, media.audio_streams(res_wav)[0], ANA_SR)
            b16 = media.decode_mono(wav, media.audio_streams(wav)[0], ANA_SR)
            ts = np.arange(1.0, min(len(a16), len(b16)) / ANA_SR - 4.0, 1.0)
            got = [r[0] for r in (segments.fine_measure(a16, b16, t, 0.0, 3.0, max_lag=0.2)
                                  for t in ts) if r is not None]
            if len(got) >= 3:
                lat = float(np.median(got))
            else:
                log(f"  piece {i}: stretch latency not measurable, taken as 0")
        # its sample 0 is the corrected source at n0 / sr * k; ref = offset + that * c
        out.append({"start": x["ref_start"], "end": x["ref_end"], "src": pi, "speed": c,
                    "offset": x["offset"] + (n0 / sr * k - x["src_offset"]) * c + lat})
    return out


def match_loudness(ref_loud, inter_wav: str, info, pieces, extra, codec: str, log, report) -> float:
    """The constant gain that gives the synced track the reference's integrated loudness,
    both measured over the synced pieces only (the parts at their own speed included). One
    multiplier for the whole track: no compression or limiting, so the dynamics and the
    sound stay the original's. Capped so it cannot push peaks into clipping."""
    try:
        # the source is measured while the reference's measure (a pool job) still runs
        try:
            src_loud = media.loudness(inter_wav, media.audio_streams(inter_wav)[0],
                                      [(p["start"] - p["offset"], p["end"] - p["offset"])
                                       for p in pieces]
                                      + [(x["src_start"], x["src_start"]
                                          + (x["ref_end"] - x["ref_start"]) / x["speed"])
                                         for x in extra])
        except BaseException:
            futures_wait([ref_loud])  # do not leave its ffmpeg running behind the error
            raise
        ref_loud = ref_loud.result()
    except media.MediaError as e:
        log(f"  loudness: {str(e).splitlines()[0][:200]}")
        ref_loud = src_loud = None
    if not ref_loud or not src_loud:
        log("  loudness: could not measure, gain left at 0 dB")
        report["loudness"] = {"gain_db": 0.0, "measured": False}
        return 1.0
    (ref_i, _), (src_i, _) = ref_loud, src_loud
    src_peak = _peak_dbfs(info)
    want = ref_i - src_i
    room = max(-HEADROOM_DB.get(codec, HEADROOM_LOSSY_DB) - src_peak, 0.0)
    gain_db = min(want, room)
    log(f"  loudness: reference {ref_i:.1f} LUFS, source {src_i:.1f} LUFS (peak {src_peak:.1f} "
        f"dBFS) -> gain {gain_db:+.2f} dB"
        + (f" (capped from {want:+.2f} dB so the peaks do not clip)" if gain_db < want else ""))
    report["loudness"] = {"measured": True, "reference_lufs": ref_i, "source_lufs": src_i,
                          "source_peak_dbfs": src_peak, "wanted_db": want, "gain_db": gain_db,
                          "capped": gain_db < want, "output_lufs": src_i + gain_db}
    return 10 ** (gain_db / 20)


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
    res = pmap(lambda t: segments.fine_measure(inter16, x16, t, 0.0, 6.0, max_lag=0.2), ts)
    ok = [(t, r[0]) for t, r in zip(ts, res) if r is not None]
    if len(ok) < 10:
        raise media.MediaError("could not measure the time-stretch latency")
    t, off = (np.array(x) for x in zip(*ok))
    return float(np.median(off)), t, off


def fill_regions(pieces: list[dict], n_out: int, sr: int, fade_ms: float,
                 margin_s: float = 2.0) -> list[tuple[int, int]]:
    """Frame ranges of the output not covered by pieces (the gaps a fill shows through),
    widened by the fades into them and a margin for the separation's context, merged."""
    pad = int(round((fade_ms * 1e-3 + margin_s) * sr))
    out: list[tuple[int, int]] = []
    cur = 0
    for p in sorted(pieces, key=lambda q: q["start"]):
        n0 = min(max(int(round(p["start"] * sr)), cur), n_out)
        if n0 > cur:
            out.append((cur, n0))
        cur = max(cur, min(int(round(p["end"] * sr)), n_out))
    if cur < n_out:
        out.append((cur, n_out))
    merged: list[tuple[int, int]] = []
    for a, b in out:
        a, b = max(0, a - pad), min(n_out, b + pad)
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def keep_pitch(src_path, src_st, k, inter16, pieces, tmp, log, report, opt, split=None):
    """Stretch the source to the film timing keeping its pitch (opt.keep_engine) and
    shift the pieces by the stretcher's latency, measured against the resampled source.
    Returns the stretched WAV and, with stems, the analysis copy of everything but the
    vocals in its timing (for verify), else None."""
    rest16 = None
    keep_wav = os.path.join(tmp, "keep.wav")
    kind = opt.keep_engine
    if kind == "auto":
        kind = "signalsmith" if signalsmith.available() else "wsola"
    if kind == "stems":
        log("time-stretching source keeping its pitch (separated stems)...")
        split = split or stems.start_split(src_path, src_st, tmp, opt, log)
        n_out = int(round(wavio.read_info(split[1]).frames * k))
        dec = {}
        def measure(path, step):
            dec[path] = media.decode_mono(path, media.audio_streams(path)[0], ANA_SR)
            return stretch_latency(inter16, dec[path], step)[0]
        parts, engine = stems.stretch(split, tmp, k, opt, log, measure)
        (voc_k, lat_v), others = parts[0], parts[1:]
        # latency > 0: content early; every stem is lined up with the vocals
        shifts = [int(round((lat - lat_v) * src_st.sample_rate)) for _, lat in others]
        log("  stem latencies: " + ", ".join(
            f"{name} {lat * 1000:+.3f} ms" for name, (_, lat) in zip(engine["order"], parts)))
        for name, (_, lat) in zip(engine["order"], parts):
            engine[name]["latency_ms"] = lat * 1000
        stems.mix(voc_k, [(p, s) for (p, _), s in zip(others, shifts)], keep_wav, n_out)
        if not opt.keep_temp:
            for p, _ in parts:
                os.remove(p)
        # the analysis copy of the mix, from the stem decodes (the resampler is linear)
        # instead of decoding keep.wav again; the other stems get their fractional shift
        # exactly (band-limited, in the frequency domain)
        n16 = int(round(n_out * ANA_SR / src_st.sample_rate))
        rest16 = np.zeros(n16, np.float32)
        for (p, _), shift in zip(others, shifts):
            r16 = dec.pop(p, None)
            if r16 is None:                   # not decoded when its measurement failed
                r16 = media.decode_mono(p, media.audio_streams(p)[0], ANA_SR)
            d = shift * ANA_SR / src_st.sample_rate
            nf = sp_fft.next_fast_len(max(n16, len(r16)) + int(abs(d)) + 64, real=True)
            R16 = sp_fft.rfft(r16, nf, workers=-1)
            del r16
            R16 *= np.exp(-2j * np.pi * np.fft.rfftfreq(nf) * d)
            rest16 += sp_fft.irfft(R16, nf, workers=-1)[:n16].astype(np.float32)
            del R16
        v16 = dec.pop(voc_k)
        keep16 = rest16.copy()
        keep16[:min(n16, len(v16))] += v16[:n16]
        del v16
    elif kind == "signalsmith":
        log("time-stretching source keeping its pitch (Signalsmith Stretch)...")
        raw_wav = os.path.join(tmp, "raw.wav")
        media.decode_to_wav(src_path, src_st, raw_wav, src_st.channels, src_st.sample_rate)
        engine = signalsmith.stretch(raw_wav, keep_wav, k)
        if not opt.keep_temp:
            os.remove(raw_wav)
        keep16 = media.decode_mono(keep_wav, media.audio_streams(keep_wav)[0], ANA_SR)
    elif kind == "wsola":
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
    return keep_wav, rest16


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
