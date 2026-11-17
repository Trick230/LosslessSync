"""Command line interface."""
from __future__ import annotations

import argparse
import sys

from . import media
from .pipeline import Options, sync


def cmd_probe(args):
    for path in args.files:
        print(path)
        for s in media.audio_streams(path):
            print("  " + s.describe())


def cmd_sync(args):
    opt = Options(
        ref_track=args.ref_track, src_track=args.src_track,
        ref_langs=tuple(x.strip() for x in args.ref_lang.split(",") if x.strip()),
        speed=args.speed, pitch=args.pitch, fill=args.fill, fade_ms=args.fade_ms, codec=args.codec,
        bitrate=args.bitrate, language=args.language, mux=args.mux, report=args.report,
        keep_temp=args.keep_temp, tmp_dir=args.tmp, analyze_only=args.analyze_only)
    if not args.output and not args.analyze_only:
        sys.exit("error: -o/--output is required (or use --analyze-only)")
    sync(args.ref, args.src, args.output, opt)


def main(argv=None):
    p = argparse.ArgumentParser(prog="losslesssync",
                                description="Sync a dubbed audio track to a reference release "
                                            "using the shared music & effects.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="list audio tracks")
    pr.add_argument("files", nargs="+")
    pr.set_defaults(fn=cmd_probe)

    s = sub.add_parser("sync", help="sync SRC audio to the REF timeline")
    s.add_argument("--ref", required=True, help="reference file (Blu-ray mkv/m2ts or audio)")
    s.add_argument("--src", required=True, help="file with the audio to sync (TV recording)")
    s.add_argument("-o", "--output", help="output audio (.flac/.wav/.ac3/.eac3/.m4a/.mka)")
    s.add_argument("--ref-track", type=int, help="reference audio track (a:N, see probe)")
    s.add_argument("--ref-lang", default="eng,jpn", help="preferred reference languages")
    s.add_argument("--src-track", type=int, help="source audio track (a:N)")
    s.add_argument("--speed", default="auto",
                   help="auto | pal (25 from 23.976) | pal24 | ntsc | 1 | ratio like 1001/960")
    s.add_argument("--pitch", default="auto", choices=["auto", "resample", "tempo"],
                   help="resample = undo PAL speed-up incl. pitch; tempo = keep pitch")
    s.add_argument("--fill", default="silence", choices=["silence", "ref"],
                   help="what to put where the source has no matching content")
    s.add_argument("--fade-ms", type=float, default=40.0,
                   help="fade length (ms) where the track meets a gap (default 40)")
    s.add_argument("--codec", choices=list(media.CODECS), help="override output codec")
    s.add_argument("--bitrate", help="bitrate for lossy codecs, e.g. 448k")
    s.add_argument("--language", default="por", help="language tag of the output track")
    s.add_argument("--mux", help="also write an MKV: reference file + synced track")
    s.add_argument("--report", help="write a JSON report")
    s.add_argument("--analyze-only", action="store_true", help="only print the edit list")
    s.add_argument("--keep-temp", action="store_true")
    s.add_argument("--tmp", help="directory for temporary files (default: next to output)")
    s.set_defaults(fn=cmd_sync)

    g = sub.add_parser("gui", help="open the graphical interface")
    g.set_defaults(fn=lambda a: __import__("losslesssync.gui", fromlist=["main"]).main())

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except media.MediaError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
