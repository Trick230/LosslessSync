<div align="center">

# LosslessSync

**Automatic, sample-accurate sync of a dubbed audio track onto another release's timeline.**

[![ci](https://github.com/Trick230/LosslessSync/actions/workflows/ci.yml/badge.svg)](https://github.com/Trick230/LosslessSync/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-0078d4)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

Take a dub from a TV recording, and LosslessSync lines it up with a Blu-ray: it corrects
the speed, works around commercial breaks and edits, and writes the result losslessly. It matches
on the music and effects both versions share, so the dialogue language does not matter.

```console
$ lls sync --ref episode.mkv --src tv.ts -o dub.flac
  reference: flac 6ch 48000 Hz lang=eng dur=7.0 min
  source   : mp2 2ch 48000 Hz lang=por dur=7.7 min
  speed refined to x1.0427083 (exact ratio)
  pitch: speed-up with pitch shift -> resample
  speed x1.0427083 (1001/960), mode resample, coverage 94.3%
  #  ref start    ref end      src start    src end      offset(ms)  method
  0  0:00:00.000  0:02:30.000  0:00:07.682  0:02:31.538   -8010.467  phat
  1  0:02:30.000  0:04:09.993  0:03:43.466  0:05:19.364  -83010.442  phat
  2  0:04:13.993  0:06:39.999  0:05:19.364  0:07:39.389  -79010.445  phat
  gap 0:04:09.993 - 0:04:13.993 (4.00 s, no source)
  76/76 windows measured: median 0.027 ms, p95 0.029 ms, 100.0% within 1 ms
  encoding flac -> dub.flac
done in 18 s
```

<sub>Abridged output on the synthetic PAL test case: a 25 fps broadcast with an ad break and a
cut scene, found and synced automatically.</sub>

## Highlights

- **Sub-millisecond accuracy.** Each segment is aligned with GCC-PHAT and the result is measured
  against the reference before it is written.
- **Hands-off.** Speed (PAL 25 vs 23.976 fps and other ratios), ad breaks, cut scenes, sped-up
  credits and signal dropouts are detected automatically.
- **Lossless.** Keeps the source's channels and sample rate; FLAC 24-bit by default, or mux
  straight into an MKV.
- **Original pitch, if you want it.** Can render on the new timing while keeping the
  broadcaster's pitch, optionally with GPU source separation for the cleanest result.
- **Whole seasons.** Batch mode pairs episodes by name and writes an HTML report per episode.
- **Self-contained.** One command sets everything up inside the project folder. Nothing is
  installed on the system.

## Accuracy

| Material | Timing error against the reference |
| --- | --- |
| Synthetic PAL case with ad break and edits (CI) | median 0.03 ms, max 0.03 ms |
| Real TV episode, resampled | median 0.01 ms, max 0.09 ms |
| Real TV episode, original pitch kept (separated stems) | p95 1.3 ms |

CI runs 13 synthetic cases with known ground truth on every push, plus a real-episode
regression check.

## Installation

```sh
git clone https://github.com/Trick230/LosslessSync.git
cd LosslessSync
./lls setup        # Windows: lls setup
```

`lls setup` downloads pinned, checksummed tools (Python, ffmpeg with Rubber Band, native DSP
kernels) into `.lls/`. The folder can be moved or copied. Windows and Linux, x64 and ARM64.

Optional: `./lls setup --stems` adds GPU source separation (NVIDIA, 3-5 GB).

## Usage

```sh
lls probe episode.mkv tv.ts                                    # list the audio tracks
lls sync --ref episode.mkv --src tv.ts -o dub.flac --report dub.html
lls sync --ref episode.mkv --src tv.ts -o dub.flac --mux episode_dub.mkv
lls sync --ref episode.mkv --src tv.ts --pitch keep -o dub.flac   # keep the original pitch
lls batch --ref-dir season5 --out-dir synced -j 2              # a whole season
lls gui                                                        # graphical interface
```

| Option | Default | Description |
| --- | --- | --- |
| `--speed` | `auto` | `auto`, `pal`, `1`, or an exact ratio such as `1001/960` |
| `--pitch` | `auto` | `resample`, `tempo`, or `keep` (original pitch on the new timing) |
| `--keep-engine` | `auto` | stretcher for `keep`: `wsola`, `rubberband`, or `stems` (GPU) |
| `--fill` | `silence` | gaps: `silence` or `ref` |
| `--codec`, `--bitrate` | from extension | `.flac`, `.wav`, `.ac3`, `.eac3`, `.m4a`, `.mka` |
| `--ref-track`, `--src-track` | auto | pick tracks (see `probe`) |

All options, batch pairing rules, the report format and the pipeline in detail:
**[docs/GUIDE.md](docs/GUIDE.md)**, or `lls sync --help`.

## How it works

1. **Speed:** candidate frame-rate ratios are scored and refined to an exact fraction.
2. **Anchors:** short chunks of the source are matched across the reference and grouped
   into segments of constant offset.
3. **Fine offsets:** each segment is refined with GCC-PHAT, or onset flux for time-stretched
   sources.
4. **Cuts and dropouts:** cut points go where similarity to the reference changes; lost or
   muted spans are found on a fine grid.
5. **Render:** pieces are copied from the original-quality source, verified against the
   reference, and encoded.

## Development

```sh
lls test           # unit tests + quick end-to-end cases
lls test all       # all 13 synthetic cases
lls lint           # ruff + mypy
```

| Path | Contents |
| --- | --- |
| `losslesssync/sync/` | the analysis pipeline |
| `losslesssync/audio/`, `losslesssync/stretch/` | decoding, rendering, pitch-keeping stretchers |
| `losslesssync/app/` | CLI, GUI, batch, reports |
| `dsp/` | native kernels in Rust (`lls-dsp`) |
| `tools/`, `tests/` | synthetic test material, checks, tests |

Tests use synthetic material only, so no copyrighted media is needed. Cloud features (remote
jobs, cloud GPU separation) are optional and need external tooling.

## Limitations

- Sources time-stretched with the pitch kept have no phase coherence, so their timing is
  limited to about 1 ms; unstretched sources sync sample-exact.
- In those sources, a dropout's position is approximate; its length is still correct.
- No macOS build (no static ffmpeg with Rubber Band for Apple Silicon).

## Acknowledgements

[FFmpeg](https://ffmpeg.org), [Rubber Band](https://breakfastquay.com/rubberband/),
[Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch),
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) with the
Mel-Band RoFormer vocals model, NumPy, SciPy and [uv](https://github.com/astral-sh/uv).

## License

Released under the [MIT License](LICENSE).
