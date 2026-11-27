# LosslessSync

[![ci](https://github.com/Trick230/LosslessSync/actions/workflows/ci.yml/badge.svg)](https://github.com/Trick230/LosslessSync/actions/workflows/ci.yml)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-0078d4)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Automatic, sample-accurate sync of a dubbed audio track onto another release's timeline.**

Take a dub from a TV recording, and LosslessSync lines it up with a Blu-ray: it corrects
the speed, works around commercial breaks and edits, and writes the result losslessly. It matches
on the music and effects both versions share, so the dialogue language does not matter.

- **Sub-millisecond accuracy:** median alignment error 0.01 ms on a real episode
- **No manual work:** speed, cuts, gaps and dropouts are found automatically
- **Lossless:** keeps the source's channels and sample rate; FLAC 24-bit by default
- **Self-contained:** one command sets everything up inside the project folder

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Development](#development)
- [Limitations](#limitations)
- [License](#license)

## Features

| Problem | How it's handled |
| --- | --- |
| PAL speed-up (25 vs 23.976 fps) and other ratios | Detected and snapped to the exact fraction |
| Pitch shifted or pitch kept | Detects which one the broadcaster did; can render with the original pitch kept (`--pitch keep`) |
| Commercial breaks, cut scenes, truncated credits | The source is split into segments, each with its own offset |
| Parts at another speed (sped-up credits, a different intro) | Found and played at their own speed |
| Signal dropouts (lost or muted audio) | Detected, reported and synced across |
| Gaps with no source material | Filled with silence or with the reference audio, with fades |

Each run can also write a JSON or HTML report: the edit list, a timeline and a per-window
timing check against the reference.

## Quick start

```sh
git clone https://github.com/Trick230/LosslessSync.git
cd LosslessSync
./lls setup        # Windows: lls setup
./lls sync --ref episode.mkv --src tv.ts -o dub.flac
```

`lls setup` downloads pinned, checksummed tools (Python, ffmpeg with Rubber Band, native DSP
kernels) into `.lls/`. Nothing is installed on the system, and the folder can be moved or
copied. Runs on Windows and Linux, x64 and ARM64.

Optional: `./lls setup --stems` adds GPU source separation (NVIDIA, 3-5 GB) for the
highest-quality pitch-kept output.

## Usage

```sh
lls probe episode.mkv tv.ts                                    # list the audio tracks
lls sync --ref episode.mkv --src tv.ts -o dub.flac --report dub.html
lls sync --ref episode.mkv --src tv.ts -o dub.flac --mux episode_dub.mkv   # also mux into an MKV
lls sync --ref episode.mkv --src tv.ts --analyze-only          # edit list only
lls batch --ref-dir season5 --out-dir synced -j 2              # a whole season
lls gui                                                        # graphical interface
```

`batch` pairs files by episode code (`S05E01`, `5x01`), skips episodes already done and
writes an index page listing any episode worth checking.

Key options (`lls sync --help` for all):

| Option | Default | Description |
| --- | --- | --- |
| `--speed` | `auto` | `auto`, `pal`, `1`, or an exact ratio such as `1001/960` |
| `--pitch` | `auto` | `resample`, `tempo`, or `keep` (original pitch on the new timing) |
| `--keep-engine` | `auto` | stretcher for `keep`: `wsola`, `rubberband`, or `stems` (GPU) |
| `--fill` | `silence` | gaps: `silence` or `ref` |
| `--codec`, `--bitrate` | from extension | `.flac`, `.wav`, `.ac3`, `.eac3`, `.m4a`, `.mka` |
| `--ref-track`, `--src-track` | auto | pick tracks (see `probe`) |

## How it works

1. **Speed:** candidate frame-rate ratios are scored and refined to an exact fraction.
2. **Anchors:** short chunks of the source are matched across the reference and grouped
   into segments of constant offset.
3. **Fine offsets:** each segment is refined with GCC-PHAT (or onset flux for
   time-stretched sources).
4. **Cuts and dropouts:** cut points are placed where similarity to the reference changes;
   lost or muted spans are detected on a fine grid.
5. **Render:** pieces are copied from the original-quality source, measured against the
   reference, and encoded.

| Path | Contents |
| --- | --- |
| `losslesssync/sync/` | the analysis pipeline |
| `losslesssync/audio/`, `losslesssync/stretch/` | decoding, rendering, pitch-keeping stretchers |
| `losslesssync/app/` | CLI, GUI, batch, reports |
| `dsp/` | native kernels in Rust (`lls-dsp`) |
| `tools/`, `tests/` | synthetic test material, checks, tests |

## Development

```sh
lls test           # unit tests + quick end-to-end cases
lls test all       # all 13 synthetic cases (ad breaks, dropouts, sped-up credits, ...)
lls lint           # ruff + mypy
```

Tests run on synthetic material with known ground truth, so no copyrighted media is needed.
CI builds the native kernels and runs the tests on Windows and Linux, x64 and ARM64, plus a
real-episode regression check. Cloud features (remote jobs, cloud GPU separation) are optional
and need external tooling.

## Limitations

- Sources time-stretched with the pitch kept have no phase coherence, so their timing is
  limited to about 1 ms; unstretched sources sync sample-exact.
- In those sources, a dropout's position is approximate; its length is still correct.
- No macOS build (no static ffmpeg with Rubber Band for Apple Silicon).

## License

MIT, see [LICENSE](LICENSE).
