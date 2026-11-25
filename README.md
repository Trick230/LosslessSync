# LosslessSync

Syncs a dubbed audio track, such as a TV recording, onto the timeline of a reference
release such as a Blu-ray.

Matching uses the music and effects that both dubs share, so the dialogue language does not
matter. The synced track keeps the source's channels and sample rate, and is written
losslessly (FLAC 24-bit) by default.

## Contents

- [What it handles](#what-it-handles)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Options](#options)
- [How it works](#how-it-works)
- [Output and report](#output-and-report)
- [Testing](#testing)
- [Development](#development)
- [Limitations](#limitations)

## What it handles

| Situation | Handling |
| --- | --- |
| PAL speed-up (25 vs 23.976 / 24 fps) and other common ratios | Detected and snapped to the exact fraction |
| Pitch kept or not | Detects whether the broadcaster resampled (pitch shifted) or time-stretched (pitch kept) |
| Commercial breaks, cut scenes, bumpers, truncated credits | The source is split into segments, each with its own offset |
| Parts at another speed (sped-up end credits, an intro from another release) | Each found in the holes the rest leaves and played at its own speed |
| Edits inside continuous source | The splice is located in the source and both edges are cut there |
| Opening titles | Detected and started without a fade-in |
| Signal dropouts (lost or muted audio), several in a row | Detected, reported, and synced across |
| Sub-millisecond offsets | GCC-PHAT, or onset flux for time-stretched sources |

Gaps where the source has no material are filled with silence, or with the reference audio
if you choose that. Edges next to a gap get a fade (300 ms by default). Joins where two
pieces touch get a 4 ms splice guard.

## Requirements

- Python 3.10 or newer
- `numpy` and `scipy`
- `ffmpeg` and `ffprobe` on `PATH` (with `librubberband` for the time-stretch mode)
- Tkinter, for the GUI (included with the standard Python installers)
- Optional, for speed: Rust, to build the native kernels (`cd dsp && cargo build --release`).
  `losslesssync` finds `dsp/target/release/lls-dsp` by itself (or `$LOSSLESSSYNC_DSP`;
  `LOSSLESSSYNC_DSP=off` forces the Python code). The WSOLA stretch is bit-identical and
  about 6x faster (T05E01 `--pitch keep`: 80 s -> 59 s end to end).
- Optional, for `--keep-engine stems`: a second Python (3.11 works) with
  `audio-separator[gpu]`, `python-stretch` and `soundfile`, ideally with CUDA torch. A venv
  keeps torch out of the main install. `scripts/setup.ps1` (Windows) or `scripts/setup.sh`
  (Linux) creates it in `~/.lls/venv-sep` and sets `LOSSLESSSYNC_EXT_PYTHON`; by hand:

  ```
  py -3.11 -m venv %USERPROFILE%\.lls\venv-sep
  %USERPROFILE%\.lls\venv-sep\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  %USERPROFILE%\.lls\venv-sep\Scripts\python -m pip install "audio-separator[gpu]" python-stretch soundfile
  %USERPROFILE%\.lls\venv-sep\Scripts\python -m pip install "triton-windows<3.10"
  ```

  Then pass `--ext-python %USERPROFILE%\.lls\venv-sep\Scripts\python.exe` (or set `LOSSLESSSYNC_EXT_PYTHON`).

  With Triton installed the model's repeated blocks are compiled (`torch.compile`):
  separation chunks run 2.15x faster on an RTX 4060 (2-episode batch 200 s -> 168-186 s),
  for ~15 s of compiling the first time and ~5 s after that (cached next to the models).

  The separation runs on CUDA (NVIDIA, or AMD with a ROCm torch build), else on an Intel
  GPU with a torch XPU build, else on the CPU (much slower). With `torch-directml`
  installed it would use DirectML, but audio-separator runs the RoFormer model on the CPU
  there (DirectML runs out of memory on it), so DirectML gives no speed-up.

  With `lls-dsp` built with `--features signalsmith`, the helper stretches the rest while
  the GPU is still separating the vocals, instead of afterwards (T05E01: 151 s -> 115-139 s;
  a 2-episode batch: 250 s -> 200 s, and it no longer holds the whole residual in memory).
  That feature compiles Signalsmith Stretch (C++), so it needs a C++ compiler and libclang;
  on Windows with the Rust GNU toolchain, a WinLibs GCC with LLVM works:

  ```
  set PATH=C:\path\to\mingw64\bin;%PATH%
  set LIBCLANG_PATH=C:\path\to\mingw64\bin
  cd dsp && cargo build --release --features signalsmith
  ```

## Installation

```
git clone https://github.com/Trick230/LosslessSync.git
cd LosslessSync
pip install -e .
```

## Usage

### List the audio tracks of a file

```
losslesssync probe episode.mkv tv.ts
```

### Sync

```
losslesssync sync --ref episode.mkv --src tv.ts -o pt.flac --report pt.json
```

### Sync and mux into a copy of the reference

```
losslesssync sync --ref episode.mkv --src tv.ts -o pt.flac --mux episode_pt.mkv
```

### Analyse only

Prints the edit list without rendering.

```
losslesssync sync --ref episode.mkv --src tv.ts --analyze-only
```

### Batch: a whole season

```
losslesssync batch --ref-dir season5 --out-dir synced -j 2
```

Pairs reference and source files by episode code in the name (`S05E01`, `T05E01`,
`5x01`). When both are in the same folder, the language word in the name (`EN` / `PT`)
tells them apart; otherwise use `--src-dir`, or `--ref-glob` / `--src-glob`. Each episode
gets `<source>_synced.flac`, a JSON and an HTML report and a log in the output folder,
plus `batch_summary.json` and `index.html` (all episodes, linking their pages) for the run. Episodes already done are skipped (`--overwrite` redoes
them), a failing episode does not stop the rest, and episodes with dropouts or a p95
error above 1 ms are listed for checking. `--dry-run` shows the pairing only. `-j 2` runs
two episodes at once (about 0.9 GB RAM each); the sync options above apply to every
episode. With `--keep-engine stems` and one job, the next episode's vocals separation
starts on the GPU as soon as the current episode's is done, so it runs while the current
one finishes on the CPU.

### Graphical interface

```
losslesssync gui
```

Or `python -m losslesssync gui`.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--ref` | required | Reference file (Blu-ray mkv/m2ts, or audio) |
| `--src` | required | File with the audio to sync (TV recording) |
| `-o`, `--output` | | Output audio: `.flac`, `.wav`, `.ac3`, `.eac3`, `.m4a`, `.mka` |
| `--ref-track N` | auto | Reference audio track (`a:N`, see `probe`) |
| `--ref-lang` | `eng,jpn` | Preferred reference languages |
| `--src-track N` | auto | Source audio track (`a:N`) |
| `--speed` | `auto` | `auto`, `pal`, `1`, or an exact ratio such as `1001/960` |
| `--pitch` | `auto` | `auto`, `resample`, `tempo` or `keep` (cut on the resampled source, render it time-stretched so the original PAL pitch is kept on the film timing) |
| `--keep-engine` | `auto` | stretcher for `--pitch keep` (`auto`: `stems` when `--ext-python` or `LOSSLESSSYNC_EXT_PYTHON` is set, else `wsola`): `wsola` (built in; no phase-vocoder smearing, timing error bounded by `--wsola-max-ms`) `rubberband` (R3), or `stems`: the vocals are separated (RoFormer, GPU) and stretched with WSOLA, the rest with Signalsmith Stretch, then summed; the two always add back to the source |
| `--ext-python` | `$LOSSLESSSYNC_EXT_PYTHON`, else this Python | Python with `audio-separator` and `python-stretch`, for `stems` |
| `--sep-model` | `vocals_mel_band_roformer.ckpt` | Vocals model for `stems` |
| `--vocal-engine` | `wsola` | `stems`: stretcher for the vocals, `wsola` or `psola` (`lls-dsp psola`: repeats or drops whole pitch periods; lower latency, but a worse p95 sync error on the test episode) |
| `--hpss` | off | `stems`: split the rest into tonal parts (Signalsmith) and attacks (short-frame WSOLA, keeps them sharp); needs `lls-dsp` built with `signalsmith`. With `--vocal-engine psola`: p95 1.20 vs 1.30 ms, but a 2-episode batch takes ~10 s longer |
| `--rest-block-ms`, `--rest-interval-ms` | `120`, `30` | `stems`: Signalsmith block and hop for the rest |
| `--sep-model-dir` | `~/.cache/losslesssync/models` | Where separation models are stored (downloaded on first use) |
| `--wsola-tol-ms` | `5` | WSOLA search tolerance: events stay within this of their exact place |
| `--wsola-max-ms` | `10` | WSOLA widens its search up to this only for a clearly better splice (fewer artefacts; set it equal to `--wsola-tol-ms` for a hard bound) |
| `--no-match-loudness` | off | Keep the source's level. By default one constant gain gives the synced track the reference's integrated loudness (EBU R128, measured over the synced parts only); no compression or limiting, and capped so the peaks stay below -0.1 dBFS (-1 dBFS for lossy codecs) |
| `--verify-deep` | off | `stems`: also separate the reference's vocals, so the check after rendering can measure windows where both dialogues hide the shared sound (S01E01: 184 instead of 171 of 251 windows); about a minute more on the GPU |
| `--fill` | `silence` | What fills gaps: `silence` or `ref` |
| `--fade-ms` | `300` | Fade at the edges of gaps |
| `--fade-opening` | off | Also fade into the opening titles |
| `--codec` | from extension | Override the output codec |
| `--bitrate` | | Bitrate for lossy codecs, for example `448k` |
| `--language` | `por` | Language tag of the output track |
| `--mux` | | Also write an MKV: the reference file plus the synced track |
| `--report` | | Write a report: JSON, or an HTML page if the name ends in `.html` |
| `--analyze-only` | off | Only print the edit list |
| `--keep-temp` | off | Keep intermediate files |
| `--tmp` | next to output | Directory for temporary files |

## How it works

1. **Decode.** Both tracks are decoded to mono 16 kHz for analysis. The original audio is
   only used for the final render.
2. **Speed.** Candidate frame-rate ratios are scored, and the winner is refined to an exact
   fraction. The source is then speed-corrected, either by resampling or by time-stretching.
3. **Anchors.** Short chunks of the source are matched over the whole reference. Confident
   matches are grouped into segments of constant offset.
4. **Fine offsets.** Each segment's offset is refined with GCC-PHAT. Time-stretched sources
   have no phase coherence, so they use onset flux at 1 ms resolution instead.
5. **Dropouts.** The offset is tracked on a fine grid between segments. Small forward jumps
   (lost audio) become cuts of their own, and offsets that step back are rejected as false
   matches. Spans where the source is digital silence but the reference is not are reported
   as muted.
6. **Cut points.** Cuts between segments are placed where the similarity to the reference
   changes. Where the source runs on across a cut, the edit is located in the source itself.
7. **Parts at another speed.** Holes of 4 s or more left in the reference are searched again
   at relative speeds from x0.75 to x1.34 against the source the neighbours leave unused;
   a match gets its own speed (refined with GCC-PHAT) and its own sped-up copy of the source.
8. **Render.** Pieces are copied from the speed-corrected source at their offsets, with fades
   at gaps. The result is then measured against the reference and encoded.

### Source layout

| Path | What is there |
|---|---|
| `losslesssync/app/` | Command line, GUI, batch mode, HTML report |
| `losslesssync/sync/` | `pipeline` (the steps above), `features`, `speed` (speed, anchors, pitch mode), `segments` (fine offsets, drift), `gaps` (dropouts, muted spans), `cuts` (cut points), `respeed` (parts at another speed) |
| `losslesssync/audio/` | ffmpeg decoding and encoding, WAV I/O, rendering |
| `losslesssync/stretch/` | Pitch-keeping stretchers: WSOLA, PSOLA, separated stems and their helper (`ext`), the native kernels (`native`) |
| `dsp/` | `lls-dsp`, the native WSOLA/PSOLA/HPSS/Signalsmith kernels |
| `tools/` | Synthetic test material, checks and stretch-quality scoring |

## Output and report

The log lists:

- the detected speed and pitch mode
- every segment (reference range, source range, offset, method)
- every gap
- every signal dropout
- the opening titles
- a final timing check against the reference

`--report` writes the same information as JSON (segments, gaps, `dropouts`,
`opening_segment` and the per-window timing check). A name ending in `.html` writes a
self-contained page instead: what to review (dropouts, windows off by more than 1 ms, low
coverage), a timeline of segments, cuts, gaps and dropouts, a plot of the timing error per
window, and the tables.

## Testing

`tools/make_synth.py` builds synthetic material with known ground truth. `tools/check.py`
compares a result against it. The variants are `pal`, `tempo` and `none`; `--dropouts` adds
a series of lost and muted dropouts, and `--ad`, `--recap`, `--intro`, `--credits` and
`--channels` add an ad break, a recap, a different intro, sped-up credits and other TV
layouts. `tools/run_synth.py` runs all thirteen cases end to end and prints PASS/FAIL per
case:

```
python tools/run_synth.py [test_out] [--cases pal,drop_pal]
```

One case by hand:

```
python tools/make_synth.py test_out/pal --variant pal
losslesssync sync --ref test_out/pal/ref.mka --src test_out/pal/tv.ts -o test_out/pal/out.flac --report test_out/pal/report.json
python tools/check.py test_out/pal
```

`tools/stretch_quality.py SOURCE [engine ...]` scores the artefacts of the pitch-keeping
stretchers (`wsola[:tol[:frame_ms[:key=value...]]]`, `rubberband`, `signalsmith`, `atempo`): four
excerpts are stretched and stretched back, and the round trip is compared with the
original on a log-mel spectrogram, forgiving +-10 ms of displacement. Lower is better.

## Development

The project runs in three places: Windows, Linux (CachyOS) and a GPU-less cloud VM.
`scripts/setup.ps1` / `scripts/setup.sh` set each one up (`-Check` / `--check` shows the
state). `gpu/run_gpu.py` sends stem separation to a free cloud GPU, and
`tools/media_sync.py` syncs a whole folder, or a Google Drive folder through rclone.
CI runs the synthetic cases on every push. Workflow, rules and commands are in
[CLAUDE.md](CLAUDE.md) (Portuguese).

## Limitations

- **Time-stretched sources.** A source time-stretched with the pitch kept has no phase
  coherence, so timing is limited to about 1 ms. The original, unstretched recording syncs
  sample-exact.
- **Dropouts in time-stretched sources.** In these sources, the moment of a lost-audio
  dropout is only located approximately. Its length is still correct.
