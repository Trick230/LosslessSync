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
- [Limitations](#limitations)

## What it handles

| Situation | Handling |
| --- | --- |
| PAL speed-up (25 vs 23.976 / 24 fps) and other common ratios | Detected and snapped to the exact fraction |
| Pitch kept or not | Detects whether the broadcaster resampled (pitch shifted) or time-stretched (pitch kept) |
| Commercial breaks, cut scenes, bumpers, truncated credits | The source is split into segments, each with its own offset |
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
- Optional, for `--keep-engine stems`: a second Python (3.11 works) with
  `audio-separator[gpu]`, `python-stretch` and `soundfile`, ideally with CUDA torch. A venv
  keeps torch out of the main install:

  ```
  py -3.11 -m venv .venv-sep
  .venv-sep\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
  .venv-sep\Scripts\pip install "audio-separator[gpu]" python-stretch soundfile
  ```

  Then pass `--ext-python .venv-sep\Scripts\python.exe` (or set `LOSSLESSSYNC_EXT_PYTHON`).

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
episode.

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
| `--keep-engine` | `wsola` | stretcher for `--pitch keep`: `wsola` (built in; no phase-vocoder smearing, timing error bounded by `--wsola-max-ms`) `rubberband` (R3), or `stems`: the vocals are separated (RoFormer, GPU) and stretched with WSOLA, the rest with Signalsmith Stretch, then summed; the two always add back to the source |
| `--ext-python` | `$LOSSLESSSYNC_EXT_PYTHON` | Python with `audio-separator` and `python-stretch`, for `stems` |
| `--sep-model` | `vocals_mel_band_roformer.ckpt` | Vocals model for `stems` |
| `--sep-model-dir` | `~/.cache/losslesssync/models` | Where separation models are stored (downloaded on first use) |
| `--wsola-tol-ms` | `5` | WSOLA search tolerance: events stay within this of their exact place |
| `--wsola-max-ms` | `10` | WSOLA widens its search up to this only for a clearly better splice (fewer artefacts; set it equal to `--wsola-tol-ms` for a hard bound) |
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
7. **Render.** Pieces are copied from the speed-corrected source at their offsets, with fades
   at gaps. The result is then measured against the reference and encoded.

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

`tests/make_synth.py` builds synthetic material with known ground truth. `tests/check.py`
compares a result against it. The variants are `pal`, `tempo` and `none`, and `--dropouts`
adds a series of lost and muted dropouts.

```
python tests/make_synth.py test_out/pal --variant pal
losslesssync sync --ref test_out/pal/ref.mka --src test_out/pal/tv.ts -o test_out/pal/out.flac --report test_out/pal/report.json
python tests/check.py test_out/pal
```

`tests/stretch_quality.py SOURCE [engine ...]` scores the artefacts of the pitch-keeping
stretchers (`wsola[:tol[:frame_ms[:key=value...]]]`, `rubberband`, `signalsmith`, `atempo`): four
excerpts are stretched and stretched back, and the round trip is compared with the
original on a log-mel spectrogram, forgiving +-10 ms of displacement. Lower is better.

## Limitations

- **Time-stretched sources.** A source time-stretched with the pitch kept has no phase
  coherence, so timing is limited to about 1 ms. The original, unstretched recording syncs
  sample-exact.
- **Dropouts in time-stretched sources.** In these sources, the moment of a lost-audio
  dropout is only located approximately. Its length is still correct.
