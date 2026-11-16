# LosslessSync

Syncs a dubbed audio track (e.g. a PT-PT TV recording) onto the timeline of a reference
release (Blu-ray, EN or JP track). Matching uses the music & effects shared by both dubs,
so the dialogue language does not matter.

What it handles automatically:

- PAL speed-up (25 vs 23.976 fps) and other common ratios, snapped to the exact fraction;
  detects whether the broadcaster kept the pitch-shift (resample) or time-stretched (tempo).
- Commercial breaks, cut scenes, bumpers, truncated credits: the source is split into
  segments with their own offset; cut points are placed with 10 ms resolution, joins get
  4 ms crossfades, missing parts are filled with silence (or the reference audio).
- Sub-millisecond offsets (GCC-PHAT); the result is measured against the reference.
- Source channels and sample rate are kept; output is FLAC 24-bit by default.

Requires Python 3.10+, numpy, scipy, and `ffmpeg`/`ffprobe` on PATH
(with librubberband for the tempo mode).

```
pip install -e .
losslesssync probe episode.mkv tv.ts
losslesssync sync --ref episode.mkv --src tv.ts -o pt.flac --report pt.json
losslesssync sync --ref episode.mkv --src tv.ts -o pt.flac --mux episode_pt.mkv
```

Useful options: `--ref-track N` / `--src-track N` (a:N from `probe`), `--speed pal|1|1001/960`,
`--pitch resample|tempo`, `--fill ref`, `--codec ac3 --bitrate 448k`, `--analyze-only`.

Tests (synthetic material with known ground truth):

```
python tests/make_synth.py test_out/pal --variant pal
losslesssync sync --ref test_out/pal/ref.mka --src test_out/pal/tv.ts -o test_out/pal/out.flac --report test_out/pal/report.json
python tests/check.py test_out/pal
```
