"""Unit tests of the pure building blocks (fast, no media, no lls-dsp): `lls test` runs them
next to the synthetic cases."""
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from losslesssync.app.batch import _lang, episode_key, needs_check, summarize  # noqa: E402
from losslesssync.audio.render import render  # noqa: E402
from losslesssync.audio.wavio import WavWriter, read_info  # noqa: E402
from losslesssync.sync.cuts import best_cut  # noqa: E402
from losslesssync.sync.features import frame_positions, mel_filterbank, pool  # noqa: E402
from losslesssync.sync.gaps import offset_runs  # noqa: E402
from losslesssync.sync.segments import FLUX_BLOCK, FLUX_HOP, Segment, _flux_bands, fine_measure, flux_slice  # noqa: E402
from losslesssync.sync.speed import Anchor  # noqa: E402


class WavIO(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "a.wav")

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip(self):
        x = np.random.default_rng(0).standard_normal((1000, 2)).astype(np.float32)
        w = WavWriter(self.path, 1000, 2, 48000, mask=3)
        w.write(x[:600])
        w.write(x[600:])
        w.close()
        info = read_info(self.path)
        self.assertEqual((info.frames, info.channels, info.sr, info.mask), (1000, 2, 48000, 3))
        m = info.memmap()
        np.testing.assert_array_equal(m, x)
        del m                                # Windows: the map holds the file open

    def test_close_pads_with_silence(self):
        w = WavWriter(self.path, 500, 1, 8000)
        w.write(np.ones((100, 1), np.float32))
        w.close()
        info = read_info(self.path)
        m = info.memmap()
        self.assertEqual(info.frames, 500)
        self.assertEqual(float(m[:100].sum()), 100.0)
        self.assertEqual(float(np.abs(m[100:]).sum()), 0.0)
        del m

    def test_write_past_length(self):
        w = WavWriter(self.path, 10, 1, 8000)
        with self.assertRaises(ValueError):
            w.write(np.zeros((11, 1), np.float32))
        w.close()

    def test_rejects_int_pcm(self):
        fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
        data = b"\0" * 20
        body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
        with open(self.path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", len(body)) + body)
        with self.assertRaises(ValueError):
            read_info(self.path)

    def test_rejects_non_wav(self):
        with open(self.path, "wb") as f:
            f.write(b"not a wav file at all")
        with self.assertRaises(ValueError):
            read_info(self.path)


class Render(unittest.TestCase):
    def test_butt_splice_keeps_continuous_sound(self):
        # two pieces of one source with the same offset: the splice's cross-fade must give
        # the source back exactly (no dip to silence)
        with tempfile.TemporaryDirectory() as d:
            sr, n = 8000, 8000
            x = np.random.default_rng(1).standard_normal((n, 2)).astype(np.float32)
            src, out = os.path.join(d, "src.wav"), os.path.join(d, "out.wav")
            w = WavWriter(src, n, 2, sr)
            w.write(x)
            w.close()
            pieces = [{"start": 0.0, "end": 0.5, "offset": 0.0},
                      {"start": 0.5, "end": 1.0, "offset": 0.0}]
            render(read_info(src), pieces, n, out)
            m = read_info(out).memmap()
            np.testing.assert_allclose(m, x, atol=1e-6)
            del m
            # an offset jump at the splice: each side reads its own source mapping
            pieces[1]["offset"] = -0.1           # the second piece reads 800 frames later
            render(read_info(src), pieces, n, out)
            m = read_info(out).memmap()
            s, h = 4000, 16                      # splice frame, splice_ms / 2 at 8 kHz
            np.testing.assert_allclose(m[:s - h], x[:s - h], atol=1e-6)
            np.testing.assert_allclose(m[s + h:n - 800], x[s + h + 800:], atol=1e-6)
            t = (np.arange(2 * h) + 0.5) / (2 * h)
            up = (0.5 - 0.5 * np.cos(np.pi * t))[:, None]
            np.testing.assert_allclose(m[s - h:s + h], x[s - h:s + h] * (1 - up)
                                       + x[s - h + 800:s + h + 800] * up, atol=1e-5)
            del m
            # a gap filled with the same signal: the pieces' fades cross-fade into the fill,
            # so the source comes back exactly
            pieces = [{"start": 0.0, "end": 0.4, "offset": 0.0},
                      {"start": 0.6, "end": 1.0, "offset": 0.0}]
            render(read_info(src), pieces, n, out, fill=read_info(src), fade_ms=50.0)
            m = read_info(out).memmap()
            np.testing.assert_allclose(m, x, atol=1e-6)
            del m


class BestCut(unittest.TestCase):
    # similarity per frame: A matches the first part, B the last; tau scores the gap state
    def test_clean_switch(self):
        sA = np.r_[np.ones(40), np.zeros(60)]
        sB = np.r_[np.zeros(40), np.ones(60)]
        self.assertEqual(best_cut(sA, sB), (40, 40))

    def test_expected_gap_is_kept(self):
        sA = np.r_[np.ones(40), np.zeros(60)]
        sB = np.r_[np.zeros(50), np.ones(50)]
        self.assertEqual(best_cut(sA, sB, expected_gap=10), (40, 50))

    def test_gap_never_shorter_than_expected(self):
        sA = np.r_[np.ones(50), np.zeros(50)]
        sB = np.r_[np.zeros(50), np.ones(50)]
        c1, c2 = best_cut(sA, sB, expected_gap=8)
        self.assertGreaterEqual(c2 - c1, 8)

    def test_only_one_side(self):
        # a gap pays off only once tau * its length beats gap_pen (0.08 * 60 > 3)
        s = np.r_[np.zeros(60), np.ones(40)]
        self.assertEqual(best_cut(None, s), (0, 60))
        self.assertEqual(best_cut(s[::-1].copy(), None), (40, 100))
        short = np.r_[np.zeros(30), np.ones(70)]
        self.assertEqual(best_cut(None, short), (0, 0))

    def test_gap_longer_than_window(self):
        self.assertEqual(best_cut(np.ones(10), np.ones(10), expected_gap=20), (0, 10))


class OffsetRuns(unittest.TestCase):
    def test_one_outlier_does_not_split(self):
        obs = [(t, 0.100) for t in range(10)] + [(10, 0.200)] + [(t, 0.100) for t in range(11, 20)]
        runs = offset_runs(obs)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0]), 19)

    def test_two_levels(self):
        obs = [(t, 0.100) for t in range(8)] + [(t, 0.150) for t in range(8, 16)]
        runs = offset_runs(obs)
        self.assertEqual([len(r) for r in runs], [8, 8])

    def test_short_run_dropped(self):
        obs = [(t, 0.100) for t in range(8)] + [(t, 0.300) for t in range(8, 11)]
        self.assertEqual([len(r) for r in offset_runs(obs)], [8])


class Features(unittest.TestCase):
    def test_frame_positions(self):
        np.testing.assert_array_equal(frame_positions(1000, 100.0), np.arange(10) * 100)
        np.testing.assert_array_equal(frame_positions(1000, 100.0, speed=2.0)[:3], [0, 50, 100])
        self.assertEqual(len(frame_positions(10, 100.0, start=50.0)), 0)

    def test_pool(self):
        F = np.arange(14, dtype=float).reshape(7, 2)
        np.testing.assert_array_equal(pool(F, 3), [[2.0, 3.0], [8.0, 9.0]])

    def test_mel_filterbank_covers_every_band(self):
        fb = mel_filterbank(16000, 512, 40, 50.0, 8000.0)
        self.assertEqual(fb.shape, (40, 257))
        self.assertTrue((fb.sum(axis=1) > 0).all())
        self.assertTrue((fb >= 0).all() and (fb <= 1).all())


class SegmentCore(unittest.TestCase):
    def test_core_spans_anchor_centres(self):
        s = Segment(anchors=[Anchor(10.0, 12.0, 0.9, 2.0), Anchor(20.0, 22.0, 0.9, 2.0)])
        self.assertEqual(s.core(1.0, 4.0), (14.0, 24.0))
        self.assertAlmostEqual(s.coarse_offset(1.0), 2.0)


class FineFlux(unittest.TestCase):
    def test_slice_across_blocks_matches_direct(self):
        x = np.random.default_rng(1).standard_normal((FLUX_BLOCK + 500) * FLUX_HOP + 256)
        f0, n = FLUX_BLOCK - 300, 700
        seg = x[f0 * FLUX_HOP:(f0 + n - 1) * FLUX_HOP + 256].astype(np.float32)
        win, W = _flux_bands(256, 8)
        frames = np.lib.stride_tricks.sliding_window_view(seg, 256)[::FLUX_HOP][:n]
        B = np.log(np.abs(np.fft.rfft(frames * win, axis=1)) @ W + 1e-5).T
        F = np.maximum(np.diff(B, axis=1), 0)
        F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-9)
        np.testing.assert_allclose(flux_slice(x, f0, n), F, atol=1e-3)
        self.assertIsNone(flux_slice(x, -1, n))

    def test_flux_fallback_off_grid_offset(self):
        # Same onsets, unrelated waveforms (no phase coherence): only the flux can measure.
        # The true offset is off the 1 ms grid, and the guess is 4 ms off.
        sr, D = 16000, 0.0123
        rng = np.random.default_rng(2)
        ref, src = np.zeros(20 * sr), np.zeros(20 * sr)
        env = np.exp(-np.arange(800) / 120.0)
        for t in np.sort(rng.uniform(0.5, 19.0, 60)):
            for y, at in ((ref, t), (src, t - D)):
                i = int(round(at * sr))
                y[i:i + 800] += rng.standard_normal(800) * env
        res = fine_measure(ref, src, 5.0, D + 0.004, 6.0, max_lag=0.03, phat_thr=1e9)
        self.assertIsNotNone(res)
        self.assertEqual(res[2], "flux")
        self.assertAlmostEqual(res[0], D, delta=0.0005)


class Batch(unittest.TestCase):
    def test_episode_key(self):
        self.assertEqual(episode_key("Show S01E02 PT.mkv"), "s1e2")
        self.assertEqual(episode_key("show.t1.e12.eng.flac"), "s1e12")
        self.assertEqual(episode_key("Show 2x05 dobrado.ts"), "s2e5")
        self.assertEqual(episode_key("O Filme Português.mkv"), "o filme")

    def test_lang(self):
        self.assertEqual(_lang("/x/Show S01E01 ENG.mkv"), "ref")
        self.assertEqual(_lang("/x/Show S01E01 PT.ts"), "src")
        self.assertIsNone(_lang("/x/Show S01E01.mkv"))

    def test_needs_check(self):
        self.assertEqual(needs_check({"status": "error"}, 5.0), ["failed, see log"])
        self.assertEqual(needs_check({"status": "ok", "median_ms": 0.1, "p95_ms": 1.0}, 5.0), [])
        why = needs_check({"status": "ok", "dropouts": 2, "median_ms": 0.1, "p95_ms": 9.0}, 5.0)
        self.assertEqual(why, ["2 dropout(s)", "p95 9.00 ms"])
        self.assertEqual(needs_check({"status": "ok"}, 5.0), ["not verified"])

    def test_summarize(self):
        r = summarize({"coverage_pct": 99.5, "gaps": [1, 2], "dropouts": None,
                       "verify": {"median_abs_ms": 0.01, "p95_abs_ms": 0.5}})
        self.assertEqual((r["coverage_pct"], r["gaps"], r["dropouts"], r["median_ms"], r["p95_ms"]),
                         (99.5, 2, 0, 0.01, 0.5))


if __name__ == "__main__":
    unittest.main()
