//! Streaming harmonic/percussive split (Fitzgerald's median filtering, Wiener masks).
//!
//! STFT with a Hann window of ~43 ms at 75 % overlap. A bin is harmonic where the magnitude is
//! steady over time (median over LT frames) and percussive where it is flat over frequency
//! (median over LF bins). The masks are computed on the channels' summed magnitude, so both
//! channels split the same way and the stereo image holds, and they sum to one, so
//! harmonic + percussive gives back the input exactly (to rounding). Both outputs are aligned
//! sample for sample with the input; the split lags the input by (LT / 2 + 3) hops.

use std::collections::VecDeque;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

const LT: usize = 17; // frames (~180 ms at 48 kHz)
const LF: usize = 17; // bins (~400 Hz at 48 kHz)

struct Frame {
    spec: Vec<Complex<f32>>, // ch * nb
    mag: Vec<f32>,           // nb, summed over channels
    pmed: Vec<f32>,          // nb, median over frequency
}

/// Inserts v into the sorted w (NaN sorts last, so it cannot break the order).
fn ins(w: &mut [f32], len: usize, v: f32) {
    let i = w[..len].partition_point(|x| x.total_cmp(&v).is_lt());
    w.copy_within(i..len, i + 1);
    w[i] = v;
}

/// Removes one v from the sorted w[..len].
fn del(w: &mut [f32], len: usize, v: f32) {
    let i = w[..len].partition_point(|x| x.total_cmp(&v).is_lt());
    w.copy_within(i + 1..len, i);
}

pub struct Hpss {
    ch: usize,
    nfft: usize,
    hop: usize,
    nb: usize,
    win: Vec<f32>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    inb: Vec<f32>,          // padded input not yet consumed by a frame, interleaved
    xin: VecDeque<f32>,     // input not yet output (percussive = input - harmonic)
    ring: VecDeque<Frame>,  // the last LT frames
    spare: Vec<Frame>,
    tsort: Vec<f32>,        // per bin, the ring's magnitudes sorted (LT slots each)
    tlen: usize,
    acc_h: Vec<f32>,        // overlap-add accumulator, nfft frames, interleaved
    skip: usize,            // leading padding frames still to drop from the output
    tmp: Vec<f32>,
    spec: Vec<Complex<f32>>,
    fwin: Vec<f32>,
    mh: Vec<f32>,
}

impl Hpss {
    pub fn new(ch: usize, sr: u32) -> Hpss {
        let nfft = (0.043 * sr as f64).log2().round().exp2() as usize;
        let hop = nfft / 4;
        let nb = nfft / 2 + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        let win = (0..nfft)
            .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / nfft as f64).cos()) as f32)
            .collect();
        let mut h = Hpss {
            ch,
            nfft,
            hop,
            nb,
            win,
            fwd: planner.plan_fft_forward(nfft),
            inv: planner.plan_fft_inverse(nfft),
            inb: vec![0.0; (nfft - hop) * ch],
            xin: VecDeque::new(),
            ring: VecDeque::with_capacity(LT + 1),
            spare: Vec::new(),
            tsort: vec![0.0; nb * LT],
            tlen: 0,
            acc_h: vec![0.0; nfft * ch],
            skip: nfft - hop,
            tmp: vec![0.0; nfft],
            spec: vec![Complex::new(0.0, 0.0); nb],
            fwin: vec![0.0; LF],
            mh: vec![0.0; nb],
        };
        // the frames before the start are silent; the first real frame is centred once
        // LT / 2 more have arrived
        for _ in 0..LT / 2 {
            h.ring.push_back(Frame {
                spec: vec![Complex::new(0.0, 0.0); ch * nb],
                mag: vec![0.0; nb],
                pmed: vec![0.0; nb],
            });
        }
        h.tlen = LT / 2; // all zeros already
        h
    }

    /// Feeds interleaved input; appends the outputs that are complete to `harm` and `perc`.
    pub fn push(&mut self, x: &[f32], harm: &mut Vec<f32>, perc: &mut Vec<f32>) {
        self.inb.extend_from_slice(x);
        self.xin.extend(x.iter().copied());
        let (ch, nfft, hop, nb) = (self.ch, self.nfft, self.hop, self.nb);
        while self.inb.len() >= nfft * ch {
            // analysis
            let mut fr = self.spare.pop().unwrap_or_else(|| Frame {
                spec: vec![Complex::new(0.0, 0.0); ch * nb],
                mag: vec![0.0; nb],
                pmed: vec![0.0; nb],
            });
            fr.mag.iter_mut().for_each(|v| *v = 0.0);
            for c in 0..ch {
                for i in 0..nfft {
                    self.tmp[i] = self.inb[i * ch + c] * self.win[i];
                }
                let _ = self.fwd.process(&mut self.tmp, &mut fr.spec[c * nb..(c + 1) * nb]);
                for b in 0..nb {
                    fr.mag[b] += fr.spec[c * nb + b].norm();
                }
            }
            self.inb.drain(..hop * ch);
            // median over frequency: a sorted window sliding along the bins
            let (half, w) = (LF / 2, &mut self.fwin);
            let mut len = 0;
            for b in 0..half.min(nb) {
                ins(w, len, fr.mag[b]);
                len += 1;
            }
            for b in 0..nb {
                if b + half < nb {
                    ins(w, len, fr.mag[b + half]);
                    len += 1;
                }
                fr.pmed[b] = w[len / 2];
                if b >= half {
                    del(w, len, fr.mag[b - half]);
                    len -= 1;
                }
            }
            // median over time: per bin, the ring's magnitudes kept sorted
            for b in 0..nb {
                ins(&mut self.tsort[b * LT..(b + 1) * LT], self.tlen, fr.mag[b]);
            }
            self.tlen += 1;
            self.ring.push_back(fr);
            if self.ring.len() < LT {
                continue;
            }
            // the centre frame's mask
            let f = &self.ring[LT / 2];
            for b in 0..nb {
                let hm = self.tsort[b * LT + LT / 2];
                let pm = f.pmed[b];
                let (h2, p2) = (hm * hm, pm * pm);
                self.mh[b] = if h2 + p2 > 1e-30 { h2 / (h2 + p2) } else { 1.0 };
            }
            // synthesis of the harmonic part (the percussive one is the input minus it)
            let scale = 1.0 / (1.5 * nfft as f32);
            for c in 0..ch {
                for b in 0..nb {
                    self.spec[b] = f.spec[c * nb + b] * self.mh[b];
                }
                let _ = self.inv.process(&mut self.spec, &mut self.tmp);
                for i in 0..nfft {
                    self.acc_h[i * ch + c] += self.tmp[i] * self.win[i] * scale;
                }
            }
            let old = self.ring.pop_front().unwrap();
            for b in 0..nb {
                del(&mut self.tsort[b * LT..(b + 1) * LT], self.tlen, old.mag[b]);
            }
            self.tlen -= 1;
            self.spare.push(old);
            // the first hop is final now
            let s = self.skip.min(hop);
            self.skip -= s;
            let out = &self.acc_h[s * ch..hop * ch];
            harm.extend_from_slice(out);
            perc.extend(out.iter().map(|h| self.xin.pop_front().unwrap_or(0.0) - h));
            self.acc_h.copy_within(hop * ch.., 0);
            self.acc_h[(nfft - hop) * ch..].iter_mut().for_each(|v| *v = 0.0);
        }
    }

    pub fn hop(&self) -> usize {
        self.hop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_sum_to_input() {
        let (ch, n) = (2usize, 100_003usize);
        let mut s = 1u32;
        let x: Vec<f32> = (0..n * ch)
            .map(|i| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                let t = (i / ch) as f32 / 48000.0;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                    + if (i / ch) % 9000 < 200 { noise } else { 0.0 }
            })
            .collect();
        let mut hp = Hpss::new(ch, 48000);
        let (mut h, mut p) = (Vec::new(), Vec::new());
        for blk in x.chunks(4096 * ch) {
            hp.push(blk, &mut h, &mut p);
        }
        let z = vec![0f32; hp.hop() * ch];
        while h.len() < x.len() {
            hp.push(&z, &mut h, &mut p);
        }
        let err = (0..x.len()).map(|i| (h[i] + p[i] - x[i]).abs()).fold(0f32, f32::max);
        assert!(err < 1e-4, "max error {err}");
        // the tone lands in the harmonic part, the clicks in the percussive one
        let e = |v: &[f32], a: usize, b: usize| v[a * ch..b * ch].iter().map(|s| s * s).sum::<f32>();
        assert!(e(&h, 20_000, 26_000) > 20.0 * e(&p, 20_000, 26_000));
        assert!(e(&p, 27_000, 27_200) > e(&h, 27_000, 27_200) * 0.5);
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    #[test]
    #[ignore]
    fn speed() {
        let (ch, sr, secs) = (2usize, 48000u32, 120usize);
        let mut s = 7u32;
        let x: Vec<f32> = (0..sr as usize * secs * ch)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let mut hp = Hpss::new(ch, sr);
        let (mut h, mut p) = (Vec::new(), Vec::new());
        let t = std::time::Instant::now();
        for blk in x.chunks(4096 * ch) {
            hp.push(blk, &mut h, &mut p);
            h.clear();
            p.clear();
        }
        let dt = t.elapsed().as_secs_f64();
        eprintln!("HPSS {secs} s stereo: {dt:.2} s ({:.0}x realtime)", secs as f64 / dt);
    }
}
