//! Native kernels for LosslessSync, driven by the Python package (losslesssync/native.py).
//!
//!   lls-dsp wsola IN OFFSET FRAMES CHANNELS OUT N_OUT K N D0 D MARGIN STEREO
//!
//! Reads FRAMES interleaved float32 frames from IN at byte OFFSET, writes N_OUT frames of
//! the WSOLA time stretch by K, appended to OUT (Python has already written the WAV
//! header). Same algorithm as losslesssync/wsola.py: frame N (hop N/2), nominal position
//! round((m*H + H)/K) - H, splice searched over +-D, leaving +-D0 only when the
//! normalised correlation is better by more than MARGIN. Prints the shift statistics
//! (median |shift|, max |shift| in samples, fraction beyond D0) on stdout.

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::process::exit;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

struct Input {
    x: Vec<f32>,
    n: i64,
    ch: usize,
}

impl Input {
    /// Input [a, a + len) into `out` as (len, ch) or mono mean (len, 1), zero outside.
    fn mid(&self, a: i64, len: usize, stereo: bool, out: &mut [f64]) {
        let w = if stereo { self.ch } else { 1 };
        out[..len * w].iter_mut().for_each(|v| *v = 0.0);
        let lo = a.max(0);
        let hi = (a + len as i64).min(self.n);
        for t in lo..hi {
            let src = &self.x[t as usize * self.ch..(t as usize + 1) * self.ch];
            let dst = (t - a) as usize * w;
            if stereo {
                for c in 0..self.ch {
                    out[dst + c] = src[c] as f64;
                }
            } else {
                out[dst] = src.iter().map(|&v| v as f64).sum::<f64>() / self.ch as f64;
            }
        }
    }
}

fn read_input(path: &str, offset: u64, frames: usize, ch: usize) -> Input {
    let mut f = File::open(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    f.seek(SeekFrom::Start(offset)).unwrap();
    let mut bytes = vec![0u8; frames * ch * 4];
    f.read_exact(&mut bytes).unwrap_or_else(|e| die(&format!("{path}: {e}")));
    let x = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Input { x, n: frames as i64, ch }
}

fn die(msg: &str) -> ! {
    eprintln!("lls-dsp: {msg}");
    exit(2)
}

struct Corr {
    nfft: usize,
    fwd: Arc<dyn RealToComplex<f64>>,
    inv: Arc<dyn ComplexToReal<f64>>,
    buf: Vec<f64>,
    spec_s: Vec<Complex<f64>>,
    spec_t: Vec<Complex<f64>>,
    acc: Vec<Complex<f64>>,
    out: Vec<f64>,
    scratch_f: Vec<Complex<f64>>,
    scratch_i: Vec<Complex<f64>>,
}

impl Corr {
    fn new(len: usize) -> Corr {
        let nfft = len.next_power_of_two();
        let mut p = RealFftPlanner::<f64>::new();
        let fwd = p.plan_fft_forward(nfft);
        let inv = p.plan_fft_inverse(nfft);
        let spec = fwd.make_output_vec();
        Corr {
            nfft,
            buf: fwd.make_input_vec(),
            spec_s: spec.clone(),
            spec_t: spec.clone(),
            acc: spec,
            out: inv.make_output_vec(),
            scratch_f: fwd.make_scratch_vec(),
            scratch_i: inv.make_scratch_vec(),
            fwd,
            inv,
        }
    }

    /// c[j] = sum_c sum_i seg[j + i, c] * tpl[i, c] for j in 0..=seg_len - tpl_len.
    fn run(&mut self, seg: &[f64], seg_len: usize, tpl: &[f64], tpl_len: usize, w: usize,
           c: &mut [f64]) {
        self.acc.iter_mut().for_each(|v| *v = Complex::new(0.0, 0.0));
        for ch in 0..w {
            self.buf.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..seg_len {
                self.buf[i] = seg[i * w + ch];
            }
            self.fwd.process_with_scratch(&mut self.buf, &mut self.spec_s, &mut self.scratch_f).unwrap();
            self.buf.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..tpl_len {
                self.buf[i] = tpl[i * w + ch];
            }
            self.fwd.process_with_scratch(&mut self.buf, &mut self.spec_t, &mut self.scratch_f).unwrap();
            for ((a, s), t) in self.acc.iter_mut().zip(&self.spec_s).zip(&self.spec_t) {
                *a += s * t.conj();
            }
        }
        // the inverse wants real DC and Nyquist bins
        let last = self.acc.len() - 1;
        self.acc[0].im = 0.0;
        self.acc[last].im = 0.0;
        self.inv.process_with_scratch(&mut self.acc, &mut self.out, &mut self.scratch_i).unwrap();
        let scale = 1.0 / self.nfft as f64;
        for (j, v) in c.iter_mut().enumerate() {
            *v = self.out[j] * scale;
        }
    }
}

fn wsola(args: &[String]) {
    if args.len() != 12 {
        die("wsola IN OFFSET FRAMES CHANNELS OUT N_OUT K N D0 D MARGIN STEREO");
    }
    let p = |i: usize| -> f64 { args[i].parse().unwrap_or_else(|_| die(&format!("bad arg {}", args[i]))) };
    let (in_path, out_path) = (&args[0], &args[4]);
    let offset = p(1) as u64;
    let frames = p(2) as usize;
    let ch = p(3) as usize;
    let n_out = p(5) as usize;
    let k = p(6);
    let nf = p(7) as usize;
    let d0 = p(8) as i64;
    let d = p(9) as i64;
    let margin = p(10);
    let stereo = p(11) != 0.0;
    let h = nf / 2;

    let x = read_input(in_path, offset, frames, ch);
    let out_f = OpenOptions::new().append(true).open(out_path)
        .unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    let mut out = BufWriter::with_capacity(1 << 20, out_f);

    let w: Vec<f32> = (0..nf)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / nf as f64).cos()) as f32)
        .collect();
    let wc = if stereo { ch } else { 1 };
    let seg_len = h + 2 * d as usize;
    let mut tpl = vec![0.0f64; h * wc];
    let mut seg = vec![0.0f64; seg_len * wc];
    let mut c = vec![0.0f64; 2 * d as usize + 1];
    let mut cs = vec![0.0f64; seg_len + 1];
    let mut score = vec![0.0f64; 2 * d as usize + 1];
    let mut corr = Corr::new(seg_len);
    let mut acc = vec![0.0f32; nf * ch];
    let mut shifts: Vec<i64> = Vec::new();
    let mut prev: Option<i64> = None;
    let mut written = 0usize;
    let mut m: u64 = 0;
    let mut bytes: Vec<u8> = Vec::with_capacity(h * ch * 4);

    while written < n_out {
        let mut a = (((m as f64) * h as f64 + h as f64) / k).round_ties_even() as i64 - h as i64;
        if let (Some(pv), true) = (prev, d > 0) {
            x.mid(pv + h as i64, h, stereo, &mut tpl);
            let e_t: f64 = tpl.iter().map(|v| v * v).sum();
            if e_t > 1e-10 {
                x.mid(a - d, seg_len, stereo, &mut seg);
                corr.run(&seg, seg_len, &tpl, h, wc, &mut c);
                cs[0] = 0.0;
                for i in 0..seg_len {
                    let mut s = 0.0;
                    for cc in 0..wc {
                        let v = seg[i * wc + cc];
                        s += v * v;
                    }
                    cs[i + 1] = cs[i] + s;
                }
                let mut best = 0usize;
                for j in 0..score.len() {
                    let e = cs[j + h] - cs[j];
                    score[j] = c[j] / (e * e_t + 1e-9 * h as f64 * e_t).sqrt();
                    if score[j] > score[best] {
                        best = j;
                    }
                }
                let mut dd = best as i64 - d;
                if dd.abs() > d0 {
                    let lo = (d - d0) as usize;
                    let hi = (d + d0) as usize;
                    let mut nb = lo;
                    for j in lo..=hi {
                        if score[j] > score[nb] {
                            nb = j;
                        }
                    }
                    if score[best] - score[nb] < margin {
                        dd = nb as i64 - d;
                    }
                }
                a += dd;
                shifts.push(dd);
            }
        }
        // overlap-add the frame at a
        for i in 0..nf {
            let t = a + i as i64;
            if t >= 0 && t < x.n {
                let src = &x.x[t as usize * ch..(t as usize + 1) * ch];
                for cc in 0..ch {
                    acc[i * ch + cc] += src[cc] * w[i];
                }
            }
        }
        prev = Some(a);
        let n = h.min(n_out - written);
        bytes.clear();
        for v in &acc[..n * ch] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        out.write_all(&bytes).unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
        written += n;
        acc.copy_within(h * ch.., 0);
        acc[h * ch..].iter_mut().for_each(|v| *v = 0.0);
        m += 1;
    }
    out.flush().unwrap_or_else(|e| die(&format!("{out_path}: {e}")));

    let mut s: Vec<i64> = shifts.iter().map(|v| v.abs()).collect();
    s.sort_unstable();
    let (med, max) = if s.is_empty() {
        (0.0, 0)
    } else if s.len() % 2 == 1 {
        (s[s.len() / 2] as f64, *s.last().unwrap())
    } else {
        ((s[s.len() / 2 - 1] + s[s.len() / 2]) as f64 / 2.0, *s.last().unwrap())
    };
    let wide = if s.is_empty() { 0.0 } else { s.iter().filter(|&&v| v > d0).count() as f64 / s.len() as f64 };
    println!("{med} {max} {wide}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("wsola") => wsola(&args[2..]),
        Some("--version") => println!("lls-dsp {}", env!("CARGO_PKG_VERSION")),
        _ => die("usage: lls-dsp wsola ..."),
    }
}
