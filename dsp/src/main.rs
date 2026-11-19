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
//!
//!   lls-dsp psola IN OFFSET FRAMES CHANNELS OUT N_OUT K SR T_MIN_MS T_MAX_MS
//!
//! Pitch-synchronous stretch of the vocals stem (see psola.rs), same input and output
//! conventions as wsola. Prints the voiced fraction, the repeated (or dropped) periods, the
//! ones forced where the period did not match, and the median and max timing error in ms.
//!
//!   lls-dsp stretch OUT CHANNELS SR K|- [--block MS --interval MS] [--perc PERC_K]
//!                                              (built with --features signalsmith)
//!
//! Signalsmith Stretch (default preset: 120 ms blocks every 30 ms) of the interleaved
//! float32 frames read from stdin until EOF, stretched by K (N input frames -> round(N*K)
//! output frames), written as an RF64 float WAV to OUT. The stretcher's latency is
//! compensated, so the output lines up with the input to within a few samples (the caller
//! measures the exact latency). With --perc, the input is first split into harmonic and
//! percussive parts (hpss.rs): the harmonic part is stretched into OUT, the percussive part
//! with short-frame WSOLA (which keeps attacks sharp) into PERC_K, once the input has ended
//! and while the harmonic part finishes. With K given as "-", stdin is framed: records of a
//! u32 frame count and that many frames, one record of u32::MAX and K as an f64, and a last
//! u32::MAX - 1 (end of input: a bare EOF means the caller died, and is an error); the
//! input is read and split as it comes, and stretched once K has come.

use std::env;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::process::exit;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

#[cfg(feature = "signalsmith")]
mod hpss;
mod psola;

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
    let ch = p(3) as usize;
    let x = read_input(in_path, p(1) as u64, p(2) as usize, ch);
    let out_f = OpenOptions::new().append(true).open(out_path)
        .unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    let mut out = BufWriter::with_capacity(1 << 20, out_f);
    let mut bytes: Vec<u8> = Vec::new();
    let d0 = p(8) as i64;
    let (med, max, wide) = wsola_run(&x, p(5) as usize, p(6), p(7) as usize, d0, p(9) as i64,
                                     p(10), p(11) != 0.0, &mut |v: &[f32]| {
        bytes.clear();
        for s in v {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        out.write_all(&bytes).unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    });
    out.flush().unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    println!("{med} {max} {wide}");
}

/// WSOLA of x by K into n_out frames, handed to emit in hops; returns the shift statistics
/// (median |shift|, max |shift| in samples, fraction beyond d0).
fn wsola_run(x: &Input, n_out: usize, k: f64, nf: usize, d0: i64, d: i64, margin: f64,
             stereo: bool, emit: &mut dyn FnMut(&[f32])) -> (f64, i64, f64) {
    let ch = x.ch;
    let h = nf / 2;

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
        emit(&acc[..n * ch]);
        written += n;
        acc.copy_within(h * ch.., 0);
        acc[h * ch..].iter_mut().for_each(|v| *v = 0.0);
        m += 1;
    }

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
    (med as f64, max, wide)
}

#[cfg(feature = "signalsmith")]
/// RF64 float WAV written as it goes; the sizes are filled in by finish().
struct Rf64 {
    f: BufWriter<File>,
    ch: usize,
    frames: u64,
}

#[cfg(feature = "signalsmith")]
impl Rf64 {
    const HEADER: u64 = 80;

    fn create(path: &str, ch: usize, sr: u32) -> Rf64 {
        let f = File::create(path).unwrap_or_else(|e| die(&format!("{path}: {e}")));
        let mut f = BufWriter::with_capacity(1 << 20, f);
        let mut h = Vec::with_capacity(80);
        h.extend_from_slice(b"RF64");
        h.extend_from_slice(&u32::MAX.to_le_bytes());
        h.extend_from_slice(b"WAVEds64");
        h.extend_from_slice(&28u32.to_le_bytes());
        h.extend_from_slice(&[0u8; 28]);
        h.extend_from_slice(b"fmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&3u16.to_le_bytes());
        h.extend_from_slice(&(ch as u16).to_le_bytes());
        h.extend_from_slice(&sr.to_le_bytes());
        h.extend_from_slice(&(sr * 4 * ch as u32).to_le_bytes());
        h.extend_from_slice(&(4 * ch as u16).to_le_bytes());
        h.extend_from_slice(&32u16.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&u32::MAX.to_le_bytes());
        f.write_all(&h).unwrap_or_else(|e| die(&format!("{path}: {e}")));
        Rf64 { f, ch, frames: 0 }
    }

    fn write(&mut self, x: &[f32]) {
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.f.write_all(&bytes).unwrap_or_else(|e| die(&format!("write: {e}")));
        self.frames += (x.len() / self.ch) as u64;
    }

    fn finish(mut self) {
        let data = self.frames * 4 * self.ch as u64;
        let mut d = Vec::with_capacity(24);
        d.extend_from_slice(&(data + 72).to_le_bytes()); // file size - 8 (80-byte header)
        d.extend_from_slice(&data.to_le_bytes());
        d.extend_from_slice(&self.frames.to_le_bytes());
        self.f.flush().unwrap_or_else(|e| die(&format!("write: {e}")));
        let f = self.f.get_mut();
        f.seek(SeekFrom::Start(20)).unwrap();
        f.write_all(&d).unwrap_or_else(|e| die(&format!("write: {e}")));
        f.flush().unwrap_or_else(|e| die(&format!("write: {e}")));
    }
}

#[cfg(feature = "signalsmith")]
/// Fills `buf` from `r` unless EOF comes first; returns the bytes read.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> usize {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => die(&format!("stdin: {e}")),
        }
    }
    got
}

#[cfg(feature = "signalsmith")]
enum Msg {
    Data(Vec<f32>),
    K(f64),
    End(u64), // input frames in all
}

#[cfg(feature = "signalsmith")]
/// The stretcher fed in fixed blocks, writing round(N * K) frames once N is known.
struct Sm {
    st: signalsmith_stretch::Stretch,
    out: Rf64,
    y: Vec<f32>,
    k: f64,
    ch: usize,
    fed: u64,
    made: u64,
    skip: u64,
}

#[cfg(feature = "signalsmith")]
impl Sm {
    fn feed(&mut self, x: &[f32], want: u64) {
        let ch = self.ch;
        self.fed += (x.len() / ch) as u64;
        let n = ((self.fed as f64 * self.k).floor() as u64 - self.made) as usize;
        self.made += n as u64;
        self.y.resize(n * ch, 0.0);
        self.st.process(x, &mut self.y);
        let s = self.skip.min(n as u64) as usize;
        self.skip -= s as u64;
        let keep = (n - s).min((want - self.out.frames).min(n as u64) as usize);
        self.out.write(&self.y[s * ch..(s + keep) * ch]);
    }
}

#[cfg(feature = "signalsmith")]
fn stretch(args: &[String]) {
    let usage = "usage: lls-dsp stretch OUT CHANNELS SR K|- [--block MS --interval MS] [--perc PERC_K]";
    if args.len() < 4 || args.len() % 2 != 0 {
        die(usage);
    }
    let ch: usize = args[1].parse().unwrap_or_else(|_| die("bad CHANNELS"));
    let sr: u32 = args[2].parse().unwrap_or_else(|_| die("bad SR"));
    // "-": framed stdin, K comes in it (the input can flow, and be split, before K is known)
    let framed = args[3] == "-";
    let k_arg: Option<f64> = if framed {
        None
    } else {
        match args[3].parse::<f64>() {
            Ok(v) if v > 0.0 => Some(v),
            _ => die("bad K"),
        }
    };
    let (mut block, mut interval, mut perc) = (None, None, None);
    for o in args[4..].chunks_exact(2) {
        let ms = || -> f64 {
            match o[1].parse::<f64>() {
                Ok(v) if v > 0.0 => v,
                _ => die(usage),
            }
        };
        match o[0].as_str() {
            "--block" => block = Some(ms()),
            "--interval" => interval = Some(ms()),
            "--perc" => perc = Some(o[1].clone()),
            _ => die(usage),
        }
    }
    if ch == 0 || sr == 0 {
        die("bad arguments");
    }
    let out_path = args[0].clone();
    let make = move |k: f64| -> Sm {
        let st = if block.is_none() && interval.is_none() {
            signalsmith_stretch::Stretch::preset_default(ch as u32, sr)
        } else {
            let b = block.unwrap_or(120.0);
            let i = interval.unwrap_or(b / 4.0);
            let smp = |ms: f64| ((ms * sr as f64 / 1000.0).round() as usize).max(16);
            signalsmith_stretch::Stretch::new(ch as u32, smp(b), smp(i))
        };
        // output sample j holds input time j / K once these leading samples are dropped
        let skip = (st.input_latency() as f64 * k).round() as u64 + st.output_latency() as u64;
        Sm { st, out: Rf64::create(&out_path, ch, sr), y: Vec::new(), k, ch, fed: 0, made: 0, skip }
    };
    const B: usize = 4096;

    // stdin (and the split, then the attacks' stretch) on one thread, the stretcher on this
    // one; the channel is unbounded since the tonal part waits in it until K comes
    let (tx, rx) = std::sync::mpsc::channel::<Msg>();
    let reader = std::thread::spawn(move || {
        let mut stdin = std::io::stdin().lock();
        let mut k = k_arg;
        let raw = perc.as_ref().map(|p| format!("{p}.raw"));
        // an error here leaves no full-length copy of the attacks behind
        let raw_c = raw.clone();
        let fail = move |m: &str| -> ! {
            if let Some(r) = raw_c.as_ref() {
                let _ = std::fs::remove_file(r);
            }
            die(m)
        };
        let mut bytes = Vec::new();
        let mut split = raw.as_ref().map(|p| (hpss::Hpss::new(ch, sr), Rf64::create(p, ch, sr)));
        let (mut h, mut p) = (Vec::new(), Vec::new());
        let mut total = 0u64;
        loop {
            // framed: records of a u32 frame count + that many frames, u32::MAX + f64 K, or
            // u32::MAX - 1 = end of input
            let frames = if framed {
                let mut hd = [0u8; 4];
                match read_full(&mut stdin, &mut hd) {
                    0 => fail("stdin: closed before the end record"),
                    4 => {}
                    _ => fail("stdin: truncated record"),
                }
                let n = u32::from_le_bytes(hd);
                if n == u32::MAX - 1 {
                    break;
                }
                if n == u32::MAX {
                    let mut kb = [0u8; 8];
                    if read_full(&mut stdin, &mut kb) < 8 {
                        fail("stdin: truncated record");
                    }
                    let v = f64::from_le_bytes(kb);
                    if !(v > 0.0) || k.is_some() {
                        fail("bad K");
                    }
                    k = Some(v);
                    if tx.send(Msg::K(v)).is_err() {
                        return;
                    }
                    continue;
                }
                bytes.resize(n as usize * ch * 4, 0);
                if read_full(&mut stdin, &mut bytes) < bytes.len() {
                    fail("stdin: truncated record");
                }
                n as usize
            } else {
                bytes.resize(B * ch * 4, 0);
                read_full(&mut stdin, &mut bytes) / (4 * ch)
            };
            let x: Vec<f32> = bytes[..frames * ch * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            total += frames as u64;
            match split.as_mut() {
                Some((hp, pw)) => {
                    hp.push(&x, &mut h, &mut p);
                    pw.write(&p);
                    p.clear();
                    if !h.is_empty() && tx.send(Msg::Data(std::mem::take(&mut h))).is_err() {
                        return;
                    }
                }
                None => {
                    if frames > 0 && tx.send(Msg::Data(x)).is_err() {
                        return;
                    }
                }
            }
            if !framed && frames < B {
                break;
            }
        }
        if let Some((mut hp, mut pw)) = split {
            // zeros push the rest of the split out; both outputs end at the input's end
            let z = vec![0f32; hp.hop() * ch];
            let mut sent = pw.frames;
            while pw.frames < total {
                hp.push(&z, &mut h, &mut p);
                let n = ((total - pw.frames) as usize).min(p.len() / ch);
                pw.write(&p[..n * ch]);
                let n = ((total - sent) as usize).min(h.len() / ch);
                sent += n as u64;
                h.truncate(n * ch);
                if n > 0 && tx.send(Msg::Data(std::mem::take(&mut h))).is_err() {
                    return;
                }
                h.clear();
                p.clear();
            }
            pw.finish();
        }
        let _ = tx.send(Msg::End(total));
        // the attacks: WSOLA in short frames with a tight search (keeps them sharp and in
        // place), while the main thread's stretcher finishes the tonal part
        if let (Some(raw), Some(out)) = (raw, perc) {
            let k = k.unwrap_or_else(|| fail("stdin: no K"));
            let x = read_input(&raw, Rf64::HEADER, total as usize, ch);
            let _ = std::fs::remove_file(&raw);
            let smp = |ms: f64| (ms * 1e-3 * sr as f64).round_ties_even() as i64;
            let nf = 2 * (PERC_FRAME_MS * 1e-3 * sr as f64 / 2.0).round_ties_even() as usize;
            let d0 = smp(PERC_TOL_MS);
            let d = d0.max(smp(PERC_MAX_MS));
            let mut w = Rf64::create(&out, ch, sr);
            wsola_run(&x, (total as f64 * k).round_ties_even() as usize, k, nf, d0, d, 0.05,
                      true, &mut |v: &[f32]| w.write(v));
            w.finish();
        }
    });

    let mut sm: Option<Sm> = k_arg.map(&make);
    // blocks kept as they came until K is known (one growing Vec would peak at 2-3x the
    // input on reallocation); pend holds less than one Signalsmith block
    let mut blocks: std::collections::VecDeque<Vec<f32>> = Default::default();
    let mut pend: Vec<f32> = Vec::new();
    let drain = |sm: &mut Sm, blocks: &mut std::collections::VecDeque<Vec<f32>>,
                 pend: &mut Vec<f32>| {
        while let Some(v) = blocks.pop_front() {
            pend.extend_from_slice(&v);
            let mut a = 0;
            while pend.len() - a >= B * ch {
                sm.feed(&pend[a..a + B * ch], u64::MAX);
                a += B * ch;
            }
            pend.drain(..a);
        }
    };
    let mut total = None;
    for msg in rx {
        match msg {
            Msg::Data(v) => blocks.push_back(v),
            Msg::K(k) => sm = Some(make(k)),
            Msg::End(n) => {
                total = Some(n);
                break;
            }
        }
        if let Some(sm) = sm.as_mut() {
            drain(sm, &mut blocks, &mut pend);
        }
    }
    let Some(n) = total else { die("stdin: reader failed") };
    let Some(mut sm) = sm else { die("stdin: no K") };
    drain(&mut sm, &mut blocks, &mut pend);
    let want = (n as f64 * sm.k).round() as u64; // round(N * K)
    // past the end: zeros push the tail out of the stretcher
    while sm.out.frames < want {
        let mut x = vec![0f32; B * ch];
        x[..pend.len()].copy_from_slice(&pend);
        pend.clear();
        sm.feed(&x, want);
    }
    sm.out.finish();
    reader.join().unwrap_or_else(|_| die("reader thread"));
}

// the attacks' WSOLA (stretch --perc): frame, preferred and widest splice search, in ms
#[cfg(feature = "signalsmith")]
const PERC_FRAME_MS: f64 = 12.0;
#[cfg(feature = "signalsmith")]
const PERC_TOL_MS: f64 = 3.0;
#[cfg(feature = "signalsmith")]
const PERC_MAX_MS: f64 = 5.0;

fn psola_cmd(args: &[String]) {
    if args.len() != 10 {
        die("psola IN OFFSET FRAMES CHANNELS OUT N_OUT K SR T_MIN_MS T_MAX_MS");
    }
    let p = |i: usize| -> f64 { args[i].parse().unwrap_or_else(|_| die(&format!("bad arg {}", args[i]))) };
    let (in_path, out_path) = (&args[0], &args[4]);
    let ch = p(3) as usize;
    let x = read_input(in_path, p(1) as u64, p(2) as usize, ch);
    let out_f = OpenOptions::new().append(true).open(out_path)
        .unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    let mut out = BufWriter::with_capacity(1 << 20, out_f);
    let mut bytes = Vec::new();
    let st = psola::stretch(&x.x, ch, p(7), p(6), p(5) as usize, p(8) / 1000.0, p(9) / 1000.0,
                            &mut |v: &[f32]| {
        bytes.clear();
        for s in v {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        out.write_all(&bytes).unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    });
    out.flush().unwrap_or_else(|e| die(&format!("{out_path}: {e}")));
    println!("{} {} {} {} {}", st.voiced, st.repeats, st.forced, st.err_med, st.err_max);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("wsola") => wsola(&args[2..]),
        Some("psola") => psola_cmd(&args[2..]),
        #[cfg(feature = "signalsmith")]
        Some("stretch") => stretch(&args[2..]),
        Some("--version") => println!("lls-dsp {}", env!("CARGO_PKG_VERSION")),
        _ => die("usage: lls-dsp wsola ..."),
    }
}
