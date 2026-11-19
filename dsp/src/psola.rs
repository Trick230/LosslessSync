//! Pitch-synchronous time stretch for the vocals stem (TD-PSOLA with raised-cosine splices).
//!
//! Marks are placed one pitch period apart (a YIN track on a decimated mono copy, refined so
//! that the period before each mark best matches the period after it); in unvoiced stretches
//! they are a fixed 5 ms apart. The output is the input read mark to mark: between two
//! consecutive marks it is copied exactly, and the stretch comes from repeating (K > 1) or
//! dropping (K < 1) single periods, cross-faded over one period with a raised cosine. A
//! repeat is placed where the period matches its neighbour best, within a timing error
//! budget: the analysis position never strays more than T_MAX from round(output / K), so
//! the timing error cannot accumulate.

use std::thread;

fn ncc(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += (x * y) as f64;
        aa += (x * x) as f64;
        bb += (y * y) as f64;
    }
    ab / (aa * bb + 1e-20).sqrt()
}

/// Mono signal `x` on [a, a + len), zero outside.
fn seg(x: &[f32], a: i64, len: usize, buf: &mut Vec<f32>) {
    buf.clear();
    buf.resize(len, 0.0);
    let lo = a.max(0);
    let hi = (a + len as i64).min(x.len() as i64);
    if hi > lo {
        buf[(lo - a) as usize..(hi - a) as usize].copy_from_slice(&x[lo as usize..hi as usize]);
    }
}

/// Similarity of the period [m - d, m) with [m, m + d): how cleanly the period ending at m can be
/// repeated. Near silence anything goes.
fn period_match(x: &[f32], m: i64, d: usize, a: &mut Vec<f32>, b: &mut Vec<f32>) -> f64 {
    seg(x, m - d as i64, d, a);
    seg(x, m, d, b);
    let e: f64 = a.iter().chain(b.iter()).map(|&v| (v * v) as f64).sum::<f64>() / (2 * d) as f64;
    if e < 1e-7 {
        1.0 // under -70 dBFS
    } else {
        ncc(a, b)
    }
}

/// YIN period (in decimated samples) per frame, or 0 when unvoiced.
fn yin(y: &[f32], hop: usize, w: usize, tmin: usize, tmax: usize) -> Vec<f32> {
    let frames = y.len() / hop + 1;
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(8);
    let per = frames.div_ceil(threads);
    let mut out = vec![0f32; frames];
    thread::scope(|s| {
        for (ci, chunk) in out.chunks_mut(per).enumerate() {
            s.spawn(move || {
                let mut d = vec![0f32; tmax + 1];
                let mut fr = vec![0f32; w + tmax];
                for (i, o) in chunk.iter_mut().enumerate() {
                    let f = ci * per + i;
                    let a = (f * hop) as i64 - ((w + tmax) / 2) as i64;
                    seg(y, a, w + tmax, &mut fr);
                    let e: f32 = fr[..w].iter().map(|v| v * v).sum::<f32>() / w as f32;
                    if e < 1e-7 {
                        continue;
                    }
                    for t in 1..=tmax {
                        let mut s = 0f32;
                        for j in 0..w {
                            let v = fr[j] - fr[j + t];
                            s += v * v;
                        }
                        d[t] = s;
                    }
                    // cumulative mean normalised difference
                    let mut acc = 0f32;
                    let mut best = (f32::MAX, 0usize);
                    let mut first = 0usize;
                    for t in 1..=tmax {
                        acc += d[t];
                        let c = if acc > 0.0 { d[t] * t as f32 / acc } else { 1.0 };
                        d[t] = c;
                        if t >= tmin {
                            if c < best.0 {
                                best = (c, t);
                            }
                            if first == 0 && c < 0.15 {
                                first = t;
                            }
                        }
                    }
                    let mut t = if first > 0 { first } else { best.1 };
                    while t < tmax && d[t + 1] < d[t] {
                        t += 1;
                    }
                    if d[t] < 0.35 {
                        *o = t as f32;
                    }
                }
            });
        }
    });
    out
}

pub struct Stats {
    pub voiced: f64,
    pub repeats: usize,
    pub forced: usize,
    pub err_med: f64,
    pub err_max: f64,
}

/// Stretches interleaved `x` (ch channels) by k into exactly n_out frames.
pub fn stretch(x: &[f32], ch: usize, sr: f64, k: f64, n_out: usize, t_min: f64, t_max: f64,
               emit: &mut dyn FnMut(&[f32])) -> Stats {
    let n = x.len() / ch;
    let mono: Vec<f32> = x.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect();

    // pitch track on a ~12 kHz copy: two box filters of r samples, then every r-th sample,
    // in one pass (the filters delay by (r - 1) / 2 each, which the decimation takes back)
    let r = ((sr / 12000.0).round() as usize).max(1);
    let shift = r - 1;
    let mut y: Vec<f32> = Vec::with_capacity(n / r + 1);
    let (mut a1, mut a2) = (0f64, 0f64);
    let mut ring = vec![0f64; r]; // the first filter's last r outputs
    for i in 0..n {
        a1 += mono[i] as f64;
        if i >= r {
            a1 -= mono[i - r] as f64;
        }
        let s1 = a1 / r as f64;
        a2 += s1 - if i >= r { ring[i % r] } else { 0.0 };
        ring[i % r] = s1;
        if i >= shift && (i - shift) % r == 0 && y.len() < n / r {
            y.push((a2 / r as f64) as f32);
        }
    }
    let srd = sr / r as f64;
    let hop_d = ((0.01 * srd).round() as usize).max(1);
    let (tmin, tmax) = ((srd / 500.0) as usize, (srd / 60.0).ceil() as usize);
    let w = (0.025 * srd).round() as usize;
    let track = yin(&y, hop_d, w, tmin.max(2), tmax);

    // marks
    let unv = ((0.005 * sr).round() as usize).max(8);
    let mut marks: Vec<i64> = vec![0];
    let mut score: Vec<f64> = Vec::new();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut voiced_n = 0usize;
    let mut m = 0i64;
    while m < n as i64 {
        let f = ((m as f64 / r as f64 / hop_d as f64).round() as usize).min(track.len() - 1);
        let p = track[f];
        let (mut d, mut c) = (unv, f64::NAN);
        if p > 0.0 {
            // the lag (decimated, then full rate) whose period matches the next one best
            let md = m / r as i64;
            let (lo, hi) = (((p * 0.85) as usize).max(2), ((p * 1.15).ceil() as usize).max(3));
            let mut best = (f64::MIN, 0usize);
            for t in lo..=hi {
                let s = period_match(&y, md, t, &mut a, &mut b);
                if s > best.0 {
                    best = (s, t);
                }
            }
            let mut fb = (f64::MIN, 0usize);
            for t in (best.1 * r).saturating_sub(r).max(2)..=best.1 * r + r {
                let s = period_match(&mono, m, t, &mut a, &mut b);
                if s > fb.0 {
                    fb = (s, t);
                }
            }
            if fb.0 > 0.5 {
                d = fb.1;
                c = fb.0;
                voiced_n += d;
            }
        }
        if c.is_nan() {
            c = period_match(&mono, m, d, &mut a, &mut b);
        }
        // score[j]: how well [m_j, m_j + d) matches the period before m_j, which is what
        // repeating segment j cross-fades
        score.push(c);
        m += d as i64;
        marks.push(m);
    }
    drop(mono);
    let nm = marks.len() - 1; // segments
    // past the last mark, fixed steps (the input is zero there)
    let mark = |i: usize| -> i64 {
        if i <= nm { marks[i] } else { marks[nm] + ((i - nm) * unv) as i64 }
    };
    let at = |t: i64, c: usize| -> f32 {
        if t >= 0 && (t as usize) < n { x[t as usize * ch + c] } else { 0.0 }
    };

    // synthesis
    let tmx = t_max * sr;
    let tmn = (t_min * sr).min(tmx / 2.0); // a ramp from t_min to t_max needs t_min < t_max
    let mut out: Vec<f32> = Vec::with_capacity(1 << 16);
    let (mut o, mut j) = (0usize, 0usize);
    let (mut repeats, mut forced) = (0usize, 0usize);
    let mut errs: Vec<f32> = Vec::new();
    let mut last_rep = false;
    while o < n_out {
        let mj = mark(j);
        let d = (mark(j + 1) - mj) as usize;
        let next_t = (o + d) as f64 / k; // where the next segment should start in the input
        let adv = (mj + d as i64) as f64 - next_t;
        let mut nj = j + 1;
        if k > 1.0 {
            let thr = 0.97 - 0.47 * ((adv - tmn) / (tmx - tmn)).clamp(0.0, 1.0);
            let c = if j < nm { score[j] } else { 1.0 };
            if adv > tmx || (adv > tmn && c >= thr && !last_rep) {
                nj = j;
                repeats += 1;
                if adv > tmx && c < 0.5 {
                    forced += 1;
                }
            }
        } else if k < 1.0 && j + 2 <= nm {
            // dropping the next period: it must look like this one
            let thr = 0.97 - 0.47 * ((-adv - tmn) / (tmx - tmn)).clamp(0.0, 1.0);
            let c = score[j + 1];
            if -adv > tmx || (-adv > tmn && c >= thr && !last_rep) {
                nj = j + 2;
                repeats += 1;
                if -adv > tmx && c < 0.5 {
                    forced += 1;
                }
            }
        }
        last_rep = nj != j + 1;
        let mn = mark(nj);
        // [o, o + d): from x[mj + u] cross-faded into x[mn - d + u]; equal when nj == j + 1
        let take = d.min(n_out - o);
        for u in 0..take {
            let (t0, t1) = (mj + u as i64, mn - d as i64 + u as i64);
            if t0 == t1 {
                (0..ch).for_each(|c| out.push(at(t0, c)));
            } else {
                let wv = 0.5 - 0.5 * (std::f64::consts::PI * (u as f64 + 0.5) / d as f64).cos();
                for c in 0..ch {
                    out.push(((1.0 - wv) * at(t0, c) as f64 + wv * at(t1, c) as f64) as f32);
                }
            }
        }
        errs.push((((mn as f64) - next_t) / sr * 1000.0) as f32);
        o += take;
        j = nj;
        if out.len() >= (1 << 16) * ch {
            emit(&out);
            out.clear();
        }
    }
    emit(&out);
    let mut e: Vec<f32> = errs.iter().map(|v| v.abs()).collect();
    e.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Stats {
        voiced: voiced_n as f64 / n.max(1) as f64,
        repeats,
        forced,
        err_med: e.get(e.len() / 2).copied().unwrap_or(0.0) as f64,
        err_max: e.last().copied().unwrap_or(0.0) as f64,
    }
}
