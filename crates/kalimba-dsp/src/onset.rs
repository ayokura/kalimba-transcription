//! Onset-detection DSP primitives — the browser-side pipeline's front end.
//!
//! Faithful ports of the librosa-derived numpy reference implementations in
//! `apps/api/app/transcription/segments.py` (`_mel_filterbank`,
//! `_rms_numpy`, `_onset_strength_numpy`, `_peak_pick_numpy`,
//! `_onset_backtrack_numpy`, `_onset_detect_numpy`). Precision is matched to the
//! numpy path on purpose: the numpy STFT runs in single precision
//! (`np.fft.rfft` on a float32 frame matrix -> complex64), so this uses
//! `rustfft::<f32>` to stay in the same precision class (~1e-5 agreement,
//! frame-exact peak picking). `mel_filterbank` evaluates in f64 then rounds to
//! f32, mirroring numpy.
//!
//! The server recognizer delegates its primary onset path to these (see
//! `segments._compute_onset_features`): they are frame-exact to the numpy
//! reference across the full fixture suite and 10-17x faster (onset features
//! were ~30% of transcription wall time). The `_*_numpy` functions are retained
//! as the differential-equivalence oracle. The same crate runs the onset
//! detection in the browser (WebAudio Float32Array -> wasm). Pinned by mechanism
//! tests (`apps/api/tests/test_onset_dsp_rust.py`), the full fixture regression
//! suite, and the wasm equivalence harness.

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Slaney-normalised mel filterbank, row-major `n_mels * (n_fft/2+1)`.
/// Bit-mirror of `segments._mel_filterbank` (fmin=0, fmax=sr/2, htk=False,
/// norm='slaney'); computed in f64 and rounded to f32 like numpy.
pub fn mel_filterbank(sample_rate: i64, n_fft: usize, n_mels: usize) -> Vec<f32> {
    let sr = sample_rate as f64;
    let nfreq = n_fft / 2 + 1;
    // fft_freqs[k] = k * sr / n_fft  (== np.fft.rfftfreq(n_fft, 1/sr))
    let fft_freqs: Vec<f64> = (0..nfreq).map(|k| k as f64 * sr / n_fft as f64).collect();

    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0_f64;
    let logstep = (6.4_f64).ln() / 27.0;
    let min_log_mel = min_log_hz / f_sp;

    let hz_to_mel = |f: f64| -> f64 {
        if f >= min_log_hz {
            min_log_mel + (f / min_log_hz).ln() / logstep
        } else {
            f / f_sp
        }
    };
    let mel_to_hz = |m: f64| -> f64 {
        if m >= min_log_mel {
            min_log_hz * (logstep * (m - min_log_mel)).exp()
        } else {
            f_sp * m
        }
    };

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sr / 2.0);
    // mel_f = mel_to_hz(linspace(mel_min, mel_max, n_mels+2))
    let npts = n_mels + 2;
    let mel_f: Vec<f64> = (0..npts)
        .map(|i| {
            let t = if npts <= 1 {
                0.0
            } else {
                i as f64 / (npts - 1) as f64
            };
            mel_to_hz(mel_min + t * (mel_max - mel_min))
        })
        .collect();
    let fdiff: Vec<f64> = (0..npts - 1).map(|i| mel_f[i + 1] - mel_f[i]).collect();

    let mut weights = vec![0.0f32; n_mels * nfreq];
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for (j, &ff) in fft_freqs.iter().enumerate() {
            // lower = -ramps[i]/fdiff[i] = (ff - mel_f[i]) / fdiff[i]
            let lower = (ff - mel_f[i]) / fdiff[i];
            // upper = ramps[i+2]/fdiff[i+1] = (mel_f[i+2] - ff) / fdiff[i+1]
            let upper = (mel_f[i + 2] - ff) / fdiff[i + 1];
            let w = lower.min(upper).max(0.0) * enorm;
            weights[i * nfreq + j] = w as f32;
        }
    }
    weights
}

/// Read `padded[t]` of a signal zero-padded by `pad` on both ends, without
/// materialising the padded buffer. `padded[t] = audio[t-pad]` inside the
/// original span, else 0.
#[inline]
fn padded_sample(audio: &[f32], pad: usize, t: usize) -> f32 {
    if t >= pad && t < pad + audio.len() {
        audio[t - pad]
    } else {
        0.0
    }
}

/// Frame count for a both-ends `pad`-padded signal framed at `frame_length`/`hop`.
/// Mirrors numpy `1 + (len(padded) - frame_length) // hop_length`.
#[inline]
fn framed_count(audio_len: usize, pad: usize, frame_length: usize, hop_length: usize) -> usize {
    let padded_len = audio_len + 2 * pad;
    if padded_len < frame_length || hop_length == 0 {
        0
    } else {
        1 + (padded_len - frame_length) / hop_length
    }
}

/// Frame-wise RMS energy. Mirror of `segments._rms_numpy`
/// (center=True, pad_mode='constant'). Squares accumulated in f64 then sqrt'd
/// and rounded to f32.
pub fn rms(audio: &[f32], frame_length: usize, hop_length: usize) -> Vec<f32> {
    let pad = frame_length / 2;
    let n_frames = framed_count(audio.len(), pad, frame_length, hop_length);
    let mut out = Vec::with_capacity(n_frames);
    for j in 0..n_frames {
        let base = hop_length * j;
        let mut sum = 0.0f64;
        for i in 0..frame_length {
            let v = padded_sample(audio, pad, base + i) as f64;
            sum += v * v;
        }
        out.push((sum / frame_length as f64).sqrt() as f32);
    }
    out
}

/// Mel spectral-flux onset strength. Mirror of `segments._onset_strength_numpy`:
/// periodic-Hann STFT power -> Slaney mel -> power_to_db(ref=1, amin=1e-10,
/// top_db=80) -> positive lag-1 diff -> mean over mel bands -> left-pad by
/// `1 + n_fft//(2*hop)` and trim to the frame count. Single precision (rustfft
/// f32) to match numpy's complex64 path.
pub fn onset_strength(
    audio: &[f32],
    sample_rate: i64,
    hop_length: usize,
    n_fft: usize,
    n_mels: usize,
) -> Vec<f32> {
    let pad = n_fft / 2;
    let n_frames = framed_count(audio.len(), pad, n_fft, hop_length);
    if n_frames == 0 {
        return Vec::new();
    }
    let nfreq = n_fft / 2 + 1;

    // Periodic Hann (0.5 - 0.5*cos(2*pi*i/n_fft)), f32 — NOT the symmetric
    // np.hanning (which divides by n_fft-1).
    let window: Vec<f32> = (0..n_fft)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / n_fft as f64).cos()) as f32)
        .collect();

    // Sparse mel rows: each triangular filter touches only a few freq bins.
    let mel = mel_filterbank(sample_rate, n_fft, n_mels);
    let mel_rows: Vec<Vec<(usize, f32)>> = (0..n_mels)
        .map(|i| {
            (0..nfreq)
                .filter_map(|k| {
                    let w = mel[i * nfreq + k];
                    if w != 0.0 {
                        Some((k, w))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buf: Vec<Complex32> = vec![Complex32 { re: 0.0, im: 0.0 }; n_fft];

    // log_mel matrix (n_mels, n_frames), row-major; track global max for the db floor.
    let mut log_mel = vec![0.0f32; n_mels * n_frames];
    let mut global_max = f32::NEG_INFINITY;
    for j in 0..n_frames {
        let base = hop_length * j;
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = Complex32 {
                re: padded_sample(audio, pad, base + i) * window[i],
                im: 0.0,
            };
        }
        fft.process(&mut buf);
        for i in 0..n_mels {
            let mut acc = 0.0f32;
            for &(k, w) in &mel_rows[i] {
                let c = buf[k];
                acc += w * (c.re * c.re + c.im * c.im);
            }
            let m = if acc > 1e-10 { acc } else { 1e-10 };
            let db = 10.0 * m.log10();
            log_mel[i * n_frames + j] = db;
            if db > global_max {
                global_max = db;
            }
        }
    }

    let floor = global_max - 80.0;
    // Positive lag-1 diff over the clipped log_mel, mean over mel bands.
    let mut core = vec![0.0f32; n_frames.saturating_sub(1)];
    for j in 1..n_frames {
        let mut sum = 0.0f32;
        for i in 0..n_mels {
            let cur = log_mel[i * n_frames + j].max(floor);
            let prev = log_mel[i * n_frames + (j - 1)].max(floor);
            let d = cur - prev;
            if d > 0.0 {
                sum += d;
            }
        }
        core[j - 1] = sum / n_mels as f32;
    }

    // Left-pad by pad_width, trim to n_frames.
    let pad_width = 1 + n_fft / (2 * hop_length);
    let mut out = vec![0.0f32; n_frames];
    for (t, &v) in core.iter().enumerate() {
        let idx = pad_width + t;
        if idx >= n_frames {
            break;
        }
        out[idx] = v;
    }
    out
}

/// Greedy peak picker. Mirror of `segments._peak_pick_numpy` (librosa
/// `util.peak_pick`, sparse=True). Window means are accumulated in f64 here,
/// whereas numpy's `np.mean` over a float32 envelope accumulates in float32
/// (it only upcasts to f64 when adding the python-float delta afterwards).
/// The two can differ by ~1e-8, so agreement with the numpy reference is not
/// guaranteed bit-for-bit when a frame value coincides with the threshold —
/// in practice unobserved (2026-07-02 audit; dtype alignment is tracked in
/// the sprint plan, Sprint 2/5).
#[allow(clippy::too_many_arguments)]
pub fn peak_pick(
    x: &[f32],
    pre_max: usize,
    post_max: usize,
    pre_avg: usize,
    post_avg: usize,
    delta: f64,
    wait: usize,
) -> Vec<u32> {
    let sz = x.len();
    let mut peaks: Vec<u32> = Vec::new();
    if sz == 0 {
        return peaks;
    }

    let slice_max = |lo: usize, hi: usize| -> f32 {
        x[lo..hi].iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    };
    let slice_mean = |lo: usize, hi: usize| -> f64 {
        let s: f64 = x[lo..hi].iter().map(|&v| v as f64).sum();
        s / (hi - lo) as f64
    };

    // Frame 0 special case.
    let max0 = slice_max(0, post_max.min(sz).max(1).min(sz));
    let avg0 = slice_mean(0, post_avg.min(sz).max(1).min(sz));
    let mut n;
    if x[0] >= max0 && (x[0] as f64) >= avg0 + delta {
        peaks.push(0);
        n = wait + 1;
    } else {
        n = 1;
    }

    while n < sz {
        let lo = n.saturating_sub(pre_max);
        let hi = (n + post_max).min(sz);
        if x[n] != slice_max(lo, hi) {
            n += 1;
            continue;
        }
        let alo = n.saturating_sub(pre_avg);
        let ahi = (n + post_avg).min(sz);
        if (x[n] as f64) < slice_mean(alo, ahi) + delta {
            n += 1;
            continue;
        }
        peaks.push(n as u32);
        n += wait + 1;
    }
    peaks
}

/// Snap each onset event back to the nearest preceding local minimum of
/// `energy`. Mirror of `segments._onset_backtrack_numpy` (frame 0 always a
/// fallback). `events` must be sorted ascending (peak_pick output is).
pub fn onset_backtrack(events: &[u32], energy: &[f32]) -> Vec<u32> {
    let n = energy.len();
    let mut minima: Vec<u32> = vec![0];
    if n >= 3 {
        for i in 1..n - 1 {
            if energy[i] <= energy[i - 1] && energy[i] < energy[i + 1] {
                minima.push(i as u32);
            }
        }
    }
    // minima is already sorted & unique (0, then strictly increasing i>0).
    events
        .iter()
        .map(|&e| {
            // searchsorted(side='right') - 1, clipped to [0, len-1].
            let pos = minima.partition_point(|&m| m <= e);
            let idx = pos.saturating_sub(1).min(minima.len() - 1);
            minima[idx]
        })
        .collect()
}

/// Full onset-frame detection. Mirror of `segments._onset_detect_numpy`:
/// max-normalise a copy, derive librosa's default peak-pick windows from sr/hop,
/// peak-pick, then (optionally) backtrack against the *un-normalised* envelope.
pub fn onset_detect(
    onset_env: &[f32],
    sample_rate: i64,
    hop_length: usize,
    backtrack: bool,
) -> Vec<u32> {
    let sr = sample_rate as f64;
    let hop = hop_length as f64;

    let mut env: Vec<f32> = onset_env.to_vec();
    let env_max = env.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if env_max > 0.0 {
        for v in env.iter_mut() {
            *v /= env_max;
        }
    }

    // int(seconds * sr // hop) — Python float floor-division then truncation.
    let floordiv = |seconds: f64| -> usize { ((seconds * sr) / hop).floor() as usize };
    let pre_max = floordiv(0.03);
    let post_max = floordiv(0.00) + 1;
    let pre_avg = floordiv(0.10);
    let post_avg = floordiv(0.10) + 1;
    let wait = floordiv(0.03);
    let delta = 0.07_f64;

    let frames = peak_pick(&env, pre_max, post_max, pre_avg, post_avg, delta, wait);
    if backtrack && !frames.is_empty() {
        onset_backtrack(&frames, onset_env)
    } else {
        frames
    }
}
