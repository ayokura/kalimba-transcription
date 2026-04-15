use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    static FFT_PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::<f32>::new());
    static HANNING_CACHE: RefCell<HashMap<usize, Vec<f32>>> = RefCell::new(HashMap::new());
}

fn cached_hanning<F: FnOnce(&[f32]) -> R, R>(n: usize, f: F) -> R {
    HANNING_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let window = cache.entry(n).or_insert_with(|| {
            (0..n)
                .map(|i| {
                    0.5 - 0.5
                        * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos()
                })
                .collect()
        });
        f(window)
    })
}

fn adaptive_n_fft(
    sample_rate: i64,
    frequency: f64,
    chunk_len: usize,
    min_bins: usize,
    harmonic_band_cents: f64,
) -> usize {
    let band_hz = frequency
        * ((2f64).powf(harmonic_band_cents / 1200.0)
            - (2f64).powf(-harmonic_band_cents / 1200.0));
    let min_n_fft = if band_hz > 0.0 {
        (sample_rate as f64 / band_hz).ceil() as usize * min_bins
    } else {
        4096
    };
    let n_fft = min_n_fft.max(chunk_len);
    let log2 = (n_fft as f64).log2().ceil() as u32;
    1usize << log2
}

/// Return peak FFT magnitude in ±band_cents around `frequency`.
/// `buffer` must already be length `n_fft`, zero-padded with chunk*hanning prefix.
fn note_band_energy_inner(
    buffer: &mut [Complex32],
    audio_chunk: &[f32],
    sample_rate: i64,
    frequency: f64,
    n_fft: usize,
    harmonic_band_cents: f64,
) -> f32 {
    // Mirror Python's `peak_energy_near` early return when center freq is non-positive.
    // Without this, log2 / negative `as usize` cast on lo_bin can produce wrap-around
    // huge indices and out-of-bounds panics when called with malformed input.
    if !(frequency > 0.0 && sample_rate > 0 && n_fft > 0) {
        return 0.0;
    }
    let chunk_len = audio_chunk.len();
    cached_hanning(chunk_len, |window| {
        for i in 0..chunk_len {
            buffer[i] = Complex32 {
                re: audio_chunk[i] * window[i],
                im: 0.0,
            };
        }
    });
    for i in chunk_len..n_fft {
        buffer[i] = Complex32 { re: 0.0, im: 0.0 };
    }

    FFT_PLANNER.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n_fft);
        fft.process(buffer);
    });

    let nbin = n_fft / 2 + 1;
    let freq_step = sample_rate as f64 / n_fft as f64;
    let log_center = frequency.log2();
    let band_delta_lo = frequency * (2f64).powf(-harmonic_band_cents / 1200.0);
    let band_delta_hi = frequency * (2f64).powf(harmonic_band_cents / 1200.0);

    let lo_bin = (band_delta_lo / freq_step).floor() as usize;
    let hi_bin = ((band_delta_hi / freq_step).ceil() as usize).min(nbin - 1);
    let lo_bin = lo_bin.max(1);

    let mut max = 0.0_f32;
    for k in lo_bin..=hi_bin {
        let f = k as f64 * freq_step;
        if f <= 0.0 {
            continue;
        }
        let cents = 1200.0 * (f.log2() - log_center).abs();
        if cents <= harmonic_band_cents {
            let mag = buffer[k].norm();
            if mag > max {
                max = mag;
            }
        }
    }
    max
}

/// Compute peak energy near `frequency` in a window centered on `center_time`.
/// Mirrors Python `_note_band_energy` semantics:
///  - window_samples = max(int(sr * window_seconds), 512)
///  - center_sample = int(t * sr) (truncation toward zero)
///  - chunk = audio[start:start+window_samples], start = max(center_sample - half, 0)
///  - end = min(start + window_samples, len(audio))
///  - if len(chunk) < 256: return 0
fn note_band_energy(
    audio: &[f32],
    sample_rate: i64,
    center_time: f64,
    frequency: f64,
    window_seconds: f64,
    fft_buffer: &mut Vec<Complex32>,
    harmonic_band_cents: f64,
) -> f32 {
    let window_samples = ((sample_rate as f64 * window_seconds) as i64).max(512) as usize;
    let center_sample = (center_time * sample_rate as f64) as i64;
    let half = (window_samples / 2) as i64;
    let start = (center_sample - half).max(0) as usize;
    let end = (start + window_samples).min(audio.len());
    if end <= start {
        return 0.0;
    }
    let chunk = &audio[start..end];
    if chunk.len() < 256 {
        return 0.0;
    }
    let n_fft = adaptive_n_fft(sample_rate, frequency, chunk.len(), 2, harmonic_band_cents);
    if fft_buffer.len() != n_fft {
        fft_buffer.resize(n_fft, Complex32 { re: 0.0, im: 0.0 });
    }
    note_band_energy_inner(
        fft_buffer,
        chunk,
        sample_rate,
        frequency,
        n_fft,
        harmonic_band_cents,
    )
}

#[pyfunction]
#[pyo3(signature = (
    audio, sample_rate, gap_start, gap_end, frequency, window_seconds,
    mute_dip_energy_window, max_dip_window, max_recovery_window,
    coarse_step, fine_step,
    min_pre_energy, max_dip_ratio, min_post_energy, min_recovery_ratio,
    harmonic_band_cents,
))]
fn scan_gap_for_mute_dip_with_window(
    audio: PyReadonlyArray1<f32>,
    sample_rate: i64,
    gap_start: f64,
    gap_end: f64,
    frequency: f64,
    window_seconds: f64,
    mute_dip_energy_window: f64,
    max_dip_window: f64,
    max_recovery_window: f64,
    coarse_step: f64,
    fine_step: f64,
    min_pre_energy: f64,
    max_dip_ratio: f64,
    min_post_energy: f64,
    min_recovery_ratio: f64,
    harmonic_band_cents: f64,
) -> Option<f64> {
    // Defensive validation against invalid params from the FFI boundary.
    // Without these, fine_step <= 0 makes the n_fine count loop never terminate;
    // coarse_step <= fine_step / 2 rounds coarse_stride to 0 and the outer
    // `i += coarse_stride` loop never advances; non-positive frequency / sample_rate
    // would NaN the log2-cents math in note_band_energy_inner.
    if !(fine_step > 0.0 && coarse_step > 0.0 && frequency > 0.0 && sample_rate > 0) {
        return None;
    }

    let audio_array = audio.as_array();
    let audio_slice = audio_array.as_slice()?;
    let audio_duration = audio_slice.len() as f64 / sample_rate as f64;
    let scan_end = gap_end.min(audio_duration - window_seconds);

    if scan_end - gap_start < max_dip_window + max_recovery_window {
        return None;
    }

    let mut fft_buffer_outer: Vec<Complex32> = Vec::new();
    let mut fft_buffer_inner: Vec<Complex32> = Vec::new();

    // Integer-indexed fine grid matches Python's np.arange(gap_start, scan_end,
    // fine_step) semantic (include i iff gap_start + i*fine_step < scan_end).
    // Using `((scan_end - gap_start) / fine_step).floor()` can undercount by 1
    // vs np.arange for gap_start/fine_step combinations where step multiplication
    // lines up just below scan_end, so we enumerate directly.
    let mut n_fine: i64 = 0;
    loop {
        let t = gap_start + (n_fine as f64) * fine_step;
        if t >= scan_end {
            break;
        }
        n_fine += 1;
    }
    let dip_span = (max_dip_window / fine_step).round().max(0.0) as i64;
    let recovery_span = (max_recovery_window / fine_step).round().max(0.0) as i64;
    let coarse_stride = ((coarse_step / fine_step).round() as i64).max(1);
    let max_i = n_fine - dip_span - recovery_span;
    if max_i <= 0 {
        return None;
    }

    let mut i: i64 = 0;
    while i < max_i {
        let t = gap_start + (i as f64) * fine_step;
        let pre_energy = note_band_energy(
            audio_slice,
            sample_rate,
            t,
            frequency,
            window_seconds,
            &mut fft_buffer_outer,
            harmonic_band_cents,
        ) as f64;

        if pre_energy < min_pre_energy {
            i += coarse_stride;
            continue;
        }

        // Dip scan: fine grid indices i+1..dip_end_idx (exclusive).
        let dip_end_idx = (i + dip_span).min(n_fine);
        let mut min_energy = pre_energy;
        let mut j = i + 1;
        while j < dip_end_idx {
            let t_fine = gap_start + (j as f64) * fine_step;
            let e = note_band_energy(
                audio_slice,
                sample_rate,
                t_fine,
                frequency,
                mute_dip_energy_window,
                &mut fft_buffer_inner,
                harmonic_band_cents,
            ) as f64;
            if e < min_energy {
                min_energy = e;
            }
            j += 1;
        }

        let dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6);
        if dip_ratio >= max_dip_ratio {
            i += coarse_stride;
            continue;
        }

        // Recovery scan: fine grid indices dip_end_idx..recovery_end_idx (exclusive).
        let recovery_end_idx = (dip_end_idx + recovery_span).min(n_fine);
        let mut j = dip_end_idx;
        while j < recovery_end_idx {
            let t_fine = gap_start + (j as f64) * fine_step;
            let e = note_band_energy(
                audio_slice,
                sample_rate,
                t_fine,
                frequency,
                mute_dip_energy_window,
                &mut fft_buffer_inner,
                harmonic_band_cents,
            ) as f64;
            if e >= min_post_energy {
                let recovery_ratio = e / (pre_energy + 1e-6);
                if recovery_ratio >= min_recovery_ratio {
                    return Some(t_fine);
                }
            }
            j += 1;
        }

        i += coarse_stride;
    }

    None
}

/// Detect a sharp energy rise near the end of a gap for `frequency`.
///
/// Two-point check: pre at `gap_end - pre_offset`, post at `gap_end - post_offset`.
/// Returns `post_time` (as a candidate segment start) iff:
///   post_energy >= min_post_energy  AND  post_energy / (pre_energy + eps) >= rise_ratio
///
/// Targets the decay-into-restrike pattern that `scan_gap_for_mute_dip_with_window`
/// cannot catch (pre_energy below its MIN_PRE_ENERGY floor but post_energy high).
/// Both offsets are measured backward from `gap_end` so `post_time` is kept inside
/// the gap — the caller uses it as a new Segment's start_time which must be
/// < gap_end for seg_end clamping to stay valid.
#[pyfunction]
#[pyo3(signature = (
    audio, sample_rate, gap_start, gap_end, frequency,
    window_seconds, pre_offset, post_offset,
    rise_ratio, min_post_energy, min_pre_energy, harmonic_band_cents,
))]
fn detect_gap_rise_attack(
    audio: PyReadonlyArray1<f32>,
    sample_rate: i64,
    gap_start: f64,
    gap_end: f64,
    frequency: f64,
    window_seconds: f64,
    pre_offset: f64,
    post_offset: f64,
    rise_ratio: f64,
    min_post_energy: f64,
    min_pre_energy: f64,
    harmonic_band_cents: f64,
) -> Option<f64> {
    if !(frequency > 0.0 && sample_rate > 0 && window_seconds > 0.0) {
        return None;
    }
    if !(pre_offset > post_offset && post_offset >= 0.0) {
        return None;
    }

    let audio_array = audio.as_array();
    let audio_slice = audio_array.as_slice()?;

    let pre_time = gap_end - pre_offset;
    let post_time = gap_end - post_offset;
    if pre_time < gap_start {
        return None;
    }
    if post_time <= pre_time {
        return None;
    }

    let mut fft_buffer: Vec<Complex32> = Vec::new();
    let pre_energy = note_band_energy(
        audio_slice, sample_rate, pre_time, frequency,
        window_seconds, &mut fft_buffer, harmonic_band_cents,
    ) as f64;
    if pre_energy < min_pre_energy {
        return None;
    }
    let post_energy = note_band_energy(
        audio_slice, sample_rate, post_time, frequency,
        window_seconds, &mut fft_buffer, harmonic_band_cents,
    ) as f64;

    if post_energy < min_post_energy {
        return None;
    }
    let ratio = post_energy / (pre_energy + 1e-6);
    if ratio < rise_ratio {
        return None;
    }
    Some(post_time)
}

#[pymodule]
fn kalimba_dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_gap_for_mute_dip_with_window, m)?)?;
    m.add_function(wrap_pyfunction!(detect_gap_rise_attack, m)?)?;
    Ok(())
}
