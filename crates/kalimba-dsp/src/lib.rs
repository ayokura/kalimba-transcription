//! kalimba-dsp: DSP primitives for kalimba transcription.
//!
//! Dual-binding crate:
//! - `python` feature (default): exposes a pyo3 extension module `kalimba_dsp`
//!   built via maturin for the API server.
//! - `wasm` feature: exposes wasm-bindgen wrappers for browser-side (WebAudio +
//!   WebAssembly) transcription. Operates on `&[f32]` slices (JS `Float32Array`).
//!
//! The numeric core (`cached_hanning`, `adaptive_n_fft`, `note_band_energy_inner`,
//! `note_band_energy`, `scan_gap_for_mute_dip_with_window_inner`,
//! `detect_gap_rise_attack_inner`) is binding-agnostic pure Rust shared by both
//! bindings. Only the thin FFI wrappers and the module/export glue are cfg-gated.
//!
//! ## Building the Python extension (default)
//!
//! ```text
//! # via maturin (uses [tool.maturin] features = ["python"])
//! maturin develop          # or: maturin build
//! # plain cargo (default feature = python):
//! cargo build
//! ```
//!
//! ## Building for WASM (browser)
//!
//! The wasm target is NOT installed by default. A user must run these once:
//!
//! ```text
//! rustup target add wasm32-unknown-unknown
//! cargo build --no-default-features --features wasm --target wasm32-unknown-unknown
//! ```
//!
//! For a full browser-ready package (JS glue + .wasm), install wasm-pack
//! (`cargo install wasm-pack`) and run (cargo flags go after `--`; wasm-pack's
//! own `--target` selects the JS output kind, not a rustc target):
//!
//! ```text
//! wasm-pack build --target web -- --no-default-features --features wasm
//! ```
//!
//! On the host (no wasm target), `cargo check --no-default-features --features wasm`
//! type-checks the wasm wrapper code without producing a wasm artifact.
//!
//! `crates/kalimba-dsp/check_wasm.sh` builds the package and asserts the wasm
//! outputs match the native pyo3 extension over a battery of inputs.

use rustfft::num_complex::Complex32;
use rustfft::FftPlanner;
use std::cell::RefCell;
use std::collections::HashMap;

/// Onset-detection DSP (mel filterbank, STFT onset strength, peak-pick,
/// backtrack) — the browser pipeline front end. See `onset.rs`.
mod onset;

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

/// One-shot convenience wrapper around `note_band_energy` that allocates its own
/// FFT scratch buffer. The thin pyo3 / wasm-bindgen wrappers delegate here so the
/// `Complex32` scratch type never has to cross a binding boundary. Mirrors the
/// Python `_note_band_energy` (windowed peak energy near a note's band) that it
/// replaces in `apps/api/app/transcription/peaks.py`.
fn note_band_energy_oneshot(
    audio: &[f32],
    sample_rate: i64,
    center_time: f64,
    frequency: f64,
    window_seconds: f64,
    harmonic_band_cents: f64,
) -> f32 {
    let mut fft_buffer: Vec<Complex32> = Vec::new();
    note_band_energy(
        audio,
        sample_rate,
        center_time,
        frequency,
        window_seconds,
        &mut fft_buffer,
        harmonic_band_cents,
    )
}

/// Peak magnitude of a precomputed (f64) spectrum within `±band_cents` of
/// `center_freq`. Mirror of `peaks.peak_energy_near`: positive-freq bins only,
/// cents distance `|1200 * log2(f / center)|`. Operates on the recognizer's
/// already-f64 rfft spectrum so it is numerically equivalent to numpy (no f32
/// downcast). Returns 0.0 when no bin falls in the band.
fn peak_energy_near(
    frequencies: &[f64],
    spectrum: &[f64],
    center_freq: f64,
    band_cents: f64,
) -> f64 {
    if !(center_freq > 0.0) {
        return 0.0;
    }
    let m = frequencies.len().min(spectrum.len());
    let mut best = f64::NEG_INFINITY;
    let mut found = false;
    for j in 0..m {
        let f = frequencies[j];
        if f > 0.0 {
            let cents = (1200.0 * (f / center_freq).log2()).abs();
            if cents <= band_cents {
                let v = spectrum[j];
                if !found || v > best {
                    best = v;
                    found = true;
                }
            }
        }
    }
    if found {
        best
    } else {
        0.0
    }
}

/// Batched `peak_energy_near` over many `center_freqs`. Mirror of
/// `peaks.batch_peak_energies`: precomputes `log2(f)` for the positive-freq bins
/// once, then per center uses the cents distance `|1200 * (log2(f) - log2(c))|`.
/// Note the cents formula differs from `peak_energy_near` (`log2(f/c)`), matching
/// the two distinct numpy implementations exactly. Invalid centers (<= 0) -> 0.
fn batch_peak_energies(
    frequencies: &[f64],
    spectrum: &[f64],
    center_freqs: &[f64],
    band_cents: f64,
) -> Vec<f64> {
    let n = center_freqs.len();
    let mut results = vec![0.0f64; n];
    if n == 0 {
        return results;
    }
    let m = frequencies.len().min(spectrum.len());
    let mut log_pos: Vec<f64> = Vec::with_capacity(m);
    let mut spec_pos: Vec<f64> = Vec::with_capacity(m);
    for j in 0..m {
        let f = frequencies[j];
        if f > 0.0 {
            log_pos.push(f.log2());
            spec_pos.push(spectrum[j]);
        }
    }
    if log_pos.is_empty() {
        return results;
    }
    for (i, &c) in center_freqs.iter().enumerate() {
        if !(c > 0.0) {
            continue;
        }
        let log_c = c.log2();
        let mut best = f64::NEG_INFINITY;
        let mut found = false;
        for (k, &lp) in log_pos.iter().enumerate() {
            if (1200.0 * (lp - log_c)).abs() <= band_cents {
                let v = spec_pos[k];
                if !found || v > best {
                    best = v;
                    found = true;
                }
            }
        }
        if found {
            results[i] = best;
        }
    }
    results
}

/// Binding-agnostic core of `scan_gap_for_mute_dip_with_window`.
///
/// Scans a gap for a mute-dip-then-recovery pattern in `frequency`'s band and
/// returns the recovery time, or `None`. Operates on a plain `&[f32]` audio
/// slice so both the pyo3 and wasm-bindgen wrappers can share it unchanged.
#[allow(clippy::too_many_arguments)]
fn scan_gap_for_mute_dip_with_window_inner(
    audio_slice: &[f32],
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
/// Returns `post_time` (as a candidate segment start) iff all three hold:
///   pre_energy  >= min_pre_energy
///   post_energy >= min_post_energy
///   post_energy / (pre_energy + eps) >= rise_ratio
///
/// Targets the decay-into-restrike pattern that `scan_gap_for_mute_dip_with_window`
/// cannot catch (pre_energy below mute-dip's MIN_PRE_ENERGY floor, but still
/// meaningfully above the noise floor — the `min_pre_energy` gate here is set
/// well below mute-dip's to keep the rise pass open for re-strikes that have
/// decayed further, while still rejecting fresh attacks that start from pure
/// noise). Both offsets are measured backward from `gap_end` so `post_time`
/// is kept inside the gap — the caller uses it as a new Segment's start_time
/// which must be < gap_end for seg_end clamping to stay valid.
/// Binding-agnostic core of `detect_gap_rise_attack`.
///
/// Two-point energy-rise check inside a gap; returns `post_time` candidate start
/// or `None`. Shared by the pyo3 and wasm-bindgen wrappers via a `&[f32]` slice.
#[allow(clippy::too_many_arguments)]
fn detect_gap_rise_attack_inner(
    audio_slice: &[f32],
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
    // Require strict `post_offset > 0` so `post_time = gap_end - post_offset`
    // stays inside the gap. Allowing post_offset == 0 would return
    // post_time == gap_end, which the Python caller clamps to a zero-length
    // segment via `seg_end = min(recovery + default, gap_end)`.
    if !(pre_offset > post_offset && post_offset > 0.0) {
        return None;
    }

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

// ===========================================================================
// Python (pyo3) binding — built by maturin into the `kalimba_dsp` extension.
// Thin wrappers: convert the numpy array to a &[f32] slice and delegate to the
// binding-agnostic *_inner core above.
// ===========================================================================
#[cfg(feature = "python")]
mod python_binding {
    use super::{
        detect_gap_rise_attack_inner, note_band_energy_oneshot,
        scan_gap_for_mute_dip_with_window_inner,
    };
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(signature = (
        audio, sample_rate, gap_start, gap_end, frequency, window_seconds,
        mute_dip_energy_window, max_dip_window, max_recovery_window,
        coarse_step, fine_step,
        min_pre_energy, max_dip_ratio, min_post_energy, min_recovery_ratio,
        harmonic_band_cents,
    ))]
    #[allow(clippy::too_many_arguments)]
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
        let audio_array = audio.as_array();
        let audio_slice = audio_array.as_slice()?;
        scan_gap_for_mute_dip_with_window_inner(
            audio_slice,
            sample_rate,
            gap_start,
            gap_end,
            frequency,
            window_seconds,
            mute_dip_energy_window,
            max_dip_window,
            max_recovery_window,
            coarse_step,
            fine_step,
            min_pre_energy,
            max_dip_ratio,
            min_post_energy,
            min_recovery_ratio,
            harmonic_band_cents,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (
        audio, sample_rate, gap_start, gap_end, frequency,
        window_seconds, pre_offset, post_offset,
        rise_ratio, min_post_energy, min_pre_energy, harmonic_band_cents,
    ))]
    #[allow(clippy::too_many_arguments)]
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
        let audio_array = audio.as_array();
        let audio_slice = audio_array.as_slice()?;
        detect_gap_rise_attack_inner(
            audio_slice,
            sample_rate,
            gap_start,
            gap_end,
            frequency,
            window_seconds,
            pre_offset,
            post_offset,
            rise_ratio,
            min_post_energy,
            min_pre_energy,
            harmonic_band_cents,
        )
    }

    #[pyfunction]
    #[pyo3(signature = (
        audio, sample_rate, center_time, frequency, window_seconds, harmonic_band_cents,
    ))]
    fn note_band_energy(
        audio: PyReadonlyArray1<f32>,
        sample_rate: i64,
        center_time: f64,
        frequency: f64,
        window_seconds: f64,
        harmonic_band_cents: f64,
    ) -> PyResult<f64> {
        let audio_array = audio.as_array();
        let audio_slice = audio_array.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("audio must be a C-contiguous float32 array")
        })?;
        Ok(note_band_energy_oneshot(
            audio_slice,
            sample_rate,
            center_time,
            frequency,
            window_seconds,
            harmonic_band_cents,
        ) as f64)
    }

    #[pyfunction]
    #[pyo3(signature = (sample_rate, frequency, chunk_len, min_bins, harmonic_band_cents))]
    fn adaptive_n_fft(
        sample_rate: i64,
        frequency: f64,
        chunk_len: usize,
        min_bins: usize,
        harmonic_band_cents: f64,
    ) -> usize {
        super::adaptive_n_fft(sample_rate, frequency, chunk_len, min_bins, harmonic_band_cents)
    }

    // --- broadband spectral energy (f64, matches the recognizer's rfft spectrum) ---

    #[pyfunction]
    fn peak_energy_near(
        frequencies: PyReadonlyArray1<f64>,
        spectrum: PyReadonlyArray1<f64>,
        center_freq: f64,
        band_cents: f64,
    ) -> PyResult<f64> {
        let fa = frequencies.as_array();
        let fs = fa.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("frequencies must be a C-contiguous float64 array")
        })?;
        let sa = spectrum.as_array();
        let ss = sa.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("spectrum must be a C-contiguous float64 array")
        })?;
        Ok(super::peak_energy_near(fs, ss, center_freq, band_cents))
    }

    #[pyfunction]
    fn batch_peak_energies<'py>(
        py: Python<'py>,
        frequencies: PyReadonlyArray1<f64>,
        spectrum: PyReadonlyArray1<f64>,
        center_freqs: PyReadonlyArray1<f64>,
        band_cents: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fa = frequencies.as_array();
        let fs = fa.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("frequencies must be a C-contiguous float64 array")
        })?;
        let sa = spectrum.as_array();
        let ss = sa.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("spectrum must be a C-contiguous float64 array")
        })?;
        let ca = center_freqs.as_array();
        let cs = ca.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("center_freqs must be a C-contiguous float64 array")
        })?;
        Ok(PyArray1::from_vec(py, super::batch_peak_energies(fs, ss, cs, band_cents)))
    }

    // --- onset DSP (browser pipeline front end; see crate::onset) ---

    #[pyfunction]
    fn mel_filterbank(
        py: Python<'_>,
        sample_rate: i64,
        n_fft: usize,
        n_mels: usize,
    ) -> Bound<'_, PyArray1<f32>> {
        PyArray1::from_vec(py, crate::onset::mel_filterbank(sample_rate, n_fft, n_mels))
    }

    #[pyfunction]
    fn rms<'py>(
        py: Python<'py>,
        audio: PyReadonlyArray1<f32>,
        frame_length: usize,
        hop_length: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = audio.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("audio must be a C-contiguous float32 array")
        })?;
        Ok(PyArray1::from_vec(py, crate::onset::rms(slice, frame_length, hop_length)))
    }

    #[pyfunction]
    fn onset_strength<'py>(
        py: Python<'py>,
        audio: PyReadonlyArray1<f32>,
        sample_rate: i64,
        hop_length: usize,
        n_fft: usize,
        n_mels: usize,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let arr = audio.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("audio must be a C-contiguous float32 array")
        })?;
        Ok(PyArray1::from_vec(
            py,
            crate::onset::onset_strength(slice, sample_rate, hop_length, n_fft, n_mels),
        ))
    }

    #[pyfunction]
    #[pyo3(signature = (x, pre_max, post_max, pre_avg, post_avg, delta, wait))]
    #[allow(clippy::too_many_arguments)]
    fn peak_pick(
        x: PyReadonlyArray1<f32>,
        pre_max: usize,
        post_max: usize,
        pre_avg: usize,
        post_avg: usize,
        delta: f64,
        wait: usize,
    ) -> PyResult<Vec<u32>> {
        let arr = x.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("x must be a C-contiguous float32 array")
        })?;
        Ok(crate::onset::peak_pick(
            slice, pre_max, post_max, pre_avg, post_avg, delta, wait,
        ))
    }

    #[pyfunction]
    fn onset_backtrack(events: Vec<u32>, energy: PyReadonlyArray1<f32>) -> PyResult<Vec<u32>> {
        let arr = energy.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("energy must be a C-contiguous float32 array")
        })?;
        Ok(crate::onset::onset_backtrack(&events, slice))
    }

    #[pyfunction]
    fn onset_detect(
        onset_env: PyReadonlyArray1<f32>,
        sample_rate: i64,
        hop_length: usize,
        backtrack: bool,
    ) -> PyResult<Vec<u32>> {
        let arr = onset_env.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("onset_env must be a C-contiguous float32 array")
        })?;
        Ok(crate::onset::onset_detect(slice, sample_rate, hop_length, backtrack))
    }

    #[pymodule]
    fn kalimba_dsp(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(scan_gap_for_mute_dip_with_window, m)?)?;
        m.add_function(wrap_pyfunction!(detect_gap_rise_attack, m)?)?;
        m.add_function(wrap_pyfunction!(note_band_energy, m)?)?;
        m.add_function(wrap_pyfunction!(adaptive_n_fft, m)?)?;
        m.add_function(wrap_pyfunction!(peak_energy_near, m)?)?;
        m.add_function(wrap_pyfunction!(batch_peak_energies, m)?)?;
        m.add_function(wrap_pyfunction!(mel_filterbank, m)?)?;
        m.add_function(wrap_pyfunction!(rms, m)?)?;
        m.add_function(wrap_pyfunction!(onset_strength, m)?)?;
        m.add_function(wrap_pyfunction!(peak_pick, m)?)?;
        m.add_function(wrap_pyfunction!(onset_backtrack, m)?)?;
        m.add_function(wrap_pyfunction!(onset_detect, m)?)?;
        Ok(())
    }
}

// ===========================================================================
// WASM (wasm-bindgen) binding — for browser-side transcription.
// Wrappers take `&[f32]` (maps to a JS `Float32Array` via wasm-bindgen) and
// delegate to the same *_inner core. Returns `Option<f64>` -> JS `number | undefined`.
// ===========================================================================
#[cfg(feature = "wasm")]
mod wasm_binding {
    use super::{
        detect_gap_rise_attack_inner, note_band_energy_oneshot,
        scan_gap_for_mute_dip_with_window_inner,
    };
    use wasm_bindgen::prelude::*;

    /// Peak FFT magnitude in `frequency`'s ±`harmonic_band_cents` band within a
    /// window centered on `center_time`. Shared core with the pyo3 binding; the
    /// browser pipeline calls this on a JS `Float32Array` of decoded audio.
    #[wasm_bindgen]
    pub fn note_band_energy(
        audio: &[f32],
        sample_rate: i64,
        center_time: f64,
        frequency: f64,
        window_seconds: f64,
        harmonic_band_cents: f64,
    ) -> f64 {
        note_band_energy_oneshot(
            audio,
            sample_rate,
            center_time,
            frequency,
            window_seconds,
            harmonic_band_cents,
        ) as f64
    }

    /// FFT size giving >= `min_bins` bins inside the ±`harmonic_band_cents` band.
    /// `u32` I/O for natural JS `number` interop (n_fft fits well within u32).
    #[wasm_bindgen]
    pub fn adaptive_n_fft(
        sample_rate: i64,
        frequency: f64,
        chunk_len: u32,
        min_bins: u32,
        harmonic_band_cents: f64,
    ) -> u32 {
        super::adaptive_n_fft(
            sample_rate,
            frequency,
            chunk_len as usize,
            min_bins as usize,
            harmonic_band_cents,
        ) as u32
    }

    // --- broadband spectral energy (f64 Float64Array) ---

    /// Peak magnitude within `±band_cents` of `center_freq` over a precomputed spectrum.
    #[wasm_bindgen]
    pub fn peak_energy_near(
        frequencies: &[f64],
        spectrum: &[f64],
        center_freq: f64,
        band_cents: f64,
    ) -> f64 {
        crate::peak_energy_near(frequencies, spectrum, center_freq, band_cents)
    }

    /// Batched `peak_energy_near` over many center frequencies.
    #[wasm_bindgen]
    pub fn batch_peak_energies(
        frequencies: &[f64],
        spectrum: &[f64],
        center_freqs: &[f64],
        band_cents: f64,
    ) -> Vec<f64> {
        crate::batch_peak_energies(frequencies, spectrum, center_freqs, band_cents)
    }

    // --- onset DSP (browser pipeline front end; see crate::onset) ---

    /// Slaney mel filterbank, row-major `n_mels * (n_fft/2+1)` Float32Array.
    #[wasm_bindgen]
    pub fn mel_filterbank(sample_rate: i64, n_fft: u32, n_mels: u32) -> Vec<f32> {
        crate::onset::mel_filterbank(sample_rate, n_fft as usize, n_mels as usize)
    }

    /// Frame-wise RMS energy (center=True, constant pad).
    #[wasm_bindgen]
    pub fn rms(audio: &[f32], frame_length: u32, hop_length: u32) -> Vec<f32> {
        crate::onset::rms(audio, frame_length as usize, hop_length as usize)
    }

    /// Mel spectral-flux onset strength envelope.
    #[wasm_bindgen]
    pub fn onset_strength(
        audio: &[f32],
        sample_rate: i64,
        hop_length: u32,
        n_fft: u32,
        n_mels: u32,
    ) -> Vec<f32> {
        crate::onset::onset_strength(
            audio,
            sample_rate,
            hop_length as usize,
            n_fft as usize,
            n_mels as usize,
        )
    }

    /// Greedy peak picker; returns peak frame indices (Uint32Array).
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn peak_pick(
        x: &[f32],
        pre_max: u32,
        post_max: u32,
        pre_avg: u32,
        post_avg: u32,
        delta: f64,
        wait: u32,
    ) -> Vec<u32> {
        crate::onset::peak_pick(
            x,
            pre_max as usize,
            post_max as usize,
            pre_avg as usize,
            post_avg as usize,
            delta,
            wait as usize,
        )
    }

    /// Snap onset events back to preceding local energy minima.
    #[wasm_bindgen]
    pub fn onset_backtrack(events: &[u32], energy: &[f32]) -> Vec<u32> {
        crate::onset::onset_backtrack(events, energy)
    }

    /// Full onset-frame detection (normalise -> peak-pick -> backtrack).
    #[wasm_bindgen]
    pub fn onset_detect(
        onset_env: &[f32],
        sample_rate: i64,
        hop_length: u32,
        backtrack: bool,
    ) -> Vec<u32> {
        crate::onset::onset_detect(onset_env, sample_rate, hop_length as usize, backtrack)
    }

    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn scan_gap_for_mute_dip_with_window(
        audio: &[f32],
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
        scan_gap_for_mute_dip_with_window_inner(
            audio,
            sample_rate,
            gap_start,
            gap_end,
            frequency,
            window_seconds,
            mute_dip_energy_window,
            max_dip_window,
            max_recovery_window,
            coarse_step,
            fine_step,
            min_pre_energy,
            max_dip_ratio,
            min_post_energy,
            min_recovery_ratio,
            harmonic_band_cents,
        )
    }

    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn detect_gap_rise_attack(
        audio: &[f32],
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
        detect_gap_rise_attack_inner(
            audio,
            sample_rate,
            gap_start,
            gap_end,
            frequency,
            window_seconds,
            pre_offset,
            post_offset,
            rise_ratio,
            min_post_energy,
            min_pre_energy,
            harmonic_band_cents,
        )
    }
}
