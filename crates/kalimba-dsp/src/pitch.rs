//! Pitch-identification DSP: f64 magnitude spectrum + tuning-candidate ranking.
//!
//! Browser-side pitch-ID front end (the step after onset detection).
//!
//! `chunk_spectrum_inner` is the f64 analogue of the recognizer's `_chunk_spectrum`
//! (`apps/api/app/transcription/profiles.py`): a *symmetric* Hann window (numpy
//! `np.hanning`, `2πi/(n-1)`) times `|rfft|`, computed in **f64**. It MUST stay f64:
//! the existing `note_band_energy`/`onset` FFT path is `Complex32` (f32) and
//! `peak_energy_near`/`batch_peak_energies` consume an f64 spectrum, so an f32
//! magnitude here would silently drift the downstream energy scores.
//!
//! `rank_tuning_inner` ports the integer-harmonic-comb branch of
//! `rank_tuning_candidates` (`peaks.py`) — the production default path
//! (`use_per_tine_partial_scoring == False`). It reuses the already-ported f64
//! `crate::batch_peak_energies` for energy extraction and returns one score per
//! input note frequency (caller sorts / argmaxes and maps to a note name).

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::cell::RefCell;

thread_local! {
    // SEPARATE f64 planner — distinct monomorphization from lib.rs's f32 planner.
    static FFT_PLANNER_F64: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::<f64>::new());
}

/// Symmetric Hann window of length `n` in f64, matching numpy `np.hanning(n)`:
/// `0.5 - 0.5*cos(2πi/(n-1))`. Edge cases match numpy: `np.hanning(0) == []`,
/// `np.hanning(1) == [1.0]`.
fn hanning_f64(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// f64 magnitude spectrum `|rfft(chunk * hanning(len(chunk)), n=n_fft)|`.
///
/// Mirror of `_chunk_spectrum(chunk, sample_rate, n_fft)[1]` (profiles.py:323-327):
/// the chunk is windowed at its own length, then zero-padded to (or truncated at)
/// `n_fft`, and the first `n_fft/2+1` rfft magnitudes are returned. `chunk` is
/// promoted to f64 before windowing, matching numpy (`float32 * np.hanning(float64)`
/// promotes to float64). Returns an empty vec on degenerate input.
pub fn chunk_spectrum_inner(chunk: &[f32], sample_rate: i64, n_fft: usize) -> Vec<f64> {
    if n_fft == 0 || sample_rate <= 0 {
        return Vec::new();
    }
    let window = hanning_f64(chunk.len());
    let mut buffer: Vec<Complex<f64>> = vec![Complex { re: 0.0, im: 0.0 }; n_fft];
    // np.fft.rfft(x, n=n_fft) zero-pads when len(x) < n_fft and truncates when longer.
    let m = chunk.len().min(n_fft);
    for i in 0..m {
        buffer[i] = Complex {
            re: chunk[i] as f64 * window[i],
            im: 0.0,
        };
    }
    FFT_PLANNER_F64.with(|planner| {
        let fft = planner.borrow_mut().plan_fft_forward(n_fft);
        fft.process(&mut buffer);
    });
    let nbin = n_fft / 2 + 1;
    let mut out = Vec::with_capacity(nbin);
    for bin in buffer.iter().take(nbin) {
        out.push(bin.norm());
    }
    out
}

/// Per-note candidate scores for the integer-harmonic-comb branch of
/// `rank_tuning_candidates` (peaks.py:226-327, `use_per_tine_partial_scoring == False`).
///
/// Builds the harmonic comb (note*1..4) plus the sub-half / sub-third subharmonic
/// targets (floored to 0 below 40 Hz), extracts band energies via the f64
/// `batch_peak_energies` core, then computes each note's score from
/// `harmonic_support`, `fundamental_ratio`, the subharmonic-alias energy and the
/// octave-alias penalty. Returns one score per `note_freqs` entry, in input order.
/// The caller (JS / parity test) sorts descending and maps the argmax to a note name.
///
/// Constants mirror `constants.py` (HARMONIC_WEIGHTS, OCTAVE_ALIAS_*, OVERTONE_DOMINANT_*);
/// the parity test imports the live Python constants so any future drift is caught.
pub fn rank_tuning_inner(
    frequencies: &[f64],
    spectrum: &[f64],
    note_freqs: &[f64],
    band_cents: f64,
) -> Vec<f64> {
    // HARMONIC_WEIGHTS = [1.0, 0.55, 0.3, 0.15]; index 0 is the fundamental (its weight
    // is not applied to fundamental_energy, matching the Python code).
    const W1: f64 = 0.55;
    const W2: f64 = 0.3;
    const W3: f64 = 0.15;
    const OCTAVE_ALIAS_RATIO_THRESHOLD: f64 = 1.15;
    const OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO: f64 = 0.34;
    const OCTAVE_ALIAS_PENALTY: f64 = 0.85;
    const OVERTONE_DOMINANT_FUNDAMENTAL_RATIO: f64 = 0.18;
    const OVERTONE_DOMINANT_PENALTY_WEIGHT: f64 = 0.0;
    const SUBHARMONIC_FLOOR_HZ: f64 = 40.0;

    let n = note_freqs.len();
    if n == 0 {
        return Vec::new();
    }

    // Target order matches Python's concat: [note*1, note*2, note*3, note*4, sub_half, sub_third].
    let mut targets: Vec<f64> = Vec::with_capacity(6 * n);
    for m in 1..=4u32 {
        for &f in note_freqs {
            targets.push(f * m as f64);
        }
    }
    for &f in note_freqs {
        let h = f / 2.0;
        targets.push(if h < SUBHARMONIC_FLOOR_HZ { 0.0 } else { h });
    }
    for &f in note_freqs {
        let t = f / 3.0;
        targets.push(if t < SUBHARMONIC_FLOOR_HZ { 0.0 } else { t });
    }

    let energies = crate::batch_peak_energies(frequencies, spectrum, &targets, band_cents);

    let mut scores = vec![0.0f64; n];
    for (i, score_slot) in scores.iter_mut().enumerate() {
        // harmonic_energy_matrix[h][i] == energies[h*n + i]
        let fundamental_energy = energies[i];
        // overtone_energy = sum(w*e for e,w in zip(harmonics[1:], weights[1:])), left-to-right.
        let overtone_energy =
            W1 * energies[n + i] + W2 * energies[2 * n + i] + W3 * energies[3 * n + i];
        let harmonic_support = fundamental_energy + overtone_energy;
        let fundamental_ratio = fundamental_energy / harmonic_support.max(1e-9);

        let sub_half = energies[4 * n + i];
        let sub_third = energies[5 * n + i];
        let subharmonic_alias_energy = 0.7 * sub_half + 0.45 * sub_third;
        let octave_alias_energy = sub_half;
        let octave_alias_ratio = octave_alias_energy / fundamental_energy.max(1e-9);

        let mut octave_alias_penalty = 0.0;
        if octave_alias_ratio >= OCTAVE_ALIAS_RATIO_THRESHOLD
            && fundamental_ratio <= OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO
        {
            octave_alias_penalty = octave_alias_energy * OCTAVE_ALIAS_PENALTY;
        }

        let mut score = harmonic_support * (0.2 + 0.8 * fundamental_ratio)
            + 0.45 * fundamental_energy
            - 0.6 * subharmonic_alias_energy
            - octave_alias_penalty;
        if fundamental_ratio < OVERTONE_DOMINANT_FUNDAMENTAL_RATIO {
            score -= OVERTONE_DOMINANT_PENALTY_WEIGHT * overtone_energy;
        }
        *score_slot = score;
    }
    scores
}
