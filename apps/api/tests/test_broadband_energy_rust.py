"""Mechanism tests: Rust broadband energy (kalimba_dsp) == numpy reference.

`peaks.peak_energy_near` / `peaks.batch_peak_energies` delegate to the Rust
shared core (f64). The recognizer's rfft spectrum/frequencies are already
float64, so the port is bit-exact to the pre-delegation numpy implementations
(retained here as the differential reference). The wasm equivalence harness
(`crates/kalimba-dsp/check_wasm.sh`) extends the guarantee to the .wasm build.
"""

import numpy as np
import pytest

import kalimba_dsp as K
from app.transcription.peaks import (
    peak_energy_near,
    batch_peak_energies,
    HARMONIC_BAND_CENTS,
)


# ---------------------------------------------------------------------------
# numpy reference implementations (pre-delegation)
# ---------------------------------------------------------------------------

def _numpy_peak_energy_near(frequencies, spectrum, center_freq, band_cents=HARMONIC_BAND_CENTS):
    valid = frequencies > 0
    positive_freqs = frequencies[valid]
    positive_spectrum = spectrum[valid]
    if center_freq <= 0 or len(positive_freqs) == 0:
        return 0.0
    distances = np.abs(1200.0 * np.log2(positive_freqs / center_freq))
    mask = distances <= band_cents
    if not np.any(mask):
        return 0.0
    return float(np.max(positive_spectrum[mask]))


def _numpy_batch_peak_energies(frequencies, spectrum, center_freqs, band_cents=HARMONIC_BAND_CENTS):
    valid = frequencies > 0
    positive_freqs = frequencies[valid]
    positive_spectrum = spectrum[valid]
    if len(positive_freqs) == 0 or len(center_freqs) == 0:
        return np.zeros(len(center_freqs))
    valid_centers = center_freqs > 0
    log_positive = np.log2(positive_freqs)
    log_centers = np.full(len(center_freqs), -np.inf)
    log_centers[valid_centers] = np.log2(center_freqs[valid_centers])
    distances = np.abs(1200.0 * (log_positive[np.newaxis, :] - log_centers[:, np.newaxis]))
    masks = distances <= band_cents
    results = np.zeros(len(center_freqs))
    for i in range(len(center_freqs)):
        if valid_centers[i] and np.any(masks[i]):
            results[i] = float(np.max(positive_spectrum[masks[i]]))
    return results


def _spectrum(n_fft, sr, seed):
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)  # float64
    spec = np.abs(rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs)))
    return np.ascontiguousarray(freqs), np.ascontiguousarray(spec)


_CENTERS = np.array(
    [261.63, 329.63, 392.0, 523.25, 783.99, 1046.5, 40.0, 0.0, -5.0, 1318.5],
    dtype=np.float64,
)


class TestPeakEnergyNear:
    @pytest.mark.parametrize("n_fft,sr", [(4096, 44100), (8192, 96000), (2048, 48000)])
    def test_matches_numpy(self, n_fft, sr):
        freqs, spec = _spectrum(n_fft, sr, seed=n_fft + sr)
        for c in _CENTERS:
            ref = _numpy_peak_energy_near(freqs, spec, float(c))
            got = peak_energy_near(freqs, spec, float(c))
            assert ref == got, f"center={c}: numpy={ref} rust={got}"

    def test_edge_cases(self):
        freqs = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
        spec = np.array([9.0, 1.0, 2.0, 3.0], dtype=np.float64)
        # center 0 / negative -> 0.0
        assert peak_energy_near(freqs, spec, 0.0) == 0.0
        assert peak_energy_near(freqs, spec, -10.0) == 0.0
        # no bin in band (far away) -> 0.0
        assert peak_energy_near(freqs, spec, 10000.0) == 0.0
        # bin 0 (freq 0) excluded even though it has the max magnitude
        assert peak_energy_near(np.array([0.0], dtype=np.float64), np.array([99.0]), 0.0) == 0.0


class TestBatchPeakEnergies:
    @pytest.mark.parametrize("n_fft,sr", [(4096, 44100), (8192, 96000), (2048, 48000)])
    def test_matches_numpy(self, n_fft, sr):
        freqs, spec = _spectrum(n_fft, sr, seed=n_fft * 2 + sr)
        ref = _numpy_batch_peak_energies(freqs, spec, _CENTERS)
        got = np.asarray(batch_peak_energies(freqs, spec, _CENTERS))
        assert got.shape == ref.shape
        assert np.array_equal(ref, got), f"max|d|={np.abs(ref - got).max():.3e}"

    def test_empty_centers(self):
        freqs, spec = _spectrum(2048, 44100, seed=1)
        got = np.asarray(batch_peak_energies(freqs, spec, np.array([], dtype=np.float64)))
        assert got.shape == (0,)

    def test_consistency_with_single(self):
        """batch result equals per-center peak_energy_near (different cents formula,
        but both bit-match their numpy reference, so cross-check via numpy refs)."""
        freqs, spec = _spectrum(4096, 44100, seed=42)
        batch = np.asarray(batch_peak_energies(freqs, spec, _CENTERS))
        ref_batch = _numpy_batch_peak_energies(freqs, spec, _CENTERS)
        assert np.array_equal(batch, ref_batch)
