"""Mechanism tests for adaptive FFT sizing and low-register energy functions.

The old max(4096, ...) FFT sizing pattern creates a blind spot at high sample
rates: at 96kHz with n_fft=4096, bin spacing is 23.4Hz while D4's +/-40 cent
band is only ~13.5Hz wide — potentially zero bins.  _adaptive_n_fft fixes this
by computing the minimum FFT size needed for >=2 bins in the target band.
"""

import numpy as np
import pytest

from app.transcription import _adaptive_n_fft, _note_band_energy
from app.transcription.peaks import onset_energy_gain, HARMONIC_BAND_CENTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(frequency: float, duration: float, sr: int, amplitude: float = 1.0) -> np.ndarray:
    """Generate a pure sine wave at the given sample rate."""
    t = np.arange(int(sr * duration)) / sr
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# _adaptive_n_fft tests
# ---------------------------------------------------------------------------

class TestAdaptiveNfft:
    """Verify _adaptive_n_fft produces sufficient frequency resolution."""

    def test_low_freq_high_sr_gets_larger_fft(self):
        """D4 (294Hz) at 96kHz needs n_fft >> 4096 for >=2 bins in band."""
        n_fft = _adaptive_n_fft(96000, 294.0, 4096)
        bin_spacing = 96000 / n_fft
        band_hz = 294.0 * (2 ** (40 / 1200) - 2 ** (-40 / 1200))
        bins_in_band = band_hz / bin_spacing
        assert bins_in_band >= 2.0, f"Only {bins_in_band:.1f} bins in band (need >=2)"

    def test_c4_at_96k(self):
        """C4 (261Hz) at 96kHz — even lower frequency, needs larger FFT."""
        n_fft = _adaptive_n_fft(96000, 261.0, 4096)
        bin_spacing = 96000 / n_fft
        band_hz = 261.0 * (2 ** (40 / 1200) - 2 ** (-40 / 1200))
        bins_in_band = band_hz / bin_spacing
        assert bins_in_band >= 2.0

    def test_high_freq_standard_sr_unchanged(self):
        """C6 (1047Hz) at 44100Hz — original 4096 is already sufficient."""
        n_fft = _adaptive_n_fft(44100, 1047.0, 4096)
        assert n_fft == 4096

    def test_power_of_two(self):
        """Result must be a power of two for FFT efficiency."""
        for sr in [44100, 48000, 96000]:
            for freq in [261.0, 294.0, 523.0, 1047.0]:
                n_fft = _adaptive_n_fft(sr, freq, 2048)
                assert n_fft & (n_fft - 1) == 0, f"n_fft={n_fft} not power of 2"

    def test_at_least_chunk_len(self):
        """n_fft must be >= chunk_len (zero-padding, not truncation)."""
        n_fft = _adaptive_n_fft(44100, 1047.0, 8192)
        assert n_fft >= 8192

    def test_min_bins_1_preserves_4096_at_44k(self):
        """With min_bins=1, C4 at 44.1kHz stays at 4096 (same as old behavior)."""
        n_fft = _adaptive_n_fft(44100, 261.0, 3528, min_bins=1)
        assert n_fft == 4096

    def test_min_bins_1_fixes_96k_blind_spot(self):
        """With min_bins=1 at 96kHz, short chunks get escalated past 4096."""
        n_fft = _adaptive_n_fft(96000, 261.0, 2880, min_bins=1)
        bin_spacing = 96000 / n_fft
        band_hz = 261.0 * (2 ** (40 / 1200) - 2 ** (-40 / 1200))
        bins_in_band = band_hz / bin_spacing
        assert bins_in_band >= 1.0, f"Only {bins_in_band:.2f} bins"


# ---------------------------------------------------------------------------
# _note_band_energy tests at low frequencies / high sample rates
# ---------------------------------------------------------------------------

class TestNoteBandEnergyLowRegister:
    """Verify _note_band_energy returns non-zero for low-register notes."""

    @pytest.mark.parametrize("freq,sr", [
        (261.63, 96000),   # C4 at 96kHz
        (293.66, 96000),   # D4 at 96kHz
        (261.63, 44100),   # C4 at 44.1kHz (baseline)
        (293.66, 44100),   # D4 at 44.1kHz (baseline)
    ])
    def test_detects_pure_tone(self, freq, sr):
        """A strong pure tone must produce non-zero band energy."""
        duration = 0.5
        audio = _sine(freq, duration, sr, amplitude=0.8)
        center_time = duration / 2
        energy = _note_band_energy(audio, sr, center_time, freq)
        assert energy > 0.01, f"Energy {energy} too low for {freq}Hz at {sr}Hz SR"

    def test_96k_not_drastically_lower_than_44k(self):
        """96kHz energy should be comparable to 44.1kHz for the same tone."""
        freq = 293.66  # D4
        duration = 0.5
        audio_96k = _sine(freq, duration, 96000, amplitude=0.8)
        audio_44k = _sine(freq, duration, 44100, amplitude=0.8)
        center = duration / 2

        energy_96k = _note_band_energy(audio_96k, 96000, center, freq)
        energy_44k = _note_band_energy(audio_44k, 44100, center, freq)

        # Energy at 96k should be at least 20% of 44k energy (was 0% before fix)
        ratio = energy_96k / (energy_44k + 1e-10)
        assert ratio > 0.2, f"96k energy ratio {ratio:.3f} too low vs 44.1k"


# ---------------------------------------------------------------------------
# onset_energy_gain tests at low frequencies / high sample rates
# ---------------------------------------------------------------------------

class TestOnsetEnergyGainLowRegister:
    """Verify onset_energy_gain detects genuine attacks for low-register notes."""

    def test_genuine_attack_d4_96k(self):
        """D4 attack at 96kHz should produce gain > 1."""
        sr = 96000
        freq = 293.66  # D4
        # Silence then tone
        silence = np.zeros(int(sr * 0.2), dtype=np.float32)
        tone = _sine(freq, 0.3, sr, amplitude=0.8)
        audio = np.concatenate([silence, tone])
        onset_time = 0.2
        end_time = 0.4

        gain = onset_energy_gain(audio, sr, onset_time, end_time, freq)
        assert gain > 2.0, f"Gain {gain:.2f} too low — attack not detected for D4 at 96kHz"

    def test_no_attack_returns_near_one(self):
        """Sustained tone with no onset should produce gain near 1."""
        sr = 96000
        freq = 293.66
        audio = _sine(freq, 1.0, sr, amplitude=0.5)
        gain = onset_energy_gain(audio, sr, 0.3, 0.6, freq)
        assert gain < 3.0, f"Gain {gain:.2f} too high for sustained tone"
