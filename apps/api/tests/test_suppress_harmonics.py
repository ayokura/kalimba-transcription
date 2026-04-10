"""Mechanism tests for suppress_harmonics fundamental guard.

When suppress_harmonics removes energy at a note's partial positions, it must
NOT suppress at frequencies that coincide with another note's fundamental.
For example, D5's 1.5× partial (≈881 Hz) should not suppress A5 (880 Hz).
"""

import numpy as np
import pytest

from app.transcription.peaks import suppress_harmonics
from app.transcription.constants import SUPPRESSION_BAND_CENTS


def _make_spectrum(
    freqs: list[float],
    energies: list[float],
    sr: int = 44100,
    n_fft: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic spectrum with energy peaks at given frequencies."""
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sr)
    spectrum = np.zeros_like(frequencies)
    for freq, energy in zip(freqs, energies):
        if freq <= 0:
            continue
        idx = np.argmin(np.abs(frequencies - freq))
        spectrum[idx] = energy
    return frequencies, spectrum


class TestFundamentalGuard:
    """suppress_harmonics must skip suppression when a partial overlaps
    another note's fundamental."""

    def test_d5_1_5x_does_not_suppress_a5(self) -> None:
        """D5 (587.33 Hz) × 1.5 = 881 Hz ≈ A5 (880 Hz).
        With the guard, A5's energy should be preserved."""
        d5 = 587.3295
        a5 = 880.0
        freqs, spectrum = _make_spectrum([d5, a5], [100.0, 80.0])
        tuning_funds = np.array([d5, a5])

        residual = suppress_harmonics(
            spectrum, freqs, d5,
            partial_ratios=[1.0, 1.5, 2.0],
            tuning_fundamentals=tuning_funds,
        )

        # A5 fundamental should be preserved (not suppressed)
        a5_idx = np.argmin(np.abs(freqs - a5))
        assert residual[a5_idx] == pytest.approx(80.0), (
            "A5 fundamental was suppressed — fundamental guard failed"
        )
        # D5 fundamental should still be suppressed (ratio=1.0 is always suppressed)
        d5_idx = np.argmin(np.abs(freqs - d5))
        assert residual[d5_idx] < 100.0 * 0.5

    def test_c4_2x_still_suppresses_c5(self) -> None:
        """C4 (261.63 Hz) × 2 = 523.25 Hz = C5 (523.25 Hz).
        Integer harmonic overlaps are NOT guarded — the pipeline's octave dyad
        rescue in secondary selection handles this case instead."""
        c4 = 261.6256
        c5 = 523.2511
        freqs, spectrum = _make_spectrum([c4, c5], [100.0, 70.0])
        tuning_funds = np.array([c4, c5])

        residual = suppress_harmonics(
            spectrum, freqs, c4,
            partial_ratios=[1.0, 2.0, 3.0],
            tuning_fundamentals=tuning_funds,
        )

        c5_idx = np.argmin(np.abs(freqs - c5))
        assert residual[c5_idx] < 70.0 * 0.5, (
            "Integer harmonic C4×2=C5 should still be suppressed"
        )

    def test_suppression_works_without_guard(self) -> None:
        """Without tuning_fundamentals, suppression proceeds normally
        (backward compatibility)."""
        d5 = 587.3295
        a5 = 880.0
        freqs, spectrum = _make_spectrum([d5, a5], [100.0, 80.0])

        residual = suppress_harmonics(
            spectrum, freqs, d5,
            partial_ratios=[1.0, 1.5, 2.0],
            # No tuning_fundamentals — no guard
        )

        a5_idx = np.argmin(np.abs(freqs - a5))
        # Without guard, A5 should be suppressed (×0.08)
        assert residual[a5_idx] < 80.0 * 0.5

    def test_non_overlapping_partials_still_suppressed(self) -> None:
        """Partials that don't overlap any fundamental are suppressed normally."""
        d5 = 587.3295
        d5_2x = d5 * 2.0  # 1174.66 Hz — not a tuning fundamental
        freqs, spectrum = _make_spectrum([d5, d5_2x], [100.0, 60.0])
        tuning_funds = np.array([d5, 880.0])  # A5 is in tuning but not at 2×

        residual = suppress_harmonics(
            spectrum, freqs, d5,
            partial_ratios=[1.0, 1.5, 2.0],
            tuning_fundamentals=tuning_funds,
        )

        d5_2x_idx = np.argmin(np.abs(freqs - d5_2x))
        assert residual[d5_2x_idx] < 60.0 * 0.5, (
            "D5's 2× partial should be suppressed (no fundamental overlap)"
        )

    def test_own_fundamental_always_suppressed(self) -> None:
        """The base note's own fundamental (ratio=1.0) is always suppressed,
        even when tuning_fundamentals is provided."""
        d5 = 587.3295
        freqs, spectrum = _make_spectrum([d5], [100.0])
        tuning_funds = np.array([d5, 880.0])

        residual = suppress_harmonics(
            spectrum, freqs, d5,
            partial_ratios=[1.0, 1.5, 2.0],
            tuning_fundamentals=tuning_funds,
        )

        d5_idx = np.argmin(np.abs(freqs - d5))
        assert residual[d5_idx] == pytest.approx(100.0 * 0.08), (
            "Base note's own fundamental should always be suppressed"
        )

    def test_guard_does_not_apply_to_integer_comb(self) -> None:
        """When using default integer comb (no partial_ratios), all ratios are
        integers so the guard never triggers."""
        c4 = 261.6256
        c5 = 523.2511
        freqs, spectrum = _make_spectrum([c4, c5], [100.0, 70.0])
        tuning_funds = np.array([c4, c5])

        residual = suppress_harmonics(
            spectrum, freqs, c4,
            # No partial_ratios — uses integer comb (2, 3, 4)
            tuning_fundamentals=tuning_funds,
        )

        c5_idx = np.argmin(np.abs(freqs - c5))
        assert residual[c5_idx] < 70.0 * 0.5, (
            "Integer harmonics should still be suppressed even with guard"
        )

    def test_guard_applies_to_non_integer_partial(self) -> None:
        """A non-integer partial like 2.247× should be guarded if it overlaps
        another note's fundamental."""
        # Hypothetical: note at 233 Hz, with partial at 2.247× ≈ 523.6 Hz
        # Another note at 523.25 Hz (within 45 cents)
        base = 233.0
        other_fund = 523.2511
        partial_freq = base * 2.247  # ≈ 523.55 Hz
        freqs, spectrum = _make_spectrum([base, other_fund], [100.0, 70.0])
        tuning_funds = np.array([base, other_fund])

        residual = suppress_harmonics(
            spectrum, freqs, base,
            partial_ratios=[1.0, 2.247],
            tuning_fundamentals=tuning_funds,
        )

        other_idx = np.argmin(np.abs(freqs - other_fund))
        assert residual[other_idx] == pytest.approx(70.0), (
            "Non-integer partial 2.247× should be guarded when overlapping another fundamental"
        )
