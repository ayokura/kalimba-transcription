"""Mechanism tests for mute-dip re-attack detection and residual decay suppression.

These tests use constructed audio signals to directly verify the per-note
frequency-band energy analysis functions, independent of the full recognizer
pipeline.
"""

import numpy as np
import pytest

from app.transcription import (
    _has_mute_dip_reattack,
    _is_residual_decay,
    _note_band_energy,
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY,
    MUTE_DIP_REATTACK_MAX_DIP_RATIO,
    MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
    RESIDUAL_DECAY_MIN_ONSET_GAIN,
)


SR = 44100


def _sine(frequency: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(SR * duration)) / SR
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _build_mute_dip_signal(
    frequency: float,
    *,
    pre_duration: float = 0.15,
    mute_duration: float = 0.04,
    post_duration: float = 0.15,
    pre_amplitude: float = 0.8,
    mute_amplitude: float = 0.01,
    post_amplitude: float = 0.8,
) -> tuple[np.ndarray, float]:
    """Build a signal with high → mute → re-attack pattern.

    Returns (audio, onset_time) where onset_time is the start of the mute.
    Mute duration (40ms) matches measured kalimba mute-dip width.
    """
    pre = _sine(frequency, pre_duration, pre_amplitude)
    mute = _sine(frequency, mute_duration, mute_amplitude)
    post = _sine(frequency, post_duration, post_amplitude)
    audio = np.concatenate([pre, mute, post])
    onset_time = pre_duration
    return audio, onset_time


def _build_residual_signal(
    frequency: float,
    *,
    duration: float = 0.25,
    amplitude: float = 0.8,
    decay_rate: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Build a smoothly decaying signal (residual, no mute-dip).

    Returns (audio, onset_time) at the midpoint.
    """
    t = np.arange(int(SR * duration)) / SR
    envelope = amplitude * np.exp(-decay_rate * t)
    audio = (envelope * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    onset_time = duration / 2
    return audio, onset_time


def _build_sympathetic_dip_signal(
    frequency: float,
    *,
    pre_duration: float = 0.10,
    dip_duration: float = 0.02,
    post_duration: float = 0.10,
    pre_amplitude: float = 0.8,
    dip_amplitude: float = 0.05,
    post_amplitude: float = 0.5,  # lower than pre: decaying
) -> tuple[np.ndarray, float]:
    """Build a signal with dip but no recovery (sympathetic interference).

    Post amplitude is lower than pre, simulating ongoing decay after a
    brief interference dip from a neighboring tine.
    """
    pre = _sine(frequency, pre_duration, pre_amplitude)
    dip = _sine(frequency, dip_duration, dip_amplitude)
    post = _sine(frequency, post_duration, post_amplitude)
    audio = np.concatenate([pre, dip, post])
    onset_time = pre_duration
    return audio, onset_time


# --- _has_mute_dip_reattack tests ---


class TestHasMuteDipReattack:
    def test_genuine_mute_dip_returns_true(self) -> None:
        freq = 523.251  # C5
        audio, onset = _build_mute_dip_signal(freq)
        assert _has_mute_dip_reattack(audio, SR, onset, freq) is True

    def test_residual_decay_returns_false(self) -> None:
        freq = 523.251  # C5
        audio, onset = _build_residual_signal(freq)
        assert _has_mute_dip_reattack(audio, SR, onset, freq) is False

    def test_sympathetic_dip_returns_false(self) -> None:
        """Dip is present but recovery ratio is too low (sympathetic interference)."""
        freq = 523.251  # C5
        audio, onset = _build_sympathetic_dip_signal(freq)
        assert _has_mute_dip_reattack(audio, SR, onset, freq) is False

    def test_silent_note_returns_false(self) -> None:
        """No meaningful pre-onset energy → not a re-attack."""
        freq = 523.251
        audio = np.zeros(int(SR * 0.25), dtype=np.float32)
        assert _has_mute_dip_reattack(audio, SR, 0.10, freq) is False

    def test_different_frequencies(self) -> None:
        """Mute-dip detection works across the kalimba range."""
        for freq in [261.626, 440.0, 659.255, 1046.502]:  # C4, A4, E5, C6
            audio, onset = _build_mute_dip_signal(freq)
            assert _has_mute_dip_reattack(audio, SR, onset, freq) is True, f"Failed for {freq}Hz"

    def test_works_with_varying_amplitude(self) -> None:
        """Verify the function works with both low and full-scale audio."""
        freq = 523.251
        # Build a mute-dip signal with low overall amplitude
        audio, onset = _build_mute_dip_signal(
            freq, pre_amplitude=0.3, mute_amplitude=0.003, post_amplitude=0.3,
        )
        assert _has_mute_dip_reattack(audio, SR, onset, freq) is True

        # Same signal normalized to full scale should also work
        normalized = audio / (np.max(np.abs(audio)) + 1e-8)
        assert _has_mute_dip_reattack(normalized, SR, onset, freq) is True


# --- _is_residual_decay tests ---


class TestIsResidualDecay:
    def test_smooth_decay_is_residual(self) -> None:
        freq = 523.251
        audio, onset = _build_residual_signal(freq, duration=0.30, decay_rate=1.5)
        end_time = onset + 0.10
        assert _is_residual_decay(audio, SR, onset, freq) is True

    def test_fresh_attack_is_not_residual(self) -> None:
        """A note with strong onset energy should not be classified as residual."""
        freq = 523.251
        # Build: silence → strong attack
        silence = np.zeros(int(SR * 0.08), dtype=np.float32)
        attack = _sine(freq, 0.15, amplitude=0.9)
        audio = np.concatenate([silence, attack])
        onset_time = 0.08
        assert _is_residual_decay(audio, SR, onset_time, freq) is False


# --- forward-scan integration test via segment_peaks ---


class TestForwardScanRecovery:
    def test_forward_scan_recovers_mute_dip_reattack(self) -> None:
        """When primary is residual, forward-scan should find a mute-dip re-attack."""
        from app.transcription import segment_peaks
        from app.tunings import get_default_tunings

        tuning = get_default_tunings()[0]
        freq_e5 = 659.255  # will be residual primary
        freq_c5 = 523.251  # will show mute-dip re-attack

        # Build full audio with pre-segment context so that residual-decay
        # and mute-dip pre-onset measurements have valid data.
        # Layout: [0.0 - 0.15] pre-context | [0.15 - 0.45] segment
        pre_context = 0.15
        seg_duration = 0.30
        total_duration = pre_context + seg_duration
        t = np.arange(int(SR * total_duration)) / SR

        # E5: steady decay across the entire signal (residual)
        e5 = 0.8 * np.exp(-1.5 * t) * np.sin(2 * np.pi * freq_e5 * t)

        # C5: ringing in pre-context, mute at segment start, re-attack after
        c5_envelope = np.ones_like(t) * 0.5
        mute_center = pre_context  # mute happens right at segment start
        mute_mask = (t >= mute_center - 0.01) & (t <= mute_center + 0.02)
        c5_envelope[mute_mask] = 0.005  # near-zero during mute
        c5 = (c5_envelope * np.sin(2 * np.pi * freq_c5 * t)).astype(np.float32)

        audio = (e5 + c5).astype(np.float32)

        start_time = pre_context
        end_time = pre_context + seg_duration

        _r = segment_peaks(
            audio, SR, start_time, end_time, tuning,
            debug=True,
            recent_note_names={"E5", "C5"},
        )
        candidates, debug, primary_hyp, _trace = _r.candidates, _r.debug, _r.primary, _r.trace

        # The segment should NOT be dropped (forward-scan should recover C5)
        assert candidates, "Forward-scan should have recovered a candidate from mute-dip"
        note_names = [c.note_name for c in candidates]
        assert "C5" in note_names, f"Expected C5 in candidates, got {note_names}"
