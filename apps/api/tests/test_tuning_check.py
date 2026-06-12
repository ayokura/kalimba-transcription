"""Mechanism tests for the tuning mismatch advisory (tuning_check.py).

Synthetic sine mixtures only — fixture-level behavior is covered by the
calibration script (scripts/audio-analysis/calibrate_tuning_mismatch.py)
against the tester corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.transcription.tuning_check import analyze_tuning_mismatch
from app.tunings import get_default_tunings, note_name_to_frequency

SR = 22050


def _tuning(tuning_id: str):
    return next(t for t in get_default_tunings() if t.id == tuning_id)


def _tone_mix(note_names: list[str], duration: float = 2.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * duration)) / SR
    mix = np.zeros_like(t)
    for name in note_names:
        mix += np.sin(2 * np.pi * note_name_to_frequency(name) * t)
    peak = np.max(np.abs(mix))
    return (mix / peak * amplitude) if peak > 0 else mix


def test_d_major_tones_flag_c_tuning():
    audio = _tone_mix(["D4", "F#4", "A4", "C#5", "D5", "F#5", "A5", "D6"])
    report = analyze_tuning_mismatch(audio, SR, _tuning("kalimba-17-c"))
    assert report is not None
    assert report.selected_coverage < 0.88
    assert "F#" in report.outside_pitch_classes
    assert "C#" in report.outside_pitch_classes
    assert report.suggested_tuning_id == "kalimba-17-d"
    assert report.suggested_coverage == pytest.approx(1.0, abs=0.01)


def test_c_major_tones_pass_c_tuning():
    audio = _tone_mix(["C4", "E4", "G4", "C5", "E5", "G5", "A4", "F4"])
    assert analyze_tuning_mismatch(audio, SR, _tuning("kalimba-17-c")) is None


def test_diatonic_suggestion_beats_chromatic_superset():
    # 34L-C (chromatic, coverage trivially 1.0) must not shadow the diatonic
    # D major preset when both cover the peaks.
    audio = _tone_mix(["D4", "F#4", "A4", "C#5", "D5", "F#5", "A5", "D6"])
    report = analyze_tuning_mismatch(audio, SR, _tuning("kalimba-17-c"))
    assert report is not None
    assert report.suggested_tuning_id != "kalimba-34l-c"


def test_silence_returns_none():
    assert analyze_tuning_mismatch(np.zeros(SR), SR, _tuning("kalimba-17-c")) is None


def test_low_gain_skipped():
    # Below the -15 dBFS peak gate the advisory must stay silent even when
    # the (noise-dominated) peaks would not match the tuning.
    audio = _tone_mix(["D4", "F#4", "A4", "C#5", "D5", "F#5"], amplitude=0.05)
    assert analyze_tuning_mismatch(audio, SR, _tuning("kalimba-17-c")) is None


def test_chromatic_tuning_never_flags():
    audio = _tone_mix(["D4", "F#4", "A4", "C#5", "D5", "F#5", "A5", "D6"])
    assert analyze_tuning_mismatch(audio, SR, _tuning("kalimba-34l-c")) is None
