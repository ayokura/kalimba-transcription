"""Mechanism tests for recover_spread_chord_via_segment_start_probe (#167).

Tests use constructed RawEvent/NoteCandidate inputs with monkeypatched
measure_narrow_fft_note_scores and onset_backward_attack_gain to verify
gate logic in isolation.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.models import InstrumentTuning, TuningNote
from app.transcription.constants import (
    MAX_POLYPHONY,
    NARROW_FFT_SPREAD_CHORD_BG_DOMINANCE_RATIO,
    NARROW_FFT_SPREAD_CHORD_MIN_BACKWARD_GAIN,
    NARROW_FFT_SPREAD_CHORD_MIN_FR,
    NARROW_FFT_SPREAD_CHORD_RISE_MIN_RATIO,
)
from app.transcription.events import recover_spread_chord_via_segment_start_probe
from app.transcription.models import Note, NoteCandidate, RawEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tuning() -> InstrumentTuning:
    """Minimal 5-note tuning for testing."""
    return InstrumentTuning(
        id="test",
        name="test-5",
        key_count=5,
        notes=[
            TuningNote(key=1, note_name="G3", frequency=196.0),
            TuningNote(key=2, note_name="B3", frequency=246.942),
            TuningNote(key=3, note_name="D4", frequency=293.665),
            TuningNote(key=4, note_name="B4", frequency=493.883),
            TuningNote(key=5, note_name="C4", frequency=261.626),
        ],
    )


def _event(
    start: float,
    end: float,
    note_names: list[str],
    *,
    primary: str | None = None,
    from_ssg: bool = False,
) -> RawEvent:
    tuning = _tuning()
    notes = []
    for nn in note_names:
        match = next(n for n in tuning.notes if n.note_name == nn)
        notes.append(
            NoteCandidate(key=match.key, note=Note.from_name(nn), score=100.0)
        )
    notes.sort(key=lambda c: c.frequency)
    return RawEvent(
        start_time=start,
        end_time=end,
        notes=notes,
        is_gliss_like=False,
        primary_note_name=primary or note_names[0],
        primary_score=100.0,
        from_short_segment_guard=from_ssg,
        sub_onsets=(start,),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dummy_audio():
    """1-second mono float32 audio at 44100 Hz (content irrelevant — mocked)."""
    return np.zeros(44100, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_PEAKS_MOD = "app.transcription.events"


class TestTruePositive:
    """Candidate passes all gates and is added to the event."""

    def test_rising_candidate_added(self, dummy_audio):
        """B3 has high bg, high rise_ratio, clean fr → added."""
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"], primary="B4")
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4")
        onset_times = [10.0, 10.25, 10.4]  # 10.25 is unconsumed

        # Segment-start scores: B3 has high energy + fr
        seg_scores = {
            "B3": (30.0, 43.0, 0.87),
            "D4": (35.0, 50.0, 0.97),
            "B4": (8.0, -4.0, 0.86),
        }
        # Onset scores: B3 has low energy (not yet attacked)
        onset_scores = {
            "B3": (0.8, 1.0, 0.11),
            "D4": (2.5, 3.5, 0.94),
            "B4": (12.0, 18.0, 0.99),
        }

        def mock_narrow_fft(audio, sr, time, tuning, **kwargs):
            if abs(time - 10.4) < 0.01:
                return seg_scores
            if abs(time - 10.25) < 0.01:
                return onset_scores
            return {}

        def mock_bg(audio, sr, time, freq, lookback_seconds=0.2):
            if abs(time - 10.4) < 0.01:
                if abs(freq - 246.942) < 1:   # B3
                    return 85.0
                if abs(freq - 493.883) < 1:   # B4
                    return 170.0
                if abs(freq - 293.665) < 1:   # D4
                    return 12.0
            return 1.0

        with (
            patch(f"{_PEAKS_MOD}.measure_narrow_fft_note_scores", side_effect=mock_narrow_fft),
            patch(f"{_PEAKS_MOD}.onset_backward_attack_gain", side_effect=mock_bg),
        ):
            result = recover_spread_chord_via_segment_start_probe(
                [prev_event, target_event],
                dummy_audio, 44100, tuning, onset_times,
            )

        assert len(result) == 2
        note_names = {n.note_name for n in result[1].notes}
        assert "B3" in note_names


class TestGate1BackwardGain:
    """Gate 1: backward_gain at segment start must be >= threshold."""

    def test_low_bg_rejected(self, dummy_audio):
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4")
        onset_times = [10.0, 10.25, 10.4]

        seg_scores = {"B3": (30.0, 43.0, 0.87), "D4": (35.0, 50.0, 0.97), "B4": (8.0, -4.0, 0.86)}
        onset_scores = {"B3": (0.8, 1.0, 0.11)}

        def mock_narrow_fft(audio, sr, time, tuning, **kwargs):
            if abs(time - 10.4) < 0.01:
                return seg_scores
            if abs(time - 10.25) < 0.01:
                return onset_scores
            return {}

        def mock_bg(audio, sr, time, freq, lookback_seconds=0.2):
            if abs(freq - 246.942) < 1:   # B3 — below threshold
                return NARROW_FFT_SPREAD_CHORD_MIN_BACKWARD_GAIN - 1.0
            return 170.0

        with (
            patch(f"{_PEAKS_MOD}.measure_narrow_fft_note_scores", side_effect=mock_narrow_fft),
            patch(f"{_PEAKS_MOD}.onset_backward_attack_gain", side_effect=mock_bg),
        ):
            result = recover_spread_chord_via_segment_start_probe(
                [prev_event, target_event],
                dummy_audio, 44100, tuning, onset_times,
            )

        note_names = {n.note_name for n in result[1].notes}
        assert "B3" not in note_names


class TestGate2BgDominance:
    """Gate 2: candidate bg must be >= max_in_event_bg * ratio."""

    def test_dominated_candidate_rejected(self, dummy_audio):
        """In-event note has extremely high bg, candidate is dominated."""
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4")
        onset_times = [10.0, 10.25, 10.4]

        seg_scores = {"B3": (30.0, 43.0, 0.87), "D4": (35.0, 50.0, 0.97), "B4": (8.0, -4.0, 0.86)}
        onset_scores = {"B3": (0.8, 1.0, 0.11)}

        def mock_narrow_fft(audio, sr, time, tuning, **kwargs):
            if abs(time - 10.4) < 0.01:
                return seg_scores
            if abs(time - 10.25) < 0.01:
                return onset_scores
            return {}

        def mock_bg(audio, sr, time, freq, lookback_seconds=0.2):
            if abs(freq - 246.942) < 1:   # B3
                return 50.0  # passes Gate 1
            if abs(freq - 493.883) < 1:   # B4 — very high
                return 1000.0
            if abs(freq - 293.665) < 1:   # D4
                return 500.0
            return 1.0

        with (
            patch(f"{_PEAKS_MOD}.measure_narrow_fft_note_scores", side_effect=mock_narrow_fft),
            patch(f"{_PEAKS_MOD}.onset_backward_attack_gain", side_effect=mock_bg),
        ):
            result = recover_spread_chord_via_segment_start_probe(
                [prev_event, target_event],
                dummy_audio, 44100, tuning, onset_times,
            )

        # 50 < 1000 * 0.25 = 250 → rejected
        note_names = {n.note_name for n in result[1].notes}
        assert "B3" not in note_names


class TestGate5RiseDiscriminator:
    """Gate 5: fund_e must be rising from onset to segment start."""

    def test_non_rising_rejected(self, dummy_audio):
        """Candidate with same energy at onset and segment start → not rising."""
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4")
        onset_times = [10.0, 10.25, 10.4]

        # B3 has similar energy at both times — sustaining, not rising
        seg_scores = {"B3": (30.0, 43.0, 0.87), "D4": (35.0, 50.0, 0.97), "B4": (8.0, -4.0, 0.86)}
        onset_scores = {"B3": (28.0, 40.0, 0.85)}  # rise = 30/28 ≈ 1.07 < 2.0

        def mock_narrow_fft(audio, sr, time, tuning, **kwargs):
            if abs(time - 10.4) < 0.01:
                return seg_scores
            if abs(time - 10.25) < 0.01:
                return onset_scores
            return {}

        def mock_bg(audio, sr, time, freq, lookback_seconds=0.2):
            if abs(freq - 246.942) < 1:
                return 85.0
            if abs(freq - 493.883) < 1:
                return 170.0
            return 12.0

        with (
            patch(f"{_PEAKS_MOD}.measure_narrow_fft_note_scores", side_effect=mock_narrow_fft),
            patch(f"{_PEAKS_MOD}.onset_backward_attack_gain", side_effect=mock_bg),
        ):
            result = recover_spread_chord_via_segment_start_probe(
                [prev_event, target_event],
                dummy_audio, 44100, tuning, onset_times,
            )

        note_names = {n.note_name for n in result[1].notes}
        assert "B3" not in note_names


class TestGate6Dissonance:
    """Gate 6: candidate must not be within 250 cents of existing note."""

    def test_dissonant_candidate_rejected(self, dummy_audio):
        """C4 (261.6 Hz) is within 250 cents of D4 (293.7 Hz) → rejected."""
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4")
        onset_times = [10.0, 10.25, 10.4]

        # C4 is the top non-event candidate
        seg_scores = {
            "C4": (40.0, 55.0, 0.95),
            "D4": (35.0, 50.0, 0.97),
            "B4": (8.0, -4.0, 0.86),
        }
        onset_scores = {"C4": (1.0, 1.5, 0.20)}

        def mock_narrow_fft(audio, sr, time, tuning, **kwargs):
            if abs(time - 10.4) < 0.01:
                return seg_scores
            if abs(time - 10.25) < 0.01:
                return onset_scores
            return {}

        def mock_bg(audio, sr, time, freq, lookback_seconds=0.2):
            if abs(freq - 261.626) < 1:   # C4
                return 90.0
            if abs(freq - 493.883) < 1:   # B4
                return 170.0
            return 12.0

        with (
            patch(f"{_PEAKS_MOD}.measure_narrow_fft_note_scores", side_effect=mock_narrow_fft),
            patch(f"{_PEAKS_MOD}.onset_backward_attack_gain", side_effect=mock_bg),
        ):
            result = recover_spread_chord_via_segment_start_probe(
                [prev_event, target_event],
                dummy_audio, 44100, tuning, onset_times,
            )

        note_names = {n.note_name for n in result[1].notes}
        # D4 is 293.7 Hz, C4 is 261.6 Hz → ~200 cents < 250 → rejected
        assert "C4" not in note_names


class TestMaxPolyphonyGuard:
    """Events at MAX_POLYPHONY should not be modified."""

    def test_full_event_skipped(self, dummy_audio):
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        # Create event with MAX_POLYPHONY notes
        full_notes = ["G3", "B3", "D4", "B4"][:MAX_POLYPHONY]
        target_event = _event(10.4, 10.5, full_notes, primary="B4")
        onset_times = [10.0, 10.25, 10.4]

        result = recover_spread_chord_via_segment_start_probe(
            [prev_event, target_event],
            dummy_audio, 44100, tuning, onset_times,
        )

        assert len(result[1].notes) == MAX_POLYPHONY


class TestShortSegmentGuardSkipped:
    """Events from short-segment guard should not be modified."""

    def test_ssg_event_skipped(self, dummy_audio):
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4", from_ssg=True)
        onset_times = [10.0, 10.25, 10.4]

        result = recover_spread_chord_via_segment_start_probe(
            [prev_event, target_event],
            dummy_audio, 44100, tuning, onset_times,
        )

        note_names = {n.note_name for n in result[1].notes}
        assert note_names == {"D4", "B4"}


class TestNoUnconsumedOnset:
    """When no unconsumed onset exists in the lookback window, skip."""

    def test_all_consumed(self, dummy_audio):
        tuning = _tuning()
        prev_event = _event(10.0, 10.2, ["B4"])
        target_event = _event(10.4, 10.5, ["D4", "B4"], primary="B4")
        # All onsets are consumed by events
        onset_times = [10.0, 10.4]

        result = recover_spread_chord_via_segment_start_probe(
            [prev_event, target_event],
            dummy_audio, 44100, tuning, onset_times,
        )

        note_names = {n.note_name for n in result[1].notes}
        assert note_names == {"D4", "B4"}
