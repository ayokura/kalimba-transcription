"""Mechanism tests for the unsupervised quality indicators (internal, v1).

Pure-function tests on constructed inputs — no audio, no recognizer. Fixture-level
behaviour (specificity / F1 correlation) is covered by the validation report
(scripts/audio-analysis/quality_indicator_report.py) against the real corpus.
"""

from __future__ import annotations

from app.transcription.quality_indicators import (
    QualityIndicators,
    compute_quality_indicators,
)


def _events(n: int, with_alts: int = 0) -> list[dict]:
    events = [{"notes": [{"pitchClass": "C", "octave": 4}]} for _ in range(n)]
    for i in range(min(with_alts, n)):
        events[i]["alternateGroupings"] = [{"reason": "soft", "confidence": 0.3}]
    return events


def test_clean_loud_recording_flags_green() -> None:
    qi = compute_quality_indicators(_events(20), [], tuning_coverage=0.999, peak_dbfs=-4.0)
    assert isinstance(qi, QualityIndicators)
    assert qi.recording_quality > 0.9
    assert qi.recognizer_confidence == 1.0  # no alts, no slots
    assert qi.flag == "green"


def test_low_gain_recording_drags_recording_quality_down() -> None:
    loud = compute_quality_indicators(_events(20), [], tuning_coverage=0.99, peak_dbfs=-4.0)
    quiet = compute_quality_indicators(_events(20), [], tuning_coverage=0.99, peak_dbfs=-19.0)
    assert quiet.recording_quality < loud.recording_quality
    assert quiet.difficulty > loud.difficulty


def test_high_ambiguity_lowers_confidence() -> None:
    confident = compute_quality_indicators(_events(20, with_alts=0), [], 0.99, -4.0)
    ambiguous = compute_quality_indicators(_events(20, with_alts=18), [], 0.99, -4.0)
    assert ambiguous.recognizer_confidence < confident.recognizer_confidence
    assert ambiguous.difficulty > confident.difficulty


def test_many_candidate_slots_lower_confidence() -> None:
    slots = [{"primaryNote": {"pitchClass": "D", "octave": 4}, "confidence": 0.2}] * 20
    no_slots = compute_quality_indicators(_events(20), [], 0.99, -4.0)
    with_slots = compute_quality_indicators(_events(20), slots, 0.99, -4.0)
    assert with_slots.recognizer_confidence < no_slots.recognizer_confidence


def test_poor_tuning_coverage_and_low_gain_flags_red() -> None:
    # peak_dbfs=-25.0 sits well above the web's -35 dBFS warning floor (so the
    # web itself would not warn) but is still well below GAIN_FULL_DBFS,
    # combined with poor coverage + high ambiguity to push difficulty > 0.6.
    # 50 events so the small-recording shrinkage (n/(n+10) ≈ 0.83) leaves the
    # density signal mostly intact — this scenario is a long ambiguous
    # recording, not a short one.
    qi = compute_quality_indicators(
        _events(50, with_alts=40), [{"confidence": 0.1}] * 40,
        tuning_coverage=0.5, peak_dbfs=-25.0,
    )
    assert qi.flag == "red"
    assert qi.difficulty > 0.6


def test_short_clean_recording_stays_green() -> None:
    # #194 recalibration (2026-07-05): a 5-event clean capture with a couple
    # of uncertainty markers must NOT be flagged — v1 read 2/5 flagged events
    # as density 0.4 and mis-flagged the two short clean corpus recordings
    # (bbeeaad8, 16b37356).  Shrinkage n/(n+10) discounts densities that have
    # little evidence behind them.
    qi = compute_quality_indicators(
        _events(5, with_alts=2), [{"confidence": 0.1}] * 2,
        tuning_coverage=0.95, peak_dbfs=-6.0,
    )
    assert qi.flag == "green"


def test_no_events_is_maximally_uncertain() -> None:
    qi = compute_quality_indicators([], [], tuning_coverage=None, peak_dbfs=float("-inf"))
    assert qi.recording_quality == 0.0  # -inf peak -> gain 0, coverage None
    assert qi.recognizer_confidence == 0.0
    assert qi.flag == "red"


def test_unmeasurable_coverage_uses_gain_only() -> None:
    # coverage None (e.g. low-gain skip) -> recording quality == gain score.
    qi = compute_quality_indicators(_events(5), [], tuning_coverage=None, peak_dbfs=-6.0)
    assert qi.recording_quality == 1.0
    assert qi.signals["tuningCoverage"] is None
