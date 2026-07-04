"""Unsupervised recognition-quality / difficulty indicators (internal, v1).

These are **internal self-assessment signals, NOT a tester-facing feature.** They
are computed without ground truth and are intended for internal triage
prioritisation and for validation against triage_verdicts.json / corrections —
not for display. See scripts/audio-analysis/quality_indicator_report.py.

Motivation (why a no-ground-truth signal is useful even at F1 ~ 1.0):
- note-level F1 over the corpus shifts as harder recordings are added, so it is
  not a stable per-recording "is this trustworthy" signal.
- Testers cannot be expected to perfectly review every transcription; a
  per-recording flag lets attention be focused where it is needed.
- Tester corrections are themselves imperfect (unnoticed performance errors), so
  a confidence signal that disagrees with a correction can surface probable
  human error ("recognizer confident here, yet it was edited").

The signals combine (a) recording-quality cues — peak level, tuning-fit coverage
— and (b) recognizer self-confidence cues — candidate ambiguity / soft-reject
density. All weights and thresholds below are PROVISIONAL; calibrating them is
the job of the validation report, not an assumption.

Pure synthesis of already-computed signals; no new spectral extraction, numpy
only (streaming/WASM portable).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Recording-quality calibration (provisional) ---------------------------
# Peak level mapped to [0, 1]. The web client's low-volume warning
# (LOW_LEVEL_PEAK_DB, SimpleHome.tsx) was relaxed from -15 to -35 dBFS on
# 2026-07-03 (tester feedback: -15 warned on recordings that were actually
# fine). Treat that -35 dBFS floor as the zero anchor here too, so a
# recording the web does *not* warn about never gets scored as
# recording_quality == 0; a comfortably-loud capture (>= -6 dBFS) scores full
# marks. Note tuning_check.LOW_GAIN_SKIP_PEAK_DBFS intentionally kept its own
# -15 dBFS calibration (see its comment) and no longer mirrors this constant.
GAIN_FULL_DBFS = -6.0
GAIN_ZERO_DBFS = -35.0
# Weight of gain vs tuning-fit coverage in the recording-quality score.
RECORDING_GAIN_WEIGHT = 0.6
RECORDING_COVERAGE_WEIGHT = 0.4

# --- Recognizer-confidence calibration (provisional) -----------------------
# How much event-level alternate-grouping density vs candidate-slot density
# lower the confidence. Both are uncertainty markers the recognizer already
# emits (soft-rejected alternates, dropped-segment slots).
CONFIDENCE_ALT_WEIGHT = 0.6
CONFIDENCE_SLOT_WEIGHT = 0.4
# Small-recording shrinkage (#194 recalibration, 2026-07-05): densities over
# few events are statistically unreliable — a 5-event recording with 2
# flagged events reads as density 0.4 from noise alone, which made v1
# mis-flag the two short clean recordings (bbeeaad8 5 GT → recConf 0.000,
# 16b37356 12 GT → 0.171) while every long clean recording scored green.
# Shrink both densities toward 0 by n/(n + prior): a density estimated from
# n events is trusted in proportion to the evidence behind it.
CONFIDENCE_DENSITY_SHRINKAGE_EVENTS = 10.0

# --- Composite difficulty -> flag (provisional) ----------------------------
# Recording quality dominates: environment is the primary driver of hard cases
# (the critical evaluation found existing warnings already cover gain/tuning).
DIFFICULTY_RECORDING_WEIGHT = 0.6
DIFFICULTY_CONFIDENCE_WEIGHT = 0.4
FLAG_GREEN_MAX_DIFFICULTY = 0.30
FLAG_YELLOW_MAX_DIFFICULTY = 0.60


@dataclass
class QualityIndicators:
    recording_quality: float  # [0, 1], higher = cleaner capture
    recognizer_confidence: float  # [0, 1], higher = less ambiguous output
    difficulty: float  # [0, 1], higher = harder / less trustworthy
    flag: str  # "green" | "yellow" | "red"
    signals: dict = field(default_factory=dict)  # raw components for calibration


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _gain_score(peak_dbfs: float) -> float:
    if peak_dbfs >= GAIN_FULL_DBFS:
        return 1.0
    if peak_dbfs <= GAIN_ZERO_DBFS:
        return 0.0
    return (peak_dbfs - GAIN_ZERO_DBFS) / (GAIN_FULL_DBFS - GAIN_ZERO_DBFS)


def peak_dbfs_of(audio) -> float:
    """Peak level in dBFS for a normalised [-1, 1] signal. -inf-safe."""
    import numpy as np

    peak = float(np.max(np.abs(np.asarray(audio)))) if getattr(audio, "size", len(audio)) else 0.0
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


def compute_quality_indicators(
    events: list[dict],
    candidate_slots: list[dict] | None,
    tuning_coverage: float | None,
    peak_dbfs: float,
) -> QualityIndicators:
    """Compute internal quality/difficulty indicators from a transcription result.

    Args:
        events: ScoreEvent dicts (each may carry ``alternateGroupings``).
        candidate_slots: CandidateSlot dicts (dropped-segment candidates), or None.
        tuning_coverage: selected-tuning spectral coverage in [0, 1] (see
            tuning_check.measure_selected_coverage), or None if unmeasurable.
        peak_dbfs: recording peak level in dBFS.
    """
    candidate_slots = candidate_slots or []
    n_events = len(events)

    gain_score = _gain_score(peak_dbfs)
    # When coverage is unmeasurable (very low gain / no tonal content) the
    # recording quality is governed by gain alone.
    if tuning_coverage is None:
        recording_quality = gain_score
        coverage_for_signal = None
    else:
        coverage_score = _clip01(tuning_coverage)
        recording_quality = (
            RECORDING_GAIN_WEIGHT * gain_score
            + RECORDING_COVERAGE_WEIGHT * coverage_score
        )
        coverage_for_signal = coverage_score

    if n_events == 0:
        # No events recognised at all: maximally uncertain.
        alt_density = 1.0
        slot_density = 1.0
        shrink = 1.0
    else:
        events_with_alts = sum(1 for e in events if e.get("alternateGroupings"))
        alt_density = events_with_alts / n_events
        slot_density = min(1.0, len(candidate_slots) / n_events)
        shrink = n_events / (n_events + CONFIDENCE_DENSITY_SHRINKAGE_EVENTS)
    recognizer_confidence = _clip01(
        1.0
        - shrink
        * (CONFIDENCE_ALT_WEIGHT * alt_density + CONFIDENCE_SLOT_WEIGHT * slot_density)
    )

    difficulty = _clip01(
        1.0
        - (
            DIFFICULTY_RECORDING_WEIGHT * recording_quality
            + DIFFICULTY_CONFIDENCE_WEIGHT * recognizer_confidence
        )
    )
    if difficulty <= FLAG_GREEN_MAX_DIFFICULTY:
        flag = "green"
    elif difficulty <= FLAG_YELLOW_MAX_DIFFICULTY:
        flag = "yellow"
    else:
        flag = "red"

    return QualityIndicators(
        recording_quality=round(recording_quality, 4),
        recognizer_confidence=round(recognizer_confidence, 4),
        difficulty=round(difficulty, 4),
        flag=flag,
        signals={
            "peakDbfs": round(peak_dbfs, 2) if peak_dbfs != float("-inf") else None,
            "gainScore": round(gain_score, 4),
            "tuningCoverage": round(coverage_for_signal, 4) if coverage_for_signal is not None else None,
            "altDensity": round(alt_density, 4),
            "slotDensity": round(slot_density, 4),
            "eventCount": n_events,
        },
    )
