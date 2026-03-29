from __future__ import annotations

import io
import json
import math
from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from .models import InstrumentTuning, NotationViews, ScoreEvent, ScoreNote, TranscriptionResult
from .tunings import build_custom_tuning, parse_note_name

FRAME_LENGTH = 2048
HOP_LENGTH = 256
TEMPO_ESTIMATION_HOP_LENGTH = 1024
RMS_MEDIAN_THRESHOLD_MAX_PEAK_RATIO = 0.45
NESTED_SEGMENT_DEDUP_MAX_START_DELTA = 0.02
SEGMENT_OVERLAP_TRIM_MAX_OVERLAP = 0.18
SEGMENT_OVERLAP_TRIM_MIN_DURATION = 0.18
TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE = 0.18
TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE = 0.7
TERMINAL_ORPHAN_SEGMENT_DURATION = 0.32
TERMINAL_TWO_ONSET_TAIL_MIN_GAP_AFTER_ACTIVE = 0.9
TERMINAL_TWO_ONSET_TAIL_MAX_GAP_AFTER_ACTIVE = 1.6
TERMINAL_TWO_ONSET_TAIL_MIN_INTERVAL = 0.6
TERMINAL_TWO_ONSET_TAIL_MAX_INTERVAL = 1.3
TERMINAL_TWO_ONSET_TAIL_SEGMENT_DURATION = 0.32
LEADING_ORPHAN_ONSET_MIN_GAP_BEFORE_ACTIVE = 0.35
LEADING_ORPHAN_ONSET_MAX_GAP_BEFORE_ACTIVE = 0.85
LEADING_ORPHAN_SEGMENT_DURATION = 0.24
MULTI_ONSET_GAP_MIN_DURATION = 1.0
MULTI_ONSET_GAP_MIN_EDGE_SPACING = 0.18
MULTI_ONSET_GAP_MIN_INTERVAL = 0.18
MULTI_ONSET_GAP_MAX_INTERVAL = 0.42
MULTI_ONSET_GAP_MIN_SHORT_INTERVALS = 2
POST_TAIL_GAP_HEAD_MIN_DURATION = 2.5
POST_TAIL_GAP_HEAD_MAX_EARLY_ONSET_OFFSET = 0.45
POST_TAIL_GAP_HEAD_MIN_LATE_ONSET_OFFSET = 0.9
POST_TAIL_GAP_HEAD_MIN_INTERVAL = 0.45
POST_TAIL_GAP_HEAD_MAX_INTERVAL = 0.75
POST_TAIL_GAP_HEAD_MIN_TRAILING_EDGE = 0.45
POST_TAIL_GAP_HEAD_MAX_NEXT_RANGE_DURATION = 0.45
GAP_RUN_LEAD_IN_MIN_FOLLOWUP_GAP = 0.9
SPARSE_GAP_TAIL_MIN_DURATION = 1.0
SPARSE_GAP_TAIL_MIN_PREVIOUS_EDGE = 0.04
SPARSE_GAP_TAIL_MAX_ONSET_OFFSET = 0.45
SPARSE_GAP_TAIL_MIN_TRAILING_EDGE = 0.6
SPARSE_GAP_TAIL_MIN_INTERVAL = 0.08
SPARSE_GAP_TAIL_MAX_INTERVAL = 0.45
SPARSE_GAP_TAIL_SEGMENT_DURATION = 0.24
SINGLE_ONSET_GAP_HEAD_MIN_DURATION = 0.85
SINGLE_ONSET_GAP_HEAD_MAX_DURATION = 1.0
SINGLE_ONSET_GAP_HEAD_MIN_PREVIOUS_EDGE = 0.3
SINGLE_ONSET_GAP_HEAD_MIN_TRAILING_EDGE = 0.55
SINGLE_ONSET_GAP_HEAD_MAX_NEXT_RANGE_DURATION = 0.35
SINGLE_ONSET_GAP_HEAD_SEGMENT_DURATION = 0.24
SHORT_BRIDGE_ACTIVE_RANGE_MAX_DURATION = 0.16
SHORT_BRIDGE_ACTIVE_RANGE_MAX_ONSET_OFFSET = 0.03
SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_ONSET_GAP = 0.1
SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_ONSET_GAP = 0.3
SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_EDGE_GAP = 0.08
SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_EDGE_GAP = 0.2
CLOSE_TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE = 0.05
CLOSE_TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE = 0.18
CLOSE_TERMINAL_ORPHAN_SEGMENT_DURATION = 0.24
DELAYED_TERMINAL_ORPHAN_MIN_GAP_AFTER_ACTIVE = 0.35
DELAYED_TERMINAL_ORPHAN_MAX_GAP_AFTER_ACTIVE = 0.65
DELAYED_TERMINAL_ORPHAN_SEGMENT_DURATION = 0.24
DELAYED_TERMINAL_ORPHAN_MIN_BASE_DURATION = 0.24
DELAYED_TERMINAL_ORPHAN_MAX_BASE_DURATION = 0.4
TERMINAL_MULTI_ONSET_MIN_COUNT = 4
TERMINAL_MULTI_ONSET_MIN_INTERVAL = 0.18
TERMINAL_MULTI_ONSET_MAX_INTERVAL = 0.55
TERMINAL_MULTI_ONSET_TAIL_DURATION = 0.32
ACTIVE_RANGE_START_CLUSTER_MIN_GAP = 0.45
ACTIVE_RANGE_START_CLUSTER_MAX_SPAN = 0.09
ACTIVE_RANGE_START_CLUSTER_MAX_DURATION = 0.35
CLUSTERED_RANGE_HEAD_MIN_DURATION = 0.05
ACTIVE_RANGE_HEAD_CLUSTER_MAX_OFFSET = 0.3
ACTIVE_RANGE_HEAD_CLUSTER_MAX_INTERVAL = 0.14
ACTIVE_RANGE_HEAD_CLUSTER_MIN_DURATION = 0.35
ATTACK_ANALYSIS_SECONDS = 0.16
ATTACK_ANALYSIS_RATIO = 0.35
ONSET_ENERGY_WINDOW_SECONDS = 0.08
SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY = 2000.0
ONSET_ATTACK_MIN_BROADBAND_GAIN = 10.0
ONSET_ATTACK_MIN_HIGH_BAND_FLUX = 1.5
ONSET_ATTACK_GAIN_REQUIRES_MIN_FLUX = 0.5
ATTACK_VALIDATED_GAP_SEGMENT_DURATION = 0.24
ATTACK_REFINED_ONSET_MAX_INTERVAL = 0.15
USE_ATTACK_VALIDATED_GAP_COLLECTOR = False
FILTER_GAP_ONSETS_BY_ATTACK_PROFILE = True
ABLATE_LEADING_ORPHAN = False
ABLATE_CLOSE_TERMINAL_ORPHAN = False
ABLATE_DELAYED_TERMINAL_ORPHAN = False
ABLATE_SINGLE_ONSET_GAP_HEAD = False
ABLATE_SPARSE_GAP_TAIL = False
ABLATE_MULTI_ONSET_GAP = False
ABLATE_POST_TAIL_GAP_HEAD = False
ABLATE_TERMINAL_MULTI_ONSET = False
ABLATE_GAP_INJECTED = False
ABLATE_TWO_ONSET_TERMINAL_TAIL = False
ABLATE_COLLAPSE_ACTIVE_RANGE_HEAD = False
ABLATE_SNAP_RANGE_START_TO_ONSET = False
MIN_RECENT_NOTE_ONSET_GAIN = 2.5
RECENT_PRIMARY_REPLACEMENT_MIN_SCORE_RATIO = 0.18
RECENT_PRIMARY_REPLACEMENT_MIN_FUNDAMENTAL_RATIO = 0.6
RECENT_PRIMARY_REPLACEMENT_RELAXED_FUNDAMENTAL_RATIO = 0.45
RECENT_PRIMARY_REPLACEMENT_STRONG_ONSET_GAIN = 100.0
RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_GAIN = 20.0
RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_RATIO = 8.0
RECENT_PRIMARY_REPLACEMENT_MAX_DURATION = 0.47
DESCENDING_REPEATED_PRIMARY_MAX_DURATION = 0.47
DESCENDING_REPEATED_PRIMARY_MAX_PRIMARY_ONSET_GAIN = 2.5
DESCENDING_REPEATED_PRIMARY_MIN_REPLACEMENT_ONSET_GAIN = 5.0
DESCENDING_REPEATED_PRIMARY_MIN_REPLACEMENT_ONSET_RATIO = 4.0
DESCENDING_REPEATED_PRIMARY_MIN_SCORE_RATIO = 0.35
DESCENDING_REPEATED_PRIMARY_MIN_FUNDAMENTAL_RATIO = 0.9
DESCENDING_REPEATED_PRIMARY_MIN_INTERVAL_CENTS = 80.0
DESCENDING_REPEATED_PRIMARY_MAX_INTERVAL_CENTS = 220.0
RECENT_UPPER_SECONDARY_MIN_DURATION = 0.22
RECENT_UPPER_SECONDARY_PRIMARY_ONSET_GAIN = 20.0
UPPER_SECONDARY_WEAK_ONSET_MIN_DURATION = 0.4
UPPER_SECONDARY_WEAK_ONSET_MAX_GAIN = 30.0
UPPER_SECONDARY_WEAK_ONSET_SCORE_RATIO = 0.14
SHORT_SECONDARY_WEAK_ONSET_MAX_DURATION = 0.36
SHORT_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN = 15.0
SHORT_SECONDARY_WEAK_ONSET_MAX_GAIN = 2.5
SHORT_SECONDARY_WEAK_ONSET_MAX_RATIO = 0.08
SHORT_SECONDARY_WEAK_ONSET_SCORE_RATIO = 0.55
LOWER_SECONDARY_WEAK_ONSET_MAX_DURATION = 0.45
LOWER_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN = 20.0
LOWER_SECONDARY_WEAK_ONSET_MAX_GAIN = 2.5
LOWER_SECONDARY_WEAK_ONSET_MAX_RATIO = 0.08
LOWER_SECONDARY_WEAK_ONSET_SCORE_RATIO = 0.35
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_DURATION = 0.32
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_DURATION = 0.12
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_PRIMARY_ONSET_GAIN = 1.5
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_PRIMARY_FUNDAMENTAL_RATIO = 0.85
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO = 0.95
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_SCORE_RATIO = 0.25
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_ALIAS_RATIO = 1.2
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_SUPPORTING_SCORE_RATIO = 0.75
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_SCORE = 121.0
RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MAX_DURATION = 0.35
RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MAX_PRIMARY_FUNDAMENTAL_RATIO = 0.7
RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO = 0.95
RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_UPPER_ALIAS_RATIO = 0.8
RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_UPPER_SCORE_RATIO = 0.7
RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_SUPPORTING_LOWER_SCORE_RATIO = 0.45
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_LOWER_SCORE_RATIO = 0.4
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_LOWER_FUNDAMENTAL_RATIO = 0.85
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO = 0.94
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_UPPER_SCORE_RATIO = 0.15
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_PRIMARY_SCORE_RATIO = 0.04
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_FUNDAMENTAL_RATIO_DELTA = 0.12
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_PRIMARY_INTERVAL_CENTS = 1000.0
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_PRIMARY_INTERVAL_CENTS = 700.0
ASCENDING_PRIMARY_RUN_MIN_LENGTH = 3
ASCENDING_PRIMARY_RUN_MAX_DURATION = 0.45
ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO = 0.35
ASCENDING_PRIMARY_RUN_RECENT_SECONDARY_ONSET_GAIN = 8.0
DESCENDING_PRIMARY_SUFFIX_MIN_LENGTH = 2
DESCENDING_PRIMARY_SUFFIX_MAX_DURATION = 0.5
DESCENDING_PRIMARY_SUFFIX_PRIMARY_ONSET_GAIN = 10.0
DESCENDING_PRIMARY_SUFFIX_UPPER_SCORE_RATIO = 0.6
DESCENDING_REPEATED_PRIMARY_STALE_UPPER_MAX_ONSET_GAIN = 1.5
DESCENDING_REPEATED_PRIMARY_STALE_UPPER_SCORE_RATIO = 0.95
ADJACENT_SEPARATED_DYAD_MAX_DURATION = 0.6
ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT = 2
PRIOR_ONSET_BACKTRACK_SECONDS = 0.55
HARMONIC_WEIGHTS = [1.0, 0.55, 0.3, 0.15]
HARMONIC_BAND_CENTS = 40.0
SUPPRESSION_BAND_CENTS = 45.0
MAX_POLYPHONY = 4
TERTIARY_MIN_SCORE_RATIO = 0.06
TERTIARY_MIN_FUNDAMENTAL_RATIO = 0.90
TERTIARY_MIN_ONSET_GAIN = 1.8
QUATERNARY_MIN_SCORE_RATIO = 0.30
GLISS_TERTIARY_MAX_RELATIVE_ONSET_GAIN = 0.12
GLISS_CLUSTER_MAX_GAP = 0.06
GLISS_CLUSTER_MAX_EVENT_DURATION = 0.85
GLISS_CLUSTER_MAX_TOTAL_DURATION = 1.2
GLISS_CLUSTER_TARGET_NOTE_COUNT = 3
GLISS_LEADING_SUBSET_MAX_DURATION = 0.18
GLISS_LEADING_SUBSET_SCORE_RATIO = 4.0
GLISS_TERTIARY_MAX_DURATION = 1.35
GLISS_TERTIARY_SCORE_RATIO = 0.02
GLISS_TERTIARY_MIN_SCORE = 20.0
GLISS_TERTIARY_STRONG_ONSET_GAIN = 5.0
GLISS_TERTIARY_WEAK_ONSET_GAIN = 2.0
GLISS_TERTIARY_MIN_FUNDAMENTAL_RATIO = 0.9
LOWER_MIXED_ROLL_EXTENSION_MAX_DURATION = 1.25
LOWER_MIXED_ROLL_EXTENSION_MIN_EXTENSION_SCORE_RATIO = 0.35
LOWER_MIXED_ROLL_EXTENSION_MIN_UPPER_SCORE_RATIO = 0.4
LOWER_MIXED_ROLL_EXTENSION_MIN_FUNDAMENTAL_RATIO = 0.97
LOWER_MIXED_ROLL_EXTENSION_MIN_PRIMARY_ONSET_GAIN = 25.0
LOWER_MIXED_ROLL_EXTENSION_MIN_EXTENSION_ONSET_GAIN = 20.0
LOWER_ROLL_TAIL_EXTENSION_MAX_DURATION = 0.4
LOWER_ROLL_TAIL_EXTENSION_MIN_FUNDAMENTAL_RATIO = 0.95
LOWER_ROLL_TAIL_EXTENSION_MIN_PRIMARY_ONSET_GAIN = 25.0
LOWER_ROLL_TAIL_EXTENSION_MAX_ONSET_GAIN = 5.0
LOWER_ROLL_TAIL_EXTENSION_MIN_SCORE_RATIO = 0.9
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_DURATION = 0.55
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_SCORE_RATIO = 0.08
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_FUNDAMENTAL_RATIO = 0.9
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_MARGIN_RATIO = 1.15
FOUR_NOTE_GLISS_EXTENSION_MAX_DURATION = 1.0
FOUR_NOTE_GLISS_EXTENSION_SCORE_RATIO = 0.02
FOUR_NOTE_GLISS_EXTENSION_MIN_SCORE = 18.0
FOUR_NOTE_GLISS_EXTENSION_MIN_FUNDAMENTAL_RATIO = 0.82
CHORD_CLUSTER_MAX_GAP = 0.08
CHORD_CLUSTER_MAX_SINGLETON_DURATION = 0.22
CHORD_CLUSTER_MAX_TOTAL_DURATION = 1.6
REPEATED_PATTERN_LOCAL_CONTEXT_MAX_GAP = 0.35
MAX_HARMONIC_MULTIPLE = 4
SECONDARY_SCORE_RATIO = 0.12
SECONDARY_MIN_FUNDAMENTAL_RATIO = 0.18
SHORT_SEGMENT_SECONDARY_SCORE_RATIO = 0.06
OVERTONE_DOMINANT_FUNDAMENTAL_RATIO = 0.18
OVERTONE_DOMINANT_PENALTY_WEIGHT = 0.0
OCTAVE_ALIAS_RATIO_THRESHOLD = 1.15
OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO = 0.34
OCTAVE_ALIAS_PENALTY = 0.85

OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO = 0.32
OCTAVE_DYAD_LOWER_OCTAVE4_MIN_FUNDAMENTAL_RATIO = 0.85
OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO = 0.06
OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO = 0.95
OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO = 0.05
OCTAVE_DYAD_UPPER_SCORE_RATIO = 0.03
LOW_CONFIDENCE_DYAD_MAX_DURATION = 0.25
LOW_CONFIDENCE_DYAD_MAX_SCORE = 120.0
SHORT_SECONDARY_STRIP_MAX_DURATION = 0.28
SHORT_SECONDARY_STRIP_MIN_SCORE = 60.0
SHORT_SECONDARY_STRIP_NEXT_SCORE_RATIO = 5.0
RESTART_STALE_UPPER_STRIP_MIN_INTERVAL_CENTS = 1800.0
RESTART_STALE_UPPER_STRIP_MAX_DURATION = 0.24
ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS = 80.0
ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS = 350.0
DESCENDING_STEP_HANDOFF_MAX_DURATION = 0.46
DESCENDING_ADJACENT_UPPER_CARRYOVER_MAX_DURATION = 0.24
DESCENDING_ADJACENT_UPPER_PRIMARY_ONSET_GAIN = 5.0
DESCENDING_ADJACENT_UPPER_SCORE_RATIO = 1.5
DESCENDING_RESTART_UPPER_CARRYOVER_MAX_DURATION = 0.45
DESCENDING_RESTART_UPPER_PRIMARY_ONSET_GAIN = 20.0
DESCENDING_RESTART_UPPER_SCORE_RATIO = 0.2
DESCENDING_PRIMARY_BAND_MIN_LENGTH = 4
DESCENDING_PRIMARY_SUFFIX_UPPER_CARRYOVER_MAX_DURATION = 0.45
DESCENDING_PRIMARY_BAND_PRIMARY_ONSET_GAIN = 5.0
DESCENDING_PRIMARY_SUFFIX_UPPER_SCORE_RATIO = 0.75
RESONANT_CARRYOVER_PHRASE_RESET_MIN_GAP = 0.45
RESONANT_CARRYOVER_HIGH_RETURN_MAX_DURATION = 0.24
RESONANT_CARRYOVER_HIGH_RETURN_MIN_INTERVAL_CENTS = 1800.0
RESONANT_CARRYOVER_HIGH_RETURN_MAX_NEXT_GAP = 0.12
LEADING_SINGLE_TRANSIENT_MAX_DURATION = 0.3
LEADING_SINGLE_TRANSIENT_MAX_SCORE = 150.0
LEADING_SINGLE_TRANSIENT_NEXT_SCORE_RATIO = 8.0
VERY_LOW_CONFIDENCE_EVENT_MAX_SCORE = 2.0
BRIDGING_OCTAVE_PAIR_MAX_DURATION = 0.4
BRIDGING_OCTAVE_PAIR_MAX_SCORE = 600.0
SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION = 0.28
SPLIT_UPPER_OCTAVE_PAIR_MAX_DURATION = 0.7
SPLIT_UPPER_OCTAVE_PAIR_PRIMARY_SCORE_MAX = 800.0
SPLIT_UPPER_OCTAVE_PAIR_FRACTION = 0.45
SAME_START_PRIMARY_SINGLETON_MAX_START_DELTA = 0.02
OVERLAPPING_PRIMARY_SINGLETON_MIN_START_DELTA = 0.05
OVERLAPPING_PRIMARY_SINGLETON_MIN_OVERLAP = 0.08
OVERLAPPING_PRIMARY_SINGLETON_MAX_DURATION = 0.4

PITCH_CLASS_TO_DOREMI = {
    "C": "\u30c9",
    "C#": "\u30c9#",
    "Db": "\u30ecb",
    "D": "\u30ec",
    "D#": "\u30ec#",
    "Eb": "\u30dfb",
    "E": "\u30df",
    "F": "\u30d5\u30a1",
    "F#": "\u30d5\u30a1#",
    "Gb": "\u30bdb",
    "G": "\u30bd",
    "G#": "\u30bd#",
    "Ab": "\u30e9b",
    "A": "\u30e9",
    "A#": "\u30e9#",
    "Bb": "\u30b7b",
    "B": "\u30b7",
}

PITCH_CLASS_TO_NUMBER = {
    "C": "1",
    "C#": "#1",
    "Db": "b2",
    "D": "2",
    "D#": "#2",
    "Eb": "b3",
    "E": "3",
    "F": "4",
    "F#": "#4",
    "Gb": "b5",
    "G": "5",
    "G#": "#5",
    "Ab": "b6",
    "A": "6",
    "A#": "#6",
    "Bb": "b7",
    "B": "7",
}

@dataclass
class NoteCandidate:
    key: int
    note_name: str
    frequency: float
    pitch_class: str
    octave: int

@dataclass
class RawEvent:
    start_time: float
    end_time: float
    notes: list[NoteCandidate]
    is_gliss_like: bool
    primary_note_name: str = ""
    primary_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RepeatedPatternPass:
    name: str
    fn: Callable[[list["RawEvent"]], list["RawEvent"]]
    merge_after: bool = True


@dataclass(slots=True)
class NoteHypothesis:
    candidate: NoteCandidate
    score: float
    fundamental_energy: float
    overtone_energy: float
    fundamental_ratio: float
    subharmonic_alias_energy: float
    octave_alias_energy: float
    octave_alias_ratio: float
    octave_alias_penalty: float
    second_harmonic_energy: float = 0.0
    harmonics: list[dict[str, float]] | None = None
    subharmonics: list[dict[str, float]] | None = None

def parse_tuning_json(tuning_json: str) -> InstrumentTuning:
    try:
        payload: dict[str, Any] = json.loads(tuning_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid tuning JSON.") from exc

    notes = payload.get("notes", [])
    if not notes:
        raise HTTPException(status_code=400, detail="Tuning must contain at least one note.")

    note_names = [note["noteName"] for note in notes if "noteName" in note]
    if len(note_names) != len(notes):
        raise HTTPException(status_code=400, detail="Each tuning note must include noteName.")

    return build_custom_tuning(payload.get("name", "Custom Tuning"), note_names)


def parse_disabled_repeated_pattern_passes(raw_value: str | None) -> frozenset[str]:
    if raw_value is None or not raw_value.strip():
        return frozenset()

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw_value.split(",") if item.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(status_code=400, detail="disabledRepeatedPatternPasses must be a JSON string array or comma-separated string.")

    disabled = frozenset(item.strip() for item in parsed if item.strip())
    unknown = sorted(disabled - set(REPEATED_PATTERN_PASS_IDS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown repeated-pattern pass ids: {', '.join(unknown)}")
    return disabled


async def read_audio(upload: UploadFile) -> tuple[np.ndarray, int]:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Send WAV audio from the web client.") from exc

    if audio.ndim > 1:
        audio = audio[:, 0]

    if float(np.max(np.abs(audio))) < 1e-4:
        raise HTTPException(status_code=422, detail="Audio appears to be silent.")

    return audio.astype(np.float32), sample_rate

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    centered = audio - np.mean(audio)
    peak = np.max(np.abs(centered))
    if peak < 1e-6:
        return centered
    return centered / peak

def cents_distance(freq_a: float, freq_b: float) -> float:
    return abs(1200.0 * math.log2(freq_a / freq_b))

def snap_frequency_to_tuning(freq: float, tuning: InstrumentTuning) -> NoteCandidate | None:
    best_note = None
    best_distance = float("inf")

    for note in tuning.notes:
        distance = cents_distance(freq, note.frequency)
        if distance < best_distance:
            best_note = note
            best_distance = distance

    if best_note is None or best_distance > 80:
        return None

    pitch_class, octave = parse_note_name(best_note.note_name)
    return NoteCandidate(
        key=best_note.key,
        note_name=best_note.note_name,
        frequency=best_note.frequency,
        pitch_class=pitch_class,
        octave=octave,
    )

def merge_time_ranges(ranges: list[tuple[float, float]], gap_tolerance: float = 0.06) -> list[tuple[float, float]]:
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + gap_tolerance:
            merged[-1] = (previous_start, max(previous_end, end))
            continue
        merged.append((start, end))

    return merged

def dedupe_nested_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(segments) < 2:
        return segments

    deduped: list[tuple[float, float]] = []
    for start_time, end_time in sorted(segments):
        if deduped:
            previous_start, previous_end = deduped[-1]
            same_start = abs(start_time - previous_start) <= NESTED_SEGMENT_DEDUP_MAX_START_DELTA
            if same_start:
                if end_time <= previous_end:
                    continue
                deduped[-1] = (previous_start, end_time)
                continue
        deduped.append((start_time, end_time))

    return deduped


def trim_small_overlapping_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(segments) < 2:
        return segments

    trimmed: list[tuple[float, float]] = [segments[0]]
    for start_time, end_time in segments[1:]:
        previous_start, previous_end = trimmed[-1]
        overlap = previous_end - start_time
        duration = end_time - start_time
        if (
            overlap > 0
            and overlap <= SEGMENT_OVERLAP_TRIM_MAX_OVERLAP
            and duration >= SEGMENT_OVERLAP_TRIM_MIN_DURATION
        ):
            adjusted_start = previous_end
            if end_time - adjusted_start >= 0.08:
                trimmed.append((adjusted_start, end_time))
                continue
        trimmed.append((start_time, end_time))

    return trimmed


def should_keep_dense_trailing_onset(
    boundary_times: list[float],
    index: int,
    range_start: float,
    range_end: float,
) -> bool:
    if index <= 0 or index >= len(boundary_times) - 1:
        return False

    previous_time = boundary_times[index - 1]
    current_time = boundary_times[index]
    next_time = boundary_times[index + 1]
    return (
        len(boundary_times) >= 4
        and range_end - range_start >= 2.5
        and current_time - previous_time >= 0.1
        and current_time - previous_time < 0.18
        and next_time - current_time >= 0.28
        and current_time - range_start >= 1.5
        and range_end - current_time <= 1.0
        and range_end - current_time >= 0.18
    )




def should_keep_short_range_trailing_onset(
    boundary_times: list[float],
    index: int,
    range_start: float,
    range_end: float,
) -> bool:
    if index < 1 or index != len(boundary_times) - 1:
        return False

    previous_time = boundary_times[index - 1]
    current_time = boundary_times[index]
    previous_previous_time = range_start if index == 1 else boundary_times[index - 2]
    range_duration = range_end - range_start
    return (
        0.9 <= range_duration <= 1.5
        and current_time - previous_time >= 0.1
        and current_time - previous_time <= 0.16
        and previous_time - previous_previous_time >= 0.25
        and previous_time - previous_previous_time <= 0.45
        and range_end - current_time >= 0.28
        and range_end - current_time <= 0.45
    )


def _lookup_onset_attack_profile(
    onset_profiles: dict[float, OnsetAttackProfile] | None,
    onset_time: float,
) -> OnsetAttackProfile | None:
    if onset_profiles is None:
        return None
    return onset_profiles.get(round(onset_time, 4))




def should_snap_range_start_to_first_onset(
    range_start: float,
    first_onset: float,
    onset_profiles: dict[float, OnsetAttackProfile],
) -> bool:
    if first_onset - range_start > ATTACK_REFINED_ONSET_MAX_INTERVAL:
        return False
    first_profile = _lookup_onset_attack_profile(onset_profiles, first_onset)
    return first_profile is not None and first_profile.is_valid_attack

def collapse_active_range_head_onsets(
    effective_range_start: float,
    range_end: float,
    range_onsets: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
) -> list[float]:
    if not range_onsets:
        return range_onsets
    if range_end - effective_range_start < ACTIVE_RANGE_HEAD_CLUSTER_MIN_DURATION:
        return range_onsets

    head_cluster: list[float] = []
    previous_time = effective_range_start
    for onset_time in range_onsets:
        if onset_time - effective_range_start > ACTIVE_RANGE_HEAD_CLUSTER_MAX_OFFSET:
            break
        if onset_time - previous_time > ACTIVE_RANGE_HEAD_CLUSTER_MAX_INTERVAL:
            break
        profile = _lookup_onset_attack_profile(onset_profiles, onset_time)
        if profile is None:
            break
        if not profile.is_valid_attack and not head_cluster:
            break
        head_cluster.append(onset_time)
        previous_time = onset_time

    anchor_profile = _lookup_onset_attack_profile(onset_profiles, effective_range_start)
    anchor_valid = anchor_profile is not None and anchor_profile.is_valid_attack
    if len(head_cluster) + (1 if anchor_valid else 0) < 2:
        return range_onsets
    if len(range_onsets) > 3:
        return range_onsets

    head_cluster_set = {round(onset_time, 4) for onset_time in head_cluster}
    return [onset_time for onset_time in range_onsets if round(onset_time, 4) not in head_cluster_set]

@dataclass(frozen=True)
class GapAttackCandidates:
    inter_ranges: list[list[float]]
    leading: list[float]
    trailing: list[float]


LEADING_GAP_START_MARGIN = 0.05
GAP_ONSET_MIN_BROADBAND_GAIN = 0.95
GAP_ONSET_MAX_KURTOSIS = 2.0
GAP_ONSET_MAX_POST_CREST = 0.0  # disabled; set to e.g. 3.8 to enable


@dataclass(frozen=True)
class OnsetWaveformStats:
    kurtosis: float
    crest: float


def precompute_onset_waveform_stats(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
) -> dict[float, OnsetWaveformStats]:
    """Pre-compute kurtosis and crest factor for each onset (20ms post-onset window)."""
    stats: dict[float, OnsetWaveformStats] = {}
    window_samples = int(sample_rate * _KURTOSIS_WINDOW_SECONDS)
    for onset_time in onset_times:
        onset_sample = max(int(onset_time * sample_rate), 0)
        seg = np.array(audio[onset_sample:min(len(audio), onset_sample + window_samples)], copy=True)
        stats[round(onset_time, 4)] = OnsetWaveformStats(
            kurtosis=_waveform_kurtosis(seg),
            crest=_waveform_crest_factor(seg),
        )
    return stats


def _valid_attack_gap_onsets(
    gap_start: float,
    gap_end: float,
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
    start_margin: float = 0.05,
    waveform_stats: dict[float, OnsetWaveformStats] | None = None,
) -> list[float]:
    valid: list[float] = []
    for time in onset_times:
        if not (gap_start + start_margin < time < gap_end - 0.05):
            continue
        profile = onset_profiles.get(round(time, 4))
        if profile is None or not profile.is_valid_attack:
            continue
        if profile.broadband_onset_gain < GAP_ONSET_MIN_BROADBAND_GAIN:
            continue
        if waveform_stats is not None:
            ws = waveform_stats.get(round(time, 4))
            if ws is not None:
                if GAP_ONSET_MAX_KURTOSIS > 0 and ws.kurtosis > GAP_ONSET_MAX_KURTOSIS:
                    continue
                if GAP_ONSET_MAX_POST_CREST > 0 and ws.crest > GAP_ONSET_MAX_POST_CREST:
                    continue
        valid.append(time)
    return valid


def collect_attack_validated_gap_candidates(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
    audio_duration: float,
    waveform_stats: dict[float, OnsetWaveformStats] | None = None,
) -> GapAttackCandidates:
    inter_ranges: list[list[float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        inter_ranges.append(_valid_attack_gap_onsets(
            previous_end, next_start, onset_times, onset_profiles,
            waveform_stats=waveform_stats,
        ))

    leading: list[float] = []
    trailing: list[float] = []
    if active_ranges:
        leading = _valid_attack_gap_onsets(
            0.0, active_ranges[0][0], onset_times, onset_profiles,
            start_margin=LEADING_GAP_START_MARGIN, waveform_stats=waveform_stats,
        )
        trailing = _valid_attack_gap_onsets(
            active_ranges[-1][1], audio_duration + 0.06, onset_times, onset_profiles,
            waveform_stats=waveform_stats,
        )

    return GapAttackCandidates(inter_ranges=inter_ranges, leading=leading, trailing=trailing)


CANDIDATE_PROMOTION_MIN_CANDIDATES = 1
CANDIDATE_PROMOTION_MIN_EDGE_DISTANCE = 0.3
CANDIDATE_PROMOTION_MIN_SEGMENT_DURATION = 0.08
CANDIDATE_PROMOTION_SEGMENT_DURATION = 0.32
CANDIDATE_PROMOTION_CLUSTER_MAX_INTERVAL = 0.1


def _cluster_gap_candidates(candidate_onsets: list[float]) -> list[float]:
    if not candidate_onsets:
        return []
    clusters: list[list[float]] = [[candidate_onsets[0]]]
    for onset_time in candidate_onsets[1:]:
        if onset_time - clusters[-1][-1] <= CANDIDATE_PROMOTION_CLUSTER_MAX_INTERVAL:
            clusters[-1].append(onset_time)
        else:
            clusters.append([onset_time])
    return [cluster[-1] for cluster in clusters]


def _promote_gap_candidates_by_structure(
    candidate_onsets: list[float],
    gap_start: float,
    gap_end: float,
) -> list[tuple[float, float]]:
    clustered = _cluster_gap_candidates(candidate_onsets)
    if len(clustered) < CANDIDATE_PROMOTION_MIN_CANDIDATES:
        return []

    eligible = [
        t for t in clustered
        if t - gap_start >= CANDIDATE_PROMOTION_MIN_EDGE_DISTANCE
        and gap_end - t >= CANDIDATE_PROMOTION_MIN_EDGE_DISTANCE
    ]
    if len(eligible) < CANDIDATE_PROMOTION_MIN_CANDIDATES:
        return []

    segments: list[tuple[float, float]] = []
    for i, onset_time in enumerate(eligible):
        if i + 1 < len(eligible):
            end_time = eligible[i + 1]
        else:
            end_time = min(onset_time + CANDIDATE_PROMOTION_SEGMENT_DURATION, gap_end - 0.08)
        if end_time - onset_time >= CANDIDATE_PROMOTION_MIN_SEGMENT_DURATION:
            segments.append((onset_time, end_time))

    return segments


def collect_multi_onset_gap_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
    gap_attack_candidates: GapAttackCandidates | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        gap_duration = next_start - previous_end
        if gap_duration < MULTI_ONSET_GAP_MIN_DURATION:
            continue

        gap_onsets = [
            time
            for time in onset_times
            if previous_end + 0.05 < time < next_start - 0.05
        ]
        if len(gap_onsets) < 3:
            continue

        if (
            gap_onsets[0] - previous_end < MULTI_ONSET_GAP_MIN_EDGE_SPACING
            or next_start - gap_onsets[-1] < MULTI_ONSET_GAP_MIN_EDGE_SPACING
        ):
            continue

        intervals = [gap_onsets[i + 1] - gap_onsets[i] for i in range(len(gap_onsets) - 1)]
        short_interval_count = sum(
            1
            for interval in intervals
            if MULTI_ONSET_GAP_MIN_INTERVAL <= interval <= MULTI_ONSET_GAP_MAX_INTERVAL
        )
        if short_interval_count >= MULTI_ONSET_GAP_MIN_SHORT_INTERVALS:
            for start_time, end_time in zip(gap_onsets, gap_onsets[1:]):
                if end_time - start_time >= 0.08:
                    segments.append((start_time, end_time))

            trailing_gap = next_start - gap_onsets[-1]
            if trailing_gap > PRIOR_ONSET_BACKTRACK_SECONDS and trailing_gap >= 0.08:
                segments.append((gap_onsets[-1], next_start))
            continue

        candidate_onsets = (
            gap_attack_candidates.inter_ranges[index]
            if gap_attack_candidates is not None and index < len(gap_attack_candidates.inter_ranges)
            else []
        )
        if candidate_onsets:
            segments.extend(_promote_gap_candidates_by_structure(candidate_onsets, previous_end, next_start))

    return segments




def collect_post_tail_gap_head_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start, next_end = active_ranges[index + 1]
        gap_duration = next_start - previous_end
        if gap_duration < POST_TAIL_GAP_HEAD_MIN_DURATION:
            continue

        gap_onsets = [
            onset_time
            for onset_time in onset_times
            if previous_end + SPARSE_GAP_TAIL_MIN_PREVIOUS_EDGE < onset_time < next_start - 0.05
        ]
        if len(gap_onsets) < 4:
            continue

        early_gap_onsets = [
            onset_time
            for onset_time in gap_onsets
            if onset_time - previous_end <= POST_TAIL_GAP_HEAD_MAX_EARLY_ONSET_OFFSET
        ]
        late_gap_onsets = [
            onset_time
            for onset_time in gap_onsets
            if onset_time - previous_end >= POST_TAIL_GAP_HEAD_MIN_LATE_ONSET_OFFSET
        ]
        if not (1 <= len(early_gap_onsets) <= 2):
            continue
        if len(late_gap_onsets) < 3:
            continue
        if next_end - next_start > POST_TAIL_GAP_HEAD_MAX_NEXT_RANGE_DURATION:
            continue
        if next_start - late_gap_onsets[-1] < POST_TAIL_GAP_HEAD_MIN_TRAILING_EDGE:
            continue

        late_intervals = [late_gap_onsets[i + 1] - late_gap_onsets[i] for i in range(len(late_gap_onsets) - 1)]
        if not all(
            POST_TAIL_GAP_HEAD_MIN_INTERVAL <= interval <= POST_TAIL_GAP_HEAD_MAX_INTERVAL
            for interval in late_intervals
        ):
            continue

        for start_time, end_time in zip(late_gap_onsets, late_gap_onsets[1:]):
            if end_time - start_time >= 0.08:
                segments.append((start_time, end_time))

    return segments

def collect_leading_orphan_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    first_range_start = active_ranges[0][0]
    leading_onsets = [
        onset_time
        for onset_time in onset_times
        if first_range_start - LEADING_ORPHAN_ONSET_MAX_GAP_BEFORE_ACTIVE
        <= onset_time
        <= first_range_start - LEADING_ORPHAN_ONSET_MIN_GAP_BEFORE_ACTIVE
    ]
    if len(leading_onsets) != 1:
        return []

    orphan_start = leading_onsets[0]
    orphan_end = min(orphan_start + LEADING_ORPHAN_SEGMENT_DURATION, first_range_start - 0.08)
    if orphan_end - orphan_start < 0.08:
        return []
    return [(orphan_start, orphan_end)]


def collect_sparse_gap_tail_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        gap_duration = next_start - previous_end
        if gap_duration < SPARSE_GAP_TAIL_MIN_DURATION:
            continue

        gap_onsets = [
            onset_time
            for onset_time in onset_times
            if previous_end + SPARSE_GAP_TAIL_MIN_PREVIOUS_EDGE < onset_time < next_start - 0.05
        ]
        early_gap_onsets = [
            onset_time
            for onset_time in gap_onsets
            if onset_time - previous_end <= SPARSE_GAP_TAIL_MAX_ONSET_OFFSET
        ]
        if not (1 <= len(early_gap_onsets) <= 2):
            continue
        if next_start - early_gap_onsets[-1] < SPARSE_GAP_TAIL_MIN_TRAILING_EDGE:
            continue

        if len(early_gap_onsets) == 1:
            orphan_start = early_gap_onsets[0]
            orphan_end = min(orphan_start + SPARSE_GAP_TAIL_SEGMENT_DURATION, next_start - 0.08)
            if orphan_end - orphan_start >= 0.08:
                segments.append((orphan_start, orphan_end))
            continue

        onset_interval = early_gap_onsets[1] - early_gap_onsets[0]
        if not (SPARSE_GAP_TAIL_MIN_INTERVAL <= onset_interval <= SPARSE_GAP_TAIL_MAX_INTERVAL):
            continue

        if early_gap_onsets[1] - early_gap_onsets[0] >= 0.08:
            segments.append((early_gap_onsets[0], early_gap_onsets[1]))
        orphan_start = early_gap_onsets[1]
        orphan_end = min(orphan_start + SPARSE_GAP_TAIL_SEGMENT_DURATION, next_start - 0.08)
        if orphan_end - orphan_start >= 0.08:
            segments.append((orphan_start, orphan_end))

    return segments


def collect_single_onset_gap_head_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start, next_end = active_ranges[index + 1]
        gap_duration = next_start - previous_end
        if not (SINGLE_ONSET_GAP_HEAD_MIN_DURATION <= gap_duration < SINGLE_ONSET_GAP_HEAD_MAX_DURATION):
            continue

        gap_onsets = [
            onset_time
            for onset_time in onset_times
            if previous_end + 0.05 < onset_time < next_start - 0.05
        ]
        if len(gap_onsets) != 1:
            continue

        gap_onset = gap_onsets[0]
        if gap_onset - previous_end < SINGLE_ONSET_GAP_HEAD_MIN_PREVIOUS_EDGE:
            continue
        if next_start - gap_onset < SINGLE_ONSET_GAP_HEAD_MIN_TRAILING_EDGE:
            continue
        if next_end - next_start > SINGLE_ONSET_GAP_HEAD_MAX_NEXT_RANGE_DURATION:
            continue

        segment_start = gap_onset
        segment_end = min(segment_start + SINGLE_ONSET_GAP_HEAD_SEGMENT_DURATION, next_start - 0.08)
        if segment_end - segment_start >= 0.08:
            segments.append((segment_start, segment_end))

    return segments


def collect_two_onset_terminal_tail_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    last_range_end = active_ranges[-1][1]
    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if last_range_end + TERMINAL_TWO_ONSET_TAIL_MIN_GAP_AFTER_ACTIVE
        <= onset_time
        <= audio_duration - 0.08
    ]
    if len(trailing_onsets) != 2:
        return []

    first_onset, second_onset = trailing_onsets
    if first_onset > last_range_end + TERMINAL_TWO_ONSET_TAIL_MAX_GAP_AFTER_ACTIVE:
        return []
    if not (TERMINAL_TWO_ONSET_TAIL_MIN_INTERVAL <= second_onset - first_onset <= TERMINAL_TWO_ONSET_TAIL_MAX_INTERVAL):
        return []

    segments: list[tuple[float, float]] = []
    first_end = min(first_onset + TERMINAL_TWO_ONSET_TAIL_SEGMENT_DURATION, second_onset - 0.08)
    if first_end - first_onset >= 0.08:
        segments.append((first_onset, first_end))

    second_end = min(second_onset + TERMINAL_TWO_ONSET_TAIL_SEGMENT_DURATION, audio_duration)
    if second_end - second_onset >= 0.08:
        segments.append((second_onset, second_end))

    return segments


def collect_attack_validated_gap_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
    audio_duration: float,
    gap_attack_candidates: GapAttackCandidates | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    candidates = gap_attack_candidates or collect_attack_validated_gap_candidates(active_ranges, onset_times, onset_profiles, audio_duration)

    # Inter-range gaps
    for index in range(len(active_ranges) - 1):
        next_start = active_ranges[index + 1][0]
        valid_onsets = candidates.inter_ranges[index] if index < len(candidates.inter_ranges) else []
        for onset_index, onset_time in enumerate(valid_onsets):
            if onset_index + 1 < len(valid_onsets):
                end_time = valid_onsets[onset_index + 1]
            else:
                end_time = min(onset_time + ATTACK_VALIDATED_GAP_SEGMENT_DURATION, next_start)
            if end_time - onset_time >= 0.08:
                segments.append((onset_time, end_time))

    # Leading gap (before first active range)
    if active_ranges:
        first_start = active_ranges[0][0]
        for onset_index, onset_time in enumerate(candidates.leading):
            if onset_index + 1 < len(candidates.leading):
                end_time = candidates.leading[onset_index + 1]
            else:
                end_time = min(onset_time + ATTACK_VALIDATED_GAP_SEGMENT_DURATION, first_start)
            if end_time - onset_time >= 0.08:
                segments.append((onset_time, end_time))

    # Trailing gap (after last active range)
    if active_ranges:
        for onset_index, onset_time in enumerate(candidates.trailing):
            if onset_index + 1 < len(candidates.trailing):
                end_time = candidates.trailing[onset_index + 1]
            else:
                end_time = min(onset_time + ATTACK_VALIDATED_GAP_SEGMENT_DURATION, audio_duration)
            if end_time - onset_time >= 0.08:
                segments.append((onset_time, end_time))

    return segments


def build_gap_ioi_diagnostics(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start, next_end = active_ranges[index + 1]
        gap_onsets = [
            onset_time
            for onset_time in onset_times
            if previous_end + 0.05 < onset_time < next_start - 0.05
        ]
        if not gap_onsets:
            continue

        previous_context = [onset_time for onset_time in onset_times if onset_time < gap_onsets[0]]
        next_context = [onset_time for onset_time in onset_times if onset_time > gap_onsets[-1]]
        previous_interval = gap_onsets[0] - previous_context[-1] if previous_context else None
        next_interval = next_context[0] - gap_onsets[-1] if next_context else None
        diagnostics.append(
            {
                "previousEnd": round(previous_end, 4),
                "nextStart": round(next_start, 4),
                "nextEnd": round(next_end, 4),
                "gapDuration": round(next_start - previous_end, 4),
                "nextRangeDuration": round(next_end - next_start, 4),
                "gapOnsets": [round(onset_time, 4) for onset_time in gap_onsets],
                "previousInterval": None if previous_interval is None else round(previous_interval, 4),
                "nextInterval": None if next_interval is None else round(next_interval, 4),
                "previousEdgeDistance": round(gap_onsets[0] - previous_end, 4),
                "nextEdgeDistance": round(next_start - gap_onsets[-1], 4),
            }
        )
    return diagnostics


def collect_close_terminal_orphan_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    last_range_end = active_ranges[-1][1]
    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE
        <= onset_time
        <= min(last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE, audio_duration - 0.08)
    ]
    if len(trailing_onsets) != 1:
        return []

    orphan_start = trailing_onsets[0]
    orphan_end = min(orphan_start + CLOSE_TERMINAL_ORPHAN_SEGMENT_DURATION, audio_duration)
    if orphan_end - orphan_start < 0.08:
        return []
    return [(orphan_start, orphan_end)]


def collect_delayed_terminal_orphan_segments(
    base_segment: tuple[float, float] | None,
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if base_segment is None:
        return []

    base_start, base_end = base_segment
    base_duration = base_end - base_start
    if not (
        DELAYED_TERMINAL_ORPHAN_MIN_BASE_DURATION
        <= base_duration
        <= DELAYED_TERMINAL_ORPHAN_MAX_BASE_DURATION
    ):
        return []

    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if base_end + DELAYED_TERMINAL_ORPHAN_MIN_GAP_AFTER_ACTIVE
        <= onset_time
        <= min(base_end + DELAYED_TERMINAL_ORPHAN_MAX_GAP_AFTER_ACTIVE, audio_duration - 0.08)
    ]
    if len(trailing_onsets) != 1:
        return []

    orphan_start = trailing_onsets[0]
    orphan_end = min(orphan_start + DELAYED_TERMINAL_ORPHAN_SEGMENT_DURATION, audio_duration)
    if orphan_end - orphan_start < 0.08:
        return []
    return [(orphan_start, orphan_end)]


def collect_terminal_multi_onset_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    last_range_end = active_ranges[-1][1]
    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if onset_time >= last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE
        and onset_time <= audio_duration - 0.08
    ]
    if len(trailing_onsets) < TERMINAL_MULTI_ONSET_MIN_COUNT:
        return []
    if trailing_onsets[0] > last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE:
        return []

    run_onsets = trailing_onsets[1:]
    if len(run_onsets) < 3:
        return []

    intervals = [run_onsets[i + 1] - run_onsets[i] for i in range(len(run_onsets) - 1)]
    if not intervals:
        return []
    if not all(TERMINAL_MULTI_ONSET_MIN_INTERVAL <= interval <= TERMINAL_MULTI_ONSET_MAX_INTERVAL for interval in intervals):
        return []

    segments: list[tuple[float, float]] = []
    for start_time, end_time in zip(run_onsets, run_onsets[1:]):
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))

    tail_start = run_onsets[-1]
    tail_end = min(tail_start + TERMINAL_MULTI_ONSET_TAIL_DURATION, audio_duration)
    if tail_end - tail_start >= 0.08:
        segments.append((tail_start, tail_end))
    return segments


def simplify_sparse_gap_tail_high_octave_dyad(candidates: list[NoteCandidate]) -> list[NoteCandidate]:
    if len(candidates) != 2:
        return candidates

    ordered = sorted(candidates, key=lambda candidate: candidate.frequency)
    lower, upper = ordered
    if upper.octave < 6:
        return candidates
    if lower.pitch_class != upper.pitch_class:
        return candidates
    if upper.octave - lower.octave != 1:
        return candidates
    return [upper]


def suppress_short_bridge_active_ranges(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(active_ranges) < 3:
        return active_ranges, []

    filtered_ranges: list[tuple[float, float]] = []
    suppressed_ranges: list[tuple[float, float]] = []

    for index, current_range in enumerate(active_ranges):
        if index == 0 or index == len(active_ranges) - 1:
            filtered_ranges.append(current_range)
            continue

        current_start, current_end = current_range
        current_duration = current_end - current_start
        if current_duration > SHORT_BRIDGE_ACTIVE_RANGE_MAX_DURATION:
            filtered_ranges.append(current_range)
            continue

        current_onsets = [time for time in onset_times if current_start <= time <= current_end]
        if len(current_onsets) != 1:
            filtered_ranges.append(current_range)
            continue

        current_onset = current_onsets[0]
        if current_onset - current_start > SHORT_BRIDGE_ACTIVE_RANGE_MAX_ONSET_OFFSET:
            filtered_ranges.append(current_range)
            continue

        next_start, _ = active_ranges[index + 1]
        next_prior_onsets = [
            time
            for time in onset_times
            if next_start - PRIOR_ONSET_BACKTRACK_SECONDS <= time <= next_start + 0.005
            and time >= current_end + 0.005
        ]
        if not next_prior_onsets:
            filtered_ranges.append(current_range)
            continue

        next_backtracked_onset = next_prior_onsets[-1]
        onset_gap = next_backtracked_onset - current_onset
        edge_gap = next_backtracked_onset - current_end
        if not (
            SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_ONSET_GAP
            <= onset_gap
            <= SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_ONSET_GAP
            and SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_EDGE_GAP
            <= edge_gap
            <= SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_EDGE_GAP
        ):
            filtered_ranges.append(current_range)
            continue

        suppressed_ranges.append(current_range)

    return filtered_ranges, suppressed_ranges


def should_keep_low_register_sparse_gap_tail(
    candidates: list[NoteCandidate],
    tuning: InstrumentTuning,
    descending_primary_suffix_floor: float | None,
    descending_primary_suffix_note_names: set[str],
) -> bool:
    if len(candidates) != 1 or descending_primary_suffix_floor is None:
        return False

    candidate = candidates[0]
    if candidate.octave >= 6:
        return False
    if candidate.note_name in descending_primary_suffix_note_names:
        return False

    sorted_notes = sorted(tuning.notes, key=lambda item: item.frequency)
    rank_by_name = {note.note_name: index for index, note in enumerate(sorted_notes)}
    suffix_floor_name = next((note.note_name for note in sorted_notes if abs(note.frequency - descending_primary_suffix_floor) < 1e-6), None)
    candidate_rank = rank_by_name.get(candidate.note_name)
    suffix_rank = rank_by_name.get(suffix_floor_name) if suffix_floor_name is not None else None
    if candidate_rank is None or suffix_rank is None:
        return False
    return candidate_rank == suffix_rank - 1


def _active_range_debug_context(
    range_index: int,
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
) -> dict[str, Any]:
    range_start, range_end = active_ranges[range_index]
    effective_range_start = range_start
    previous_range_end = active_ranges[range_index - 1][1] if range_index > 0 else None
    prior_onsets = [
        time
        for time in onset_times
        if range_start - PRIOR_ONSET_BACKTRACK_SECONDS <= time <= range_start + 0.005
        and (previous_range_end is None or time >= previous_range_end + 0.005)
    ]
    if prior_onsets:
        effective_range_start = prior_onsets[-1]
        if (
            previous_range_end is not None
            and range_start - previous_range_end >= ACTIVE_RANGE_START_CLUSTER_MIN_GAP
            and range_end - range_start <= ACTIVE_RANGE_START_CLUSTER_MAX_DURATION
        ):
            trailing_cluster = [
                time for time in prior_onsets if effective_range_start - time <= ACTIVE_RANGE_START_CLUSTER_MAX_SPAN
            ]
            if len(trailing_cluster) >= 2:
                effective_range_start = trailing_cluster[0]

    range_onsets = [time for time in onset_times if effective_range_start + 0.005 < time < range_end - 0.05]
    return {
        "activeRangeStart": round(range_start, 4),
        "activeRangeEnd": round(range_end, 4),
        "backtrackedStartTime": round(effective_range_start, 4),
        "activeRangeOnsets": [round(time, 4) for time in range_onsets],
        "activeRangeOnsetCount": len(range_onsets),
    }


def build_segment_debug_contexts(
    segments: list[tuple[float, float]],
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
) -> dict[tuple[float, float], dict[str, Any]]:
    active_contexts = [_active_range_debug_context(index, active_ranges, onset_times) for index in range(len(active_ranges))]
    segment_contexts: dict[tuple[float, float], dict[str, Any]] = {}
    for index, (start_time, end_time) in enumerate(segments):
        segment_key = (round(start_time, 4), round(end_time, 4))
        segment_onsets = [time for time in onset_times if start_time <= time <= end_time]
        context = {
            "previousGapSec": None if index == 0 else round(start_time - segments[index - 1][1], 4),
            "nextGapSec": None if index + 1 >= len(segments) else round(segments[index + 1][0] - end_time, 4),
            "segmentOnsets": [round(time, 4) for time in segment_onsets],
            "localOnsetCount": len(segment_onsets),
        }
        active_index = next(
            (range_index for range_index, (range_start, range_end) in enumerate(active_ranges) if start_time < range_end and end_time > range_start),
            None,
        )
        if active_index is not None:
            context.update(active_contexts[active_index])
        else:
            context.update(
                {
                    "activeRangeStart": None,
                    "activeRangeEnd": None,
                    "backtrackedStartTime": None,
                    "activeRangeOnsets": [],
                    "activeRangeOnsetCount": 0,
                }
            )
        segment_contexts[segment_key] = context
    return segment_contexts


def detect_segments(audio: np.ndarray, sample_rate: int) -> tuple[list[tuple[float, float]], float, dict[str, Any]]:
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=HOP_LENGTH)
    max_rms = float(np.max(rms))
    median_rms = float(np.median(rms))
    threshold = max(max_rms * 0.18, min(median_rms * 2.2, max_rms * RMS_MEDIAN_THRESHOLD_MAX_PEAK_RATIO), 0.01)
    active_frames = rms >= threshold

    active_ranges: list[tuple[float, float]] = []
    active_start = None
    for index, is_active in enumerate(active_frames):
        if is_active and active_start is None:
            active_start = index
        elif not is_active and active_start is not None:
            start_time = max(float(frame_times[active_start]) - 0.02, 0.0)
            end_time = float(frame_times[min(index, len(frame_times) - 1)]) + 0.08
            active_ranges.append((start_time, end_time))
            active_start = None

    if active_start is not None:
        active_ranges.append((max(float(frame_times[active_start]) - 0.02, 0.0), librosa.get_duration(y=audio, sr=sample_rate)))

    raw_active_ranges = active_ranges.copy()
    active_ranges = merge_time_ranges(active_ranges)

    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sample_rate, hop_length=HOP_LENGTH, backtrack=True)
    onset_times = [float(value) for value in librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=HOP_LENGTH)]
    onset_attack_profiles = precompute_onset_attack_profiles(audio, sample_rate, onset_times)
    onset_times = refine_onset_times_by_attack_profile(onset_times, onset_attack_profiles)
    active_ranges, short_bridge_active_ranges = suppress_short_bridge_active_ranges(active_ranges, onset_times)
    gap_onset_times = filter_gap_onsets_by_attack(onset_times, active_ranges, onset_attack_profiles) if FILTER_GAP_ONSETS_BY_ATTACK_PROFILE else onset_times
    gap_ioi_diagnostics = build_gap_ioi_diagnostics(active_ranges, onset_times)

    audio_duration = float(librosa.get_duration(y=audio, sr=sample_rate))
    waveform_stats = precompute_onset_waveform_stats(audio, sample_rate, gap_onset_times)
    # Unfiltered candidates for multi_onset_gap_segments (shared path)
    attack_validated_gap_candidates = collect_attack_validated_gap_candidates(
        active_ranges, gap_onset_times, onset_attack_profiles, audio_duration,
    )
    # Filtered candidates for attack-validated gap collector only
    filtered_gap_candidates = collect_attack_validated_gap_candidates(
        active_ranges, gap_onset_times, onset_attack_profiles, audio_duration,
        waveform_stats=waveform_stats,
    ) if USE_ATTACK_VALIDATED_GAP_COLLECTOR else None
    leading_orphan_segments = [] if ABLATE_LEADING_ORPHAN else collect_leading_orphan_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    gap_injected_segments: list[tuple[float, float]] = []
    multi_onset_gap_segments = [] if ABLATE_MULTI_ONSET_GAP else collect_multi_onset_gap_segments(
        active_ranges,
        gap_onset_times,
        onset_attack_profiles,
        attack_validated_gap_candidates,
    )
    post_tail_gap_head_segments = [] if ABLATE_POST_TAIL_GAP_HEAD else collect_post_tail_gap_head_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    single_onset_gap_head_segments = [] if ABLATE_SINGLE_ONSET_GAP_HEAD else collect_single_onset_gap_head_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    sparse_gap_tail_segments = [] if ABLATE_SPARSE_GAP_TAIL else collect_sparse_gap_tail_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    if not ABLATE_GAP_INJECTED:
        qualifying_gap_run: list[tuple[float, float]] = []
        for index in range(len(active_ranges) - 1):
            previous_end = active_ranges[index][1]
            next_start = active_ranges[index + 1][0]
            gap_onsets = [time for time in gap_onset_times if previous_end + 0.05 < time < next_start - 0.05]
            qualifies = (
                len(gap_onsets) == 1
                and gap_onsets[0] - previous_end >= 0.35
                and next_start - gap_onsets[0] >= 0.35
            )
            if qualifies:
                qualifying_gap_run.append((gap_onsets[0], next_start))
                continue
            if len(qualifying_gap_run) >= 3:
                gap_injected_segments.extend(qualifying_gap_run)
            qualifying_gap_run = []

        if len(qualifying_gap_run) >= 3:
            gap_injected_segments.extend(qualifying_gap_run)

    terminal_orphan_segments: list[tuple[float, float]] = []
    close_terminal_orphan_segments: list[tuple[float, float]] = []
    delayed_terminal_orphan_segments: list[tuple[float, float]] = []
    terminal_multi_onset_segments: list[tuple[float, float]] = []
    terminal_two_onset_tail_segments: list[tuple[float, float]] = []
    attack_validated_gap_segments: list[tuple[float, float]] = []
    if active_ranges:
        audio_duration = float(librosa.get_duration(y=audio, sr=sample_rate))
        last_range_end = active_ranges[-1][1]
        trailing_onsets = [
            onset_time
            for onset_time in gap_onset_times
            if last_range_end + TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE
            <= onset_time
            <= min(last_range_end + TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE, audio_duration - 0.08)
        ]
        if len(trailing_onsets) == 1:
            orphan_start = trailing_onsets[0]
            orphan_end = min(orphan_start + TERMINAL_ORPHAN_SEGMENT_DURATION, audio_duration)
            if orphan_end - orphan_start >= 0.08:
                terminal_orphan_segments.append((orphan_start, orphan_end))
        close_terminal_orphan_segments = [] if ABLATE_CLOSE_TERMINAL_ORPHAN else collect_close_terminal_orphan_segments(active_ranges, gap_onset_times, audio_duration, onset_attack_profiles)
        delayed_base_segment = close_terminal_orphan_segments[-1] if close_terminal_orphan_segments else (terminal_orphan_segments[-1] if terminal_orphan_segments else None)
        delayed_terminal_orphan_segments = [] if ABLATE_DELAYED_TERMINAL_ORPHAN else collect_delayed_terminal_orphan_segments(delayed_base_segment, gap_onset_times, audio_duration, onset_attack_profiles)
        terminal_multi_onset_segments = [] if ABLATE_TERMINAL_MULTI_ONSET else collect_terminal_multi_onset_segments(active_ranges, gap_onset_times, audio_duration, onset_attack_profiles)
        if not terminal_orphan_segments and not close_terminal_orphan_segments and not delayed_terminal_orphan_segments and not terminal_multi_onset_segments:
            terminal_two_onset_tail_segments = [] if ABLATE_TWO_ONSET_TERMINAL_TAIL else collect_two_onset_terminal_tail_segments(active_ranges, gap_onset_times, audio_duration, onset_attack_profiles)
        if USE_ATTACK_VALIDATED_GAP_COLLECTOR:
            attack_validated_gap_segments = collect_attack_validated_gap_segments(
                active_ranges,
                gap_onset_times,
                onset_attack_profiles,
                audio_duration,
                filtered_gap_candidates,
            )

    segments: list[tuple[float, float]] = []
    for range_index, (range_start, range_end) in enumerate(active_ranges):
        effective_range_start = range_start
        previous_range_end = active_ranges[range_index - 1][1] if range_index > 0 else None
        prior_onsets = [
            time
            for time in onset_times
            if range_start - PRIOR_ONSET_BACKTRACK_SECONDS <= time <= range_start + 0.005
            and (previous_range_end is None or time >= previous_range_end + 0.005)
        ]
        relaxed_head_segment = False
        if prior_onsets:
            effective_range_start = prior_onsets[-1]
            if (
                previous_range_end is not None
                and range_start - previous_range_end >= ACTIVE_RANGE_START_CLUSTER_MIN_GAP
                and range_end - range_start <= ACTIVE_RANGE_START_CLUSTER_MAX_DURATION
            ):
                trailing_cluster = [
                    time for time in prior_onsets
                    if effective_range_start - time <= ACTIVE_RANGE_START_CLUSTER_MAX_SPAN
                ]
                if len(trailing_cluster) >= 2:
                    effective_range_start = trailing_cluster[0]
                    relaxed_head_segment = True

        range_onsets = [time for time in onset_times if effective_range_start + 0.005 < time < range_end - 0.05]
        if not ABLATE_COLLAPSE_ACTIVE_RANGE_HEAD:
            range_onsets = collapse_active_range_head_onsets(effective_range_start, range_end, range_onsets, onset_attack_profiles)
        if not ABLATE_SNAP_RANGE_START_TO_ONSET and not prior_onsets and not relaxed_head_segment and range_onsets:
            first_range_onset = range_onsets[0]
            if should_snap_range_start_to_first_onset(effective_range_start, first_range_onset, onset_attack_profiles):
                effective_range_start = first_range_onset
                range_onsets = [time for time in range_onsets if effective_range_start + 0.005 < time < range_end - 0.05]
        boundary_times = sorted(range_onsets)
        deduped_onsets: list[float] = []
        for boundary_index, time in enumerate(boundary_times):
            if not deduped_onsets:
                deduped_onsets.append(time)
                continue

            previous_time = deduped_onsets[-1]
            if (
                time - previous_time >= 0.18
                or should_keep_dense_trailing_onset(boundary_times, boundary_index, effective_range_start, range_end)
                or should_keep_short_range_trailing_onset(boundary_times, boundary_index, effective_range_start, range_end)
            ):
                deduped_onsets.append(time)

        starts = [effective_range_start, *deduped_onsets]
        for index, start_time in enumerate(starts):
            end_time = starts[index + 1] if index + 1 < len(starts) else range_end
            min_duration = CLUSTERED_RANGE_HEAD_MIN_DURATION if relaxed_head_segment and index == 0 else 0.08
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))

    for start_time, end_time in gap_injected_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in leading_orphan_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in multi_onset_gap_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in post_tail_gap_head_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in single_onset_gap_head_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in sparse_gap_tail_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in terminal_orphan_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in close_terminal_orphan_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in delayed_terminal_orphan_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in terminal_multi_onset_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in terminal_two_onset_tail_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))
    for start_time, end_time in attack_validated_gap_segments:
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))

    segments = dedupe_nested_segments(segments)
    segments = trim_small_overlapping_segments(segments)

    if not segments:
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        segments = [(0.0, duration)]

    tempo_audio_duration_sec = float(librosa.get_duration(y=audio, sr=sample_rate))
    tempo_start = perf_counter()
    tempo_onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=TEMPO_ESTIMATION_HOP_LENGTH)
    tempo_array, _ = librosa.beat.beat_track(
        onset_envelope=tempo_onset_env,
        sr=sample_rate,
        hop_length=TEMPO_ESTIMATION_HOP_LENGTH,
    )
    tempo_estimation_ms = (perf_counter() - tempo_start) * 1000.0
    tempo = float(np.asarray(tempo_array).reshape(-1)[0]) if np.asarray(tempo_array).size else 90.0
    if tempo <= 1.0:
        tempo = 90.0

    debug_info = {
        "onsetTimes": onset_times,
        "gapValidatedOnsetTimes": gap_onset_times if FILTER_GAP_ONSETS_BY_ATTACK_PROFILE else None,
        "attackValidatedGapCandidates": {
            "interRanges": [[round(time, 4) for time in gap] for gap in attack_validated_gap_candidates.inter_ranges],
            "leading": [round(time, 4) for time in attack_validated_gap_candidates.leading],
            "trailing": [round(time, 4) for time in attack_validated_gap_candidates.trailing],
        },
        "gapIoiDiagnostics": gap_ioi_diagnostics,
        "activeRanges": [[round(start, 4), round(end, 4)] for start, end in active_ranges],
        "rawActiveRanges": [[round(start, 4), round(end, 4)] for start, end in raw_active_ranges],
        "shortBridgeActiveRanges": [[round(start, 4), round(end, 4)] for start, end in short_bridge_active_ranges],
        "gapInjectedSegments": [[round(start, 4), round(end, 4)] for start, end in gap_injected_segments],
        "leadingOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in leading_orphan_segments],
        "multiOnsetGapSegments": [[round(start, 4), round(end, 4)] for start, end in multi_onset_gap_segments],
        "postTailGapHeadSegments": [[round(start, 4), round(end, 4)] for start, end in post_tail_gap_head_segments],
        "singleOnsetGapHeadSegments": [[round(start, 4), round(end, 4)] for start, end in single_onset_gap_head_segments],
        "sparseGapTailSegments": [[round(start, 4), round(end, 4)] for start, end in sparse_gap_tail_segments],
        "terminalOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in terminal_orphan_segments],
        "closeTerminalOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in close_terminal_orphan_segments],
        "delayedTerminalOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in delayed_terminal_orphan_segments],
        "terminalMultiOnsetSegments": [[round(start, 4), round(end, 4)] for start, end in terminal_multi_onset_segments],
        "terminalTwoOnsetTailSegments": [[round(start, 4), round(end, 4)] for start, end in terminal_two_onset_tail_segments],
        "attackValidatedGapSegments": [[round(start, 4), round(end, 4)] for start, end in attack_validated_gap_segments],
        "segments": [[round(start, 4), round(end, 4)] for start, end in segments],
        "rmsThreshold": round(threshold, 6),
        "tempoRaw": round(tempo, 4),
        "tempoHopLength": TEMPO_ESTIMATION_HOP_LENGTH,
        "tempoAudioDurationSec": round(tempo_audio_duration_sec, 4),
        "tempoEstimationMs": round(tempo_estimation_ms, 3),
        "onsetAttackProfiles": {
            str(key): {
                "broadbandOnsetGain": round(profile.broadband_onset_gain, 6),
                "highBandSpectralFlux": round(profile.high_band_spectral_flux, 6),
                "broadbandSpectralFlux": round(profile.broadband_spectral_flux, 6),
                "isValidAttack": profile.is_valid_attack,
            }
            for key, profile in onset_attack_profiles.items()
        },
    }
    return segments, tempo, debug_info

def peak_energy_near(frequencies: np.ndarray, spectrum: np.ndarray, center_freq: float, band_cents: float = HARMONIC_BAND_CENTS) -> float:
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

def batch_peak_energies(frequencies: np.ndarray, spectrum: np.ndarray, center_freqs: np.ndarray, band_cents: float = HARMONIC_BAND_CENTS) -> np.ndarray:
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

def suppress_harmonics(spectrum: np.ndarray, frequencies: np.ndarray, base_frequency: float) -> np.ndarray:
    residual = spectrum.copy()
    valid = frequencies > 0
    positive_freqs = frequencies[valid]
    for multiple in range(1, MAX_HARMONIC_MULTIPLE + 1):
        center_freq = base_frequency * multiple
        if center_freq > frequencies[-1]:
            break
        distances = np.abs(1200.0 * np.log2(positive_freqs / center_freq))
        positive_mask = distances <= SUPPRESSION_BAND_CENTS
        if np.any(positive_mask):
            global_mask = np.zeros_like(frequencies, dtype=bool)
            global_mask[np.where(valid)[0][positive_mask]] = True
            residual[global_mask] *= 0.08
    return residual

def build_raw_peaks(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    tuning: InstrumentTuning,
    *,
    limit: int = 8,
    min_frequency: float = 40.0,
) -> list[dict[str, Any]]:
    valid = (frequencies >= min_frequency) & (spectrum > 0)
    if not np.any(valid):
        return []

    candidate_freqs = frequencies[valid]
    candidate_spectrum = spectrum[valid]
    ranked_indexes = np.argsort(candidate_spectrum)[::-1]

    peaks: list[dict[str, Any]] = []
    used_frequencies: list[float] = []
    for index in ranked_indexes:
        frequency = float(candidate_freqs[index])
        if any(cents_distance(frequency, existing) < 35.0 for existing in used_frequencies):
            continue

        amplitude = float(candidate_spectrum[index])
        snapped = snap_frequency_to_tuning(frequency, tuning)
        peaks.append(
            {
                "frequency": round(frequency, 3),
                "amplitude": round(amplitude, 6),
                "snappedNote": snapped.note_name if snapped else None,
                "centsToSnapped": round(cents_distance(frequency, snapped.frequency), 3) if snapped else None,
            }
        )
        used_frequencies.append(frequency)
        if len(peaks) >= limit:
            break

    return peaks

def rank_tuning_candidates(frequencies: np.ndarray, spectrum: np.ndarray, tuning: InstrumentTuning, *, debug: bool = False) -> list[NoteHypothesis]:
    note_freqs = np.array([note.frequency for note in tuning.notes])
    harmonic_targets = np.concatenate([note_freqs * m for m in range(1, MAX_HARMONIC_MULTIPLE + 1)])
    sub_half_targets = note_freqs / 2.0
    sub_third_targets = note_freqs / 3.0
    sub_half_targets[sub_half_targets < 40.0] = 0.0
    sub_third_targets[sub_third_targets < 40.0] = 0.0
    all_target_freqs = np.concatenate([harmonic_targets, sub_half_targets, sub_third_targets])
    all_energies = batch_peak_energies(frequencies, spectrum, all_target_freqs)

    n_notes = len(tuning.notes)
    harmonic_energy_matrix = all_energies[: n_notes * MAX_HARMONIC_MULTIPLE].reshape(MAX_HARMONIC_MULTIPLE, n_notes)
    sub_half_energies = all_energies[n_notes * MAX_HARMONIC_MULTIPLE : n_notes * MAX_HARMONIC_MULTIPLE + n_notes]
    sub_third_energies = all_energies[n_notes * MAX_HARMONIC_MULTIPLE + n_notes :]

    hypotheses: list[NoteHypothesis] = []

    for note_index, note in enumerate(tuning.notes):
        pitch_class, octave = parse_note_name(note.note_name)
        candidate = NoteCandidate(
            key=note.key,
            note_name=note.note_name,
            frequency=note.frequency,
            pitch_class=pitch_class,
            octave=octave,
        )

        harmonic_energies = [float(harmonic_energy_matrix[h, note_index]) for h in range(MAX_HARMONIC_MULTIPLE)]
        subharmonic_frequencies = [note.frequency / 2.0, note.frequency / 3.0]
        subharmonic_energies = [float(sub_half_energies[note_index]), float(sub_third_energies[note_index])]

        fundamental_energy = harmonic_energies[0]
        overtone_energy = sum(weight * energy for weight, energy in zip(HARMONIC_WEIGHTS[1:], harmonic_energies[1:]))
        harmonic_support = fundamental_energy + overtone_energy
        fundamental_ratio = fundamental_energy / max(harmonic_support, 1e-9)
        subharmonic_alias_energy = (0.7 * subharmonic_energies[0]) + (0.45 * subharmonic_energies[1])
        octave_alias_energy = subharmonic_energies[0]
        octave_alias_ratio = octave_alias_energy / max(fundamental_energy, 1e-9)
        octave_alias_penalty = 0.0
        if octave_alias_ratio >= OCTAVE_ALIAS_RATIO_THRESHOLD and fundamental_ratio <= OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO:
            octave_alias_penalty = octave_alias_energy * OCTAVE_ALIAS_PENALTY

        score = (
            harmonic_support * (0.2 + 0.8 * fundamental_ratio)
            + (0.45 * fundamental_energy)
            - (0.6 * subharmonic_alias_energy)
            - octave_alias_penalty
        )
        if fundamental_ratio < OVERTONE_DOMINANT_FUNDAMENTAL_RATIO:
            score -= OVERTONE_DOMINANT_PENALTY_WEIGHT * overtone_energy

        harmonics = None
        subharmonics = None
        if debug:
            harmonics = [
                {
                    "multiple": float(index),
                    "frequency": round(note.frequency * index, 3),
                    "energy": round(energy, 6),
                    "weight": HARMONIC_WEIGHTS[index - 1],
                }
                for index, energy in enumerate(harmonic_energies, start=1)
            ]
            subharmonics = [
                {
                    "multiple": 1.0 / float(index + 1),
                    "frequency": round(subharmonic_frequencies[index], 3),
                    "energy": round(subharmonic_energies[index], 6),
                }
                for index in range(len(subharmonic_frequencies))
            ]

        hypotheses.append(
            NoteHypothesis(
                candidate=candidate,
                score=score,
                fundamental_energy=fundamental_energy,
                overtone_energy=overtone_energy,
                fundamental_ratio=fundamental_ratio,
                subharmonic_alias_energy=subharmonic_alias_energy,
                octave_alias_energy=octave_alias_energy,
                octave_alias_ratio=octave_alias_ratio,
                octave_alias_penalty=octave_alias_penalty,
                second_harmonic_energy=harmonic_energies[1] if len(harmonic_energies) > 1 else 0.0,
                harmonics=harmonics,
                subharmonics=subharmonics,
            )
        )

    return sorted(hypotheses, key=lambda item: item.score, reverse=True)

def are_harmonic_related(note_a: NoteCandidate, note_b: NoteCandidate) -> bool:
    high = max(note_a.frequency, note_b.frequency)
    low = min(note_a.frequency, note_b.frequency)
    ratio = high / low if low else 0.0
    if ratio <= 0.0:
        return False
    return any(abs(1200.0 * math.log2(ratio / multiple)) <= 30 for multiple in (2, 3, 4))

def harmonic_relation_multiple(note_a: NoteCandidate, note_b: NoteCandidate) -> float | None:
    high = max(note_a.frequency, note_b.frequency)
    low = min(note_a.frequency, note_b.frequency)
    ratio = high / low if low else 0.0
    if ratio <= 0.0:
        return None
    for multiple in (2.0, 3.0, 4.0):
        if abs(1200.0 * math.log2(ratio / multiple)) <= 30:
            return multiple
    return None

def allow_octave_secondary(primary: NoteHypothesis, hypothesis: NoteHypothesis, selected: list[NoteCandidate]) -> bool:
    for existing in selected:
        relation = harmonic_relation_multiple(hypothesis.candidate, existing)
        if relation is None:
            continue
        if relation != 2.0:
            return False
        if hypothesis.candidate.frequency > existing.frequency and existing.octave > 4:
            return False
        if hypothesis.candidate.frequency > existing.frequency:
            primary_octave_energy = primary.second_harmonic_energy
            if hypothesis.fundamental_ratio < OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO:
                return False
            if primary_octave_energy > 0.0 and hypothesis.fundamental_energy < primary_octave_energy * OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO:
                return False
            return True
        if hypothesis.candidate.octave <= 3:
            return False
        if hypothesis.candidate.octave == 4:
            if hypothesis.fundamental_ratio < OCTAVE_DYAD_LOWER_OCTAVE4_MIN_FUNDAMENTAL_RATIO:
                return False
        elif hypothesis.fundamental_ratio < OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO:
            return False
        if hypothesis.fundamental_energy < primary.fundamental_energy * OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO:
            return False
        return True
    return False


def extend_contiguous_gliss_cluster(
    selected: list[NoteCandidate],
    ranked: list[NoteHypothesis],
    residual_ranked: list[NoteHypothesis],
    *,
    primary_score: float,
    duration: float,
    target_note_count: int,
) -> tuple[list[NoteCandidate], list[dict[str, Any]]]:
    if len(selected) != 2 or duration > FOUR_NOTE_GLISS_EXTENSION_MAX_DURATION:
        return selected, []

    selected_keys = sorted(note.key for note in selected)
    if selected_keys[-1] - selected_keys[0] < 2:
        return selected, []

    min_selected = selected_keys[0]
    max_selected = selected_keys[-1]
    start_min = max_selected - target_note_count + 1
    start_max = min_selected
    candidate_windows = [
        list(range(start_key, start_key + target_note_count))
        for start_key in range(start_min, start_max + 1)
        if min_selected >= start_key and max_selected <= start_key + target_note_count - 1
    ]
    if not candidate_windows:
        return selected, []

    selected_names = {note.note_name for note in selected}
    best_by_key: dict[int, NoteHypothesis] = {}
    for hypotheses in (residual_ranked[:10], ranked[:10]):
        for hypothesis in hypotheses:
            if hypothesis.candidate.note_name in selected_names:
                continue
            existing = best_by_key.get(hypothesis.candidate.key)
            if existing is None or hypothesis.score > existing.score:
                best_by_key[hypothesis.candidate.key] = hypothesis

    best_missing: list[NoteHypothesis] | None = None
    best_window_score = -1.0
    for window_keys in candidate_windows:
        missing_keys = [key for key in window_keys if key not in selected_keys]
        missing_hypotheses: list[NoteHypothesis] = []
        valid_window = True
        for key in missing_keys:
            hypothesis = best_by_key.get(key)
            if hypothesis is None:
                valid_window = False
                break
            if hypothesis.score < primary_score * FOUR_NOTE_GLISS_EXTENSION_SCORE_RATIO:
                valid_window = False
                break
            if hypothesis.score < FOUR_NOTE_GLISS_EXTENSION_MIN_SCORE:
                valid_window = False
                break
            if hypothesis.fundamental_ratio < FOUR_NOTE_GLISS_EXTENSION_MIN_FUNDAMENTAL_RATIO:
                valid_window = False
                break
            missing_hypotheses.append(hypothesis)
        if not valid_window:
            continue
        total_score = sum(hypothesis.score for hypothesis in missing_hypotheses)
        if total_score > best_window_score:
            best_missing = missing_hypotheses
            best_window_score = total_score

    if best_missing is None:
        return selected, []

    extended = list(selected)
    debug_entries: list[dict[str, Any]] = []
    for hypothesis in best_missing:
        extended.append(hypothesis.candidate)
        debug_entries.append(
            {
                'noteName': hypothesis.candidate.note_name,
                'score': round(hypothesis.score, 6),
                'fundamentalRatio': round(hypothesis.fundamental_ratio, 6),
                'onsetGain': None,
                'accepted': True,
                'reasons': [f'contiguous-{target_note_count}-note-gliss-extension'],
                'octaveDyadAllowed': False,
            }
        )

    return sorted(extended, key=lambda item: item.frequency), debug_entries

def onset_energy_gain(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    target_frequency: float,
) -> float:
    window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
    start_sample = max(int(start_time * sample_rate), 0)
    end_sample = min(int(end_time * sample_rate), len(audio))
    if end_sample - start_sample < 512:
        return 0.0

    early_chunk = audio[start_sample:min(start_sample + window_samples, end_sample)]
    pre_start = max(0, start_sample - window_samples)
    pre_chunk = audio[pre_start:start_sample]
    if len(pre_chunk) < 512 or len(early_chunk) < 512:
        return 0.0

    def _energy(chunk: np.ndarray) -> float:
        n_fft = max(4096, 1 << int(np.ceil(np.log2(len(chunk)))))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
        frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        return peak_energy_near(frequencies, spectrum, target_frequency)

    pre_energy = _energy(pre_chunk)
    early_energy = _energy(early_chunk)
    return (early_energy + 1e-6) / (pre_energy + 1e-6)


def _build_analysis_window_chunks(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
    start_sample = max(int(start_time * sample_rate), 0)
    end_sample = min(int(end_time * sample_rate), len(audio))
    if end_sample - start_sample < 512:
        return None

    attack_end = min(start_sample + window_samples, end_sample)
    pre_start = max(0, start_sample - window_samples)
    pre_chunk = audio[pre_start:start_sample]
    attack_chunk = audio[start_sample:attack_end]
    sustain_start = max(start_sample, end_sample - window_samples)
    sustain_chunk = audio[sustain_start:end_sample]
    if len(pre_chunk) < 512 or len(attack_chunk) < 512 or len(sustain_chunk) < 512:
        return None
    return pre_chunk, attack_chunk, sustain_chunk


def _chunk_spectrum(chunk: np.ndarray, sample_rate: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return frequencies, spectrum


def _broadband_chunk_energy(chunk: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(chunk, dtype=np.float64))))


def _positive_spectral_flux(
    reference_spectrum: np.ndarray,
    target_spectrum: np.ndarray,
    frequencies: np.ndarray,
    *,
    min_frequency: float = 0.0,
) -> float:
    if min_frequency > 0.0:
        mask = frequencies >= min_frequency
        if not np.any(mask):
            return 0.0
        reference = reference_spectrum[mask]
        target = target_spectrum[mask]
    else:
        reference = reference_spectrum
        target = target_spectrum
    positive_delta = np.maximum(target - reference, 0.0)
    return float(np.sum(positive_delta) / (np.sum(reference) + 1e-6))


@dataclass
class OnsetAttackProfile:
    onset_time: float
    broadband_onset_gain: float
    high_band_spectral_flux: float
    broadband_spectral_flux: float
    is_valid_attack: bool


_KURTOSIS_WINDOW_SECONDS = 0.02


def _waveform_kurtosis(signal: np.ndarray) -> float:
    """Excess kurtosis (Fisher). Normal distribution = 0, periodic ≈ -1.5."""
    if len(signal) < 4:
        return 0.0
    var = float(np.var(signal))
    if var < 1e-20:
        return 0.0
    mean = float(np.mean(signal))
    return float(np.mean((signal - mean) ** 4) / (var * var)) - 3.0


def _waveform_crest_factor(signal: np.ndarray) -> float:
    """Peak / RMS. Sine wave ≈ 1.41, impulsive spike >> 1."""
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms < 1e-20:
        return 0.0
    return float(np.max(np.abs(signal))) / rms


def compute_onset_attack_profile(
    audio: np.ndarray,
    sample_rate: int,
    onset_time: float,
    *,
    window_seconds: float = ONSET_ENERGY_WINDOW_SECONDS,
) -> OnsetAttackProfile | None:
    window_samples = max(int(sample_rate * window_seconds), 512)
    onset_sample = max(int(onset_time * sample_rate), 0)
    pre_start = max(0, onset_sample - window_samples)
    pre_chunk = audio[pre_start:onset_sample]
    attack_end = min(onset_sample + window_samples, len(audio))
    attack_chunk = audio[onset_sample:attack_end]
    if len(pre_chunk) < 512 or len(attack_chunk) < 512:
        return None

    n_fft = max(4096, 1 << int(np.ceil(np.log2(max(len(pre_chunk), len(attack_chunk))))))
    frequencies, pre_spectrum = _chunk_spectrum(pre_chunk, sample_rate, n_fft)
    _, attack_spectrum = _chunk_spectrum(attack_chunk, sample_rate, n_fft)

    pre_energy = _broadband_chunk_energy(pre_chunk)
    attack_energy = _broadband_chunk_energy(attack_chunk)
    broadband_gain = (attack_energy + 1e-6) / (pre_energy + 1e-6)
    broadband_flux = _positive_spectral_flux(pre_spectrum, attack_spectrum, frequencies)
    high_band_flux = _positive_spectral_flux(
        pre_spectrum, attack_spectrum, frequencies, min_frequency=SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY,
    )
    is_valid = high_band_flux >= ONSET_ATTACK_MIN_HIGH_BAND_FLUX or (broadband_gain >= ONSET_ATTACK_MIN_BROADBAND_GAIN and high_band_flux >= ONSET_ATTACK_GAIN_REQUIRES_MIN_FLUX)

    return OnsetAttackProfile(
        onset_time=onset_time,
        broadband_onset_gain=broadband_gain,
        high_band_spectral_flux=high_band_flux,
        broadband_spectral_flux=broadband_flux,
        is_valid_attack=is_valid,
    )


def precompute_onset_attack_profiles(
    audio: np.ndarray,
    sample_rate: int,
    onset_times: list[float],
) -> dict[float, OnsetAttackProfile]:
    profiles: dict[float, OnsetAttackProfile] = {}
    for onset_time in onset_times:
        profile = compute_onset_attack_profile(audio, sample_rate, onset_time)
        if profile is not None:
            profiles[round(onset_time, 4)] = profile
    return profiles


def refine_onset_times_by_attack_profile(
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
) -> list[float]:
    if not onset_times:
        return onset_times

    refined: list[float] = []
    for onset_time in onset_times:
        if not refined:
            refined.append(onset_time)
            continue

        previous_time = refined[-1]
        if onset_time - previous_time >= ATTACK_REFINED_ONSET_MAX_INTERVAL:
            refined.append(onset_time)
            continue

        previous_profile = onset_profiles.get(round(previous_time, 4))
        current_profile = onset_profiles.get(round(onset_time, 4))
        if current_profile is None:
            continue

        should_replace_previous = previous_profile is None
        if not should_replace_previous and not previous_profile.is_valid_attack and current_profile.is_valid_attack:
            should_replace_previous = True

        if should_replace_previous:
            refined[-1] = onset_time
            continue

        refined.append(onset_time)

    return refined


GAP_ONSET_REJECT_MAX_BROADBAND_GAIN = 2.0
GAP_ONSET_REJECT_MAX_HIGH_BAND_FLUX = 0.5


def filter_gap_onsets_by_attack(
    onset_times: list[float],
    active_ranges: list[tuple[float, float]],
    onset_profiles: dict[float, OnsetAttackProfile],
) -> list[float]:
    """Return onset_times with obvious-noise gap onsets removed.

    Onsets inside active ranges are kept unconditionally (used for segment
    boundary splitting).  Gap-region onsets are rejected only when BOTH
    broadband gain AND high-band spectral flux are below the reject
    thresholds — i.e. clearly not a real note attack.  Borderline onsets
    are kept so that existing timing heuristics can provide the second
    layer of validation.
    """
    if not active_ranges:
        return onset_times

    def _in_active_range(time: float) -> bool:
        for range_start, range_end in active_ranges:
            if range_start - 0.05 <= time <= range_end + 0.05:
                return True
        return False

    filtered: list[float] = []
    for time in onset_times:
        if _in_active_range(time):
            filtered.append(time)
            continue
        profile = onset_profiles.get(round(time, 4))
        if profile is None:
            filtered.append(time)
            continue
        if (
            profile.broadband_onset_gain < GAP_ONSET_REJECT_MAX_BROADBAND_GAIN
            and profile.high_band_spectral_flux < GAP_ONSET_REJECT_MAX_HIGH_BAND_FLUX
        ):
            continue
        filtered.append(time)
    return filtered


def prepare_attack_debug_context(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> dict[str, Any] | None:
    chunks = _build_analysis_window_chunks(audio, sample_rate, start_time, end_time)
    if chunks is None:
        return None

    pre_chunk, attack_chunk, sustain_chunk = chunks
    n_fft = max(4096, 1 << int(np.ceil(np.log2(max(len(pre_chunk), len(attack_chunk), len(sustain_chunk))))))
    frequencies, pre_spectrum = _chunk_spectrum(pre_chunk, sample_rate, n_fft)
    _, attack_spectrum = _chunk_spectrum(attack_chunk, sample_rate, n_fft)
    _, sustain_spectrum = _chunk_spectrum(sustain_chunk, sample_rate, n_fft)

    pre_energy = _broadband_chunk_energy(pre_chunk)
    attack_energy = _broadband_chunk_energy(attack_chunk)
    sustain_energy = _broadband_chunk_energy(sustain_chunk)
    return {
        "frequencies": frequencies,
        "preSpectrum": pre_spectrum,
        "attackSpectrum": attack_spectrum,
        "sustainSpectrum": sustain_spectrum,
        "broadband": {
            "broadbandPreAttackEnergy": round(pre_energy, 6),
            "broadbandAttackEnergy": round(attack_energy, 6),
            "broadbandSustainEnergy": round(sustain_energy, 6),
            "broadbandOnsetGain": round((attack_energy + 1e-6) / (pre_energy + 1e-6), 6),
            "broadbandAttackToSustainRatio": round((attack_energy + 1e-6) / (sustain_energy + 1e-6), 6),
            "spectralFlux": round(_positive_spectral_flux(pre_spectrum, attack_spectrum, frequencies), 6),
            "highBandSpectralFlux": round(
                _positive_spectral_flux(
                    pre_spectrum,
                    attack_spectrum,
                    frequencies,
                    min_frequency=SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY,
                ),
                6,
            ),
        },
    }


def build_candidate_attack_debug(attack_context: dict[str, Any], target_frequency: float) -> dict[str, Any]:
    frequencies = attack_context["frequencies"]
    pre_spectrum = attack_context["preSpectrum"]
    attack_spectrum = attack_context["attackSpectrum"]
    sustain_spectrum = attack_context["sustainSpectrum"]
    pre_energy = peak_energy_near(frequencies, pre_spectrum, target_frequency)
    attack_energy = peak_energy_near(frequencies, attack_spectrum, target_frequency)
    sustain_energy = peak_energy_near(frequencies, sustain_spectrum, target_frequency)
    return {
        "preAttackEnergy": round(pre_energy, 6),
        "attackEnergy": round(attack_energy, 6),
        "sustainEnergy": round(sustain_energy, 6),
        "attackToSustainRatio": round((attack_energy + 1e-6) / (sustain_energy + 1e-6), 6),
        "candidateOnsetGain": round((attack_energy + 1e-6) / (pre_energy + 1e-6), 6),
    }


def build_debug_candidates(
    ranked: list[NoteHypothesis],
    limit: int = 5,
    attack_profiles: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for hypothesis in ranked[:limit]:
        item = {
            "noteName": hypothesis.candidate.note_name,
            "score": round(hypothesis.score, 6),
            "fundamentalEnergy": round(hypothesis.fundamental_energy, 6),
            "overtoneEnergy": round(hypothesis.overtone_energy, 6),
            "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
            "subharmonicAliasEnergy": round(hypothesis.subharmonic_alias_energy, 6),
            "octaveAliasEnergy": round(hypothesis.octave_alias_energy, 6),
            "octaveAliasRatio": round(hypothesis.octave_alias_ratio, 6),
            "octaveAliasPenalty": round(hypothesis.octave_alias_penalty, 6),
            "harmonics": hypothesis.harmonics or [],
            "subharmonics": hypothesis.subharmonics or [],
        }
        if attack_profiles is not None:
            profile = attack_profiles.get(hypothesis.candidate.note_name)
            if profile is not None:
                item.update(profile)
        payload.append(item)
    return payload

def maybe_replace_stale_recent_primary(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    primary: NoteHypothesis,
    ranked: list[NoteHypothesis],
    recent_note_names: set[str] | None,
    previous_primary_note_name: str | None = None,
    previous_primary_frequency: float | None = None,
    previous_primary_was_singleton: bool = False,
) -> tuple[NoteHypothesis, float | None, dict[str, Any] | None]:
    if not recent_note_names or primary.candidate.note_name not in recent_note_names:
        return primary, None, None

    duration = end_time - start_time
    if duration > RECENT_PRIMARY_REPLACEMENT_MAX_DURATION:
        return primary, None, None

    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
    if primary_onset_gain >= MIN_RECENT_NOTE_ONSET_GAIN:
        return primary, primary_onset_gain, None

    for hypothesis in ranked[1:6]:
        if hypothesis.candidate.note_name == primary.candidate.note_name:
            continue
        if hypothesis.candidate.frequency >= primary.candidate.frequency:
            continue
        if hypothesis.score < primary.score * RECENT_PRIMARY_REPLACEMENT_MIN_SCORE_RATIO:
            continue
        onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
        relaxed_recent_primary = (
            hypothesis.fundamental_ratio >= RECENT_PRIMARY_REPLACEMENT_RELAXED_FUNDAMENTAL_RATIO
            and onset_gain >= RECENT_PRIMARY_REPLACEMENT_STRONG_ONSET_GAIN
        )
        descending_repeated_primary = (
            previous_primary_was_singleton
            and previous_primary_note_name == primary.candidate.note_name
            and previous_primary_frequency is not None
            and duration <= DESCENDING_REPEATED_PRIMARY_MAX_DURATION
            and primary_onset_gain <= DESCENDING_REPEATED_PRIMARY_MAX_PRIMARY_ONSET_GAIN
            and hypothesis.candidate.frequency < primary.candidate.frequency
            and hypothesis.candidate.frequency < previous_primary_frequency
            and DESCENDING_REPEATED_PRIMARY_MIN_INTERVAL_CENTS
            <= abs(cents_distance(hypothesis.candidate.frequency, primary.candidate.frequency))
            <= DESCENDING_REPEATED_PRIMARY_MAX_INTERVAL_CENTS
            and onset_gain >= DESCENDING_REPEATED_PRIMARY_MIN_REPLACEMENT_ONSET_GAIN
            and onset_gain >= max(primary_onset_gain, 1e-6) * DESCENDING_REPEATED_PRIMARY_MIN_REPLACEMENT_ONSET_RATIO
            and hypothesis.score >= primary.score * DESCENDING_REPEATED_PRIMARY_MIN_SCORE_RATIO
            and hypothesis.fundamental_ratio >= DESCENDING_REPEATED_PRIMARY_MIN_FUNDAMENTAL_RATIO
        )
        if hypothesis.fundamental_ratio < RECENT_PRIMARY_REPLACEMENT_MIN_FUNDAMENTAL_RATIO and not relaxed_recent_primary and not descending_repeated_primary:
            continue

        if (
            (
                onset_gain >= RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_GAIN
                and onset_gain >= max(primary_onset_gain, 1e-6) * RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_RATIO
            )
            or descending_repeated_primary
        ):
            replacement_debug = {
                "replacedPrimaryNote": primary.candidate.note_name,
                "replacementNote": hypothesis.candidate.note_name,
                "replacedPrimaryOnsetGain": round(primary_onset_gain, 6),
                "replacementOnsetGain": round(onset_gain, 6),
            }
            if descending_repeated_primary:
                replacement_debug["reason"] = "descending-repeated-primary"
            return (
                hypothesis,
                onset_gain,
                replacement_debug,
            )

    return primary, primary_onset_gain, None

def maybe_promote_lower_secondary_to_recent_upper_octave(
    primary: NoteHypothesis,
    accepted_secondary: NoteHypothesis,
    residual_ranked: list[NoteHypothesis],
    segment_duration: float,
    recent_note_names: set[str] | None = None,
) -> tuple[NoteHypothesis, str | None]:
    if segment_duration > LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_DURATION:
        return accepted_secondary, None
    if accepted_secondary.candidate.frequency >= primary.candidate.frequency:
        return accepted_secondary, None
    if recent_note_names and accepted_secondary.candidate.note_name in recent_note_names:
        return accepted_secondary, None
    if accepted_secondary.score > primary.score * LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_LOWER_SCORE_RATIO:
        return accepted_secondary, None
    if accepted_secondary.fundamental_ratio > LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_LOWER_FUNDAMENTAL_RATIO:
        return accepted_secondary, None
    if (
        abs(cents_distance(primary.candidate.frequency, accepted_secondary.candidate.frequency))
        < LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_PRIMARY_INTERVAL_CENTS
    ):
        return accepted_secondary, None

    target_octave = accepted_secondary.candidate.octave + 1
    for hypothesis in residual_ranked[:8]:
        if hypothesis.candidate.note_name == accepted_secondary.candidate.note_name:
            continue
        if hypothesis.candidate.pitch_class != accepted_secondary.candidate.pitch_class:
            continue
        if hypothesis.candidate.octave != target_octave:
            continue
        if hypothesis.candidate.frequency >= primary.candidate.frequency:
            continue
        if (
            abs(cents_distance(primary.candidate.frequency, hypothesis.candidate.frequency))
            > LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_PRIMARY_INTERVAL_CENTS
        ):
            continue
        if hypothesis.fundamental_ratio < LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO:
            continue
        if (
            hypothesis.fundamental_ratio - accepted_secondary.fundamental_ratio
            < LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_FUNDAMENTAL_RATIO_DELTA
        ):
            continue
        if hypothesis.score < accepted_secondary.score * LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_UPPER_SCORE_RATIO:
            continue
        if hypothesis.score < primary.score * LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_PRIMARY_SCORE_RATIO:
            continue
        return hypothesis, accepted_secondary.candidate.note_name

    return accepted_secondary, None


def maybe_promote_stale_primary_to_upper_octave(
    primary: NoteHypothesis,
    ranked: list[NoteHypothesis],
    segment_duration: float,
    primary_onset_gain: float | None,
    recent_note_names: set[str] | None,
) -> tuple[NoteHypothesis, dict[str, Any] | None]:
    if segment_duration > STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_DURATION:
        return primary, None
    if not recent_note_names or primary.candidate.note_name not in recent_note_names:
        return primary, None
    if primary.candidate.octave > 4:
        return primary, None
    if primary.fundamental_ratio > STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_PRIMARY_FUNDAMENTAL_RATIO:
        return primary, None
    if primary_onset_gain is not None and primary_onset_gain > STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_PRIMARY_ONSET_GAIN:
        return primary, None

    target_octave = primary.candidate.octave + 1
    upper_candidate: NoteHypothesis | None = None
    for hypothesis in ranked[1:6]:
        if hypothesis.candidate.pitch_class != primary.candidate.pitch_class:
            continue
        if hypothesis.candidate.octave != target_octave:
            continue
        if hypothesis.score < primary.score * STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_SCORE_RATIO:
            continue
        if hypothesis.fundamental_ratio < STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO:
            continue
        if hypothesis.octave_alias_ratio < STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_ALIAS_RATIO:
            continue
        upper_candidate = hypothesis
        break
    if upper_candidate is None:
        return primary, None

    has_supporting_high = any(
        hypothesis.candidate.note_name != upper_candidate.candidate.note_name
        and hypothesis.candidate.frequency > upper_candidate.candidate.frequency
        and hypothesis.score >= primary.score * STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_SUPPORTING_SCORE_RATIO
        and hypothesis.fundamental_ratio >= STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO
        and not are_harmonic_related(hypothesis.candidate, upper_candidate.candidate)
        for hypothesis in ranked[1:6]
    )
    if not has_supporting_high:
        return primary, None

    upper_candidate = NoteHypothesis(
        candidate=upper_candidate.candidate,
        score=max(upper_candidate.score, STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_SCORE),
        fundamental_energy=upper_candidate.fundamental_energy,
        overtone_energy=upper_candidate.overtone_energy,
        fundamental_ratio=upper_candidate.fundamental_ratio,
        subharmonic_alias_energy=upper_candidate.subharmonic_alias_energy,
        octave_alias_energy=upper_candidate.octave_alias_energy,
        octave_alias_ratio=upper_candidate.octave_alias_ratio,
        octave_alias_penalty=upper_candidate.octave_alias_penalty,
        second_harmonic_energy=upper_candidate.second_harmonic_energy,
        harmonics=upper_candidate.harmonics,
        subharmonics=upper_candidate.subharmonics,
    )
    return upper_candidate, {
        'replacedPrimaryNote': primary.candidate.note_name,
        'replacementNote': upper_candidate.candidate.note_name,
        'reason': 'stale-lower-primary-promoted-to-upper-octave',
    }


def maybe_promote_recent_upper_octave_alias_primary(
    primary: NoteHypothesis,
    ranked: list[NoteHypothesis],
    segment_duration: float,
    recent_note_names: set[str] | None,
) -> tuple[NoteHypothesis, dict[str, Any] | None]:
    if segment_duration > RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MAX_DURATION:
        return primary, None
    if not recent_note_names:
        return primary, None
    if primary.candidate.octave >= 6:
        return primary, None
    if primary.fundamental_ratio > RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MAX_PRIMARY_FUNDAMENTAL_RATIO:
        return primary, None

    target_octave = primary.candidate.octave + 1
    upper_candidate: NoteHypothesis | None = None
    for hypothesis in ranked[1:6]:
        if hypothesis.candidate.pitch_class != primary.candidate.pitch_class:
            continue
        if hypothesis.candidate.octave != target_octave:
            continue
        if hypothesis.candidate.note_name not in recent_note_names:
            continue
        if hypothesis.score < primary.score * RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_UPPER_SCORE_RATIO:
            continue
        if hypothesis.fundamental_ratio < RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO:
            continue
        if hypothesis.octave_alias_ratio < RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_UPPER_ALIAS_RATIO:
            continue
        upper_candidate = hypothesis
        break
    if upper_candidate is None:
        return primary, None

    has_supporting_lower = any(
        hypothesis.candidate.note_name != upper_candidate.candidate.note_name
        and hypothesis.candidate.frequency < upper_candidate.candidate.frequency
        and not are_harmonic_related(hypothesis.candidate, upper_candidate.candidate)
        and hypothesis.score >= primary.score * RECENT_UPPER_OCTAVE_ALIAS_PROMOTION_MIN_SUPPORTING_LOWER_SCORE_RATIO
        for hypothesis in ranked[1:6]
    )
    if not has_supporting_lower:
        return primary, None

    promoted = NoteHypothesis(
        candidate=upper_candidate.candidate,
        score=upper_candidate.score,
        fundamental_energy=upper_candidate.fundamental_energy,
        overtone_energy=upper_candidate.overtone_energy,
        fundamental_ratio=upper_candidate.fundamental_ratio,
        subharmonic_alias_energy=upper_candidate.subharmonic_alias_energy,
        octave_alias_energy=upper_candidate.octave_alias_energy,
        octave_alias_ratio=upper_candidate.octave_alias_ratio,
        octave_alias_penalty=upper_candidate.octave_alias_penalty,
        second_harmonic_energy=upper_candidate.second_harmonic_energy,
        harmonics=upper_candidate.harmonics,
        subharmonics=upper_candidate.subharmonics,
    )
    return promoted, {
        'replacedPrimaryNote': primary.candidate.note_name,
        'replacementNote': promoted.candidate.note_name,
        'reason': 'recent-upper-octave-alias-primary',
    }


def is_physically_playable_chord(keys: list[int]) -> bool:
    """Check if a set of keys can be played simultaneously on a kalimba.

    One thumb can slide across consecutive keys (2-4 tines).
    The other thumb can strict-press 1-2 adjacent keys.
    Either thumb can reach any part of the instrument.
    Valid chords must be splittable into:
      - a slide group (consecutive keys, any length) + a strict group (≤2 adjacent keys)
    """
    if len(keys) <= 2:
        return True
    unique = sorted(set(keys))
    n = len(unique)
    if n > 4:
        return False

    def _consecutive(ks: list[int]) -> bool:
        return len(ks) <= 1 or all(ks[i + 1] - ks[i] == 1 for i in range(len(ks) - 1))

    def _strict_ok(ks: list[int]) -> bool:
        return len(ks) <= 1 or (len(ks) == 2 and ks[1] - ks[0] == 1)

    # Try all ways to split into slide group + strict group
    for mask in range(1 << n):
        slide = [unique[i] for i in range(n) if mask & (1 << i)]
        strict = [unique[i] for i in range(n) if not (mask & (1 << i))]
        if len(slide) < 1 or len(strict) > 2:
            continue
        if _consecutive(slide) and _strict_ok(strict):
            return True
    return False


def select_contiguous_four_note_cluster(
    primary: NoteHypothesis,
    ranked: list[NoteHypothesis],
    segment_duration: float,
) -> list[NoteCandidate] | None:
    if segment_duration < FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_DURATION:
        return None
    if len(ranked) < 4:
        return None

    top_four = ranked[:4]
    top_four_keys = sorted(hypothesis.candidate.key for hypothesis in top_four)
    if len(set(top_four_keys)) != 4 or top_four_keys[-1] - top_four_keys[0] != 3:
        return None
    if primary.candidate.key not in top_four_keys:
        return None
    if any(hypothesis.fundamental_ratio < FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_FUNDAMENTAL_RATIO for hypothesis in top_four):
        return None

    fourth = top_four[-1]
    if fourth.score < primary.score * FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_SCORE_RATIO:
        return None
    if len(ranked) >= 5 and fourth.score < ranked[4].score * FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_MARGIN_RATIO:
        return None

    return sorted((hypothesis.candidate for hypothesis in top_four), key=lambda candidate: candidate.frequency)

def is_slide_playable_contiguous_cluster(notes: list[NoteCandidate], tuning: InstrumentTuning) -> bool:
    if len(notes) != 3:
        return False

    sorted_keys = sorted(note.key for note in notes)
    if sorted_keys[-1] - sorted_keys[0] != 2 or len(set(sorted_keys)) != 3:
        return False

    rank_by_name = {
        note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))
    }
    ranks: list[int] = []
    for note in sorted(notes, key=lambda item: item.frequency):
        rank = rank_by_name.get(note.note_name)
        if rank is None:
            return False
        ranks.append(rank)

    lower_gap = ranks[1] - ranks[0]
    upper_gap = ranks[2] - ranks[1]
    return lower_gap == upper_gap and lower_gap >= 2


def is_adjacent_tuning_step(note_a: NoteCandidate, note_b: NoteCandidate, tuning: InstrumentTuning) -> bool:
    rank_by_name = {
        note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))
    }
    rank_a = rank_by_name.get(note_a.note_name)
    rank_b = rank_by_name.get(note_b.note_name)
    if rank_a is None or rank_b is None:
        return False
    return abs(rank_a - rank_b) == 1


def suppress_leading_descending_overlap(raw_events: list[RawEvent], tuning: InstrumentTuning) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    first_event = raw_events[0]
    second_event = raw_events[1]
    third_event = raw_events[2]
    first_duration = first_event.end_time - first_event.start_time
    if (
        len(first_event.notes) != 2
        or len(second_event.notes) != 1
        or len(third_event.notes) != 1
        or first_duration > 0.16
    ):
        return raw_events

    sorted_notes = sorted(first_event.notes, key=lambda note: note.frequency)
    lower_note = sorted_notes[0]
    upper_note = sorted_notes[1]
    middle_note = second_event.notes[0]
    tail_note = third_event.notes[0]
    if (
        tail_note.note_name != lower_note.note_name
        or middle_note.frequency >= upper_note.frequency
        or middle_note.frequency <= lower_note.frequency
        or not is_adjacent_tuning_step(upper_note, middle_note, tuning)
        or not is_adjacent_tuning_step(middle_note, lower_note, tuning)
        or first_event.primary_note_name != upper_note.note_name
    ):
        return raw_events

    updated_first = RawEvent(
        start_time=first_event.start_time,
        end_time=first_event.end_time,
        notes=[upper_note],
        is_gliss_like=first_event.is_gliss_like,
        primary_note_name=upper_note.note_name,
        primary_score=first_event.primary_score,
    )
    return [updated_first, *raw_events[1:]]


def simplify_descending_adjacent_dyad_residue(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        updated_event = event
        if 0 < index < len(raw_events) - 1 and len(event.notes) == 2 and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION:
            previous_event = raw_events[index - 1]
            next_event = raw_events[index + 1]
            if len(previous_event.notes) == 1 and len(next_event.notes) == 1:
                lower_note, upper_note = sorted(event.notes, key=lambda note: note.frequency)
                if (
                    previous_event.notes[0].note_name == upper_note.note_name
                    and next_event.notes[0].frequency < lower_note.frequency
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= cents_distance(lower_note.frequency, upper_note.frequency)
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                ):
                    updated_event = RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=[lower_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=lower_note.note_name,
                        primary_score=event.primary_score,
                    )
        cleaned.append(updated_event)

    return cleaned




def collapse_ascending_restart_lower_residue_singletons(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned = list(raw_events)
    for index in range(1, len(cleaned) - 1):
        previous_event = cleaned[index - 1]
        event = cleaned[index]
        next_event = cleaned[index + 1]
        duration = event.end_time - event.start_time
        if (
            len(previous_event.notes) != 1
            or len(event.notes) != 1
            or len(next_event.notes) != 1
            or previous_event.is_gliss_like
            or next_event.is_gliss_like
            or duration > SHORT_SECONDARY_STRIP_MAX_DURATION
        ):
            continue

        previous_note = previous_event.notes[0]
        residue_note = event.notes[0]
        next_note = next_event.notes[0]
        if (
            previous_note.frequency >= next_note.frequency
            or not is_adjacent_tuning_step(previous_note, next_note, tuning)
            or residue_note.frequency >= previous_note.frequency
            or is_adjacent_tuning_step(residue_note, previous_note, tuning)
            or is_adjacent_tuning_step(residue_note, next_note, tuning)
            or abs(cents_distance(residue_note.frequency, previous_note.frequency)) < 250.0
            or abs(cents_distance(residue_note.frequency, previous_note.frequency)) > 550.0
        ):
            continue

        cleaned[index] = RawEvent(
            start_time=event.start_time,
            end_time=event.end_time,
            notes=[previous_note],
            is_gliss_like=event.is_gliss_like,
            primary_note_name=previous_note.note_name,
            primary_score=event.primary_score,
        )

    return cleaned


def collapse_high_register_adjacent_bridge_dyads(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    cleaned = list(raw_events)
    for index in range(1, len(cleaned) - 2):
        previous_event = cleaned[index - 1]
        event = cleaned[index]
        next_event = cleaned[index + 1]
        next_next_event = cleaned[index + 2]
        duration = event.end_time - event.start_time
        if (
            len(previous_event.notes) != 1
            or len(event.notes) != 2
            or len(next_event.notes) != 1
            or len(next_next_event.notes) != 1
            or previous_event.is_gliss_like
            or next_event.is_gliss_like
            or next_next_event.is_gliss_like
            or duration > ADJACENT_SEPARATED_DYAD_MAX_DURATION
        ):
            continue

        lower_note, upper_note = sorted(event.notes, key=lambda note: note.frequency)
        if (
            lower_note.octave < 6
            or upper_note.octave < 6
            or previous_event.notes[0].note_name != upper_note.note_name
            or next_event.notes[0].note_name != upper_note.note_name
            or next_next_event.notes[0].note_name != lower_note.note_name
            or event.primary_note_name != lower_note.note_name
            or not is_adjacent_tuning_step(lower_note, upper_note, tuning)
        ):
            continue

        cleaned[index] = RawEvent(
            start_time=event.start_time,
            end_time=event.end_time,
            notes=[lower_note],
            is_gliss_like=event.is_gliss_like,
            primary_note_name=lower_note.note_name,
            primary_score=event.primary_score,
        )

    return cleaned
def suppress_descending_upper_singleton_spikes(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        if index == 0 or index + 1 >= len(raw_events):
            cleaned.append(raw_events[index])
            index += 1
            continue

        previous_event = raw_events[index - 1]
        event = raw_events[index]
        next_event = raw_events[index + 1]

        if (
            len(previous_event.notes) == 2
            and len(event.notes) == 1
            and len(next_event.notes) == 1
            and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION
        ):
            previous_sorted = sorted(previous_event.notes, key=lambda note: note.frequency)
            lower_note = previous_sorted[0]
            upper_note = previous_sorted[1]
            current_note = event.notes[0]
            next_note = next_event.notes[0]
            is_descending_upper_spike = (
                current_note.frequency > upper_note.frequency > lower_note.frequency > next_note.frequency
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(current_note.frequency, upper_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(upper_note.frequency, lower_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(lower_note.frequency, next_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
            )
            if is_descending_upper_spike:
                index += 1
                continue

        cleaned.append(event)
        index += 1

    return cleaned



def suppress_short_descending_return_singletons(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if index == 0 or index + 1 >= len(raw_events):
            cleaned.append(event)
            continue

        previous_event = raw_events[index - 1]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time
        gap_before = event.start_time - previous_event.end_time
        gap_after = next_event.start_time - event.end_time
        if (
            len(previous_event.notes) == 1
            and len(event.notes) == 1
            and len(next_event.notes) == 1
            and not previous_event.is_gliss_like
            and not next_event.is_gliss_like
            and duration <= 0.14
            and gap_before <= 0.02
            and gap_after >= 0.18
        ):
            previous_note = previous_event.notes[0]
            current_note = event.notes[0]
            next_note = next_event.notes[0]
            if (
                previous_note.octave >= 6
                and current_note.octave >= 6
                and next_note.octave >= 6
                and previous_note.frequency < current_note.frequency < next_note.frequency
                and is_adjacent_tuning_step(previous_note, current_note, tuning)
                and is_adjacent_tuning_step(current_note, next_note, tuning)
            ):
                continue

        cleaned.append(event)

    return cleaned
def suppress_descending_upper_return_overlap(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        if index < 2 or index + 1 >= len(raw_events):
            cleaned.append(raw_events[index])
            index += 1
            continue

        previous_previous_event = raw_events[index - 2]
        previous_event = raw_events[index - 1]
        event = raw_events[index]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time

        if (
            len(previous_previous_event.notes) == 1
            and len(previous_event.notes) == 1
            and len(event.notes) == 2
            and len(next_event.notes) == 1
            and not previous_previous_event.is_gliss_like
            and not previous_event.is_gliss_like
            and not event.is_gliss_like
            and not next_event.is_gliss_like
            and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
        ):
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                upper_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency > primary_note.frequency]
                if len(upper_notes) == 1:
                    upper_note = upper_notes[0]
                    if (
                        previous_previous_event.notes[0].note_name == upper_note.note_name
                        and previous_event.notes[0].note_name == primary_note.note_name
                        and next_event.notes[0].frequency < primary_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= cents_distance(primary_note.frequency, upper_note.frequency)
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    ):
                        index += 1
                        continue

        cleaned.append(event)
        index += 1

    return cleaned


def should_block_descending_repeated_primary_tertiary_extension(
    *,
    selected: list[NoteCandidate],
    extension: NoteCandidate,
    segment_duration: float,
    previous_primary_was_singleton: bool,
    descending_primary_suffix_floor: float | None,
    descending_primary_suffix_ceiling: float | None,
    descending_primary_suffix_note_names: set[str] | None,
) -> bool:
    if len(selected) != 2:
        return False

    selected_keys = sorted(note.key for note in selected)
    is_upper_contiguous_extension = extension.key == selected_keys[-1] + 1
    if not is_upper_contiguous_extension:
        return False

    return (
        previous_primary_was_singleton
        and descending_primary_suffix_floor is not None
        and descending_primary_suffix_ceiling is not None
        and bool(descending_primary_suffix_note_names)
        and segment_duration <= DESCENDING_PRIMARY_SUFFIX_MAX_DURATION
        and selected[0].frequency <= descending_primary_suffix_floor
        and extension.frequency > descending_primary_suffix_ceiling
    )


def segment_peaks(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    tuning: InstrumentTuning,
    *,
    debug: bool = False,
    recent_note_names: set[str] | None = None,
    ascending_primary_run_ceiling: float | None = None,
    ascending_singleton_suffix_ceiling: float | None = None,
    ascending_singleton_suffix_note_names: set[str] | None = None,
    descending_primary_suffix_floor: float | None = None,
    descending_primary_suffix_ceiling: float | None = None,
    descending_primary_suffix_note_names: set[str] | None = None,
    previous_primary_note_name: str | None = None,
    previous_primary_frequency: float | None = None,
    previous_primary_was_singleton: bool = False,
) -> tuple[list[NoteCandidate], dict[str, Any] | None, NoteHypothesis | None]:
    start = int(start_time * sample_rate)
    end = int(end_time * sample_rate)
    segment = audio[start:end]
    if len(segment) < 512:
        return [], None, None

    analysis_samples = len(segment)
    if len(segment) > int(sample_rate * 0.1):
        analysis_samples = min(
            len(segment),
            max(int(sample_rate * ATTACK_ANALYSIS_SECONDS), int(len(segment) * ATTACK_ANALYSIS_RATIO)),
        )
    analysis_segment = segment[:analysis_samples]

    n_fft = max(4096, 1 << int(np.ceil(np.log2(len(analysis_segment)))))
    window = np.hanning(len(analysis_segment))
    spectrum = np.abs(np.fft.rfft(analysis_segment * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

    ranked = rank_tuning_candidates(frequencies, spectrum, tuning, debug=debug)
    if not ranked or ranked[0].score <= 1e-6:
        return [], None, None

    primary = ranked[0]
    primary, primary_onset_gain, primary_promotion_debug = maybe_replace_stale_recent_primary(
        audio,
        sample_rate,
        start_time,
        end_time,
        primary,
        ranked,
        recent_note_names,
        previous_primary_note_name=previous_primary_note_name,
        previous_primary_frequency=previous_primary_frequency,
        previous_primary_was_singleton=previous_primary_was_singleton,
    )
    primary, stale_upper_promotion_debug = maybe_promote_stale_primary_to_upper_octave(
        primary,
        ranked,
        end_time - start_time,
        primary_onset_gain,
        recent_note_names,
    )
    if stale_upper_promotion_debug is not None:
        primary_promotion_debug = stale_upper_promotion_debug
    primary, recent_upper_alias_promotion_debug = maybe_promote_recent_upper_octave_alias_primary(
        primary,
        ranked,
        end_time - start_time,
        recent_note_names,
    )
    if recent_upper_alias_promotion_debug is not None:
        primary_promotion_debug = recent_upper_alias_promotion_debug
    selected = [primary.candidate]
    residual_ranked: list[NoteHypothesis] = []
    promoted_secondary_to_recent_upper_octave = False
    secondary_decision_trail: list[dict[str, Any]] = []
    contiguous_four_note_cluster = select_contiguous_four_note_cluster(primary, ranked, end_time - start_time)
    if contiguous_four_note_cluster is not None:
        selected = contiguous_four_note_cluster
        for candidate in selected:
            if candidate.note_name == primary.candidate.note_name:
                continue
            matching = next(hypothesis for hypothesis in ranked[:4] if hypothesis.candidate.note_name == candidate.note_name)
            secondary_decision_trail.append(
                {
                    "noteName": matching.candidate.note_name,
                    "score": round(matching.score, 6),
                    "fundamentalRatio": round(matching.fundamental_ratio, 6),
                    "onsetGain": None,
                    "accepted": True,
                    "reasons": ["contiguous-four-note-cluster"],
                    "octaveDyadAllowed": False,
                }
            )
    secondary_score_ratio = SECONDARY_SCORE_RATIO
    if end_time - start_time <= 0.14:
        secondary_score_ratio = SHORT_SEGMENT_SECONDARY_SCORE_RATIO
    secondary_min_fundamental_ratio = SECONDARY_MIN_FUNDAMENTAL_RATIO

    if MAX_POLYPHONY > 1 and contiguous_four_note_cluster is None:
        residual_spectrum = suppress_harmonics(spectrum, frequencies, primary.candidate.frequency)
        residual_ranked = rank_tuning_candidates(frequencies, residual_spectrum, tuning, debug=debug)
        secondary_onset_gain: float | None = None
        secondary_fundamental_ratio: float | None = None
        for hypothesis in residual_ranked[:8]:
            reasons: list[str] = []
            onset_gain: float | None = None
            octave_dyad_allowed = allow_octave_secondary(primary, hypothesis, selected)
            segment_duration = end_time - start_time
            score_ratio = secondary_score_ratio
            if octave_dyad_allowed and hypothesis.candidate.frequency > primary.candidate.frequency:
                score_ratio = min(score_ratio, OCTAVE_DYAD_UPPER_SCORE_RATIO)
            is_tertiary_or_beyond = len(selected) >= 2
            is_quaternary_or_beyond = len(selected) >= 3
            if is_tertiary_or_beyond:
                test_keys = [n.key for n in selected] + [hypothesis.candidate.key]
                tertiary_score_ratio = QUATERNARY_MIN_SCORE_RATIO if is_quaternary_or_beyond else TERTIARY_MIN_SCORE_RATIO
                if not is_physically_playable_chord(test_keys):
                    reasons.append("tertiary-physically-impossible")
                elif hypothesis.score < primary.score * tertiary_score_ratio:
                    reasons.append("tertiary-score-below-threshold")
                elif hypothesis.fundamental_ratio < TERTIARY_MIN_FUNDAMENTAL_RATIO:
                    reasons.append("tertiary-fundamental-ratio-too-low")
                elif any(hypothesis.candidate.note_name == existing.note_name for existing in selected):
                    reasons.append("tertiary-duplicate-note")
                if not reasons and is_tertiary_or_beyond:
                    secondary = selected[1]
                    if secondary_onset_gain is None:
                        secondary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, secondary.frequency)
                    if secondary_onset_gain < 1.0:
                        reasons.append("tertiary-secondary-no-attack")
                    elif secondary_fundamental_ratio is not None and secondary_fundamental_ratio < TERTIARY_MIN_FUNDAMENTAL_RATIO:
                        reasons.append("tertiary-secondary-weak-fundamental")
                if not reasons and is_tertiary_or_beyond:
                    if onset_gain is None:
                        onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                    if onset_gain < TERTIARY_MIN_ONSET_GAIN:
                        reasons.append("tertiary-weak-onset")
            if hypothesis.candidate.note_name == primary.candidate.note_name:
                reasons.append("same-as-primary")
            if (
                primary_promotion_debug is not None
                and primary_promotion_debug.get("reason") == "recent-upper-octave-alias-primary"
                and hypothesis.candidate.pitch_class == primary.candidate.pitch_class
                and hypothesis.candidate.octave == primary.candidate.octave - 1
            ):
                reasons.append("recent-upper-octave-alias-secondary-blocked")
            if hypothesis.score < primary.score * score_ratio and not octave_dyad_allowed:
                reasons.append("score-below-threshold")
            if hypothesis.fundamental_ratio < secondary_min_fundamental_ratio:
                reasons.append("fundamental-ratio-too-low")
            if any(are_harmonic_related(hypothesis.candidate, existing) for existing in selected) and not octave_dyad_allowed:
                reasons.append("harmonic-related-to-selected")
            if recent_note_names and hypothesis.candidate.note_name in recent_note_names:
                onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if hypothesis.candidate.frequency < primary.candidate.frequency:
                    if (
                        onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                        or (
                            ascending_primary_run_ceiling is not None
                            and segment_duration <= ASCENDING_PRIMARY_RUN_MAX_DURATION
                            and primary.candidate.frequency >= ascending_primary_run_ceiling
                            and hypothesis.candidate.frequency < ascending_primary_run_ceiling
                            and onset_gain < ASCENDING_PRIMARY_RUN_RECENT_SECONDARY_ONSET_GAIN
                        )
                    ):
                        reasons.append("recent-carryover-candidate")
                else:
                    if primary_onset_gain is None:
                        primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                    if (
                        primary_onset_gain >= RECENT_UPPER_SECONDARY_PRIMARY_ONSET_GAIN
                        and segment_duration >= RECENT_UPPER_SECONDARY_MIN_DURATION
                        and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    ):
                        reasons.append("recent-carryover-candidate")
            if (
                not reasons
                and previous_primary_was_singleton
                and previous_primary_note_name == hypothesis.candidate.note_name
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and segment_duration <= DESCENDING_ADJACENT_UPPER_CARRYOVER_MAX_DURATION
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= cents_distance(primary.candidate.frequency, hypothesis.candidate.frequency)
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if onset_gain is None:
                    onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if (
                    primary_onset_gain >= DESCENDING_ADJACENT_UPPER_PRIMARY_ONSET_GAIN
                    and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_ADJACENT_UPPER_SCORE_RATIO
                ):
                    reasons.append("descending-adjacent-upper-carryover")
            if (
                not reasons
                and previous_primary_was_singleton
                and previous_primary_frequency is not None
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and primary.candidate.frequency < previous_primary_frequency
                and hypothesis.candidate.frequency >= previous_primary_frequency
                and segment_duration <= DESCENDING_RESTART_UPPER_CARRYOVER_MAX_DURATION
                and not octave_dyad_allowed
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if onset_gain is None:
                    onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if (
                    primary_onset_gain >= DESCENDING_RESTART_UPPER_PRIMARY_ONSET_GAIN
                    and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_RESTART_UPPER_SCORE_RATIO
                ):
                    reasons.append("descending-restart-upper-carryover")
            if (
                not reasons
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and descending_primary_suffix_floor is not None
                and descending_primary_suffix_ceiling is not None
                and descending_primary_suffix_note_names
                and primary.candidate.frequency <= descending_primary_suffix_floor
                and segment_duration <= DESCENDING_PRIMARY_SUFFIX_MAX_DURATION
                and not octave_dyad_allowed
                and (
                    hypothesis.candidate.note_name in descending_primary_suffix_note_names
                    or hypothesis.candidate.frequency > descending_primary_suffix_ceiling
                )
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if onset_gain is None:
                    onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if (
                    primary_onset_gain >= DESCENDING_PRIMARY_SUFFIX_PRIMARY_ONSET_GAIN
                    and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_PRIMARY_SUFFIX_UPPER_SCORE_RATIO
                ):
                    reasons.append("descending-primary-suffix-upper-carryover")
            if (
                not reasons
                and primary_promotion_debug is not None
                and primary_promotion_debug.get("reason") == "descending-repeated-primary"
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and not octave_dyad_allowed
            ):
                replaced_primary_note = primary_promotion_debug.get("replacedPrimaryNote")
                if onset_gain is None:
                    onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if (
                    onset_gain < DESCENDING_REPEATED_PRIMARY_STALE_UPPER_MAX_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_REPEATED_PRIMARY_STALE_UPPER_SCORE_RATIO
                    and (
                        hypothesis.candidate.note_name == replaced_primary_note
                        or (
                            descending_primary_suffix_ceiling is not None
                            and hypothesis.candidate.frequency > descending_primary_suffix_ceiling
                        )
                    )
                ):
                    reasons.append("descending-repeated-primary-stale-upper")
            if (
                not reasons
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and segment_duration >= UPPER_SECONDARY_WEAK_ONSET_MIN_DURATION
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if primary_onset_gain >= RECENT_UPPER_SECONDARY_PRIMARY_ONSET_GAIN:
                    if onset_gain is None:
                        onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                    if (
                        onset_gain < UPPER_SECONDARY_WEAK_ONSET_MAX_GAIN
                        and hypothesis.score < primary.score * UPPER_SECONDARY_WEAK_ONSET_SCORE_RATIO
                    ):
                        reasons.append("weak-upper-secondary")
            if (
                not reasons
                and segment_duration <= SHORT_SECONDARY_WEAK_ONSET_MAX_DURATION
                and not octave_dyad_allowed
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if primary_onset_gain >= SHORT_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN:
                    if onset_gain is None:
                        onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                    if (
                        onset_gain < SHORT_SECONDARY_WEAK_ONSET_MAX_GAIN
                        and onset_gain < primary_onset_gain * SHORT_SECONDARY_WEAK_ONSET_MAX_RATIO
                        and hypothesis.score < primary.score * SHORT_SECONDARY_WEAK_ONSET_SCORE_RATIO
                    ):
                        reasons.append("weak-secondary-onset")
            if (
                not reasons
                and segment_duration <= LOWER_SECONDARY_WEAK_ONSET_MAX_DURATION
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and not octave_dyad_allowed
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if primary_onset_gain >= LOWER_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN:
                    if onset_gain is None:
                        onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                    if (
                        onset_gain < LOWER_SECONDARY_WEAK_ONSET_MAX_GAIN
                        and onset_gain < primary_onset_gain * LOWER_SECONDARY_WEAK_ONSET_MAX_RATIO
                        and hypothesis.score < primary.score * LOWER_SECONDARY_WEAK_ONSET_SCORE_RATIO
                    ):
                        reasons.append("weak-lower-secondary")
            if (
                reasons == ["score-below-threshold"]
                and segment_duration >= 0.75
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and hypothesis.fundamental_ratio >= 0.95
                and hypothesis.score >= primary.score * 0.1
                and not octave_dyad_allowed
            ):
                interval_cents = abs(cents_distance(primary.candidate.frequency, hypothesis.candidate.frequency))
                if 250.0 <= interval_cents <= 550.0:
                    reasons = []

            if (
                not reasons
                and ascending_primary_run_ceiling is not None
                and segment_duration <= ASCENDING_PRIMARY_RUN_MAX_DURATION
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and primary.candidate.frequency >= ascending_primary_run_ceiling
                and hypothesis.candidate.frequency < ascending_primary_run_ceiling
                and hypothesis.score < primary.score * ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO
                and hypothesis.candidate.note_name != primary.candidate.note_name
                and primary.candidate.note_name not in (recent_note_names or set())
                and not octave_dyad_allowed
            ):
                if onset_gain is None:
                    onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if onset_gain < MIN_RECENT_NOTE_ONSET_GAIN:
                    reasons.append("ascending-singleton-carryover")
            if (
                not reasons
                and segment_duration <= 0.22
                and previous_primary_was_singleton
                and previous_primary_note_name == primary.candidate.note_name
                and ascending_singleton_suffix_ceiling is not None
                and primary.candidate.frequency >= ascending_singleton_suffix_ceiling
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and hypothesis.score < primary.score * ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO
                and hypothesis.candidate.note_name != primary.candidate.note_name
                and hypothesis.candidate.note_name not in (ascending_singleton_suffix_note_names or set())
                and not octave_dyad_allowed
            ):
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if onset_gain is None:
                    onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency)
                if primary_onset_gain < MIN_RECENT_NOTE_ONSET_GAIN and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN:
                    reasons.append("repeated-primary-carryover")
            accepted = len(reasons) == 0
            accepted_hypothesis = hypothesis
            debug_reasons = reasons
            if accepted and len(selected) == 1 and hypothesis.candidate.frequency < primary.candidate.frequency and not octave_dyad_allowed:
                accepted_hypothesis, promoted_from = maybe_promote_lower_secondary_to_recent_upper_octave(
                    primary,
                    hypothesis,
                    residual_ranked,
                    segment_duration,
                    recent_note_names,
                )
                if promoted_from is not None:
                    promoted_secondary_to_recent_upper_octave = True
                    debug_reasons = [f"promoted-from-{promoted_from}"]
            secondary_decision_trail.append(
                {
                    "noteName": accepted_hypothesis.candidate.note_name if accepted else hypothesis.candidate.note_name,
                    "score": round(accepted_hypothesis.score if accepted else hypothesis.score, 6),
                    "fundamentalRatio": round(accepted_hypothesis.fundamental_ratio if accepted else hypothesis.fundamental_ratio, 6),
                    "onsetGain": None if onset_gain is None else round(onset_gain, 6),
                    "accepted": accepted,
                    "reasons": debug_reasons,
                    "octaveDyadAllowed": octave_dyad_allowed,
                }
            )
            if accepted:
                if len(selected) == 1:
                    secondary_fundamental_ratio = accepted_hypothesis.fundamental_ratio
                selected.append(accepted_hypothesis.candidate)
                if len(selected) >= MAX_POLYPHONY:
                    break

    if (
        len(selected) == 2
        and not promoted_secondary_to_recent_upper_octave
        and end_time - start_time <= GLISS_TERTIARY_MAX_DURATION
    ):
        selected_keys = sorted(note.key for note in selected)
        if selected_keys[-1] - selected_keys[0] == 1:
            extension_keys = {selected_keys[0] - 1, selected_keys[-1] + 1}
            selected_names = {note.note_name for note in selected}
            viable_extensions: list[tuple[NoteHypothesis, float]] = []
            for hypothesis in residual_ranked[:6]:
                candidate = hypothesis.candidate
                if candidate.note_name in selected_names:
                    continue
                if candidate.key not in extension_keys:
                    continue
                if hypothesis.score < primary.score * GLISS_TERTIARY_SCORE_RATIO:
                    continue
                if hypothesis.score < GLISS_TERTIARY_MIN_SCORE:
                    continue
                if hypothesis.fundamental_ratio < GLISS_TERTIARY_MIN_FUNDAMENTAL_RATIO:
                    continue
                if any(are_harmonic_related(candidate, existing) for existing in selected):
                    continue
                onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, candidate.frequency)
                if primary_onset_gain is not None and primary_onset_gain > 0 and primary.score > 0:
                    relative_og = onset_gain / primary_onset_gain
                    score_ratio = hypothesis.score / primary.score
                    if relative_og > GLISS_TERTIARY_MAX_RELATIVE_ONSET_GAIN and score_ratio < GLISS_TERTIARY_SCORE_RATIO:
                        continue
                viable_extensions.append((hypothesis, onset_gain))

            chosen_extension: tuple[NoteHypothesis, float] | None = None
            if viable_extensions:
                strongest_by_score = max(viable_extensions, key=lambda item: item[0].score)
                strong_onset_candidates = [
                    item
                    for item in viable_extensions
                    if item[1] >= GLISS_TERTIARY_STRONG_ONSET_GAIN and item[0].score >= GLISS_TERTIARY_MIN_SCORE
                ]
                if strong_onset_candidates and strongest_by_score[1] < GLISS_TERTIARY_WEAK_ONSET_GAIN:
                    chosen_extension = max(strong_onset_candidates, key=lambda item: item[1])
                else:
                    chosen_extension = strongest_by_score

            if chosen_extension is not None:
                hypothesis, onset_gain = chosen_extension
                extended_cluster = [*selected, hypothesis.candidate]
                extension_blocked = should_block_descending_repeated_primary_tertiary_extension(
                    selected=selected,
                    extension=hypothesis.candidate,
                    segment_duration=end_time - start_time,
                    previous_primary_was_singleton=previous_primary_was_singleton,
                    descending_primary_suffix_floor=descending_primary_suffix_floor,
                    descending_primary_suffix_ceiling=descending_primary_suffix_ceiling,
                    descending_primary_suffix_note_names=descending_primary_suffix_note_names,
                )
                if extension_blocked:
                    secondary_decision_trail.append(
                        {
                            "noteName": hypothesis.candidate.note_name,
                            "score": round(hypothesis.score, 6),
                            "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                            "onsetGain": round(onset_gain, 6),
                            "accepted": False,
                            "reasons": ["descending-repeated-primary-tertiary-blocked"],
                            "octaveDyadAllowed": False,
                        }
                    )
                elif is_slide_playable_contiguous_cluster(extended_cluster, tuning):
                    selected.append(hypothesis.candidate)
                    secondary_decision_trail.append(
                        {
                            "noteName": hypothesis.candidate.note_name,
                            "score": round(hypothesis.score, 6),
                            "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                            "onsetGain": round(onset_gain, 6),
                            "accepted": True,
                            "reasons": ["contiguous-tertiary-extension"],
                            "octaveDyadAllowed": False,
                        }
                    )
                else:
                    secondary_decision_trail.append(
                        {
                            "noteName": hypothesis.candidate.note_name,
                            "score": round(hypothesis.score, 6),
                            "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                            "onsetGain": round(onset_gain, 6),
                            "accepted": False,
                            "reasons": ["non-slide-playable-contiguous-cluster"],
                            "octaveDyadAllowed": False,
                        }
                    )

    if (
        len(selected) == 2
        and end_time - start_time <= LOWER_MIXED_ROLL_EXTENSION_MAX_DURATION
        and primary.candidate.frequency == min(note.frequency for note in selected)
        and primary.candidate.octave <= 4
    ):
        upper_note = max(selected, key=lambda note: note.frequency)
        selected_names = {note.note_name for note in selected}
        if upper_note.key - primary.candidate.key >= 3:
            selected_secondary_scores = [
                secondary.score
                for secondary in residual_ranked[:4]
                if secondary.candidate.note_name in selected_names
            ]
            extension_candidate: tuple[NoteHypothesis, float] | None = None
            for hypothesis in residual_ranked[:8]:
                candidate = hypothesis.candidate
                if candidate.note_name in selected_names:
                    continue
                if not (primary.candidate.frequency < candidate.frequency < upper_note.frequency):
                    continue
                if upper_note.key - candidate.key > 1:
                    continue
                if candidate.key - primary.candidate.key < 2:
                    continue
                if hypothesis.score < primary.score * LOWER_MIXED_ROLL_EXTENSION_MIN_EXTENSION_SCORE_RATIO:
                    continue
                if hypothesis.score < GLISS_TERTIARY_MIN_SCORE:
                    continue
                if hypothesis.fundamental_ratio < LOWER_MIXED_ROLL_EXTENSION_MIN_FUNDAMENTAL_RATIO:
                    continue
                if any(are_harmonic_related(candidate, existing) for existing in selected):
                    continue
                if selected_secondary_scores and hypothesis.score < max(selected_secondary_scores) * LOWER_MIXED_ROLL_EXTENSION_MIN_UPPER_SCORE_RATIO:
                    continue
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if primary_onset_gain < LOWER_MIXED_ROLL_EXTENSION_MIN_PRIMARY_ONSET_GAIN:
                    continue
                onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, candidate.frequency)
                if onset_gain < LOWER_MIXED_ROLL_EXTENSION_MIN_EXTENSION_ONSET_GAIN:
                    continue
                extension_candidate = (hypothesis, onset_gain)
                break

            if extension_candidate is not None:
                hypothesis, onset_gain = extension_candidate
                selected.append(hypothesis.candidate)
                secondary_decision_trail.append(
                    {
                        "noteName": hypothesis.candidate.note_name,
                        "score": round(hypothesis.score, 6),
                        "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                        "onsetGain": round(onset_gain, 6),
                        "accepted": True,
                        "reasons": ["lower-mixed-roll-extension"],
                        "octaveDyadAllowed": False,
                    }
                )

    if (
        len(selected) == 3
        and end_time - start_time <= LOWER_ROLL_TAIL_EXTENSION_MAX_DURATION
        and primary.candidate.frequency == max(note.frequency for note in selected)
    ):
        selected_keys = sorted(note.key for note in selected)
        if selected_keys[-1] - selected_keys[0] == 2:
            selected_names = {note.note_name for note in selected}
            lowest_selected = min(selected, key=lambda note: note.frequency)
            extension_key = selected_keys[0] - 1
            extension_candidate: tuple[NoteHypothesis, float] | None = None
            for hypothesis in residual_ranked[:8]:
                candidate = hypothesis.candidate
                if candidate.note_name in selected_names:
                    continue
                if candidate.key != extension_key:
                    continue
                if hypothesis.fundamental_ratio < LOWER_ROLL_TAIL_EXTENSION_MIN_FUNDAMENTAL_RATIO:
                    continue
                if hypothesis.score < GLISS_TERTIARY_MIN_SCORE:
                    continue
                if any(are_harmonic_related(candidate, existing) for existing in selected):
                    continue
                lowest_selected_score = next((item.score for item in residual_ranked[:8] if item.candidate.note_name == lowest_selected.note_name), None)
                if lowest_selected_score is not None and hypothesis.score < lowest_selected_score * LOWER_ROLL_TAIL_EXTENSION_MIN_SCORE_RATIO:
                    continue
                if primary_onset_gain is None:
                    primary_onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, primary.candidate.frequency)
                if primary_onset_gain < LOWER_ROLL_TAIL_EXTENSION_MIN_PRIMARY_ONSET_GAIN:
                    continue
                onset_gain = onset_energy_gain(audio, sample_rate, start_time, end_time, candidate.frequency)
                if onset_gain > LOWER_ROLL_TAIL_EXTENSION_MAX_ONSET_GAIN:
                    continue
                extension_candidate = (hypothesis, onset_gain)
                break

            if extension_candidate is not None:
                hypothesis, onset_gain = extension_candidate
                selected.append(hypothesis.candidate)
                secondary_decision_trail.append(
                    {
                        "noteName": hypothesis.candidate.note_name,
                        "score": round(hypothesis.score, 6),
                        "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                        "onsetGain": round(onset_gain, 6),
                        "accepted": True,
                        "reasons": ["lower-roll-tail-extension"],
                        "octaveDyadAllowed": False,
                    }
                )

    debug_payload = None
    if debug:
        attack_context = prepare_attack_debug_context(audio, sample_rate, start_time, end_time)
        segment_attack_debug = attack_context["broadband"] if attack_context is not None else {}
        attack_profiles: dict[str, dict[str, Any]] = {}
        if attack_context is not None:
            for hypothesis in [primary, *ranked[:5], *residual_ranked[:5]]:
                note_name = hypothesis.candidate.note_name
                if note_name not in attack_profiles:
                    attack_profiles[note_name] = build_candidate_attack_debug(attack_context, hypothesis.candidate.frequency)
        debug_payload = {
            "startTime": round(start_time, 4),
            "endTime": round(end_time, 4),
            "durationSec": round(end_time - start_time, 4),
            "selectedNotes": [candidate.note_name for candidate in selected],
            **segment_attack_debug,
            "primaryCandidate": build_debug_candidates([primary], limit=1, attack_profiles=attack_profiles)[0],
            "primaryOnsetGain": None if primary_onset_gain is None else round(primary_onset_gain, 6),
            "primaryPromotion": primary_promotion_debug,
            "rankedCandidates": build_debug_candidates(ranked, attack_profiles=attack_profiles),
            "residualCandidates": build_debug_candidates(residual_ranked, attack_profiles=attack_profiles),
            "secondaryDecisionTrail": secondary_decision_trail,
            "rawPeaks": build_raw_peaks(frequencies, spectrum, tuning),
        }

    return sorted(selected, key=lambda item: item.frequency), debug_payload, primary

def split_ambiguous_upper_octave_pairs(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    split_events: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        duration = event.end_time - event.start_time
        if (
            len(event.notes) == 2
            and event.primary_score <= SPLIT_UPPER_OCTAVE_PAIR_PRIMARY_SCORE_MAX
            and harmonic_relation_multiple(event.notes[0], event.notes[1]) == 2.0
        ):
            higher_note = max(event.notes, key=lambda note: note.frequency)
            lower_note = min(event.notes, key=lambda note: note.frequency)
            next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
            previous_event = split_events[-1] if split_events else None
            current_note_names = {note.note_name for note in event.notes}
            previous_note_names = {note.note_name for note in previous_event.notes} if previous_event else set()
            next_note_names = {note.note_name for note in next_event.notes} if next_event else set()
            terminal_reset_octave_pair = (
                previous_event is not None
                and next_event is not None
                and len(previous_event.notes) == 1
                and len(next_event.notes) == 1
                and next_event.notes[0].frequency < lower_note.frequency
                and lower_note.frequency < previous_event.notes[0].frequency < higher_note.frequency
                and duration >= max(0.22, SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION - 0.08)
            )
            if (
                (SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION <= duration <= SPLIT_UPPER_OCTAVE_PAIR_MAX_DURATION or terminal_reset_octave_pair)
                and event.primary_note_name == higher_note.note_name
                and current_note_names != previous_note_names
                and current_note_names != next_note_names
            ):
                split_offset = max(0.08, min(duration * SPLIT_UPPER_OCTAVE_PAIR_FRACTION, duration - 0.08))
                split_time = event.start_time + split_offset
                split_events.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=split_time,
                        notes=[lower_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=lower_note.note_name,
                        primary_score=event.primary_score,
                    )
                )
                split_events.append(
                    RawEvent(
                        start_time=split_time,
                        end_time=event.end_time,
                        notes=[higher_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=higher_note.note_name,
                        primary_score=event.primary_score,
                    )
                )
                continue
        split_events.append(event)

    return split_events

def suppress_resonant_carryover(raw_events: list[RawEvent], tuning: InstrumentTuning | None = None) -> list[RawEvent]:
    if not raw_events:
        return []

    cleaned = [raw_events[0]]
    for index, event in enumerate(raw_events[1:], start=1):
        immediate_previous_notes = cleaned[-1].notes
        immediate_recent = {note.note_name for note in immediate_previous_notes}
        older_recent: set[str] = set()
        if len(cleaned) > 1:
            older_recent = {
                note.note_name
                for note in cleaned[-2].notes
                if any(are_harmonic_related(note, previous_note) for previous_note in immediate_previous_notes)
            }
        recent_notes = immediate_recent | older_recent

        updated_event = event
        if len(event.notes) == 2:
            repeated_notes = [note for note in event.notes if note.note_name in recent_notes]
            fresh_notes = [note for note in event.notes if note.note_name not in recent_notes]
            duration = event.end_time - event.start_time
            keep_short_octave_dyad = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1:
                keep_short_octave_dyad = (
                    harmonic_relation_multiple(repeated_notes[0], fresh_notes[0]) == 2.0
                    and duration <= 0.35
                )
            keep_phrase_reset_ascending_dyad = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                next_event = raw_events[index + 1]
                next_gap = next_event.start_time - event.end_time
                if (
                    len(next_event.notes) == 1
                    and next_gap >= RESONANT_CARRYOVER_PHRASE_RESET_MIN_GAP
                    and next_event.notes[0].frequency < repeated_notes[0].frequency
                ):
                    keep_phrase_reset_ascending_dyad = True
            keep_phrase_reset_lower_repeated = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                next_event = raw_events[index + 1]
                next_gap = next_event.start_time - event.end_time
                if (
                    len(next_event.notes) == 1
                    and duration <= RESONANT_CARRYOVER_HIGH_RETURN_MAX_DURATION
                    and next_gap <= RESONANT_CARRYOVER_HIGH_RETURN_MAX_NEXT_GAP
                    and repeated_notes[0].frequency < next_event.notes[0].frequency < fresh_notes[0].frequency
                    and cents_distance(repeated_notes[0].frequency, fresh_notes[0].frequency) >= RESONANT_CARRYOVER_HIGH_RETURN_MIN_INTERVAL_CENTS
                ):
                    keep_phrase_reset_lower_repeated = True
            keep_repeated_short_restart_overlap = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                previous_event = cleaned[-1]
                next_event = raw_events[index + 1]
                if (
                    len(previous_event.notes) == 1
                    and previous_event.notes[0].note_name == repeated_notes[0].note_name
                    and len(next_event.notes) == 1
                    and fresh_notes[0].frequency < repeated_notes[0].frequency < next_event.notes[0].frequency
                    and duration <= 0.14
                    and cents_distance(fresh_notes[0].frequency, repeated_notes[0].frequency) >= 300.0
                ):
                    keep_repeated_short_restart_overlap = True
            keep_descending_lower_repeated = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                next_event = raw_events[index + 1]
                if (
                    len(next_event.notes) == 1
                    and repeated_notes[0].frequency < fresh_notes[0].frequency
                    and next_event.notes[0].frequency < repeated_notes[0].frequency
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= cents_distance(repeated_notes[0].frequency, fresh_notes[0].frequency)
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(next_event.notes[0].frequency, repeated_notes[0].frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                ):
                    keep_descending_lower_repeated = True
            keep_high_register_repeated_lower_restart = False
            if (
                tuning is not None
                and len(repeated_notes) == 1
                and len(fresh_notes) == 1
                and index > 1
                and index + 1 < len(raw_events)
            ):
                previous_event = cleaned[-1]
                previous_previous_event = cleaned[-2]
                next_event = raw_events[index + 1]
                repeated_note = repeated_notes[0]
                fresh_note = fresh_notes[0]
                if (
                    len(previous_event.notes) == 1
                    and len(previous_previous_event.notes) == 1
                    and len(next_event.notes) == 1
                    and previous_event.notes[0].note_name == repeated_note.note_name
                    and previous_previous_event.notes[0].note_name == fresh_note.note_name
                    and repeated_note.octave >= 6
                    and fresh_note.octave >= 6
                    and next_event.notes[0].octave >= 6
                    and repeated_note.frequency < fresh_note.frequency < next_event.notes[0].frequency
                    and is_adjacent_tuning_step(repeated_note, fresh_note, tuning)
                    and is_adjacent_tuning_step(fresh_note, next_event.notes[0], tuning)
                    and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION
                ):
                    keep_high_register_repeated_lower_restart = True
            keep_post_triad_upper_tail_repeated = False
            if (
                tuning is not None
                and len(repeated_notes) == 1
                and len(fresh_notes) == 1
                and index + 1 < len(raw_events)
            ):
                previous_event = cleaned[-1]
                next_event = raw_events[index + 1]
                repeated_note = repeated_notes[0]
                fresh_note = fresh_notes[0]
                if (
                    len(previous_event.notes) >= 3
                    and len(next_event.notes) == 1
                    and repeated_note.note_name in {note.note_name for note in previous_event.notes}
                    and event.primary_note_name == repeated_note.note_name
                    and repeated_note.frequency < fresh_note.frequency
                    and next_event.notes[0].frequency < repeated_note.frequency
                    and is_adjacent_tuning_step(repeated_note, fresh_note, tuning)
                    and duration <= 0.14
                ):
                    keep_post_triad_upper_tail_repeated = True
            if (
                keep_phrase_reset_lower_repeated
                or keep_repeated_short_restart_overlap
                or keep_descending_lower_repeated
                or keep_high_register_repeated_lower_restart
                or keep_post_triad_upper_tail_repeated
            ):
                updated_event = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=repeated_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=repeated_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and (
                repeated_notes[0].frequency < fresh_notes[0].frequency
                or duration <= 0.14
            ) and not keep_short_octave_dyad and not keep_phrase_reset_ascending_dyad and not keep_phrase_reset_lower_repeated and not keep_repeated_short_restart_overlap and not keep_descending_lower_repeated and not keep_high_register_repeated_lower_restart and not keep_post_triad_upper_tail_repeated:
                updated_event = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=fresh_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=fresh_notes[0].note_name,
                    primary_score=event.primary_score,
                )
        cleaned.append(updated_event)

    return cleaned

def collapse_same_start_primary_singletons(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            cleaned.append(current)
            break

        following = raw_events[index + 1]
        shared_start = abs(current.start_time - following.start_time) <= SAME_START_PRIMARY_SINGLETON_MAX_START_DELTA
        if not shared_start:
            cleaned.append(current)
            index += 1
            continue

        current_primary = next((note for note in current.notes if note.note_name == current.primary_note_name), None)
        following_primary = next((note for note in following.notes if note.note_name == following.primary_note_name), None)
        if current_primary is None or following_primary is None:
            cleaned.append(current)
            index += 1
            continue

        current_non_primary = [note for note in current.notes if note.note_name != current.primary_note_name]
        following_non_primary = [note for note in following.notes if note.note_name != following.primary_note_name]
        current_is_primary_singleton = len(current.notes) == 1
        following_is_primary_singleton = len(following.notes) == 1
        current_note_names = {note.note_name for note in current.notes}
        following_note_names = {note.note_name for note in following.notes}
        shared_primary = current.primary_note_name and current.primary_note_name == following.primary_note_name

        prefer_current = (
            shared_primary
            and current_is_primary_singleton
            and following_non_primary
            and all(note.frequency < current_primary.frequency for note in following_non_primary)
        ) or (
            current_is_primary_singleton
            and current.primary_note_name in following_note_names
            and any(note.note_name != current.primary_note_name for note in following.notes)
            and all(note.note_name == current.primary_note_name or note.frequency < current_primary.frequency for note in following.notes)
        )
        prefer_following = (
            shared_primary
            and following_is_primary_singleton
            and current_non_primary
            and all(note.frequency < following_primary.frequency for note in current_non_primary)
        ) or (
            following_is_primary_singleton
            and following.primary_note_name in current_note_names
            and any(note.note_name != following.primary_note_name for note in current.notes)
            and all(note.note_name == following.primary_note_name or note.frequency < following_primary.frequency for note in current.notes)
        )

        if not prefer_current and not prefer_following:
            cleaned.append(current)
            index += 1
            continue

        phrase_reset_lower: NoteCandidate | None = None
        if (
            prefer_following
            and shared_primary
            and len(current_non_primary) == 1
            and following_is_primary_singleton
            and index + 2 < len(raw_events)
        ):
            lower_note = current_non_primary[0]
            next_event = raw_events[index + 2]
            next_primary = next((note for note in next_event.notes if note.note_name == next_event.primary_note_name), None)
            if (
                next_primary is not None
                and len(next_event.notes) == 1
                and next_primary.frequency > following_primary.frequency
                and cents_distance(lower_note.frequency, following_primary.frequency) >= 350.0
            ):
                phrase_reset_lower = lower_note

        if phrase_reset_lower is not None:
            preferred_notes = [phrase_reset_lower]
            preferred_primary = phrase_reset_lower.note_name
        else:
            preferred = current if prefer_current else following
            preferred_notes = preferred.notes
            preferred_primary = preferred.primary_note_name
        cleaned.append(
            RawEvent(
                start_time=min(current.start_time, following.start_time),
                end_time=max(current.end_time, following.end_time),
                notes=preferred_notes,
                is_gliss_like=current.is_gliss_like or following.is_gliss_like,
                primary_note_name=preferred_primary,
                primary_score=max(current.primary_score, following.primary_score),
            )
        )
        index += 2

    return cleaned


def collapse_restart_tail_subset_into_following_chord(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 2 >= len(raw_events):
            cleaned.append(current)
            index += 1
            continue

        following = raw_events[index + 1]
        trailing = raw_events[index + 2]
        if len(current.notes) == 1 and len(following.notes) == 1 and len(trailing.notes) == 2:
            current_note = current.notes[0]
            following_note = following.notes[0]
            trailing_note_names = {note.note_name for note in trailing.notes}
            trailing_lower_notes = [note for note in trailing.notes if note.frequency < following_note.frequency]
            following_duration = following.end_time - following.start_time
            gap_to_trailing = trailing.start_time - following.end_time
            if (
                following_note.note_name in trailing_note_names
                and trailing.primary_note_name == following_note.note_name
                and current_note.frequency < following_note.frequency
                and is_adjacent_tuning_step(current_note, following_note, tuning)
                and trailing_lower_notes
                and gap_to_trailing <= 0.04
                and following_duration <= 0.32
                and all(note.frequency < current_note.frequency for note in trailing_lower_notes)
            ):
                cleaned.append(current)
                cleaned.append(trailing)
                index += 3
                continue

        cleaned.append(current)
        index += 1

    return cleaned

def suppress_leading_single_transient(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    first_event = raw_events[0]
    next_event = raw_events[1]
    first_duration = first_event.end_time - first_event.start_time
    next_note_names = {note.note_name for note in next_event.notes}
    if (
        len(first_event.notes) == 1
        and first_event.notes[0].note_name in next_note_names
        and len(next_event.notes) >= 1
        and first_duration <= LEADING_SINGLE_TRANSIENT_MAX_DURATION
        and first_event.primary_score <= LEADING_SINGLE_TRANSIENT_MAX_SCORE
        and next_event.primary_score >= first_event.primary_score * LEADING_SINGLE_TRANSIENT_NEXT_SCORE_RATIO
    ):
        return raw_events[1:]

    return raw_events



def suppress_leading_gliss_subset_transients(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            current_duration = current.end_time - current.start_time
            gap = next_event.start_time - current.end_time
            current_note_names = {note.note_name for note in current.notes}
            next_note_names = {note.note_name for note in next_event.notes}
            if (
                len(current.notes) == 1
                and len(next_event.notes) >= 3
                and current_note_names < next_note_names
                and current_duration <= GLISS_LEADING_SUBSET_MAX_DURATION
                and gap <= GLISS_CLUSTER_MAX_GAP
                and next_event.primary_score >= current.primary_score * GLISS_LEADING_SUBSET_SCORE_RATIO
            ):
                index += 1
                continue
        cleaned.append(current)
        index += 1

    return cleaned

def suppress_low_confidence_dyad_transients(raw_events: list[RawEvent]) -> list[RawEvent]:
    cleaned: list[RawEvent] = []
    for event in raw_events:
        duration = event.end_time - event.start_time
        if len(event.notes) == 2 and ((duration <= LOW_CONFIDENCE_DYAD_MAX_DURATION and event.primary_score <= LOW_CONFIDENCE_DYAD_MAX_SCORE) or event.primary_score <= VERY_LOW_CONFIDENCE_EVENT_MAX_SCORE):
            continue
        cleaned.append(event)
    return cleaned

def simplify_short_secondary_bleed(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        updated_event = event
        duration = event.end_time - event.start_time
        if (
            len(event.notes) == 2
            and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
            and event.primary_score >= SHORT_SECONDARY_STRIP_MIN_SCORE
        ):
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                previous_event = raw_events[index - 1] if index > 0 else None
                next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
                lower_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency < primary_note.frequency]
                upper_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency > primary_note.frequency]
                sorted_notes = sorted(event.notes, key=lambda note: note.frequency)
                descending_bridge_to_upper = False
                upper_bridge_note: NoteCandidate | None = None
                if (
                    len(sorted_notes) == 2
                    and previous_event is not None
                    and len(previous_event.notes) == 1
                    and next_event is not None
                    and len(next_event.notes) == 1
                ):
                    lower_bridge_note = sorted_notes[0]
                    upper_bridge_note = sorted_notes[1]
                    previous_note = previous_event.notes[0]
                    next_note = next_event.notes[0]
                    descending_bridge_to_upper = (
                        previous_note.frequency > upper_bridge_note.frequency > next_note.frequency > lower_bridge_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(previous_note.frequency, upper_bridge_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(upper_bridge_note.frequency, next_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_note.frequency, lower_bridge_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    )
                if descending_bridge_to_upper and upper_bridge_note is not None:
                    updated_event = RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=[upper_bridge_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=upper_bridge_note.note_name,
                        primary_score=event.primary_score,
                    )
                    cleaned.append(updated_event)
                    continue
                if len(lower_notes) == 1:
                    next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
                    if (
                        next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].note_name == primary_note.note_name
                        and next_event.primary_score >= event.primary_score * SHORT_SECONDARY_STRIP_NEXT_SCORE_RATIO
                    ):
                        updated_event = RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=[primary_note],
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=primary_note.note_name,
                            primary_score=event.primary_score,
                        )
                    if previous_event is not None and next_event is not None and updated_event is event:
                        lower_note = lower_notes[0]
                    if previous_event is not None and next_event is not None and updated_event is event:
                        lower_note = lower_notes[0]
                        descending_restart_primary_repeat = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].frequency > primary_note.frequency
                            and next_event.primary_note_name == primary_note.note_name
                            and any(note.note_name == primary_note.note_name for note in next_event.notes)
                            and all(note.note_name == primary_note.note_name or note.frequency > primary_note.frequency for note in next_event.notes)
                        )
                        descending_step_handoff = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].note_name == primary_note.note_name
                            and len(next_event.notes) == 1
                            and next_event.notes[0].frequency < lower_note.frequency
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(lower_note.frequency, primary_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        )
                        descending_upper_residue = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].note_name == lower_note.note_name
                            and len(next_event.notes) == 1
                            and next_event.notes[0].frequency < lower_note.frequency
                            and 250.0
                            <= abs(cents_distance(lower_note.frequency, primary_note.frequency))
                            <= 550.0
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        )
                        descending_restart_upper_sandwich = (
                            len(previous_event.notes) == 1
                            and len(next_event.notes) == 1
                            and previous_event.notes[0].note_name == next_event.notes[0].note_name
                            and previous_event.notes[0].frequency > primary_note.frequency
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(lower_note.frequency, primary_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(previous_event.notes[0].frequency, primary_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        )
                        if (descending_step_handoff or descending_upper_residue or descending_restart_upper_sandwich or descending_restart_primary_repeat) and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION:
                            kept_note = lower_note
                            if descending_restart_upper_sandwich or descending_restart_primary_repeat:
                                kept_note = primary_note
                            updated_event = RawEvent(
                                start_time=event.start_time,
                                end_time=event.end_time,
                                notes=[kept_note],
                                is_gliss_like=event.is_gliss_like,
                                primary_note_name=kept_note.note_name,
                                primary_score=event.primary_score,
                            )
                elif len(upper_notes) == 1:
                    upper_note = upper_notes[0]
                    descending_restart_bridge = (
                        previous_event is not None
                        and len(previous_event.notes) == 1
                        and previous_event.notes[0].frequency < primary_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(previous_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency > upper_note.frequency
                        and cents_distance(upper_note.frequency, next_event.notes[0].frequency) >= 1200.0
                    )
                    if descending_restart_bridge:
                        continue
                    repeated_high_return = (
                        duration <= SHORT_SECONDARY_STRIP_MAX_DURATION
                        and previous_event is not None
                        and any(note.note_name == upper_note.note_name for note in previous_event.notes)
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and primary_note.frequency < next_event.notes[0].frequency < upper_note.frequency
                        and cents_distance(primary_note.frequency, upper_note.frequency) >= RESONANT_CARRYOVER_HIGH_RETURN_MIN_INTERVAL_CENTS
                    )
                    descending_step_handoff = (
                        previous_event is not None
                        and len(previous_event.notes) == 1
                        and previous_event.notes[0].note_name == upper_note.note_name
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency < primary_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= cents_distance(primary_note.frequency, upper_note.frequency)
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    )
                    descending_upper_residue = (
                        previous_event is not None
                        and len(previous_event.notes) == 1
                        and previous_event.notes[0].note_name == primary_note.note_name
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency < primary_note.frequency
                        and 250.0
                        <= cents_distance(primary_note.frequency, upper_note.frequency)
                        <= 550.0
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    )
                    mirrored_adjacent_run = False
                    if duration <= SHORT_SECONDARY_STRIP_MAX_DURATION and previous_event is not None and next_event is not None and index + 2 < len(raw_events):
                        next_next_event = raw_events[index + 2]
                        mirrored_adjacent_run = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].note_name == primary_note.note_name
                            and len(next_event.notes) == 2
                            and {note.note_name for note in next_event.notes} == {primary_note.note_name, upper_note.note_name}
                            and len(next_next_event.notes) == 1
                            and next_next_event.notes[0].frequency > upper_note.frequency
                            and cents_distance(primary_note.frequency, upper_note.frequency) <= 250.0
                        )
                    repeated_descending_handoff = False
                    if duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION and previous_event is not None and next_event is not None:
                        repeated_descending_handoff_head = False
                        if index + 2 < len(raw_events):
                            next_next_event = raw_events[index + 2]
                            repeated_descending_handoff_head = (
                                len(previous_event.notes) == 1
                                and previous_event.notes[0].note_name == upper_note.note_name
                                and {note.note_name for note in next_event.notes} == {primary_note.note_name, upper_note.note_name}
                                and len(next_next_event.notes) == 1
                                and next_next_event.notes[0].frequency < primary_note.frequency
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= cents_distance(primary_note.frequency, upper_note.frequency)
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= abs(cents_distance(next_next_event.notes[0].frequency, primary_note.frequency))
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            )
                        repeated_descending_handoff_tail = False
                        if index >= 2:
                            previous_previous_event = raw_events[index - 2]
                            repeated_descending_handoff_tail = (
                                {note.note_name for note in previous_event.notes} == {primary_note.note_name, upper_note.note_name}
                                and len(previous_previous_event.notes) == 1
                                and previous_previous_event.notes[0].note_name == upper_note.note_name
                                and len(next_event.notes) == 1
                                and next_event.notes[0].frequency < primary_note.frequency
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= cents_distance(primary_note.frequency, upper_note.frequency)
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            )
                        repeated_descending_handoff = repeated_descending_handoff_head or repeated_descending_handoff_tail
                    repeated_adjacent_tail = (
                        duration <= SHORT_SECONDARY_STRIP_MAX_DURATION
                        and previous_event is not None
                        and {note.note_name for note in previous_event.notes} == {primary_note.note_name, upper_note.note_name}
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency > upper_note.frequency
                        and cents_distance(primary_note.frequency, upper_note.frequency) <= 250.0
                    )
                    if repeated_high_return or repeated_descending_handoff or ((descending_step_handoff or descending_upper_residue) and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION) or mirrored_adjacent_run or repeated_adjacent_tail:
                        kept_note = primary_note if repeated_high_return else upper_note
                        if repeated_descending_handoff or ((descending_step_handoff or descending_upper_residue) and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION):
                            kept_note = primary_note
                        updated_event = RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=[kept_note],
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=kept_note.note_name,
                            primary_score=event.primary_score,
                        )
        cleaned.append(updated_event)

    return cleaned


def suppress_post_tail_gap_bridge_dyads(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        previous_event = raw_events[index - 1] if index > 0 else None
        next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
        if (
            previous_event is not None
            and next_event is not None
            and len(event.notes) == 2
            and len(previous_event.notes) == 1
            and len(next_event.notes) == 2
            and {note.note_name for note in next_event.notes} == {note.note_name for note in event.notes}
        ):
            duration = event.end_time - event.start_time
            gap_after = next_event.start_time - event.end_time
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                upper_note = next((note for note in event.notes if note.note_name != primary_note.note_name and note.frequency > primary_note.frequency), None)
                if upper_note is not None and (
                    duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
                    and gap_after >= 0.35
                    and previous_event.notes[0].frequency < primary_note.frequency
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(previous_event.notes[0].frequency, primary_note.frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(primary_note.frequency, upper_note.frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                ):
                    continue
        cleaned.append(event)

    return cleaned


def merge_short_gliss_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            merged.append(current)
            break

        following = raw_events[index + 1]
        gap = following.start_time - current.end_time
        current_duration = current.end_time - current.start_time
        following_duration = following.end_time - following.start_time
        combined_notes: dict[str, NoteCandidate] = {note.note_name: note for note in current.notes}
        for note in following.notes:
            combined_notes.setdefault(note.note_name, note)
        combined_keys = sorted(note.key for note in combined_notes.values())
        contiguous_keys = bool(combined_keys) and (combined_keys[-1] - combined_keys[0] + 1 == len(combined_keys))
        overlap_count = len(current.notes) + len(following.notes) - len(combined_notes)
        merge_pattern_ok = (
            (
                {len(current.notes), len(following.notes)} == {1, 2}
                and max(current_duration, following_duration) <= 0.18
            )
            or (
                len(current.notes) == 2
                and len(following.notes) == 2
                and overlap_count == 1
                and (following.is_gliss_like or following_duration <= 0.18)
            )
            or (min(len(current.notes), len(following.notes)) == 1 and max(len(current.notes), len(following.notes)) == 3 and overlap_count >= 1)
        )
        can_merge = (
            gap <= GLISS_CLUSTER_MAX_GAP
            and current_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and following_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and (following.end_time - current.start_time) <= GLISS_CLUSTER_MAX_TOTAL_DURATION
            and len(combined_notes) == GLISS_CLUSTER_TARGET_NOTE_COUNT
            and contiguous_keys
            and merge_pattern_ok
        )
        if can_merge:
            merged.append(
                RawEvent(
                    start_time=current.start_time,
                    end_time=following.end_time,
                    notes=sorted(combined_notes.values(), key=lambda note: note.frequency),
                    is_gliss_like=True,
                    primary_note_name=current.primary_note_name,
                    primary_score=max(current.primary_score, following.primary_score),
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def simplify_short_gliss_prefix_to_contiguous_singleton(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        updated_event = current
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            duration = current.end_time - current.start_time
            gap = next_event.start_time - current.end_time
            if len(current.notes) == 2 and duration <= 0.14 and gap <= GLISS_CLUSTER_MAX_GAP:
                candidate_unions: list[tuple[NoteCandidate, list[int]]] = []
                next_keys = sorted(note.key for note in next_event.notes)
                target_union_size = 4 if len(next_event.notes) >= 3 else 3
                if len(next_event.notes) >= 2 and not next_event.is_gliss_like:
                    for note in current.notes:
                        merged_keys = sorted(set(next_keys + [note.key]))
                        if merged_keys[-1] - merged_keys[0] + 1 == len(merged_keys) == target_union_size:
                            candidate_unions.append((note, merged_keys))
                if len(next_event.notes) >= 3:
                    candidate_unions = []
                    for note in current.notes:
                        merged_keys = sorted(set(next_keys + [note.key]))
                        if merged_keys[-1] - merged_keys[0] + 1 == len(merged_keys) == 4:
                            candidate_unions.append((note, merged_keys))
                if len(candidate_unions) == 1:
                    chosen_note = candidate_unions[0][0]
                    updated_event = RawEvent(
                        start_time=current.start_time,
                        end_time=current.end_time,
                        notes=[chosen_note],
                        is_gliss_like=True,
                        primary_note_name=chosen_note.note_name,
                        primary_score=current.primary_score,
                    )
        cleaned.append(updated_event)
        index += 1

    return cleaned


def merge_four_note_gliss_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            merged.append(current)
            break

        following = raw_events[index + 1]
        gap = following.start_time - current.end_time
        current_duration = current.end_time - current.start_time
        following_duration = following.end_time - following.start_time
        combined_notes: dict[str, NoteCandidate] = {note.note_name: note for note in current.notes}
        for note in following.notes:
            combined_notes.setdefault(note.note_name, note)
        combined_keys = sorted(note.key for note in combined_notes.values())
        contiguous_keys = bool(combined_keys) and (combined_keys[-1] - combined_keys[0] + 1 == len(combined_keys))
        overlap_count = len(current.notes) + len(following.notes) - len(combined_notes)
        merge_pattern_ok = (
            {len(current.notes), len(following.notes)} == {1, 3}
            or ({len(current.notes), len(following.notes)} == {2, 3} and overlap_count >= 1)
        )
        can_merge = (
            gap <= GLISS_CLUSTER_MAX_GAP
            and current_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and following_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and (following.end_time - current.start_time) <= GLISS_CLUSTER_MAX_TOTAL_DURATION
            and len(combined_notes) == 4
            and contiguous_keys
            and merge_pattern_ok
        )
        if can_merge:
            merged.append(
                RawEvent(
                    start_time=current.start_time,
                    end_time=following.end_time,
                    notes=sorted(combined_notes.values(), key=lambda note: note.frequency),
                    is_gliss_like=True,
                    primary_note_name=current.primary_note_name,
                    primary_score=max(current.primary_score, following.primary_score),
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def suppress_leading_gliss_neighbor_noise(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            current_duration = current.end_time - current.start_time
            gap = next_event.start_time - current.end_time
            current_note_names = {note.note_name for note in current.notes}
            next_note_names = {note.note_name for note in next_event.notes}
            if (
                len(current.notes) == 2
                and len(next_event.notes) >= 4
                and len(current_note_names & next_note_names) == 1
                and current_duration <= 0.14
                and gap <= GLISS_CLUSTER_MAX_GAP
                and next_event.primary_score >= current.primary_score * 1.2
            ):
                index += 1
                continue
        cleaned.append(current)
        index += 1

    return cleaned


def suppress_bridging_octave_pairs(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = [raw_events[0]]
    for index in range(1, len(raw_events) - 1):
        event = raw_events[index]
        previous = cleaned[-1]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time
        if (
            len(event.notes) == 2
            and len(previous.notes) == 1
            and len(next_event.notes) == 1
            and duration <= BRIDGING_OCTAVE_PAIR_MAX_DURATION
            and event.primary_score <= BRIDGING_OCTAVE_PAIR_MAX_SCORE
            and harmonic_relation_multiple(event.notes[0], event.notes[1]) == 2.0
        ):
            event_note_names = {note.note_name for note in event.notes}
            neighbor_note_names = {previous.notes[0].note_name, next_event.notes[0].note_name}
            if event_note_names == neighbor_note_names:
                continue
        cleaned.append(event)

    cleaned.append(raw_events[-1])
    return cleaned

def suppress_subset_decay_events(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = [raw_events[0]]
    for event in raw_events[1:]:
        previous = cleaned[-1]
        gap = event.start_time - previous.end_time
        previous_note_names = {note.note_name for note in previous.notes}
        event_note_names = {note.note_name for note in event.notes}
        if gap <= 0.02 and event_note_names < previous_note_names:
            continue
        cleaned.append(event)

    return cleaned

def suppress_short_residual_tails(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = [raw_events[0]]
    for index in range(1, len(raw_events) - 1):
        event = raw_events[index]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time
        gap_to_next = next_event.start_time - event.end_time
        if len(event.notes) == 1 and duration <= 0.14 and gap_to_next <= 0.12:
            recent_notes = {note.note_name for note in cleaned[-1].notes}
            if len(cleaned) > 1:
                recent_notes |= {note.note_name for note in cleaned[-2].notes}
            current_note = event.notes[0]
            next_note_names = {note.note_name for note in next_event.notes}
            if current_note.note_name in recent_notes and current_note.note_name not in next_note_names:
                continue
        cleaned.append(event)

    cleaned.append(raw_events[-1])
    return cleaned

def suppress_descending_terminal_residual_cluster(raw_events: list[RawEvent], tuning: InstrumentTuning) -> list[RawEvent]:
    if len(raw_events) < 5:
        return raw_events

    trailing = raw_events[-1]
    previous = raw_events[-2]
    if trailing.is_gliss_like or previous.is_gliss_like or len(trailing.notes) < 2 or len(trailing.notes) > 3:
        return raw_events
    if len(previous.notes) != 1 or (trailing.end_time - trailing.start_time) > 0.28:
        return raw_events

    rank_by_name = {
        note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))
    }
    previous_rank = rank_by_name.get(previous.notes[0].note_name)
    primary_rank = rank_by_name.get(trailing.primary_note_name)
    if previous_rank is None or primary_rank is None or primary_rank != previous_rank + 1:
        return raw_events

    suffix_names: set[str] = {previous.notes[0].note_name}
    expected_rank = previous_rank + 1
    for event in reversed(raw_events[:-2]):
        if len(event.notes) != 1 or event.is_gliss_like:
            break
        note_name = event.notes[0].note_name
        note_rank = rank_by_name.get(note_name)
        if note_rank != expected_rank:
            break
        suffix_names.add(note_name)
        expected_rank += 1
        if len(suffix_names) >= 4:
            break

    trailing_names = {note.note_name for note in trailing.notes}
    if len(suffix_names) < 3 or not trailing_names.issubset(suffix_names):
        return raw_events
    if not any(rank_by_name.get(name, -1) > primary_rank for name in trailing_names):
        return raw_events

    return raw_events[:-1]


def suppress_descending_restart_residual_cluster(raw_events: list[RawEvent], tuning: InstrumentTuning) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    rank_by_name = {
        note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))
    }
    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        previous_event = cleaned[-1] if cleaned else None
        event = raw_events[index]
        if (
            previous_event is not None
            and len(previous_event.notes) == 1
            and not previous_event.is_gliss_like
            and len(event.notes) == 2
            and not event.is_gliss_like
            and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION
            and index + 1 < len(raw_events)
        ):
            event_names = {note.note_name for note in event.notes}
            event_ranks = sorted(rank_by_name.get(note.note_name, -1) for note in event.notes)
            previous_rank = rank_by_name.get(previous_event.notes[0].note_name)
            if previous_rank is not None and len(event_ranks) == 2 and event_ranks[0] == previous_rank + 1 and event_ranks[1] == previous_rank + 2:
                run_end = index
                while run_end + 1 < len(raw_events) and {note.note_name for note in raw_events[run_end + 1].notes} == event_names:
                    repeated_event = raw_events[run_end + 1]
                    if repeated_event.is_gliss_like or (repeated_event.end_time - repeated_event.start_time) > DESCENDING_STEP_HANDOFF_MAX_DURATION:
                        break
                    run_end += 1
                if run_end + 1 < len(raw_events):
                    restart_event = raw_events[run_end + 1]
                    restart_rank = rank_by_name.get(restart_event.primary_note_name)
                    restart_gap = restart_event.start_time - raw_events[run_end].end_time
                    restart_gap_limit = 0.8
                    if restart_rank is not None and restart_rank >= event_ranks[1] + 10:
                        restart_gap_limit = 1.5
                    if (
                        len(restart_event.notes) == 1
                        and not restart_event.is_gliss_like
                        and restart_rank is not None
                        and restart_rank >= event_ranks[1] + 6
                        and restart_gap <= restart_gap_limit
                    ):
                        index = run_end + 1
                        continue
        cleaned.append(event)
        index += 1

    return cleaned


def collapse_late_descending_step_handoffs(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned = list(raw_events)
    for index in range(1, len(cleaned) - 1):
        previous_event = cleaned[index - 1]
        event = cleaned[index]
        next_event = cleaned[index + 1]
        duration = event.end_time - event.start_time
        if (
            len(previous_event.notes) == 1
            and len(event.notes) == 2
            and len(next_event.notes) == 1
            and not previous_event.is_gliss_like
            and not next_event.is_gliss_like
            and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
        ):
            lower_note, upper_note = sorted(event.notes, key=lambda note: note.frequency)
            if (
                previous_event.notes[0].note_name == upper_note.note_name
                and next_event.notes[0].frequency < lower_note.frequency
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= cents_distance(lower_note.frequency, upper_note.frequency)
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
            ):
                cleaned[index] = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=[lower_note],
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=lower_note.note_name,
                    primary_score=event.primary_score,
                )

    return cleaned


def merge_adjacent_events(raw_events: list[RawEvent]) -> list[RawEvent]:
    if not raw_events:
        return []

    merged = [raw_events[0]]
    for event in raw_events[1:]:
        previous = merged[-1]
        gap = event.start_time - previous.end_time
        same_notes = [note.note_name for note in previous.notes] == [note.note_name for note in event.notes]
        if same_notes and gap <= 0.12:
            merged[-1] = RawEvent(
                start_time=previous.start_time,
                end_time=event.end_time,
                notes=previous.notes,
                is_gliss_like=previous.is_gliss_like or event.is_gliss_like,
                primary_note_name=previous.primary_note_name,
                primary_score=max(previous.primary_score, event.primary_score),
            )
            continue
        merged.append(event)

    return merged


def split_adjacent_step_dyads_in_ascending_runs(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    note_rank = {note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))}

    def event_rank_range(event: RawEvent) -> tuple[int, int] | None:
        ranks = sorted({note_rank.get(note.note_name, -1) for note in event.notes})
        if not ranks or ranks[0] < 0:
            return None
        return ranks[0], ranks[-1]

    def is_split_candidate(event: RawEvent) -> tuple[int, int] | None:
        if event.is_gliss_like or len(event.notes) != 2 or (event.end_time - event.start_time) > ADJACENT_SEPARATED_DYAD_MAX_DURATION:
            return None
        rank_range = event_rank_range(event)
        if rank_range is None:
            return None
        low_rank, high_rank = rank_range
        if high_rank != low_rank + 1:
            return None
        return low_rank, high_rank

    def forward_support(index: int, current_high_rank: int) -> int:
        support = 0
        previous_high_rank = current_high_rank
        for next_index in range(index + 1, len(raw_events)):
            next_event = raw_events[next_index]
            if len(next_event.notes) > 2:
                break
            if next_event.is_gliss_like and len(next_event.notes) != 1:
                break
            next_range = event_rank_range(next_event)
            if next_range is None:
                break
            next_low_rank, next_high_rank = next_range
            if next_high_rank <= previous_high_rank or next_low_rank < previous_high_rank:
                break
            support += 1
            previous_high_rank = next_high_rank
            if support >= ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT:
                break
        return support

    split_events: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        split_candidate = is_split_candidate(event)
        if split_candidate is None:
            split_events.append(event)
            continue

        low_rank, high_rank = split_candidate
        if forward_support(index, high_rank) < ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT:
            split_events.append(event)
            continue

        previous_same_dyad = False
        if index > 0:
            previous_split_candidate = is_split_candidate(raw_events[index - 1])
            previous_same_dyad = previous_split_candidate == split_candidate
        previous_step_sequence = False
        if index > 1 and len(raw_events[index - 1].notes) == 1 and len(raw_events[index - 2].notes) == 1:
            previous_singleton_ranks = event_rank_range(raw_events[index - 1])
            older_singleton_ranks = event_rank_range(raw_events[index - 2])
            previous_step_sequence = (
                previous_singleton_ranks is not None
                and older_singleton_ranks is not None
                and previous_singleton_ranks[0] == previous_singleton_ranks[1] == high_rank
                and older_singleton_ranks[0] == older_singleton_ranks[1] == low_rank
            )
        next_is_higher_singleton = False
        if index + 1 < len(raw_events):
            next_range = event_rank_range(raw_events[index + 1])
            next_is_higher_singleton = next_range is not None and next_range[0] == next_range[1] and next_range[0] > high_rank
        if previous_step_sequence and next_is_higher_singleton:
            continue
        if previous_same_dyad and next_is_higher_singleton:
            upper_note = max(event.notes, key=lambda note: note.frequency)
            split_events.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=[upper_note],
                    is_gliss_like=False,
                    primary_note_name=upper_note.note_name,
                    primary_score=event.primary_score,
                )
            )
            continue

        ordered_notes = sorted(event.notes, key=lambda note: note.frequency)
        midpoint = event.start_time + ((event.end_time - event.start_time) * 0.5)
        split_events.append(
            RawEvent(
                start_time=event.start_time,
                end_time=midpoint,
                notes=[ordered_notes[0]],
                is_gliss_like=False,
                primary_note_name=ordered_notes[0].note_name,
                primary_score=event.primary_score,
            )
        )
        split_events.append(
            RawEvent(
                start_time=midpoint,
                end_time=event.end_time,
                notes=[ordered_notes[1]],
                is_gliss_like=False,
                primary_note_name=ordered_notes[1].note_name,
                primary_score=event.primary_score,
            )
        )

    return split_events

def merge_short_chord_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            merged.append(current)
            break

        following = raw_events[index + 1]
        gap = following.start_time - current.end_time
        current_names = {note.note_name for note in current.notes}
        following_names = {note.note_name for note in following.notes}
        combined: dict[str, NoteCandidate] = {note.note_name: note for note in current.notes}
        for note in following.notes:
            combined.setdefault(note.note_name, note)
        combined_keys = sorted(note.key for note in combined.values())
        contiguous_keys = bool(combined_keys) and (combined_keys[-1] - combined_keys[0] + 1 == len(combined_keys))
        combined_count = len(combined)
        total_duration = following.end_time - current.start_time
        is_singleton_plus_dyad = (
            {len(current.notes), len(following.notes)} == {1, 2}
            and not current_names.intersection(following_names)
            and not current.is_gliss_like
            and not following.is_gliss_like
            and min(current.end_time - current.start_time, following.end_time - following.start_time) <= CHORD_CLUSTER_MAX_SINGLETON_DURATION
        )
        is_subset_to_triad = (
            combined_count == 3
            and (current_names < following_names or following_names < current_names)
            and max(len(current.notes), len(following.notes)) == 3
        )
        if (
            gap <= CHORD_CLUSTER_MAX_GAP
            and total_duration <= CHORD_CLUSTER_MAX_TOTAL_DURATION
            and combined_count == 3
            and contiguous_keys
            and (is_singleton_plus_dyad or is_subset_to_triad)
        ):
            merged.append(
                RawEvent(
                    start_time=current.start_time,
                    end_time=following.end_time,
                    notes=sorted(combined.values(), key=lambda note: note.frequency),
                    is_gliss_like=current.is_gliss_like or following.is_gliss_like,
                    primary_note_name=following.primary_note_name,
                    primary_score=max(current.primary_score, following.primary_score),
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def normalize_repeated_four_note_family(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        if len(event_set) == 3:
            family_notes_by_name = {note.note_name: note for note in event.notes}
            family_set = set(event_set)
            merged_end_time = event.end_time
            merged_primary_name = event.primary_note_name
            merged_primary_score = event.primary_score
            merged_is_gliss_like = event.is_gliss_like
            consumed = 1
            added_new_note = False
            merged_family = False
            step_index = index + 1

            while step_index < len(raw_events) and consumed < 3:
                following = raw_events[step_index]
                following_set = frozenset(note.note_name for note in following.notes)
                gap = following.start_time - merged_end_time
                new_notes = following_set - family_set

                if gap > GLISS_CLUSTER_MAX_GAP or len(following_set) != 1:
                    break
                if len(new_notes) != 1 or len(family_set | following_set) != 4:
                    break

                if new_notes:
                    following_duration = following.end_time - following.start_time
                    if following_duration > 0.18 and not following.is_gliss_like:
                        break
                    added_new_note = True

                for note in following.notes:
                    family_notes_by_name.setdefault(note.note_name, note)
                family_set |= following_set
                merged_end_time = following.end_time
                merged_is_gliss_like = merged_is_gliss_like or following.is_gliss_like
                if merged_primary_name not in family_set and following.primary_note_name in family_set:
                    merged_primary_name = following.primary_note_name
                merged_primary_score = max(merged_primary_score, following.primary_score)
                consumed += 1
                step_index += 1

                if added_new_note and len(family_set) == 4:
                    family_notes = sorted(
                        (family_notes_by_name[name] for name in family_set),
                        key=lambda note: note.frequency,
                    )
                    normalized.append(
                        RawEvent(
                            start_time=event.start_time,
                            end_time=merged_end_time,
                            notes=family_notes,
                            is_gliss_like=merged_is_gliss_like,
                            primary_note_name=merged_primary_name if merged_primary_name in family_set else family_notes[0].note_name,
                            primary_score=merged_primary_score,
                        )
                    )
                    index += consumed
                    merged_family = True
                    break

            if merged_family:
                continue

        normalized.append(event)
        index += 1

    return normalized

def normalize_repeated_four_note_gliss_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    # Legacy no-op. Practical gliss behavior is now recovered by earlier local
    # front-end logic and by the anchor-bounded explicit four-note pass below.
    return raw_events

def collect_local_four_note_family_runs(raw_events: list[RawEvent]) -> list[dict[str, Any]]:
    exact_family_events: dict[frozenset[str], list[tuple[int, RawEvent]]] = {}
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        if len(event_set) == 4:
            exact_family_events.setdefault(event_set, []).append((index, event))

    candidate_runs: list[dict[str, Any]] = []
    for family_set, anchors in exact_family_events.items():
        if len(anchors) < 2:
            continue

        family_notes_by_name: dict[str, NoteCandidate] = {}
        for _, anchor in anchors:
            for note in anchor.notes:
                family_notes_by_name.setdefault(note.note_name, note)
        if len(family_notes_by_name) != 4:
            continue

        family_notes = sorted((family_notes_by_name[name] for name in family_set), key=lambda note: note.frequency)
        anchor_indices = [index for index, _ in anchors]
        dominant_score = float(np.median([anchor.primary_score for _, anchor in anchors]))
        runs_for_family: list[dict[str, Any]] = []

        for anchor_index in anchor_indices:
            start_index = anchor_index
            while start_index > 0:
                previous_event = raw_events[start_index - 1]
                gap = raw_events[start_index].start_time - previous_event.end_time
                previous_set = frozenset(note.note_name for note in previous_event.notes)
                previous_duration = previous_event.end_time - previous_event.start_time
                if previous_set <= family_set and previous_duration <= 1.25:
                    start_index -= 1
                    continue
                if (
                    previous_set
                    and not previous_set <= family_set
                    and len(previous_event.notes) <= 2
                    and previous_duration <= 0.18
                    and gap <= 0.12
                ):
                    start_index -= 1
                    continue
                break

            end_index = anchor_index
            while end_index + 1 < len(raw_events):
                next_event = raw_events[end_index + 1]
                gap = next_event.start_time - raw_events[end_index].end_time
                next_set = frozenset(note.note_name for note in next_event.notes)
                next_duration = next_event.end_time - next_event.start_time
                if next_set <= family_set and next_duration <= 1.25:
                    end_index += 1
                    continue
                if (
                    next_set
                    and not next_set <= family_set
                    and len(next_event.notes) <= 2
                    and next_duration <= 0.18
                    and gap <= 0.12
                ):
                    end_index += 1
                    continue
                break

            if runs_for_family and start_index <= runs_for_family[-1]["end_index"] + 1:
                runs_for_family[-1]["end_index"] = max(runs_for_family[-1]["end_index"], end_index)
            else:
                runs_for_family.append(
                    {
                        "family_set": family_set,
                        "family_notes": family_notes,
                        "dominant_score": dominant_score,
                        "start_index": start_index,
                        "end_index": end_index,
                    }
                )

        for run in runs_for_family:
            run_anchor_indices = [
                index for index in anchor_indices if run["start_index"] <= index <= run["end_index"]
            ]
            if len(run_anchor_indices) < 2:
                continue
            run["anchor_indices"] = run_anchor_indices
            candidate_runs.append(run)

    claimed_indices: dict[int, frozenset[str]] = {}
    ambiguous_indices: set[int] = set()
    for run in candidate_runs:
        for index in range(run["start_index"], run["end_index"] + 1):
            existing_family = claimed_indices.get(index)
            if existing_family is None:
                claimed_indices[index] = run["family_set"]
            elif existing_family != run["family_set"]:
                ambiguous_indices.add(index)

    resolved_runs: list[dict[str, Any]] = []
    for run in candidate_runs:
        if any(index in ambiguous_indices for index in range(run["start_index"], run["end_index"] + 1)):
            continue
        resolved_runs.append(run)

    return resolved_runs


def find_local_four_note_family_run(
    index: int,
    family_runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for run in family_runs:
        if run["start_index"] <= index <= run["end_index"]:
            return run
    return None


def normalize_repeated_explicit_four_note_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    family_runs = collect_local_four_note_family_runs(raw_events)
    if not family_runs:
        return raw_events

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        run = find_local_four_note_family_run(index, family_runs)
        if run is None:
            normalized.append(event)
            index += 1
            continue

        dominant_set = run["family_set"]
        dominant_notes = run["family_notes"]
        anchor_indices = run["anchor_indices"]
        event_set = frozenset(note.note_name for note in event.notes)
        duration = event.end_time - event.start_time
        lower_anchor_name = dominant_notes[0].note_name

        if event_set == dominant_set:
            normalized.append(event)
            index += 1
            continue

        previous_exact_distance = min((index - anchor_index for anchor_index in anchor_indices if anchor_index < index), default=999)
        next_exact_distance = min((anchor_index - index for anchor_index in anchor_indices if anchor_index > index), default=999)
        exact_support_before = previous_exact_distance <= 2
        exact_support_after = next_exact_distance <= 2

        next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
        next_set = frozenset(note.note_name for note in next_event.notes) if next_event is not None else frozenset()
        next_gap = next_event.start_time - event.end_time if next_event is not None else 1.0

        if next_event is not None and find_local_four_note_family_run(index + 1, family_runs) == run:
            next_exact_after = min((anchor_index - (index + 1) for anchor_index in anchor_indices if anchor_index > index + 1), default=999) <= 2
            if (
                next_set == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and event_set < dominant_set
                and len(event_set) == 1
                and event.primary_note_name == lower_anchor_name
                and duration <= 0.12
                and event.is_gliss_like
                and (exact_support_before or next_exact_after)
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=True,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

            if (
                next_set == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and event_set < dominant_set
                and len(event_set) == 3
                and duration <= 1.25
                and event.is_gliss_like
                and (exact_support_before or next_exact_after)
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=True,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

            if (
                next_set < dominant_set
                and len(event_set) >= 2
                and len(next_set) >= 2
                and event_set < dominant_set
                and (event_set | next_set) == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and max(duration, next_event.end_time - next_event.start_time) <= 1.25
                and not event.is_gliss_like
                and not next_event.is_gliss_like
                and next_exact_after
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=False,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

        if (
            event_set < dominant_set
            and len(event_set) in {2, 3}
            and duration <= 1.25
            and exact_support_before
            and exact_support_after
        ):
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=dominant_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name if event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            index += 1
            continue

        normalized.append(event)
        index += 1

    return normalized


def normalize_strict_four_note_subsets(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            next_set = frozenset(note.note_name for note in next_event.notes)
            next_gap = next_event.start_time - event.end_time
            previous_set = frozenset(note.note_name for note in normalized[-1].notes) if normalized else frozenset()
            future_support = any(
                frozenset(note.note_name for note in raw_events[future_index].notes) == next_set
                for future_index in range(index + 2, min(len(raw_events), index + 4))
            )
            if (
                len(event_set) == 2
                and len(next_set) == 4
                and event_set < next_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and not event.is_gliss_like
                and not next_event.is_gliss_like
                and (previous_set == next_set or future_support)
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=next_event.notes,
                        is_gliss_like=event.is_gliss_like or next_event.is_gliss_like,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in next_set else next_event.notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue
        normalized.append(event)
        index += 1

    return normalized


def normalize_repeated_triad_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    anchor_indices_by_set: dict[frozenset[str], list[int]] = {}
    anchor_notes_by_set: dict[frozenset[str], dict[str, NoteCandidate]] = {}
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        note_set_counts[event_set] = note_set_counts.get(event_set, 0) + 1
        if len(event_set) != 3:
            continue
        anchor_indices_by_set.setdefault(event_set, []).append(index)
        notes_by_name = anchor_notes_by_set.setdefault(event_set, {})
        for note in event.notes:
            notes_by_name.setdefault(note.note_name, note)

    if not anchor_indices_by_set:
        return raw_events

    def event_set_at(index: int) -> frozenset[str]:
        return frozenset(note.note_name for note in raw_events[index].notes)

    def is_local_fragment(index: int, anchor_set: frozenset[str]) -> bool:
        event_set = event_set_at(index)
        overlap = len(event_set & anchor_set)
        if overlap == 0:
            return False
        if event_set < anchor_set and len(event_set) <= 2:
            return True
        if len(event_set) == 2 and overlap >= 1:
            return True
        if len(event_set) == 3 and overlap == 2 and note_set_counts.get(event_set, 0) <= 1:
            return True
        return False

    local_actions: dict[int, tuple[str, frozenset[str]]] = {}
    conflicting_indices: set[int] = set()

    def assign_local_action(index: int, action: str, anchor_set: frozenset[str]) -> None:
        existing = local_actions.get(index)
        if existing is None:
            local_actions[index] = (action, anchor_set)
            return
        if existing != (action, anchor_set):
            conflicting_indices.add(index)

    def schedule_region(indices: list[int], anchor_set: frozenset[str], *, region: str, anchor_count: int) -> None:
        if not indices:
            return
        if region in {'head', 'tail'} and anchor_count < 3:
            return
        if not all(is_local_fragment(index, anchor_set) for index in indices):
            return
        for index in indices:
            event = raw_events[index]
            event_set = event_set_at(index)
            overlap = len(event_set & anchor_set)
            duration = event.end_time - event.start_time
            if event_set < anchor_set and len(event_set) <= 2:
                if region == 'head' and len(event_set) == 1 and duration <= 0.18:
                    assign_local_action(index, 'drop', anchor_set)
                else:
                    assign_local_action(index, 'rewrite', anchor_set)
                continue
            if len(event_set) == 3 and overlap == 2 and note_set_counts.get(event_set, 0) <= 1:
                assign_local_action(index, 'rewrite', anchor_set)
                continue
            if len(event_set) == 2 and overlap >= 1 and duration <= 0.2:
                assign_local_action(index, 'drop', anchor_set)

    candidate_sets = sorted(
        (
            (anchor_set, indices)
            for anchor_set, indices in anchor_indices_by_set.items()
            if len(indices) >= 2 and len(anchor_notes_by_set.get(anchor_set, {})) == 3
        ),
        key=lambda item: (-len(item[1]), item[1][0]),
    )

    for anchor_set, anchor_indices in candidate_sets:
        for left_anchor_index, right_anchor_index in zip(anchor_indices, anchor_indices[1:]):
            interior_indices = list(range(left_anchor_index + 1, right_anchor_index))
            schedule_region(interior_indices, anchor_set, region='between', anchor_count=len(anchor_indices))

        first_anchor_index = anchor_indices[0]
        head_indices = list(range(0, first_anchor_index))
        if first_anchor_index <= 2:
            schedule_region(head_indices, anchor_set, region='head', anchor_count=len(anchor_indices))

        last_anchor_index = anchor_indices[-1]
        tail_indices = list(range(last_anchor_index + 1, len(raw_events)))
        if 0 < len(tail_indices) <= 2:
            schedule_region(tail_indices, anchor_set, region='tail', anchor_count=len(anchor_indices))

    normalized: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if index in conflicting_indices:
            normalized.append(event)
            continue

        local_action = local_actions.get(index)
        if not local_action:
            normalized.append(event)
            continue

        action, anchor_set = local_action
        if action == 'drop':
            continue
        if action == 'rewrite':
            anchor_notes = anchor_notes_by_set.get(anchor_set)
            if anchor_notes is None or len(anchor_notes) != 3:
                normalized.append(event)
                continue
            dominant_notes = sorted((anchor_notes[name] for name in anchor_set), key=lambda note: note.frequency)
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=dominant_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name if event.primary_note_name in anchor_set else dominant_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            continue

        normalized.append(event)

    return normalized


def suppress_repeated_triad_blips(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        if len(event_set) == 3 and 0 < index < len(raw_events) - 1:
            previous_event = raw_events[index - 1]
            next_event = raw_events[index + 1]
            previous_set = frozenset(note.note_name for note in previous_event.notes)
            next_set = frozenset(note.note_name for note in next_event.notes)
            duration = event.end_time - event.start_time
            previous_duration = previous_event.end_time - previous_event.start_time
            next_duration = next_event.end_time - next_event.start_time
            if (
                previous_set == event_set
                and next_set == event_set
                and duration <= 0.35
                and duration <= min(previous_duration, next_duration) * 0.45
            ):
                continue
        cleaned.append(event)
    return cleaned


def suppress_isolated_triad_extensions(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if event.is_gliss_like:
            cleaned.append(event)
            continue
        event_note_set = frozenset(note.note_name for note in event.notes)
        if len(event.notes) == 3:
            candidate_subsets = [
                frozenset(event_note_set - {note.note_name})
                for note in event.notes
                if len(event_note_set - {note.note_name}) == 2
            ]
            best_subset: frozenset[str] | None = None
            best_support: tuple[int, float] | None = None
            for subset in candidate_subsets:
                if event.primary_note_name not in subset:
                    continue
                previous_support: tuple[int, float] | None = None
                for offset in range(max(0, index - 2), index):
                    previous_event = raw_events[offset]
                    previous_set = frozenset(note.note_name for note in previous_event.notes)
                    if previous_event.is_gliss_like or previous_set != subset:
                        continue
                    previous_support = (index - offset, event.start_time - previous_event.end_time)
                next_support: tuple[int, float] | None = None
                for offset in range(index + 1, min(len(raw_events), index + 3)):
                    next_event = raw_events[offset]
                    next_set = frozenset(note.note_name for note in next_event.notes)
                    if next_event.is_gliss_like or next_set != subset:
                        continue
                    next_support = (offset - index, next_event.start_time - event.end_time)
                    break
                if previous_support is None or next_support is None:
                    continue
                support_score = (
                    previous_support[0] + next_support[0],
                    previous_support[1] + next_support[1],
                )
                if best_support is None or support_score < best_support:
                    best_support = support_score
                    best_subset = subset
            if best_subset is not None:
                target_notes = [note for note in event.notes if note.note_name in best_subset]
                cleaned.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=sorted(target_notes, key=lambda note: note.frequency),
                        is_gliss_like=False,
                        primary_note_name=event.primary_note_name if event.primary_note_name in best_subset else target_notes[0].note_name,
                        primary_score=event.primary_score,
                    )
                )
                continue
        cleaned.append(event)

    return cleaned


def suppress_recent_upper_echo_mixed_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 5:
        return raw_events

    without_echo: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if 2 <= index < len(raw_events) - 2 and len(event.notes) == 1:
            previous_previous_event = raw_events[index - 2]
            previous_event = raw_events[index - 1]
            next_event = raw_events[index + 1]
            next_next_event = raw_events[index + 2]
            current_note = event.notes[0]
            if (
                len(previous_previous_event.notes) == 1
                and len(previous_event.notes) == 1
                and len(next_event.notes) == 3
                and len(next_next_event.notes) == 1
                and event.is_gliss_like
                and event.end_time - event.start_time <= 0.14
                and current_note.note_name == previous_previous_event.notes[0].note_name
                and previous_event.notes[0].frequency < current_note.frequency
                and 250.0 <= abs(cents_distance(previous_event.notes[0].frequency, current_note.frequency)) <= 350.0
                and previous_event.notes[0].note_name in {note.note_name for note in next_event.notes}
                and all(note.frequency < current_note.frequency for note in next_event.notes)
                and max(note.frequency for note in next_event.notes if note.note_name != previous_event.notes[0].note_name)
                < next_next_event.notes[0].frequency
                < previous_event.notes[0].frequency
            ):
                continue
        without_echo.append(event)

    cleaned: list[RawEvent] = []
    for index, event in enumerate(without_echo):
        updated_event = event
        if 2 <= index < len(without_echo) - 1 and len(event.notes) == 3:
            previous_previous_event = without_echo[index - 2]
            previous_event = without_echo[index - 1]
            next_event = without_echo[index + 1]
            if (
                len(previous_previous_event.notes) == 1
                and len(previous_event.notes) == 1
                and len(next_event.notes) == 1
            ):
                repeated_notes = [note for note in event.notes if note.note_name == previous_event.notes[0].note_name]
                fresh_notes = [note for note in event.notes if note.note_name != previous_event.notes[0].note_name]
                if len(repeated_notes) == 1 and len(fresh_notes) == 2:
                    repeated_note = repeated_notes[0]
                    if (
                        repeated_note.frequency == max(note.frequency for note in event.notes)
                        and previous_previous_event.notes[0].frequency > repeated_note.frequency
                        and 250.0 <= abs(cents_distance(previous_previous_event.notes[0].frequency, repeated_note.frequency)) <= 350.0
                        and max(note.frequency for note in fresh_notes) < next_event.notes[0].frequency < repeated_note.frequency
                        and event.end_time - event.start_time <= 0.55
                    ):
                        sorted_fresh_notes = sorted(fresh_notes, key=lambda note: note.frequency)
                        updated_event = RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=sorted_fresh_notes,
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=sorted_fresh_notes[-1].note_name,
                            primary_score=event.primary_score,
                        )
        cleaned.append(updated_event)

    return cleaned


def _debug_event_signature(raw_event: RawEvent) -> tuple[float, float, tuple[str, ...], bool]:
    return (
        round(raw_event.start_time, 4),
        round(raw_event.end_time, 4),
        tuple(sorted(note.note_name for note in raw_event.notes)),
        raw_event.is_gliss_like,
    )


def _debug_event_payload_from_signature(signature: tuple[float, float, tuple[str, ...], bool]) -> dict[str, Any]:
    start_time, end_time, notes, is_gliss_like = signature
    return {
        "startTime": start_time,
        "endTime": end_time,
        "notes": list(notes),
        "isGlissLike": is_gliss_like,
    }


def repeated_pattern_passes() -> tuple[RepeatedPatternPass, ...]:
    return (
        RepeatedPatternPass("normalize_repeated_explicit_four_note_patterns", normalize_repeated_explicit_four_note_patterns, merge_after=True),
        RepeatedPatternPass("normalize_repeated_triad_patterns", normalize_repeated_triad_patterns, merge_after=True),
        RepeatedPatternPass("normalize_strict_four_note_subsets", normalize_strict_four_note_subsets, merge_after=True),
        RepeatedPatternPass("suppress_repeated_triad_blips", suppress_repeated_triad_blips, merge_after=False),
        RepeatedPatternPass("suppress_isolated_triad_extensions", suppress_isolated_triad_extensions, merge_after=False),
    )


REPEATED_PATTERN_PASS_IDS = tuple(pass_entry.name for pass_entry in repeated_pattern_passes())


def apply_repeated_pattern_passes(
    raw_events: list[RawEvent],
    *,
    disabled_passes: frozenset[str] | None = None,
    debug: bool = False,
) -> tuple[list[RawEvent], list[dict[str, Any]]]:
    disabled = disabled_passes or frozenset()
    trace: list[dict[str, Any]] = []
    current = raw_events

    for pass_entry in repeated_pattern_passes():
        before_signatures = [_debug_event_signature(event) for event in current]
        next_events = current if pass_entry.name in disabled else pass_entry.fn(current)
        if pass_entry.merge_after:
            next_events = merge_adjacent_events(next_events)
        after_signatures = [_debug_event_signature(event) for event in next_events]

        if debug:
            before_counter = Counter(before_signatures)
            after_counter = Counter(after_signatures)
            removed: list[dict[str, Any]] = []
            added: list[dict[str, Any]] = []
            for signature, count in (before_counter - after_counter).items():
                removed.extend(_debug_event_payload_from_signature(signature) for _ in range(count))
            for signature, count in (after_counter - before_counter).items():
                added.extend(_debug_event_payload_from_signature(signature) for _ in range(count))
            trace.append(
                {
                    "pass": pass_entry.name,
                    "enabled": pass_entry.name not in disabled,
                    "mergeAfter": pass_entry.merge_after,
                    "beforeEventCount": len(before_signatures),
                    "afterEventCount": len(after_signatures),
                    "changed": before_signatures != after_signatures,
                    "removed": removed,
                    "added": added,
                }
            )

        current = next_events

    return current, trace


def quantize_beat(value: float, step: float = 0.25) -> float:
    return round(value / step) * step

def format_doremi(candidate: NoteCandidate) -> str:
    base = PITCH_CLASS_TO_DOREMI[candidate.pitch_class]
    if candidate.octave >= 6:
        return f"{base}.."
    if candidate.octave == 5:
        return f"{base}."
    if candidate.octave == 3:
        return f"_{base}"
    if candidate.octave <= 2:
        return f"__{base}"
    return base

def format_number(candidate: NoteCandidate) -> str:
    base = PITCH_CLASS_TO_NUMBER[candidate.pitch_class]
    if candidate.octave >= 6:
        return f"{base}''"
    if candidate.octave == 5:
        return f"{base}'"
    if candidate.octave == 3:
        return f".{base}"
    if candidate.octave <= 2:
        return f"..{base}"
    return base

def contiguous_note_cluster(raw_event: RawEvent) -> bool:
    keys = sorted(note.key for note in raw_event.notes)
    if len(keys) < 2:
        return False
    return keys[-1] - keys[0] + 1 == len(keys)
def classify_event_gesture(event: RawEvent, index: int, raw_events: list[RawEvent], merged_events: list[RawEvent]) -> str:
    if len(event.notes) <= 1:
        return "ambiguous"

    support_events = [
        raw_event
        for raw_event in raw_events
        if raw_event.start_time <= event.end_time + 0.03 and raw_event.end_time >= event.start_time - 0.03
    ]
    support_count = len(support_events)
    event_note_names = {note.note_name for note in event.notes}
    support_subsets = [
        {note.note_name for note in raw_event.notes}
        for raw_event in support_events
        if {note.note_name for note in raw_event.notes} < event_note_names
    ]
    all_support_gliss_like = bool(support_events) and all(raw_event.is_gliss_like for raw_event in support_events)
    support_key_ranges = [
        (min(note.key for note in raw_event.notes), max(note.key for note in raw_event.notes), raw_event.start_time)
        for raw_event in support_events
        if raw_event.notes
    ]
    support_key_ranges.sort(key=lambda item: item[2])

    previous_event = merged_events[index - 1] if index > 0 else None
    next_event = merged_events[index + 1] if index + 1 < len(merged_events) else None
    neighbor_progression = False
    for neighbor in (previous_event, next_event):
        if neighbor is None or len(neighbor.notes) != len(event.notes):
            continue
        if not contiguous_note_cluster(event) or not contiguous_note_cluster(neighbor):
            continue
        event_keys = sorted(note.key for note in event.notes)
        neighbor_keys = sorted(note.key for note in neighbor.notes)
        average_shift = float(np.mean([abs(a - b) for a, b in zip(event_keys, neighbor_keys)]))
        if 0.4 <= average_shift <= 3.5 and set(event_keys) != set(neighbor_keys):
            neighbor_progression = True
            break

    if event.is_gliss_like and contiguous_note_cluster(event) and neighbor_progression:
        return "slide_chord"

    if event.is_gliss_like and contiguous_note_cluster(event) and support_subsets:
        event_keys = sorted(note.key for note in event.notes)
        event_max_key = event_keys[-1]
        if support_key_ranges:
            first_min_key, first_max_key, _ = support_key_ranges[0]
            last_min_key, last_max_key, _ = support_key_ranges[-1]
            has_ascending_support_progression = (
                support_count >= 3
                and first_max_key < event_max_key
                and last_max_key >= event_max_key
                and last_min_key > first_min_key
            )
            if has_ascending_support_progression:
                return "slide_chord"
        if all_support_gliss_like:
            return "slide_chord"

    if support_count >= 2 and support_subsets:
        return "slide_chord"

    if not event.is_gliss_like:
        return "strict_chord"

    return "ambiguous"

def build_notation_views(events: list[ScoreEvent]) -> NotationViews:
    western = [" | ".join(note.pitch_class + str(note.octave) for note in event.notes) for event in events]
    numbered = [" ".join(note.label_number for note in event.notes) for event in events]
    vertical = [[note.label_doremi for note in event.notes] for event in events]
    return NotationViews(western=western, numbered=numbered, verticalDoReMi=vertical)

def build_recent_note_names(raw_events: list[RawEvent]) -> set[str] | None:
    if not raw_events:
        return None

    deduped_recent_events: list[RawEvent] = []
    last_note_set: tuple[str, ...] | None = None
    for recent_event in reversed(raw_events):
        note_set = tuple(note.note_name for note in recent_event.notes)
        if note_set == last_note_set:
            continue
        deduped_recent_events.append(recent_event)
        last_note_set = note_set
        if len(deduped_recent_events) >= 4:
            break

    recent_note_names: set[str] = set()
    for recent_event in deduped_recent_events:
        recent_note_names |= {note.note_name for note in recent_event.notes}
    return recent_note_names

def build_recent_ascending_primary_run_ceiling(raw_events: list[RawEvent]) -> float | None:
    if not raw_events:
        return None

    recent_primary_frequencies: list[float] = []
    for recent_event in reversed(raw_events):
        if recent_event.is_gliss_like or len(recent_event.notes) > 2 or recent_event.primary_note_name is None:
            break
        primary_note = next((note for note in recent_event.notes if note.note_name == recent_event.primary_note_name), None)
        if primary_note is None:
            break
        if recent_primary_frequencies and primary_note.frequency > recent_primary_frequencies[-1]:
            break
        recent_primary_frequencies.append(primary_note.frequency)
        if len(recent_primary_frequencies) >= 4:
            break

    if len(recent_primary_frequencies) < ASCENDING_PRIMARY_RUN_MIN_LENGTH:
        return None

    recent_primary_frequencies.reverse()
    if any(current < previous for previous, current in zip(recent_primary_frequencies, recent_primary_frequencies[1:])):
        return None
    if recent_primary_frequencies[-1] <= recent_primary_frequencies[0]:
        return None
    return recent_primary_frequencies[-1]



def build_recent_ascending_singleton_suffix(raw_events: list[RawEvent]) -> tuple[float | None, set[str]]:
    if not raw_events:
        return None, set()

    recent_primary_notes: list[NoteCandidate] = []
    for recent_event in reversed(raw_events):
        if len(recent_event.notes) != 1 or recent_event.primary_note_name is None:
            break
        primary_note = recent_event.notes[0]
        if primary_note.note_name != recent_event.primary_note_name:
            break
        if recent_primary_notes and primary_note.frequency > recent_primary_notes[-1].frequency:
            break
        recent_primary_notes.append(primary_note)
        if len(recent_primary_notes) >= 4:
            break

    if len(recent_primary_notes) < ASCENDING_PRIMARY_RUN_MIN_LENGTH:
        return None, set()

    recent_primary_notes.reverse()
    recent_primary_frequencies = [note.frequency for note in recent_primary_notes]
    if any(current < previous for previous, current in zip(recent_primary_frequencies, recent_primary_frequencies[1:])):
        return None, set()
    if recent_primary_frequencies[-1] <= recent_primary_frequencies[0]:
        return None, set()

    return recent_primary_frequencies[-1], {note.note_name for note in recent_primary_notes}


def build_recent_descending_primary_suffix(raw_events: list[RawEvent]) -> tuple[float | None, float | None, set[str]]:
    if not raw_events:
        return None, None, set()

    recent_primary_notes: list[NoteCandidate] = []
    for recent_event in reversed(raw_events):
        if recent_event.is_gliss_like or len(recent_event.notes) > 2 or recent_event.primary_note_name is None:
            break
        primary_note = next((note for note in recent_event.notes if note.note_name == recent_event.primary_note_name), None)
        if primary_note is None:
            break
        if recent_primary_notes and primary_note.frequency < recent_primary_notes[-1].frequency:
            break
        recent_primary_notes.append(primary_note)
        if len(recent_primary_notes) >= 4:
            break

    if len(recent_primary_notes) < DESCENDING_PRIMARY_SUFFIX_MIN_LENGTH:
        return None, None, set()

    recent_primary_notes.reverse()
    recent_primary_frequencies = [note.frequency for note in recent_primary_notes]
    if any(current > previous for previous, current in zip(recent_primary_frequencies, recent_primary_frequencies[1:])):
        return None, None, set()
    if recent_primary_frequencies[-1] >= recent_primary_frequencies[0]:
        return None, None, set()

    return (
        recent_primary_frequencies[-1],
        recent_primary_frequencies[0],
        {note.note_name for note in recent_primary_notes},
    )


async def transcribe_audio(
    upload: UploadFile,
    tuning: InstrumentTuning,
    *,
    debug: bool = False,
    disabled_repeated_pattern_passes: frozenset[str] | None = None,
) -> TranscriptionResult:
    audio, sample_rate = await read_audio(upload)
    normalized = normalize_audio(audio)
    segments, tempo, segment_debug = detect_segments(normalized, sample_rate)

    raw_events: list[RawEvent] = []
    segment_candidates_debug: list[dict[str, Any]] = []
    segment_contexts = build_segment_debug_contexts(
        segments,
        [tuple(item) for item in segment_debug.get("activeRanges", [])],
        [float(value) for value in segment_debug.get("onsetTimes", [])],
    ) if debug else {}
    sparse_gap_tail_segment_keys = {
        (round(start_time, 4), round(end_time, 4))
        for start_time, end_time in segment_debug.get("sparseGapTailSegments", [])
    }
    multi_onset_gap_segments = [
        (float(start_time), float(end_time))
        for start_time, end_time in segment_debug.get("multiOnsetGapSegments", [])
    ]
    multi_onset_gap_segment_keys = {
        (round(start_time, 4), round(end_time, 4))
        for start_time, end_time in multi_onset_gap_segments
    }
    single_onset_gap_head_segment_keys = {
        (round(start_time, 4), round(end_time, 4))
        for start_time, end_time in segment_debug.get("singleOnsetGapHeadSegments", [])
    }

    for start_time, end_time in segments:
        duration = max(end_time - start_time, 0.08)
        recent_note_names = build_recent_note_names(raw_events)
        ascending_primary_run_ceiling = build_recent_ascending_primary_run_ceiling(raw_events)
        ascending_singleton_suffix_ceiling, ascending_singleton_suffix_note_names = build_recent_ascending_singleton_suffix(raw_events)
        descending_primary_suffix_floor, descending_primary_suffix_ceiling, descending_primary_suffix_note_names = build_recent_descending_primary_suffix(raw_events)
        previous_primary_note_name = raw_events[-1].primary_note_name if raw_events else None
        previous_primary_frequency = None
        if raw_events and raw_events[-1].primary_note_name is not None:
            previous_primary = next((note for note in raw_events[-1].notes if note.note_name == raw_events[-1].primary_note_name), None)
            previous_primary_frequency = previous_primary.frequency if previous_primary is not None else None
        previous_primary_was_singleton = bool(raw_events and len(raw_events[-1].notes) == 1)
        candidates, candidate_debug, primary = segment_peaks(
            normalized,
            sample_rate,
            start_time,
            end_time,
            tuning,
            debug=debug,
            recent_note_names=recent_note_names,
            ascending_primary_run_ceiling=ascending_primary_run_ceiling,
            ascending_singleton_suffix_ceiling=ascending_singleton_suffix_ceiling,
            ascending_singleton_suffix_note_names=ascending_singleton_suffix_note_names,
            descending_primary_suffix_floor=descending_primary_suffix_floor,
            descending_primary_suffix_ceiling=descending_primary_suffix_ceiling,
            descending_primary_suffix_note_names=descending_primary_suffix_note_names,
            previous_primary_note_name=previous_primary_note_name,
            previous_primary_frequency=previous_primary_frequency,
            previous_primary_was_singleton=previous_primary_was_singleton,
        )
        if not candidates or primary is None:
            continue

        segment_key = (round(start_time, 4), round(end_time, 4))
        if debug and candidate_debug:
            candidate_debug.update(segment_contexts.get(segment_key, {}))
            candidate_debug["segmentSource"] = (
                "single_onset_gap_head" if segment_key in single_onset_gap_head_segment_keys
                else "multi_onset_gap" if segment_key in multi_onset_gap_segment_keys
                else "sparse_gap_tail" if segment_key in sparse_gap_tail_segment_keys
                else "active_or_gap"
            )
        if segment_key in sparse_gap_tail_segment_keys and max(note.octave for note in candidates) < 6:
            keep_gap_run_lead_in = any(
                later_start >= end_time + GAP_RUN_LEAD_IN_MIN_FOLLOWUP_GAP
                for later_start, _ in multi_onset_gap_segments
            )
            if not keep_gap_run_lead_in and not should_keep_low_register_sparse_gap_tail(
                candidates,
                tuning,
                descending_primary_suffix_floor,
                descending_primary_suffix_note_names,
            ):
                if debug and candidate_debug:
                    candidate_debug["droppedBy"] = "low_register_sparse_gap_tail"
                    segment_candidates_debug.append(candidate_debug)
                continue
            if debug and candidate_debug:
                candidate_debug["sparseGapTailAdjustment"] = (
                    "gap_run_lead_in_kept"
                    if keep_gap_run_lead_in
                    else "descending_restart_kept"
                )
        if segment_key in sparse_gap_tail_segment_keys:
            simplified_candidates = simplify_sparse_gap_tail_high_octave_dyad(candidates)
            if len(simplified_candidates) != len(candidates):
                candidates = simplified_candidates
                if debug and candidate_debug:
                    candidate_debug["selectedNotes"] = [candidate.note_name for candidate in candidates]
                    candidate_debug["sparseGapTailAdjustment"] = "high_register_octave_simplified"

        raw_events.append(
            RawEvent(
                start_time=start_time,
                end_time=end_time,
                notes=candidates,
                is_gliss_like=duration < 0.18,
                primary_note_name=primary.candidate.note_name,
                primary_score=primary.score,
            )
        )
        if debug and candidate_debug:
            segment_candidates_debug.append(candidate_debug)

    processed_events = suppress_low_confidence_dyad_transients(raw_events)
    processed_events = suppress_resonant_carryover(processed_events, tuning)
    processed_events = collapse_restart_tail_subset_into_following_chord(processed_events, tuning)
    processed_events = collapse_same_start_primary_singletons(processed_events)
    processed_events = simplify_short_secondary_bleed(processed_events)
    processed_events = suppress_post_tail_gap_bridge_dyads(processed_events)
    processed_events = suppress_leading_descending_overlap(processed_events, tuning)
    processed_events = simplify_descending_adjacent_dyad_residue(processed_events)
    processed_events = collapse_high_register_adjacent_bridge_dyads(processed_events, tuning)
    processed_events = suppress_descending_upper_singleton_spikes(processed_events)
    processed_events = suppress_short_descending_return_singletons(processed_events, tuning)
    processed_events = suppress_descending_upper_return_overlap(processed_events)
    processed_events = merge_short_gliss_clusters(processed_events)
    processed_events = simplify_short_gliss_prefix_to_contiguous_singleton(processed_events)
    processed_events = merge_four_note_gliss_clusters(processed_events)
    processed_events = suppress_leading_gliss_subset_transients(processed_events)
    processed_events = suppress_leading_gliss_neighbor_noise(processed_events)
    processed_events = suppress_leading_single_transient(processed_events)
    processed_events = suppress_subset_decay_events(processed_events)
    processed_events = split_ambiguous_upper_octave_pairs(processed_events)
    processed_events = suppress_bridging_octave_pairs(processed_events)
    processed_events = suppress_short_residual_tails(processed_events)
    processed_events = suppress_descending_terminal_residual_cluster(processed_events, tuning)
    processed_events = suppress_descending_restart_residual_cluster(processed_events, tuning)
    processed_events = collapse_late_descending_step_handoffs(processed_events)
    merged_events = merge_adjacent_events(processed_events)
    merged_events = collapse_late_descending_step_handoffs(merged_events)
    merged_events = merge_short_chord_clusters(merged_events)
    merged_events = merge_adjacent_events(merged_events)
    merged_events = suppress_recent_upper_echo_mixed_clusters(merged_events)
    merged_events = collapse_ascending_restart_lower_residue_singletons(merged_events, tuning)
    merged_events = merge_adjacent_events(merged_events)
    merged_events, repeated_pattern_pass_trace = apply_repeated_pattern_passes(
        merged_events,
        disabled_passes=disabled_repeated_pattern_passes,
        debug=debug,
    )
    merged_events = split_adjacent_step_dyads_in_ascending_runs(merged_events, tuning)
    if not merged_events:
        raise HTTPException(status_code=422, detail="No musical notes were detected. Try a clearer recording or a different tuning.")

    beat_seconds = 60.0 / tempo
    events: list[ScoreEvent] = []
    warnings: list[str] = []

    for index, event in enumerate(merged_events, start=1):
        start_beat = quantize_beat(event.start_time / beat_seconds)
        duration_beat = max(quantize_beat((event.end_time - event.start_time) / beat_seconds), 0.25)
        notes = [
            ScoreNote(
                key=candidate.key,
                pitchClass=candidate.pitch_class,
                octave=candidate.octave,
                labelDoReMi=format_doremi(candidate),
                labelNumber=format_number(candidate),
                frequency=round(candidate.frequency, 3),
            )
            for candidate in event.notes
        ]
        events.append(
            ScoreEvent(
                id=f"evt-{index}",
                startBeat=start_beat,
                durationBeat=duration_beat,
                notes=notes,
                isGlissLike=event.is_gliss_like,
                gesture=classify_event_gesture(event, index - 1, raw_events, merged_events),
            )
        )

    if len(events) < 3:
        warnings.append("Only a small number of note events were detected.")

    result_debug = None
    if debug:
        result_debug = {
            **segment_debug,
            "segmentCandidates": segment_candidates_debug,
            "mergedEvents": [
                {
                    "startTime": round(event.start_time, 4),
                    "endTime": round(event.end_time, 4),
                    "notes": [candidate.note_name for candidate in event.notes],
                    "gesture": classify_event_gesture(event, index, raw_events, merged_events),
                }
                for index, event in enumerate(merged_events)
            ],
            "disabledRepeatedPatternPasses": sorted(disabled_repeated_pattern_passes or ()),
            "repeatedPatternPassTrace": repeated_pattern_pass_trace,
        }

    return TranscriptionResult(
        instrumentTuning=tuning,
        tempo=round(tempo, 2),
        events=events,
        notationViews=build_notation_views(events),
        warnings=warnings,
        debug=result_debug,
    )
















