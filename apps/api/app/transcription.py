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
ATTACK_ANALYSIS_SECONDS = 0.16
ATTACK_ANALYSIS_RATIO = 0.35
ONSET_ENERGY_WINDOW_SECONDS = 0.08
MIN_RECENT_NOTE_ONSET_GAIN = 2.5
RECENT_PRIMARY_REPLACEMENT_MIN_SCORE_RATIO = 0.18
RECENT_PRIMARY_REPLACEMENT_MIN_FUNDAMENTAL_RATIO = 0.6
RECENT_PRIMARY_REPLACEMENT_RELAXED_FUNDAMENTAL_RATIO = 0.45
RECENT_PRIMARY_REPLACEMENT_STRONG_ONSET_GAIN = 100.0
RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_GAIN = 20.0
RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_RATIO = 8.0
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
ASCENDING_PRIMARY_RUN_MIN_LENGTH = 3
ASCENDING_PRIMARY_RUN_MAX_DURATION = 0.45
ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO = 0.35
ASCENDING_PRIMARY_RUN_RECENT_SECONDARY_ONSET_GAIN = 8.0
ADJACENT_SEPARATED_DYAD_MAX_DURATION = 0.6
ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT = 2
PRIOR_ONSET_BACKTRACK_SECONDS = 0.55
HARMONIC_WEIGHTS = [1.0, 0.55, 0.3, 0.15]
HARMONIC_BAND_CENTS = 40.0
SUPPRESSION_BAND_CENTS = 45.0
MAX_POLYPHONY = 2
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
FOUR_NOTE_GLISS_EXTENSION_MAX_DURATION = 1.0
FOUR_NOTE_GLISS_EXTENSION_SCORE_RATIO = 0.02
FOUR_NOTE_GLISS_EXTENSION_MIN_SCORE = 18.0
FOUR_NOTE_GLISS_EXTENSION_MIN_FUNDAMENTAL_RATIO = 0.82
CHORD_CLUSTER_MAX_GAP = 0.08
CHORD_CLUSTER_MAX_SINGLETON_DURATION = 0.22
CHORD_CLUSTER_MAX_TOTAL_DURATION = 1.6
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
OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO = 0.06
OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO = 0.95
OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO = 0.05
OCTAVE_DYAD_UPPER_SCORE_RATIO = 0.03
LOW_CONFIDENCE_DYAD_MAX_DURATION = 0.25
LOW_CONFIDENCE_DYAD_MAX_SCORE = 120.0
SHORT_SECONDARY_STRIP_MAX_DURATION = 0.28
SHORT_SECONDARY_STRIP_MIN_SCORE = 60.0
SHORT_SECONDARY_STRIP_NEXT_SCORE_RATIO = 5.0
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
    "C": "ド",
    "C#": "ド#",
    "Db": "レb",
    "D": "レ",
    "D#": "レ#",
    "Eb": "ミb",
    "E": "ミ",
    "F": "ファ",
    "F#": "ファ#",
    "Gb": "ソb",
    "G": "ソ",
    "G#": "ソ#",
    "Ab": "ラb",
    "A": "ラ",
    "A#": "ラ#",
    "Bb": "シb",
    "B": "シ",
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
        audio = np.mean(audio, axis=1)

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

    gap_injected_segments: list[tuple[float, float]] = []
    qualifying_gap_run: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        gap_onsets = [time for time in onset_times if previous_end + 0.05 < time < next_start - 0.05]
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

    segments: list[tuple[float, float]] = []
    for range_start, range_end in active_ranges:
        effective_range_start = range_start
        prior_onsets = [time for time in onset_times if range_start - PRIOR_ONSET_BACKTRACK_SECONDS <= time <= range_start + 0.005]
        if prior_onsets:
            effective_range_start = prior_onsets[-1]

        range_onsets = [time for time in onset_times if effective_range_start + 0.005 < time < range_end - 0.05]
        supplemental_starts = [
            start
            for start, _ in raw_active_ranges
            if effective_range_start + 0.18 < start < range_end - 0.05
            and all(abs(start - onset) >= 0.24 for onset in range_onsets)
        ]
        boundary_times = sorted([*range_onsets, *supplemental_starts])
        deduped_onsets: list[float] = []
        for time in boundary_times:
            if not deduped_onsets or time - deduped_onsets[-1] >= 0.18:
                deduped_onsets.append(time)

        starts = [effective_range_start, *deduped_onsets]
        for index, start_time in enumerate(starts):
            end_time = starts[index + 1] if index + 1 < len(starts) else range_end
            if end_time - start_time >= 0.08:
                segments.append((start_time, end_time))

    for start_time, end_time in gap_injected_segments:
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
        "activeRanges": [[round(start, 4), round(end, 4)] for start, end in active_ranges],
        "gapInjectedSegments": [[round(start, 4), round(end, 4)] for start, end in gap_injected_segments],
        "segments": [[round(start, 4), round(end, 4)] for start, end in segments],
        "rmsThreshold": round(threshold, 6),
        "tempoRaw": round(tempo, 4),
        "tempoHopLength": TEMPO_ESTIMATION_HOP_LENGTH,
        "tempoAudioDurationSec": round(tempo_audio_duration_sec, 4),
        "tempoEstimationMs": round(tempo_estimation_ms, 3),
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
        if hypothesis.candidate.octave <= 4:
            return False
        if hypothesis.fundamental_ratio < OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO:
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

def build_debug_candidates(ranked: list[NoteHypothesis], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
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
        for hypothesis in ranked[:limit]
    ]

def maybe_replace_stale_recent_primary(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    primary: NoteHypothesis,
    ranked: list[NoteHypothesis],
    recent_note_names: set[str] | None,
) -> tuple[NoteHypothesis, float | None, dict[str, Any] | None]:
    if not recent_note_names or primary.candidate.note_name not in recent_note_names:
        return primary, None, None

    duration = end_time - start_time
    if duration > 0.45:
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
        if hypothesis.fundamental_ratio < RECENT_PRIMARY_REPLACEMENT_MIN_FUNDAMENTAL_RATIO and not relaxed_recent_primary:
            continue

        if (
            onset_gain >= RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_GAIN
            and onset_gain >= max(primary_onset_gain, 1e-6) * RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_RATIO
        ):
            return (
                hypothesis,
                onset_gain,
                {
                    "replacedPrimaryNote": primary.candidate.note_name,
                    "replacementNote": hypothesis.candidate.note_name,
                    "replacedPrimaryOnsetGain": round(primary_onset_gain, 6),
                    "replacementOnsetGain": round(onset_gain, 6),
                },
            )

    return primary, primary_onset_gain, None

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
    previous_primary_note_name: str | None = None,
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
    )
    selected = [primary.candidate]
    residual_ranked: list[NoteHypothesis] = []
    secondary_decision_trail: list[dict[str, Any]] = []
    secondary_score_ratio = SECONDARY_SCORE_RATIO
    if end_time - start_time <= 0.14:
        secondary_score_ratio = SHORT_SEGMENT_SECONDARY_SCORE_RATIO
    secondary_min_fundamental_ratio = SECONDARY_MIN_FUNDAMENTAL_RATIO

    if MAX_POLYPHONY > 1:
        residual_spectrum = suppress_harmonics(spectrum, frequencies, primary.candidate.frequency)
        residual_ranked = rank_tuning_candidates(frequencies, residual_spectrum, tuning, debug=debug)
        for hypothesis in residual_ranked[:8]:
            reasons: list[str] = []
            onset_gain: float | None = None
            octave_dyad_allowed = allow_octave_secondary(primary, hypothesis, selected)
            segment_duration = end_time - start_time
            score_ratio = secondary_score_ratio
            if octave_dyad_allowed and hypothesis.candidate.frequency > primary.candidate.frequency:
                score_ratio = min(score_ratio, OCTAVE_DYAD_UPPER_SCORE_RATIO)
            if hypothesis.candidate.note_name == primary.candidate.note_name:
                reasons.append("same-as-primary")
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
            secondary_decision_trail.append(
                {
                    "noteName": hypothesis.candidate.note_name,
                    "score": round(hypothesis.score, 6),
                    "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                    "onsetGain": None if onset_gain is None else round(onset_gain, 6),
                    "accepted": accepted,
                    "reasons": reasons,
                    "octaveDyadAllowed": octave_dyad_allowed,
                }
            )
            if accepted:
                selected.append(hypothesis.candidate)
                break

    if len(selected) == 2 and end_time - start_time <= GLISS_TERTIARY_MAX_DURATION:
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

    debug_payload = None
    if debug:
        debug_payload = {
            "startTime": round(start_time, 4),
            "endTime": round(end_time, 4),
            "durationSec": round(end_time - start_time, 4),
            "selectedNotes": [candidate.note_name for candidate in selected],
            "primaryCandidate": build_debug_candidates([primary], limit=1)[0],
            "primaryOnsetGain": None if primary_onset_gain is None else round(primary_onset_gain, 6),
            "primaryPromotion": primary_promotion_debug,
            "rankedCandidates": build_debug_candidates(ranked),
            "residualCandidates": build_debug_candidates(residual_ranked),
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
            and SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION <= duration <= SPLIT_UPPER_OCTAVE_PAIR_MAX_DURATION
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
            if (
                event.primary_note_name == higher_note.note_name
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

def suppress_resonant_carryover(raw_events: list[RawEvent]) -> list[RawEvent]:
    if not raw_events:
        return []

    cleaned = [raw_events[0]]
    for event in raw_events[1:]:
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
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and (
                repeated_notes[0].frequency < fresh_notes[0].frequency
                or duration <= 0.14
            ) and not keep_short_octave_dyad:
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

        preferred = current if prefer_current else following
        cleaned.append(
            RawEvent(
                start_time=min(current.start_time, following.start_time),
                end_time=max(current.end_time, following.end_time),
                notes=preferred.notes,
                is_gliss_like=current.is_gliss_like or following.is_gliss_like,
                primary_note_name=preferred.primary_note_name,
                primary_score=max(current.primary_score, following.primary_score),
            )
        )
        index += 2

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
            and duration <= SHORT_SECONDARY_STRIP_MAX_DURATION
            and event.primary_score >= SHORT_SECONDARY_STRIP_MIN_SCORE
        ):
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                lower_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency < primary_note.frequency]
                if len(lower_notes) == 1 and index + 1 < len(raw_events):
                    next_event = raw_events[index + 1]
                    if (
                        len(next_event.notes) == 1
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
        cleaned.append(updated_event)

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
            {len(current.notes), len(following.notes)} == {1, 2}
            or (len(current.notes) == 2 and len(following.notes) == 2 and overlap_count == 1)
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
            if len(current.notes) == 2 and len(next_event.notes) >= 3 and duration <= 0.14 and gap <= GLISS_CLUSTER_MAX_GAP:
                candidate_unions: list[tuple[NoteCandidate, list[int]]] = []
                next_keys = sorted(note.key for note in next_event.notes)
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
    if len(raw_events) < 4:
        return raw_events

    triad_counts: dict[frozenset[str], int] = {}
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        if len(note_set) == 3:
            triad_counts[note_set] = triad_counts.get(note_set, 0) + 1

    if len(triad_counts) < 2:
        return raw_events

    best_family: tuple[frozenset[str], frozenset[str], frozenset[str], int] | None = None
    triad_items = list(triad_counts.items())
    for index, (first_set, first_count) in enumerate(triad_items):
        for second_set, second_count in triad_items[index + 1:]:
            union = first_set | second_set
            overlap = first_set & second_set
            score = first_count + second_count
            dominant_count = max(first_count, second_count)
            if len(union) != 4 or len(overlap) < 2 or min(first_count, second_count) < 1 or dominant_count != 3:
                continue
            candidate = (union, first_set, second_set, score)
            if best_family is None or candidate[3] > best_family[3]:
                best_family = candidate

    if best_family is None:
        return raw_events

    family_set, triad_a, triad_b, _ = best_family
    family_triads = {triad_a, triad_b}
    exclusive_family_notes = triad_a ^ triad_b
    has_supporting_subset = any(
        0 < len(frozenset(note.note_name for note in event.notes)) < 3
        and frozenset(note.note_name for note in event.notes) < family_set
        and bool(frozenset(note.note_name for note in event.notes) & exclusive_family_notes)
        for event in raw_events
    )
    if not has_supporting_subset:
        return raw_events
    family_notes_by_name: dict[str, NoteCandidate] = {}
    family_events = [event for event in raw_events if frozenset(note.note_name for note in event.notes) in family_triads]
    for event in family_events:
        for note in event.notes:
            if note.note_name in family_set:
                family_notes_by_name.setdefault(note.note_name, note)
    if len(family_notes_by_name) != 4:
        return raw_events
    family_notes = sorted((family_notes_by_name[name] for name in family_set), key=lambda note: note.frequency)

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        duration = event.end_time - event.start_time

        if event_set in family_triads:
            if index + 1 < len(raw_events):
                following = raw_events[index + 1]
                following_set = frozenset(note.note_name for note in following.notes)
                gap = following.start_time - event.end_time
                if gap <= 0.02 and following_set < family_set and len(following_set) <= 2 and following_set & (family_set - event_set):
                    normalized.append(
                        RawEvent(
                            start_time=event.start_time,
                            end_time=following.end_time,
                            notes=family_notes,
                            is_gliss_like=event.is_gliss_like or following.is_gliss_like,
                            primary_note_name=event.primary_note_name if event.primary_note_name in family_set else family_notes[0].note_name,
                            primary_score=max(event.primary_score, following.primary_score),
                        )
                    )
                    index += 2
                    continue
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=family_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name if event.primary_note_name in family_set else family_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            index += 1
            continue

        if event_set < family_set:
            prev_family = len(normalized) > 0 and frozenset(note.note_name for note in normalized[-1].notes) == family_set
            next_family = index + 1 < len(raw_events) and frozenset(note.note_name for note in raw_events[index + 1].notes) in family_triads
            if len(event_set) <= 2 and ((prev_family and duration <= 1.0) or (next_family and duration <= 0.24)):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=family_notes,
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=event.primary_note_name if event.primary_note_name in family_set else family_notes[0].note_name,
                        primary_score=event.primary_score,
                    )
                )
                index += 1
                continue

        if len(event_set) == 1 and not (event_set <= family_set) and duration <= 0.24:
            prev_family = len(normalized) > 0 and frozenset(note.note_name for note in normalized[-1].notes) == family_set
            next_family = index + 1 < len(raw_events) and frozenset(note.note_name for note in raw_events[index + 1].notes) in family_triads
            if prev_family or next_family:
                index += 1
                continue

        normalized.append(event)
        index += 1

    collapsed: list[RawEvent] = []
    index = 0
    while index < len(normalized):
        event = normalized[index]
        if index + 1 < len(normalized):
            next_event = normalized[index + 1]
            event_set = frozenset(note.note_name for note in event.notes)
            next_set = frozenset(note.note_name for note in next_event.notes)
            next_gap = next_event.start_time - event.end_time
            if (
                len(event_set) == 2
                and event_set < family_set
                and next_set == family_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
            ):
                collapsed.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=family_notes,
                        is_gliss_like=False,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in family_set else family_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue
        collapsed.append(event)
        index += 1

    return collapsed



def normalize_repeated_four_note_gliss_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 5:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        if len(note_set) == 4:
            note_set_counts[note_set] = note_set_counts.get(note_set, 0) + 1

    if not note_set_counts:
        return raw_events

    dominant_set, dominant_count = max(note_set_counts.items(), key=lambda item: item[1])
    if dominant_count < 2 or dominant_count >= 4:
        return raw_events

    dominant_events = [event for event in raw_events if frozenset(note.note_name for note in event.notes) == dominant_set]
    dominant_notes_by_name: dict[str, NoteCandidate] = {}
    for event in dominant_events:
        for note in event.notes:
            dominant_notes_by_name.setdefault(note.note_name, note)
    if len(dominant_notes_by_name) != 4:
        return raw_events

    dominant_notes = sorted((dominant_notes_by_name[name] for name in dominant_set), key=lambda note: note.frequency)
    dominant_keys = sorted(note.key for note in dominant_notes)
    if dominant_keys[-1] - dominant_keys[0] + 1 != len(dominant_keys):
        return raw_events

    family_subset_events = [event for event in raw_events if frozenset(note.note_name for note in event.notes) < dominant_set]
    singleton_family_events = [event for event in family_subset_events if len(event.notes) == 1]
    off_family_noise_events = [
        event
        for event in raw_events
        if frozenset(note.note_name for note in event.notes) and not frozenset(note.note_name for note in event.notes) <= dominant_set and (event.end_time - event.start_time) <= 0.18
    ]
    if len(family_subset_events) < 2 or not singleton_family_events or not off_family_noise_events:
        return raw_events

    normalized: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        duration = event.end_time - event.start_time
        previous_event = raw_events[index - 1] if index > 0 else None
        next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
        previous_gap = event.start_time - previous_event.end_time if previous_event is not None else 1.0
        next_gap = next_event.start_time - event.end_time if next_event is not None else 1.0
        nearby_dominant = any(
            0 <= offset < len(raw_events) and frozenset(note.note_name for note in raw_events[offset].notes) == dominant_set
            for offset in (index - 3, index - 2, index - 1, index + 1, index + 2, index + 3)
        )

        if event_set == dominant_set:
            normalized.append(event)
            continue

        if event_set and event_set < dominant_set and nearby_dominant and duration <= 1.4:
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=dominant_notes,
                    is_gliss_like=True,
                    primary_note_name=event.primary_note_name if event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            continue

        if (
            event_set
            and not event_set <= dominant_set
            and len(event.notes) <= 2
            and duration <= 0.18
            and ((previous_event is not None and frozenset(note.note_name for note in previous_event.notes) <= dominant_set and previous_gap <= 0.12)
                 or (next_event is not None and frozenset(note.note_name for note in next_event.notes) <= dominant_set and next_gap <= 0.12))
        ):
            continue

        normalized.append(event)

    return normalized
def normalize_repeated_explicit_four_note_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 5:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        if len(note_set) == 4:
            note_set_counts[note_set] = note_set_counts.get(note_set, 0) + 1

    if not note_set_counts:
        return raw_events

    dominant_set, dominant_count = max(note_set_counts.items(), key=lambda item: item[1])
    subset_support_events = [
        event
        for event in raw_events
        if 1 < len(frozenset(note.note_name for note in event.notes)) < 4
        and frozenset(note.note_name for note in event.notes) < dominant_set
        and not event.is_gliss_like
    ]
    has_mergeable_strict_pair = any(
        1 < len(frozenset(note.note_name for note in raw_events[index].notes)) < 4
        and 1 < len(frozenset(note.note_name for note in raw_events[index + 1].notes)) < 4
        and not raw_events[index].is_gliss_like
        and not raw_events[index + 1].is_gliss_like
        and frozenset(note.note_name for note in raw_events[index].notes) < dominant_set
        and frozenset(note.note_name for note in raw_events[index + 1].notes) < dominant_set
        and (frozenset(note.note_name for note in raw_events[index].notes) | frozenset(note.note_name for note in raw_events[index + 1].notes)) == dominant_set
        and raw_events[index + 1].start_time - raw_events[index].end_time <= GLISS_CLUSTER_MAX_GAP
        for index in range(len(raw_events) - 1)
    )
    if dominant_count < 4 and not (
        dominant_count >= 2
        and dominant_count + len(subset_support_events) >= 5
        and has_mergeable_strict_pair
    ):
        return raw_events

    dominant_events = [event for event in raw_events if frozenset(note.note_name for note in event.notes) == dominant_set]
    dominant_score = float(np.median([event.primary_score for event in dominant_events])) if dominant_events else 0.0
    dominant_notes_by_name: dict[str, NoteCandidate] = {}
    for event in dominant_events:
        for note in event.notes:
            dominant_notes_by_name.setdefault(note.note_name, note)
    if len(dominant_notes_by_name) != 4:
        return raw_events
    dominant_notes = sorted((dominant_notes_by_name[name] for name in dominant_set), key=lambda note: note.frequency)

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        duration = event.end_time - event.start_time

        if event_set == dominant_set:
            normalized.append(event)
            index += 1
            continue

        previous_set = frozenset(note.note_name for note in raw_events[index - 1].notes) if index > 0 else frozenset()
        previous_gap = event.start_time - raw_events[index - 1].end_time if index > 0 else 1.0
        next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
        next_set = frozenset(note.note_name for note in next_event.notes) if next_event is not None else frozenset()
        next_gap = next_event.start_time - event.end_time if next_event is not None else 1.0
        nearby_dominant = any(
            0 <= offset < len(raw_events) and frozenset(note.note_name for note in raw_events[offset].notes) == dominant_set
            for offset in (index - 3, index - 2, index - 1, index + 1, index + 2, index + 3)
        )
        future_dominant = any(
            0 <= offset < len(raw_events) and frozenset(note.note_name for note in raw_events[offset].notes) == dominant_set
            for offset in (index + 1, index + 2, index + 3)
        )
        shared_note_count = len(event_set & dominant_set)

        if (
            next_event is not None
            and event_set
            and next_set
            and event_set < dominant_set
            and next_set < dominant_set
            and len(event_set) >= 2
            and len(next_set) >= 2
            and (event_set | next_set) == dominant_set
            and next_gap <= GLISS_CLUSTER_MAX_GAP
            and max(duration, next_event.end_time - next_event.start_time) <= 1.25
            and nearby_dominant
            and not event.is_gliss_like
            and not next_event.is_gliss_like
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

        if next_event is not None and next_set == dominant_set and next_gap <= GLISS_CLUSTER_MAX_GAP:
            if len(event_set) >= 2 and event_set < dominant_set and shared_note_count >= 2 and duration <= 1.25:
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=event.is_gliss_like or next_event.is_gliss_like,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

            if (
                len(event_set) == 1
                and duration <= 0.12
                and nearby_dominant
                and event.primary_score <= dominant_score * 0.35
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=event.is_gliss_like or next_event.is_gliss_like,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

        if (
            len(event_set) == 2
            and event_set < dominant_set
            and shared_note_count == 2
            and duration <= 1.25
            and nearby_dominant
            and future_dominant
            and not event.is_gliss_like
        ):
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=dominant_notes,
                    is_gliss_like=False,
                    primary_note_name=event.primary_note_name if event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            index += 1
            continue

        if (
            len(event_set) == 3
            and shared_note_count >= 2
            and duration <= 1.0
            and nearby_dominant
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

        if len(event_set) <= 2 and event_set < dominant_set and nearby_dominant and duration <= 0.45:
            if previous_set == dominant_set and previous_gap <= 0.02:
                index += 1
                continue
            if index == len(raw_events) - 1 and previous_set == dominant_set and previous_gap <= 0.35:
                index += 1
                continue
            if (
                event.primary_score <= dominant_score * 0.8
                and (
                    (previous_set == dominant_set and previous_gap <= 0.35)
                    or (next_set == dominant_set and next_gap <= 0.35)
                    or (index > 0 and previous_gap <= 0.02)
                )
            ):
                index += 1
                continue

        normalized.append(event)
        index += 1

    return normalized

def normalize_strict_four_note_subsets(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        if len(note_set) == 4:
            note_set_counts[note_set] = note_set_counts.get(note_set, 0) + 1
    if not note_set_counts:
        return raw_events

    dominant_set, dominant_count = max(note_set_counts.items(), key=lambda item: item[1])
    if dominant_count < 4:
        return raw_events

    dominant_notes_by_name: dict[str, NoteCandidate] = {}
    for event in raw_events:
        if frozenset(note.note_name for note in event.notes) != dominant_set:
            continue
        for note in event.notes:
            dominant_notes_by_name.setdefault(note.note_name, note)
    if len(dominant_notes_by_name) != 4:
        return raw_events
    dominant_notes = sorted((dominant_notes_by_name[name] for name in dominant_set), key=lambda note: note.frequency)

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            next_set = frozenset(note.note_name for note in next_event.notes)
            next_gap = next_event.start_time - event.end_time
            if (
                len(event_set) == 2
                and event_set < dominant_set
                and next_set == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=event.is_gliss_like or next_event.is_gliss_like,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
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
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        if len(note_set) == 3:
            note_set_counts[note_set] = note_set_counts.get(note_set, 0) + 1

    if not note_set_counts:
        return raw_events

    dominant_set, dominant_count = max(note_set_counts.items(), key=lambda item: item[1])
    if dominant_count < 3:
        return raw_events

    dominant_events = [event for event in raw_events if frozenset(note.note_name for note in event.notes) == dominant_set]
    competing_four_note_family = any(
        other_set != dominant_set and len(other_set & dominant_set) >= 2 and len(other_set | dominant_set) == 4
        for other_set in note_set_counts
    )
    if dominant_count < 4 and competing_four_note_family:
        return raw_events
    dominant_score = float(np.median([event.primary_score for event in dominant_events])) if dominant_events else 0.0
    dominant_notes_by_name: dict[str, NoteCandidate] = {}
    for event in dominant_events:
        for note in event.notes:
            dominant_notes_by_name.setdefault(note.note_name, note)
    dominant_notes = sorted((dominant_notes_by_name[name] for name in dominant_set), key=lambda note: note.frequency)

    normalized: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        if event_set == dominant_set:
            normalized.append(event)
            continue

        previous_set = frozenset(note.note_name for note in raw_events[index - 1].notes) if index > 0 else frozenset()
        next_set = frozenset(note.note_name for note in raw_events[index + 1].notes) if index + 1 < len(raw_events) else frozenset()
        previous_gap = event.start_time - raw_events[index - 1].end_time if index > 0 else 1.0
        next_gap = raw_events[index + 1].start_time - event.end_time if index + 1 < len(raw_events) else 1.0
        nearby_dominant = any(
            0 <= offset < len(raw_events) and frozenset(note.note_name for note in raw_events[offset].notes) == dominant_set
            for offset in (index - 3, index - 2, index - 1, index + 1, index + 2, index + 3)
        )
        between_dominant = (
            index > 0
            and index + 1 < len(raw_events)
            and previous_set == dominant_set
            and next_set == dominant_set
            and previous_gap <= 0.18
            and next_gap <= 0.18
        )
        duration = event.end_time - event.start_time
        shared_note_count = len(event_set & dominant_set)

        if (
            len(event_set) <= 2
            and duration <= 0.32
            and event.primary_score <= dominant_score * 0.28
            and (
                (shared_note_count >= 1 and (
                    (previous_set == dominant_set and previous_gap <= CHORD_CLUSTER_MAX_GAP)
                    or (next_set == dominant_set and next_gap <= CHORD_CLUSTER_MAX_GAP)
                    or between_dominant
                ))
                or (len(event_set) == 1 and nearby_dominant and event.primary_score <= dominant_score * 0.25)
                or (len(event_set) == 2 and shared_note_count >= 1 and nearby_dominant and duration <= 0.2 and event.primary_score <= dominant_score * 0.12)
            )
        ):
            continue

        if event_set < dominant_set and nearby_dominant:
            if len(event_set) == 2 or (len(event_set) == 1 and next(iter(event_set), None) in dominant_set):
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
                continue

        if (
            len(event_set) == 3
            and len(event_set & dominant_set) == 2
            and note_set_counts.get(event_set, 0) <= 1
            and nearby_dominant
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
            continue

        normalized.append(event)

    return normalized


def suppress_repeated_triad_blips(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        if len(note_set) == 3:
            note_set_counts[note_set] = note_set_counts.get(note_set, 0) + 1
    if not note_set_counts:
        return raw_events

    dominant_set, dominant_count = max(note_set_counts.items(), key=lambda item: item[1])
    if dominant_count < 4:
        return raw_events
    dominant_events = [event for event in raw_events if frozenset(note.note_name for note in event.notes) == dominant_set]
    dominant_score = float(np.median([event.primary_score for event in dominant_events])) if dominant_events else 0.0

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        if event_set == dominant_set:
            previous_set = frozenset(note.note_name for note in raw_events[index - 1].notes) if index > 0 else frozenset()
            next_set = frozenset(note.note_name for note in raw_events[index + 1].notes) if index + 1 < len(raw_events) else frozenset()
            duration = event.end_time - event.start_time
            if (
                duration <= 0.35
                and event.primary_score <= dominant_score * 0.6
                and previous_set == dominant_set
                and next_set == dominant_set
            ):
                continue
        cleaned.append(event)
    return cleaned


def suppress_isolated_triad_extensions(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    for event in raw_events:
        note_set = frozenset(note.note_name for note in event.notes)
        note_set_counts[note_set] = note_set_counts.get(note_set, 0) + 1

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        event_note_set = frozenset(note.note_name for note in event.notes)
        if len(event.notes) == 3 and note_set_counts.get(event_note_set, 0) == 1:
            candidate_subsets = [
                frozenset(subset)
                for subset in (
                    event_note_set - {note.note_name}
                    for note in event.notes
                )
                if len(subset) == 2 and note_set_counts.get(frozenset(subset), 0) >= 3
            ]
            if candidate_subsets:
                nearby_sets = [
                    frozenset(note.note_name for note in raw_events[offset].notes)
                    for offset in range(max(0, index - 2), min(len(raw_events), index + 3))
                    if offset != index
                ]
                matching_subsets = [subset for subset in candidate_subsets if subset in nearby_sets]
                if matching_subsets:
                    target_subset = max(matching_subsets, key=lambda subset: note_set_counts[subset])
                    target_notes = [note for note in event.notes if note.note_name in target_subset]
                    cleaned.append(
                        RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=sorted(target_notes, key=lambda note: note.frequency),
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=event.primary_note_name if event.primary_note_name in target_subset else target_notes[0].note_name,
                            primary_score=event.primary_score,
                        )
                    )
                    continue
        cleaned.append(event)

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
        RepeatedPatternPass("normalize_repeated_four_note_family", normalize_repeated_four_note_family, merge_after=True),
        RepeatedPatternPass("normalize_repeated_four_note_gliss_patterns", normalize_repeated_four_note_gliss_patterns, merge_after=True),
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

    recent_note_names: set[str] = set()
    for recent_event in raw_events[-4:]:
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

    for start_time, end_time in segments:
        duration = max(end_time - start_time, 0.08)
        recent_note_names = build_recent_note_names(raw_events)
        ascending_primary_run_ceiling = build_recent_ascending_primary_run_ceiling(raw_events)
        ascending_singleton_suffix_ceiling, ascending_singleton_suffix_note_names = build_recent_ascending_singleton_suffix(raw_events)
        previous_primary_note_name = raw_events[-1].primary_note_name if raw_events else None
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
            previous_primary_note_name=previous_primary_note_name,
            previous_primary_was_singleton=previous_primary_was_singleton,
        )
        if not candidates or primary is None:
            continue

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
    processed_events = suppress_resonant_carryover(processed_events)
    processed_events = collapse_same_start_primary_singletons(processed_events)
    processed_events = simplify_short_secondary_bleed(processed_events)
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
    merged_events = merge_adjacent_events(processed_events)
    merged_events = merge_short_chord_clusters(merged_events)
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



