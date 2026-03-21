from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from .models import InstrumentTuning, NotationViews, ScoreEvent, ScoreNote, TranscriptionResult
from .tunings import build_custom_tuning, parse_note_name


FRAME_LENGTH = 2048
HOP_LENGTH = 256
ATTACK_ANALYSIS_SECONDS = 0.16
ATTACK_ANALYSIS_RATIO = 0.35
HARMONIC_WEIGHTS = [1.0, 0.55, 0.3, 0.15]
HARMONIC_BAND_CENTS = 40.0
SUPPRESSION_BAND_CENTS = 45.0
MAX_POLYPHONY = 2
MAX_HARMONIC_MULTIPLE = 4
SECONDARY_SCORE_RATIO = 0.12
SECONDARY_MIN_FUNDAMENTAL_RATIO = 0.18
OCTAVE_ALIAS_RATIO_THRESHOLD = 1.15
OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO = 0.34
OCTAVE_ALIAS_PENALTY = 0.85

OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO = 0.32
OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO = 0.16


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


@dataclass
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
    harmonics: list[dict[str, float]]
    subharmonics: list[dict[str, float]]


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
    return abs(1200.0 * np.log2(freq_a / freq_b))


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


def detect_segments(audio: np.ndarray, sample_rate: int) -> tuple[list[tuple[float, float]], float, dict[str, Any]]:
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=HOP_LENGTH)
    threshold = max(float(np.max(rms)) * 0.18, float(np.median(rms)) * 2.2, 0.01)
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

    segments: list[tuple[float, float]] = []
    for range_start, range_end in active_ranges:
        effective_range_start = range_start
        prior_onsets = [time for time in onset_times if range_start - 0.55 <= time <= range_start + 0.005]
        if prior_onsets:
            effective_range_start = prior_onsets[-1]

        range_onsets = [time for time in onset_times if effective_range_start + 0.005 < time < range_end - 0.05]
        supplemental_starts = [
            start
            for start, _ in raw_active_ranges
            if effective_range_start + 0.05 < start < range_end - 0.05
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

    if not segments:
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        segments = [(0.0, duration)]

    tempo_array, _ = librosa.beat.beat_track(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)
    tempo = float(np.asarray(tempo_array).reshape(-1)[0]) if np.asarray(tempo_array).size else 90.0
    if tempo <= 1.0:
        tempo = 90.0

    debug_info = {
        "onsetTimes": onset_times,
        "activeRanges": [[round(start, 4), round(end, 4)] for start, end in active_ranges],
        "segments": [[round(start, 4), round(end, 4)] for start, end in segments],
        "rmsThreshold": round(threshold, 6),
        "tempoRaw": round(tempo, 4),
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


def rank_tuning_candidates(frequencies: np.ndarray, spectrum: np.ndarray, tuning: InstrumentTuning) -> list[NoteHypothesis]:
    hypotheses: list[NoteHypothesis] = []

    for note in tuning.notes:
        pitch_class, octave = parse_note_name(note.note_name)
        candidate = NoteCandidate(
            key=note.key,
            note_name=note.note_name,
            frequency=note.frequency,
            pitch_class=pitch_class,
            octave=octave,
        )

        harmonic_energies = [peak_energy_near(frequencies, spectrum, note.frequency * harmonic_index) for harmonic_index in range(1, MAX_HARMONIC_MULTIPLE + 1)]
        subharmonic_frequencies = [note.frequency / 2.0, note.frequency / 3.0]
        subharmonic_energies = [peak_energy_near(frequencies, spectrum, sub_freq) if sub_freq >= 40 else 0.0 for sub_freq in subharmonic_frequencies]

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
                harmonics=harmonics,
                subharmonics=subharmonics,
            )
        )

    return sorted(hypotheses, key=lambda item: item.score, reverse=True)


def are_harmonic_related(note_a: NoteCandidate, note_b: NoteCandidate) -> bool:
    high = max(note_a.frequency, note_b.frequency)
    low = min(note_a.frequency, note_b.frequency)
    ratio = high / low if low else 0.0
    return any(abs(1200.0 * np.log2(ratio / multiple)) <= 30 for multiple in (2, 3, 4))


def harmonic_relation_multiple(note_a: NoteCandidate, note_b: NoteCandidate) -> float | None:
    high = max(note_a.frequency, note_b.frequency)
    low = min(note_a.frequency, note_b.frequency)
    ratio = high / low if low else 0.0
    for multiple in (2.0, 3.0, 4.0):
        if abs(1200.0 * np.log2(ratio / multiple)) <= 30:
            return multiple
    return None


def allow_octave_secondary(primary: NoteHypothesis, hypothesis: NoteHypothesis, selected: list[NoteCandidate]) -> bool:
    for existing in selected:
        relation = harmonic_relation_multiple(hypothesis.candidate, existing)
        if relation is None:
            continue
        if relation != 2.0:
            return False
        if hypothesis.fundamental_ratio < OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO:
            return False
        if hypothesis.fundamental_energy < primary.fundamental_energy * OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO:
            return False
        return True
    return False


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
            "harmonics": hypothesis.harmonics,
            "subharmonics": hypothesis.subharmonics,
        }
        for hypothesis in ranked[:limit]
    ]


def segment_peaks(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    tuning: InstrumentTuning,
    *,
    debug: bool = False,
) -> tuple[list[NoteCandidate], dict[str, Any] | None]:
    start = int(start_time * sample_rate)
    end = int(end_time * sample_rate)
    segment = audio[start:end]
    if len(segment) < 512:
        return [], None

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

    ranked = rank_tuning_candidates(frequencies, spectrum, tuning)
    if not ranked or ranked[0].score <= 1e-6:
        return [], None

    primary = ranked[0]
    selected = [primary.candidate]
    residual_ranked: list[NoteHypothesis] = []
    secondary_decision_trail: list[dict[str, Any]] = []
    secondary_score_ratio = SECONDARY_SCORE_RATIO
    secondary_min_fundamental_ratio = SECONDARY_MIN_FUNDAMENTAL_RATIO

    if MAX_POLYPHONY > 1:
        residual_spectrum = suppress_harmonics(spectrum, frequencies, primary.candidate.frequency)
        residual_ranked = rank_tuning_candidates(frequencies, residual_spectrum, tuning)
        for hypothesis in residual_ranked[:8]:
            reasons: list[str] = []
            if hypothesis.candidate.note_name == primary.candidate.note_name:
                reasons.append("same-as-primary")
            if hypothesis.score < primary.score * secondary_score_ratio:
                reasons.append("score-below-threshold")
            if hypothesis.fundamental_ratio < secondary_min_fundamental_ratio:
                reasons.append("fundamental-ratio-too-low")
            if any(are_harmonic_related(hypothesis.candidate, existing) for existing in selected) and not allow_octave_secondary(primary, hypothesis, selected):
                reasons.append("harmonic-related-to-selected")
            accepted = len(reasons) == 0
            secondary_decision_trail.append(
                {
                    "noteName": hypothesis.candidate.note_name,
                    "score": round(hypothesis.score, 6),
                    "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
                    "accepted": accepted,
                    "reasons": reasons,
                    "octaveDyadAllowed": allow_octave_secondary(primary, hypothesis, selected),
                }
            )
            if accepted:
                selected.append(hypothesis.candidate)
                break

    debug_payload = None
    if debug:
        debug_payload = {
            "startTime": round(start_time, 4),
            "endTime": round(end_time, 4),
            "durationSec": round(end_time - start_time, 4),
            "selectedNotes": [candidate.note_name for candidate in selected],
            "primaryCandidate": build_debug_candidates([primary], limit=1)[0],
            "rankedCandidates": build_debug_candidates(ranked),
            "residualCandidates": build_debug_candidates(residual_ranked),
            "secondaryDecisionTrail": secondary_decision_trail,
            "rawPeaks": build_raw_peaks(frequencies, spectrum, tuning),
        }

    return sorted(selected, key=lambda item: item.frequency), debug_payload


def suppress_resonant_carryover(raw_events: list[RawEvent]) -> list[RawEvent]:
    if not raw_events:
        return []

    cleaned = [raw_events[0]]
    for event in raw_events[1:]:
        recent_notes = {note.note_name for note in cleaned[-1].notes}
        if len(cleaned) > 1:
            recent_notes.update(note.note_name for note in cleaned[-2].notes)
        updated_event = event
        if len(event.notes) == 2:
            repeated_notes = [note for note in event.notes if note.note_name in recent_notes]
            fresh_notes = [note for note in event.notes if note.note_name not in recent_notes]
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and repeated_notes[0].frequency < fresh_notes[0].frequency:
                updated_event = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=fresh_notes,
                    is_gliss_like=event.is_gliss_like,
                )
        cleaned.append(updated_event)

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
            )
            continue
        merged.append(event)

    return merged


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


def build_notation_views(events: list[ScoreEvent]) -> NotationViews:
    western = [" | ".join(note.pitch_class + str(note.octave) for note in event.notes) for event in events]
    numbered = [" ".join(note.label_number for note in event.notes) for event in events]
    vertical = [[note.label_doremi for note in event.notes] for event in events]
    return NotationViews(western=western, numbered=numbered, verticalDoReMi=vertical)


async def transcribe_audio(upload: UploadFile, tuning: InstrumentTuning, *, debug: bool = False) -> TranscriptionResult:
    audio, sample_rate = await read_audio(upload)
    normalized = normalize_audio(audio)
    segments, tempo, segment_debug = detect_segments(normalized, sample_rate)

    raw_events: list[RawEvent] = []
    segment_candidates_debug: list[dict[str, Any]] = []

    for start_time, end_time in segments:
        duration = max(end_time - start_time, 0.08)
        candidates, candidate_debug = segment_peaks(
            normalized,
            sample_rate,
            start_time,
            end_time,
            tuning,
            debug=debug,
        )
        if not candidates:
            continue

        raw_events.append(
            RawEvent(
                start_time=start_time,
                end_time=end_time,
                notes=candidates,
                is_gliss_like=duration < 0.18,
            )
        )
        if candidate_debug:
            segment_candidates_debug.append(candidate_debug)

    merged_events = merge_adjacent_events(suppress_resonant_carryover(raw_events))
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
                }
                for event in merged_events
            ],
        }

    return TranscriptionResult(
        instrumentTuning=tuning,
        tempo=round(tempo, 2),
        events=events,
        notationViews=build_notation_views(events),
        warnings=warnings,
        debug=result_debug,
    )

