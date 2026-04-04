from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from ..models import InstrumentTuning
from ..tunings import parse_note_name
from . import settings
from .audio import cents_distance, snap_frequency_to_tuning
from .constants import *
from .models import NoteCandidate, NoteHypothesis
from .profiles import (
    _build_analysis_window_chunks,
    _broadband_chunk_energy,
    _chunk_spectrum,
    _positive_spectral_flux,
)


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


def onset_backward_attack_gain(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    target_frequency: float,
    lookback_seconds: float = TERTIARY_BACKWARD_LOOKBACK_SECONDS,
) -> float:
    """Compute the ratio of onset energy to energy at a past reference point.

    Traces the note's own frequency component backward in time.
    Genuine attacks show high ratio (note absent in the past).
    Residual notes show low ratio (note was already present and decaying).
    """
    window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
    start_sample = max(int(start_time * sample_rate), 0)
    lookback_sample = int((start_time - lookback_seconds) * sample_rate)

    # Not enough audio history for a reliable lookback (e.g. first 200ms of
    # file or evaluation window).  Return 0 so the tertiary gate rejects.
    if lookback_sample < 0 or lookback_sample + window_samples > start_sample:
        return 0.0

    early_chunk = audio[start_sample:start_sample + window_samples]
    past_chunk = audio[lookback_sample:lookback_sample + window_samples]
    if len(past_chunk) < 512 or len(early_chunk) < 512:
        return 0.0

    def _energy(chunk: np.ndarray) -> float:
        n_fft = max(4096, 1 << int(np.ceil(np.log2(len(chunk)))))
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
        frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        return peak_energy_near(frequencies, spectrum, target_frequency)

    past_energy = _energy(past_chunk)
    onset_energy = _energy(early_chunk)
    return (onset_energy + 1e-6) / (past_energy + 1e-6)




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

    # If the primary shows a genuine mute-dip re-attack, keep it even though
    # the broadband onset_gain is low — the per-note frequency band confirms
    # the tine was touched and replucked.
    if _has_mute_dip_reattack(audio, sample_rate, start_time, primary.candidate.frequency):
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


def _try_gap_fill(
    test_keys: list[int],
    selected: list[NoteCandidate],
    hypothesis: "NoteHypothesis",
    residual_ranked: list["NoteHypothesis"],
    primary: "NoteHypothesis",
    audio: "np.ndarray",
    sample_rate: int,
    start_time: float,
    end_time: float,
) -> list[NoteCandidate] | None:
    """Find gap-filling candidates that make a chord physically playable.

    When adding *hypothesis* to *selected* creates a physically impossible
    chord, look for candidates whose keys fill the gaps between existing
    keys.  Returns the list of gap-fill candidates to insert, or None.
    """
    sorted_keys = sorted(set(test_keys))
    if len(sorted_keys) + 1 > MAX_POLYPHONY:
        return None  # no room for gap-fill

    selected_names = {n.note_name for n in selected} | {hypothesis.candidate.note_name}
    gap_keys: set[int] = set()
    for i in range(len(sorted_keys) - 1):
        lo, hi = sorted_keys[i], sorted_keys[i + 1]
        if hi - lo == 2:
            gap_keys.add(lo + 1)

    if not gap_keys:
        return None

    gap_candidates: list[NoteCandidate] = []
    for gk in gap_keys:
        for h in residual_ranked:
            if h.candidate.key == gk and h.candidate.note_name not in selected_names:
                if (
                    h.score >= primary.score * TERTIARY_MIN_SCORE_RATIO
                    and h.score >= TERTIARY_MIN_SCORE
                    and h.fundamental_ratio >= TERTIARY_MIN_FUNDAMENTAL_RATIO
                    and onset_energy_gain(audio, sample_rate, start_time, end_time, h.candidate.frequency) >= TERTIARY_MIN_ONSET_GAIN
                ):
                    gap_candidates.append(h.candidate)
                break

    if len(gap_candidates) != len(gap_keys):
        return None

    filled_keys = test_keys + [c.key for c in gap_candidates]
    if not is_physically_playable_chord(filled_keys):
        return None
    if len(set(filled_keys)) > MAX_POLYPHONY:
        return None

    return gap_candidates


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


class SegmentPeaksResult(NamedTuple):
    """Return type for segment_peaks. Supports tuple unpacking."""
    candidates: list[NoteCandidate]
    debug: dict[str, Any] | None
    primary: NoteHypothesis | None
    trace: SegmentDecisionTrace | None = None


@dataclass(slots=True)
class _SegmentContext:
    """Immutable context shared by all segment_peaks phases."""
    audio: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    tuning: InstrumentTuning
    debug: bool
    recent_note_names: set[str] | None
    ascending_primary_run_ceiling: float | None
    ascending_singleton_suffix_ceiling: float | None
    ascending_singleton_suffix_note_names: set[str] | None
    descending_primary_suffix_floor: float | None
    descending_primary_suffix_ceiling: float | None
    descending_primary_suffix_note_names: set[str] | None
    previous_primary_note_name: str | None
    previous_primary_frequency: float | None
    previous_primary_was_singleton: bool

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass(slots=True)
class _SpectralData:
    """FFT results from signal acquisition."""
    frequencies: np.ndarray
    spectrum: np.ndarray
    ranked: list[NoteHypothesis]


@dataclass(slots=True)
class _PrimaryDecision:
    """Record of primary note resolution decisions."""
    initial_primary: str
    final_primary: str
    onset_gain: float | None
    promotions: list[str]
    rejected: bool
    rejection_reason: str | None


@dataclass(slots=True)
class _CandidateDecision:
    """Record of a single candidate evaluation."""
    note_name: str
    frequency: float
    score: float
    fundamental_ratio: float
    onset_gain: float | None
    accepted: bool
    reasons: list[str]
    octave_dyad_allowed: bool
    source: str

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "noteName": self.note_name,
            "score": round(self.score, 6),
            "fundamentalRatio": round(self.fundamental_ratio, 6),
            "onsetGain": None if self.onset_gain is None else round(self.onset_gain, 6),
            "accepted": self.accepted,
            "reasons": self.reasons,
            "octaveDyadAllowed": self.octave_dyad_allowed,
        }

    @property
    def reason_categories(self) -> set[str]:
        """Categories of gates that produced the rejection reasons."""
        return {GATE_CATEGORIES.get(r, "unknown") for r in self.reasons} if not self.accepted else set()


@dataclass(slots=True)
class _CandidateVerdict:
    """Phase A evaluation result for a single candidate (selected-independent)."""
    hypothesis: NoteHypothesis
    phase_a_reasons: list[str]
    onset_gain: float | None
    octave_dyad_allowed: bool
    phase_b_reasons: list[str] = field(default_factory=list)
    accepted: bool = False


# Gate classification for 5-layer architecture planning.
# "structural": no audio needed, always run — stay in selection layer
# "evidence": uses onset/backward/mute-dip from cache — candidate for final-decision layer
# "context": depends on previous event state — candidate for final-decision layer
# "iterative": iterative harmonic suppression pass
# "extension": pattern extension phases
# "promotion": secondary-to-upper-octave promotion
GATE_CATEGORIES: dict[str, str] = {
    # Structural (Layer 3: selection, stay here)
    "tertiary-physically-impossible": "structural",
    "tertiary-score-below-threshold": "structural",
    "tertiary-fundamental-ratio-too-low": "structural",
    "tertiary-duplicate-note": "structural",
    "same-as-primary": "structural",
    "recent-upper-octave-alias-secondary-blocked": "structural",
    "score-below-threshold": "structural",
    "fundamental-ratio-too-low": "structural",
    "harmonic-related-to-selected": "structural",
    # Evidence (candidate for Layer 4: final decision)
    "tertiary-weak-onset": "evidence",
    "tertiary-weak-backward-attack": "evidence",
    "recent-carryover-candidate": "evidence",
    "weak-upper-secondary": "evidence",
    "weak-secondary-onset": "evidence",
    "weak-lower-secondary": "evidence",
    # Context (candidate for Layer 4: final decision)
    "descending-adjacent-upper-carryover": "context",
    "descending-restart-upper-carryover": "context",
    "descending-primary-suffix-upper-carryover": "context",
    "descending-repeated-primary-stale-upper": "context",
    "ascending-singleton-carryover": "context",
    "repeated-primary-carryover": "context",
    # Iterative suppression
    "iterative-tertiary-physically-impossible": "iterative",
    "iterative-tertiary-score-below-threshold": "iterative",
    "iterative-tertiary-fundamental-ratio-too-low": "iterative",
    "iterative-tertiary-weak-onset": "iterative",
    "iterative-tertiary-weak-backward-attack": "iterative",
    "iterative-suppression-tertiary": "iterative",
    # Extension
    "contiguous-four-note-cluster": "extension",
    "gap-fill-for-physical-playability": "extension",
    "contiguous-tertiary-extension": "extension",
    "non-slide-playable-contiguous-cluster": "extension",
    "descending-repeated-primary-tertiary-blocked": "extension",
    "lower-mixed-roll-extension": "extension",
    "lower-roll-tail-extension": "extension",
    # Rescue (Layer 3.5: final decision)
    "evidence-rescue-weak-secondary-onset": "rescue",
    "evidence-rescue-weak-lower-secondary": "rescue",
    "evidence-rescue-recent-carryover": "rescue",
}


@dataclass(slots=True)
class SegmentDecisionTrace:
    """Complete record of all decisions made for a segment."""
    primary: _PrimaryDecision
    candidates: list[_CandidateDecision]


@dataclass(slots=True)
class _PrimaryResult:
    """Output of the primary promotion gauntlet."""
    primary: NoteHypothesis
    primary_onset_gain: float | None
    promotion_debug: dict[str, Any] | None
    decision: _PrimaryDecision


@dataclass(slots=True)
class _SelectionState:
    """Mutable accumulator for secondary selection and extension phases."""
    selected: list[NoteCandidate]
    residual_ranked: list[NoteHypothesis]
    candidate_decisions: list[_CandidateDecision]
    promoted_secondary_to_recent_upper_octave: bool


class _NoteEvidenceCache:
    """Lazy, cached per-note evidence for a single segment.

    All phases share one instance so that evidence computed during
    primary resolution is reused in secondary selection and extensions.
    Internally calls module-level functions (monkeypatch-compatible).
    """

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
    ) -> None:
        self._audio = audio
        self._sr = sample_rate
        self._start = start_time
        self._end = end_time
        self._onset_gains: dict[float, float] = {}
        self._backward_gains: dict[float, float] = {}
        self._mute_dip: dict[float, bool] = {}
        self._residual_decay: dict[float, bool] = {}

    def onset_gain(self, frequency: float) -> float:
        if frequency not in self._onset_gains:
            self._onset_gains[frequency] = onset_energy_gain(
                self._audio, self._sr, self._start, self._end, frequency,
            )
        return self._onset_gains[frequency]

    def backward_attack_gain(self, frequency: float) -> float:
        if frequency not in self._backward_gains:
            self._backward_gains[frequency] = onset_backward_attack_gain(
                self._audio, self._sr, self._start, frequency,
            )
        return self._backward_gains[frequency]

    def has_mute_dip_reattack(self, frequency: float) -> bool:
        if frequency not in self._mute_dip:
            self._mute_dip[frequency] = _has_mute_dip_reattack(
                self._audio, self._sr, self._start, frequency,
            )
        return self._mute_dip[frequency]

    def is_residual_decay(self, frequency: float) -> bool:
        if frequency not in self._residual_decay:
            self._residual_decay[frequency] = _is_residual_decay(
                self._audio, self._sr, self._start, frequency,
            )
        return self._residual_decay[frequency]

    def get_onset_gain_if_cached(self, frequency: float) -> float | None:
        return self._onset_gains.get(frequency)


def _acquire_spectrum(
    ctx: _SegmentContext,
) -> tuple[_SpectralData, _NoteEvidenceCache] | None:
    """Layer 1: Signal acquisition — FFT and initial candidate ranking.

    Returns None when the segment is too short or has no meaningful signal.
    """
    start = int(ctx.start_time * ctx.sample_rate)
    end = int(ctx.end_time * ctx.sample_rate)
    segment = ctx.audio[start:end]
    if len(segment) < 512:
        return None

    analysis_samples = len(segment)
    if len(segment) > int(ctx.sample_rate * 0.1):
        analysis_samples = min(
            len(segment),
            max(int(ctx.sample_rate * ATTACK_ANALYSIS_SECONDS), int(len(segment) * ATTACK_ANALYSIS_RATIO)),
        )
    analysis_segment = segment[:analysis_samples]

    n_fft = max(4096, 1 << int(np.ceil(np.log2(len(analysis_segment)))))
    window = np.hanning(len(analysis_segment))
    spectrum = np.abs(np.fft.rfft(analysis_segment * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / ctx.sample_rate)

    ranked = rank_tuning_candidates(frequencies, spectrum, ctx.tuning, debug=ctx.debug)
    if not ranked or ranked[0].score <= 1e-6:
        return None

    spectral = _SpectralData(frequencies=frequencies, spectrum=spectrum, ranked=ranked)
    evidence = _NoteEvidenceCache(ctx.audio, ctx.sample_rate, ctx.start_time, ctx.end_time)
    return spectral, evidence


def _resolve_primary(
    ctx: _SegmentContext,
    spectral: _SpectralData,
    evidence: _NoteEvidenceCache,
) -> _PrimaryResult:
    """Layer 2: Primary promotion gauntlet + residual decay check.

    Always returns a _PrimaryResult. Check result.decision.rejected for rejections.
    """
    ranked = spectral.ranked
    initial_primary_name = ranked[0].candidate.note_name
    promotions: list[str] = []
    primary = ranked[0]
    primary, primary_onset_gain, primary_promotion_debug = maybe_replace_stale_recent_primary(
        ctx.audio,
        ctx.sample_rate,
        ctx.start_time,
        ctx.end_time,
        primary,
        ranked,
        ctx.recent_note_names,
        previous_primary_note_name=ctx.previous_primary_note_name,
        previous_primary_frequency=ctx.previous_primary_frequency,
        previous_primary_was_singleton=ctx.previous_primary_was_singleton,
    )
    if primary_promotion_debug is not None:
        promotions.append(primary_promotion_debug.get("reason", "stale-recent-primary"))
    primary, stale_upper_promotion_debug = maybe_promote_stale_primary_to_upper_octave(
        primary,
        ranked,
        ctx.duration,
        primary_onset_gain,
        ctx.recent_note_names,
    )
    if stale_upper_promotion_debug is not None:
        primary_promotion_debug = stale_upper_promotion_debug
        promotions.append(stale_upper_promotion_debug.get("reason", "stale-upper-octave"))
    primary, recent_upper_alias_promotion_debug = maybe_promote_recent_upper_octave_alias_primary(
        primary,
        ranked,
        ctx.duration,
        ctx.recent_note_names,
    )
    if recent_upper_alias_promotion_debug is not None:
        primary_promotion_debug = recent_upper_alias_promotion_debug
        promotions.append(recent_upper_alias_promotion_debug.get("reason", "recent-upper-octave-alias"))
    # Rejection is deferred: record reason but continue so secondary
    # evaluation always runs.  _apply_final_decisions handles the final call.
    _rejected = False
    _rejection_reason: str | None = None
    if (
        primary.score < PRIMARY_REJECTION_MAX_SCORE
        and primary.fundamental_ratio < PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO
    ):
        _rejected = True
        _rejection_reason = "primary-score-too-low"
    elif (
        # Suppress residual-decay segments: if the primary is a recent note
        # and shows no mute-dip (smooth decay rather than finger-touch-then-
        # repluck), the segment is likely resonance from a previous event.
        ctx.recent_note_names
        and primary.candidate.note_name in ctx.recent_note_names
        and evidence.is_residual_decay(primary.candidate.frequency)
        and not evidence.has_mute_dip_reattack(primary.candidate.frequency)
    ):
        # Forward-scan: check all recent notes for genuine re-attack (mute-dip).
        _ranked_by_name: dict[str, NoteHypothesis] = {}
        for h in ranked:
            if h.candidate.note_name not in _ranked_by_name:
                _ranked_by_name[h.candidate.note_name] = h
        alternative_primary = None
        for note_name in ctx.recent_note_names:
            hyp = _ranked_by_name.get(note_name)
            if hyp is None:
                continue
            if evidence.has_mute_dip_reattack(hyp.candidate.frequency):
                if alternative_primary is None or hyp.score > alternative_primary.score:
                    alternative_primary = hyp
        if alternative_primary is None:
            # Octave-up rescue: check 1 and 2 octaves above.
            for octave_mult in (2, 4):
                target_freq = primary.candidate.frequency * octave_mult
                for h in ranked:
                    if abs(h.candidate.frequency - target_freq) / target_freq < 0.03:
                        octave_gain = _note_onset_energy_gain(ctx.audio, ctx.sample_rate, ctx.start_time, h.candidate.frequency)
                        if octave_gain is not None and octave_gain >= RESIDUAL_DECAY_MIN_ONSET_GAIN:
                            alternative_primary = h
                        break
                if alternative_primary is not None:
                    break
        if alternative_primary is None:
            _rejected = True
            _rejection_reason = "residual-decay-no-reattack"
        else:
            primary = alternative_primary
            primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
            primary_promotion_debug = {
                "reason": "residual-forward-scan",
                "replacedPrimaryNote": ranked[0].candidate.note_name,
                "replacementNote": primary.candidate.note_name,
            }
            promotions.append("residual-forward-scan")
    decision = _PrimaryDecision(
        initial_primary=initial_primary_name, final_primary=primary.candidate.note_name,
        onset_gain=primary_onset_gain, promotions=promotions,
        rejected=_rejected, rejection_reason=_rejection_reason,
    )
    return _PrimaryResult(primary, primary_onset_gain, primary_promotion_debug, decision)


def _extend_gliss_tertiary(
    ctx: _SegmentContext,
    primary: NoteHypothesis,
    state: _SelectionState,
    evidence: _NoteEvidenceCache,
) -> None:
    """Layer 4a: Contiguous gliss tertiary extension."""
    if (
        len(state.selected) != 2
        or state.promoted_secondary_to_recent_upper_octave
        or ctx.duration > GLISS_TERTIARY_MAX_DURATION
    ):
        return
    selected_keys = sorted(note.key for note in state.selected)
    if selected_keys[-1] - selected_keys[0] != 1:
        return
    extension_keys = {selected_keys[0] - 1, selected_keys[-1] + 1}
    selected_names = {note.note_name for note in state.selected}
    viable_extensions: list[tuple[NoteHypothesis, float]] = []
    for hypothesis in state.residual_ranked[:6]:
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
        if any(are_harmonic_related(candidate, existing) for existing in state.selected):
            continue
        og = evidence.onset_gain(candidate.frequency)
        bg = evidence.backward_attack_gain(candidate.frequency)
        if bg < TERTIARY_MIN_BACKWARD_ATTACK_GAIN:
            continue
        viable_extensions.append((hypothesis, og))

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

    if chosen_extension is None:
        return
    hypothesis, og = chosen_extension
    extended_cluster = [*state.selected, hypothesis.candidate]
    extension_blocked = should_block_descending_repeated_primary_tertiary_extension(
        selected=state.selected,
        extension=hypothesis.candidate,
        segment_duration=ctx.duration,
        previous_primary_was_singleton=ctx.previous_primary_was_singleton,
        descending_primary_suffix_floor=ctx.descending_primary_suffix_floor,
        descending_primary_suffix_ceiling=ctx.descending_primary_suffix_ceiling,
        descending_primary_suffix_note_names=ctx.descending_primary_suffix_note_names,
    )
    if extension_blocked:
        state.candidate_decisions.append(_CandidateDecision(
            note_name=hypothesis.candidate.note_name,
            frequency=hypothesis.candidate.frequency,
            score=hypothesis.score,
            fundamental_ratio=hypothesis.fundamental_ratio,
            onset_gain=og,
            accepted=False,
            reasons=["descending-repeated-primary-tertiary-blocked"],
            octave_dyad_allowed=False,
            source="extension-gliss",
        ))
    elif is_slide_playable_contiguous_cluster(extended_cluster, ctx.tuning):
        hypothesis.candidate.onset_gain = og
        state.selected.append(hypothesis.candidate)
        state.candidate_decisions.append(_CandidateDecision(
            note_name=hypothesis.candidate.note_name,
            frequency=hypothesis.candidate.frequency,
            score=hypothesis.score,
            fundamental_ratio=hypothesis.fundamental_ratio,
            onset_gain=og,
            accepted=True,
            reasons=["contiguous-tertiary-extension"],
            octave_dyad_allowed=False,
            source="extension-gliss",
        ))
    else:
        state.candidate_decisions.append(_CandidateDecision(
            note_name=hypothesis.candidate.note_name,
            frequency=hypothesis.candidate.frequency,
            score=hypothesis.score,
            fundamental_ratio=hypothesis.fundamental_ratio,
            onset_gain=og,
            accepted=False,
            reasons=["non-slide-playable-contiguous-cluster"],
            octave_dyad_allowed=False,
            source="extension-gliss",
        ))


def _extend_lower_mixed_roll(
    ctx: _SegmentContext,
    primary: NoteHypothesis,
    state: _SelectionState,
    evidence: _NoteEvidenceCache,
) -> None:
    """Layer 4b: Lower mixed-roll extension."""
    if (
        len(state.selected) != 2
        or ctx.duration > LOWER_MIXED_ROLL_EXTENSION_MAX_DURATION
        or primary.candidate.frequency != min(note.frequency for note in state.selected)
        or primary.candidate.octave > 4
    ):
        return
    upper_note = max(state.selected, key=lambda note: note.frequency)
    selected_names = {note.note_name for note in state.selected}
    if upper_note.key - primary.candidate.key < 3:
        return
    selected_secondary_scores = [
        secondary.score
        for secondary in state.residual_ranked[:4]
        if secondary.candidate.note_name in selected_names
    ]
    extension_candidate: tuple[NoteHypothesis, float] | None = None
    for hypothesis in state.residual_ranked[:8]:
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
        if any(are_harmonic_related(candidate, existing) for existing in state.selected):
            continue
        if selected_secondary_scores and hypothesis.score < max(selected_secondary_scores) * LOWER_MIXED_ROLL_EXTENSION_MIN_UPPER_SCORE_RATIO:
            continue
        pog = evidence.onset_gain(primary.candidate.frequency)
        if pog < LOWER_MIXED_ROLL_EXTENSION_MIN_PRIMARY_ONSET_GAIN:
            continue
        og = evidence.onset_gain(candidate.frequency)
        if og < LOWER_MIXED_ROLL_EXTENSION_MIN_EXTENSION_ONSET_GAIN:
            continue
        extension_candidate = (hypothesis, og)
        break

    if extension_candidate is not None:
        hypothesis, og = extension_candidate
        hypothesis.candidate.onset_gain = og
        state.selected.append(hypothesis.candidate)
        state.candidate_decisions.append(_CandidateDecision(
            note_name=hypothesis.candidate.note_name,
            frequency=hypothesis.candidate.frequency,
            score=hypothesis.score,
            fundamental_ratio=hypothesis.fundamental_ratio,
            onset_gain=og,
            accepted=True,
            reasons=["lower-mixed-roll-extension"],
            octave_dyad_allowed=False,
            source="extension-mixed-roll",
        ))


def _extend_lower_roll_tail(
    ctx: _SegmentContext,
    primary: NoteHypothesis,
    state: _SelectionState,
    evidence: _NoteEvidenceCache,
) -> None:
    """Layer 4c: Lower roll-tail extension."""
    if (
        len(state.selected) != 3
        or ctx.duration > LOWER_ROLL_TAIL_EXTENSION_MAX_DURATION
        or primary.candidate.frequency != max(note.frequency for note in state.selected)
    ):
        return
    selected_keys = sorted(note.key for note in state.selected)
    if selected_keys[-1] - selected_keys[0] != 2:
        return
    selected_names = {note.note_name for note in state.selected}
    lowest_selected = min(state.selected, key=lambda note: note.frequency)
    extension_key = selected_keys[0] - 1
    extension_candidate: tuple[NoteHypothesis, float] | None = None
    for hypothesis in state.residual_ranked[:8]:
        candidate = hypothesis.candidate
        if candidate.note_name in selected_names:
            continue
        if candidate.key != extension_key:
            continue
        if hypothesis.fundamental_ratio < LOWER_ROLL_TAIL_EXTENSION_MIN_FUNDAMENTAL_RATIO:
            continue
        if hypothesis.score < GLISS_TERTIARY_MIN_SCORE:
            continue
        if any(are_harmonic_related(candidate, existing) for existing in state.selected):
            continue
        lowest_selected_score = next((item.score for item in state.residual_ranked[:8] if item.candidate.note_name == lowest_selected.note_name), None)
        if lowest_selected_score is not None and hypothesis.score < lowest_selected_score * LOWER_ROLL_TAIL_EXTENSION_MIN_SCORE_RATIO:
            continue
        pog = evidence.onset_gain(primary.candidate.frequency)
        if pog < LOWER_ROLL_TAIL_EXTENSION_MIN_PRIMARY_ONSET_GAIN:
            continue
        og = evidence.onset_gain(candidate.frequency)
        if og > LOWER_ROLL_TAIL_EXTENSION_MAX_ONSET_GAIN:
            continue
        extension_candidate = (hypothesis, og)
        break

    if extension_candidate is not None:
        hypothesis, og = extension_candidate
        hypothesis.candidate.onset_gain = og
        state.selected.append(hypothesis.candidate)
        state.candidate_decisions.append(_CandidateDecision(
            note_name=hypothesis.candidate.note_name,
            frequency=hypothesis.candidate.frequency,
            score=hypothesis.score,
            fundamental_ratio=hypothesis.fundamental_ratio,
            onset_gain=og,
            accepted=True,
            reasons=["lower-roll-tail-extension"],
            octave_dyad_allowed=False,
            source="extension-roll-tail",
        ))



def _select_candidates(
    ctx: _SegmentContext,
    spectral: _SpectralData,
    primary_result: _PrimaryResult,
    evidence: _NoteEvidenceCache,
) -> _SelectionState:
    """Layer 3: Candidate selection — 4-note cluster, Phase A/B, iterative suppression."""
    primary = primary_result.primary
    primary_onset_gain = primary_result.primary_onset_gain
    primary_promotion_debug = primary_result.promotion_debug
    selected = [primary.candidate]
    residual_ranked: list[NoteHypothesis] = []
    promoted_secondary_to_recent_upper_octave = False
    candidate_decisions: list[_CandidateDecision] = []
    contiguous_four_note_cluster = select_contiguous_four_note_cluster(primary, spectral.ranked, ctx.duration)
    if contiguous_four_note_cluster is not None:
        selected = contiguous_four_note_cluster
        for candidate in selected:
            if candidate.note_name == primary.candidate.note_name:
                continue
            matching = next(hypothesis for hypothesis in spectral.ranked[:4] if hypothesis.candidate.note_name == candidate.note_name)
            candidate_decisions.append(_CandidateDecision(
                note_name=matching.candidate.note_name,
                frequency=matching.candidate.frequency,
                score=matching.score,
                fundamental_ratio=matching.fundamental_ratio,
                onset_gain=None,
                accepted=True,
                reasons=["contiguous-four-note-cluster"],
                octave_dyad_allowed=False,
                source="cluster",
            ))
    secondary_score_ratio = SECONDARY_SCORE_RATIO
    if ctx.duration <= 0.14:
        secondary_score_ratio = SHORT_SEGMENT_SECONDARY_SCORE_RATIO
    secondary_min_fundamental_ratio = SECONDARY_MIN_FUNDAMENTAL_RATIO
    _disabled = settings.get().disabled_gates

    if MAX_POLYPHONY > 1 and contiguous_four_note_cluster is None:
        residual_spectrum = suppress_harmonics(spectral.spectrum, spectral.frequencies, primary.candidate.frequency)
        residual_ranked = rank_tuning_candidates(spectral.frequencies, residual_spectrum, ctx.tuning, debug=ctx.debug)
        # ══ Phase A: Independent candidate evaluation (selected-independent) ══
        verdicts: list[_CandidateVerdict] = []
        for hypothesis in residual_ranked[:8]:
            phase_a_reasons: list[str] = []
            onset_gain: float | None = None
            # octave_dyad_allowed approximated against primary only (Phase A)
            octave_dyad_allowed = allow_octave_secondary(primary, hypothesis, [primary.candidate])
            segment_duration = ctx.duration
            score_ratio = secondary_score_ratio
            if octave_dyad_allowed and hypothesis.candidate.frequency > primary.candidate.frequency:
                score_ratio = min(score_ratio, OCTAVE_DYAD_UPPER_SCORE_RATIO)
            # ── Structural gates (selected-independent) ──────────────
            if hypothesis.candidate.note_name == primary.candidate.note_name:
                if "same-as-primary" not in _disabled:
                    phase_a_reasons.append("same-as-primary")
            if (
                primary_promotion_debug is not None
                and primary_promotion_debug.get("reason") == "recent-upper-octave-alias-primary"
                and hypothesis.candidate.pitch_class == primary.candidate.pitch_class
                and hypothesis.candidate.octave == primary.candidate.octave - 1
            ):
                if "recent-upper-octave-alias-secondary-blocked" not in _disabled:
                    phase_a_reasons.append("recent-upper-octave-alias-secondary-blocked")
            if hypothesis.score < primary.score * score_ratio and not octave_dyad_allowed:
                if "score-below-threshold" not in _disabled:
                    phase_a_reasons.append("score-below-threshold")
            if hypothesis.fundamental_ratio < secondary_min_fundamental_ratio:
                if "fundamental-ratio-too-low" not in _disabled:
                    phase_a_reasons.append("fundamental-ratio-too-low")
            _structural_snapshot = list(phase_a_reasons)
            # ── Evidence + Context gates (selected-independent) ──────
            if ctx.recent_note_names and hypothesis.candidate.note_name in ctx.recent_note_names:
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if hypothesis.candidate.frequency < primary.candidate.frequency:
                    if (
                        onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                        or (
                            ctx.ascending_primary_run_ceiling is not None
                            and segment_duration <= ASCENDING_PRIMARY_RUN_MAX_DURATION
                            and primary.candidate.frequency >= ctx.ascending_primary_run_ceiling
                            and hypothesis.candidate.frequency < ctx.ascending_primary_run_ceiling
                            and onset_gain < ASCENDING_PRIMARY_RUN_RECENT_SECONDARY_ONSET_GAIN
                        )
                    ):
                        if not evidence.has_mute_dip_reattack(hypothesis.candidate.frequency):
                            if "recent-carryover-candidate" not in _disabled:
                                phase_a_reasons.append("recent-carryover-candidate")
                else:
                    primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                    if (
                        primary_onset_gain >= RECENT_UPPER_SECONDARY_PRIMARY_ONSET_GAIN
                        and segment_duration >= RECENT_UPPER_SECONDARY_MIN_DURATION
                        and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                        and not evidence.has_mute_dip_reattack(hypothesis.candidate.frequency)
                    ):
                        if "recent-carryover-candidate" not in _disabled:
                            phase_a_reasons.append("recent-carryover-candidate")
            # ── Context gates (previous event dependent) ──
            if (
                ctx.previous_primary_was_singleton
                and ctx.previous_primary_note_name == hypothesis.candidate.note_name
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and segment_duration <= DESCENDING_ADJACENT_UPPER_CARRYOVER_MAX_DURATION
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= cents_distance(primary.candidate.frequency, hypothesis.candidate.frequency)
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if (
                    primary_onset_gain >= DESCENDING_ADJACENT_UPPER_PRIMARY_ONSET_GAIN
                    and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_ADJACENT_UPPER_SCORE_RATIO
                ):
                    if "descending-adjacent-upper-carryover" not in _disabled:
                        phase_a_reasons.append("descending-adjacent-upper-carryover")
            if (
                ctx.previous_primary_was_singleton
                and ctx.previous_primary_frequency is not None
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and primary.candidate.frequency < ctx.previous_primary_frequency
                and hypothesis.candidate.frequency >= ctx.previous_primary_frequency
                and segment_duration <= DESCENDING_RESTART_UPPER_CARRYOVER_MAX_DURATION
                and not octave_dyad_allowed
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if (
                    primary_onset_gain >= DESCENDING_RESTART_UPPER_PRIMARY_ONSET_GAIN
                    and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_RESTART_UPPER_SCORE_RATIO
                ):
                    if "descending-restart-upper-carryover" not in _disabled:
                        phase_a_reasons.append("descending-restart-upper-carryover")
            if (
                hypothesis.candidate.frequency > primary.candidate.frequency
                and ctx.descending_primary_suffix_floor is not None
                and ctx.descending_primary_suffix_ceiling is not None
                and ctx.descending_primary_suffix_note_names
                and primary.candidate.frequency <= ctx.descending_primary_suffix_floor
                and segment_duration <= DESCENDING_PRIMARY_SUFFIX_MAX_DURATION
                and not octave_dyad_allowed
                and (
                    hypothesis.candidate.note_name in ctx.descending_primary_suffix_note_names
                    or hypothesis.candidate.frequency > ctx.descending_primary_suffix_ceiling
                )
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if (
                    primary_onset_gain >= DESCENDING_PRIMARY_SUFFIX_PRIMARY_ONSET_GAIN
                    and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_PRIMARY_SUFFIX_UPPER_SCORE_RATIO
                ):
                    if "descending-primary-suffix-upper-carryover" not in _disabled:
                        phase_a_reasons.append("descending-primary-suffix-upper-carryover")
            if (
                primary_promotion_debug is not None
                and primary_promotion_debug.get("reason") == "descending-repeated-primary"
                and hypothesis.candidate.frequency > primary.candidate.frequency
                and not octave_dyad_allowed
            ):
                replaced_primary_note = primary_promotion_debug.get("replacedPrimaryNote")
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if (
                    onset_gain < DESCENDING_REPEATED_PRIMARY_STALE_UPPER_MAX_ONSET_GAIN
                    and hypothesis.score < primary.score * DESCENDING_REPEATED_PRIMARY_STALE_UPPER_SCORE_RATIO
                    and (
                        hypothesis.candidate.note_name == replaced_primary_note
                        or (
                            ctx.descending_primary_suffix_ceiling is not None
                            and hypothesis.candidate.frequency > ctx.descending_primary_suffix_ceiling
                        )
                    )
                ):
                    if "descending-repeated-primary-stale-upper" not in _disabled:
                        phase_a_reasons.append("descending-repeated-primary-stale-upper")
            # ── Weak-onset evidence gates ─────────────────────────────
            if (
                hypothesis.candidate.frequency > primary.candidate.frequency
                and segment_duration >= UPPER_SECONDARY_WEAK_ONSET_MIN_DURATION
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                if primary_onset_gain >= RECENT_UPPER_SECONDARY_PRIMARY_ONSET_GAIN:
                    onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                    if (
                        onset_gain < UPPER_SECONDARY_WEAK_ONSET_MAX_GAIN
                        and hypothesis.score < primary.score * UPPER_SECONDARY_WEAK_ONSET_SCORE_RATIO
                    ):
                        if "weak-upper-secondary" not in _disabled:
                            phase_a_reasons.append("weak-upper-secondary")
            if (
                segment_duration <= SHORT_SECONDARY_WEAK_ONSET_MAX_DURATION
                and not octave_dyad_allowed
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                if primary_onset_gain >= SHORT_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN:
                    onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                    if (
                        onset_gain < SHORT_SECONDARY_WEAK_ONSET_MAX_GAIN
                        and onset_gain < primary_onset_gain * SHORT_SECONDARY_WEAK_ONSET_MAX_RATIO
                        and hypothesis.score < primary.score * SHORT_SECONDARY_WEAK_ONSET_SCORE_RATIO
                    ):
                        if "weak-secondary-onset" not in _disabled:
                            phase_a_reasons.append("weak-secondary-onset")
            if (
                segment_duration <= LOWER_SECONDARY_WEAK_ONSET_MAX_DURATION
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and not octave_dyad_allowed
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                if primary_onset_gain >= LOWER_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN:
                    onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                    if (
                        onset_gain < LOWER_SECONDARY_WEAK_ONSET_MAX_GAIN
                        and onset_gain < primary_onset_gain * LOWER_SECONDARY_WEAK_ONSET_MAX_RATIO
                        and hypothesis.score < primary.score * LOWER_SECONDARY_WEAK_ONSET_SCORE_RATIO
                    ):
                        if "weak-lower-secondary" not in _disabled:
                            phase_a_reasons.append("weak-lower-secondary")
            # ── Score rescue (clears structural rejection) ─────────────
            if (
                _structural_snapshot == ["score-below-threshold"]
                and segment_duration >= 0.75
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and hypothesis.fundamental_ratio >= 0.95
                and hypothesis.score >= primary.score * 0.1
                and not octave_dyad_allowed
            ):
                interval_cents = abs(cents_distance(primary.candidate.frequency, hypothesis.candidate.frequency))
                if 250.0 <= interval_cents <= 550.0:
                    if "score-below-threshold" in phase_a_reasons:
                        phase_a_reasons.remove("score-below-threshold")

            # ── Run-pattern context gates ──────────────────────────────
            if (
                ctx.ascending_primary_run_ceiling is not None
                and segment_duration <= ASCENDING_PRIMARY_RUN_MAX_DURATION
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and primary.candidate.frequency >= ctx.ascending_primary_run_ceiling
                and hypothesis.candidate.frequency < ctx.ascending_primary_run_ceiling
                and hypothesis.score < primary.score * ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO
                and hypothesis.candidate.note_name != primary.candidate.note_name
                and primary.candidate.note_name not in (ctx.recent_note_names or set())
                and not octave_dyad_allowed
            ):
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if onset_gain < MIN_RECENT_NOTE_ONSET_GAIN:
                    if "ascending-singleton-carryover" not in _disabled:
                        phase_a_reasons.append("ascending-singleton-carryover")
            if (
                segment_duration <= 0.22
                and ctx.previous_primary_was_singleton
                and ctx.previous_primary_note_name == primary.candidate.note_name
                and ctx.ascending_singleton_suffix_ceiling is not None
                and primary.candidate.frequency >= ctx.ascending_singleton_suffix_ceiling
                and hypothesis.candidate.frequency < primary.candidate.frequency
                and hypothesis.score < primary.score * ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO
                and hypothesis.candidate.note_name != primary.candidate.note_name
                and hypothesis.candidate.note_name not in (ctx.ascending_singleton_suffix_note_names or set())
                and not octave_dyad_allowed
            ):
                primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if primary_onset_gain < MIN_RECENT_NOTE_ONSET_GAIN and onset_gain < MIN_RECENT_NOTE_ONSET_GAIN:
                    if "repeated-primary-carryover" not in _disabled:
                        phase_a_reasons.append("repeated-primary-carryover")
            verdicts.append(_CandidateVerdict(
                hypothesis=hypothesis,
                phase_a_reasons=phase_a_reasons,
                onset_gain=onset_gain,
                octave_dyad_allowed=octave_dyad_allowed,
            ))

        # ══ Phase B: Greedy set construction (selected-dependent) ═════
        for verdict in verdicts:
            hypothesis = verdict.hypothesis
            if verdict.phase_a_reasons:
                # Phase A rejected — record decision and skip
                candidate_decisions.append(_CandidateDecision(
                    note_name=hypothesis.candidate.note_name,
                    frequency=hypothesis.candidate.frequency,
                    score=hypothesis.score,
                    fundamental_ratio=hypothesis.fundamental_ratio,
                    onset_gain=verdict.onset_gain,
                    accepted=False,
                    reasons=verdict.phase_a_reasons,
                    octave_dyad_allowed=verdict.octave_dyad_allowed,
                    source="secondary",
                ))
                continue
            phase_b_reasons: list[str] = []
            onset_gain = verdict.onset_gain
            # ── harmonic-related check against full selected set ──
            octave_dyad_allowed_full = allow_octave_secondary(primary, hypothesis, selected)
            if any(are_harmonic_related(hypothesis.candidate, existing) for existing in selected) and not octave_dyad_allowed_full:
                if "harmonic-related-to-selected" not in _disabled:
                    phase_b_reasons.append("harmonic-related-to-selected")
            # ── Tertiary gates (selected-dependent) ──────────────────
            is_tertiary_or_beyond = len(selected) >= 2
            if is_tertiary_or_beyond:
                test_keys = [n.key for n in selected] + [hypothesis.candidate.key]
                if not is_physically_playable_chord(test_keys):
                    gap_filled = _try_gap_fill(
                        test_keys, selected, hypothesis, residual_ranked,
                        primary, ctx.audio, ctx.sample_rate, ctx.start_time, ctx.end_time,
                    )
                    if gap_filled:
                        for gf_candidate in gap_filled:
                            selected.append(gf_candidate)
                            candidate_decisions.append(_CandidateDecision(
                                note_name=gf_candidate.note_name,
                                frequency=gf_candidate.frequency,
                                score=0.0,
                                fundamental_ratio=0.0,
                                onset_gain=None,
                                accepted=True,
                                reasons=["gap-fill-for-physical-playability"],
                                octave_dyad_allowed=False,
                                source="gap-fill",
                            ))
                    else:
                        phase_b_reasons.append("tertiary-physically-impossible")
                elif hypothesis.score < TERTIARY_MIN_SCORE:
                    phase_b_reasons.append("tertiary-score-below-threshold")
                elif any(hypothesis.candidate.note_name == existing.note_name for existing in selected):
                    phase_b_reasons.append("tertiary-duplicate-note")
                # ── Tertiary evidence gates ──
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if onset_gain < TERTIARY_MIN_ONSET_GAIN:
                    if "tertiary-weak-onset" not in _disabled:
                        phase_b_reasons.append("tertiary-weak-onset")
                backward_gain = evidence.backward_attack_gain(hypothesis.candidate.frequency)
                if backward_gain < TERTIARY_MIN_BACKWARD_ATTACK_GAIN:
                    if "tertiary-weak-backward-attack" not in _disabled:
                        phase_b_reasons.append("tertiary-weak-backward-attack")
            verdict.phase_b_reasons = phase_b_reasons
            accepted = len(phase_b_reasons) == 0
            verdict.accepted = accepted
            accepted_hypothesis = hypothesis
            debug_reasons: list[str] = phase_b_reasons
            segment_duration = ctx.duration
            if accepted and len(selected) == 1 and hypothesis.candidate.frequency < primary.candidate.frequency and not verdict.octave_dyad_allowed:
                accepted_hypothesis, promoted_from = maybe_promote_lower_secondary_to_recent_upper_octave(
                    primary,
                    hypothesis,
                    residual_ranked,
                    segment_duration,
                    ctx.recent_note_names,
                )
                if promoted_from is not None:
                    promoted_secondary_to_recent_upper_octave = True
                    debug_reasons = [f"promoted-from-{promoted_from}"]
            _hyp = accepted_hypothesis if accepted else hypothesis
            candidate_decisions.append(_CandidateDecision(
                note_name=_hyp.candidate.note_name,
                frequency=_hyp.candidate.frequency,
                score=_hyp.score,
                fundamental_ratio=_hyp.fundamental_ratio,
                onset_gain=onset_gain,
                accepted=accepted,
                reasons=debug_reasons,
                octave_dyad_allowed=verdict.octave_dyad_allowed,
                source="secondary",
            ))
            if accepted:
                accepted_hypothesis.candidate.onset_gain = evidence.get_onset_gain_if_cached(accepted_hypothesis.candidate.frequency)
                selected.append(accepted_hypothesis.candidate)
                if len(selected) >= MAX_POLYPHONY:
                    break

        # --- Iterative harmonic suppression: second pass for tertiary ---
        # After the first pass selects primary+secondary on the primary-only
        # residual, suppress ALL accepted notes' harmonics and re-rank.
        # Candidates that the first pass rejected (harmonic-related, inflated
        # fundamental-ratio) get a fresh evaluation on the cleaner residual.
        # Each round accepts at most one candidate, then re-suppresses and
        # re-ranks before the next round (true iterative suppression).
        while (
            settings.get().use_iterative_harmonic_suppression
            and len(selected) >= 2
            and len(selected) < MAX_POLYPHONY
        ):
            _iter_residual = spectral.spectrum
            for _sel_note in selected:
                _iter_residual = suppress_harmonics(_iter_residual, spectral.frequencies, _sel_note.frequency)
            _iter_ranked = rank_tuning_candidates(spectral.frequencies, _iter_residual, ctx.tuning, debug=ctx.debug)
            _already_selected = {n.note_name for n in selected}
            _iter_round_accepted = False
            for _iter_hyp in _iter_ranked[:8]:
                if _iter_hyp.candidate.note_name in _already_selected:
                    continue
                # Only consider octave-related candidates (the specific
                # pattern where harmonic-related binary check over-rejects).
                _iter_is_octave = any(
                    harmonic_relation_multiple(_iter_hyp.candidate, existing) == 2.0
                    for existing in selected
                )
                if not _iter_is_octave:
                    continue
                _iter_reasons: list[str] = []
                _iter_onset_gain: float | None = None
                _test_keys = [n.key for n in selected] + [_iter_hyp.candidate.key]
                if not is_physically_playable_chord(_test_keys):
                    _iter_reasons.append("iterative-tertiary-physically-impossible")
                elif (
                    _iter_hyp.score < primary.score * TERTIARY_MIN_SCORE_RATIO
                    or _iter_hyp.score < TERTIARY_MIN_SCORE
                ):
                    # Rescue: if the candidate has a genuine attack and
                    # retains high fundamentalRatio in the residual, the low
                    # score is explained by harmonic overlap, not missing energy.
                    _iter_onset_gain = evidence.onset_gain(_iter_hyp.candidate.frequency)
                    if not (
                        _iter_onset_gain >= TERTIARY_MIN_ONSET_GAIN
                        and _iter_hyp.fundamental_ratio >= ITERATIVE_RESCUE_MIN_FUNDAMENTAL_RATIO
                    ):
                        _iter_reasons.append("iterative-tertiary-score-below-threshold")
                else:
                    # All candidates reaching here are octave-related
                    # (non-octave filtered by _iter_is_octave above).
                    _iter_fr_threshold = ITERATIVE_TERTIARY_OCTAVE_MIN_FUNDAMENTAL_RATIO
                    if _iter_hyp.fundamental_ratio < _iter_fr_threshold:
                        _iter_reasons.append("iterative-tertiary-fundamental-ratio-too-low")
                if not _iter_reasons:
                    _iter_onset_gain = evidence.onset_gain(_iter_hyp.candidate.frequency)
                    if _iter_onset_gain < TERTIARY_MIN_ONSET_GAIN:
                        _iter_reasons.append("iterative-tertiary-weak-onset")
                if not _iter_reasons:
                    _iter_backward_gain = evidence.backward_attack_gain(_iter_hyp.candidate.frequency)
                    if _iter_backward_gain < TERTIARY_MIN_BACKWARD_ATTACK_GAIN:
                        _iter_reasons.append("iterative-tertiary-weak-backward-attack")
                _iter_accepted = len(_iter_reasons) == 0
                _iter_cached_og = evidence.get_onset_gain_if_cached(_iter_hyp.candidate.frequency)
                candidate_decisions.append(_CandidateDecision(
                    note_name=_iter_hyp.candidate.note_name,
                    frequency=_iter_hyp.candidate.frequency,
                    score=_iter_hyp.score,
                    fundamental_ratio=_iter_hyp.fundamental_ratio,
                    onset_gain=_iter_cached_og,
                    accepted=_iter_accepted,
                    reasons=_iter_reasons if _iter_reasons else ["iterative-suppression-tertiary"],
                    octave_dyad_allowed=False,
                    source="iterative",
                ))
                if _iter_accepted:
                    _iter_hyp.candidate.onset_gain = _iter_cached_og
                    selected.append(_iter_hyp.candidate)
                    _iter_round_accepted = True
                    break  # re-suppress and re-rank in next while iteration
            if not _iter_round_accepted:
                break  # no more candidates to recover


    return _SelectionState(
        selected=selected,
        residual_ranked=residual_ranked,
        candidate_decisions=candidate_decisions,
        promoted_secondary_to_recent_upper_octave=promoted_secondary_to_recent_upper_octave,
    )


def _build_segment_debug(
    ctx: _SegmentContext,
    spectral: _SpectralData,
    primary_result: _PrimaryResult,
    selection: _SelectionState,
    evidence: _NoteEvidenceCache,
) -> dict[str, Any]:
    """Layer 5: Assemble the debug payload."""
    primary = primary_result.primary
    attack_context = prepare_attack_debug_context(ctx.audio, ctx.sample_rate, ctx.start_time, ctx.end_time)
    segment_attack_debug = attack_context["broadband"] if attack_context is not None else {}
    attack_profiles: dict[str, dict[str, Any]] = {}
    if attack_context is not None:
        for hypothesis in [primary, *spectral.ranked[:5], *selection.residual_ranked[:5]]:
            note_name = hypothesis.candidate.note_name
            if note_name not in attack_profiles:
                attack_profiles[note_name] = build_candidate_attack_debug(attack_context, hypothesis.candidate.frequency)
    primary_og = evidence.get_onset_gain_if_cached(primary.candidate.frequency)
    return {
        "startTime": round(ctx.start_time, 4),
        "endTime": round(ctx.end_time, 4),
        "durationSec": round(ctx.duration, 4),
        "selectedNotes": [candidate.note_name for candidate in selection.selected],
        **segment_attack_debug,
        "primaryCandidate": build_debug_candidates([primary], limit=1, attack_profiles=attack_profiles)[0],
        "primaryOnsetGain": None if primary_og is None else round(primary_og, 6),
        "primaryPromotion": primary_result.promotion_debug,
        "rankedCandidates": build_debug_candidates(spectral.ranked, attack_profiles=attack_profiles),
        "residualCandidates": build_debug_candidates(selection.residual_ranked, attack_profiles=attack_profiles),
        "secondaryDecisionTrail": [d.to_debug_dict() for d in selection.candidate_decisions],
        "rawPeaks": build_raw_peaks(spectral.frequencies, spectral.spectrum, ctx.tuning),
    }


def _apply_final_decisions(
    ctx: _SegmentContext,
    selection: _SelectionState,
    primary_result: _PrimaryResult,
    evidence: _NoteEvidenceCache,
) -> None:
    """Layer 3.5: Final decision — post-selection review.

    1. Deferred primary rejection → clear selected.
    2. Evidence gate rescue → re-admit candidates rejected only by
       evidence gates if they have strong spectral quality.
    """
    if primary_result.decision.rejected:
        selection.selected.clear()
        return

    if not settings.get().use_evidence_gate_rescue:
        return

    _disabled = settings.get().disabled_gates
    primary = primary_result.primary
    selected_names = {n.note_name for n in selection.selected}

    # Collect evidence-only rejected candidates (no structural rejection)
    rescue_pool: list[tuple[_CandidateDecision, NoteHypothesis]] = []
    for decision in selection.candidate_decisions:
        if decision.accepted or decision.source != "secondary":
            continue
        cats = decision.reason_categories
        if not cats or cats != {"evidence"}:
            continue
        # Find matching hypothesis from residual_ranked
        hyp = _find_hypothesis_by_frequency(selection.residual_ranked, decision.frequency)
        if hyp is None:
            continue
        rescue_pool.append((decision, hyp))

    # Sort by score descending (strongest candidates first)
    rescue_pool.sort(key=lambda pair: pair[0].score, reverse=True)

    for decision, hypothesis in rescue_pool:
        # Common score floor: catches octave-dyad candidates that
        # bypassed structural score-below-threshold.
        if hypothesis.score < primary.score * RESCUE_MIN_SCORE_RATIO:
            continue
        if len(selection.selected) >= MAX_POLYPHONY:
            break
        if hypothesis.candidate.note_name in selected_names:
            continue
        if any(are_harmonic_related(hypothesis.candidate, s) for s in selection.selected):
            continue
        test_keys = [n.key for n in selection.selected] + [hypothesis.candidate.key]
        if not is_physically_playable_chord(test_keys):
            continue

        gate_name = _evidence_rescue_gate(decision, hypothesis, primary, evidence)
        if gate_name is None:
            continue
        if gate_name in _disabled:
            continue

        hypothesis.candidate.onset_gain = evidence.get_onset_gain_if_cached(hypothesis.candidate.frequency)
        selection.selected.append(hypothesis.candidate)
        selected_names.add(hypothesis.candidate.note_name)
        selection.candidate_decisions.append(_CandidateDecision(
            note_name=hypothesis.candidate.note_name,
            frequency=hypothesis.candidate.frequency,
            score=hypothesis.score,
            fundamental_ratio=hypothesis.fundamental_ratio,
            onset_gain=decision.onset_gain,
            accepted=True,
            reasons=[gate_name],
            octave_dyad_allowed=decision.octave_dyad_allowed,
            source="rescue",
        ))


def _find_hypothesis_by_frequency(
    ranked: list[NoteHypothesis],
    frequency: float,
) -> NoteHypothesis | None:
    """Find a NoteHypothesis matching *frequency* in *ranked*."""
    for hyp in ranked:
        if abs(hyp.candidate.frequency - frequency) < 0.01:
            return hyp
    return None


def _evidence_rescue_gate(
    decision: _CandidateDecision,
    hypothesis: NoteHypothesis,
    primary: NoteHypothesis,
    evidence: _NoteEvidenceCache,
) -> str | None:
    """Return rescue gate name if the candidate qualifies, else None.

    Dispatches on which evidence gate originally rejected the candidate
    and applies gate-specific rescue thresholds.
    """
    reasons = set(decision.reasons)

    backward_gain = evidence.backward_attack_gain(hypothesis.candidate.frequency)
    if backward_gain < RESCUE_MIN_BACKWARD_GAIN:
        return None

    # weak-lower-secondary (check before weak-secondary-onset since both may appear)
    if "weak-lower-secondary" in reasons:
        if hypothesis.fundamental_ratio >= RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO:
            return "evidence-rescue-weak-lower-secondary"
        return None

    # weak-secondary-onset
    if "weak-secondary-onset" in reasons:
        if hypothesis.fundamental_ratio >= RESCUE_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO:
            return "evidence-rescue-weak-secondary-onset"
        return None

    # recent-carryover-candidate
    if "recent-carryover-candidate" in reasons:
        if hypothesis.score >= primary.score * RESCUE_CARRYOVER_MIN_SCORE_RATIO:
            return "evidence-rescue-recent-carryover"
        return None

    return None


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
) -> SegmentPeaksResult:
    ctx = _SegmentContext(
        audio=audio, sample_rate=sample_rate,
        start_time=start_time, end_time=end_time,
        tuning=tuning, debug=debug,
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

    # Layer 1: Signal acquisition
    acquired = _acquire_spectrum(ctx)
    if acquired is None:
        return SegmentPeaksResult([], None, None)
    spectral, evidence = acquired

    # Layer 2: Primary resolution (rejection deferred to Layer 3.5)
    primary_result = _resolve_primary(ctx, spectral, evidence)
    primary = primary_result.primary
    primary.candidate.onset_gain = primary_result.primary_onset_gain

    # Layer 3: Candidate selection (always runs, even if primary rejected)
    selection = _select_candidates(ctx, spectral, primary_result, evidence)

    # Layer 3.5: Final decision (handles deferred primary rejection)
    _apply_final_decisions(ctx, selection, primary_result, evidence)

    if not selection.selected:
        # Primary rejected + no rescue → empty segment
        trace = SegmentDecisionTrace(primary=primary_result.decision, candidates=selection.candidate_decisions)
        rejection_debug = None
        if ctx.debug and primary_result.decision.rejection_reason == "residual-decay-no-reattack":
            rejection_debug = {
                "startTime": round(ctx.start_time, 6),
                "endTime": round(ctx.end_time, 6),
                "durationSec": round(ctx.duration, 6),
                "selectedNotes": [],
                "primaryNote": primary.candidate.note_name,
                "droppedBy": primary_result.decision.rejection_reason,
            }
        return SegmentPeaksResult([], rejection_debug, None, trace)

    # Layer 4: Extension phases
    _extend_gliss_tertiary(ctx, primary, selection, evidence)
    _extend_lower_mixed_roll(ctx, primary, selection, evidence)
    _extend_lower_roll_tail(ctx, primary, selection, evidence)

    # Layer 5: Evidence freeze + trace assembly + debug
    for note in selection.selected:
        cached_og = evidence.get_onset_gain_if_cached(note.frequency)
        if cached_og is not None and note.onset_gain is None:
            note.onset_gain = cached_og

    trace = SegmentDecisionTrace(primary=primary_result.decision, candidates=selection.candidate_decisions)

    debug_payload = None
    if ctx.debug:
        debug_payload = _build_segment_debug(ctx, spectral, primary_result, selection, evidence)

    return SegmentPeaksResult(
        sorted(selection.selected, key=lambda item: item.frequency),
        debug_payload,
        primary,
        trace,
    )

def _note_band_energy(
    audio: np.ndarray,
    sample_rate: int,
    center_time: float,
    frequency: float,
    window_seconds: float = ONSET_ENERGY_WINDOW_SECONDS,
) -> float:
    """Compute peak energy near *frequency* in a short window centred on *center_time*."""
    window_samples = max(int(sample_rate * window_seconds), 512)
    center_sample = int(center_time * sample_rate)
    half = window_samples // 2
    start = max(center_sample - half, 0)
    end = min(start + window_samples, len(audio))
    chunk = audio[start:end]
    if len(chunk) < 256:
        return 0.0
    band_hz = frequency * (2 ** (HARMONIC_BAND_CENTS / 1200) - 2 ** (-HARMONIC_BAND_CENTS / 1200))
    min_n_fft = int(np.ceil(sample_rate / band_hz)) * 2 if band_hz > 0 else 4096
    n_fft = max(min_n_fft, len(chunk))
    n_fft = 1 << int(np.ceil(np.log2(n_fft)))
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return peak_energy_near(frequencies, spectrum, frequency)


def _find_note_attack_time(
    audio: np.ndarray,
    sample_rate: int,
    onset_time: float,
    frequency: float,
    max_lookahead_seconds: float = 0.08,
) -> float:
    """Find the attack time for a specific note frequency after the onset.

    Instead of using broadband amplitude, scans the note's frequency band
    energy to find the steepest increase.  This handles chords where each
    note attacks at a slightly different time.
    """
    hop_seconds = 0.005
    window_seconds = 0.02
    t = onset_time
    end_t = min(onset_time + max_lookahead_seconds, len(audio) / sample_rate - window_seconds)

    best_diff = 0.0
    best_time = onset_time
    prev_energy = _note_band_energy(audio, sample_rate, t, frequency,
                                    window_seconds=window_seconds)
    t += hop_seconds
    while t < end_t:
        energy = _note_band_energy(audio, sample_rate, t, frequency,
                                   window_seconds=window_seconds)
        diff = energy - prev_energy
        if diff > best_diff:
            best_diff = diff
            best_time = t
        prev_energy = energy
        t += hop_seconds

    return best_time


MUTE_DIP_ENERGY_WINDOW = 0.05
MUTE_DIP_REATTACK_MIN_PRE_ENERGY = 3.0
MUTE_DIP_REATTACK_MIN_POST_ENERGY = 3.0
MUTE_DIP_REATTACK_MAX_DIP_RATIO = 0.1
MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO = 0.9


def _has_mute_dip_reattack(
    audio: np.ndarray,
    sample_rate: int,
    onset_time: float,
    frequency: float,
) -> bool:
    """Check if a recently-played note shows a mute-dip re-attack pattern.

    On a kalimba, replucking a ringing tine requires touching it first,
    which causes a characteristic energy profile in the note's frequency band:
        high (ringing) → low (finger mute) → high (re-attack)

    Unlike ``_is_residual_decay`` (which compares pre vs post energy), this
    function explicitly looks for the energy *dip* near the onset time.
    """
    # Pre-onset: the note must have been ringing before the onset.
    pre_time = onset_time - 0.04
    if pre_time < 0:
        return False
    pre_energy = _note_band_energy(audio, sample_rate, pre_time, frequency,
                                   window_seconds=MUTE_DIP_ENERGY_WINDOW)
    if pre_energy < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
        return False  # Note wasn't ringing; can't be a re-attack.

    # Scan for minimum energy around the onset (the mute dip).
    # The dip sits between pre-onset and the new attack, typically ±30ms of
    # the detected onset.  Window (50ms) exceeds the physical dip (~40ms)
    # but fine-step scanning still detects deep genuine dips while smoothing
    # out narrow noise artifacts.
    min_energy = float("inf")
    scan_start = max(onset_time - 0.01, 0.0)
    scan_end = min(onset_time + 0.05, len(audio) / sample_rate - MUTE_DIP_ENERGY_WINDOW)
    t = scan_start
    while t < scan_end:
        energy = _note_band_energy(audio, sample_rate, t, frequency,
                                   window_seconds=MUTE_DIP_ENERGY_WINDOW)
        if energy < min_energy:
            min_energy = energy
        t += 0.005

    # Post-attack: find the per-note attack time and measure energy after.
    attack_time = _find_note_attack_time(audio, sample_rate, onset_time, frequency)
    post_time = attack_time + 0.02
    if post_time > len(audio) / sample_rate - MUTE_DIP_ENERGY_WINDOW:
        return False
    post_energy = _note_band_energy(audio, sample_rate, post_time, frequency,
                                    window_seconds=MUTE_DIP_ENERGY_WINDOW)
    if post_energy < MUTE_DIP_REATTACK_MIN_POST_ENERGY:
        return False  # No meaningful re-attack energy.

    dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6)
    if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
        return False

    # Recovery check: a genuine re-attack restores the note's energy to near
    # its pre-onset level.  Sympathetic interference from plucking a neighboring
    # tine can cause a brief energy dip, but the note continues to decay
    # afterwards (recovery < 1.0).
    recovery_ratio = post_energy / (pre_energy + 1e-6)
    return recovery_ratio >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO


def _note_onset_energy_gain(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    frequency: float,
) -> float | None:
    """Return post/pre note-band energy gain at *frequency*, or None.

    Returns ``None`` when post-attack energy is too low to make a reliable
    determination (noise floor).  A positive return means a meaningful
    signal is present; values below ``RESIDUAL_DECAY_MIN_ONSET_GAIN``
    indicate residual decay while higher values indicate a genuine attack.
    """
    attack_time = _find_note_attack_time(audio, sample_rate, start_time, frequency)

    pre_time = start_time - 0.03
    post_time = attack_time + 0.03

    if pre_time < 0 or post_time > len(audio) / sample_rate:
        return None

    pre_energy = _note_band_energy(audio, sample_rate, pre_time, frequency,
                                   window_seconds=0.04)
    post_energy = _note_band_energy(audio, sample_rate, post_time, frequency,
                                    window_seconds=0.04)

    if post_energy < 1.0:
        return None  # Below noise floor; cannot determine.

    return (post_energy + 1e-6) / (pre_energy + 1e-6)


def _is_residual_decay(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    frequency: float,
) -> bool:
    """Check whether a note at *frequency* is residual decay at this onset."""
    gain = _note_onset_energy_gain(audio, sample_rate, start_time, frequency)
    if gain is None:
        return False  # No meaningful energy; not a residual context.
    return gain < RESIDUAL_DECAY_MIN_ONSET_GAIN
