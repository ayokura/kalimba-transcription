from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from ..models import InstrumentTuning
from . import settings
from .audio import cents_distance, snap_frequency_to_tuning
from .constants import *
from .models import Note, NoteCandidate, NoteHypothesis, RawAlternateGrouping
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


def _adaptive_n_fft(sample_rate: int, min_frequency: float, chunk_len: int, *, min_bins: int = 2) -> int:
    """Compute FFT size ensuring >= *min_bins* bins in the +/-40 cents band.

    Without this, high sample rates (e.g. 96kHz) with a fixed n_fft=4096 produce
    bin spacings too coarse for low-register notes — the ±40-cent band may contain
    zero bins, causing energy functions to return 0.

    *min_bins=2* (default) is suitable for accurate energy measurement.
    *min_bins=1* is a conservative setting that only prevents zero-bin blind spots
    while preserving existing behavior at standard sample rates.
    """
    band_hz = min_frequency * (2 ** (HARMONIC_BAND_CENTS / 1200) - 2 ** (-HARMONIC_BAND_CENTS / 1200))
    min_n_fft = int(np.ceil(sample_rate / band_hz)) * min_bins if band_hz > 0 else 4096
    n_fft = max(min_n_fft, chunk_len)
    return 1 << int(np.ceil(np.log2(n_fft)))


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

def suppress_harmonics(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    base_frequency: float,
    *,
    partial_ratios: list[float] | None = None,
    tuning_fundamentals: np.ndarray | None = None,
) -> np.ndarray:
    """Suppress energy at harmonic/partial positions of *base_frequency*.

    When *partial_ratios* is provided (e.g. [1.0, 1.5, 2.0, 2.908, 3.672]),
    suppress at those positions *in addition to* integer multiples
    1..MAX_HARMONIC_MULTIPLE.  Integer harmonics always carry some energy on
    kalimba tines (alongside beam partials), so suppressing only the measured
    beam positions would leak C4's 4× at C6 into the residual and confuse
    downstream secondary selection.  Union of integer comb and beam partials
    is the safe choice.

    When *tuning_fundamentals* is provided, skip suppression at positions
    that coincide with another note's fundamental (within ±SUPPRESSION_BAND_CENTS).
    This prevents e.g. D5's 1.5× partial from suppressing A5's fundamental.
    """
    residual = spectrum.copy()
    valid = frequencies > 0
    positive_freqs = frequencies[valid]
    if partial_ratios is not None:
        # Union of per-tine partials and integer comb, deduped with ~0.005
        # tolerance so 1.0/2.0 etc. from the partial list don't get counted
        # twice alongside their integer equivalents.
        ratios: list[float] = list(partial_ratios)
        for m in range(1, MAX_HARMONIC_MULTIPLE + 1):
            integer_ratio = float(m)
            if not any(abs(r - integer_ratio) <= 0.005 for r in ratios):
                ratios.append(integer_ratio)
    else:
        ratios = [float(m) for m in range(1, MAX_HARMONIC_MULTIPLE + 1)]
    for ratio in ratios:
        center_freq = base_frequency * ratio
        if center_freq > frequencies[-1]:
            continue
        # Guard: skip suppression when a *non-integer* partial (beam vibration
        # ratio like 1.5×) overlaps another note's fundamental.  Integer
        # harmonics (2×, 3×, 4×) are left alone — the pipeline's octave dyad
        # rescue and secondary selection logic already handle those overlaps.
        if ratio != 1.0 and tuning_fundamentals is not None and abs(ratio - round(ratio)) > 0.05:
            other_funds = tuning_fundamentals[
                np.abs(tuning_fundamentals - base_frequency) >= 0.01
            ]
            if len(other_funds) > 0:
                dists = np.abs(1200.0 * np.log2(other_funds / center_freq))
                if np.min(dists) <= SUPPRESSION_BAND_CENTS:
                    continue
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
    n_notes = len(tuning.notes)

    # ── Build per-note partial targets ────────────────────────────
    # When a note has explicit partials, use those ratios + weights.
    # Otherwise fall back to the integer harmonic comb.
    # Gated behind a feature flag — see settings.use_per_tine_partial_scoring.
    has_any_partials = (
        settings.get().use_per_tine_partial_scoring
        and any(note.partials for note in tuning.notes)
    )

    if has_any_partials:
        # Collect all partial targets and their weights per note
        partial_targets: list[list[tuple[float, float]]] = []  # per note: [(freq, weight), ...]
        all_freqs_list: list[float] = []
        for note in tuning.notes:
            if note.partials:
                pts = [(note.frequency * p.ratio, p.weight) for p in note.partials]
            else:
                # Fallback to integer comb for notes without explicit partials
                pts = [(note.frequency * m, HARMONIC_WEIGHTS[m - 1])
                       for m in range(1, MAX_HARMONIC_MULTIPLE + 1)]
            partial_targets.append(pts)
            all_freqs_list.extend(f for f, _w in pts)

        # Subharmonic targets (shared across both paths)
        sub_half_targets = note_freqs / 2.0
        sub_third_targets = note_freqs / 3.0
        sub_half_targets[sub_half_targets < 40.0] = 0.0
        sub_third_targets[sub_third_targets < 40.0] = 0.0
        all_freqs_list.extend(sub_half_targets)
        all_freqs_list.extend(sub_third_targets)

        all_target_freqs = np.array(all_freqs_list)
        all_energies = batch_peak_energies(frequencies, spectrum, all_target_freqs)

        # Unpack energies
        offset = 0
        per_note_energies: list[list[tuple[float, float, float]]] = []  # [(energy, weight, freq), ...]
        for pts in partial_targets:
            note_e = []
            for i, (freq, weight) in enumerate(pts):
                note_e.append((float(all_energies[offset + i]), weight, freq))
            per_note_energies.append(note_e)
            offset += len(pts)
        sub_half_energies_arr = all_energies[offset:offset + n_notes]
        sub_third_energies_arr = all_energies[offset + n_notes:offset + 2 * n_notes]
    else:
        # Fast path: no partials defined, use original integer comb
        harmonic_targets = np.concatenate([note_freqs * m for m in range(1, MAX_HARMONIC_MULTIPLE + 1)])
        sub_half_targets = note_freqs / 2.0
        sub_third_targets = note_freqs / 3.0
        sub_half_targets[sub_half_targets < 40.0] = 0.0
        sub_third_targets[sub_third_targets < 40.0] = 0.0
        all_target_freqs = np.concatenate([harmonic_targets, sub_half_targets, sub_third_targets])
        all_energies = batch_peak_energies(frequencies, spectrum, all_target_freqs)

        harmonic_energy_matrix = all_energies[:n_notes * MAX_HARMONIC_MULTIPLE].reshape(MAX_HARMONIC_MULTIPLE, n_notes)
        sub_half_energies_arr = all_energies[n_notes * MAX_HARMONIC_MULTIPLE:n_notes * MAX_HARMONIC_MULTIPLE + n_notes]
        sub_third_energies_arr = all_energies[n_notes * MAX_HARMONIC_MULTIPLE + n_notes:]

        per_note_energies = []
        for note_index, note in enumerate(tuning.notes):
            note_e = [
                (float(harmonic_energy_matrix[h, note_index]), HARMONIC_WEIGHTS[h], note.frequency * (h + 1))
                for h in range(MAX_HARMONIC_MULTIPLE)
            ]
            per_note_energies.append(note_e)

    # ── Score each candidate ──────────────────────────────────────
    hypotheses: list[NoteHypothesis] = []

    for note_index, note in enumerate(tuning.notes):
        candidate = NoteCandidate(
            key=note.key,
            note=Note.from_name(note.note_name),
        )

        energies_weights = per_note_energies[note_index]
        subharmonic_energies = [float(sub_half_energies_arr[note_index]), float(sub_third_energies_arr[note_index])]

        fundamental_energy = energies_weights[0][0]  # First partial is always fundamental
        overtone_energy = sum(w * e for e, w, _f in energies_weights[1:])
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

        # second_harmonic_energy: find the partial closest to 2.0× for backward compat
        second_harmonic_energy = 0.0
        for e, _w, f in energies_weights[1:]:
            ratio = f / note.frequency
            if abs(ratio - 2.0) < 0.1:
                second_harmonic_energy = e
                break

        harmonics = None
        subharmonics = None
        if debug:
            harmonics = [
                {
                    "multiple": round(f / note.frequency, 3),
                    "frequency": round(f, 3),
                    "energy": round(e, 6),
                    "weight": w,
                }
                for e, w, f in energies_weights
            ]
            subharmonics = [
                {
                    "multiple": 1.0 / float(index + 2),
                    "frequency": round(note.frequency / (index + 2), 3),
                    "energy": round(subharmonic_energies[index], 6),
                }
                for index in range(len(subharmonic_energies))
            ]

        candidate.score = score
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
                second_harmonic_energy=second_harmonic_energy,
                harmonics=harmonics,
                subharmonics=subharmonics,
            )
        )

    return sorted(hypotheses, key=lambda item: item.score, reverse=True)

def _get_partial_ratios(tuning: InstrumentTuning, frequency: float) -> list[float] | None:
    """Look up partial ratios for a note by frequency from the tuning definition."""
    for note in tuning.notes:
        if abs(note.frequency - frequency) < 0.01 and note.partials:
            return [p.ratio for p in note.partials]
    return None


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

def _aligned_band_energy(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    target_frequency: float,
) -> float:
    """Compute peak band energy for *target_frequency* in [start_time, end_time].

    Unlike :func:`_note_band_energy`, this uses an explicit time range (not
    centered on a point), so callers can construct strictly non-overlapping
    pre/post windows around an onset.
    """
    start_sample = max(int(start_time * sample_rate), 0)
    end_sample = min(int(end_time * sample_rate), len(audio))
    chunk = audio[start_sample:end_sample]
    if len(chunk) < 256:
        return 0.0
    n_fft = _adaptive_n_fft(sample_rate, target_frequency, len(chunk), min_bins=1)
    spectrum = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return peak_energy_near(frequencies, spectrum, target_frequency)


def pick_matching_sub_onset(
    audio: np.ndarray,
    sample_rate: int,
    target_frequency: float,
    sub_onsets: tuple[float, ...],
    *,
    measurement_window: float = SUB_ONSET_ANCHOR_MEASUREMENT_WINDOW,
    min_energy_ratio: float = SUB_ONSET_ANCHOR_MIN_RATIO,
    min_post_energy: float = SUB_ONSET_ANCHOR_MIN_POST_ENERGY,
) -> float | None:
    """Pick the sub-onset where *target_frequency* has the strongest attack.

    For each sub-onset, measures per-note energy in a window strictly before
    (``[onset - measurement_window, onset)``) and strictly after
    (``[onset, onset + measurement_window]``), then returns the sub-onset
    whose post/pre ratio is largest.

    Returns ``None`` when no sub-onset shows a meaningful energy rise — either
    the post energy is below ``min_post_energy`` (FFT leakage from a different
    note's attack) or the post/pre ratio is below ``min_energy_ratio`` (the
    note was already ringing or did not actually attack at any sub-onset).

    Used by :func:`onset_energy_gain` to anchor its attack window at the
    actual attack time of *target_frequency* in slide-chord-like segments
    where notes attack staggered.
    """
    if not sub_onsets:
        return None

    audio_duration = len(audio) / sample_rate
    best_post_energy = 0.0
    best_onset: float | None = None

    for onset in sub_onsets:
        pre_start = onset - measurement_window
        post_end = onset + measurement_window
        if pre_start < 0 or post_end > audio_duration:
            continue
        pre_energy = _aligned_band_energy(
            audio, sample_rate, pre_start, onset, target_frequency,
        )
        post_energy = _aligned_band_energy(
            audio, sample_rate, onset, post_end, target_frequency,
        )
        # Reject sub-onsets where the post window has negligible energy:
        # the note isn't actually attacking here, any post/pre ratio is
        # only large because of FFT spectral leakage from another note.
        if post_energy < min_post_energy:
            continue
        ratio = (post_energy + 1e-6) / (pre_energy + 1e-6)
        # Reject sub-onsets where the post/pre ratio is too low: the note
        # was already ringing through this sub-onset, not attacking at it.
        if ratio < min_energy_ratio:
            continue
        # Among qualifying sub-onsets, prefer the one with the strongest
        # absolute attack energy.  Using post_energy (not ratio) avoids
        # rewarding "first noisy appearance" over "biggest real attack",
        # which is a problem when pre is near the noise floor.
        if post_energy > best_post_energy:
            best_post_energy = post_energy
            best_onset = onset

    return best_onset


def onset_energy_gain(
    audio: np.ndarray,
    sample_rate: int,
    start_time: float,
    end_time: float,
    target_frequency: float,
    sub_onsets: tuple[float, ...] = (),
    *,
    target_note_name: str | None = None,
    recent_note_names: set[str] | None = None,
) -> float:
    """Compute the early/pre energy ratio for *target_frequency* in a segment.

    The pre window always sits immediately before the segment (legacy
    semantics).  The early window normally starts at *start_time*; when
    sub-onset anchoring is enabled AND the target note is in
    ``recent_note_names`` (i.e., would otherwise be at risk of being rejected
    as residual carryover), the early window slides to the sub-onset showing
    the strongest per-note rise.  This rescues delayed attacks of recently-
    played notes (e.g. C5 returning at the end of a slide chord) without
    inflating onset_gain for unrelated notes whose post-attack sustain just
    happens to leak into a neighboring note's frequency band.
    """
    window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
    start_sample = max(int(start_time * sample_rate), 0)
    end_sample = min(int(end_time * sample_rate), len(audio))
    if end_sample - start_sample < 512:
        return 0.0

    # Pre window is always anchored before the segment (legacy semantics).
    pre_start = max(0, start_sample - window_samples)
    pre_chunk = audio[pre_start:start_sample]

    # Early window normally starts at segment start; sub-onset anchor only
    # slides the early window forward for notes that need carryover rescue.
    anchor_sample = start_sample
    if (
        SUB_ONSET_ANCHOR_ENABLED
        and sub_onsets
        and target_note_name is not None
        and recent_note_names is not None
        and target_note_name in recent_note_names
    ):
        matched = pick_matching_sub_onset(
            audio, sample_rate, target_frequency, sub_onsets,
        )
        if matched is not None:
            anchor_sample = max(int(matched * sample_rate), 0)

    early_chunk = audio[anchor_sample:min(anchor_sample + window_samples, end_sample)]
    if len(pre_chunk) < 512 or len(early_chunk) < 512:
        return 0.0

    def _energy(chunk: np.ndarray) -> float:
        n_fft = _adaptive_n_fft(sample_rate, target_frequency, len(chunk), min_bins=1)
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
        n_fft = _adaptive_n_fft(sample_rate, target_frequency, len(chunk))
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
    min_frequency: float = 0.0,
) -> dict[str, Any] | None:
    chunks = _build_analysis_window_chunks(audio, sample_rate, start_time, end_time)
    if chunks is None:
        return None

    pre_chunk, attack_chunk, sustain_chunk = chunks
    max_chunk_len = max(len(pre_chunk), len(attack_chunk), len(sustain_chunk))
    if min_frequency > 0:
        n_fft = _adaptive_n_fft(sample_rate, min_frequency, max_chunk_len, min_bins=1)
    else:
        n_fft = max(4096, 1 << int(np.ceil(np.log2(max_chunk_len))))
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


def build_candidate_attack_debug(
    attack_context: dict[str, Any],
    target_frequency: float,
    *,
    audio: np.ndarray | None = None,
    sample_rate: int | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    sub_onsets: tuple[float, ...] = (),
    target_note_name: str | None = None,
    recent_note_names: set[str] | None = None,
) -> dict[str, Any]:
    """Build per-candidate attack debug info.

    Mirrors :func:`onset_energy_gain` semantics: by default uses the shared
    segment-start-anchored spectra from *attack_context*, but when the
    target note is in *recent_note_names* (i.e., would otherwise be at risk
    of carryover rejection), the attack chunk is re-anchored at the matching
    sub-onset.  This keeps debug numbers consistent with the values used by
    carryover gates.
    """
    payload: dict[str, Any] = {}
    anchor_time: float | None = None
    if (
        SUB_ONSET_ANCHOR_ENABLED
        and sub_onsets
        and audio is not None
        and sample_rate is not None
        and start_time is not None
        and target_note_name is not None
        and recent_note_names is not None
        and target_note_name in recent_note_names
    ):
        matched = pick_matching_sub_onset(
            audio, sample_rate, target_frequency, sub_onsets,
        )
        if matched is not None:
            anchor_time = matched

    if anchor_time is not None and end_time is not None and start_time is not None:
        # Re-compute attack chunk anchored at the matching sub-onset, but
        # keep pre/sustain chunks at their segment-relative positions
        # (matches the gating logic in :func:`onset_energy_gain`).
        window_samples = max(int(sample_rate * ONSET_ENERGY_WINDOW_SECONDS), 512)
        start_sample = max(int(start_time * sample_rate), 0)
        anchor_sample = max(int(anchor_time * sample_rate), 0)
        end_sample = min(int(end_time * sample_rate), len(audio))
        sustain_start = max(start_sample, end_sample - window_samples)

        pre_chunk = audio[max(0, start_sample - window_samples):start_sample]
        attack_chunk = audio[anchor_sample:min(anchor_sample + window_samples, end_sample)]
        sustain_chunk = audio[sustain_start:end_sample]

        if len(pre_chunk) >= 512 and len(attack_chunk) >= 512 and len(sustain_chunk) >= 512:
            n_fft = _adaptive_n_fft(sample_rate, target_frequency, max(
                len(pre_chunk), len(attack_chunk), len(sustain_chunk),
            ), min_bins=1)
            _, pre_spec = _chunk_spectrum(pre_chunk, sample_rate, n_fft)
            _, atk_spec = _chunk_spectrum(attack_chunk, sample_rate, n_fft)
            _, sus_spec = _chunk_spectrum(sustain_chunk, sample_rate, n_fft)
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
            pre_e = peak_energy_near(freqs, pre_spec, target_frequency)
            atk_e = peak_energy_near(freqs, atk_spec, target_frequency)
            sus_e = peak_energy_near(freqs, sus_spec, target_frequency)
            payload["subOnsetAnchorTime"] = round(anchor_time, 6)
            return {
                "preAttackEnergy": round(pre_e, 6),
                "attackEnergy": round(atk_e, 6),
                "sustainEnergy": round(sus_e, 6),
                "attackToSustainRatio": round((atk_e + 1e-6) / (sus_e + 1e-6), 6),
                "candidateOnsetGain": round((atk_e + 1e-6) / (pre_e + 1e-6), 6),
                **payload,
            }

    # Default: use shared segment-start-anchored spectra (legacy behaviour).
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
    sub_onsets: tuple[float, ...] = (),
) -> tuple[NoteHypothesis, float | None, dict[str, Any] | None]:
    if not recent_note_names or primary.candidate.note_name not in recent_note_names:
        return primary, None, None

    duration = end_time - start_time
    if duration > RECENT_PRIMARY_REPLACEMENT_MAX_DURATION:
        return primary, None, None

    primary_onset_gain = onset_energy_gain(
        audio, sample_rate, start_time, end_time, primary.candidate.frequency,
        sub_onsets=sub_onsets,
        target_note_name=primary.candidate.note_name,
        recent_note_names=recent_note_names,
    )
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
        onset_gain = onset_energy_gain(
            audio, sample_rate, start_time, end_time, hypothesis.candidate.frequency,
            sub_onsets=sub_onsets,
            target_note_name=hypothesis.candidate.note_name,
            recent_note_names=recent_note_names,
        )
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


def maybe_promote_onset_strong_sibling(
    primary: NoteHypothesis,
    primary_onset_gain: float | None,
    ranked: list[NoteHypothesis],
    evidence: "_NoteEvidenceCache",
) -> tuple[NoteHypothesis, float | None, dict[str, Any] | None]:
    """#166: Narrow onset-based primary rescue.

    When the top-ranked candidate has both a very weak attack and a diluted
    fundamental ratio (signs of spectral leakage rather than a genuine
    pluck), search the top-K siblings for a candidate with clean fR AND
    strong attack evidence.  If found, promote the sibling to primary.

    Target pattern (E100 in kalimba-17-g-low-bwv147-sequence-163-01):
        primary  B3: score=270.7  fR=0.599  onsetGain=1.8   (alias of B4)
        sibling  B4: score=229.4  fR=0.979  onsetGain=62.0  (real attack)

    Conditions are intentionally narrow to avoid disturbing well-tuned
    solo-note cases where the top-ranked candidate is legitimately primary.
    """
    # fR gate runs first so we can skip the (cached but still non-trivial)
    # onset_gain lookup for clearly-clean primaries.
    if primary.fundamental_ratio >= ONSET_RESCUE_PRIMARY_MAX_FUNDAMENTAL_RATIO:
        return primary, primary_onset_gain, None
    # If the caller didn't compute a primary onset_gain (the stale-recent
    # rescue path only computes it for recent notes), look it up now via
    # the evidence cache so this rescue isn't unable to evaluate fresh
    # primaries like B3 at E100 in the G-low BWV147 fixture.
    effective_gain = (
        primary_onset_gain
        if primary_onset_gain is not None
        else evidence.onset_gain(primary.candidate.frequency)
    )
    if effective_gain is None or effective_gain >= ONSET_RESCUE_PRIMARY_MAX_ONSET_GAIN:
        return primary, primary_onset_gain, None
    # Use the effective gain for the sibling comparison.
    primary_gain_for_check = effective_gain

    best_sibling: NoteHypothesis | None = None
    best_sibling_gain: float | None = None
    for hypothesis in ranked[1:ONSET_RESCUE_MAX_SIBLING_RANK + 1]:
        if hypothesis.fundamental_ratio < ONSET_RESCUE_SIBLING_MIN_FUNDAMENTAL_RATIO:
            continue
        if hypothesis.score < primary.score * ONSET_RESCUE_SIBLING_MIN_SCORE_RATIO:
            continue
        sibling_gain = evidence.onset_gain(hypothesis.candidate.frequency)
        if sibling_gain < ONSET_RESCUE_SIBLING_MIN_ONSET_GAIN:
            continue
        if sibling_gain < primary_gain_for_check * ONSET_RESCUE_SIBLING_GAIN_RATIO:
            continue
        if best_sibling is None or sibling_gain > (best_sibling_gain or 0.0):
            best_sibling = hypothesis
            best_sibling_gain = sibling_gain

    if best_sibling is None:
        return primary, primary_onset_gain, None

    debug = {
        "reason": "onset-strong-sibling",
        "replacedPrimaryNote": primary.candidate.note_name,
        "replacementNote": best_sibling.candidate.note_name,
        "primaryOnsetGain": round(primary_gain_for_check, 2),
        "siblingOnsetGain": round(best_sibling_gain or 0.0, 2),
        "primaryFundamentalRatio": round(primary.fundamental_ratio, 3),
        "siblingFundamentalRatio": round(best_sibling.fundamental_ratio, 3),
    }
    return best_sibling, best_sibling_gain, debug


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
    key_layers: dict[int, int] | None = None,
    sub_onsets: tuple[float, ...] = (),
    recent_note_names: set[str] | None = None,
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
                    and onset_energy_gain(
                        audio, sample_rate, start_time, end_time, h.candidate.frequency,
                        sub_onsets=sub_onsets,
                        target_note_name=h.candidate.note_name,
                        recent_note_names=recent_note_names,
                    ) >= TERTIARY_MIN_ONSET_GAIN
                ):
                    gap_candidates.append(h.candidate)
                break

    if len(gap_candidates) != len(gap_keys):
        return None

    filled_keys = test_keys + [c.key for c in gap_candidates]
    if not is_physically_playable_chord(filled_keys, key_layers=key_layers):
        return None
    if len(set(filled_keys)) > MAX_POLYPHONY:
        return None

    return gap_candidates


def is_physically_playable_chord(keys: list[int], key_layers: dict[int, int] | None = None) -> bool:
    """Check if a set of keys can be played simultaneously on a kalimba.

    One thumb can slide across consecutive keys (2-4 tines).
    The other thumb can strict-press 1-2 adjacent keys.
    Either thumb can reach any part of the instrument.
    Valid chords must be splittable into:
      - a slide group (consecutive keys, any length) + a strict group (≤2 adjacent keys)

    When *key_layers* is provided, consecutive-key checks also require that
    adjacent keys share the same layer.  On a 34-key kalimba, keys 17 (bottom
    layer) and 18 (top layer) are numerically adjacent but physically separated.
    """
    if len(keys) <= 2:
        if key_layers is not None and len(keys) == 2:
            k0, k1 = sorted(keys)
            if abs(k1 - k0) == 1 and key_layers.get(k0) != key_layers.get(k1):
                return False
        return True
    unique = sorted(set(keys))
    n = len(unique)
    if n > 4:
        return False

    def _same_layer_adjacent(a: int, b: int) -> bool:
        if b - a != 1:
            return False
        if key_layers is not None and key_layers.get(a) != key_layers.get(b):
            return False
        return True

    def _consecutive(ks: list[int]) -> bool:
        return len(ks) <= 1 or all(_same_layer_adjacent(ks[i], ks[i + 1]) for i in range(len(ks) - 1))

    def _strict_ok(ks: list[int]) -> bool:
        return len(ks) <= 1 or (len(ks) == 2 and _same_layer_adjacent(ks[0], ks[1]))

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
    soft_alternates: list[RawAlternateGrouping] = []
    # #178 Phase 2: when a segment is dropped, carry the rejected primary and
    # top-ranked ranked-candidates so the pipeline can build a candidate slot.
    dropped_primary: NoteCandidate | None = None
    dropped_candidates: list[NoteCandidate] = []
    dropped_reason: str = ""
    # #178 Phase 2 (sub-onset rescue): times within a rejected segment where
    # mute-dip re-attack was detected — high-confidence slots for those notes.
    dropped_sub_onset_rescues: list[float] = []


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
    sub_onsets: tuple[float, ...] = ()
    segment_sources: frozenset[str] = frozenset()

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def key_layers(self) -> dict[int, int] | None:
        """Build key→layer mapping from tuning, or None if all on layer 0."""
        layers = {n.key: n.layer for n in self.tuning.notes}
        if all(v == 0 for v in layers.values()):
            return None
        return layers


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
    "tertiary-duplicate-note": "structural",
    "semitone-leakage": "structural",
    "same-as-primary": "structural",
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
    "broadband-transient-leak": "evidence",
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
    "lower-roll-tail-extension": "extension",
    # Rescue (Layer 3.5: final decision)
    "evidence-rescue-weak-secondary-onset": "rescue",
    "evidence-rescue-weak-lower-secondary": "rescue",
    "evidence-rescue-tertiary-score-override": "rescue",
    "evidence-rescue-recent-carryover": "rescue",
    # Guard (Layer 4.5: short-segment defensive guard)
    "short-segment-secondary-guarded": "guard",
}

# Rejection reasons that indicate noise or structural impossibility — these
# candidates should never be presented as alternates (#178 Phase 1).
HARD_REJECT_REASONS: frozenset[str] = frozenset({
    "same-as-primary",
    "tertiary-physically-impossible",
    "iterative-tertiary-physically-impossible",
    "tertiary-duplicate-note",
    "harmonic-related-to-selected",
    "semitone-leakage",
    "broadband-transient-leak",
    "residual-forward-scan-replaced-primary",
})

# Max soft alternates to keep per segment.
_SOFT_ALT_MAX_COUNT = 3
# Minimum score as a fraction of primary score to be considered.
# (Retained as a floor but not the primary filter — see _SOFT_ALT_MIN_ONSET_GAIN.)
_SOFT_ALT_MIN_SCORE_RATIO = 0.005
# Minimum per-note onset_gain for a soft alternate. Primary filter: a candidate
# with strong attack evidence is likely a real note even if score is low
# (e.g., spectral masking by a stronger neighbor). Calibrated from
# free-performance-01: C5 on E1 has score=12 but og=2640 (genuine 3-note chord);
# carryover residuals have og<2. Threshold 10 separates these cleanly.
_SOFT_ALT_MIN_ONSET_GAIN = 10.0
# Minimum fundamental_ratio for a soft alternate. Low fr indicates the
# candidate's "fundamental" is actually a harmonic of another note (alias).
# Genuine masked notes (e.g., middle of a 3-note chord) still have fr>=0.9.
# Calibrated: C5 on E1 (genuine) fr=0.974; harmonic aliases fr<0.7.
_SOFT_ALT_MIN_FUNDAMENTAL_RATIO = 0.7


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
    soft_alternates: list[RawAlternateGrouping] = field(default_factory=list)


@dataclass(slots=True)
class _BranchResult:
    """Result of evaluating a single primary branch (L3–L4)."""
    primary: NoteHypothesis
    primary_onset_gain: float | None
    promotion_debug: dict[str, Any] | None
    decision: _PrimaryDecision
    selected: list[NoteCandidate]
    candidate_decisions: list[_CandidateDecision]
    residual_ranked: list[NoteHypothesis]
    total_score: float

    @property
    def note_count(self) -> int:
        return len(self.selected)


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
        sub_onsets: tuple[float, ...] = (),
        recent_note_names: set[str] | None = None,
    ) -> None:
        self._audio = audio
        self._sr = sample_rate
        self._start = start_time
        self._end = end_time
        self._sub_onsets = sub_onsets
        self._recent_note_names = recent_note_names
        self._frequency_to_note: dict[float, str] = {}
        self._onset_gains: dict[float, float] = {}
        self._backward_gains: dict[float, float] = {}
        self._mute_dip: dict[float, bool] = {}
        self._residual_decay: dict[float, bool] = {}
        self._attack_spectrum_computed: bool = False
        self._attack_spectrum_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._attack_energies: dict[float, float] = {}
        self._sustain_spectrum_computed: bool = False
        self._sustain_spectrum_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._sustain_energies: dict[float, float] = {}
        self._broadband_onset_gain: float | None = None

    def register_note(self, frequency: float, note_name: str) -> None:
        """Associate a note name with its frequency for sub-onset gating."""
        self._frequency_to_note[frequency] = note_name

    def onset_gain(self, frequency: float) -> float:
        if frequency not in self._onset_gains:
            self._onset_gains[frequency] = onset_energy_gain(
                self._audio, self._sr, self._start, self._end, frequency,
                sub_onsets=self._sub_onsets,
                target_note_name=self._frequency_to_note.get(frequency),
                recent_note_names=self._recent_note_names,
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

    def broadband_onset_gain(self) -> float:
        """Broadband onset gain: attack_energy / pre_energy for the segment."""
        if self._broadband_onset_gain is None:
            chunks = _build_analysis_window_chunks(
                self._audio, self._sr, self._start, self._end,
            )
            if chunks is None:
                self._broadband_onset_gain = 0.0
            else:
                pre_chunk, attack_chunk, _sustain = chunks
                pre = _broadband_chunk_energy(pre_chunk)
                attack = _broadband_chunk_energy(attack_chunk)
                self._broadband_onset_gain = (attack + 1e-6) / (pre + 1e-6)
        return self._broadband_onset_gain

    def attack_energy(self, frequency: float) -> float:
        """Per-note attack energy (peak around fundamental in attack window).

        Used by gates that need to distinguish a true fresh attack from a
        candidate that only carries spectral leakage at its fundamental
        frequency.  Computes a single attack-window spectrum lazily and
        caches both the spectrum and per-frequency lookups.
        """
        if frequency in self._attack_energies:
            return self._attack_energies[frequency]
        if not self._attack_spectrum_computed:
            self._attack_spectrum_cache = self._compute_chunk_spectrum(attack=True)
            self._attack_spectrum_computed = True
        if self._attack_spectrum_cache is None:
            result = 0.0
        else:
            freqs, atk_spec = self._attack_spectrum_cache
            result = float(peak_energy_near(freqs, atk_spec, frequency))
        self._attack_energies[frequency] = result
        return result

    def sustain_energy(self, frequency: float) -> float:
        """Per-note sustain energy (peak around fundamental in sustain window).

        Pairs with `attack_energy` to compute attack-to-sustain ratios that
        distinguish noise-like spurious tertiary candidates (high a/s ratio)
        from true sustained tertiary notes (low a/s ratio).
        """
        if frequency in self._sustain_energies:
            return self._sustain_energies[frequency]
        if not self._sustain_spectrum_computed:
            self._sustain_spectrum_cache = self._compute_chunk_spectrum(attack=False)
            self._sustain_spectrum_computed = True
        if self._sustain_spectrum_cache is None:
            result = 0.0
        else:
            freqs, sus_spec = self._sustain_spectrum_cache
            result = float(peak_energy_near(freqs, sus_spec, frequency))
        self._sustain_energies[frequency] = result
        return result

    def attack_to_sustain_ratio(self, frequency: float) -> float:
        attack = self.attack_energy(frequency)
        sustain = self.sustain_energy(frequency)
        return (attack + 1e-6) / (sustain + 1e-6)

    def _compute_chunk_spectrum(self, *, attack: bool) -> tuple[np.ndarray, np.ndarray] | None:
        window_samples = max(int(self._sr * ONSET_ENERGY_WINDOW_SECONDS), 512)
        start_sample = max(int(self._start * self._sr), 0)
        end_sample = min(int(self._end * self._sr), len(self._audio))
        if attack:
            chunk_end = min(start_sample + window_samples, end_sample)
            chunk = self._audio[start_sample:chunk_end]
        else:
            sustain_start = max(start_sample, end_sample - window_samples)
            chunk = self._audio[sustain_start:end_sample]
        if len(chunk) < 512:
            return None
        n_fft = _adaptive_n_fft(self._sr, 40.0, len(chunk), min_bins=1)
        return _chunk_spectrum(chunk, self._sr, n_fft)


def analyze_spectrum_at_onset(
    audio: np.ndarray,
    sample_rate: int,
    onset_time: float,
    tuning: InstrumentTuning,
    window_seconds: float = 0.15,
) -> list[NoteCandidate]:
    """Lightweight FFT analysis at a time point to produce ranked note candidates.

    Used for orphan onset recovery (#178 Phase 2.5) where an onset was detected
    but no segment was constructed (falls outside all active ranges). Returns
    top-4 NoteCandidate objects ranked by spectral score, or empty list if
    the window is too short or contains no clear signal.
    """
    start_sample = int(onset_time * sample_rate)
    end_sample = min(int((onset_time + window_seconds) * sample_rate), len(audio))
    segment = audio[start_sample:end_sample]
    if len(segment) < 512:
        return []
    min_freq = min(n.frequency for n in tuning.notes)
    n_fft = _adaptive_n_fft(sample_rate, min_freq, len(segment))
    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    ranked = rank_tuning_candidates(frequencies, spectrum, tuning)
    if not ranked or ranked[0].score <= 1e-6:
        return []
    # Compute onset_gain for the top candidate (used for confidence)
    top = ranked[0]
    og = onset_energy_gain(
        audio, sample_rate, onset_time, onset_time + window_seconds,
        top.candidate.frequency,
    )
    return [
        NoteCandidate(
            key=h.candidate.key,
            note=h.candidate.note,
            score=h.score,
            onset_gain=og if i == 0 else None,
        )
        for i, h in enumerate(ranked[:4])
    ]


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

    min_freq = min(n.frequency for n in ctx.tuning.notes)
    n_fft = _adaptive_n_fft(ctx.sample_rate, min_freq, len(analysis_segment))
    window = np.hanning(len(analysis_segment))
    spectrum = np.abs(np.fft.rfft(analysis_segment * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / ctx.sample_rate)

    ranked = rank_tuning_candidates(frequencies, spectrum, ctx.tuning, debug=ctx.debug)
    if not ranked or ranked[0].score <= 1e-6:
        return None

    spectral = _SpectralData(frequencies=frequencies, spectrum=spectrum, ranked=ranked)
    evidence = _NoteEvidenceCache(
        ctx.audio, ctx.sample_rate, ctx.start_time, ctx.end_time,
        sub_onsets=ctx.sub_onsets,
        recent_note_names=ctx.recent_note_names,
    )
    # Register tuning notes so onset_gain queries can look up the note name
    # from a frequency.  Use the frequencies on the ranked candidates (which
    # come from Note.from_name) to avoid float precision mismatches with the
    # rounded values stored on TuningNote.
    for hypothesis in ranked:
        evidence.register_note(
            hypothesis.candidate.frequency, hypothesis.candidate.note_name,
        )
    return spectral, evidence


def _narrow_fft_at_sub_onset(
    audio: np.ndarray,
    sample_rate: int,
    sub_onset_time: float,
    tuning: InstrumentTuning,
    *,
    window_seconds: float = NARROW_FFT_WINDOW_SECONDS,
    debug: bool = False,
) -> _SpectralData | None:
    """Compute narrow FFT centred on *sub_onset_time* and rank candidates.

    #153 Phase A: detect notes that are spectrally hidden in the segment-wide
    FFT but visible in a narrow attack window.  Motivating physics:
    octave-coincident chord (e.g., C6 fundamental at 1046 Hz collides with
    C5 2nd harmonic) is only separable in the early ~30 ms before C5 sustain
    dominates the 1046 Hz bin.

    Returns the same `_SpectralData` shape as `_acquire_spectrum` so existing
    ranking / evidence logic can be reused for cross-validation.
    """
    window_samples = max(int(sample_rate * window_seconds), 256)
    center_sample = int(sub_onset_time * sample_rate)
    half = window_samples // 2
    start = max(center_sample - half, 0)
    end = min(start + window_samples, len(audio))
    chunk = audio[start:end]
    if len(chunk) < 256:
        return None
    min_freq = min(n.frequency for n in tuning.notes)
    n_fft = _adaptive_n_fft(sample_rate, min_freq, len(chunk))
    window = np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window, n=n_fft))
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    ranked = rank_tuning_candidates(frequencies, spectrum, tuning, debug=debug)
    if not ranked or ranked[0].score <= 1e-6:
        return None
    return _SpectralData(frequencies=frequencies, spectrum=spectrum, ranked=ranked)


def measure_narrow_fft_note_scores(
    audio: np.ndarray,
    sample_rate: int,
    sub_onset_time: float,
    tuning: InstrumentTuning,
    *,
    window_seconds: float = NARROW_FFT_WINDOW_SECONDS,
) -> dict[str, tuple[float, float, float]] | None:
    """Return ``{note_name: (fundamental_energy, score, fundamental_ratio)}``
    from a narrow FFT centred on *sub_onset_time*.

    Public wrapper around :func:`_narrow_fft_at_sub_onset` for cross-module
    use (e.g., merge passes in :mod:`events`).  The narrow FFT is computed
    once and energies + scores + fundamental_ratio for every ranked tuning
    candidate are returned so callers can compare arbitrary pairs of notes
    (e.g., compare a guarded primary against the next event's primary at
    the same instant) and apply fundamental_ratio guards (a real fresh
    attack has fundamental >> harmonics, while spectral leakage from a
    nearby note shows a lower fundamental ratio).

    Returns ``None`` if the narrow window has no signal.  Notes that did
    not surface in the ranked hypotheses are absent from the returned dict
    (callers should treat their values as 0.0).
    """
    spectral = _narrow_fft_at_sub_onset(
        audio, sample_rate, sub_onset_time, tuning,
        window_seconds=window_seconds,
    )
    if spectral is None:
        return None
    by_name: dict[str, tuple[float, float, float]] = {}
    for hypothesis in spectral.ranked:
        name = hypothesis.candidate.note_name
        if name not in by_name:
            by_name[name] = (
                float(hypothesis.fundamental_energy),
                float(hypothesis.score),
                float(hypothesis.fundamental_ratio),
            )
    return by_name


def _resolve_confirmed_primary(
    confirmed: Note,
    spectral: _SpectralData,
    evidence: _NoteEvidenceCache,
) -> _PrimaryResult:
    """Resolve primary using a per-note-pass confirmed note (mute-dip).

    Skips the full promotion gauntlet and residual-decay check.  Falls back
    to normal ranking if the confirmed note has no FFT presence.
    """
    # Find the confirmed note in the ranked hypotheses.
    match = next(
        (h for h in spectral.ranked if h.candidate.note_name == confirmed.name),
        None,
    )
    if match is None:
        # Confirmed note not found in FFT — fall back to rank-1.
        match = spectral.ranked[0]
    onset_gain = evidence.onset_gain(match.candidate.frequency)
    decision = _PrimaryDecision(
        initial_primary=match.candidate.note_name,
        final_primary=match.candidate.note_name,
        onset_gain=onset_gain,
        promotions=["confirmed-primary"],
        rejected=False,
        rejection_reason=None,
    )
    return _PrimaryResult(match, onset_gain, None, decision)


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
        sub_onsets=ctx.sub_onsets,
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
    # #166: Onset-strong sibling rescue — catches the pattern where the
    # top-ranked candidate has weak attack + diluted fR (spectral alias) and
    # a top-K sibling has clean fR + strong attack.  Runs AFTER the existing
    # rescue paths so their specific patterns take precedence.
    primary, rescued_onset_gain, onset_rescue_debug = maybe_promote_onset_strong_sibling(
        primary,
        primary_onset_gain,
        ranked,
        evidence,
    )
    if onset_rescue_debug is not None:
        primary_promotion_debug = onset_rescue_debug
        primary_onset_gain = rescued_onset_gain
        promotions.append(onset_rescue_debug.get("reason", "onset-strong-sibling"))
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
                        # Score gate: candidate must pass the same bar as a
                        # normal primary — otherwise we rescue with noise.
                        if h.score < PRIMARY_REJECTION_MAX_SCORE:
                            break
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
    # Ensure primary_onset_gain is computed (some promotion paths leave it None).
    if primary_onset_gain is None:
        primary_onset_gain = evidence.onset_gain(primary.candidate.frequency)
    # Onset gate (#141): reject primary with no onset evidence from any
    # source — broadband, per-note, or backward attack.  Catches
    # resonance-only segments that the residual-decay check misses
    # (when the note is not in recent_note_names).
    # Exempt: first segment, promoted primaries, confirmed_primary (mute-dip).
    if (
        not _rejected
        and settings.get().use_onset_gate
        and primary_promotion_debug is None
        and ctx.start_time > 0.1
        and evidence.broadband_onset_gain() < ONSET_GATE_MIN_BROADBAND_GAIN
        and primary_onset_gain < ONSET_GATE_MIN_ONSET_GAIN
        and evidence.backward_attack_gain(primary.candidate.frequency) < ONSET_GATE_MIN_BACKWARD_GAIN
    ):
        _rejected = True
        _rejection_reason = "onset-gate-no-evidence"
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
        if evidence.attack_to_sustain_ratio(candidate.frequency) > GLISS_TERTIARY_MAX_ATTACK_TO_SUSTAIN_RATIO:
            continue
        og = evidence.onset_gain(candidate.frequency)
        bg = evidence.backward_attack_gain(candidate.frequency)
        if bg < TERTIARY_MIN_BACKWARD_ATTACK_GAIN and og < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE:
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
        _primary_partials = _get_partial_ratios(ctx.tuning, primary.candidate.frequency)
        _tuning_funds = np.array([n.frequency for n in ctx.tuning.notes])
        residual_spectrum = suppress_harmonics(spectral.spectrum, spectral.frequencies, primary.candidate.frequency, partial_ratios=_primary_partials, tuning_fundamentals=_tuning_funds)
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
            # residual-forward-scan: the original primary was a recent note showing
            # residual decay with no mute-dip reattack, so the forward-scan replaced
            # it with a different recent note that DOES have a fresh attack. The
            # replaced note has been internally classified as sympathetic resonance,
            # so it should not be readmitted as a secondary. Same philosophy as the
            # bg-ordered iteration in recover_pre_segment_attack_via_narrow_fft:
            # once a stronger fresh-attack signal is found, earlier sustain
            # candidates from the same sub-onset are dropped, not promoted.
            if (
                primary_promotion_debug is not None
                and primary_promotion_debug.get("reason") == "residual-forward-scan"
                and hypothesis.candidate.note_name == primary_promotion_debug.get("replacedPrimaryNote")
            ):
                if "residual-forward-scan-replaced-primary" not in _disabled:
                    phase_a_reasons.append("residual-forward-scan-replaced-primary")
            if hypothesis.score < primary.score * score_ratio and not octave_dyad_allowed:
                if "score-below-threshold" not in _disabled:
                    phase_a_reasons.append("score-below-threshold")
            if hypothesis.fundamental_ratio < secondary_min_fundamental_ratio:
                if "fundamental-ratio-too-low" not in _disabled:
                    phase_a_reasons.append("fundamental-ratio-too-low")
            elif hypothesis.fundamental_ratio < SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO:
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                if onset_gain < TERTIARY_MIN_ONSET_GAIN:
                    if "fundamental-ratio-too-low" not in _disabled:
                        phase_a_reasons.append("fundamental-ratio-too-low")
            elif (
                # Subharmonic-alias-of-recent-primary (G-low E137):
                # candidate is a subharmonic of a recent note AND has fR
                # suggesting alias leakage.  Force onset_gain verification.
                # Skip when the upper octave is also genuinely playing in
                # the current segment (current primary OR a strong residual
                # candidate) — in that case the low fR is from a legitimate
                # lower-octave-of-dyad, not decay leakage.  See 34L-C E163
                # [C5,<G4,E4,C4>] where C5 is a secondary with high fR.
                hypothesis.fundamental_ratio < SECONDARY_SUBHARMONIC_ALIAS_MAX_FUNDAMENTAL_RATIO
                and ctx.recent_note_names
            ):
                upper_octave_name = (
                    f"{hypothesis.candidate.pitch_class}{hypothesis.candidate.octave + 1}"
                )
                if (
                    upper_octave_name in ctx.recent_note_names
                    and upper_octave_name != primary.candidate.note_name
                ):
                    upper_is_strong_now = any(
                        h.candidate.note_name == upper_octave_name
                        and h.fundamental_ratio >= 0.85
                        and evidence.onset_gain(h.candidate.frequency) >= MIN_RECENT_NOTE_ONSET_GAIN
                        for h in residual_ranked[:6]
                    )
                    if not upper_is_strong_now:
                        if onset_gain is None:
                            onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                        if onset_gain < TERTIARY_MIN_ONSET_GAIN:
                            if "subharmonic-alias-of-recent" not in _disabled:
                                phase_a_reasons.append("subharmonic-alias-of-recent")
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
            # ── Broadband transient leak gate ─────────────────────────
            if (
                hypothesis.candidate.frequency < primary.candidate.frequency
                and segment_duration <= BROADBAND_TRANSIENT_LEAK_MAX_DURATION
                and evidence.attack_to_sustain_ratio(hypothesis.candidate.frequency) > BROADBAND_TRANSIENT_LEAK_MIN_AS_RATIO
                and hypothesis.score < primary.score * BROADBAND_TRANSIENT_LEAK_MAX_SCORE_RATIO
            ):
                if "broadband-transient-leak" not in _disabled:
                    phase_a_reasons.append("broadband-transient-leak")
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
                # Bypass: a high backward_attack_gain is independent evidence
                # that this note had its own recent strong attack (just before
                # the segment), so it should not be suppressed as an alias of
                # a different selected note.  Restrict the bypass to cases
                # where the hypothesis is BELOW every harmonically-related
                # selected note: subharmonic alias is physically less likely
                # than upward harmonic leakage, so a lower-octave candidate
                # with independent attack evidence is more likely a real
                # note.  An upper-octave candidate (e.g., E6 vs E5) is much
                # more often the true note's 2nd harmonic.  See #152.
                hyp_backward_gain = evidence.backward_attack_gain(hypothesis.candidate.frequency)
                lower_than_all_related = all(
                    hypothesis.candidate.frequency < existing.frequency
                    for existing in selected
                    if are_harmonic_related(hypothesis.candidate, existing)
                )
                independent_attack_evidence = (
                    lower_than_all_related
                    and hyp_backward_gain >= TERTIARY_MIN_BACKWARD_ATTACK_GAIN
                )
                if not independent_attack_evidence:
                    if "harmonic-related-to-selected" not in _disabled:
                        phase_b_reasons.append("harmonic-related-to-selected")
            # ── Semitone leakage gate (all candidates, not just tertiary) ──
            for existing in selected:
                interval = abs(cents_distance(hypothesis.candidate.frequency, existing.frequency))
                if interval <= SEMITONE_LEAKAGE_MAX_CENTS and existing.score > 0 and hypothesis.score < existing.score * SEMITONE_LEAKAGE_MAX_SCORE_RATIO:
                    if "semitone-leakage" not in _disabled:
                        phase_b_reasons.append("semitone-leakage")
                    break
            # ── Tertiary gates (selected-dependent) ──────────────────
            is_tertiary_or_beyond = len(selected) >= 2
            # residual-forward-scan: the segment's original primary was internally
            # classified as sustain (no mute-dip reattack), so the segment as a
            # whole is carryover-prone. Force the tertiary-style evidence gates
            # onto secondary slot candidates too, otherwise weaker carryover
            # notes simply slide up into the secondary slot once A1 strips the
            # strongest one (e.g., 34-key BWV147 E83: D5 stripped → B4 promoted).
            # Same Phase B philosophy: when a segment lacks fresh broadband
            # attack, demand explicit per-note attack evidence (onset_gain or
            # backward_attack_gain) for every accepted note.
            forced_evidence_gates = (
                not is_tertiary_or_beyond
                and primary_promotion_debug is not None
                and primary_promotion_debug.get("reason") == "residual-forward-scan"
            )
            if is_tertiary_or_beyond:
                test_keys = [n.key for n in selected] + [hypothesis.candidate.key]
                if not is_physically_playable_chord(test_keys, key_layers=ctx.key_layers):
                    gap_filled = _try_gap_fill(
                        test_keys, selected, hypothesis, residual_ranked,
                        primary, ctx.audio, ctx.sample_rate, ctx.start_time, ctx.end_time,
                        key_layers=ctx.key_layers,
                        sub_onsets=ctx.sub_onsets,
                        recent_note_names=ctx.recent_note_names,
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
            if is_tertiary_or_beyond or forced_evidence_gates:
                # ── Tertiary evidence gates ──
                # The two evidence gates are symmetric: each can be overridden
                # by the other.  tertiary-weak-onset (onset_gain low) is
                # waived when backward_gain proves the note had a recent
                # strong attack just before the segment.  tertiary-weak-
                # backward-attack (backward_gain low) is waived when
                # onset_gain proves a fresh attack inside the segment.
                onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)
                backward_gain = evidence.backward_attack_gain(hypothesis.candidate.frequency)
                if onset_gain < TERTIARY_MIN_ONSET_GAIN and backward_gain < TERTIARY_MIN_BACKWARD_ATTACK_GAIN:
                    if "tertiary-weak-onset" not in _disabled:
                        phase_b_reasons.append("tertiary-weak-onset")
                if backward_gain < TERTIARY_MIN_BACKWARD_ATTACK_GAIN and onset_gain < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE:
                    if "tertiary-weak-backward-attack" not in _disabled:
                        phase_b_reasons.append("tertiary-weak-backward-attack")
            verdict.phase_b_reasons = phase_b_reasons
            accepted = len(phase_b_reasons) == 0
            verdict.accepted = accepted
            debug_reasons: list[str] = phase_b_reasons
            candidate_decisions.append(_CandidateDecision(
                note_name=hypothesis.candidate.note_name,
                frequency=hypothesis.candidate.frequency,
                score=hypothesis.score,
                fundamental_ratio=hypothesis.fundamental_ratio,
                onset_gain=onset_gain,
                accepted=accepted,
                reasons=debug_reasons,
                octave_dyad_allowed=verdict.octave_dyad_allowed,
                source="secondary",
            ))
            if accepted:
                hypothesis.candidate.onset_gain = evidence.get_onset_gain_if_cached(hypothesis.candidate.frequency)
                selected.append(hypothesis.candidate)
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
            _iter_tuning_funds = np.array([n.frequency for n in ctx.tuning.notes])
            for _sel_note in selected:
                _iter_residual = suppress_harmonics(_iter_residual, spectral.frequencies, _sel_note.frequency, partial_ratios=_get_partial_ratios(ctx.tuning, _sel_note.frequency), tuning_fundamentals=_iter_tuning_funds)
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
                if not is_physically_playable_chord(_test_keys, key_layers=ctx.key_layers):
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
                    if _iter_backward_gain < TERTIARY_MIN_BACKWARD_ATTACK_GAIN and _iter_onset_gain < TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE:
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


    # soft_alternates are collected in _apply_final_decisions, after Phase C
    # has had a chance to promote candidates to selected.
    return _SelectionState(
        selected=selected,
        residual_ranked=residual_ranked,
        candidate_decisions=candidate_decisions,
        soft_alternates=[],
    )


def _collect_soft_alternates(
    selection: _SelectionState,
    primary: NoteHypothesis,
    spectral_ranked: list[NoteHypothesis],
) -> None:
    """Populate selection.soft_alternates from candidate_decisions.

    Runs after Phase A/B/iterative/evidence-rescue/Phase-C so candidates
    promoted by any later stage are excluded from the alt list.
    """
    if not settings.get().use_soft_candidate_alternates or not selection.selected:
        return
    selected_names = {c.note_name for c in selection.selected}
    seen_names: set[str] = set()
    primary_score = primary.score
    # Primary filter: onset_gain + fundamental_ratio based. Candidates
    # with strong attack evidence AND high fundamental ratio are likely
    # real masked notes (e.g., middle of a 3-note chord); low fr signals
    # harmonic alias of a stronger neighbor.
    soft_candidates = [
        d for d in selection.candidate_decisions
        if not d.accepted
        and d.note_name not in selected_names
        and d.note_name not in seen_names
        and d.onset_gain is not None
        and d.onset_gain >= _SOFT_ALT_MIN_ONSET_GAIN
        and d.fundamental_ratio >= _SOFT_ALT_MIN_FUNDAMENTAL_RATIO
        and d.score >= primary_score * _SOFT_ALT_MIN_SCORE_RATIO
        and not any(r in HARD_REJECT_REASONS for r in d.reasons)
    ]
    soft_candidates.sort(key=lambda d: (d.onset_gain or 0), reverse=True)
    for d in soft_candidates[:_SOFT_ALT_MAX_COUNT]:
        if d.note_name in seen_names:
            continue
        seen_names.add(d.note_name)
        # Confidence: scale by onset_gain ratio relative to a reference
        # (og=100 → conf≈0.5, og>=500 → conf≈0.7, capped at 0.7).
        og = d.onset_gain or 0
        if og >= 500:
            conf = 0.70
        elif og >= 100:
            conf = 0.50
        elif og >= 30:
            conf = 0.35
        else:
            conf = 0.20
        note_candidate: NoteCandidate | None = None
        for hyp in selection.residual_ranked:
            if hyp.candidate.note_name == d.note_name:
                note_candidate = NoteCandidate(
                    key=hyp.candidate.key,
                    note=hyp.candidate.note,
                    score=d.score,
                    onset_gain=d.onset_gain,
                )
                break
        if note_candidate is None:
            for hyp in spectral_ranked:
                if hyp.candidate.note_name == d.note_name:
                    note_candidate = NoteCandidate(
                        key=hyp.candidate.key,
                        note=hyp.candidate.note,
                        score=d.score,
                        onset_gain=d.onset_gain,
                    )
                    break
        if note_candidate is not None:
            selection.soft_alternates.append(RawAlternateGrouping(
                alternate_note=note_candidate,
                reason=f"soft_rejected:{','.join(d.reasons[:2])}",
                confidence=round(conf, 3),
            ))


def _build_segment_debug(
    ctx: _SegmentContext,
    spectral: _SpectralData,
    primary_result: _PrimaryResult,
    selection: _SelectionState,
    evidence: _NoteEvidenceCache,
) -> dict[str, Any]:
    """Layer 5: Assemble the debug payload."""
    primary = primary_result.primary
    min_freq = min(n.frequency for n in ctx.tuning.notes)
    attack_context = prepare_attack_debug_context(ctx.audio, ctx.sample_rate, ctx.start_time, ctx.end_time, min_frequency=min_freq)
    segment_attack_debug = attack_context["broadband"] if attack_context is not None else {}
    attack_profiles: dict[str, dict[str, Any]] = {}
    if attack_context is not None:
        for hypothesis in [primary, *spectral.ranked[:5], *selection.residual_ranked[:5]]:
            note_name = hypothesis.candidate.note_name
            if note_name not in attack_profiles:
                attack_profiles[note_name] = build_candidate_attack_debug(
                    attack_context, hypothesis.candidate.frequency,
                    audio=ctx.audio,
                    sample_rate=ctx.sample_rate,
                    start_time=ctx.start_time,
                    end_time=ctx.end_time,
                    sub_onsets=ctx.sub_onsets,
                    target_note_name=note_name,
                    recent_note_names=ctx.recent_note_names,
                )
    primary_og = evidence.get_onset_gain_if_cached(primary.candidate.frequency)
    short_segment_guard_skipped = [
        cd.note_name for cd in selection.candidate_decisions
        if "short-segment-secondary-guarded" in cd.reasons
    ]
    short_segment_guard_active = (
        ctx.duration < SHORT_SEGMENT_SECONDARY_GUARD_DURATION
        and len(selection.selected) >= 1
    )
    # #153 Phase A.1: per-sub-onset narrow FFT (read-only observation).
    # Compute a narrow-window FFT centred on each sub-onset and emit the
    # ranked candidates so we can compare against energy traces for E148,
    # E97, E100, etc. and verify that hidden notes (C6, F5, C4, ...)
    # surface in the narrow window before wiring this into selection logic.
    narrow_fft_by_sub_onset: list[dict[str, Any]] = []
    for sub_onset_time in ctx.sub_onsets:
        narrow = _narrow_fft_at_sub_onset(
            ctx.audio, ctx.sample_rate, sub_onset_time, ctx.tuning,
            debug=False,
        )
        if narrow is None:
            narrow_fft_by_sub_onset.append({
                "subOnsetTime": round(sub_onset_time, 4),
                "topRanked": [],
            })
            continue
        top_ranked = []
        for hypothesis in narrow.ranked[:5]:
            top_ranked.append({
                "noteName": hypothesis.candidate.note_name,
                "frequency": round(hypothesis.candidate.frequency, 2),
                "score": round(hypothesis.score, 6),
                "fundamentalEnergy": round(hypothesis.fundamental_energy, 6),
                "fundamentalRatio": round(hypothesis.fundamental_ratio, 6),
            })
        narrow_fft_by_sub_onset.append({
            "subOnsetTime": round(sub_onset_time, 4),
            "topRanked": top_ranked,
        })
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
        # Short-segment guard markers (#152 follow-up).
        # When active, this segment's primary is the only selected note because
        # the FFT window is too narrow to resolve secondaries.  Downstream
        # merge / per-sub-onset logic can use this flag as a hint that this
        # primary may need to be combined with an adjacent segment to recover
        # the full chord.  shortSegmentGuardSkipped lists what would have been
        # selected (for diagnostic / future-recovery use).
        "shortSegmentGuardActive": short_segment_guard_active,
        "shortSegmentGuardSkipped": short_segment_guard_skipped,
        "narrowFftBySubOnset": narrow_fft_by_sub_onset,
        "rawPeaks": build_raw_peaks(spectral.frequencies, spectral.spectrum, ctx.tuning),
    }


def _apply_final_decisions(
    ctx: _SegmentContext,
    spectral: _SpectralData,
    selection: _SelectionState,
    primary_result: _PrimaryResult,
    evidence: _NoteEvidenceCache,
) -> None:
    """Layer 3.5: Final decision — post-selection review.

    1. Deferred primary rejection → clear selected.
    2. Evidence gate rescue → re-admit candidates rejected only by
       evidence gates if they have strong spectral quality.
    3. Phase C octave-dyad rescue → re-admit candidates rejected only by
       structural score-below-threshold in Phase A, when allow_octave_secondary
       now returns True against the full selected set.
    4. soft_alternates collection (runs after every stage that may promote,
       so alt display excludes any note now in selected).
    """
    if primary_result.decision.rejected:
        selection.selected.clear()
        return

    if settings.get().use_evidence_gate_rescue:
        _apply_evidence_gate_rescue(ctx, selection, primary_result, evidence)

    if settings.get().use_phase_c_octave_dyad_rescue:
        _apply_phase_c_octave_dyad_rescue(ctx, selection, primary_result, evidence)

    _collect_soft_alternates(selection, primary_result.primary, spectral.ranked)


def _apply_evidence_gate_rescue(
    ctx: _SegmentContext,
    selection: _SelectionState,
    primary_result: _PrimaryResult,
    evidence: _NoteEvidenceCache,
) -> None:
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
            if not decision.octave_dyad_allowed:
                continue
        test_keys = [n.key for n in selection.selected] + [hypothesis.candidate.key]
        if not is_physically_playable_chord(test_keys, key_layers=ctx.key_layers):
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


def _apply_phase_c_octave_dyad_rescue(
    ctx: _SegmentContext,
    selection: _SelectionState,
    primary_result: _PrimaryResult,
    evidence: _NoteEvidenceCache,
) -> None:
    """Phase C — context-aware rescue for octave-dyad upper notes.

    Upper-octave notes in octave-dyad chords (e.g., C5 in C4+C5+E5) score
    low in rank_tuning_candidates because subharmonic_alias_energy at f/2
    is penalised — but when the f/2 position carries a genuine simultaneous
    note (here C4), the penalty treats a real chord partner as ghost-octave
    evidence.  Phase A rejects such candidates with score-below-threshold
    because allow_octave_secondary is evaluated against primary only.

    Phase C revisits these candidates after Phase A/B/iterative settle,
    and re-evaluates allow_octave_secondary against the final selected
    set.  If an octave partner is in selected, the candidate is admitted
    despite its low score — matching the semantics already used by the
    existing Phase B full-selected check at harmonic-related guard.
    """
    if len(selection.selected) >= MAX_POLYPHONY:
        return

    _disabled = settings.get().disabled_gates
    gate_name = "phase-c-octave-dyad-rescue"
    if gate_name in _disabled:
        return

    primary = primary_result.primary
    selected_names = {n.note_name for n in selection.selected}

    for decision in list(selection.candidate_decisions):
        if len(selection.selected) >= MAX_POLYPHONY:
            break
        if decision.accepted or decision.source != "secondary":
            continue
        # Strict trigger: rejected ONLY by structural score-below-threshold
        # in Phase A.  Other rejection reasons (fundamental-ratio-too-low,
        # harmonic-related-to-selected non-octave, semitone-leakage, etc.)
        # remain authoritative.
        if decision.reasons != ["score-below-threshold"]:
            continue
        if decision.note_name in selected_names:
            continue
        hyp = _find_hypothesis_by_frequency(selection.residual_ranked, decision.frequency)
        if hyp is None:
            continue
        if not allow_octave_secondary(primary, hyp, selection.selected):
            continue
        test_keys = [n.key for n in selection.selected] + [hyp.candidate.key]
        if not is_physically_playable_chord(test_keys, key_layers=ctx.key_layers):
            continue

        hyp.candidate.onset_gain = evidence.get_onset_gain_if_cached(hyp.candidate.frequency)
        selection.selected.append(hyp.candidate)
        selected_names.add(hyp.candidate.note_name)
        selection.candidate_decisions.append(_CandidateDecision(
            note_name=hyp.candidate.note_name,
            frequency=hyp.candidate.frequency,
            score=hyp.score,
            fundamental_ratio=hyp.fundamental_ratio,
            onset_gain=decision.onset_gain,
            accepted=True,
            reasons=[gate_name],
            octave_dyad_allowed=True,
            source="phase-c",
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
    onset_gain = evidence.onset_gain(hypothesis.candidate.frequency)

    # Carryover rescue with onset override: when the backward_gain is low
    # (note was present in past) but onset_gain confirms a genuine re-attack,
    # rescue based on onset + score instead of backward_gain.
    if "recent-carryover-candidate" in reasons:
        # Block rescue for obvious broadband transients (high attack, no sustain)
        if evidence.attack_to_sustain_ratio(hypothesis.candidate.frequency) > RESCUE_CARRYOVER_MAX_AS_RATIO:
            return None
        if backward_gain >= RESCUE_MIN_BACKWARD_GAIN:
            return "evidence-rescue-recent-carryover"
        if (
            onset_gain >= RESCUE_CARRYOVER_MIN_ONSET_GAIN
            and hypothesis.score >= primary.score * RESCUE_CARRYOVER_MIN_SCORE_RATIO
            and hypothesis.fundamental_ratio >= RESCUE_CARRYOVER_MIN_FUNDAMENTAL_RATIO
        ):
            return "evidence-rescue-recent-carryover"
        return None

    # Score+FR override for tertiary evidence gates: when backward_gain is
    # low but onset_gain confirms a genuine attack and spectral evidence is
    # strong, rescue tertiary candidates rejected only by evidence gates.
    if (
        backward_gain < RESCUE_MIN_BACKWARD_GAIN
        and onset_gain >= RESCUE_CARRYOVER_MIN_ONSET_GAIN
        and hypothesis.score >= primary.score * RESCUE_SCORE_FR_OVERRIDE_MIN_SCORE_RATIO
        and hypothesis.fundamental_ratio >= RESCUE_SCORE_FR_OVERRIDE_MIN_FR
    ):
        if "tertiary-weak-onset" in reasons or "tertiary-weak-backward-attack" in reasons:
            return "evidence-rescue-tertiary-score-override"

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

    return None


def _select_alternative_primaries(
    ranked: list[NoteHypothesis],
    authoritative: NoteHypothesis,
    max_alternatives: int = 2,
) -> list[NoteHypothesis]:
    """Pick non-redundant alternative primary candidates from *ranked*."""
    alternatives: list[NoteHypothesis] = []
    for hyp in ranked:
        if hyp.score <= 0:
            break
        if hyp is authoritative:
            continue
        if are_harmonic_related(hyp.candidate, authoritative.candidate):
            continue
        alternatives.append(hyp)
        if len(alternatives) >= max_alternatives:
            break
    return alternatives


def _evaluate_branch(
    ctx: _SegmentContext,
    spectral: _SpectralData,
    primary_hyp: NoteHypothesis,
    evidence: _NoteEvidenceCache,
    *,
    promotion_debug: dict[str, Any] | None = None,
) -> _BranchResult:
    """Evaluate a complete branch for a given primary hypothesis (L3–L4).

    Creates a synthetic _PrimaryResult and runs candidate selection,
    final decisions, and extensions.  Shared evidence caches
    (onset_gain, backward_attack_gain) may be populated as a side
    effect; branch-specific selection state is isolated.
    """
    onset_gain = evidence.onset_gain(primary_hyp.candidate.frequency)
    # Check rejection conditions (same as _resolve_primary)
    _rejected = False
    _rejection_reason: str | None = None
    if (
        primary_hyp.score < PRIMARY_REJECTION_MAX_SCORE
        and primary_hyp.fundamental_ratio < PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO
    ):
        _rejected = True
        _rejection_reason = "primary-score-too-low"
    elif (
        ctx.recent_note_names
        and primary_hyp.candidate.note_name in ctx.recent_note_names
        and evidence.is_residual_decay(primary_hyp.candidate.frequency)
        and not evidence.has_mute_dip_reattack(primary_hyp.candidate.frequency)
    ):
        _rejected = True
        _rejection_reason = "residual-decay-no-reattack"

    decision = _PrimaryDecision(
        initial_primary=primary_hyp.candidate.note_name,
        final_primary=primary_hyp.candidate.note_name,
        onset_gain=onset_gain,
        promotions=[],
        rejected=_rejected,
        rejection_reason=_rejection_reason,
    )
    primary_result = _PrimaryResult(
        primary=primary_hyp,
        primary_onset_gain=onset_gain,
        promotion_debug=promotion_debug,
        decision=decision,
    )
    primary_hyp.candidate.onset_gain = onset_gain

    # L3: Candidate selection
    selection = _select_candidates(ctx, spectral, primary_result, evidence)

    # L3.5: Final decisions (evidence rescue + primary rejection)
    _apply_final_decisions(ctx, spectral, selection, primary_result, evidence)

    if not selection.selected:
        return _BranchResult(
            primary=primary_hyp, primary_onset_gain=onset_gain,
            promotion_debug=promotion_debug, decision=decision,
            selected=[], candidate_decisions=selection.candidate_decisions,
            residual_ranked=selection.residual_ranked,
            total_score=0.0,
        )

    # L4: Extensions
    _extend_gliss_tertiary(ctx, primary_hyp, selection, evidence)
    _extend_lower_roll_tail(ctx, primary_hyp, selection, evidence)

    # Evidence freeze
    for note in selection.selected:
        cached_og = evidence.get_onset_gain_if_cached(note.frequency)
        if cached_og is not None and note.onset_gain is None:
            note.onset_gain = cached_og

    # Use accepted candidate scores (from spectral.ranked + residual_ranked)
    # to avoid under-scoring residual-only notes.
    score_by_name: dict[str, float] = {}
    for h in spectral.ranked:
        if h.candidate.note_name not in score_by_name:
            score_by_name[h.candidate.note_name] = h.score
    for h in selection.residual_ranked:
        if h.candidate.note_name not in score_by_name:
            score_by_name[h.candidate.note_name] = h.score
    total_score = sum(score_by_name.get(note.note_name, 0.0) for note in selection.selected)

    return _BranchResult(
        primary=primary_hyp, primary_onset_gain=onset_gain,
        promotion_debug=promotion_debug, decision=decision,
        selected=sorted(selection.selected, key=lambda n: n.frequency),
        candidate_decisions=selection.candidate_decisions,
        residual_ranked=selection.residual_ranked,
        total_score=total_score,
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
    confirmed_primary: Note | None = None,
    sub_onsets: tuple[float, ...] = (),
    segment_sources: frozenset[str] = frozenset(),
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
        sub_onsets=sub_onsets,
        segment_sources=segment_sources,
    )

    # Layer 1: Signal acquisition
    acquired = _acquire_spectrum(ctx)
    if acquired is None:
        return SegmentPeaksResult([], None, None)
    spectral, evidence = acquired

    # Layer 2: Primary resolution
    if confirmed_primary is not None:
        primary_result = _resolve_confirmed_primary(
            confirmed_primary, spectral, evidence,
        )
    else:
        primary_result = _resolve_primary(ctx, spectral, evidence)
    primary = primary_result.primary
    primary.candidate.onset_gain = primary_result.primary_onset_gain

    # Handle rejected primary: try alternative primaries before giving up
    if primary_result.decision.rejected:
        best_alt: _BranchResult | None = None
        if (
            settings.get().use_multi_primary_branching
            and len(spectral.ranked) >= 2
        ):
            alt_primaries = _select_alternative_primaries(spectral.ranked, primary)
            for alt_hyp in alt_primaries:
                # Alternative must show genuine attack (not just residual leakage)
                alt_og = evidence.onset_gain(alt_hyp.candidate.frequency)
                if alt_og < MIN_RECENT_NOTE_ONSET_GAIN:
                    continue
                # Must have meaningful spectral score (not just harmonic leakage)
                if alt_hyp.score < spectral.ranked[0].score * RESCUE_MIN_SCORE_RATIO:
                    continue
                alt_branch = _evaluate_branch(ctx, spectral, alt_hyp, evidence)
                if alt_branch.selected and (
                    best_alt is None
                    or alt_branch.total_score > best_alt.total_score
                ):
                    best_alt = alt_branch
        if best_alt is not None:
            # Alternative branch rescued the segment — patch trace to
            # record the original rejected primary and rescue provenance.
            primary = best_alt.primary
            rescue_decision = _PrimaryDecision(
                initial_primary=primary_result.decision.final_primary,
                final_primary=best_alt.primary.candidate.note_name,
                onset_gain=best_alt.primary_onset_gain,
                promotions=["multi-primary-rescue"],
                rejected=False,
                rejection_reason=None,
            )
            trace = SegmentDecisionTrace(primary=rescue_decision, candidates=best_alt.candidate_decisions)
            debug_payload = None
            if ctx.debug:
                _pr = _PrimaryResult(primary, best_alt.primary_onset_gain, best_alt.promotion_debug, rescue_decision)
                _sel = _SelectionState(
                    selected=best_alt.selected, residual_ranked=best_alt.residual_ranked,
                    candidate_decisions=best_alt.candidate_decisions,
                )
                debug_payload = _build_segment_debug(ctx, spectral, _pr, _sel, evidence)
            return SegmentPeaksResult(best_alt.selected, debug_payload, primary, trace)

        # No alternative found — reject segment
        selection = _select_candidates(ctx, spectral, primary_result, evidence)
        _apply_final_decisions(ctx, spectral, selection, primary_result, evidence)
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
        # #178 Phase 2: carry rejected primary + top-ranked for candidate slot.
        dropped_ranked = [h.candidate for h in spectral.ranked[:4] if h.candidate.note_name != primary.candidate.note_name]
        # #178 Phase 2 sub-onset rescue: even when the segment is rejected as
        # residual-decay, later sub-onsets may show mute-dip re-attack. These
        # become high-confidence candidate slots for the re-attack time.
        sub_onset_rescues: list[float] = []
        for sub_onset in ctx.sub_onsets:
            # Skip sub-onsets too close to the segment start (already checked there).
            if sub_onset - ctx.start_time < 0.05:
                continue
            if _has_sub_onset_mute_dip_reattack(ctx.audio, ctx.sample_rate, sub_onset, primary.candidate.frequency):
                sub_onset_rescues.append(sub_onset)
        return SegmentPeaksResult(
            [], rejection_debug, None, trace,
            dropped_primary=primary.candidate,
            dropped_candidates=dropped_ranked[:3],
            dropped_reason=primary_result.decision.rejection_reason or "rejected",
            dropped_sub_onset_rescues=sub_onset_rescues,
        )

    # Layer 3: Candidate selection
    selection = _select_candidates(ctx, spectral, primary_result, evidence)

    # Layer 3.5: Final decision
    _apply_final_decisions(ctx, spectral, selection, primary_result, evidence)

    if not selection.selected:
        trace = SegmentDecisionTrace(primary=primary_result.decision, candidates=selection.candidate_decisions)
        return SegmentPeaksResult([], None, None, trace)

    # Layer 4: Extension phases
    _extend_gliss_tertiary(ctx, primary, selection, evidence)
    _extend_lower_roll_tail(ctx, primary, selection, evidence)

    # Layer 4.5: Short-segment secondary guard.
    # Below SHORT_SEGMENT_SECONDARY_GUARD_DURATION the FFT window is too narrow
    # to reliably resolve secondary peaks; observed cases (e.g., gap-mute-dip
    # 6.7ms segments at octave-coincident chord attacks) include noise-level
    # spectral leakage that gets promoted via evidence-rescue paths.  Strip
    # non-primary notes here and mark them in the trail so downstream logic
    # (e.g., per-sub-onset narrow FFT) can re-attempt their detection.
    if (
        ctx.duration < SHORT_SEGMENT_SECONDARY_GUARD_DURATION
        and len(selection.selected) > 1
    ):
        primary_name = primary.candidate.note_name
        skipped_notes = [n for n in selection.selected if n.note_name != primary_name]
        skipped_names = {n.note_name for n in skipped_notes}
        selection.selected = [n for n in selection.selected if n.note_name == primary_name]
        # Demote any existing accepted decisions for the skipped notes.
        decision_names: set[str] = set()
        for cd in selection.candidate_decisions:
            decision_names.add(cd.note_name)
            if cd.note_name in skipped_names and cd.accepted:
                cd.accepted = False
                cd.reasons.append("short-segment-secondary-guarded")
        # Synthesize decisions for notes added by extension phases (which do
        # not always have a corresponding _CandidateDecision entry).
        for note in skipped_notes:
            if note.note_name in decision_names:
                continue
            selection.candidate_decisions.append(_CandidateDecision(
                note_name=note.note_name,
                frequency=note.frequency,
                score=0.0,
                fundamental_ratio=0.0,
                onset_gain=note.onset_gain,
                accepted=False,
                reasons=["short-segment-secondary-guarded"],
                octave_dyad_allowed=False,
                source="extension",
            ))

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
        selection.soft_alternates,
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
    n_fft = _adaptive_n_fft(sample_rate, frequency, len(chunk))
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
MUTE_DIP_ENERGY_WINDOW_NARROW = 0.03
MUTE_DIP_REATTACK_MIN_PRE_ENERGY = 3.0
MUTE_DIP_REATTACK_MIN_POST_ENERGY = 3.0
MUTE_DIP_REATTACK_MAX_DIP_RATIO = 0.1
MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO = 0.9


def _check_mute_dip_with_window(
    audio: np.ndarray,
    sample_rate: int,
    onset_time: float,
    frequency: float,
    window_seconds: float,
) -> bool:
    """Core mute-dip check with a configurable energy window."""
    pre_time = onset_time - 0.04
    if pre_time < 0:
        return False
    pre_energy = _note_band_energy(audio, sample_rate, pre_time, frequency,
                                   window_seconds=window_seconds)
    if pre_energy < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
        return False

    min_energy = float("inf")
    scan_start = max(onset_time - 0.01, 0.0)
    scan_end = min(onset_time + 0.05, len(audio) / sample_rate - window_seconds)
    t = scan_start
    while t < scan_end:
        energy = _note_band_energy(audio, sample_rate, t, frequency,
                                   window_seconds=window_seconds)
        if energy < min_energy:
            min_energy = energy
        t += 0.005

    attack_time = _find_note_attack_time(audio, sample_rate, onset_time, frequency)
    post_time = attack_time + 0.02
    if post_time > len(audio) / sample_rate - window_seconds:
        return False
    post_energy = _note_band_energy(audio, sample_rate, post_time, frequency,
                                    window_seconds=window_seconds)
    if post_energy < MUTE_DIP_REATTACK_MIN_POST_ENERGY:
        return False

    dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6)
    if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
        return False

    recovery_ratio = post_energy / (pre_energy + 1e-6)
    return recovery_ratio >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO


def _has_sub_onset_mute_dip_reattack(
    audio: np.ndarray,
    sample_rate: int,
    sub_onset_time: float,
    frequency: float,
    pre_lookback_seconds: float = 0.15,
) -> bool:
    """Check for mute-dip re-attack at a sub-onset WITHIN a segment (#178 Phase 2).

    Unlike `_has_mute_dip_reattack` which uses pre_time = onset - 40ms
    (appropriate for segment starts where the previous note ended just
    before), this function looks further back to find the ringing period
    of the previous attack within the same segment.

    Pattern: max energy in [sub_onset - lookback, sub_onset - 40ms] must be high,
    min energy in [sub_onset - 80ms, sub_onset + 20ms] must be low,
    post-attack energy must recover.
    """
    window = MUTE_DIP_ENERGY_WINDOW
    # Find max energy in the lookback region (ringing period)
    pre_scan_start = max(sub_onset_time - pre_lookback_seconds, 0.0)
    pre_scan_end = sub_onset_time - 0.04
    if pre_scan_end <= pre_scan_start:
        return False
    pre_energy = 0.0
    t = pre_scan_start
    while t < pre_scan_end:
        e = _note_band_energy(audio, sample_rate, t, frequency, window_seconds=window)
        if e > pre_energy:
            pre_energy = e
        t += 0.01
    if pre_energy < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
        return False

    # Find min energy in the dip region
    min_energy = float("inf")
    scan_start = max(sub_onset_time - 0.08, 0.0)
    scan_end = min(sub_onset_time + 0.02, len(audio) / sample_rate - window)
    t = scan_start
    while t < scan_end:
        e = _note_band_energy(audio, sample_rate, t, frequency, window_seconds=window)
        if e < min_energy:
            min_energy = e
        t += 0.005

    # Post-attack energy (after re-attack settles)
    attack_time = _find_note_attack_time(audio, sample_rate, sub_onset_time, frequency)
    post_time = attack_time + 0.02
    if post_time > len(audio) / sample_rate - window:
        return False
    post_energy = _note_band_energy(audio, sample_rate, post_time, frequency, window_seconds=window)
    if post_energy < MUTE_DIP_REATTACK_MIN_POST_ENERGY:
        return False

    dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6)
    if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
        return False

    recovery_ratio = post_energy / (pre_energy + 1e-6)
    return recovery_ratio >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO


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

    Uses a two-pass approach: first tries the standard 50ms window, then
    falls back to a narrower 30ms window for fast mute-dips (~20ms) that
    the wider window smooths out.
    """
    if _check_mute_dip_with_window(audio, sample_rate, onset_time, frequency,
                                    MUTE_DIP_ENERGY_WINDOW):
        return True
    # Fallback: narrower window catches fast mute-dips (~20ms) where the
    # 50ms window averages the dip with surrounding high-energy frames.
    return _check_mute_dip_with_window(audio, sample_rate, onset_time, frequency,
                                        MUTE_DIP_ENERGY_WINDOW_NARROW)


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
