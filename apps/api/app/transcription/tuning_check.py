"""Tuning mismatch advisory (UX safeguard, not recognizer logic).

Detects the "D major recording transcribed with a C major tuning" failure mode
(real case: tx b5972bbb — 21 odd events, no F#/C# available). The recognizer
can only assign notes that exist in the selected tuning, so a wrong tuning
produces silently-degraded output. This module inspects the raw spectrum
*independently of the tuning-constrained recognizer*:

1. Take the mean power spectrum and pick the strongest spectral peaks within
   the tuning's fundamental range.
2. Snap each peak to the nearest equal-tempered semitone (reject > 45 cents).
3. Compute per-pitch-class weight; coverage(tuning) = weight share of pitch
   classes present in the tuning.
4. If the selected tuning's coverage is poor and another preset covers the
   peaks distinctly better, report a mismatch with that suggestion.

Kalimba inharmonic partials (2nd partial 1.9-2.2x, AGENTS.md) can leak weight
into off-scale semitone bins (e.g. 2.1x = +86 cents above the octave), so
coverage is rarely exactly 1.0 even for correct tunings — thresholds below are
calibrated on the 2026-06 tester corpus rather than assumed.

Pure numpy on the mean spectrum; no librosa, streaming/WASM-portable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..models import InstrumentTuning
from ..tunings import get_default_tunings, note_name_to_frequency

# Calibrated 2026-06-13 on the 15 unique tester audios (scripts/audio-analysis
# /calibrate_tuning_mismatch.py): after the low-gain gate, every correct-tuning
# recording scores coverage >= 0.9975, while the known mismatch case b5972bbb
# (D recording / C tuning, hash 137757d0) scores 0.7937 with kalimba-17-d at
# ~0.998. Threshold 0.88 sits between the populations with wide margin.
MISMATCH_MAX_SELECTED_COVERAGE = 0.88
SUGGESTION_MIN_COVERAGE_GAIN = 0.08
# Coverages within this distance of the best candidate count as ties; the tie
# is broken toward fewer pitch classes so a diatonic preset (7 pcs) beats the
# chromatic 34L superset (12 pcs, coverage trivially 1.0).
#
# Fragility note (2026-06-13 stress test, forcing every wrong preset onto each
# tester audio): this tie-break does NOT guarantee a diatonic suggestion — it
# only protects the diatonic when the correct preset scores >= ~0.98. Clean
# diatonic recordings all sit at 0.99+ so they win, but two cases yield a 34L
# suggestion: (a) genuinely chromatic recordings (correct — 34L is the right
# answer), and (b) spectrally messy/ambiguous audio where the correct preset's
# coverage falls below the window (real case: tx 5b7608b4, correct=17-c but a
# loud low-C drone + D-ish upper content drops cov(17-c) to ~0.91 vs 34L 1.0).
# Case (b) is low-risk in production because the mismatch only fires when the
# *selected* tuning's coverage is already poor, and 5b7608b4 under its correct
# 17-c tuning is silenced by the MIN_PEAKS gate (only 2 peaks in-range) anyway.
# The deeper cause is that _tuning_frequency_range() makes the analysis window
# depend on the (possibly wrong) selected tuning, so the same audio yields
# different pitch-class weights per selection. If this ever needs hardening,
# prefer a tuning-independent union frequency range over widening this epsilon.
SUGGESTION_COVERAGE_TIE_EPSILON = 0.02
PEAK_SNAP_MAX_CENTS = 45.0
MIN_PEAKS = 5
TOP_N_PEAKS = 24
# Peaks below this fraction of the strongest peak are noise-floor texture, not
# tonal content.
PEAK_MIN_RELATIVE_POWER = 1e-4
OUTSIDE_PC_MIN_SHARE = 0.03
# Low-gain recordings (calibration: 4bc99bd1 peak -17.4 dBFS scored a spurious
# 0.828 from noise peaks): skip the mismatch advisory below this peak level.
# Note: the web client's low-volume warning (LOW_LEVEL_PEAK_DB, SimpleHome.tsx)
# was relaxed to -35 dB (2026-07-03) and no longer mirrors this constant; this
# skip keeps its own calibration.
LOW_GAIN_SKIP_PEAK_DBFS = -15.0

_PC_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC_INDEX = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


@dataclass
class TuningMismatchReport:
    selected_coverage: float
    outside_pitch_classes: list[str]
    suggested_tuning_id: str | None
    suggested_tuning_name: str | None
    suggested_coverage: float | None


def _tuning_pitch_classes(tuning: InstrumentTuning) -> set[int]:
    pcs: set[int] = set()
    for note in tuning.notes:
        name = note.note_name.rstrip("0123456789")
        index = _PC_INDEX.get(name)
        if index is not None:
            pcs.add(index)
    return pcs


def _tuning_frequency_range(tuning: InstrumentTuning) -> tuple[float, float]:
    freqs = [note_name_to_frequency(note.note_name) for note in tuning.notes]
    return (min(freqs) * 0.94, max(freqs) * 1.06)


def _mean_power_spectrum(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    # <= ~4 Hz bin resolution regardless of sample rate (96 kHz captures included)
    n_fft = 1 << max(13, math.ceil(math.log2(sample_rate / 4.0)))
    if audio.shape[0] < n_fft:
        n_fft = 1 << max(10, int(math.log2(max(audio.shape[0], 2))))
    hop = n_fft // 2
    window = np.hanning(n_fft)
    acc = np.zeros(n_fft // 2 + 1)
    count = 0
    for start in range(0, audio.shape[0] - n_fft + 1, hop):
        frame = audio[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        acc += np.abs(spectrum) ** 2
        count += 1
    if count == 0:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return freqs, acc / count


def _pick_peaks(freqs: np.ndarray, power: np.ndarray, fmin: float, fmax: float) -> list[tuple[float, float]]:
    """Local maxima in [fmin, fmax], strongest first, >= 50 cents apart."""
    lo = int(np.searchsorted(freqs, fmin))
    hi = int(np.searchsorted(freqs, fmax))
    if hi - lo < 3:
        return []
    band = power[lo:hi]
    local_max = np.flatnonzero((band[1:-1] > band[:-2]) & (band[1:-1] >= band[2:])) + 1 + lo
    if local_max.size == 0:
        return []
    order = local_max[np.argsort(power[local_max])[::-1]]
    floor = power[order[0]] * PEAK_MIN_RELATIVE_POWER
    picked: list[tuple[float, float]] = []
    for idx in order:
        if power[idx] < floor:
            break
        freq = float(freqs[idx])
        if any(abs(1200.0 * math.log2(freq / f)) < 50.0 for f, _ in picked):
            continue
        picked.append((freq, float(power[idx])))
        if len(picked) >= TOP_N_PEAKS:
            break
    return picked


def _pitch_class_weights(peaks: list[tuple[float, float]]) -> tuple[dict[int, float], int]:
    weights: dict[int, float] = {}
    accepted = 0
    for freq, power in peaks:
        midi_exact = 69.0 + 12.0 * math.log2(freq / 440.0)
        midi = round(midi_exact)
        cents = (midi_exact - midi) * 100.0
        if abs(cents) > PEAK_SNAP_MAX_CENTS:
            continue
        pc = midi % 12
        weights[pc] = weights.get(pc, 0.0) + power
        accepted += 1
    return weights, accepted


def _coverage(weights: dict[int, float], pcs: set[int]) -> float:
    total = sum(weights.values())
    if total <= 0.0:
        return 1.0
    return sum(w for pc, w in weights.items() if pc in pcs) / total


def measure_selected_coverage(
    audio: np.ndarray, sample_rate: int, tuning: InstrumentTuning
) -> float | None:
    """Return the selected tuning's spectral pitch-class coverage in [0, 1],
    regardless of whether a mismatch is flagged (analyze_tuning_mismatch only
    reports on poor coverage). Returns None when there is too little tonal
    content / gain to judge -- the same gates analyze_tuning_mismatch uses.

    Reused by the unsupervised quality indicators (quality_indicators.py) as a
    recording-fit signal; kept here so the spectral-peak logic lives in one place."""
    audio = np.asarray(audio, dtype=np.float64)
    peak_amplitude = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_amplitude <= 0.0:
        return None
    if 20.0 * math.log10(peak_amplitude) < LOW_GAIN_SKIP_PEAK_DBFS:
        return None
    freqs, power = _mean_power_spectrum(audio, sample_rate)
    if freqs.size == 0:
        return None
    fmin, fmax = _tuning_frequency_range(tuning)
    peaks = _pick_peaks(freqs, power, fmin, fmax)
    weights, accepted = _pitch_class_weights(peaks)
    if accepted < MIN_PEAKS:
        return None
    return _coverage(weights, _tuning_pitch_classes(tuning))


def analyze_tuning_mismatch(
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    candidate_tunings: list[InstrumentTuning] | None = None,
) -> TuningMismatchReport | None:
    """Return a mismatch report, or None when the tuning fits the recording."""
    audio = np.asarray(audio, dtype=np.float64)
    peak_amplitude = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak_amplitude <= 0.0:
        return None
    if 20.0 * math.log10(peak_amplitude) < LOW_GAIN_SKIP_PEAK_DBFS:
        return None  # low-gain capture: spectral peaks are noise, not playing

    fmin, fmax = _tuning_frequency_range(tuning)
    freqs, power = _mean_power_spectrum(audio, sample_rate)
    if freqs.size == 0:
        return None
    peaks = _pick_peaks(freqs, power, fmin, fmax)
    weights, accepted = _pitch_class_weights(peaks)
    if accepted < MIN_PEAKS:
        return None  # too little tonal content (low-gain / silence) to judge

    selected_pcs = _tuning_pitch_classes(tuning)
    selected_coverage = _coverage(weights, selected_pcs)
    if selected_coverage >= MISMATCH_MAX_SELECTED_COVERAGE:
        return None

    total = sum(weights.values())
    outside = sorted(
        (pc for pc, w in weights.items() if pc not in selected_pcs and w / total >= OUTSIDE_PC_MIN_SHARE),
        key=lambda pc: -weights[pc],
    )

    scored: list[tuple[float, int, str, str]] = []
    for candidate in candidate_tunings if candidate_tunings is not None else get_default_tunings():
        if candidate.id == tuning.id:
            continue
        candidate_pcs = _tuning_pitch_classes(candidate)
        coverage = _coverage(weights, candidate_pcs)
        if coverage - selected_coverage < SUGGESTION_MIN_COVERAGE_GAIN:
            continue
        scored.append((coverage, len(candidate_pcs), candidate.id, candidate.name))

    suggestion: tuple[float, int, str, str] | None = None
    if scored:
        best_coverage = max(s[0] for s in scored)
        near_best = [s for s in scored if s[0] >= best_coverage - SUGGESTION_COVERAGE_TIE_EPSILON]
        suggestion = min(near_best, key=lambda s: (s[1], -s[0]))

    return TuningMismatchReport(
        selected_coverage=round(selected_coverage, 4),
        outside_pitch_classes=[_PC_NAMES_SHARP[pc] for pc in outside],
        suggested_tuning_id=suggestion[2] if suggestion else None,
        suggested_tuning_name=suggestion[3] if suggestion else None,
        suggested_coverage=round(suggestion[0], 4) if suggestion else None,
    )
