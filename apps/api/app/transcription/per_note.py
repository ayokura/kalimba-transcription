"""Per-note onset detection passes.

Pass 1 (gap mute-dip rescue): scans gaps between segments for same-note
re-attack patterns that broadband onset detection missed.
"""
from __future__ import annotations

import numpy as np

try:
    import kalimba_dsp
except ImportError as exc:
    raise ImportError(
        "kalimba_dsp Rust extension is not installed. Build it with "
        "`uv sync --dev` (requires a Rust toolchain) or "
        "`uv run maturin develop --release --manifest-path crates/kalimba-dsp/Cargo.toml`."
    ) from exc

from ..models import InstrumentTuning
from .constants import HARMONIC_BAND_CENTS
from .models import Note, Segment
from .peaks import (
    MUTE_DIP_ENERGY_WINDOW,
    MUTE_DIP_ENERGY_WINDOW_NARROW,
    MUTE_DIP_REATTACK_MAX_DIP_RATIO,
    MUTE_DIP_REATTACK_MIN_POST_ENERGY,
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY,
    MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
    _note_band_energy,
)

# Coarse scan step (20 ms) — used to find candidate dip regions quickly.
_GAP_DIP_COARSE_STEP = 0.02
# Fine scan step (5 ms) — used inside the compact dip/recovery window.
_GAP_DIP_FINE_STEP = 0.005
# Maximum time window in which the dip must occur after a high-energy point.
# A real mute-dip drops 100x within ~40 ms; natural decay takes seconds.
_GAP_DIP_MAX_DIP_WINDOW = 0.06
# Maximum time window to scan for recovery after a confirmed dip.
_GAP_DIP_MAX_RECOVERY_WINDOW = 0.10
# Minimum gap duration worth scanning.
_GAP_DIP_MIN_GAP_SECONDS = 0.15
# Default segment duration for rescued segments.
_GAP_DIP_DEFAULT_DURATION = 0.24

# Rise rescue (fallback when mute-dip cannot fire because the note's pre-energy
# has already decayed below MUTE_DIP_REATTACK_MIN_PRE_ENERGY before the
# re-strike). Two-point check near gap_end: pre at `gap_end - _GAP_RISE_PRE_OFFSET`,
# post at `gap_end - _GAP_RISE_POST_OFFSET`.
# Calibration: bwv147-sequence-163-01 E148 C6 @ 260.60s, _note_band_energy(C6)
# = 1.03 at 260.56 vs 39.55 at 260.60 (38x rise across 40ms). Threshold 10x
# rejects sympathetic-resonance coupling from a neighbor strike (typically
# <3x on the quiet tine) while catching genuine re-strikes.
_GAP_RISE_PRE_OFFSET = 0.040
_GAP_RISE_POST_OFFSET = 0.005
_GAP_RISE_RATIO = 10.0
_GAP_RISE_MIN_POST_ENERGY = 10.0
# Require the note to be actively ringing (not near-silent) at pre_time.
# Fresh attacks start from noise floor (~0.05-0.2) — a sympathetic-coupling
# extra on a neighbor tine looks like this. A genuine re-strike picks up
# where a prior decay trailed off, so pre_energy sits comfortably above
# noise floor (E148 C6 pre = 0.82 @ 260.56s, 2.2s after the E146 strike).
_GAP_RISE_MIN_PRE_ENERGY = 0.5

# Dominance check: the rescued note must dominate briefly then lose dominance.
# This is the signature of a note that broadband onset missed — it peaks fast
# but gets masked (by slide-chord sustain or louder concurrent strikes) within
# ~50 ms, so the segment-wide FFT doesn't rank it highly.
#
#   +15ms ratio = target_energy / max(other 16 tines at +15ms)  must be >= 1.0
#   +50ms ratio = target_energy / max(others at +50ms)          must be <= 0.8
#
# Calibrated against:
#   E148 C6 (miss, target): +15=1.80 +50=0.23  ← included
#   E21 B5 (broadband got it):  +15=1.64 +50=1.98  ← excluded (sustained)
#   E100 E5 (broadband got it): +15=2.39 +50=2.38  ← excluded (sustained)
#   E40/E137 B4 (sympathetic):  +15=0.75 / 0.54    ← excluded (not dominant)
_GAP_RISE_DOMINANCE_PEAK_OFFSET = 0.015
_GAP_RISE_DOMINANCE_DECAY_OFFSET = 0.050
_GAP_RISE_DOMINANCE_PEAK_RATIO = 1.0
_GAP_RISE_DOMINANCE_DECAY_RATIO = 0.8


def _scan_gap_for_mute_dip(
    audio: np.ndarray,
    sample_rate: int,
    gap_start: float,
    gap_end: float,
    frequency: float,
) -> float | None:
    """Scan a gap for a *rapid* mute-dip re-attack on *frequency*.

    Uses a sliding compact-window approach: at each coarse step where the note
    has substantial energy, checks the next 60 ms for a 100x drop and the
    following 100 ms for recovery.  Tries the standard 50ms energy window
    first, then falls back to a narrower 30ms window for fast mute-dips.

    Returns the recovery time (suitable as a new segment start) or ``None``.
    """
    result = _scan_gap_for_mute_dip_with_window(
        audio, sample_rate, gap_start, gap_end, frequency, MUTE_DIP_ENERGY_WINDOW,
    )
    if result is not None:
        return result
    return _scan_gap_for_mute_dip_with_window(
        audio, sample_rate, gap_start, gap_end, frequency, MUTE_DIP_ENERGY_WINDOW_NARROW,
    )


def _scan_gap_for_mute_dip_with_window(
    audio: np.ndarray,
    sample_rate: int,
    gap_start: float,
    gap_end: float,
    frequency: float,
    window_seconds: float,
) -> float | None:
    """Scan a gap with a specific energy window size.

    Delegated to the Rust implementation in ``kalimba_dsp``. Integer-indexed
    fine grid matching the np.arange-based clean semantic (no float
    accumulation drift); see docs/performance/20260415-profiling-baseline.md
    for the rationale.
    """
    audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
    result = kalimba_dsp.scan_gap_for_mute_dip_with_window(
        audio_f32,
        int(sample_rate),
        float(gap_start),
        float(gap_end),
        float(frequency),
        float(window_seconds),
        float(MUTE_DIP_ENERGY_WINDOW),
        float(_GAP_DIP_MAX_DIP_WINDOW),
        float(_GAP_DIP_MAX_RECOVERY_WINDOW),
        float(_GAP_DIP_COARSE_STEP),
        float(_GAP_DIP_FINE_STEP),
        float(MUTE_DIP_REATTACK_MIN_PRE_ENERGY),
        float(MUTE_DIP_REATTACK_MAX_DIP_RATIO),
        float(MUTE_DIP_REATTACK_MIN_POST_ENERGY),
        float(MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO),
        float(HARMONIC_BAND_CENTS),
    )
    return result


def _detect_gap_rise_attack(
    audio: np.ndarray,
    sample_rate: int,
    gap_start: float,
    gap_end: float,
    frequency: float,
) -> float | None:
    """Detect a sharp energy rise near ``gap_end`` for ``frequency``.

    Complements mute-dip rescue: catches re-strikes where the note decayed
    near-silent before the attack, so pre_energy falls under
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY and the dip scan never enters.
    """
    audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
    return kalimba_dsp.detect_gap_rise_attack(
        audio_f32,
        int(sample_rate),
        float(gap_start),
        float(gap_end),
        float(frequency),
        float(MUTE_DIP_ENERGY_WINDOW),
        float(_GAP_RISE_PRE_OFFSET),
        float(_GAP_RISE_POST_OFFSET),
        float(_GAP_RISE_RATIO),
        float(_GAP_RISE_MIN_POST_ENERGY),
        float(_GAP_RISE_MIN_PRE_ENERGY),
        float(HARMONIC_BAND_CENTS),
    )


def rescue_gap_mute_dips(
    segments: list[Segment],
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    *,
    rescue_log: list[dict] | None = None,
) -> list[Segment]:
    """Pass 1: create new segments for same-note re-attacks hidden in gaps.

    Two detectors run per gap:

    1. mute-dip scan: high → 100x drop → recovery pattern; catches re-strikes
       where the note was still audibly ringing at strike time.
    2. rise fallback: near-silent pre + sharp post rise; catches re-strikes
       where the note decayed below the mute-dip pre-energy floor. Only tried
       when (1) returns no rescue for the gap.

    When found, inserts a new segment with ``confirmed_primary`` set, telling
    the peaks layer which note to adopt without the full primary resolution.
    """
    if len(segments) < 2:
        return segments

    tuning_notes = [Note.from_name(tn.note_name) for tn in tuning.notes]
    rescued: list[Segment] = []

    for i in range(len(segments) - 1):
        gap_start = segments[i].end_time
        gap_end = segments[i + 1].start_time
        if gap_end - gap_start < _GAP_DIP_MIN_GAP_SECONDS:
            continue

        # Pass 1a: mute-dip scan across all tuning notes.
        best_recovery: float | None = None
        best_note: Note | None = None
        best_peak_ratio: float | None = None
        best_decay_ratio: float | None = None
        source = "gap-mute-dip"
        for note in tuning_notes:
            recovery = _scan_gap_for_mute_dip(
                audio, sample_rate, gap_start, gap_end, note.frequency,
            )
            if recovery is not None:
                if best_recovery is None or recovery < best_recovery:
                    best_recovery = recovery
                    best_note = note

        # Pass 1b: rise-attack fallback when mute-dip finds nothing in this gap.
        if best_recovery is None:
            # Two-snapshot dominance check (see _GAP_RISE_DOMINANCE_* comments):
            # Broadband misses notes that peak-then-mask, not notes that stay
            # loud. So we require dominant at +15ms AND non-dominant at +50ms.
            peak_time = gap_end + _GAP_RISE_DOMINANCE_PEAK_OFFSET
            decay_time = gap_end + _GAP_RISE_DOMINANCE_DECAY_OFFSET
            peak_energies = {}
            decay_energies = {}
            for n in tuning_notes:
                peak_energies[n.name] = _note_band_energy(
                    audio, sample_rate, peak_time, n.frequency,
                    window_seconds=MUTE_DIP_ENERGY_WINDOW,
                )
                decay_energies[n.name] = _note_band_energy(
                    audio, sample_rate, decay_time, n.frequency,
                    window_seconds=MUTE_DIP_ENERGY_WINDOW,
                )

            for note in tuning_notes:
                recovery = _detect_gap_rise_attack(
                    audio, sample_rate, gap_start, gap_end, note.frequency,
                )
                if recovery is None:
                    continue
                peak_max_other = max(
                    (e for n, e in peak_energies.items() if n != note.name),
                    default=0.0,
                )
                decay_max_other = max(
                    (e for n, e in decay_energies.items() if n != note.name),
                    default=0.0,
                )
                peak_ratio = peak_energies[note.name] / (peak_max_other + 1e-6)
                decay_ratio = decay_energies[note.name] / (decay_max_other + 1e-6)
                if peak_ratio < _GAP_RISE_DOMINANCE_PEAK_RATIO:
                    continue  # sympathetic / sidelobe leakage, target not dominant
                if decay_ratio > _GAP_RISE_DOMINANCE_DECAY_RATIO:
                    continue  # sustained dominance → broadband already detected it
                if best_recovery is None or recovery < best_recovery:
                    best_recovery = recovery
                    best_note = note
                    best_peak_ratio = peak_ratio
                    best_decay_ratio = decay_ratio
                    source = "gap-rise"

        if best_recovery is not None and best_note is not None:
            seg_end = min(best_recovery + _GAP_DIP_DEFAULT_DURATION, gap_end)
            rescued.append(Segment(
                start_time=best_recovery,
                end_time=seg_end,
                sources=frozenset({source}),
                confirmed_primary=best_note,
            ))
            if rescue_log is not None:
                entry: dict = {
                    "source": source,
                    "gapStart": round(gap_start, 4),
                    "gapEnd": round(gap_end, 4),
                    "recoveryTime": round(best_recovery, 4),
                    "note": best_note.name,
                }
                if best_peak_ratio is not None:
                    entry["peakRatio"] = round(best_peak_ratio, 3)
                if best_decay_ratio is not None:
                    entry["decayRatio"] = round(best_decay_ratio, 3)
                rescue_log.append(entry)

    if not rescued:
        return segments

    # Merge rescued segments into the original list, sorted by start_time.
    combined = list(segments) + rescued
    combined.sort(key=lambda s: s.start_time)
    return combined
