"""Per-note onset detection passes.

Pass 1 (gap mute-dip rescue): scans gaps between segments for same-note
re-attack patterns that broadband onset detection missed.
"""
from __future__ import annotations

import numpy as np

from ..models import InstrumentTuning
from .models import Note, Segment
from .peaks import (
    MUTE_DIP_ENERGY_WINDOW,
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
    following 100 ms for recovery.  This distinguishes genuine finger-mute dips
    (~40 ms) from gradual natural decay (seconds).

    Returns the recovery time (suitable as a new segment start) or ``None``.
    """
    audio_duration = len(audio) / sample_rate
    scan_end = min(gap_end, audio_duration - MUTE_DIP_ENERGY_WINDOW)

    # Need room for at least dip_window + recovery_window.
    if scan_end - gap_start < _GAP_DIP_MAX_DIP_WINDOW + _GAP_DIP_MAX_RECOVERY_WINDOW:
        return None

    t = gap_start
    while t < scan_end - _GAP_DIP_MAX_DIP_WINDOW - _GAP_DIP_MAX_RECOVERY_WINDOW:
        pre_energy = _note_band_energy(
            audio, sample_rate, t, frequency,
            window_seconds=MUTE_DIP_ENERGY_WINDOW,
        )
        if pre_energy < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
            t += _GAP_DIP_COARSE_STEP
            continue

        # Fine-scan forward for a rapid dip within the compact window.
        min_energy = pre_energy
        dip_window_end = min(t + _GAP_DIP_MAX_DIP_WINDOW, scan_end)
        t_fine = t + _GAP_DIP_FINE_STEP
        while t_fine < dip_window_end:
            energy = _note_band_energy(
                audio, sample_rate, t_fine, frequency,
                window_seconds=MUTE_DIP_ENERGY_WINDOW,
            )
            if energy < min_energy:
                min_energy = energy
            t_fine += _GAP_DIP_FINE_STEP

        dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6)
        if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
            t += _GAP_DIP_COARSE_STEP
            continue

        # Dip confirmed — scan for recovery in the next window.
        recovery_end = min(dip_window_end + _GAP_DIP_MAX_RECOVERY_WINDOW, scan_end)
        t_fine = dip_window_end
        while t_fine < recovery_end:
            energy = _note_band_energy(
                audio, sample_rate, t_fine, frequency,
                window_seconds=MUTE_DIP_ENERGY_WINDOW,
            )
            if energy >= MUTE_DIP_REATTACK_MIN_POST_ENERGY:
                recovery_ratio = energy / (pre_energy + 1e-6)
                if recovery_ratio >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO:
                    return t_fine
            t_fine += _GAP_DIP_FINE_STEP

        # Dip found but no recovery — skip past this region.
        t += _GAP_DIP_COARSE_STEP

    return None


def rescue_gap_mute_dips(
    segments: list[Segment],
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
) -> list[Segment]:
    """Pass 1: create new segments for same-note re-attacks hidden in gaps.

    For each gap between consecutive segments, scans all tuning notes for a
    compact mute-dip pattern (high → rapid 100x drop → recovery within ~160 ms).
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

        # Scan all tuning notes; pre_energy filter eliminates non-ringing ones.
        best_recovery: float | None = None
        best_note: Note | None = None
        for note in tuning_notes:
            recovery = _scan_gap_for_mute_dip(
                audio, sample_rate, gap_start, gap_end, note.frequency,
            )
            if recovery is not None:
                if best_recovery is None or recovery < best_recovery:
                    best_recovery = recovery
                    best_note = note

        if best_recovery is not None and best_note is not None:
            seg_end = min(best_recovery + _GAP_DIP_DEFAULT_DURATION, gap_end)
            rescued.append(Segment(
                start_time=best_recovery,
                end_time=seg_end,
                sources=frozenset({"gap-mute-dip"}),
                confirmed_primary=best_note,
            ))

    if not rescued:
        return segments

    # Merge rescued segments into the original list, sorted by start_time.
    combined = list(segments) + rescued
    combined.sort(key=lambda s: s.start_time)
    return combined
