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
    MUTE_DIP_ENERGY_WINDOW_NARROW,
    MUTE_DIP_REATTACK_MAX_DIP_RATIO,
    MUTE_DIP_REATTACK_MIN_POST_ENERGY,
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY,
    MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
    _note_band_energy,
    batch_note_band_energies,
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

    Pre-computes energies on the fine 5ms grid in batched form (one rfft for
    all grid points instead of per-point rfft) and replaces the original
    three-level loop with array lookups.  Bit-exact with the per-call form:
    same window, same n_fft, same peak_energy_near band mask.
    """
    audio_duration = len(audio) / sample_rate
    scan_end = min(gap_end, audio_duration - window_seconds)

    # Need room for at least dip_window + recovery_window.
    if scan_end - gap_start < _GAP_DIP_MAX_DIP_WINDOW + _GAP_DIP_MAX_RECOVERY_WINDOW:
        return None

    fine_step = _GAP_DIP_FINE_STEP
    times = np.arange(gap_start, scan_end, fine_step)
    n_fine = times.size
    if n_fine == 0:
        return None

    pre_energies = batch_note_band_energies(
        audio, sample_rate, times, frequency, window_seconds=window_seconds,
    )
    if window_seconds == MUTE_DIP_ENERGY_WINDOW:
        fine_energies = pre_energies
    else:
        fine_energies = batch_note_band_energies(
            audio, sample_rate, times, frequency, window_seconds=MUTE_DIP_ENERGY_WINDOW,
        )

    coarse_stride = max(int(round(_GAP_DIP_COARSE_STEP / fine_step)), 1)
    dip_span = int(round(_GAP_DIP_MAX_DIP_WINDOW / fine_step))
    recovery_span = int(round(_GAP_DIP_MAX_RECOVERY_WINDOW / fine_step))
    max_i = n_fine - dip_span - recovery_span

    i = 0
    while i < max_i:
        pre_energy = float(pre_energies[i])
        if pre_energy < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
            i += coarse_stride
            continue

        # Fine-scan forward for a rapid dip within [i+1, i+dip_span).
        dip_end_idx = min(i + dip_span, n_fine)
        if dip_end_idx > i + 1:
            min_energy = float(min(pre_energy, float(fine_energies[i + 1:dip_end_idx].min())))
        else:
            min_energy = pre_energy

        dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6)
        if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
            i += coarse_stride
            continue

        # Dip confirmed — scan for recovery in [dip_end_idx, recovery_end_idx).
        recovery_end_idx = min(dip_end_idx + recovery_span, n_fine)
        if recovery_end_idx > dip_end_idx:
            segment = fine_energies[dip_end_idx:recovery_end_idx]
            post_ok = segment >= MUTE_DIP_REATTACK_MIN_POST_ENERGY
            recovery_ok = (segment / (pre_energy + 1e-6)) >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO
            both = post_ok & recovery_ok
            if np.any(both):
                first = int(np.argmax(both))
                return float(times[dip_end_idx + first])

        # Dip found but no recovery — skip past this region.
        i += coarse_stride

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
