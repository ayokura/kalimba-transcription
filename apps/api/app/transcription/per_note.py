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
