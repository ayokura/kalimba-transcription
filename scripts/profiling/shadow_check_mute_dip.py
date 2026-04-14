"""Shadow-check: run old (per-call) and new (batched) implementations of
`_scan_gap_for_mute_dip_with_window` on the same inputs seen during bwv147
transcription and report divergences.

Strategy:
- Monkey-patch `per_note._scan_gap_for_mute_dip_with_window` to call both
  implementations, compare returned values, print the first divergent case
  with per-energy diff.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api" / "tests"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from app.transcription import per_note  # noqa: E402
from app.transcription.peaks import (  # noqa: E402
    MUTE_DIP_ENERGY_WINDOW,
    MUTE_DIP_ENERGY_WINDOW_NARROW,
    MUTE_DIP_REATTACK_MAX_DIP_RATIO,
    MUTE_DIP_REATTACK_MIN_POST_ENERGY,
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY,
    MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
    _note_band_energy,
)
from conftest import _transcribe_manual_capture_fixture  # noqa: E402

_GAP_DIP_COARSE_STEP = per_note._GAP_DIP_COARSE_STEP
_GAP_DIP_FINE_STEP = per_note._GAP_DIP_FINE_STEP
_GAP_DIP_MAX_DIP_WINDOW = per_note._GAP_DIP_MAX_DIP_WINDOW
_GAP_DIP_MAX_RECOVERY_WINDOW = per_note._GAP_DIP_MAX_RECOVERY_WINDOW


def _scan_original(audio, sample_rate, gap_start, gap_end, frequency, window_seconds):
    audio_duration = len(audio) / sample_rate
    scan_end = min(gap_end, audio_duration - window_seconds)
    if scan_end - gap_start < _GAP_DIP_MAX_DIP_WINDOW + _GAP_DIP_MAX_RECOVERY_WINDOW:
        return None
    t = gap_start
    while t < scan_end - _GAP_DIP_MAX_DIP_WINDOW - _GAP_DIP_MAX_RECOVERY_WINDOW:
        pre_energy = _note_band_energy(audio, sample_rate, t, frequency, window_seconds=window_seconds)
        if pre_energy < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
            t += _GAP_DIP_COARSE_STEP
            continue
        min_energy = pre_energy
        dip_window_end = min(t + _GAP_DIP_MAX_DIP_WINDOW, scan_end)
        t_fine = t + _GAP_DIP_FINE_STEP
        while t_fine < dip_window_end:
            energy = _note_band_energy(audio, sample_rate, t_fine, frequency, window_seconds=MUTE_DIP_ENERGY_WINDOW)
            if energy < min_energy:
                min_energy = energy
            t_fine += _GAP_DIP_FINE_STEP
        dip_ratio = (min_energy + 1e-6) / (pre_energy + 1e-6)
        if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
            t += _GAP_DIP_COARSE_STEP
            continue
        recovery_end = min(dip_window_end + _GAP_DIP_MAX_RECOVERY_WINDOW, scan_end)
        t_fine = dip_window_end
        while t_fine < recovery_end:
            energy = _note_band_energy(audio, sample_rate, t_fine, frequency, window_seconds=MUTE_DIP_ENERGY_WINDOW)
            if energy >= MUTE_DIP_REATTACK_MIN_POST_ENERGY:
                recovery_ratio = energy / (pre_energy + 1e-6)
                if recovery_ratio >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO:
                    return t_fine
            t_fine += _GAP_DIP_FINE_STEP
        t += _GAP_DIP_COARSE_STEP
    return None


divergences: list[dict] = []
_new_impl = per_note._scan_gap_for_mute_dip_with_window


def _shadow_scan(audio, sample_rate, gap_start, gap_end, frequency, window_seconds):
    old = _scan_original(audio, sample_rate, gap_start, gap_end, frequency, window_seconds)
    new = _new_impl(audio, sample_rate, gap_start, gap_end, frequency, window_seconds)
    if old != new:
        divergences.append(
            {
                "gap_start": gap_start,
                "gap_end": gap_end,
                "frequency": frequency,
                "window_seconds": window_seconds,
                "old": old,
                "new": new,
            }
        )
    return new


per_note._scan_gap_for_mute_dip_with_window = _shadow_scan

print("Running bwv147 with shadow check ...", flush=True)
_transcribe_manual_capture_fixture.cache_clear()
_transcribe_manual_capture_fixture("kalimba-17-c-bwv147-sequence-163-01", True, None, True)

print(f"\n=== {len(divergences)} divergences ===")
for d in divergences[:10]:
    print(f"  gap=[{d['gap_start']:.4f}, {d['gap_end']:.4f}] f={d['frequency']:.2f} "
          f"win={d['window_seconds']}: old={d['old']} new={d['new']}")
if len(divergences) > 10:
    print(f"  ... {len(divergences) - 10} more")
