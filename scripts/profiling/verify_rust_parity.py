"""Compare Rust scan_gap_for_mute_dip_with_window against the current Python
implementation on every (gap, note, window) triple actually encountered in
bwv147 transcription. Reports divergences.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api" / "tests"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import kalimba_dsp  # noqa: E402

from app.transcription import per_note  # noqa: E402
from app.transcription.peaks import (  # noqa: E402
    MUTE_DIP_ENERGY_WINDOW,
    MUTE_DIP_REATTACK_MAX_DIP_RATIO,
    MUTE_DIP_REATTACK_MIN_POST_ENERGY,
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY,
    MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
    HARMONIC_BAND_CENTS,
)
from conftest import _transcribe_manual_capture_fixture  # noqa: E402

MAX_DIP = per_note._GAP_DIP_MAX_DIP_WINDOW
MAX_REC = per_note._GAP_DIP_MAX_RECOVERY_WINDOW
COARSE = per_note._GAP_DIP_COARSE_STEP
FINE = per_note._GAP_DIP_FINE_STEP

_py_impl = per_note._scan_gap_for_mute_dip_with_window

divergences: list[dict] = []


def _shadow(audio, sample_rate, gap_start, gap_end, frequency, window_seconds):
    py = _py_impl(audio, sample_rate, gap_start, gap_end, frequency, window_seconds)
    rs = kalimba_dsp.scan_gap_for_mute_dip_with_window(
        audio.astype(np.float32, copy=False),
        int(sample_rate), gap_start, gap_end, frequency, window_seconds,
        MUTE_DIP_ENERGY_WINDOW, MAX_DIP, MAX_REC, COARSE, FINE,
        MUTE_DIP_REATTACK_MIN_PRE_ENERGY, MUTE_DIP_REATTACK_MAX_DIP_RATIO,
        MUTE_DIP_REATTACK_MIN_POST_ENERGY, MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
        float(HARMONIC_BAND_CENTS),
    )
    # Consider float-LSB equality
    both_none = py is None and rs is None
    if not both_none:
        if py is None or rs is None or abs(py - rs) > 1e-9:
            divergences.append({
                "gap": (gap_start, gap_end), "freq": frequency, "win": window_seconds,
                "py": py, "rs": rs,
            })
    return py


per_note._scan_gap_for_mute_dip_with_window = _shadow

print("Running bwv147 with Rust-parity shadow ...", flush=True)
_transcribe_manual_capture_fixture.cache_clear()
_transcribe_manual_capture_fixture("kalimba-17-c-bwv147-sequence-163-01", True, None, True)

print(f"\n=== {len(divergences)} divergences ===")
for d in divergences[:20]:
    print(f"  gap=[{d['gap'][0]:.4f}, {d['gap'][1]:.4f}] f={d['freq']:.2f} win={d['win']}: py={d['py']} rs={d['rs']}")
if len(divergences) > 20:
    print(f"  ... {len(divergences) - 20} more")
