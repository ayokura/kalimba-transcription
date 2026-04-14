"""Verify batch_note_band_energies produces bit-identical results to
per-call _note_band_energy on the specific gap that diverged.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api" / "tests"))
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.transcription.peaks import (  # noqa: E402
    _note_band_energy,
    batch_note_band_energies,
)

fx = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "manual-captures" / "kalimba-17-c-bwv147-sequence-163-01"
audio, sr = sf.read(fx / "audio.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1).astype(np.float32)

# Gap from the first divergence
gap_start = 4.7173
gap_end = 5.4533
freq = 293.66  # D4
win = 0.05

fine = 0.005
times = np.arange(gap_start, gap_end, fine)
print(f"gap [{gap_start}, {gap_end}) freq={freq} win={win}")
print(f"times.size={times.size}, first={times[0]:.6f}, last={times[-1]:.6f}")

batched = batch_note_band_energies(audio, sr, times, freq, window_seconds=win)
per_call = np.array([_note_band_energy(audio, sr, float(t), freq, window_seconds=win) for t in times])

diffs = np.abs(batched - per_call)
max_diff = float(diffs.max())
n_diff = int((diffs > 0).sum())
print(f"max_abs_diff={max_diff:.6e}, num_diff={n_diff}/{len(times)}")

if n_diff > 0:
    idx = int(np.argmax(diffs))
    print(f"worst at times[{idx}]={times[idx]:.6f}: batch={batched[idx]:.6e} per_call={per_call[idx]:.6e}")
    for i in range(max(0, idx-2), min(len(times), idx+3)):
        print(f"  t={times[i]:.6f} batch={batched[i]:.6e} per={per_call[i]:.6e} diff={diffs[i]:.3e}")
