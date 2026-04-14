"""Trace the exact divergent gap to find why old returns 5.4173 and new returns None."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api" / "tests"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.transcription.peaks import (  # noqa: E402
    MUTE_DIP_ENERGY_WINDOW,
    MUTE_DIP_REATTACK_MAX_DIP_RATIO,
    MUTE_DIP_REATTACK_MIN_POST_ENERGY,
    MUTE_DIP_REATTACK_MIN_PRE_ENERGY,
    MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO,
    _note_band_energy,
    batch_note_band_energies,
)

fx = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "manual-captures" / "kalimba-17-c-bwv147-sequence-163-01"
audio, sr = sf.read(fx / "audio.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1).astype(np.float32)

gap_start = 4.7173
gap_end = 5.4533
freq = 293.66
win = 0.05

# Reproduce old outer t progression
audio_duration = len(audio) / sr
scan_end = min(gap_end, audio_duration - win)
print(f"scan_end={scan_end:.6f}")

DIP = 0.06
RECOVERY = 0.10
COARSE = 0.02
FINE = 0.005

print(f"\n=== OLD (per-call) trace ===")
t = gap_start
iter_count = 0
while t < scan_end - DIP - RECOVERY:
    pre = _note_band_energy(audio, sr, t, freq, window_seconds=win)
    if pre < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
        t += COARSE
        iter_count += 1
        continue
    print(f"\n  iter {iter_count}: t={t:.6f} pre_energy={pre:.4f} (>= {MUTE_DIP_REATTACK_MIN_PRE_ENERGY})")
    min_energy = pre
    dip_window_end = min(t + DIP, scan_end)
    t_fine = t + FINE
    fines = []
    while t_fine < dip_window_end:
        e = _note_band_energy(audio, sr, t_fine, freq, window_seconds=MUTE_DIP_ENERGY_WINDOW)
        fines.append((t_fine, e))
        if e < min_energy:
            min_energy = e
        t_fine += FINE
    dip_ratio = (min_energy + 1e-6) / (pre + 1e-6)
    print(f"    dip min_energy={min_energy:.4f} dip_ratio={dip_ratio:.4f} (must < {MUTE_DIP_REATTACK_MAX_DIP_RATIO})")
    if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
        t += COARSE
        iter_count += 1
        continue
    # Recovery
    recovery_end = min(dip_window_end + RECOVERY, scan_end)
    t_fine = dip_window_end
    while t_fine < recovery_end:
        e = _note_band_energy(audio, sr, t_fine, freq, window_seconds=MUTE_DIP_ENERGY_WINDOW)
        if e >= MUTE_DIP_REATTACK_MIN_POST_ENERGY:
            rr = e / (pre + 1e-6)
            if rr >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO:
                print(f"    RECOVERY at t_fine={t_fine:.6f} energy={e:.4f} rr={rr:.4f}  ✓ RETURN")
                break
        t_fine += FINE
    else:
        print(f"    no recovery found")
        t += COARSE
        iter_count += 1
        continue
    break

# New (batched) trace at same outer iterations
print(f"\n=== NEW (batched) trace ===")
times = np.arange(gap_start, scan_end, FINE)
pre_e = batch_note_band_energies(audio, sr, times, freq, window_seconds=win)
fine_e = pre_e  # win == MUTE_DIP_ENERGY_WINDOW
n_fine = times.size
print(f"n_fine={n_fine}, last_time={times[-1]:.6f}")

dip_span = int(round(DIP / FINE))
recovery_span = int(round(RECOVERY / FINE))
coarse_stride = int(round(COARSE / FINE))
max_i = n_fine - dip_span - recovery_span
print(f"max_i={max_i}, dip_span={dip_span}, recovery_span={recovery_span}, coarse_stride={coarse_stride}")

i = 0
while i < max_i:
    pre = float(pre_e[i])
    if pre < MUTE_DIP_REATTACK_MIN_PRE_ENERGY:
        i += coarse_stride
        continue
    print(f"\n  i={i} t={times[i]:.6f} pre_energy={pre:.4f}")
    dip_end_idx = min(i + dip_span, n_fine)
    if dip_end_idx > i + 1:
        min_e = float(min(pre, float(fine_e[i+1:dip_end_idx].min())))
    else:
        min_e = pre
    dip_ratio = (min_e + 1e-6) / (pre + 1e-6)
    print(f"    dip min_energy={min_e:.4f} dip_ratio={dip_ratio:.4f}")
    if dip_ratio >= MUTE_DIP_REATTACK_MAX_DIP_RATIO:
        i += coarse_stride
        continue
    recovery_end_idx = min(dip_end_idx + recovery_span, n_fine)
    if recovery_end_idx > dip_end_idx:
        seg = fine_e[dip_end_idx:recovery_end_idx]
        post_ok = seg >= MUTE_DIP_REATTACK_MIN_POST_ENERGY
        rec_ok = (seg / (pre + 1e-6)) >= MUTE_DIP_REATTACK_MIN_RECOVERY_RATIO
        both = post_ok & rec_ok
        print(f"    recovery seg[:5]={seg[:5]}, post_ok any={np.any(post_ok)}, rec_ok any={np.any(rec_ok)}, both any={np.any(both)}")
        if np.any(both):
            first = int(np.argmax(both))
            print(f"    RECOVERY at t={times[dip_end_idx+first]:.6f}  ✓ RETURN")
            break
    i += coarse_stride
