#!/usr/bin/env python3
"""Calibrate tuning_check.py thresholds against the tester corpus.

For every unique (audio, tuning) pair under data/transactions/, print the
selected tuning's pitch-class coverage and the best alternative preset, so
MISMATCH_MAX_SELECTED_COVERAGE / SUGGESTION_MIN_COVERAGE_GAIN can be chosen
with real separation data.

Usage:
  uv run python scripts/audio-analysis/calibrate_tuning_mismatch.py
"""

from __future__ import annotations

import io
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from apps.api.app.transcription.tuning_check import (  # noqa: E402
    MIN_PEAKS,
    _coverage,
    _mean_power_spectrum,
    _pick_peaks,
    _pitch_class_weights,
    _tuning_frequency_range,
    _tuning_pitch_classes,
)
from apps.api.app.tunings import get_default_tunings  # noqa: E402
from apps.api.app.transcription.audio import parse_tuning_json  # noqa: E402

DATA_DIR = Path(os.environ.get("KALIMBA_DATA_DIR", str(REPO_ROOT / "data"))) / "transactions"


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(path.read_bytes()), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def main() -> int:
    seen: set[tuple[str, str]] = set()
    presets = get_default_tunings()
    rows = []
    for tx_dir in sorted(DATA_DIR.iterdir()):
        audio_path = tx_dir / "audio.wav"
        request_path = tx_dir / "request.json"
        if not (audio_path.is_file() and request_path.is_file()):
            continue
        request = json.loads(request_path.read_text(encoding="utf-8"))
        tuning = parse_tuning_json(json.dumps(request["tuning"]))
        import hashlib
        sha = hashlib.sha256(audio_path.read_bytes()).hexdigest()[:8]
        key = (sha, tuning.id)
        if key in seen:
            continue
        seen.add(key)

        audio, sr = read_wav(audio_path)
        fmin, fmax = _tuning_frequency_range(tuning)
        freqs, power = _mean_power_spectrum(audio, sr)
        peaks = _pick_peaks(freqs, power, fmin, fmax)
        weights, accepted = _pitch_class_weights(peaks)
        if accepted < MIN_PEAKS:
            rows.append((sha, tuning.id, None, None, None, f"skipped ({accepted} peaks)"))
            continue
        selected = _coverage(weights, _tuning_pitch_classes(tuning))
        best = max(
            ((_coverage(weights, _tuning_pitch_classes(c)), len(_tuning_pitch_classes(c)), c.id)
             for c in presets if c.id != tuning.id),
            key=lambda t: (t[0], -t[1]),
        )
        rows.append((sha, tuning.id, selected, best[2], best[0], f"{accepted} peaks"))

    print(f"{'hash':8}  {'selected tuning':18} {'cov':>7}  {'best alt':18} {'altcov':>7}  note")
    for sha, tid, sel, alt, altcov, note in sorted(rows, key=lambda r: (r[2] is None, r[2] or 0)):
        sel_s = f"{sel:.4f}" if sel is not None else "-"
        altcov_s = f"{altcov:.4f}" if altcov is not None else "-"
        print(f"{sha:8}  {tid:18} {sel_s:>7}  {alt or '-':18} {altcov_s:>7}  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
