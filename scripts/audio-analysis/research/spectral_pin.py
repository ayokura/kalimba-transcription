"""Spectral onset pinning for GT recordings (third-term S3, guardrail 13).

GT timeSec is approximate (raw-onset backtrack / hand placement); timing-
sensitive work (phase-tracking ROC calibration, timing calibration #173/#174)
needs a defined, computable anchor. This script pins each GT note to the
per-note band energy rise and records the offset vs the GT time.

Operational semantics (v1, recorded in #201):
- For each (onset, note): compute the note-band energy trace E(t)
  (STFT n_fft=2048 hop=256 Hann; band = f0 +- max(1.5 bins, 30 cents)).
- Search window: gtTime +- max(row toleranceSec, 0.12 s).
- floor = median E over [gt-0.35, gt-0.15] (clamped to window start);
  peak = max E in the search window at/after the first sample.
- **pin = the last time before the peak where E crosses 10% of
  (peak - floor) above floor** (the "10% rise point", anchor (a) of #201
  with a rise-confirmation), searched left from the peak.
- t_maxslope (anchor (b): max positive slope of log E) is recorded too so
  the (a)-(b) spread is measurable per recording.
- If peak < floor + 6 dB the note is "unpinnable" (masked / low SNR /
  re-strike of a still-ringing tine without a clean rise) and must be
  excluded from timing-sensitive evaluation rather than trusted.
- Chords: notes are pinned independently (glissando/strum-safe); no
  row-level single anchor is asserted.

Output: spectral_pins.json next to the recording's ground_truth.json
(local-only when the GT itself is local-only), plus a stdout summary of
pin-vs-GT offsets.

Usage:
  uv run python scripts/audio-analysis/research/spectral_pin.py 70cc6637 47902d34
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tine_partial_collision_probe import (  # noqa: E402
    GT_SOURCES,
    REPO,
    audio_for,
    load_audio,
    resolve_source,
)

N_FFT = 2048
HOP = 256
SEARCH_PAD_SEC = 0.12
FLOOR_WIN = (-0.35, -0.15)   # relative to GT time
RISE_FRAC = 0.10
MIN_RISE_DB = 6.0


def gt_path_for(tx_full: str) -> Path | None:
    for base, _ in GT_SOURCES:
        p = REPO / base / tx_full / "ground_truth.json"
        if p.is_file():
            return p
    return None


def resolve_tx(prefix: str) -> str | None:
    for base, _ in GT_SOURCES:
        for d in (REPO / base).iterdir():
            if d.name.startswith(prefix) and (d / "ground_truth.json").is_file():
                return d.name
    return None


def band_trace(audio: np.ndarray, sr: int, f0: float) -> tuple[np.ndarray, np.ndarray]:
    """(times, band energy) via short STFT around f0."""
    n_frames = 1 + max(0, (len(audio) - N_FFT)) // HOP
    win = np.hanning(N_FFT)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    bin_hz = sr / N_FFT
    half = max(1.5 * bin_hz, f0 * (2 ** (30 / 1200) - 1))
    m = (freqs >= f0 - half) & (freqs <= f0 + half)
    e = np.empty(n_frames)
    for i in range(n_frames):
        seg = audio[i * HOP:i * HOP + N_FFT]
        spec = np.abs(np.fft.rfft(seg * win))
        e[i] = float(np.sum(spec[m] ** 2))
    times = (np.arange(n_frames) * HOP + N_FFT / 2) / sr
    return times, e


def pin_note(times: np.ndarray, e: np.ndarray, t_gt: float, tol: float) -> dict:
    lo, hi = t_gt - max(tol, SEARCH_PAD_SEC), t_gt + max(tol, SEARCH_PAD_SEC)
    w = (times >= lo) & (times <= hi)
    if not w.any():
        return {"status": "no_window"}
    fw = (times >= t_gt + FLOOR_WIN[0]) & (times <= t_gt + FLOOR_WIN[1])
    floor = float(np.median(e[fw])) if fw.any() else float(np.min(e[w]))
    idx = np.where(w)[0]
    pk_rel = int(np.argmax(e[idx]))
    pk_i = idx[pk_rel]
    peak = float(e[pk_i])
    rise_db = 10 * np.log10((peak + 1e-12) / (floor + 1e-12))
    if rise_db < MIN_RISE_DB:
        return {"status": "unpinnable", "riseDb": round(rise_db, 1)}
    thresh = floor + RISE_FRAC * (peak - floor)
    i = pk_i
    while i > idx[0] and e[i - 1] > thresh:
        i -= 1
    t_pin = float(times[i])
    # anchor (b): max positive slope of log energy up to the peak
    seg = np.log10(e[idx[0]:pk_i + 1] + 1e-12)
    t_slope = float(times[idx[0] + int(np.argmax(np.diff(seg)))]) if len(seg) > 1 else t_pin
    return {
        "status": "pinned",
        "pinSec": round(t_pin, 4),
        "maxSlopeSec": round(t_slope, 4),
        "riseDb": round(rise_db, 1),
        "deltaMs": round((t_pin - t_gt) * 1000, 1),
    }


def process(tx_prefix: str) -> None:
    tx = resolve_tx(tx_prefix)
    if tx is None:
        print(f"{tx_prefix}: no GT found"); return
    src = resolve_source(tx)
    ap = audio_for(tx)
    if src is None or ap is None:
        print(f"{tx_prefix}: missing request.json/audio"); return
    _tuning, group, freqs_map = src
    audio, sr = load_audio(ap)
    gt = json.loads(gt_path_for(tx).read_text())
    default_tol = float(gt.get("toleranceSec", 0.08))
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    pins, n_pinned, deltas = [], 0, []
    for i, onset in enumerate(gt["onsets"]):
        t_gt = float(onset["timeSec"])
        tol = float(onset.get("toleranceSec", default_tol))
        for note in onset.get("notes") or []:
            if note not in freqs_map:
                pins.append({"index": i, "note": note, "gtTimeSec": t_gt, "status": "unknown_note"})
                continue
            if note not in traces:
                traces[note] = band_trace(audio, sr, freqs_map[note])
            r = pin_note(*traces[note], t_gt, tol)
            pins.append({"index": i, "note": note, "gtTimeSec": t_gt, **r})
            if r["status"] == "pinned":
                n_pinned += 1
                deltas.append(r["deltaMs"])
    out = {
        "version": 1,
        "semantics": "per-note band-energy 10% rise point (v1, #201); maxSlopeSec = anchor (b)",
        "generator": "scripts/audio-analysis/research/spectral_pin.py",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "group": group,
        "params": {"nFft": N_FFT, "hop": HOP, "riseFrac": RISE_FRAC, "minRiseDb": MIN_RISE_DB,
                   "searchPadSec": SEARCH_PAD_SEC, "floorWinSec": list(FLOOR_WIN)},
        "pins": pins,
    }
    dest = gt_path_for(tx).parent / "spectral_pins.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    d = np.array(deltas) if deltas else np.array([0.0])
    unpin = sum(1 for p in pins if p["status"] == "unpinnable")
    print(f"{tx[:8]} [{group}]: notes={len(pins)} pinned={n_pinned} unpinnable={unpin} "
          f"delta(pin-gt)ms median={np.median(d):+.1f} p10={np.percentile(d,10):+.1f} p90={np.percentile(d,90):+.1f}"
          f" -> {dest.relative_to(REPO)}")


def main() -> int:
    for prefix in sys.argv[1:] or ["70cc6637", "47902d34"]:
        process(prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
