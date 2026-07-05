"""Prep script for the bp-only verify UI (#S2 GT decontamination, sprint-plan-2026-07c).

`basic_pitch_disagreement.py` classifies every GT onset into
both-hit / recognizer-only / bp-only / both-miss. **bp-only** = the GT says a
note is there but the recognizer did not detect it, while Basic Pitch (an
independent second opinion) did. These 23 rows (as of 2026-07-05) are the
`user_corrected` GT's highest-suspicion points: either a real recognizer miss
(keep in GT) or GT contamination from an over-eager review-UI correction
(remove from GT). A human must listen and decide — this script only prepares
the material and a rough pre-triage tag.

For each bp-only row this script measures the note-band energy (via
`kalimba_dsp.note_band_energy`, the exact quantity the recognizer scores —
±HARMONIC_BAND_CENTS peak FFT magnitude in a `window_seconds` window centered
on the onset time) against a per-recording, per-note noise floor (median of
the same measurement sampled across the whole recording). `likelyAudible`
flags rows where the onset-time energy clears the noise floor by a wide
margin, as a pre-triage hint for the /debug/bp-verify UI — NOT a verdict.

Output: data/gt_drafts/bp_verify.rows.json (gitignored)
    {generatedAt, rows: [{txId, tx8, timeSec, note, bandEnergy, noiseFloor,
                          energyRatio, likelyAudible}]}

Usage (from repo root):
    uv run python scripts/audio-analysis/research/bp_verify_prep.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import kalimba_dsp as K  # noqa: E402
from app.main import app  # noqa: E402
from app.transcription.constants import HARMONIC_BAND_CENTS  # noqa: E402
from app.transcription.models import Note  # noqa: E402

import basic_pitch_disagreement as bpd  # noqa: E402

OUT_PATH = REPO_ROOT / "data" / "gt_drafts" / "bp_verify.rows.json"

# note_band_energy window — matches the recognizer / GtEnergyTrace default
# (apps/web/src/lib/wasm/energy.ts WINDOW_SECONDS), so "audible" here means
# audible to the same measurement the recognizer itself uses.
WINDOW_SECONDS = 0.05

# Noise-floor sampling grid across the full recording (per note frequency).
NOISE_FLOOR_STEP_SEC = 0.2

# energyRatio = bandEnergy / max(noiseFloor, eps) threshold for the
# "likely audible, can fast-track" pre-triage hint. Chosen generously (an
# under-confident hint just means more rows get the "要精聴" bucket, which is
# safe — the human still listens to every row regardless of this flag).
LIKELY_AUDIBLE_RATIO = 3.0


def _load_audio(tx_id: str) -> tuple[np.ndarray, int]:
    tx_dir = bpd.tx_dir_for(tx_id)
    audio, sr = sf.read(tx_dir / "audio.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _noise_floor(audio: np.ndarray, sr: int, frequency: float) -> float:
    """Median note-band energy for `frequency` sampled across the whole
    recording. Median (not min) is robust to the few real onsets of that
    note among many silent/other-note samples."""
    duration_sec = len(audio) / sr
    n_steps = max(1, int(duration_sec / NOISE_FLOOR_STEP_SEC))
    samples = []
    for i in range(n_steps):
        t = i * NOISE_FLOOR_STEP_SEC
        samples.append(
            K.note_band_energy(audio, sr, t, frequency, WINDOW_SECONDS, HARMONIC_BAND_CENTS)
        )
    return float(np.median(samples)) if samples else 0.0


def main() -> int:
    client = TestClient(app)
    rows: list[dict] = []
    # Cache noise floors per (tx_id, note) since many bp-only rows in the
    # same recording repeat notes (e.g. a9e30986 has 8 x C4).
    noise_floor_cache: dict[tuple[str, str], float] = {}
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for tx_id in bpd.TX_IDS:
        result = bpd.classify_recording(tx_id, client)
        bp_only = result["bp_only"]
        if not bp_only:
            continue
        if tx_id not in audio_cache:
            audio_cache[tx_id] = _load_audio(tx_id)
        audio, sr = audio_cache[tx_id]
        for entry in bp_only:
            time_sec = float(entry["timeSec"])
            note_name = entry["note"]
            frequency = Note.from_name(note_name).frequency
            cache_key = (tx_id, note_name)
            if cache_key not in noise_floor_cache:
                noise_floor_cache[cache_key] = _noise_floor(audio, sr, frequency)
            noise_floor = noise_floor_cache[cache_key]
            band_energy = float(
                K.note_band_energy(audio, sr, time_sec, frequency, WINDOW_SECONDS, HARMONIC_BAND_CENTS)
            )
            energy_ratio = band_energy / max(noise_floor, 1e-9)
            rows.append(
                {
                    "txId": tx_id,
                    "tx8": tx_id[:8],
                    "timeSec": time_sec,
                    "note": note_name,
                    "bandEnergy": band_energy,
                    "noiseFloor": noise_floor,
                    "energyRatio": energy_ratio,
                    "likelyAudible": energy_ratio >= LIKELY_AUDIBLE_RATIO,
                }
            )

    rows.sort(key=lambda r: (r["tx8"], r["timeSec"]))

    doc = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    n_audible = sum(1 for r in rows if r["likelyAudible"])
    print(f"Wrote {len(rows)} bp-only rows to {OUT_PATH}")
    print(f"  likelyAudible=True (fast-track candidates): {n_audible}")
    print(f"  likelyAudible=False (要精聴):                {len(rows) - n_audible}")
    for r in rows:
        tag = "audible" if r["likelyAudible"] else "unclear"
        print(
            f"  {r['tx8']} {r['timeSec']:8.3f}s {r['note']:4s} "
            f"ratio={r['energyRatio']:7.2f} [{tag}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
