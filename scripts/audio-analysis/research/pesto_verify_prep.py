"""Prep script for the PESTO verify session (S6 bets #3, #203 pre-fixed rules).

Converts docs/research/pesto-verify-candidates.json (the top-20
PESTO-unique candidates by confidence — notes PESTO detected that neither
the recognizer 1-best nor Basic Pitch nor the current GT has) into the
bp-verify UI row schema, so the human can run the same one-tap listening
session at /debug/bp-verify?set=pesto_verify.

Verdict rule (pre-fixed in #203, 2026-07-06): if fewer than 5 of the 20
candidates are judged real played notes, PESTO is dropped (bets #3
打ち切り確定). The verdict lands in data/gt_drafts/pesto_verify.verdict.json
via the UI; aggregation is the agent's job.

Per-row bandEnergy / noiseFloor / likelyAudible are measured the same way
as bp_verify_prep.py (kalimba_dsp.note_band_energy vs a median noise floor
sampled across the recording) — a pre-triage hint only, not a verdict.

Output: data/gt_drafts/pesto_verify.rows.json (gitignored)
    {generatedAt, rows: [{txId, tx8, timeSec, note, bandEnergy, noiseFloor,
                          energyRatio, likelyAudible}]}

Usage (from repo root):
    uv run python scripts/audio-analysis/research/pesto_verify_prep.py \
        [--data-dir data] [--candidates docs/research/pesto-verify-candidates.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import kalimba_dsp as K  # noqa: E402
from app.transcription.constants import HARMONIC_BAND_CENTS  # noqa: E402
from app.transcription.models import Note  # noqa: E402

# Measurement parameters mirror bp_verify_prep.py (see its comments for the
# rationale); kept in sync manually — both scripts are dev-only temporaries.
WINDOW_SECONDS = 0.05
NOISE_FLOOR_STEP_SEC = 0.2
LIKELY_AUDIBLE_RATIO = 3.0


def _load_audio(tx_dir: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(tx_dir / "audio.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _noise_floor(audio: np.ndarray, sr: int, frequency: float) -> float:
    duration_sec = len(audio) / sr
    n_steps = max(1, int(duration_sec / NOISE_FLOOR_STEP_SEC))
    samples = [
        K.note_band_energy(
            audio, sr, i * NOISE_FLOOR_STEP_SEC, frequency, WINDOW_SECONDS, HARMONIC_BAND_CENTS
        )
        for i in range(n_steps)
    ]
    return float(np.median(samples)) if samples else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "pesto-verify-candidates.json",
    )
    args = parser.parse_args()

    doc = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = doc["candidates"]

    rows: list[dict] = []
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    noise_floor_cache: dict[tuple[str, str], float] = {}
    for cand in candidates:
        tx_id = cand["tx"]
        time_sec = float(cand["t"])
        note_name = cand["note"]
        if tx_id not in audio_cache:
            audio_cache[tx_id] = _load_audio(args.data_dir / "transactions" / tx_id)
        audio, sr = audio_cache[tx_id]
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

    out_path = args.data_dir / "gt_drafts" / "pesto_verify.rows.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(rows)} rows, "
          f"{sum(r['likelyAudible'] for r in rows)} likelyAudible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
