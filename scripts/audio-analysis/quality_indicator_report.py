#!/usr/bin/env python3
"""Validation report for the unsupervised quality indicators (internal, v1).

The quality indicators (apps/api/app/transcription/quality_indicators.py) are an
INTERNAL self-assessment signal, not a tester-facing feature. This report tests
whether they are actually predictive enough to be worth surfacing, BEFORE any UI
is built. It checks three things against the real corpus:

1. F1 correlation (the "F1 is corpus-dependent" motivation): does a higher
   difficulty score track lower per-recording note F1? If so, the indicator is a
   GT-free proxy for "is this transcription trustworthy".
2. Specificity on completed fixtures: clean, F1=1.0 captures must NOT be flagged
   red/yellow (false alarms would erode trust).
3. Triage-verdict alignment: do flags agree with human triage_verdicts.json
   (correct_detection -> green-ish; recording/recognizer issues -> non-green)?

Run after recognizer changes (the indicator reads recognizer output):
  uv run python scripts/audio-analysis/quality_indicator_report.py
  uv run python scripts/audio-analysis/quality_indicator_report.py --json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.main import app  # noqa: E402
from apps.api.app.transcription.audio import parse_tuning_json  # noqa: E402
from apps.api.app.transcription.quality_indicators import (  # noqa: E402
    compute_quality_indicators,
    peak_dbfs_of,
)
from apps.api.app.transcription.tuning_check import measure_selected_coverage  # noqa: E402
from note_f1_benchmark import (  # noqa: E402
    CAPTURES_DIR,
    DATA_DIR,
    discover_tx_ids,
    ground_truth_path_for,
    load_ground_truth,
    match_pairs,
    transaction_dir_for,
)

client = TestClient(app)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(path.read_bytes()), "rb") as w:
        sr, n, sw, ch = w.getframerate(), w.getnframes(), w.getsampwidth(), w.getnchannels()
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


def _transcribe(tx_dir: Path) -> dict:
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    r = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request["tuning"]), "debug": "false", "dryRun": "true", "force": "true"},
        files={"file": ("audio.wav", (tx_dir / "audio.wav").read_bytes(), "audio/wav")},
    )
    r.raise_for_status()
    return r.json()


def _indicators_for(tx_dir: Path):
    payload = _transcribe(tx_dir)
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    tuning = parse_tuning_json(json.dumps(request["tuning"]))
    audio, sr = _read_wav(tx_dir / "audio.wav")
    coverage = measure_selected_coverage(audio, sr, tuning)
    qi = compute_quality_indicators(
        payload.get("events", []), payload.get("candidateSlots"), coverage, peak_dbfs_of(audio)
    )
    return qi, payload


def _predicted_pairs(payload: dict) -> list[dict]:
    pairs = []
    for ev in payload.get("events", []):
        for n in ev["notes"]:
            pairs.append({"time": float(ev["startTimeSec"]), "note": f"{n['pitchClass']}{n['octave']}"})
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality-indicator validation report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out: dict = {"f1Correlation": [], "triageAlignment": [], "notes": []}

    # --- 1. F1 corpus correlation -----------------------------------------
    for tx_id in discover_tx_ids():
        tx_dir = transaction_dir_for(tx_id)
        if not (tx_dir / "audio.wav").is_file():
            continue
        qi, payload = _indicators_for(tx_dir)
        truth = load_ground_truth(ground_truth_path_for(tx_id))
        m = match_pairs(truth, _predicted_pairs(payload))
        out["f1Correlation"].append(
            {"txId": tx_id[:8], "f1": round(m["f1"], 4), "difficulty": qi.difficulty,
             "flag": qi.flag, "recQuality": qi.recording_quality, "recConf": qi.recognizer_confidence}
        )

    # --- 2. triage-verdict alignment --------------------------------------
    verdicts_path = CAPTURES_DIR / "triage_verdicts.json"
    sha_to_dir: dict[str, Path] = {}
    if verdicts_path.is_file() and DATA_DIR.is_dir():
        for d in DATA_DIR.iterdir():
            wav = d / "audio.wav"
            if wav.is_file():
                sha_to_dir[hashlib.sha256(wav.read_bytes()).hexdigest()] = d
        for v in json.loads(verdicts_path.read_text())["verdicts"]:
            tx_dir = sha_to_dir.get(v["audioSha256"])
            if tx_dir is None:
                continue
            qi, _ = _indicators_for(tx_dir)
            out["triageAlignment"].append(
                {"sha": v["audioSha256"][:8], "verdict": v["verdict"], "flag": qi.flag,
                 "difficulty": qi.difficulty}
            )

    # --- summary ----------------------------------------------------------
    fc = out["f1Correlation"]
    if len(fc) >= 2:
        diff = np.array([r["difficulty"] for r in fc])
        err = np.array([1.0 - r["f1"] for r in fc])
        if diff.std() > 0 and err.std() > 0:
            out["difficultyVsErrorCorr"] = round(float(np.corrcoef(diff, err)[0, 1]), 4)
        else:
            out["difficultyVsErrorCorr"] = None
            out["notes"].append("difficulty or error has zero variance (corpus too clean/uniform) — correlation undefined")
    clean = [r for r in fc if r["f1"] >= 0.999]
    out["cleanFixtureFalseAlarmRate"] = (
        round(sum(1 for r in clean if r["flag"] != "green") / len(clean), 4) if clean else None
    )

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"{'tx':10} {'F1':>6} {'difficulty':>10} {'flag':>7} {'recQual':>8} {'recConf':>8}")
    for r in fc:
        print(f"{r['txId']:10} {r['f1']:6.3f} {r['difficulty']:10.3f} {r['flag']:>7} {r['recQuality']:8.3f} {r['recConf']:8.3f}")
    print(f"\ndifficulty vs (1-F1) correlation: {out.get('difficultyVsErrorCorr')}")
    print(f"clean-fixture (F1>=0.999) false-alarm rate (non-green): {out['cleanFixtureFalseAlarmRate']}")
    if out["triageAlignment"]:
        print("\n--- triage verdict alignment ---")
        for r in out["triageAlignment"]:
            print(f"  {r['sha']} verdict={r['verdict']:18} flag={r['flag']:>7} difficulty={r['difficulty']:.3f}")
    for n in out["notes"]:
        print(f"\nNote: {n}")
    print(
        "\nReminder: corpus is small and mostly clean (F1~1.0, verdicts mostly correct_detection),"
        "\nso this report mainly checks specificity (no false alarms). Sensitivity (catching genuinely"
        "\nhard recordings) needs harder free-performance recordings (#18) before thresholds are trusted."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
