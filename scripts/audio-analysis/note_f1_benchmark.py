#!/usr/bin/env python3
"""Note-level F1 benchmark for free-performance recordings.

Evaluates the currently-loaded recognizer against human-verified ground truth
(`ground_truth.json`, AGENTS.md schema) instead of exact event-sequence match.
This is the free-performance counterpart of the fixture regression suite:
fixtures assert "the transcription equals the score", this benchmark measures
"how close the transcription is to what was physically played" via note-level
precision / recall / F1.

Ground truth discovery (default):
  apps/api/tests/fixtures/transaction-captures/<tx-id>/ground_truth.json
with audio + tuning taken from data/transactions/<tx-id>/.

Usage:
  uv run python scripts/audio-analysis/note_f1_benchmark.py            # all GT
  uv run python scripts/audio-analysis/note_f1_benchmark.py <tx-id> ...
  uv run python scripts/audio-analysis/note_f1_benchmark.py --json
  uv run python scripts/audio-analysis/note_f1_benchmark.py --verbose  # FP/FN 明細

Matching: each ground-truth (timeSec, note) pair is matched one-to-one to the
nearest predicted (startTimeSec, note) pair with the same note name within
toleranceSec (per-onset override supported). Unmatched predicted pairs are
false positives, unmatched ground-truth pairs are false negatives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from apps.api.app.fingerprints import recognizer_fingerprint  # noqa: E402
from apps.api.app.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("KALIMBA_DATA_DIR", str(REPO_ROOT / "data"))) / "transactions"
CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"

DEFAULT_TOLERANCE_SEC = 0.05


def load_ground_truth(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    default_tol = float(doc.get("toleranceSec", DEFAULT_TOLERANCE_SEC))
    pairs: list[dict] = []
    for onset in doc.get("onsets", []):
        tol = float(onset.get("toleranceSec", default_tol))
        for note in onset["notes"]:
            pairs.append({"time": float(onset["timeSec"]), "note": note, "tol": tol})
    return pairs


def transcribe(client: TestClient, tx_id: str) -> list[dict]:
    tx_dir = DATA_DIR / tx_id
    audio_bytes = (tx_dir / "audio.wav").read_bytes()
    request = json.loads((tx_dir / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(request["tuning"]),
            "debug": "false",
            "dryRun": "true",
            "force": "true",
        },
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    response.raise_for_status()
    pairs: list[dict] = []
    for event in response.json()["events"]:
        for note in event["notes"]:
            pairs.append(
                {
                    "time": float(event["startTimeSec"]),
                    "note": f"{note['pitchClass']}{note['octave']}",
                }
            )
    return pairs


def match_pairs(truth: list[dict], predicted: list[dict]) -> dict:
    used = [False] * len(predicted)
    matched: list[tuple[dict, dict]] = []
    false_negatives: list[dict] = []
    for gt in sorted(truth, key=lambda p: p["time"]):
        best_index = -1
        best_dt = None
        for i, pred in enumerate(predicted):
            if used[i] or pred["note"] != gt["note"]:
                continue
            dt = abs(pred["time"] - gt["time"])
            if dt > gt["tol"]:
                continue
            if best_dt is None or dt < best_dt:
                best_index, best_dt = i, dt
        if best_index >= 0:
            used[best_index] = True
            matched.append((gt, predicted[best_index]))
        else:
            false_negatives.append(gt)
    false_positives = [pred for i, pred in enumerate(predicted) if not used[i]]
    tp = len(matched)
    precision = tp / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "truthNotes": len(truth),
        "predictedNotes": len(predicted),
        "tp": tp,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def discover_tx_ids() -> list[str]:
    if not CAPTURES_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in CAPTURES_DIR.iterdir()
        if (d / "ground_truth.json").is_file() and (DATA_DIR / d.name / "audio.wav").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Note-level F1 benchmark")
    parser.add_argument("tx_ids", nargs="*", help="transaction IDs (default: all with ground_truth.json)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--verbose", action="store_true", help="list FP/FN pairs per recording")
    args = parser.parse_args()

    tx_ids = args.tx_ids or discover_tx_ids()
    if not tx_ids:
        print("No ground_truth.json found under", CAPTURES_DIR, file=sys.stderr)
        return 1

    client = TestClient(app)
    results = []
    total = {"tp": 0, "truth": 0, "predicted": 0}
    for tx_id in tx_ids:
        truth = load_ground_truth(CAPTURES_DIR / tx_id / "ground_truth.json")
        predicted = transcribe(client, tx_id)
        outcome = match_pairs(truth, predicted)
        outcome["txId"] = tx_id
        results.append(outcome)
        total["tp"] += outcome["tp"]
        total["truth"] += outcome["truthNotes"]
        total["predicted"] += outcome["predictedNotes"]

    micro_p = total["tp"] / total["predicted"] if total["predicted"] else 0.0
    micro_r = total["tp"] / total["truth"] if total["truth"] else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    summary = {
        "recognizerFingerprint": recognizer_fingerprint()[:16],
        "recordings": len(results),
        "microPrecision": micro_p,
        "microRecall": micro_r,
        "microF1": micro_f1,
    }

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2))
        return 0

    print(f"{'tx':38} {'GT':>4} {'pred':>4} {'TP':>4} {'P':>6} {'R':>6} {'F1':>6}")
    for r in results:
        print(
            f"{r['txId'][:36]:38} {r['truthNotes']:>4} {r['predictedNotes']:>4} {r['tp']:>4}"
            f" {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f}"
        )
        if args.verbose:
            for fp in r["falsePositives"]:
                print(f"    FP {fp['time']:8.3f}s {fp['note']}")
            for fn in r["falseNegatives"]:
                print(f"    FN {fn['time']:8.3f}s {fn['note']}")
    print(
        f"\nmicro P={summary['microPrecision']:.3f} R={summary['microRecall']:.3f}"
        f" F1={summary['microF1']:.3f}  ({summary['recordings']} recordings,"
        f" recognizer {summary['recognizerFingerprint']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
