#!/usr/bin/env python3
"""Join free-performance benchmark metrics with review-queue status.

The note-level benchmark (note_f1_benchmark.py) measures HOW WRONG the
recognizer is per recording; the review status (review_status.json) records
WHERE that recording sits in the human collection workflow. This report joins
the two so reviewers can answer: "which recordings should I check next to
improve the recognizer the most?".

A recording is high priority when it BOTH (a) still needs human attention
(status is recorded_only / review_started / uncertain / unset) AND (b) the
recognizer struggles on it (low onset F1, high hard-miss rate, high correction
burden). Recordings already ``review_completed`` or flagged ``unusable`` /
``rerecord_needed`` are de-prioritized — they are not where review effort pays
off next.

Usage:
  uv run python scripts/audio-analysis/review_priority_report.py
  uv run python scripts/audio-analysis/review_priority_report.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Statuses that still want human attention (review effort pays off).
OPEN_STATUSES = {None, "recorded_only", "review_started", "uncertain"}
# Statuses that should not be re-surfaced as "needs review next".
CLOSED_STATUSES = {"review_completed", "unusable", "rerecord_needed"}


def compute_priority(metrics: dict, review_status: str | None) -> dict:
    """Pure priority scorer (no IO) so it is unit-testable.

    ``metrics`` uses the note_f1_benchmark per-recording shape:
      { "oneBest": {"onsetF1": float},
        "candidates": {"hardMissRate": float},
        "correction": {"costPerTruthNote": float} }

    Returns a dict with the numeric score and a short reason. Higher score =
    more worth reviewing next.
    """
    one_best = metrics.get("oneBest") or {}
    candidates = metrics.get("candidates") or {}
    correction = metrics.get("correction") or {}

    onset_f1 = float(one_best.get("onsetF1", 1.0) or 0.0)
    hard_miss_rate = float(candidates.get("hardMissRate", 0.0) or 0.0)
    cost_per_truth = float(correction.get("costPerTruthNote", 0.0) or 0.0)

    # Recognizer difficulty in [0, ~]: F1 gap dominates, hard misses are the
    # heaviest correction action, cost-per-note is a normalized burden proxy.
    difficulty = (1.0 - onset_f1) + 2.0 * hard_miss_rate + min(cost_per_truth, 2.0)

    is_open = review_status in OPEN_STATUSES
    # Closed recordings keep a residual (so they are still listed) but never
    # outrank an open recording with the same difficulty.
    openness = 1.0 if is_open else 0.2
    score = round(difficulty * openness, 4)

    reasons: list[str] = []
    if onset_f1 < 0.999:
        reasons.append(f"F1={onset_f1:.3f}")
    if hard_miss_rate > 0:
        reasons.append(f"hardMiss={hard_miss_rate:.3f}")
    if cost_per_truth > 0:
        reasons.append(f"cost/GT={cost_per_truth:.3f}")
    if not is_open:
        reasons.append(f"status={review_status}")
    return {
        "priority": score,
        "difficulty": round(difficulty, 4),
        "open": is_open,
        "reason": ", ".join(reasons) or "clean",
    }


def _load_review_status(data_dir: Path, tx_id: str) -> str | None:
    path = data_dir / tx_id / "review_status.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = doc.get("status")
    return status if isinstance(status, str) else None


def build_rows() -> list[dict]:
    from fastapi.testclient import TestClient

    from apps.api.app.main import app
    import note_f1_benchmark as bench

    client = TestClient(app)
    rows: list[dict] = []
    for tx_id in bench.discover_tx_ids():
        truth = bench.load_ground_truth(bench.ground_truth_path_for(tx_id))
        payload = bench.transcribe_payload(client, tx_id, debug=True)
        metrics = bench.evaluate_payload(payload, truth)
        review_status_path = bench.review_status_path_for(tx_id)
        if review_status_path is not None:
            try:
                status_doc = json.loads(review_status_path.read_text(encoding="utf-8"))
                status = status_doc.get("status") if isinstance(status_doc, dict) else None
            except (OSError, json.JSONDecodeError):
                status = None
        else:
            status = None
        priority = compute_priority(metrics, status)
        rows.append(
            {
                "txId": tx_id,
                "reviewStatus": status,
                "onsetF1": metrics["oneBest"]["onsetF1"],
                "hardMissRate": metrics["candidates"]["hardMissRate"],
                "costPerTruthNote": metrics["correction"]["costPerTruthNote"],
                **priority,
            }
        )
    rows.sort(key=lambda r: r["priority"], reverse=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    rows = build_rows()
    if not rows:
        print("No ground_truth.json corpus found — nothing to prioritize.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"rows": rows}, indent=2))
        return 0

    print(f"{'tx':38} {'status':16} {'F1':>6} {'hard':>6} {'cost':>6} {'prio':>6}  reason")
    for r in rows:
        print(
            f"{r['txId'][:36]:38} {str(r['reviewStatus'] or '-'):16}"
            f" {r['onsetF1']:6.3f} {r['hardMissRate']:6.3f} {r['costPerTruthNote']:6.3f}"
            f" {r['priority']:6.3f}  {r['reason']}"
        )
    print(
        "\nHigh priority = still open for review AND recognizer struggles."
        " review_completed / unusable / rerecord_needed are de-prioritized."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
