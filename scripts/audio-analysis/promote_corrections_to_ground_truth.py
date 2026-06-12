#!/usr/bin/env python3
"""Promote review-UI corrections.json to F1-benchmark ground_truth.json.

The review UI (`/score/{id}/review`) saves user-corrected timelines as
`data/transactions/<tx-id>/corrections.json` (CorrectionsPayload schema,
absolute seconds + note names — deliberately ground_truth-compatible).
This tool closes the loop: it converts a reviewed timeline into
`apps/api/tests/fixtures/transaction-captures/<tx-id>/ground_truth.json`
so `note_f1_benchmark.py` picks it up as corpus.

Safety rails:
- Existing ground_truth.json is never overwritten without --force.
  Human-verified GT (ear_verified / spectrogram_verified) outranks
  user corrections, which are saved casually from the UI.
- Duplicate-audio detection: if another capture dir already has GT for the
  same audio SHA-256, the promotion is skipped (the benchmark would
  double-count the recording) unless --allow-duplicate is given.

Usage:
  uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py            # list candidates
  uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py <tx-id> ...
  uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py --all
  uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py <tx-id> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os  # noqa: E402

DATA_DIR = Path(os.environ.get("KALIMBA_DATA_DIR", str(REPO_ROOT / "data"))) / "transactions"
CAPTURES_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "transaction-captures"

DEFAULT_TOLERANCE_SEC = 0.05
# inserted-slot: startTime is the dropped segment's boundary, not a measured onset.
# inserted-manual: time is set by hand on the review timeline (least precise).
ORIGIN_TOLERANCE_SEC = {"inserted-slot": 0.08, "inserted-manual": 0.10}

HUMAN_VERIFIED_METHODS = {"ear_verified", "spectrogram_verified", "aubio_cross_checked"}


def audio_sha256(tx_id: str) -> str | None:
    path = DATA_DIR / tx_id / "audio.wav"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corrections(tx_id: str) -> dict | None:
    path = DATA_DIR / tx_id / "corrections.json"
    if not path.is_file():
        return None
    try:
        from apps.api.app.models import CorrectionsPayload

        return CorrectionsPayload.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        ).model_dump(by_alias=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN {tx_id}: corrections.json failed validation: {exc}", file=sys.stderr)
        return None


def existing_gt_hashes() -> dict[str, str]:
    """audio SHA-256 → tx-id for every capture dir that already has GT."""
    result: dict[str, str] = {}
    if not CAPTURES_DIR.is_dir():
        return result
    for d in sorted(CAPTURES_DIR.iterdir()):
        if not (d / "ground_truth.json").is_file():
            continue
        sha = audio_sha256(d.name)
        if sha:
            result[sha] = d.name
    return result


def gt_is_human_verified(path: Path) -> bool:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    methods = {o.get("method") for o in doc.get("onsets", [])}
    return bool(methods & HUMAN_VERIFIED_METHODS)


def build_ground_truth(tx_id: str, corrections: dict) -> dict:
    onsets = []
    origin_counts: dict[str, int] = {}
    for event in corrections["events"]:
        origin = event.get("origin", "recognizer")
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        onset: dict = {
            "timeSec": round(float(event["timeSec"]), 4),
            "notes": list(event["notes"]),
            "method": "user_corrected",
        }
        if origin in ORIGIN_TOLERANCE_SEC:
            onset["toleranceSec"] = ORIGIN_TOLERANCE_SEC[origin]
        if origin != "recognizer":
            onset["comment"] = f"origin={origin}"
        onsets.append(onset)
    return {
        "version": 1,
        "toleranceSec": DEFAULT_TOLERANCE_SEC,
        "source": {
            "type": "review-corrections",
            "transactionId": tx_id,
            "correctionsUpdatedAt": corrections.get("updatedAt"),
            "originCounts": origin_counts,
        },
        "onsets": onsets,
    }


def discover_candidates() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir() if (d / "corrections.json").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tx_ids", nargs="*", help="transaction IDs to promote")
    parser.add_argument("--all", action="store_true", help="promote every candidate")
    parser.add_argument("--dry-run", action="store_true", help="print GT without writing")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing ground_truth.json (including human-verified)",
    )
    parser.add_argument(
        "--allow-duplicate", action="store_true",
        help="promote even if another capture already has GT for the same audio",
    )
    args = parser.parse_args()

    candidates = discover_candidates()
    if not args.tx_ids and not args.all:
        if not candidates:
            print("No corrections.json found under", DATA_DIR)
            return 0
        print(f"{'tx':38} {'events':>6} {'updatedAt':25} GT?")
        for tx_id in candidates:
            corrections = load_corrections(tx_id)
            if corrections is None:
                continue
            has_gt = (CAPTURES_DIR / tx_id / "ground_truth.json").is_file()
            print(
                f"{tx_id:38} {len(corrections['events']):>6}"
                f" {corrections.get('updatedAt') or '-':25} {'yes' if has_gt else 'no'}"
            )
        print("\nRun with <tx-id> or --all to promote.")
        return 0

    targets = candidates if args.all else args.tx_ids
    gt_hashes = existing_gt_hashes()
    promoted = 0
    for tx_id in targets:
        corrections = load_corrections(tx_id)
        if corrections is None:
            print(f"SKIP {tx_id}: no valid corrections.json")
            continue

        gt_path = CAPTURES_DIR / tx_id / "ground_truth.json"
        if gt_path.is_file() and not args.force:
            verified = " (human-verified)" if gt_is_human_verified(gt_path) else ""
            print(f"SKIP {tx_id}: ground_truth.json exists{verified} — use --force to overwrite")
            continue

        sha = audio_sha256(tx_id)
        dup_tx = gt_hashes.get(sha) if sha else None
        if dup_tx and dup_tx != tx_id and not args.allow_duplicate:
            print(
                f"SKIP {tx_id}: same audio ({sha[:12]}) already has GT under {dup_tx}"
                " — use --allow-duplicate to add anyway"
            )
            continue

        gt = build_ground_truth(tx_id, corrections)
        if args.dry_run:
            print(f"--- {tx_id} (dry-run) ---")
            print(json.dumps(gt, ensure_ascii=False, indent=2))
            continue

        gt_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.write_text(
            json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if sha:
            gt_hashes[sha] = tx_id
        promoted += 1
        print(f"WROTE {gt_path.relative_to(REPO_ROOT)} ({len(gt['onsets'])} onsets)")

    if not args.dry_run:
        print(f"\nPromoted {promoted} recording(s)."
              " Run note_f1_benchmark.py to refresh the corpus baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
