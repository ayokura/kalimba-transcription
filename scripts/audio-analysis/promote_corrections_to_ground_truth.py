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
- Review-status gate: by default only recordings whose review_status.json is
  ``review_completed`` are promoted. A tester who only submitted a recording
  (``recorded_only``) or flagged it (``rerecord_needed`` / ``unusable`` /
  ``uncertain``) is NOT a ground-truth signal. Use --require-status to change
  the gate, or --ignore-status to bypass it (e.g. legacy corrections with no
  status file).

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
CORPUS_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "free-performance-corpus"

DEFAULT_TOLERANCE_SEC = 0.05
# inserted-slot: startTime is the dropped segment's boundary, not a measured onset.
# inserted-manual: time is set by hand on the review timeline (least precise).
ORIGIN_TOLERANCE_SEC = {"inserted-slot": 0.08, "inserted-manual": 0.10}

HUMAN_VERIFIED_METHODS = {"ear_verified", "spectrogram_verified", "aubio_cross_checked"}

# review_completed is the only status that means "the tester finished checking
# and the timeline reflects what was played" — the one signal worth promoting
# to a GT candidate by default.
DEFAULT_REQUIRED_STATUS = "review_completed"
VALID_REVIEW_STATUSES = {
    "recorded_only",
    "review_started",
    "review_completed",
    "rerecord_needed",
    "unusable",
    "uncertain",
}


def audio_sha256(tx_id: str) -> str | None:
    path = DATA_DIR / tx_id / "audio.wav"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_review_status(tx_id: str) -> str | None:
    path = DATA_DIR / tx_id / "review_status.json"
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    status = doc.get("status")
    return status if isinstance(status, str) else None


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
    """audio SHA-256 → tx-id for every capture/corpus dir that already has GT.

    Covers both GT layers that note_f1_benchmark.discover_tx_ids() reads:
    the repo-managed free-performance corpus (committed audio.wav) and the
    local transaction-captures (audio resolved via DATA_DIR). Without the
    corpus layer, a re-upload of already-promoted audio would get a second
    GT under a new tx-id and be double-counted by the benchmark.
    """
    result: dict[str, str] = {}
    if CORPUS_DIR.is_dir():
        for d in sorted(CORPUS_DIR.iterdir()):
            if not (d / "ground_truth.json").is_file():
                continue
            audio = d / "audio.wav"
            if audio.is_file():
                sha = hashlib.sha256(audio.read_bytes()).hexdigest()
                result[sha] = d.name
    if CAPTURES_DIR.is_dir():
        for d in sorted(CAPTURES_DIR.iterdir()):
            if not (d / "ground_truth.json").is_file():
                continue
            sha = audio_sha256(d.name)
            if sha:
                result.setdefault(sha, d.name)
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
    review_status = load_review_status(tx_id)
    return {
        "version": 1,
        "toleranceSec": DEFAULT_TOLERANCE_SEC,
        "source": {
            "type": "review-corrections",
            "transactionId": tx_id,
            "correctionsUpdatedAt": corrections.get("updatedAt"),
            "originCounts": origin_counts,
            "reviewStatus": review_status,
            # provenance tier: tester-confirmed timeline, NOT human-verified onset
            # annotation. promote_corrections never claims ear/spectrogram tiers.
            "provenance": "tester_corrected",
            "timingAccuracy": {
                "onsetTiming": "approximate",
                "noteIdentity": "review_corrected",
                "caveat": (
                    "Review UI corrections are reliable primarily for note identity, "
                    "ordering, and grouping. Onset times are approximate: recognizer "
                    "event starts can deviate from perceptual/spectral onsets, "
                    "inserted-slot times use dropped-segment boundaries, and "
                    "inserted-manual times are hand-placed. Use spectral/human "
                    "verification before timing-sensitive training or calibration."
                ),
            },
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
    parser.add_argument(
        "--require-status",
        default=DEFAULT_REQUIRED_STATUS,
        choices=sorted(VALID_REVIEW_STATUSES),
        help=f"only promote recordings with this review status (default: {DEFAULT_REQUIRED_STATUS})",
    )
    parser.add_argument(
        "--ignore-status", action="store_true",
        help="bypass the review-status gate (e.g. legacy corrections without review_status.json)",
    )
    args = parser.parse_args()

    candidates = discover_candidates()
    if not args.tx_ids and not args.all:
        if not candidates:
            print("No corrections.json found under", DATA_DIR)
            return 0
        print(f"{'tx':38} {'events':>6} {'reviewStatus':16} {'updatedAt':25} GT?")
        for tx_id in candidates:
            corrections = load_corrections(tx_id)
            if corrections is None:
                continue
            has_gt = (CAPTURES_DIR / tx_id / "ground_truth.json").is_file()
            status = load_review_status(tx_id) or "-"
            print(
                f"{tx_id:38} {len(corrections['events']):>6}"
                f" {status:16} {corrections.get('updatedAt') or '-':25} {'yes' if has_gt else 'no'}"
            )
        print(
            f"\nRun with <tx-id> or --all to promote."
            f" Default gate: review status == {DEFAULT_REQUIRED_STATUS}"
            f" (use --require-status / --ignore-status to change)."
        )
        return 0

    targets = candidates if args.all else args.tx_ids
    gt_hashes = existing_gt_hashes()
    promoted = 0
    for tx_id in targets:
        corrections = load_corrections(tx_id)
        if corrections is None:
            print(f"SKIP {tx_id}: no valid corrections.json")
            continue

        if not args.ignore_status:
            status = load_review_status(tx_id)
            if status != args.require_status:
                print(
                    f"SKIP {tx_id}: review status is {status or 'unset'},"
                    f" need {args.require_status} (use --ignore-status to bypass)"
                )
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
        try:
            shown = gt_path.relative_to(REPO_ROOT)
        except ValueError:
            shown = gt_path
        print(f"WROTE {shown} ({len(gt['onsets'])} onsets)")

    if not args.dry_run:
        print(f"\nPromoted {promoted} recording(s)."
              " Run note_f1_benchmark.py to refresh the corpus baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
