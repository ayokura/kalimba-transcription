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

Repo-managed corpus promotion (docs/corpus-management.md):
  uv run python scripts/audio-analysis/promote_corrections_to_ground_truth.py <tx-id> \
      --to-corpus --copyright-status original_performance \
      --rights-reviewed-by "human requester" [--device "..."] [--microphone "..."]

  --to-corpus scaffolds apps/api/tests/fixtures/free-performance-corpus/<tx-id>/
  (audio + request + corrections + review_status + ground_truth + metadata.json).
  The rights review itself stays a human decision: the script REFUSES to run
  without an explicit --copyright-status and --rights-reviewed-by, and simply
  records that decision in metadata.json. Add a baseline entry
  (note_f1_benchmark.py --write-baseline) and commit everything together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------
# Repo-managed corpus promotion (--to-corpus)
# ---------------------------------------------------------------------------

CLEARED_COPYRIGHT_STATUSES = ("original_performance", "public_domain")


def _recording_stats(audio_path: Path) -> dict:
    """Best-effort WAV stats for corpus metadata; every field may be None."""
    stats: dict = {
        "sampleRate": None,
        "channels": None,
        "durationSec": None,
        "peakDb": None,
        "rmsDb": None,
    }
    try:
        import array
        import math
        import wave

        with wave.open(str(audio_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            n_frames = wav.getnframes()
            stats["sampleRate"] = sample_rate
            stats["channels"] = channels
            if sample_rate:
                stats["durationSec"] = round(n_frames / sample_rate, 2)
            raw = wav.readframes(n_frames)
        typecode = {2: "h", 4: "i"}.get(sample_width)
        if typecode is None or not raw:
            return stats
        samples = array.array(typecode)
        samples.frombytes(raw)
        if not samples:
            return stats
        full_scale = float(2 ** (8 * sample_width - 1))
        peak = max(abs(s) for s in samples) / full_scale
        rms = math.sqrt(sum((s / full_scale) ** 2 for s in samples) / len(samples))
        if peak > 0:
            stats["peakDb"] = round(20 * math.log10(peak), 2)
        if rms > 0:
            stats["rmsDb"] = round(20 * math.log10(rms), 2)
    except Exception as exc:  # noqa: BLE001 — stats are optional metadata
        print(f"  WARN {audio_path.name}: recording stats unavailable: {exc}", file=sys.stderr)
    return stats


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_corpus_metadata(tx_id: str, args: argparse.Namespace) -> dict:
    src = DATA_DIR / tx_id
    request = _load_json(src / "request.json") or {}
    response = _load_json(src / "response.json") or {}
    corrections = _load_json(src / "corrections.json") or {}
    review_status = _load_json(src / "review_status.json") or {}

    events = corrections.get("events") or []
    origin_counts = Counter(e.get("origin", "recognizer") for e in events)
    response_events = response.get("events") or []
    candidate_slots = response.get("candidateSlots")
    if candidate_slots is None:
        candidate_slot_count = sum(len(e.get("candidateSlots") or []) for e in response_events)
    else:
        candidate_slot_count = len(candidate_slots)

    created_at = request.get("createdAt") or request.get("recordedAt")
    if not created_at:
        try:
            created_at = datetime.fromtimestamp(
                (src / "audio.wav").stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        except OSError:
            created_at = None
    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")

    work_title = args.work_title or (
        "Original free-form kalimba performance"
        if args.copyright_status == "original_performance"
        else "Public-domain work (see notes)"
    )

    return {
        "version": 1,
        "corpusType": "free-performance",
        "repositoryManaged": True,
        "transactionId": tx_id,
        "createdAt": created_at,
        "promotedAt": now_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "status": review_status.get("status"),
        "categories": [
            "tester_capture",
            "free_performance",
            "review_corrected",
            "repo_managed_corpus",
        ],
        "recording": {
            "device": args.device,
            "microphone": args.microphone,
            "deviceNotes": None,
            **_recording_stats(src / "audio.wav"),
        },
        "tuning": {
            "selectedId": (request.get("tuning") or {}).get("id"),
            "correctedId": None,
        },
        "aggregates": {
            "recognizerEventCount": len(response_events),
            "correctedEventCount": len(events),
            "candidateSlotCount": candidate_slot_count,
            "originCounts": dict(origin_counts),
        },
        "copyright": {
            "status": args.copyright_status,
            "workTitle": work_title,
            "publicDomainReason": args.public_domain_reason,
            "repositoryAllowed": True,
        },
        "timingAccuracy": {
            "noteIdentity": "reliable",
            "noteOrderAndCombination": "reliable",
            "onsetTiming": "approximate",
            "method": "user_corrected",
            "reason": (
                "Onset times come from the review UI: recognizer-origin events use "
                "the recognizer's detected event start (which itself can deviate "
                "slightly from the perceptual/spectral onset), inserted-manual "
                "events are hand-placed on the timeline, and inserted-slot events "
                "use a dropped-segment boundary."
            ),
            "safeFor": [
                "score-level evaluation",
                "note identity / order / combination evaluation",
            ],
            "cautionFor": [
                "timing-sensitive training",
                "tempo / rhythm modeling",
                "onset-time learning",
                "anything requiring spectral-grade onset accuracy",
            ],
            "refinement": (
                "Optional. An agent or human can re-verify onset times with "
                "spectral analysis and, if done, raise the per-onset method to "
                "spectrogram_verified / aubio_cross_checked. Not required unless "
                "a timing-sensitive use needs it."
            ),
        },
        "rightsReview": {
            "requiredForRepositoryManagement": True,
            "status": "approved_for_repository",
            "reviewedAt": today,
            "reviewedBy": args.rights_reviewed_by,
            "decision": "include_audio_and_teacher_data_in_repository",
            "reason": args.rights_reason,
        },
        "captureConsent": {
            "granted": True,
            "grantedBy": args.rights_reviewed_by,
            "grantedAt": today,
        },
        "sourceFiles": {
            "audio": "audio.wav",
            "request": "request.json",
            "corrections": "corrections.json",
            "reviewStatus": "review_status.json",
            "groundTruth": "ground_truth.json",
        },
        "notes": args.corpus_notes,
    }


def promote_to_corpus(tx_id: str, args: argparse.Namespace) -> bool:
    dest = CORPUS_DIR / tx_id
    if dest.exists() and not args.force_corpus:
        print(f"SKIP corpus {tx_id}: {dest.name}/ already exists — use --force-corpus to rebuild")
        return False
    gt_path = CAPTURES_DIR / tx_id / "ground_truth.json"
    if not gt_path.is_file():
        print(f"SKIP corpus {tx_id}: no ground_truth.json under transaction-captures")
        return False
    src = DATA_DIR / tx_id
    sources = {
        "audio.wav": src / "audio.wav",
        "request.json": src / "request.json",
        "corrections.json": src / "corrections.json",
        "review_status.json": src / "review_status.json",
    }
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        print(f"SKIP corpus {tx_id}: missing source files: {', '.join(missing)}")
        return False

    metadata = build_corpus_metadata(tx_id, args)
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        (dest / name).write_bytes(path.read_bytes())
    (dest / "ground_truth.json").write_bytes(gt_path.read_bytes())
    (dest / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        shown = dest.relative_to(REPO_ROOT)
    except ValueError:
        shown = dest
    print(
        f"CORPUS {shown}/ scaffolded"
        f" (copyright: {args.copyright_status}, rights review by: {args.rights_reviewed_by})"
    )
    return True


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
    parser.add_argument(
        "--to-corpus", action="store_true",
        help="also scaffold the repo-managed free-performance-corpus entry "
             "(requires --copyright-status and --rights-reviewed-by: the human "
             "rights decision is recorded, never assumed)",
    )
    parser.add_argument(
        "--copyright-status", choices=CLEARED_COPYRIGHT_STATUSES,
        help="cleared copyright status decided by the human rights review",
    )
    parser.add_argument(
        "--rights-reviewed-by",
        help="who made the rights decision (e.g. 'human requester')",
    )
    parser.add_argument(
        "--rights-reason",
        default="The requester confirmed this recording is useful and has no copyright issue.",
        help="short reason recorded in metadata.rightsReview.reason",
    )
    parser.add_argument("--work-title", help="metadata.copyright.workTitle override")
    parser.add_argument(
        "--public-domain-reason",
        help="required context when --copyright-status public_domain",
    )
    parser.add_argument("--device", help="recording device, when known (metadata.recording.device)")
    parser.add_argument("--microphone", help="microphone, when known (metadata.recording.microphone)")
    parser.add_argument("--corpus-notes", help="free-text metadata.notes")
    parser.add_argument(
        "--force-corpus", action="store_true",
        help="rebuild an existing free-performance-corpus/<tx-id>/ directory",
    )
    args = parser.parse_args()

    if args.to_corpus and (not args.copyright_status or not args.rights_reviewed_by):
        parser.error(
            "--to-corpus requires --copyright-status and --rights-reviewed-by "
            "(repo corpus inclusion needs an explicit human rights decision; "
            "see docs/corpus-management.md)"
        )

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
        reuse_existing_gt = gt_path.is_file() and not args.force
        if reuse_existing_gt:
            verified = " (human-verified)" if gt_is_human_verified(gt_path) else ""
            if not args.to_corpus:
                print(f"SKIP {tx_id}: ground_truth.json exists{verified} — use --force to overwrite")
                continue
            print(f"KEEP {tx_id}: existing ground_truth.json{verified} reused for corpus promotion")
        else:
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

        if args.to_corpus and not args.dry_run:
            promote_to_corpus(tx_id, args)

    if not args.dry_run:
        print(f"\nPromoted {promoted} recording(s)."
              " Run note_f1_benchmark.py --write-baseline to refresh the corpus baseline"
              + (" and commit the corpus scaffold together with it." if args.to_corpus else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
