#!/usr/bin/env python3
"""Bulk re-recognition across local transactions (#204 Phase 3).

Issue #204's core problem: force=true re-uploading a stored recording used to
mint a brand-new transaction, so re-recognizing many recordings after a
recognizer change produced duplicate transactions instead of a tracked
history. Phase 1 fixed the single-recording case with
``POST /api/transcriptions/{id}/runs`` (append a run, no new transaction id).
This script is the "do that for every recording that needs it" tool the
issue's Phase 3 section calls for ("corpus 一括再認識ツール"), calling the
same endpoint in-process (via ``fastapi.testclient.TestClient``, no server
needed) for every transaction under a data directory whose saved recognizer
fingerprint no longer matches the recognizer currently loaded.

Before/after comparison: reuses ``note_f1_benchmark.collect_one_best`` /
``match_pairs`` exactly like ``metamorphic_alarm.py`` does, treating the
previously-resolved response as a pseudo ground truth and the freshly
created run's output as "predicted" — so "false negatives" are notes the new
run dropped and "false positives" are notes the new run added relative to
what was there before. This is the same shape of before/after evaluation the
per-tine research line's dual-run comparisons want (AGENTS.md, #204 issue
body), reused rather than reimplemented.

Staleness default vs. the review-queue badge: the review queue
(GET /api/review-queue, ``app.storage.resolved_recognizer_fingerprint``)
treats an unknown saved fingerprint (pre-#204 recordings) as "isStale: None"
— it deliberately does not guess, because a wrong guess would mislead a
human triaging the queue. This tool's *default* target set is broader on
purpose: "verified current" (saved fingerprint == running recognizer) is
excluded, and everything else (fingerprint differs OR fingerprint unknown)
is a candidate, because re-running is cheap, append-only, and safe, and
backfilling provenance for old recordings is one of the useful side effects
of running this tool at all. Pass ``--all`` to bypass the filter entirely
(re-run every matching transaction regardless of freshness) or ``--tx`` to
restrict to specific ids.

Data dir: like the API server itself, this reads ``KALIMBA_DATA_DIR`` (via
``app.storage``'s own env lookup, re-evaluated on every call) or defaults to
`./data`. Pass ``--data-dir`` to point at a scratch/synthetic directory
instead of touching real local data — this is REQUIRED for any test/demo run
of this script per AGENTS.md ("一括再認識の実行テストは必ず一時ディレクトリ
の合成データで行う"); never point it at the shared repo `data/` directory or
at `apps/api/tests/fixtures/free-performance-corpus/` from an automated
run.

REPORT + PERSIST (not report-only): unlike metamorphic_alarm.py, this tool
actually appends recognition runs (unless --dry-run is passed). It has no
CI/pytest wiring and is not a regression gate; it is an operator tool for
"my recognizer changed, bring my locally-recorded corpus up to date."

Usage:
  uv run python scripts/audio-analysis/bulk_recognition_runs.py --dry-run
  uv run python scripts/audio-analysis/bulk_recognition_runs.py --data-dir /tmp/scratch-data
  uv run python scripts/audio-analysis/bulk_recognition_runs.py --tx <tx-id> --tx <tx-id>
  uv run python scripts/audio-analysis/bulk_recognition_runs.py --all --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import note_f1_benchmark as nfb  # noqa: E402

DEFAULT_JSON_OUT = SCRIPT_DIR / "reports" / "bulk_recognition_runs.json"


def candidate_transaction_ids(tx_root: Path) -> list[str]:
    if not tx_root.is_dir():
        return []
    return sorted(
        d.name
        for d in tx_root.iterdir()
        if d.is_dir() and (d / "audio.wav").is_file() and (d / "request.json").is_file()
    )


def process_transaction(client, storage, tx_id: str, *, dry_run: bool) -> dict:
    before_response = storage.load_latest_response(tx_id) or {}
    before_notes = nfb.collect_one_best(before_response) if before_response.get("events") else []
    before_run_id = storage.latest_run_id(tx_id) or "legacy"
    saved_fingerprint = storage.resolved_recognizer_fingerprint(tx_id)

    if dry_run:
        return {
            "txId": tx_id,
            "beforeRunId": before_run_id,
            "savedFingerprint": saved_fingerprint,
            "beforeEventCount": len(before_notes),
            "dryRun": True,
        }

    response = client.post(f"/api/transcriptions/{tx_id}/runs")
    if response.status_code != 200:
        return {
            "txId": tx_id,
            "beforeRunId": before_run_id,
            "savedFingerprint": saved_fingerprint,
            "error": f"HTTP {response.status_code}: {response.text[:300]}",
        }
    body = response.json()
    after_notes = nfb.collect_one_best(body["result"])
    pseudo_truth = [
        {"time": n["time"], "note": n["note"], "tol": nfb.DEFAULT_TOLERANCE_SEC} for n in before_notes
    ]
    # Reuse note_f1_benchmark's matcher for output-to-output comparison (same
    # technique as metamorphic_alarm.py): "before" stands in for ground truth,
    # so falseNegatives = notes the new run dropped, falsePositives = notes it
    # added, relative to what was there before this run.
    match = nfb.match_pairs(pseudo_truth, after_notes)
    dropped = match["falseNegatives"]
    added = match["falsePositives"]
    return {
        "txId": tx_id,
        "beforeRunId": before_run_id,
        "afterRunId": body["runId"],
        "savedFingerprintBefore": saved_fingerprint,
        "recognizerFingerprintAfter": body["meta"]["recognizerFingerprint"],
        "beforeEventCount": len(before_notes),
        "afterEventCount": len(after_notes),
        "unchangedCount": match["tp"],
        "addedCount": len(added),
        "droppedCount": len(dropped),
        "added": [{"time": round(a["time"], 3), "note": a["note"]} for a in added],
        "dropped": [{"time": round(d_["time"], 3), "note": d_["note"]} for d_ in dropped],
    }


def print_table(output: dict) -> None:
    summary = output["summary"]
    print(
        f"recognizer {summary['recognizerFingerprint']}  dataDir={summary['dataDir']}"
        f"  candidates={summary['candidateCount']}  targeted={summary['targetedCount']}"
        f"  skipped(fresh)={summary['skippedFreshCount']}"
        f"{'  DRY RUN' if summary['dryRun'] else ''}"
    )
    print()
    print(f"{'txId':38} {'before':>7} {'after':>7} {'same':>5} {'add':>4} {'drop':>5}  note")
    for r in output["results"]:
        if r.get("dryRun"):
            print(f"{r['txId'][:36]:38} {r['beforeEventCount']:>7}  (dry-run, no new run created)")
            continue
        if "error" in r:
            print(f"{r['txId'][:36]:38} ERROR: {r['error']}")
            continue
        note = "" if (r["addedCount"] or r["droppedCount"]) else "unchanged"
        print(
            f"{r['txId'][:36]:38} {r['beforeEventCount']:>7} {r['afterEventCount']:>7}"
            f" {r['unchangedCount']:>5} {r['addedCount']:>4} {r['droppedCount']:>5}  {note}"
        )
    changed = [r for r in output["results"] if not r.get("dryRun") and "error" not in r and (r["addedCount"] or r["droppedCount"])]
    errors = [r for r in output["results"] if "error" in r]
    print()
    if changed:
        print(f"{len(changed)} of {len(output['results'])} re-recognitions changed the note-level output:")
        for r in changed:
            print(f"  {r['txId']}: +{r['addedCount']} added / -{r['droppedCount']} dropped")
    if errors:
        print(f"{len(errors)} transaction(s) errored: {[e['txId'] for e in errors]}")
    if not changed and not errors:
        print("No changes in note-level output for any re-recognized transaction.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="KALIMBA_DATA_DIR override (defaults to the env var, or ./data). "
             "REQUIRED to be a scratch/synthetic dir for any test/demo run.",
    )
    parser.add_argument("--tx", action="append", dest="tx_ids", help="restrict to these tx ids (repeatable)")
    parser.add_argument(
        "--all", action="store_true",
        help="re-recognize every candidate regardless of saved fingerprint (default: skip "
             "recordings whose saved fingerprint already matches the current recognizer)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report which transactions would be targeted, without calling the recognizer "
             "or persisting any run",
    )
    parser.add_argument("--json", action="store_true", help="print JSON to stdout instead of a table")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="write full JSON results here")
    parser.add_argument("--no-write", action="store_true", help="do not write the JSON report file, stdout only")
    args = parser.parse_args()

    if args.data_dir is not None:
        os.environ["KALIMBA_DATA_DIR"] = str(args.data_dir)

    from fastapi.testclient import TestClient
    from apps.api.app import storage
    from apps.api.app.fingerprints import recognizer_fingerprint
    from apps.api.app.main import app

    tx_root = storage.get_transactions_dir()
    candidates = args.tx_ids or candidate_transaction_ids(tx_root)
    if not candidates:
        print(f"No transactions found under {tx_root}.", file=sys.stderr)
        return 1

    current_fp = recognizer_fingerprint()
    targets: list[str] = []
    skipped_fresh: list[str] = []
    for tx_id in candidates:
        saved_fp = storage.resolved_recognizer_fingerprint(tx_id)
        # Unknown fingerprint counts as a target (see module docstring): unlike
        # the review-queue badge, this tool does not need to be conservative
        # about guessing, since re-running is safe and append-only.
        is_fresh = saved_fp is not None and saved_fp == current_fp
        if args.all or not is_fresh:
            targets.append(tx_id)
        else:
            skipped_fresh.append(tx_id)

    client = TestClient(app)
    results = [process_transaction(client, storage, tx_id, dry_run=args.dry_run) for tx_id in targets]

    summary = {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "recognizerFingerprint": current_fp,
        "dataDir": str(tx_root.parent),
        "candidateCount": len(candidates),
        "targetedCount": len(targets),
        "skippedFreshCount": len(skipped_fresh),
        "dryRun": args.dry_run,
        "all": args.all,
    }
    output = {"summary": summary, "results": results}

    if not args.no_write:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print_table(output)

    if any("error" in r for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
