from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TESTS_DIR = Path(__file__).resolve().parents[1] / "apps" / "api" / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from app.transcription import REPEATED_PATTERN_PASS_IDS
from explain_manual_capture import build_explanation
from manual_capture_helpers import fixture_dirs_for_status, fixture_id, fixture_status, load_fixture

DEFAULT_STATUSES = ("completed", "pending", "rerecord", "review_needed", "reference_only")


def fixture_taxonomy(request_payload: dict[str, Any]) -> str:
    performance = request_payload.get("expectedPerformance") or {}
    events = performance.get("events") or []
    if not events:
        return "no_expected_events"

    note_sets = [
        tuple(sorted(key.get("noteName") for key in (event.get("keys") or []) if key.get("noteName")))
        for event in events
    ]
    unique_note_sets = {note_set for note_set in note_sets if note_set}
    event_intents = [event.get("intent") for event in events if event.get("intent") is not None]

    if len(unique_note_sets) <= 1 and len(events) >= 2:
        return "single_event_repeat"
    if len(events) >= 4 and len(unique_note_sets) <= max(2, len(events) // 2):
        return "small_repeated_phrase"
    if len(set(event_intents)) > 1:
        return "mixed_phrase"
    if len(events) >= 8:
        return "free_performance"
    return "mixed_phrase"


def summarize_delta(baseline: dict[str, Any], ablated: dict[str, Any]) -> dict[str, Any]:
    return {
        "baselineEventCount": baseline["eventCount"],
        "ablatedEventCount": ablated["eventCount"],
        "eventCountDelta": ablated["eventCount"] - baseline["eventCount"],
        "baselineFailures": baseline["assertionFailures"],
        "ablatedFailures": ablated["assertionFailures"],
        "newFailureCodes": sorted(set(ablated["reasonCodes"]) - set(baseline["reasonCodes"])),
        "resolvedFailureCodes": sorted(set(baseline["reasonCodes"]) - set(ablated["reasonCodes"])),
        "normalizationDelta": {
            key: ablated["normalizationSummary"].get(key, 0) - baseline["normalizationSummary"].get(key, 0)
            for key in ("segmentCount", "rawEventCount", "mergedEventCount", "rawToMergedDelta")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repeated-pattern pass dependence across manual fixtures.")
    parser.add_argument("--status", action="append", choices=DEFAULT_STATUSES, default=[])
    parser.add_argument("--pass-id", action="append", dest="pass_ids", choices=REPEATED_PATTERN_PASS_IDS, default=[])
    parser.add_argument("--only-changed", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    statuses = tuple(args.status) if args.status else DEFAULT_STATUSES
    pass_ids = tuple(args.pass_ids) if args.pass_ids else REPEATED_PATTERN_PASS_IDS

    fixture_dirs = []
    for status in statuses:
        fixture_dirs.extend(fixture_dirs_for_status(status))
    fixture_dirs = sorted({fixture_dir.resolve(): fixture_dir for fixture_dir in fixture_dirs}.values(), key=fixture_id)

    report: dict[str, Any] = {"passes": {}, "fixtures": [fixture_id(fixture_dir) for fixture_dir in fixture_dirs]}

    for pass_id in pass_ids:
        changed: list[dict[str, Any]] = []
        status_counts: dict[str, int] = defaultdict(int)
        taxonomy_counts: dict[str, int] = defaultdict(int)
        for fixture_dir in fixture_dirs:
            request_payload, expected = load_fixture(fixture_dir)
            baseline = build_explanation(fixture_dir)
            ablated = build_explanation(fixture_dir, disabled_passes=[pass_id])
            delta = summarize_delta(baseline, ablated)
            if baseline["eventCount"] == ablated["eventCount"] and baseline["reasonCodes"] == ablated["reasonCodes"] and baseline["assertionFailures"] == ablated["assertionFailures"]:
                if args.only_changed:
                    continue
            status = fixture_status(expected)
            taxonomy = fixture_taxonomy(request_payload)
            status_counts[status] += 1
            taxonomy_counts[taxonomy] += 1
            changed.append(
                {
                    "fixtureId": fixture_id(fixture_dir),
                    "status": status,
                    "taxonomy": taxonomy,
                    **delta,
                }
            )
        report["passes"][pass_id] = {
            "statusCounts": dict(sorted(status_counts.items())),
            "taxonomyCounts": dict(sorted(taxonomy_counts.items())),
            "fixtures": changed,
        }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    for pass_id in pass_ids:
        entry = report["passes"][pass_id]
        print(f"## {pass_id}")
        if entry["statusCounts"]:
            print(f"- statuses: {entry['statusCounts']}")
        if entry["taxonomyCounts"]:
            print(f"- taxonomies: {entry['taxonomyCounts']}")
        fixtures = entry["fixtures"]
        if not fixtures:
            print("- no changed fixtures")
            print()
            continue
        for fixture in fixtures:
            print(
                f"- {fixture['fixtureId']} [{fixture['status']} / {fixture['taxonomy']}] "
                f"events {fixture['baselineEventCount']} -> {fixture['ablatedEventCount']} "
                f"new={fixture['newFailureCodes'] or '(none)'} resolved={fixture['resolvedFailureCodes'] or '(none)'}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
