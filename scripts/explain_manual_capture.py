from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parents[1] / "apps" / "api" / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from app.main import app
from manual_capture_helpers import (
    FIXTURE_ROOT,
    build_evaluation_audio_bytes,
    explain_fixture_output,
    fixture_status,
    load_fixture,
    parse_ranges,
    validate_expected_metadata,
    validate_request_metadata,
)

client = TestClient(app)


def _scope_for_event(start_time: float, end_time: float, evaluation_windows: list[dict[str, float]], ignored_ranges: list[dict[str, float]]) -> str:
    if evaluation_windows:
        for entry in evaluation_windows:
            if start_time < entry["endSec"] and end_time > entry["startSec"]:
                return "in_scope"
        return "out_of_scope"
    if ignored_ranges:
        for entry in ignored_ranges:
            if start_time < entry["endSec"] and end_time > entry["startSec"]:
                return "out_of_scope"
        return "in_scope"
    return "in_scope"


def _evaluation_duration(source_duration_sec: float, evaluation_windows: list[dict[str, float]], ignored_ranges: list[dict[str, float]]) -> float:
    if evaluation_windows:
        return sum(entry["endSec"] - entry["startSec"] for entry in evaluation_windows)
    if ignored_ranges:
        return max(0.0, source_duration_sec - sum(entry["endSec"] - entry["startSec"] for entry in ignored_ranges))
    return source_duration_sec


def build_explanation(fixture_dir: Path) -> dict[str, Any]:
    request_payload, expected = load_fixture(fixture_dir)
    validate_request_metadata(fixture_dir, request_payload)
    validate_expected_metadata(fixture_dir, expected)
    evaluation_windows = parse_ranges(expected, "evaluationWindows")
    ignored_ranges = parse_ranges(expected, "ignoredRanges")
    source_audio_path = fixture_dir / "audio.wav"
    source_audio_bytes = source_audio_path.read_bytes()
    scoped_audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", scoped_audio_bytes, "audio/wav")},
    )
    response.raise_for_status()
    payload = response.json()

    full_payload = payload
    if evaluation_windows or ignored_ranges:
        full_response = client.post(
            "/api/transcriptions",
            data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
            files={"file": ("audio.wav", source_audio_bytes, "audio/wav")},
        )
        full_response.raise_for_status()
        full_payload = full_response.json()

    summary = explain_fixture_output(fixture_dir, payload, expected)
    source_duration_sec = float(sf.info(source_audio_path).duration)
    evaluation_duration_sec = _evaluation_duration(source_duration_sec, evaluation_windows, ignored_ranges)
    coverage_events = []
    for event in full_payload.get("debug", {}).get("mergedEvents", []):
        note_set = "+".join(sorted(event.get("notes") or []))
        coverage_events.append(
            {
                "startTime": event["startTime"],
                "endTime": event["endTime"],
                "noteSet": note_set,
                "gesture": event.get("gesture"),
                "scope": _scope_for_event(float(event["startTime"]), float(event["endTime"]), evaluation_windows, ignored_ranges),
            }
        )
    summary.update(
        {
            "status": fixture_status(expected),
            "reason": expected.get("reason"),
            "auditVerdict": expected.get("auditVerdict"),
            "sourceProfile": request_payload.get("sourceProfile"),
            "captureIntent": request_payload.get("captureIntent"),
            "scenario": request_payload.get("scenario"),
            "expectedSummary": (request_payload.get("expectedPerformance") or {}).get("summary"),
            "evaluationWindows": evaluation_windows,
            "ignoredRanges": ignored_ranges,
            "recommendedRecapture": expected.get("recommendedRecapture") or [],
            "evaluationMode": "evaluation_windows" if evaluation_windows else "ignored_ranges" if ignored_ranges else "full_audio",
            "sourceDurationSec": round(source_duration_sec, 4),
            "evaluationDurationSec": round(evaluation_duration_sec, 4),
            "fullEventCount": len(full_payload.get("events", [])),
            "eventCoverage": coverage_events,
        }
    )
    return summary


def print_text(summary: dict[str, Any]) -> None:
    print(f"fixture: {summary['fixtureId']}")
    print(f"status: {summary['status']}")
    print(f"sourceProfile: {summary['sourceProfile']}")
    print(f"captureIntent: {summary['captureIntent']}")
    if summary.get("scenario"):
        print(f"scenario: {summary['scenario']}")
    if summary.get("expectedSummary"):
        print(f"expected: {summary['expectedSummary']}")
    print(f"detected event count: {summary['eventCount']}")
    print(f"detected primary notes: {' / '.join(summary['primaryNotes']) if summary['primaryNotes'] else '(none)'}")
    print(f"detected note sets: {' / '.join(summary['eventNoteSets']) if summary['eventNoteSets'] else '(none)'}")
    if summary.get("evaluationWindows"):
        print(f"evaluationWindows: {summary['evaluationWindows']}")
    if summary.get("ignoredRanges"):
        print(f"ignoredRanges: {summary['ignoredRanges']}")
    print(f"evaluationMode: {summary['evaluationMode']}")
    print(f"sourceDurationSec: {summary['sourceDurationSec']}")
    print(f"evaluationDurationSec: {summary['evaluationDurationSec']}")
    if summary.get("eventCoverage"):
        in_scope = [event for event in summary['eventCoverage'] if event['scope'] == 'in_scope']
        out_of_scope = [event for event in summary['eventCoverage'] if event['scope'] == 'out_of_scope']
        print(f"coverage: {len(in_scope)} in-scope / {len(out_of_scope)} out-of-scope")
        for event in summary['eventCoverage']:
            print(f"  [{event['scope']}] {event['startTime']}->{event['endTime']} {event['noteSet']} ({event.get('gesture')})")
    if summary.get("reason"):
        print(f"reason: {summary['reason']}")
    if summary.get("auditVerdict"):
        print(f"auditVerdict: {summary['auditVerdict']}")
    if summary.get("recommendedRecapture"):
        print("recommendedRecapture:")
        for item in summary['recommendedRecapture']:
            print(f"  - {item}")
    failures = summary.get("assertionFailures") or []
    if failures:
        print("assertionFailures:")
        for item in failures:
            print(f"  - {item}")
    else:
        print("assertionFailures: (none)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain a manual capture fixture against current recognizer output.")
    parser.add_argument("fixture_id", help="Fixture directory name under apps/api/tests/fixtures/manual-captures")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    fixture_dir = FIXTURE_ROOT / args.fixture_id
    if not fixture_dir.exists():
        raise SystemExit(f"Fixture not found: {fixture_dir}")

    summary = build_explanation(fixture_dir)
    if args.as_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
