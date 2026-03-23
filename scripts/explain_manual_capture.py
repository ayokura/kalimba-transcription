from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def build_explanation(fixture_dir: Path) -> dict[str, Any]:
    request_payload, expected = load_fixture(fixture_dir)
    validate_request_metadata(fixture_dir, request_payload)
    validate_expected_metadata(fixture_dir, expected)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )
    response.raise_for_status()
    payload = response.json()
    summary = explain_fixture_output(fixture_dir, payload, expected)
    summary.update(
        {
            "status": fixture_status(expected),
            "reason": expected.get("reason"),
            "auditVerdict": expected.get("auditVerdict"),
            "sourceProfile": request_payload.get("sourceProfile"),
            "captureIntent": request_payload.get("captureIntent"),
            "scenario": request_payload.get("scenario"),
            "expectedSummary": (request_payload.get("expectedPerformance") or {}).get("summary"),
            "evaluationWindows": parse_ranges(expected, "evaluationWindows"),
            "ignoredRanges": parse_ranges(expected, "ignoredRanges"),
            "recommendedRecapture": expected.get("recommendedRecapture") or [],
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
