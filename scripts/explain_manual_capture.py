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


def _dominant_gesture_mix(coverage_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in coverage_events:
        gesture = event.get("gesture") or "ambiguous"
        counts[gesture] = counts.get(gesture, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _phrase_break_guess(coverage_events: list[dict[str, Any]]) -> list[int]:
    breaks: list[int] = []
    for index in range(len(coverage_events) - 1):
        current = coverage_events[index]
        following = coverage_events[index + 1]
        gap = float(following["startTime"]) - float(current["endTime"])
        if gap >= 0.45:
            breaks.append(index + 1)
    return breaks


def _isolated_singleton_count(coverage_events: list[dict[str, Any]]) -> int:
    count = 0
    for index, event in enumerate(coverage_events):
        note_set = event.get("noteSet") or ""
        if "+" in note_set:
            continue
        start = float(event["startTime"])
        end = float(event["endTime"])
        duration = end - start
        previous = coverage_events[index - 1] if index > 0 else None
        following = coverage_events[index + 1] if index + 1 < len(coverage_events) else None
        prev_gap = start - float(previous["endTime"]) if previous is not None else 1.0
        next_gap = float(following["startTime"]) - end if following is not None else 1.0
        if duration <= 0.18 and (prev_gap <= 0.2 or next_gap <= 0.2):
            count += 1
    return count


def _event_compression(expected_event_count: int, detected_event_count: int) -> dict[str, Any] | None:
    if expected_event_count <= 0:
        return None
    return {
        "expected": expected_event_count,
        "detected": detected_event_count,
        "compressionRatio": round(detected_event_count / expected_event_count, 3),
    }


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
    expected_event_count = len(((request_payload.get("expectedPerformance") or {}).get("events") or []))
    summary.update(
        {
            "status": fixture_status(expected),
            "reason": expected.get("reason"),
            "auditVerdict": expected.get("auditVerdict"),
            "sourceProfile": request_payload.get("sourceProfile"),
            "captureIntent": request_payload.get("captureIntent"),
            "defaultCaptureIntent": (request_payload.get("expectedPerformance") or {}).get("defaultCaptureIntent"),
            "eventIntents": [event.get("intent") for event in ((request_payload.get("expectedPerformance") or {}).get("events") or [])],
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
            "eventCompression": _event_compression(expected_event_count, len(payload.get("events", []))),
            "dominantGestureMix": _dominant_gesture_mix(coverage_events),
            "isolatedSingletonCount": _isolated_singleton_count(coverage_events),
            "phraseBreakGuess": _phrase_break_guess(coverage_events),
        }
    )
    return summary


def print_text(summary: dict[str, Any]) -> None:
    print(f"fixture: {summary['fixtureId']}")
    print(f"status: {summary['status']}")
    print(f"sourceProfile: {summary['sourceProfile']}")
    print(f"captureIntent: {summary['captureIntent']}")
    if summary.get("defaultCaptureIntent") is not None:
        print(f"defaultCaptureIntent: {summary['defaultCaptureIntent']}")
    event_intents = [intent for intent in summary.get("eventIntents", []) if intent is not None]
    if event_intents:
        print(f"eventIntents: {' / '.join(event_intents)}")
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
    if summary.get("eventCompression") is not None:
        compression = summary["eventCompression"]
        print(f"eventCompression: expected {compression['expected']} / detected {compression['detected']} / ratio {compression['compressionRatio']}")
    if summary.get("dominantGestureMix"):
        mix = " / ".join(f"{key}:{value}" for key, value in summary["dominantGestureMix"].items())
        print(f"dominantGestureMix: {mix}")
    print(f"isolatedSingletonCount: {summary.get('isolatedSingletonCount', 0)}")
    if summary.get("phraseBreakGuess"):
        print(f"phraseBreakGuess: {summary['phraseBreakGuess']}")
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
    reason_codes = summary.get("reasonCodes") or []
    if reason_codes:
        print(f"reasonCodes: {' / '.join(reason_codes)}")
    failures = summary.get("assertionFailureDetails") or []
    if failures:
        print("assertionFailures:")
        for item in failures:
            print(f"  - [{item['code']}] {item['message']}")
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
