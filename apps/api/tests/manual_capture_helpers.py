from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "manual-captures"
VALID_STATUSES = {"completed", "pending", "rerecord", "review_needed", "reference_only"}
VALID_SOURCE_PROFILES = {"acoustic_real", "app_synth"}
VALID_CAPTURE_INTENTS = {"strict_chord", "slide_chord", "arpeggio", "separated_notes", "unknown"}
ASSERTION_KEYS = {
    "minEvents",
    "maxEvents",
    "requiredPrimaryNoteOccurrences",
    "maxPrimaryNoteOccurrences",
    "requiredEventNoteSetOccurrences",
    "maxEventNoteSetOccurrences",
    "expectedEventNoteSetsOrdered",
}
DEFAULT_ASSERTIONS = {
    "minEvents": None,
    "maxEvents": None,
    "requiredPrimaryNoteOccurrences": {},
    "maxPrimaryNoteOccurrences": {},
    "requiredEventNoteSetOccurrences": {},
    "maxEventNoteSetOccurrences": {},
}
RangeSpec = dict[str, float]
AssertionFailureDetail = dict[str, Any]


def list_fixture_dirs() -> list[Path]:
    return sorted(path for path in FIXTURE_ROOT.iterdir() if path.is_dir())


def fixture_id(fixture_dir: Path) -> str:
    return fixture_dir.name


def load_fixture(fixture_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    return request_payload, expected


def validate_request_metadata(fixture_dir: Path, request_payload: dict[str, Any]) -> None:
    source_profile = request_payload.get("sourceProfile")
    if not isinstance(source_profile, str) or source_profile not in VALID_SOURCE_PROFILES:
        raise AssertionError(f"{fixture_dir.name}: request.sourceProfile must be one of {sorted(VALID_SOURCE_PROFILES)}")

    capture_intent = request_payload.get("captureIntent")
    if capture_intent is not None and (not isinstance(capture_intent, str) or capture_intent not in VALID_CAPTURE_INTENTS):
        raise AssertionError(f"{fixture_dir.name}: request.captureIntent must be one of {sorted(VALID_CAPTURE_INTENTS)} or null")

    expected_performance = request_payload.get("expectedPerformance") or {}
    default_capture_intent = expected_performance.get("defaultCaptureIntent")
    if default_capture_intent is not None and (not isinstance(default_capture_intent, str) or default_capture_intent not in VALID_CAPTURE_INTENTS):
        raise AssertionError(f"{fixture_dir.name}: expectedPerformance.defaultCaptureIntent must be one of {sorted(VALID_CAPTURE_INTENTS)} or null")

    for index, event in enumerate(expected_performance.get("events") or [], start=1):
        event_intent = event.get("intent")
        if event_intent is not None and (not isinstance(event_intent, str) or event_intent not in VALID_CAPTURE_INTENTS):
            raise AssertionError(f"{fixture_dir.name}: expectedPerformance.events[{index}].intent must be one of {sorted(VALID_CAPTURE_INTENTS)} or null")


def fixture_status(expected: dict[str, Any]) -> str:
    status = expected.get("status")
    if status is not None:
        if status not in VALID_STATUSES:
            raise AssertionError(f"Unknown fixture status: {status}")
        return status
    return "pending" if expected.get("pending") else "completed"


def fixture_dirs_for_status(status: str) -> list[Path]:
    return [fixture_dir for fixture_dir in list_fixture_dirs() if fixture_status(load_fixture(fixture_dir)[1]) == status]


def normalized_assertions(expected: dict[str, Any]) -> dict[str, Any]:
    assertions = dict(DEFAULT_ASSERTIONS)
    assertions.update(expected.get("assertions") or {})
    return assertions


def primary_note_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for event in payload["events"]:
        if event["notes"]:
            note = event["notes"][0]
            names.append(f"{note['pitchClass']}{note['octave']}")
    return names


def event_note_sets(payload: dict[str, Any]) -> list[str]:
    note_sets: list[str] = []
    for event in payload["events"]:
        notes = sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"])
        if notes:
            note_sets.append("+".join(notes))
    return note_sets


def parse_ranges(expected: dict[str, Any], key: str) -> list[RangeSpec]:
    ranges = expected.get(key) or []
    parsed: list[RangeSpec] = []
    for entry in ranges:
        start = float(entry["startSec"])
        end = float(entry["endSec"])
        if start < 0 or end <= start:
            raise AssertionError(f"Invalid {key} range: {entry}")
        parsed.append({"startSec": start, "endSec": end})
    return parsed


def validate_expected_ranges(expected: dict[str, Any]) -> None:
    windows = parse_ranges(expected, "evaluationWindows")
    ignored = parse_ranges(expected, "ignoredRanges")
    if windows and ignored:
        raise AssertionError("Use either evaluationWindows or ignoredRanges, not both")

    previous_end = None
    for entry in windows:
        start = entry["startSec"]
        end = entry["endSec"]
        if previous_end is not None and start < previous_end:
            raise AssertionError(f"Overlapping evaluationWindows: {windows}")
        previous_end = end

    previous_end = None
    for entry in ignored:
        start = entry["startSec"]
        end = entry["endSec"]
        if previous_end is not None and start < previous_end:
            raise AssertionError(f"Overlapping ignoredRanges: {ignored}")
        previous_end = end


def validate_expected_metadata(fixture_dir: Path, expected: dict[str, Any]) -> None:
    status = fixture_status(expected)
    assertions = expected.get("assertions")
    if not isinstance(assertions, dict):
        raise AssertionError(f"{fixture_dir.name}: expected.assertions must be an object")

    unknown_assertion_keys = set(assertions.keys()) - ASSERTION_KEYS
    if unknown_assertion_keys:
        raise AssertionError(f"{fixture_dir.name}: unknown assertion keys {sorted(unknown_assertion_keys)}")

    assertions = normalized_assertions(expected)
    for key in [
        "requiredPrimaryNoteOccurrences",
        "maxPrimaryNoteOccurrences",
        "requiredEventNoteSetOccurrences",
        "maxEventNoteSetOccurrences",
    ]:
        if not isinstance(assertions.get(key), dict):
            raise AssertionError(f"{fixture_dir.name}: assertion '{key}' must be an object")

    ordered_note_sets = assertions.get("expectedEventNoteSetsOrdered")
    if ordered_note_sets is not None and not isinstance(ordered_note_sets, list):
        raise AssertionError(f"{fixture_dir.name}: expectedEventNoteSetsOrdered must be a list")

    validate_expected_ranges(expected)

    if status == "completed" and expected.get("pending", False):
        raise AssertionError(f"{fixture_dir.name}: completed fixtures cannot keep pending=true")

    if status in {"pending", "rerecord", "review_needed", "reference_only"}:
        reason = expected.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise AssertionError(f"{fixture_dir.name}: status '{status}' requires a reason")

    if "auditVerdict" in expected:
        verdict = expected["auditVerdict"]
        if not isinstance(verdict, str) or not verdict.strip():
            raise AssertionError(f"{fixture_dir.name}: auditVerdict must be a non-empty string when present")

    if status == "rerecord":
        recapture = expected.get("recommendedRecapture")
        if not isinstance(recapture, list) or not recapture or not all(isinstance(item, str) and item.strip() for item in recapture):
            raise AssertionError(f"{fixture_dir.name}: rerecord fixtures require non-empty recommendedRecapture")

    if status == "completed":
        has_signal = any(
            [
                assertions.get("minEvents") is not None,
                assertions.get("maxEvents") is not None,
                bool(assertions.get("requiredPrimaryNoteOccurrences")),
                bool(assertions.get("maxPrimaryNoteOccurrences")),
                bool(assertions.get("requiredEventNoteSetOccurrences")),
                bool(assertions.get("maxEventNoteSetOccurrences")),
                assertions.get("expectedEventNoteSetsOrdered") is not None,
            ]
        )
        if not has_signal:
            raise AssertionError(f"{fixture_dir.name}: completed fixtures require at least one concrete assertion")




def assertion_failure_details(fixture_dir: Path, payload: dict[str, Any], expected: dict[str, Any]) -> list[AssertionFailureDetail]:
    primary_notes = primary_note_names(payload)
    note_sets = event_note_sets(payload)
    assertions = normalized_assertions(expected)
    failures: list[AssertionFailureDetail] = []

    min_events = assertions.get("minEvents")
    event_count = len(payload["events"])
    if min_events is not None and event_count < min_events:
        failures.append(
            {
                "code": "event_count_too_low",
                "name": "Expected event count not reached",
                "assertionKey": "minEvents",
                "subject": "event_count",
                "expected": {"operator": ">=", "value": min_events},
                "actual": event_count,
                "message": f"minEvents expected >= {min_events}, got {event_count}",
            }
        )

    max_events = assertions.get("maxEvents")
    if max_events is not None and event_count > max_events:
        failures.append(
            {
                "code": "event_count_too_high",
                "name": "Detected event count exceeds expected range",
                "assertionKey": "maxEvents",
                "subject": "event_count",
                "expected": {"operator": "<=", "value": max_events},
                "actual": event_count,
                "message": f"maxEvents expected <= {max_events}, got {event_count}",
            }
        )

    for note_name, min_occurrences in assertions.get("requiredPrimaryNoteOccurrences", {}).items():
        actual = primary_notes.count(note_name)
        if actual < min_occurrences:
            failures.append(
                {
                    "code": "primary_note_missing",
                    "name": "Primary note under-detected",
                    "assertionKey": "requiredPrimaryNoteOccurrences",
                    "subject": note_name,
                    "expected": {"operator": ">=", "value": min_occurrences},
                    "actual": actual,
                    "message": f"requiredPrimaryNoteOccurrences[{note_name}] expected >= {min_occurrences}, got {actual}",
                }
            )

    for note_name, max_occurrences in assertions.get("maxPrimaryNoteOccurrences", {}).items():
        actual = primary_notes.count(note_name)
        if actual > max_occurrences:
            failures.append(
                {
                    "code": "primary_note_excess",
                    "name": "Primary note over-detected",
                    "assertionKey": "maxPrimaryNoteOccurrences",
                    "subject": note_name,
                    "expected": {"operator": "<=", "value": max_occurrences},
                    "actual": actual,
                    "message": f"maxPrimaryNoteOccurrences[{note_name}] expected <= {max_occurrences}, got {actual}",
                }
            )

    for note_set, min_occurrences in assertions.get("requiredEventNoteSetOccurrences", {}).items():
        actual = note_sets.count(note_set)
        if actual < min_occurrences:
            failures.append(
                {
                    "code": "note_set_missing",
                    "name": "Expected note-set under-detected",
                    "assertionKey": "requiredEventNoteSetOccurrences",
                    "subject": note_set,
                    "expected": {"operator": ">=", "value": min_occurrences},
                    "actual": actual,
                    "message": f"requiredEventNoteSetOccurrences[{note_set}] expected >= {min_occurrences}, got {actual}",
                }
            )

    for note_set, max_occurrences in assertions.get("maxEventNoteSetOccurrences", {}).items():
        actual = note_sets.count(note_set)
        if actual > max_occurrences:
            failures.append(
                {
                    "code": "note_set_excess",
                    "name": "Detected note-set over-produced",
                    "assertionKey": "maxEventNoteSetOccurrences",
                    "subject": note_set,
                    "expected": {"operator": "<=", "value": max_occurrences},
                    "actual": actual,
                    "message": f"maxEventNoteSetOccurrences[{note_set}] expected <= {max_occurrences}, got {actual}",
                }
            )

    expected_note_sets_ordered = assertions.get("expectedEventNoteSetsOrdered")
    if expected_note_sets_ordered is not None and note_sets != expected_note_sets_ordered:
        failures.append(
            {
                "code": "event_order_mismatch",
                "name": "Detected event order differs from expected sequence",
                "assertionKey": "expectedEventNoteSetsOrdered",
                "subject": "ordered_note_sets",
                "expected": expected_note_sets_ordered,
                "actual": note_sets,
                "message": "expectedEventNoteSetsOrdered mismatch",
            }
        )

    return failures


def assertion_failures(fixture_dir: Path, payload: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    return [detail["message"] for detail in assertion_failure_details(fixture_dir, payload, expected)]


def explain_fixture_output(fixture_dir: Path, payload: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    assertion_failure_info = assertion_failure_details(fixture_dir, payload, expected)
    return {
        "fixtureId": fixture_dir.name,
        "eventCount": len(payload.get("events", [])),
        "primaryNotes": primary_note_names(payload),
        "eventNoteSets": event_note_sets(payload),
        "assertionFailures": [detail["message"] for detail in assertion_failure_info],
        "assertionFailureDetails": assertion_failure_info,
        "reasonCodes": [detail["code"] for detail in assertion_failure_info],
    }

def build_evaluation_audio_bytes(fixture_dir: Path, expected: dict[str, Any]) -> bytes:
    windows = parse_ranges(expected, "evaluationWindows")
    ignored = parse_ranges(expected, "ignoredRanges")
    if not windows and not ignored:
        return (fixture_dir / "audio.wav").read_bytes()

    audio, sample_rate = sf.read(fixture_dir / "audio.wav", always_2d=True)
    total_duration = audio.shape[0] / sample_rate

    if windows:
        segments = windows
    else:
        cursor = 0.0
        segments = []
        for entry in ignored:
            if cursor < entry["startSec"]:
                segments.append({"startSec": cursor, "endSec": entry["startSec"]})
            cursor = max(cursor, entry["endSec"])
        if cursor < total_duration:
            segments.append({"startSec": cursor, "endSec": total_duration})

    clips = []
    for entry in segments:
        if entry["endSec"] > total_duration + 1e-6:
            raise AssertionError(f"Range exceeds audio duration for {fixture_dir.name}: {entry}")
        start_index = max(0, int(round(entry["startSec"] * sample_rate)))
        end_index = min(audio.shape[0], int(round(entry["endSec"] * sample_rate)))
        if end_index > start_index:
            clips.append(audio[start_index:end_index])
    if not clips:
        raise AssertionError(f"No audio remained after applying evaluation ranges for {fixture_dir.name}")

    combined = np.concatenate(clips, axis=0)
    buffer = io.BytesIO()
    sf.write(buffer, combined, sample_rate, format="WAV")
    return buffer.getvalue()
