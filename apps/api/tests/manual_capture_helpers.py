from __future__ import annotations

import io
import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "manual-captures"
VALID_STATUSES = {"completed", "pending", "rerecord", "review_needed", "reference_only"}
VALID_SOURCE_PROFILES = {"acoustic_real", "app_synth"}
VALID_CAPTURE_INTENTS = {"strict_chord", "slide_chord", "arpeggio", "separated_notes", "unknown"}
VALID_GROUND_TRUTH_METHODS = {"ear_verified", "spectrogram_verified", "aubio_cross_checked"}
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
NOTES_VERDICT_PATTERN = re.compile(r"^- verdict:\s*(.+)$", re.MULTILINE)


@lru_cache(maxsize=1)
def _fixture_dirs() -> tuple[Path, ...]:
    return tuple(sorted(path for path in FIXTURE_ROOT.iterdir() if path.is_dir()))


def list_fixture_dirs() -> list[Path]:
    return list(_fixture_dirs())


def fixture_id(fixture_dir: Path) -> str:
    return fixture_dir.name


@lru_cache(maxsize=None)
def _load_request_payload(fixture_dir: Path) -> dict[str, Any]:
    return json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _load_expected_payload(fixture_dir: Path) -> dict[str, Any]:
    return json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))


def load_fixture(fixture_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return deepcopy(_load_request_payload(fixture_dir)), deepcopy(_load_expected_payload(fixture_dir))


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
    return [fixture_dir for fixture_dir in _fixture_dirs() if fixture_status(_load_expected_payload(fixture_dir)) == status]


def load_notes_verdict(fixture_dir: Path) -> str:
    notes_text = (fixture_dir / "notes.md").read_text(encoding="utf-8")
    match = NOTES_VERDICT_PATTERN.search(notes_text)
    if match is None:
        raise AssertionError(f"{fixture_dir.name}: notes.md must include a '- verdict: ...' line")
    verdict = match.group(1).strip()
    if not verdict:
        raise AssertionError(f"{fixture_dir.name}: notes.md verdict must not be empty")
    return verdict


def validate_notes_metadata(fixture_dir: Path, expected: dict[str, Any]) -> None:
    expected_status = fixture_status(expected)
    notes_verdict = load_notes_verdict(fixture_dir)
    if notes_verdict != expected_status:
        raise AssertionError(
            f"{fixture_dir.name}: notes.md verdict '{notes_verdict}' must match expected status '{expected_status}'"
        )


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


def _note_sort_key(note_name: str) -> float:
    """Sort key by frequency (low to high) for consistent note set ordering."""
    from app.tunings import note_name_to_frequency
    try:
        return note_name_to_frequency(note_name)
    except Exception:
        return 0.0


def event_note_sets(payload: dict[str, Any]) -> list[str]:
    note_sets: list[str] = []
    for event in payload["events"]:
        notes = sorted(
            (f"{note['pitchClass']}{note['octave']}" for note in event["notes"]),
            key=_note_sort_key,
        )
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

def build_transcription_form_data(request_payload: dict[str, Any]) -> dict[str, str]:
    """Build the form data dict for /api/transcriptions from a fixture request payload."""
    form_data: dict[str, str] = {"tuning": json.dumps(request_payload["tuning"])}
    if request_payload.get("midPerformanceStart"):
        form_data["midPerformanceStart"] = "true"
    if request_payload.get("midPerformanceEnd"):
        form_data["midPerformanceEnd"] = "true"
    return form_data


def build_evaluation_audio_bytes(fixture_dir: Path, expected: dict[str, Any]) -> bytes:
    windows = parse_ranges(expected, "evaluationWindows")
    ignored = parse_ranges(expected, "ignoredRanges")
    return _build_evaluation_audio_bytes_cached(
        fixture_dir,
        tuple((entry["startSec"], entry["endSec"]) for entry in windows),
        tuple((entry["startSec"], entry["endSec"]) for entry in ignored),
    )


@lru_cache(maxsize=32)
def _build_evaluation_audio_bytes_cached(
    fixture_dir: Path,
    windows: tuple[tuple[float, float], ...],
    ignored: tuple[tuple[float, float], ...],
) -> bytes:
    if not windows and not ignored:
        return (fixture_dir / "audio.wav").read_bytes()

    audio, sample_rate = sf.read(fixture_dir / "audio.wav", always_2d=True)
    total_duration = audio.shape[0] / sample_rate

    if windows:
        segments = [{"startSec": start, "endSec": end} for start, end in windows]
    else:
        cursor = 0.0
        segments = []
        for start, end in ignored:
            if cursor < start:
                segments.append({"startSec": cursor, "endSec": start})
            cursor = max(cursor, end)
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


def load_ground_truth(fixture_dir: Path) -> dict[str, Any] | None:
    """Load ground_truth.json if it exists, return None otherwise."""
    gt_path = fixture_dir / "ground_truth.json"
    if not gt_path.exists():
        return None
    ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
    validate_ground_truth_metadata(fixture_dir, ground_truth)
    return ground_truth


def validate_ground_truth_metadata(fixture_dir: Path, ground_truth: dict[str, Any]) -> None:
    version = ground_truth.get("version")
    if version != 1:
        raise AssertionError(f"{fixture_dir.name}: ground_truth.version must be 1")

    default_tolerance = ground_truth.get("toleranceSec", 0.05)
    if not isinstance(default_tolerance, (int, float)) or float(default_tolerance) <= 0:
        raise AssertionError(f"{fixture_dir.name}: ground_truth.toleranceSec must be a positive number")

    onsets = ground_truth.get("onsets")
    if not isinstance(onsets, list) or not onsets:
        raise AssertionError(f"{fixture_dir.name}: ground_truth.onsets must be a non-empty array")

    previous_time = None
    for index, onset in enumerate(onsets, start=1):
        if not isinstance(onset, dict):
            raise AssertionError(f"{fixture_dir.name}: ground_truth.onsets[{index}] must be an object")

        time_sec = onset.get("timeSec")
        if not isinstance(time_sec, (int, float)) or float(time_sec) < 0:
            raise AssertionError(
                f"{fixture_dir.name}: ground_truth.onsets[{index}].timeSec must be a non-negative number"
            )
        onset_time = float(time_sec)
        if previous_time is not None and onset_time <= previous_time:
            raise AssertionError(f"{fixture_dir.name}: ground_truth.onsets must be strictly increasing by timeSec")

        notes = onset.get("notes")
        if not isinstance(notes, list) or not notes or not all(isinstance(note, str) and note.strip() for note in notes):
            raise AssertionError(
                f"{fixture_dir.name}: ground_truth.onsets[{index}].notes must be a non-empty string array"
            )

        tolerance = onset.get("toleranceSec")
        if tolerance is not None and (not isinstance(tolerance, (int, float)) or float(tolerance) <= 0):
            raise AssertionError(
                f"{fixture_dir.name}: ground_truth.onsets[{index}].toleranceSec must be a positive number"
            )

        method = onset.get("method")
        if not isinstance(method, str) or method not in VALID_GROUND_TRUTH_METHODS:
            raise AssertionError(
                f"{fixture_dir.name}: ground_truth.onsets[{index}].method must be one of {sorted(VALID_GROUND_TRUTH_METHODS)}"
            )

        comment = onset.get("comment")
        if comment is not None and (not isinstance(comment, str) or not comment.strip()):
            raise AssertionError(
                f"{fixture_dir.name}: ground_truth.onsets[{index}].comment must be a non-empty string when present"
            )

        previous_time = onset_time


def ground_truth_timing_failures(
    fixture_dir: Path,
    payload: dict[str, Any],
    ground_truth: dict[str, Any],
) -> list[str]:
    """Check detected event onsets against ground truth timing.

    Returns a list of failure messages. Empty list means all checks passed.
    """
    failures: list[str] = []
    default_tolerance = ground_truth.get("toleranceSec", 0.05)
    expected_onsets = ground_truth.get("onsets", [])

    # Get detected event start times and note sets
    detected: list[tuple[float, list[str]]] = []
    for seg in payload.get("debug", {}).get("segmentCandidates", []):
        notes = seg.get("selectedNotes", [])
        if notes:
            detected.append((seg["startTime"], sorted(notes)))

    # Check each expected onset has a matching detection
    matched_detected: set[int] = set()
    for gt_onset in expected_onsets:
        gt_time = gt_onset["timeSec"]
        gt_notes = sorted(gt_onset["notes"])
        tolerance = gt_onset.get("toleranceSec", default_tolerance)

        found = False
        for idx, (det_time, det_notes) in enumerate(detected):
            if idx in matched_detected:
                continue
            if abs(det_time - gt_time) <= tolerance:
                matched_detected.add(idx)
                if det_notes != gt_notes:
                    failures.append(
                        f"onset@{gt_time:.3f}s: expected notes {gt_notes}, got {det_notes}"
                    )
                found = True
                break

        if not found:
            failures.append(
                f"onset@{gt_time:.3f}s: expected {gt_notes} not detected within {tolerance}s tolerance"
            )

    return failures
