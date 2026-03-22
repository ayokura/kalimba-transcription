import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from app.main import app
from manual_capture_helpers import (
    build_evaluation_audio_bytes,
    event_note_sets,
    fixture_dirs_for_status,
    fixture_id,
    load_fixture,
    normalized_assertions,
    primary_note_names,
    validate_expected_metadata,
)

client = TestClient(app)
COMPLETED_FIXTURES = fixture_dirs_for_status("completed")


def assert_fixture_output(fixture_dir: Path, payload: dict, expected: dict) -> None:
    primary_notes = primary_note_names(payload)
    note_sets = event_note_sets(payload)
    assertions = normalized_assertions(expected)

    min_events = assertions.get("minEvents")
    if min_events is not None:
        assert len(payload["events"]) >= min_events, fixture_dir.name

    max_events = assertions.get("maxEvents")
    if max_events is not None:
        assert len(payload["events"]) <= max_events, fixture_dir.name

    for note_name, min_occurrences in assertions.get("requiredPrimaryNoteOccurrences", {}).items():
        assert primary_notes.count(note_name) >= min_occurrences, fixture_dir.name

    for note_name, max_occurrences in assertions.get("maxPrimaryNoteOccurrences", {}).items():
        assert primary_notes.count(note_name) <= max_occurrences, fixture_dir.name

    for note_set, min_occurrences in assertions.get("requiredEventNoteSetOccurrences", {}).items():
        assert note_sets.count(note_set) >= min_occurrences, fixture_dir.name

    for note_set, max_occurrences in assertions.get("maxEventNoteSetOccurrences", {}).items():
        assert note_sets.count(note_set) <= max_occurrences, fixture_dir.name

    expected_note_sets_ordered = assertions.get("expectedEventNoteSetsOrdered")
    if expected_note_sets_ordered is not None:
        assert note_sets == expected_note_sets_ordered, fixture_dir.name


@pytest.mark.parametrize("fixture_dir", COMPLETED_FIXTURES, ids=fixture_id)
def test_completed_manual_capture_regression(fixture_dir: Path) -> None:
    request_payload, expected = load_fixture(fixture_dir)
    validate_expected_metadata(fixture_dir, expected)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, fixture_dir.name
    assert_fixture_output(fixture_dir, response.json(), expected)


if not COMPLETED_FIXTURES:
    pytest.skip("No completed manual capture fixtures available", allow_module_level=True)
