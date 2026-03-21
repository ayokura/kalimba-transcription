import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "manual-captures"


def primary_note_names(payload: dict) -> list[str]:
    names: list[str] = []
    for event in payload["events"]:
        if event["notes"]:
            note = event["notes"][0]
            names.append(f"{note['pitchClass']}{note['octave']}")
    return names


def event_note_sets(payload: dict) -> list[str]:
    note_sets: list[str] = []
    for event in payload["events"]:
        notes = sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"])
        if notes:
            note_sets.append("+".join(notes))
    return note_sets


def test_manual_capture_regressions() -> None:
    fixture_dirs = sorted(path for path in FIXTURE_ROOT.iterdir() if path.is_dir())
    assert fixture_dirs, "No manual capture fixtures found"
    executed_fixtures = 0

    for fixture_dir in fixture_dirs:
        request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
        expected = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
        if expected.get("pending"):
            continue

        executed_fixtures += 1
        audio_bytes = (fixture_dir / "audio.wav").read_bytes()

        response = client.post(
            "/api/transcriptions",
            data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        )

        assert response.status_code == 200, fixture_dir.name
        payload = response.json()
        primary_notes = primary_note_names(payload)
        note_sets = event_note_sets(payload)
        assertions = expected["assertions"]

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

    if executed_fixtures == 0:
        pytest.skip("No completed manual capture fixtures available")
