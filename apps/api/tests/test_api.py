import json

from conftest import client


def test_health_check() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tunings_endpoint_returns_presets() -> None:
    response = client.get("/api/tunings")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 3
    assert payload[0]["notes"]


def test_custom_tuning_accepts_duplicate_notes() -> None:
    """Chromatic kalimbas (e.g. Lingting 34-key) have duplicate notes across layers."""
    tuning = {"name": "Chromatic 34", "notes": [{"noteName": "C4"}, {"noteName": "D4"}, {"noteName": "C4"}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )
    # Should not reject duplicates - 400 here would be for audio format, not tuning
    assert response.status_code != 400 or "Duplicate" not in response.json().get("detail", "")


def test_custom_tuning_rejects_non_object_json_payload() -> None:
    response = client.post(
        "/api/transcriptions",
        data={"tuning": "[]"},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Tuning JSON must be an object."}


def test_custom_tuning_rejects_non_string_note_name() -> None:
    tuning = {"name": "Invalid", "notes": [{"noteName": 7}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Each tuning noteName must be a non-empty string."}


def test_custom_tuning_rejects_non_object_note_entries() -> None:
    tuning = {"name": "Invalid", "notes": ["C4"]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Each tuning note must be an object."}


def test_custom_tuning_rejects_blank_note_name() -> None:
    tuning = {"name": "Invalid", "notes": [{"noteName": "  "}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Each tuning noteName must be a non-empty string."}


def test_custom_tuning_rejects_non_string_name() -> None:
    tuning = {"name": 17, "notes": [{"noteName": "C4"}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Tuning name must be a string."}
