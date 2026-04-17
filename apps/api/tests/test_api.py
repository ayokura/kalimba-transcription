import json
import re

from conftest import client, synthesize_note, wav_bytes


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


UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _create_transcription():
    """Helper: POST a synthetic audio to /api/transcriptions, return response JSON."""
    import numpy as np
    audio = synthesize_note(261.63, duration=0.5)
    audio_data = wav_bytes(audio)
    tuning = {
        "name": "Test 17-C",
        "notes": [{"noteName": "C4"}],
    }
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("audio.wav", audio_data, "audio/wav")},
    )
    assert response.status_code == 200
    return response.json()


def test_transcription_response_includes_transaction_id():
    payload = _create_transcription()
    assert "transactionId" in payload
    assert UUID_RE.match(payload["transactionId"])


def test_get_transcription_by_id():
    payload = _create_transcription()
    tid = payload["transactionId"]

    response = client.get(f"/api/transcriptions/{tid}")
    assert response.status_code == 200
    data = response.json()
    assert data["transactionId"] == tid
    assert data["events"] == payload["events"]


def test_get_transcription_audio_by_id():
    payload = _create_transcription()
    tid = payload["transactionId"]

    response = client.get(f"/api/transcriptions/{tid}/audio")
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert len(response.content) > 44  # WAV header + some data


def test_get_transcription_not_found():
    response = client.get("/api/transcriptions/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_get_transcription_invalid_id_format():
    response = client.get("/api/transcriptions/not-a-uuid")
    assert response.status_code == 400
    assert "Invalid transaction ID" in response.json()["detail"]


def test_get_transcription_audio_not_found():
    response = client.get("/api/transcriptions/00000000-0000-0000-0000-000000000000/audio")
    assert response.status_code == 404


def test_memo_get_default_empty():
    payload = _create_transcription()
    tid = payload["transactionId"]

    response = client.get(f"/api/transcriptions/{tid}/memo")
    assert response.status_code == 200
    assert response.json() == {"memo": ""}


def test_memo_put_and_get():
    payload = _create_transcription()
    tid = payload["transactionId"]

    put_response = client.put(
        f"/api/transcriptions/{tid}/memo",
        json={"memo": "BWV147 練習メモ。テンポ遅めで。"},
    )
    assert put_response.status_code == 200

    get_response = client.get(f"/api/transcriptions/{tid}/memo")
    assert get_response.status_code == 200
    assert get_response.json() == {"memo": "BWV147 練習メモ。テンポ遅めで。"}


def test_memo_put_overwrites():
    payload = _create_transcription()
    tid = payload["transactionId"]

    client.put(f"/api/transcriptions/{tid}/memo", json={"memo": "first"})
    client.put(f"/api/transcriptions/{tid}/memo", json={"memo": "second"})

    response = client.get(f"/api/transcriptions/{tid}/memo")
    assert response.json() == {"memo": "second"}


def test_memo_get_not_found():
    response = client.get("/api/transcriptions/00000000-0000-0000-0000-000000000000/memo")
    assert response.status_code == 404


def test_memo_put_not_found():
    response = client.put(
        "/api/transcriptions/00000000-0000-0000-0000-000000000000/memo",
        json={"memo": "x"},
    )
    assert response.status_code == 404


def test_memo_get_invalid_id():
    response = client.get("/api/transcriptions/not-a-uuid/memo")
    assert response.status_code == 400


def test_transcription_response_includes_start_time_sec():
    payload = _create_transcription()
    assert len(payload["events"]) > 0
    first = payload["events"][0]
    assert "startTimeSec" in first
    assert isinstance(first["startTimeSec"], float)
    assert first["startTimeSec"] >= 0


def test_event_start_times_are_monotonic():
    payload = _create_transcription()
    events = payload["events"]
    assert len(events) > 0
    prev = -1.0
    for event in events:
        assert event["startTimeSec"] >= prev, "startTimeSec must be non-decreasing"
        prev = event["startTimeSec"]
