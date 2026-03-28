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


def test_custom_tuning_rejects_duplicate_notes() -> None:
    tuning = {"name": "Bad Tuning", "notes": [{"noteName": "C4"}, {"noteName": "D4"}, {"noteName": "C4"}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("test.wav", b"\x00" * 44, "audio/wav")},
    )
    assert response.status_code == 400
    assert "Duplicate" in response.json()["detail"]
