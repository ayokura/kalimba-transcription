"""Review status + review queue endpoint tests (tester collection workflow).

review_status.json records where a recording sits in the collection lifecycle so
a tester can submit a recording WITHOUT a full manual transcription:
``recorded_only`` is a valid terminal contribution. ``review_completed`` is the
only status the GT-promotion script treats as ready for ground-truth candidacy.
"""

from io import BytesIO
import json

import numpy as np
import soundfile as sf

from conftest import client


def _make_audio_bytes(freq: float = 261.63) -> bytes:
    sample_rate = 44100
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    wave_data = 0.4 * np.sin(2 * np.pi * freq * t) * np.exp(-4 * t)
    buffer = BytesIO()
    sf.write(buffer, wave_data, sample_rate, format="WAV")
    return buffer.getvalue()


def _create_transaction(freq: float = 261.63) -> str:
    # The tuning only declares C4, so the synthesized note must be C4 for the
    # recognizer to detect events (force=true keeps each transaction distinct
    # regardless of identical audio).
    tuning = {"name": "Test 17-C", "notes": [{"noteName": "C4"}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning), "force": "true"},
        files={"file": ("audio.wav", _make_audio_bytes(freq), "audio/wav")},
    )
    assert response.status_code == 200
    return response.json()["transactionId"]


def test_review_status_missing_returns_null():
    tid = _create_transaction()
    response = client.get(f"/api/transcriptions/{tid}/review-status")
    assert response.status_code == 200
    assert response.json() == {"reviewStatus": None}


def test_review_status_roundtrip():
    tid = _create_transaction()
    payload = {"version": 1, "status": "review_completed", "note": "確認済み", "reviewer": "tester-a"}
    put_response = client.put(f"/api/transcriptions/{tid}/review-status", json=payload)
    assert put_response.status_code == 200
    saved = put_response.json()["reviewStatus"]
    assert saved["status"] == "review_completed"
    assert saved["note"] == "確認済み"
    assert saved["reviewer"] == "tester-a"
    assert saved["updatedAt"]

    get_response = client.get(f"/api/transcriptions/{tid}/review-status")
    assert get_response.status_code == 200
    assert get_response.json()["reviewStatus"] == saved


def test_review_status_overwrite_replaces_previous():
    tid = _create_transaction()
    client.put(f"/api/transcriptions/{tid}/review-status", json={"version": 1, "status": "recorded_only"})
    client.put(f"/api/transcriptions/{tid}/review-status", json={"version": 1, "status": "rerecord_needed"})
    response = client.get(f"/api/transcriptions/{tid}/review-status")
    assert response.json()["reviewStatus"]["status"] == "rerecord_needed"


def test_review_status_rejects_unknown_status():
    tid = _create_transaction()
    response = client.put(
        f"/api/transcriptions/{tid}/review-status",
        json={"version": 1, "status": "definitely_not_a_status"},
    )
    assert response.status_code == 422


def test_review_status_rejects_unsupported_version():
    tid = _create_transaction()
    response = client.put(
        f"/api/transcriptions/{tid}/review-status",
        json={"version": 2, "status": "recorded_only"},
    )
    assert response.status_code == 422


def test_review_status_unknown_transaction_404():
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/transcriptions/{missing}/review-status").status_code == 404
    assert (
        client.put(
            f"/api/transcriptions/{missing}/review-status",
            json={"version": 1, "status": "recorded_only"},
        ).status_code
        == 404
    )


def test_review_status_invalid_id_400():
    assert client.get("/api/transcriptions/not-a-uuid/review-status").status_code == 400


def test_review_queue_includes_metadata_and_status():
    tid = _create_transaction()
    client.put(f"/api/transcriptions/{tid}/review-status", json={"version": 1, "status": "review_started"})
    client.put(
        f"/api/transcriptions/{tid}/corrections",
        json={"version": 1, "events": [{"timeSec": 0.1, "notes": ["C4"]}]},
    )

    response = client.get("/api/review-queue?limit=200")
    assert response.status_code == 200
    rows = response.json()
    entry = next((r for r in rows if r["transactionId"] == tid), None)
    assert entry is not None
    assert entry["reviewStatus"] == "review_started"
    assert entry["hasCorrections"] is True
    assert "warningCount" in entry
    assert "candidateSlotCount" in entry


def test_review_queue_status_filter():
    tid = _create_transaction()
    client.put(f"/api/transcriptions/{tid}/review-status", json={"version": 1, "status": "unusable"})

    response = client.get("/api/review-queue?status=unusable&limit=200")
    assert response.status_code == 200
    rows = response.json()
    assert all(r["reviewStatus"] == "unusable" for r in rows)
    assert any(r["transactionId"] == tid for r in rows)


def test_review_queue_recorded_only_default_for_untriaged():
    tid = _create_transaction()
    # No review-status set: should appear under the recorded_only filter.
    response = client.get("/api/review-queue?status=recorded_only&limit=200")
    assert response.status_code == 200
    rows = response.json()
    assert any(r["transactionId"] == tid for r in rows)


def test_review_queue_rejects_invalid_status_filter():
    assert client.get("/api/review-queue?status=bogus").status_code == 400
