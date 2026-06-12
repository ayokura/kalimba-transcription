"""Corrections endpoint tests (review UI persistence).

corrections.json stores the user-verified event timeline in the same
time/notes vocabulary as ground_truth.json so saved corrections can be
promoted to F1-benchmark ground truth without transformation.
"""

from io import BytesIO
import json

import numpy as np
import soundfile as sf

from conftest import client


def _make_audio_bytes() -> bytes:
    sample_rate = 44100
    t = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
    wave_data = 0.4 * np.sin(2 * np.pi * 261.63 * t) * np.exp(-4 * t)
    buffer = BytesIO()
    sf.write(buffer, wave_data, sample_rate, format="WAV")
    return buffer.getvalue()


def _create_transaction() -> str:
    tuning = {"name": "Test 17-C", "notes": [{"noteName": "C4"}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning), "force": "true"},
        files={"file": ("audio.wav", _make_audio_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    return response.json()["transactionId"]


def test_corrections_missing_returns_null():
    tid = _create_transaction()
    response = client.get(f"/api/transcriptions/{tid}/corrections")
    assert response.status_code == 200
    assert response.json() == {"corrections": None}


def test_corrections_roundtrip():
    tid = _create_transaction()
    payload = {
        "version": 1,
        "events": [
            {"timeSec": 1.05, "notes": ["C4"], "origin": "recognizer"},
            {"timeSec": 2.5, "notes": ["D4", "D5"], "origin": "edited"},
            {"timeSec": 3.75, "notes": ["E4"], "origin": "inserted-manual"},
        ],
    }
    put_response = client.put(f"/api/transcriptions/{tid}/corrections", json=payload)
    assert put_response.status_code == 200
    saved = put_response.json()["corrections"]
    assert saved["version"] == 1
    assert saved["updatedAt"]
    assert [e["notes"] for e in saved["events"]] == [["C4"], ["D4", "D5"], ["E4"]]

    get_response = client.get(f"/api/transcriptions/{tid}/corrections")
    assert get_response.status_code == 200
    assert get_response.json()["corrections"] == saved


def test_corrections_overwrite_replaces_previous():
    tid = _create_transaction()
    first = {"version": 1, "events": [{"timeSec": 1.0, "notes": ["C4"]}]}
    second = {"version": 1, "events": []}
    client.put(f"/api/transcriptions/{tid}/corrections", json=first)
    client.put(f"/api/transcriptions/{tid}/corrections", json=second)
    response = client.get(f"/api/transcriptions/{tid}/corrections")
    assert response.json()["corrections"]["events"] == []


def test_corrections_rejects_empty_notes():
    tid = _create_transaction()
    payload = {"version": 1, "events": [{"timeSec": 1.0, "notes": []}]}
    response = client.put(f"/api/transcriptions/{tid}/corrections", json=payload)
    assert response.status_code == 422


def test_corrections_rejects_unknown_origin():
    tid = _create_transaction()
    payload = {"version": 1, "events": [{"timeSec": 1.0, "notes": ["C4"], "origin": "guessed"}]}
    response = client.put(f"/api/transcriptions/{tid}/corrections", json=payload)
    assert response.status_code == 422


def test_corrections_rejects_unsupported_version():
    tid = _create_transaction()
    payload = {"version": 2, "events": [{"timeSec": 1.0, "notes": ["C4"]}]}
    response = client.put(f"/api/transcriptions/{tid}/corrections", json=payload)
    assert response.status_code == 422


def test_corrections_invalid_file_is_quarantined_on_read():
    from app.storage import get_transaction_dir

    tid = _create_transaction()
    tx_dir = get_transaction_dir(tid)
    corrections_path = tx_dir / "corrections.json"
    # スキーマ非互換 (origin typo) のファイルを直接置く
    corrections_path.write_text(
        json.dumps(
            {"version": 1, "events": [{"timeSec": 1.0, "notes": ["C4"], "origin": "typo"}]}
        ),
        encoding="utf-8",
    )

    response = client.get(f"/api/transcriptions/{tid}/corrections")
    assert response.status_code == 200
    assert response.json() == {"corrections": None}
    # 原本は .invalid に退避され、データは失われない
    assert not corrections_path.exists()
    assert (tx_dir / "corrections.json.invalid").exists()


def test_corrections_unknown_transaction_404():
    missing_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/transcriptions/{missing_id}/corrections").status_code == 404
    payload = {"version": 1, "events": []}
    assert (
        client.put(f"/api/transcriptions/{missing_id}/corrections", json=payload).status_code
        == 404
    )
