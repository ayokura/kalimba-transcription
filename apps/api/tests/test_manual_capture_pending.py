import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from manual_capture_helpers import build_evaluation_audio_bytes, build_transcription_form_data, fixture_dirs_for_status, fixture_id, load_fixture
from app.main import app

pytestmark = [pytest.mark.manual_capture, pytest.mark.slow]

client = TestClient(app)
PENDING_FIXTURES = fixture_dirs_for_status("pending")


@pytest.mark.parametrize("fixture_dir", PENDING_FIXTURES, ids=fixture_id)
def test_pending_manual_capture_smoke(fixture_dir: Path) -> None:
    request_payload, expected = load_fixture(fixture_dir)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions", data=build_transcription_form_data(request_payload), files={"file": ("audio.wav", audio_bytes, "audio/wav")}
    )

    assert response.status_code == 200, fixture_dir.name
    payload = response.json()
    assert isinstance(payload.get("events"), list), fixture_dir.name
    assert payload["events"], fixture_dir.name


if not PENDING_FIXTURES:
    pytest.skip("No pending manual capture fixtures available", allow_module_level=True)
