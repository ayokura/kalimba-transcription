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
    assertion_failures,
)

pytestmark = [pytest.mark.manual_capture, pytest.mark.slow]

client = TestClient(app)
COMPLETED_FIXTURES = fixture_dirs_for_status("completed")


def assert_fixture_output(fixture_dir: Path, payload: dict, expected: dict) -> None:
    failures = assertion_failures(fixture_dir, payload, expected)
    assert not failures, f"{fixture_dir.name}: {'; '.join(failures)}"




@pytest.mark.parametrize("fixture_dir", COMPLETED_FIXTURES, ids=fixture_id)
def test_completed_manual_capture_regression(fixture_dir: Path) -> None:
    request_payload, expected = load_fixture(fixture_dir)
    validate_expected_metadata(fixture_dir, expected)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"])},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, fixture_dir.name
    assert_fixture_output(fixture_dir, response.json(), expected)


if not COMPLETED_FIXTURES:
    pytest.skip("No completed manual capture fixtures available", allow_module_level=True)
