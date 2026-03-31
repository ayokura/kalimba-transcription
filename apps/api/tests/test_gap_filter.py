"""Test: gap onset filtering by attack profile enabled."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from manual_capture_helpers import (
    build_evaluation_audio_bytes,
    build_transcription_form_data,
    fixture_dirs_for_status,
    fixture_id,
    load_fixture,
    validate_expected_metadata,
    assertion_failures,
)

pytestmark = [pytest.mark.manual_capture, pytest.mark.slow]

COMPLETED_FIXTURES = fixture_dirs_for_status("completed")


@pytest.fixture(autouse=True)
def _enable_gap_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.transcription as mod
    monkeypatch.setattr(mod.segments, "FILTER_GAP_ONSETS_BY_ATTACK_PROFILE", True)


@pytest.mark.parametrize("fixture_dir", COMPLETED_FIXTURES, ids=fixture_id)
def test_gap_filter(fixture_dir: Path) -> None:
    from app.main import app
    client = TestClient(app)

    request_payload, expected = load_fixture(fixture_dir)
    validate_expected_metadata(fixture_dir, expected)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data=build_transcription_form_data(request_payload),
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, fixture_dir.name
    failures = assertion_failures(fixture_dir, response.json(), expected)
    if failures:
        pytest.fail(f"{fixture_dir.name}: {'; '.join(failures)}")


if not COMPLETED_FIXTURES:
    pytest.skip("No completed manual capture fixtures available", allow_module_level=True)
