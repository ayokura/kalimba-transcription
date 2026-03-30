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
    build_transcription_form_data,
    event_note_sets,
    fixture_dirs_for_status,
    fixture_id,
    ground_truth_timing_failures,
    load_fixture,
    load_ground_truth,
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
        data=build_transcription_form_data(request_payload),
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, fixture_dir.name
    payload = response.json()
    assert_fixture_output(fixture_dir, payload, expected)

    ground_truth = load_ground_truth(fixture_dir)
    if ground_truth is not None:
        # Re-run with debug=true for timing check (uses full audio, not windowed)
        debug_form_data = build_transcription_form_data(request_payload)
        debug_form_data["debug"] = "true"
        debug_response = client.post(
            "/api/transcriptions",
            data=debug_form_data,
            files={"file": ("audio.wav", (fixture_dir / "audio.wav").read_bytes(), "audio/wav")},
        )
        timing_failures = ground_truth_timing_failures(fixture_dir, debug_response.json(), ground_truth)
        assert not timing_failures, f"{fixture_dir.name} timing: {'; '.join(timing_failures)}"


if not COMPLETED_FIXTURES:
    pytest.skip("No completed manual capture fixtures available", allow_module_level=True)
