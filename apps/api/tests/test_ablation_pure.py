"""Pure ablation: disable each legacy collector individually WITHOUT the attack-validated replacement."""
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

pytestmark = [pytest.mark.manual_capture, pytest.mark.slow, pytest.mark.ablation]

COMPLETED_FIXTURES = fixture_dirs_for_status("completed")

ABLATION_FLAGS = [
    "ablate_sparse_gap_tail",
    "ablate_multi_onset_gap",
    "ablate_collapse_active_range_head",
    "ablate_snap_range_start_to_onset",
]


@pytest.fixture(params=ABLATION_FLAGS)
def ablated_flag(request: pytest.FixtureRequest) -> str:
    from app.transcription import settings
    flag_name = request.param
    ctx = settings.override(**{flag_name: True})
    ctx.__enter__()
    request.addfinalizer(lambda: ctx.__exit__(None, None, None))
    return flag_name


@pytest.mark.parametrize("fixture_dir", COMPLETED_FIXTURES, ids=fixture_id)
def test_pure_ablation(fixture_dir: Path, ablated_flag: str) -> None:
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

    assert response.status_code == 200, f"{fixture_dir.name} [{ablated_flag}]"
    failures = assertion_failures(fixture_dir, response.json(), expected)
    if failures:
        pytest.fail(f"{fixture_dir.name} [{ablated_flag}]: {'; '.join(failures)}")


if not COMPLETED_FIXTURES:
    pytest.skip("No completed manual capture fixtures available", allow_module_level=True)
