"""Ablation matrix: enable attack-validated collector, disable each legacy collector individually."""
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
    fixture_dirs_for_status,
    fixture_id,
    load_fixture,
    validate_expected_metadata,
    assertion_failures,
)

pytestmark = [pytest.mark.manual_capture, pytest.mark.slow, pytest.mark.ablation]

COMPLETED_FIXTURES = fixture_dirs_for_status("completed")

ABLATION_FLAGS = [
    "ABLATE_LEADING_ORPHAN",
    "ABLATE_CLOSE_TERMINAL_ORPHAN",
    "ABLATE_DELAYED_TERMINAL_ORPHAN",
    "ABLATE_SINGLE_ONSET_GAP_HEAD",
    "ABLATE_SPARSE_GAP_TAIL",
    "ABLATE_MULTI_ONSET_GAP",
    "ABLATE_POST_TAIL_GAP_HEAD",
    "ABLATE_TERMINAL_MULTI_ONSET",
    "ABLATE_GAP_INJECTED",
    "ABLATE_TWO_ONSET_TERMINAL_TAIL",
    "ABLATE_SUPPLEMENTAL_STARTS",
]


@pytest.fixture()
def _enable_attack_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.transcription as mod
    monkeypatch.setattr(mod, "USE_ATTACK_VALIDATED_GAP_COLLECTOR", True)


@pytest.fixture(params=ABLATION_FLAGS)
def ablated_flag(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, _enable_attack_validated: None) -> str:
    import app.transcription as mod
    flag_name = request.param
    monkeypatch.setattr(mod, flag_name, True)
    return flag_name


@pytest.mark.parametrize("fixture_dir", COMPLETED_FIXTURES, ids=fixture_id)
def test_ablation(fixture_dir: Path, ablated_flag: str) -> None:
    from app.main import app
    client = TestClient(app)

    request_payload, expected = load_fixture(fixture_dir)
    validate_expected_metadata(fixture_dir, expected)
    audio_bytes = build_evaluation_audio_bytes(fixture_dir, expected)

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"])},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, f"{fixture_dir.name} [{ablated_flag}]"
    failures = assertion_failures(fixture_dir, response.json(), expected)
    if failures:
        pytest.fail(f"{fixture_dir.name} [{ablated_flag}]: {'; '.join(failures)}")


if not COMPLETED_FIXTURES:
    pytest.skip("No completed manual capture fixtures available", allow_module_level=True)
