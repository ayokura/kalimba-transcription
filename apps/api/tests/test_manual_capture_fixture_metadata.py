import json
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.append(str(TESTS_DIR))

from manual_capture_helpers import build_evaluation_audio_bytes, fixture_id, list_fixture_dirs, load_fixture, validate_expected_metadata, validate_request_metadata

pytestmark = pytest.mark.manual_capture

ALL_FIXTURES = list_fixture_dirs()


@pytest.mark.parametrize("fixture_dir", ALL_FIXTURES, ids=fixture_id)
def test_manual_capture_fixture_metadata(fixture_dir: Path) -> None:
    request_payload, expected = load_fixture(fixture_dir)

    assert (fixture_dir / "audio.wav").exists(), fixture_dir.name
    assert (fixture_dir / "request.json").exists(), fixture_dir.name
    assert (fixture_dir / "expected.json").exists(), fixture_dir.name
    assert (fixture_dir / "notes.md").exists(), fixture_dir.name
    assert isinstance(request_payload.get("tuning"), dict), fixture_dir.name

    validate_request_metadata(fixture_dir, request_payload)
    validate_expected_metadata(fixture_dir, expected)
    build_evaluation_audio_bytes(fixture_dir, expected)


@pytest.mark.parametrize("fixture_dir", ALL_FIXTURES, ids=fixture_id)
def test_manual_capture_request_and_expected_are_json_serializable(fixture_dir: Path) -> None:
    request_payload, expected = load_fixture(fixture_dir)
    assert json.loads(json.dumps(request_payload)) == request_payload
    assert json.loads(json.dumps(expected)) == expected
