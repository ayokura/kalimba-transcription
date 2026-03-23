import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "explain_manual_capture.py"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "apps" / "api"))
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=True,
    )


def test_explain_manual_capture_text_output() -> None:
    result = run_script("kalimba-17-c-c4-repeat-01")
    assert "fixture: kalimba-17-c-c4-repeat-01" in result.stdout
    assert "status: completed" in result.stdout
    assert "sourceProfile: acoustic_real" in result.stdout
    assert "evaluationMode: full_audio" in result.stdout


def test_explain_manual_capture_json_output() -> None:
    result = run_script("kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03", "--json")
    payload = json.loads(result.stdout)
    assert payload["fixtureId"] == "kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03"
    assert payload["status"] == "completed"
    assert payload["sourceProfile"] == "acoustic_real"
    assert payload["captureIntent"] == "strict_chord"
    assert payload["defaultCaptureIntent"] == "strict_chord"
    assert all(intent == "strict_chord" for intent in payload["eventIntents"])
    assert payload["expectedSummary"] == "E4 + G4 + B4 + D5 x 6"

def test_explain_manual_capture_json_output_with_ignored_ranges_reports_scope() -> None:
    result = run_script("kalimba-17-c-e4-g4-b4-d5-four-note-rolled-repeat-01", "--json")
    payload = json.loads(result.stdout)
    assert payload["evaluationMode"] == "ignored_ranges"
    assert payload["sourceDurationSec"] > payload["evaluationDurationSec"]
    assert any(event["scope"] == "out_of_scope" for event in payload["eventCoverage"])
    assert any(event["scope"] == "in_scope" for event in payload["eventCoverage"])


