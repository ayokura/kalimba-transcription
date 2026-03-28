import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
from manual_capture_helpers import assertion_failure_details
from scripts.explain_manual_capture import REPEATED_PATTERN_PASS_IDS, build_explanation, print_text

ROOT = Path(__file__).resolve().parents[3]
pytestmark = [pytest.mark.manual_capture, pytest.mark.slow]


def explain_fixture(fixture_id: str, *disabled_passes: str) -> dict:
    fixture_dir = ROOT / "apps" / "api" / "tests" / "fixtures" / "manual-captures" / fixture_id
    return build_explanation(fixture_dir, disabled_passes=list(disabled_passes))


def render_text(summary: dict) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_text(summary)
    return buffer.getvalue()


def run_explain_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT / "apps" / "api"))
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "explain_manual_capture.py"), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        check=True,
    )


def test_explain_manual_capture_text_output() -> None:
    summary = explain_fixture("kalimba-17-c-c4-repeat-01")
    output = render_text(summary)
    assert "fixture: kalimba-17-c-c4-repeat-01" in output
    assert "status: completed" in output
    assert "sourceProfile: acoustic_real" in output
    assert "evaluationMode: full_audio" in output


def test_explain_manual_capture_cli_smoke() -> None:
    result = run_explain_cli("kalimba-17-c-c4-repeat-01", "--json")
    payload = json.loads(result.stdout)
    assert payload["fixtureId"] == "kalimba-17-c-c4-repeat-01"
    assert payload["status"] == "completed"
    assert payload["eventCount"] >= 1


def test_explain_manual_capture_json_output() -> None:
    payload = explain_fixture("kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03")
    assert payload["fixtureId"] == "kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03"
    assert payload["status"] == "completed"
    assert payload["sourceProfile"] == "acoustic_real"
    assert payload["captureIntent"] == "strict_chord"
    assert payload["defaultCaptureIntent"] == "strict_chord"
    assert all(intent == "strict_chord" for intent in payload["eventIntents"])
    assert len(payload["expectedEvents"]) == 6
    assert all(event["intent"] == "strict_chord" for event in payload["expectedEvents"])
    assert payload["expectedSummary"] == "E4 + G4 + B4 + D5 x 6"
    assert payload["segmentCandidates"]
    assert payload["mergedEvents"]
    assert payload["segmentDetection"]["segments"]
    first_segment = payload["segmentCandidates"][0]
    assert "broadbandOnsetGain" in first_segment
    assert "spectralFlux" in first_segment
    assert "localOnsetCount" in first_segment
    first_ranked = first_segment["rankedCandidates"][0]
    assert "attackEnergy" in first_ranked
    assert "attackToSustainRatio" in first_ranked
    assert "candidateOnsetGain" in first_ranked


def test_explain_manual_capture_json_output_with_ignored_ranges_reports_scope() -> None:
    payload = explain_fixture("kalimba-17-c-e4-g4-b4-d5-four-note-rolled-repeat-01")
    assert payload["evaluationMode"] == "ignored_ranges"
    assert payload["sourceDurationSec"] > payload["evaluationDurationSec"]
    assert any(event["scope"] == "out_of_scope" for event in payload["eventCoverage"])
    assert any(event["scope"] == "in_scope" for event in payload["eventCoverage"])




def test_assertion_failure_details_report_order_mismatch() -> None:
    payload = {
        "events": [
            {"notes": [{"pitchClass": "C", "octave": 4}]},
            {"notes": [{"pitchClass": "E", "octave": 4}]},
        ]
    }
    expected = {
        "assertions": {
            "expectedEventNoteSetsOrdered": ["E4", "C4"],
        }
    }

    details = assertion_failure_details(ROOT / "apps" / "api" / "tests" / "fixtures" / "manual-captures" / "kalimba-17-c-c4-repeat-01", payload, expected)

    assert any(item["code"] == "event_order_mismatch" for item in details)


def test_explain_manual_capture_reports_pending_summary_hints() -> None:
    payload = explain_fixture("kalimba-17-c-c4-to-g4-sequence-17-01")
    assert payload["eventCompression"]["expected"] == 17
    assert payload["eventCompression"]["detected"] == 17
    assert len(payload["expectedEvents"]) == 17
    assert all(event["intent"] is None for event in payload["expectedEvents"])
    assert payload["dominantGestureMix"]["ambiguous"] >= 1
    assert payload["normalizationSummary"]["segmentCount"] >= payload["normalizationSummary"]["rawEventCount"]
    assert payload["normalizationSummary"]["rawEventCount"] >= payload["normalizationSummary"]["mergedEventCount"]
    assert payload["normalizationReasonCounts"]
    assert any(item["rule"] in ("merged_duplicate", "simplified_to_subset") for item in payload["normalizationDecisions"])
    assert isinstance(payload["phraseBreakGuess"], list)


def test_explain_manual_capture_reports_disabled_pass_trace() -> None:
    payload = explain_fixture("kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03", "normalize_repeated_triad_patterns")
    assert payload["disabledRepeatedPatternPasses"] == ["normalize_repeated_triad_patterns"]
    pass_trace = next(item for item in payload["repeatedPatternPassTrace"] if item["pass"] == "normalize_repeated_triad_patterns")
    assert pass_trace["enabled"] is False


def test_explain_manual_capture_lists_repeated_pattern_passes() -> None:
    lines = [line.strip() for line in REPEATED_PATTERN_PASS_IDS if line.strip()]
    assert "normalize_repeated_triad_patterns" in lines
