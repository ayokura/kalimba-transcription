from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "audio-analysis" / "note_f1_benchmark.py"
    spec = importlib.util.spec_from_file_location("note_f1_benchmark", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def _note(name: str) -> dict:
    return {"pitchClass": name[:-1], "octave": int(name[-1])}


def test_candidate_metrics_correction_burden_and_source_counts() -> None:
    payload = {
        "events": [
            {
                "id": "evt-1",
                "startTimeSec": 1.0,
                "notes": [_note("C4")],
                "alternateGroupings": [
                    {
                        "alternateNote": _note("D4"),
                        "reason": "soft_rejected:test",
                        "confidence": 0.8,
                    }
                ],
            },
            {"id": "evt-2", "startTimeSec": 2.0, "notes": [_note("E4")]},
            {"id": "evt-3", "startTimeSec": 3.0, "notes": [_note("G4")]},
        ],
        "candidateSlots": [
            {
                "startTime": 2.05,
                "endTime": 2.2,
                "primaryNote": _note("F4"),
                "candidates": [],
                "dropReason": "orphan-onset-no-segment",
                "confidence": 0.5,
            }
        ],
        "debug": {
            "segmentCandidates": [
                {
                    "startTime": 1.0,
                    "endTime": 1.1,
                    "rankedCandidates": [{"noteName": "D4", "score": 10.0}],
                    "residualCandidates": [{"noteName": "A4", "score": 1.0}],
                }
            ]
        },
    }
    truth = [
        {"time": 1.0, "note": "C4", "tol": 0.05},
        {"time": 1.0, "note": "D4", "tol": 0.05},
        {"time": 2.05, "note": "F4", "tol": 0.05},
        {"time": 4.0, "note": "A4", "tol": 0.05},
    ]

    outcome = bench.evaluate_payload(payload, truth)

    assert outcome["oneBest"]["onsetPrecision"] == pytest.approx(1 / 3)
    assert outcome["oneBest"]["onsetRecall"] == pytest.approx(1 / 4)
    assert outcome["candidates"]["recallAt1"] == pytest.approx(1 / 4)
    assert outcome["candidates"]["recallAt3"] == pytest.approx(3 / 4)
    assert outcome["candidates"]["hardMisses"] == 1
    assert outcome["candidates"]["candidateAssistedHits"] == 2
    assert outcome["candidates"]["rankedDiagnostic"]["recallAtK"]["1"] == pytest.approx(1 / 4)
    # hardMissNotes must be a public (non ``_``-prefixed) key so the --json and
    # --verbose hard-miss listings are reachable; A4 is the only unrecoverable GT.
    assert outcome["candidates"]["hardMissNotes"] == [{"time": 4.0, "note": "A4"}]
    assert not any(k.startswith("_") for k in outcome["candidates"])

    assert outcome["correction"]["estimatedCost"] == 7
    assert outcome["correction"]["noteRemoves"] == 2
    assert outcome["correction"]["candidateEnabled"] == 2
    assert outcome["correction"]["manualInserts"] == 1
    assert outcome["correction"]["candidateAssistedFixRate"] == pytest.approx(2 / 3)

    assert outcome["candidateSources"] == {
        "oneBestNotes": 3,
        "alternateGroupingNotes": 1,
        "softAlternateNotes": 1,
        "droppedCandidateNotes": 1,
        "debugRankedCandidateNotes": 1,
        "debugResidualCandidateNotes": 1,
    }


def test_confidence_calibration_uses_alternate_groupings_as_event_flags() -> None:
    payload = {
        "events": [
            {
                "id": "evt-1",
                "startTimeSec": 1.0,
                "notes": [_note("C4")],
                "alternateGroupings": [
                    {
                        "alternateNote": _note("D4"),
                        "reason": "soft_rejected:test",
                        "confidence": 0.8,
                    }
                ],
            },
            {"id": "evt-2", "startTimeSec": 2.0, "notes": [_note("E4")]},
        ],
        "candidateSlots": [],
        "debug": {},
    }
    truth = [
        {"time": 1.0, "note": "C4", "tol": 0.05},
        {"time": 1.0, "note": "D4", "tol": 0.05},
    ]

    outcome = bench.evaluate_payload(payload, truth)

    assert outcome["confidence"]["flaggedEventPrecision"] == pytest.approx(1.0)
    assert outcome["confidence"]["missedErrorRate"] == pytest.approx(0.5)
    assert outcome["confidence"]["highConfidenceWrongRate"] == pytest.approx(1.0)
    assert outcome["confidence"]["lowConfidenceCorrectRate"] == pytest.approx(0.0)
    assert outcome["confidence"]["candidateHighConfidenceWrongRate"] == pytest.approx(0.0)
