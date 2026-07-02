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


def test_repo_managed_corpus_discovery_precedes_local_transactions(tmp_path, monkeypatch) -> None:
    tx_id = "tx-corpus"
    corpus_root = tmp_path / "free-performance-corpus"
    corpus_dir = corpus_root / tx_id
    corpus_dir.mkdir(parents=True)
    for name in ("audio.wav", "request.json", "ground_truth.json"):
        (corpus_dir / name).write_text("{}", encoding="utf-8")

    local_root = tmp_path / "transactions"
    local_dir = local_root / tx_id
    local_dir.mkdir(parents=True)
    (local_dir / "audio.wav").write_text("local-audio", encoding="utf-8")
    (local_dir / "request.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(bench, "FREE_PERFORMANCE_CORPUS_DIR", corpus_root)
    monkeypatch.setattr(bench, "DATA_DIR", local_root)
    monkeypatch.setattr(bench, "CAPTURES_DIR", tmp_path / "transaction-captures")

    assert bench.discover_tx_ids() == [tx_id]
    assert bench.transaction_dir_for(tx_id) == corpus_dir
    assert bench.ground_truth_path_for(tx_id) == corpus_dir / "ground_truth.json"


def test_mir_eval_compat_uses_fixed_tolerance() -> None:
    """Per-onset toleranceSec (e.g. 0.2 for inserted-manual) must not leak
    into the strict mir_eval-compatible report value (fixed 50ms)."""
    payload = {"events": [{"id": "e1", "startTimeSec": 1.10, "notes": [_note("C4")]}]}
    truth = [{"time": 1.0, "note": "C4", "tol": 0.2}]
    lenient = bench.match_pairs(truth, bench.collect_one_best(payload))
    assert lenient["tp"] == 1  # wide per-onset tolerance matches at dt=0.10
    strict = bench.mir_eval_compat_metrics(payload, truth)
    assert strict["tp"] == 0  # 50ms fixed tolerance does not
    assert strict["toleranceSec"] == 0.05
    assert strict["onsetF1"] == 0.0


def test_bootstrap_ci_is_deterministic_and_bounded() -> None:
    results = [
        {"tp": 10, "truthNotes": 10, "predictedNotes": 10},
        {"tp": 8, "truthNotes": 10, "predictedNotes": 9},
        {"tp": 5, "truthNotes": 10, "predictedNotes": 11},
    ]
    first = bench.bootstrap_micro_f1_ci(results, iterations=200)
    second = bench.bootstrap_micro_f1_ci(results, iterations=200)
    assert first == second  # seeded → reproducible
    low, high = first["microF1CI95"]
    assert 0.0 <= low <= high <= 1.0
    assert bench.bootstrap_micro_f1_ci([]) is None
