"""Tests for the benchmark<->queue join (review priority scoring).

The scorer is the contract: a recording is worth reviewing next when it is both
OPEN for human attention and HARD for the recognizer. Closed recordings keep a
residual score (still listed) but never outrank an equally-hard open one.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audio-analysis" / "review_priority_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("review_priority_report", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_module()


def _metrics(f1: float, hard: float, cost: float) -> dict:
    return {
        "oneBest": {"onsetF1": f1},
        "candidates": {"hardMissRate": hard},
        "correction": {"costPerTruthNote": cost},
    }


def test_clean_completed_recording_is_low_priority():
    out = report.compute_priority(_metrics(1.0, 0.0, 0.0), "review_completed")
    assert out["priority"] == 0.0
    assert out["open"] is False


def test_hard_open_recording_outranks_hard_closed_recording():
    metrics = _metrics(0.6, 0.2, 1.0)
    open_score = report.compute_priority(metrics, "recorded_only")["priority"]
    closed_score = report.compute_priority(metrics, "review_completed")["priority"]
    assert open_score > closed_score


def test_missing_status_is_treated_as_open():
    out = report.compute_priority(_metrics(0.5, 0.1, 0.5), None)
    assert out["open"] is True
    assert out["priority"] > 0


def test_difficulty_increases_with_worse_metrics():
    easy = report.compute_priority(_metrics(0.95, 0.0, 0.1), "recorded_only")["difficulty"]
    hard = report.compute_priority(_metrics(0.5, 0.3, 1.5), "recorded_only")["difficulty"]
    assert hard > easy


def test_cost_per_truth_is_capped():
    # Runaway cost-per-note should not dominate the score without bound.
    capped = report.compute_priority(_metrics(1.0, 0.0, 50.0), "recorded_only")
    reference = report.compute_priority(_metrics(1.0, 0.0, 2.0), "recorded_only")
    assert capped["difficulty"] == reference["difficulty"]


def test_reason_summarizes_drivers():
    out = report.compute_priority(_metrics(0.7, 0.1, 0.4), "recorded_only")
    assert "F1=" in out["reason"]
    assert "hardMiss=" in out["reason"]
    assert "cost/GT=" in out["reason"]


def test_unusable_is_closed():
    out = report.compute_priority(_metrics(0.4, 0.4, 1.0), "unusable")
    assert out["open"] is False
