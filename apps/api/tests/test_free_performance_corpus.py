"""Tier 4: Corpus benchmark regression gate for the repo-managed corpus.

Unlike the Tier 2 fixture suite (exact event-sequence match against
``expected.json``), this layer asserts a *distance floor* against human-derived
ground truth: per-recording note F1 must not fall below the recorded baseline
and hard misses must not exceed the recorded ceiling. The assertion authority
is ``free-performance-corpus/benchmark_baseline.json`` (see docs/testing.md
"Tier 4" and docs/corpus-management.md for the update discipline: baselines
move in the improvement direction only, via
``note_f1_benchmark.py --write-baseline``).

Also machine-checks the corpus governance rules from docs/corpus-management.md:
committed recordings must carry a human rights review
(``rightsReview.status == "approved_for_repository"``) and a cleared copyright
status. This is what lets an agent trust that whatever sits in this directory
was human-approved.

Scope note: only git-tracked corpus recordings run here. Local-only recordings
(transaction-captures + data/transactions) are covered by running
``note_f1_benchmark.py --check-baseline`` manually after recognizer changes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = REPO_ROOT / "apps" / "api" / "tests" / "fixtures" / "free-performance-corpus"
REQUIRED_FILES = ("audio.wav", "request.json", "ground_truth.json", "metadata.json")
# Extend deliberately (docs/corpus-management.md); never add a status here to
# make a failing recording pass.
CLEARED_COPYRIGHT_STATUSES = {"original_performance", "public_domain"}


def _load_benchmark_module():
    script = REPO_ROOT / "scripts" / "audio-analysis" / "note_f1_benchmark.py"
    spec = importlib.util.spec_from_file_location("note_f1_benchmark", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def _corpus_tx_ids() -> list[str]:
    if not CORPUS_DIR.is_dir():
        return []
    return sorted(
        d.name
        for d in CORPUS_DIR.iterdir()
        if d.is_dir() and (d / "ground_truth.json").is_file()
    )


CORPUS_TX_IDS = _corpus_tx_ids()

if not CORPUS_TX_IDS:
    pytest.skip(
        "no repo-managed free-performance corpus recordings present",
        allow_module_level=True,
    )

BASELINE = bench.load_baseline()


@pytest.mark.parametrize("tx_id", CORPUS_TX_IDS)
def test_corpus_recording_governance(tx_id: str) -> None:
    tx_dir = CORPUS_DIR / tx_id
    for name in REQUIRED_FILES:
        assert (tx_dir / name).is_file(), (
            f"{tx_id}: missing required corpus file {name} (docs/corpus-management.md)"
        )
    metadata = json.loads((tx_dir / "metadata.json").read_text(encoding="utf-8"))
    rights_status = (metadata.get("rightsReview") or {}).get("status")
    assert rights_status == "approved_for_repository", (
        f"{tx_id}: rightsReview.status is {rights_status!r} — repo-managed corpus "
        "requires a human rights review before commit"
    )
    copyright_status = (metadata.get("copyright") or {}).get("status")
    assert copyright_status in CLEARED_COPYRIGHT_STATUSES, (
        f"{tx_id}: copyright.status {copyright_status!r} is not a cleared status "
        f"({sorted(CLEARED_COPYRIGHT_STATUSES)})"
    )


@pytest.mark.parametrize("tx_id", CORPUS_TX_IDS)
def test_corpus_recording_has_baseline_entry(tx_id: str) -> None:
    assert tx_id in BASELINE.get("recordings", {}), (
        f"{tx_id}: no entry in benchmark_baseline.json — run "
        "`uv run python scripts/audio-analysis/note_f1_benchmark.py --write-baseline` "
        "and commit the result together with the recording"
    )


@pytest.fixture(scope="module")
def corpus_outcomes() -> dict[str, dict]:
    """Transcribe each corpus recording once and evaluate against its GT."""
    client = TestClient(bench.app)
    outcomes: dict[str, dict] = {}
    for tx_id in CORPUS_TX_IDS:
        truth = bench.load_ground_truth(bench.ground_truth_path_for(tx_id))
        payload = bench.transcribe_payload(client, tx_id, debug=True)
        outcomes[tx_id] = bench.evaluate_payload(payload, truth)
    return outcomes


@pytest.mark.parametrize("tx_id", CORPUS_TX_IDS)
def test_corpus_benchmark_gate(corpus_outcomes: dict[str, dict], tx_id: str) -> None:
    entry = BASELINE.get("recordings", {}).get(tx_id)
    if entry is None:
        pytest.fail(f"{tx_id}: baseline entry missing (see companion test)")
    outcome = corpus_outcomes[tx_id]
    f1 = outcome["f1"]
    hard_misses = int(outcome["candidates"]["hardMisses"])
    assert f1 >= float(entry["minF1"]) - 1e-9, (
        f"{tx_id}: note F1 {f1:.3f} fell below baseline minF1 {entry['minF1']:.3f}. "
        "This is a free-performance regression. If it is an intentional tradeoff, "
        "follow the fixture-policy procedure and update the baseline with "
        "--write-baseline --allow-baseline-regression; never lower the baseline "
        "just to make CI pass."
    )
    assert hard_misses <= int(entry["maxHardMisses"]), (
        f"{tx_id}: hardMisses {hard_misses} exceeds baseline max "
        f"{entry['maxHardMisses']} (notes absent from one-best AND all candidates)"
    )
