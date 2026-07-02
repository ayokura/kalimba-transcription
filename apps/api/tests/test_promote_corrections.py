"""Tests for the corrections -> ground-truth promotion script.

The review-status gate is the key safety rail: only ``review_completed``
recordings are promoted by default, so an untriaged ``recorded_only`` recording
never becomes a GT candidate by accident.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audio-analysis" / "promote_corrections_to_ground_truth.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("promote_corrections", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promote = _load_module()


def _write_tx(data_dir: Path, tx_id: str, *, status: str | None, with_corrections: bool = True) -> None:
    tx = data_dir / tx_id
    tx.mkdir(parents=True, exist_ok=True)
    (tx / "audio.wav").write_bytes(b"RIFFfake")
    if with_corrections:
        (tx / "corrections.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "updatedAt": "2026-06-27T00:00:00+00:00",
                    "events": [
                        {"timeSec": 1.0, "notes": ["C4"], "origin": "recognizer"},
                        {"timeSec": 2.0, "notes": ["E4"], "origin": "inserted-slot"},
                    ],
                }
            ),
            encoding="utf-8",
        )
    if status is not None:
        (tx / "review_status.json").write_text(
            json.dumps({"version": 1, "status": status}), encoding="utf-8"
        )


@pytest.fixture()
def patched(tmp_path, monkeypatch):
    data_dir = tmp_path / "transactions"
    captures_dir = tmp_path / "captures"
    corpus_dir = tmp_path / "corpus"
    data_dir.mkdir()
    captures_dir.mkdir()
    corpus_dir.mkdir()
    monkeypatch.setattr(promote, "DATA_DIR", data_dir)
    monkeypatch.setattr(promote, "CAPTURES_DIR", captures_dir)
    monkeypatch.setattr(promote, "CORPUS_DIR", corpus_dir)
    return data_dir, captures_dir, corpus_dir


def _run(*argv: str) -> int:
    import sys

    old = sys.argv
    sys.argv = ["promote", *argv]
    try:
        return promote.main()
    finally:
        sys.argv = old


def test_load_review_status(patched):
    data_dir, _, _ = patched
    _write_tx(data_dir, "tx-1", status="review_completed")
    assert promote.load_review_status("tx-1") == "review_completed"
    _write_tx(data_dir, "tx-2", status=None)
    assert promote.load_review_status("tx-2") is None


def test_completed_status_is_promoted(patched):
    data_dir, captures_dir, _ = patched
    _write_tx(data_dir, "tx-done", status="review_completed")
    rc = _run("tx-done")
    assert rc == 0
    gt_path = captures_dir / "tx-done" / "ground_truth.json"
    assert gt_path.is_file()
    gt = json.loads(gt_path.read_text())
    assert gt["source"]["reviewStatus"] == "review_completed"
    assert gt["source"]["provenance"] == "tester_corrected"
    assert gt["source"]["timingAccuracy"]["onsetTiming"] == "approximate"
    assert {o["method"] for o in gt["onsets"]} == {"user_corrected"}


def test_recorded_only_is_skipped_by_default(patched):
    data_dir, captures_dir, _ = patched
    _write_tx(data_dir, "tx-raw", status="recorded_only")
    rc = _run("tx-raw")
    assert rc == 0
    assert not (captures_dir / "tx-raw" / "ground_truth.json").exists()


def test_missing_status_is_skipped_by_default(patched):
    data_dir, captures_dir, _ = patched
    _write_tx(data_dir, "tx-nostatus", status=None)
    rc = _run("tx-nostatus")
    assert rc == 0
    assert not (captures_dir / "tx-nostatus" / "ground_truth.json").exists()


def test_ignore_status_bypasses_gate(patched):
    data_dir, captures_dir, _ = patched
    _write_tx(data_dir, "tx-legacy", status=None)
    rc = _run("tx-legacy", "--ignore-status")
    assert rc == 0
    assert (captures_dir / "tx-legacy" / "ground_truth.json").is_file()


def test_require_status_override(patched):
    data_dir, captures_dir, _ = patched
    _write_tx(data_dir, "tx-uncertain", status="uncertain")
    _run("tx-uncertain")
    assert not (captures_dir / "tx-uncertain" / "ground_truth.json").exists()
    _run("tx-uncertain", "--require-status", "uncertain")
    assert (captures_dir / "tx-uncertain" / "ground_truth.json").is_file()


def test_existing_gt_not_overwritten_without_force(patched):
    data_dir, captures_dir, _ = patched
    _write_tx(data_dir, "tx-existing", status="review_completed")
    gt_dir = captures_dir / "tx-existing"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "ground_truth.json").write_text(
        json.dumps({"version": 1, "onsets": [{"timeSec": 9.9, "notes": ["G4"], "method": "ear_verified"}]}),
        encoding="utf-8",
    )
    _run("tx-existing")
    gt = json.loads((gt_dir / "ground_truth.json").read_text())
    assert gt["onsets"][0]["method"] == "ear_verified"


def test_duplicate_audio_in_corpus_blocks_promotion(patched):
    """Same audio already committed to the repo-managed corpus → skip.

    Without the corpus layer in the SHA dedup, a re-upload of promoted audio
    would gain a second GT under a new tx-id and be double-counted by the
    benchmark (2026-07-02 audit finding)."""
    data_dir, captures_dir, corpus_dir = patched
    _write_tx(data_dir, "tx-reupload", status="review_completed")
    corpus_tx = corpus_dir / "tx-original"
    corpus_tx.mkdir(parents=True)
    # Same bytes as _write_tx writes for audio.wav.
    (corpus_tx / "audio.wav").write_bytes(b"RIFFfake")
    (corpus_tx / "ground_truth.json").write_text(
        json.dumps({"version": 1, "onsets": []}), encoding="utf-8"
    )
    rc = _run("tx-reupload")
    assert rc == 0
    assert not (captures_dir / "tx-reupload" / "ground_truth.json").exists()

    rc = _run("tx-reupload", "--allow-duplicate")
    assert rc == 0
    assert (captures_dir / "tx-reupload" / "ground_truth.json").is_file()


def test_to_corpus_requires_explicit_rights_decision(patched):
    data_dir, _, _ = patched
    _write_tx(data_dir, "tx-c", status="review_completed")
    with pytest.raises(SystemExit) as excinfo:
        _run("tx-c", "--to-corpus")
    assert excinfo.value.code == 2


def test_to_corpus_scaffolds_repo_corpus_entry(patched):
    data_dir, captures_dir, corpus_dir = patched
    _write_tx(data_dir, "tx-c", status="review_completed")
    tx = data_dir / "tx-c"
    (tx / "request.json").write_text(
        json.dumps({"tuning": {"id": "kalimba-17-c"}}), encoding="utf-8"
    )
    (tx / "response.json").write_text(
        json.dumps({"events": [{"id": "e1"}], "candidateSlots": [{"x": 1}]}),
        encoding="utf-8",
    )

    rc = _run(
        "tx-c",
        "--to-corpus",
        "--copyright-status", "original_performance",
        "--rights-reviewed-by", "human requester",
        "--device", "Test Phone",
    )
    assert rc == 0

    dest = corpus_dir / "tx-c"
    for name in (
        "audio.wav",
        "request.json",
        "corrections.json",
        "review_status.json",
        "ground_truth.json",
        "metadata.json",
    ):
        assert (dest / name).is_file(), name
    meta = json.loads((dest / "metadata.json").read_text())
    assert meta["rightsReview"]["status"] == "approved_for_repository"
    assert meta["rightsReview"]["reviewedBy"] == "human requester"
    assert meta["copyright"]["status"] == "original_performance"
    assert meta["tuning"]["selectedId"] == "kalimba-17-c"
    assert meta["aggregates"]["correctedEventCount"] == 2
    assert meta["aggregates"]["recognizerEventCount"] == 1
    assert meta["aggregates"]["candidateSlotCount"] == 1
    assert meta["aggregates"]["originCounts"] == {"recognizer": 1, "inserted-slot": 1}
    assert meta["recording"]["device"] == "Test Phone"
    # b"RIFFfake" is not a decodable WAV: stats must degrade to None, not crash.
    assert meta["recording"]["peakDb"] is None

    # GT already existed (written by the same run); a second --to-corpus run
    # without --force-corpus must not clobber the scaffold.
    rc = _run(
        "tx-c",
        "--to-corpus",
        "--copyright-status", "original_performance",
        "--rights-reviewed-by", "human requester",
    )
    assert rc == 0  # prints SKIP corpus, exits cleanly
