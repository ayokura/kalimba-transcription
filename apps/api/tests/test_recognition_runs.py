"""Recognition run endpoints (#204 Phase 1).

A recording (audio.wav + request.json) is immutable; each re-recognition is
appended as a new run under ``runs/<runId>/`` without minting a new
transaction id. Reads resolve to the newest run, falling back to the legacy
upload-time response.json so pre-#204 clients keep working unchanged.

These tests exercise the wire path (POST /runs actually re-runs the recognizer
on the stored audio) for structural invariants, and use marshaled sentinel run
dicts via ``app.storage`` to assert exact read-resolution without depending on
recognizer output.
"""

import json
import os
import re
from pathlib import Path

from conftest import client, synthesize_note, wav_bytes
from app import storage
from app.fingerprints import kalimba_dsp_fingerprint, recognizer_fingerprint

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-z]{8}$")
MISSING_ID = "00000000-0000-0000-0000-000000000000"


def _tx_root() -> Path:
    return Path(os.environ["KALIMBA_DATA_DIR"]) / "transactions"


def _create_transaction(*, duration: float = 0.5, frequency: float = 261.63) -> str:
    audio = wav_bytes(synthesize_note(frequency, duration=duration))
    tuning = {"name": "Test 17-C", "notes": [{"noteName": "C4"}]}
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning), "force": "true"},
        files={"file": ("audio.wav", audio, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return response.json()["transactionId"]


def test_post_run_appends_without_new_transaction():
    tid = _create_transaction()
    before = {p.name for p in _tx_root().iterdir() if p.is_dir()}

    resp = client.post(f"/api/transcriptions/{tid}/runs")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["transactionId"] == tid
    assert RUN_ID_RE.match(body["runId"]), body["runId"]

    meta = body["meta"]
    assert {"runId", "commitSha", "recognizerFingerprint", "dspFingerprint", "ranAt"} <= set(meta)
    # runId = <UTC ts>-<recognizerFingerprint first 8>
    assert body["runId"].endswith("-" + meta["recognizerFingerprint"][:8])
    assert isinstance(body["result"]["events"], list)

    # The canonical fix for force=true duplication: no new transaction id.
    after = {p.name for p in _tx_root().iterdir() if p.is_dir()}
    assert after == before

    run_dir = _tx_root() / tid / "runs" / body["runId"]
    assert (run_dir / "response.json").exists()
    assert (run_dir / "meta.json").exists()
    # Runs always execute with debug=True internally; the payload goes to
    # debug.json only (see test_run_debug_is_stored_separately_and_kept_out_of_reads).
    assert (run_dir / "debug.json").exists()


def test_run_debug_is_stored_separately_and_kept_out_of_reads():
    """Debug lives only in runs/<runId>/debug.json (no double storage).

    Pins the fix for the review finding: the run executes with debug=True, so
    without popping ``debug`` the run's response.json would embed the full
    debug payload a second time, and every read resolved through
    load_latest_response (GET /api/transcriptions/{id}, dedup) would return the
    inflated document after a re-recognition."""
    tid = _create_transaction()
    body = client.post(f"/api/transcriptions/{tid}/runs").json()

    # The endpoint's own result mirrors the stored (lean) response.
    assert "debug" not in body["result"]

    run_dir = _tx_root() / tid / "runs" / body["runId"]
    stored_response = json.loads((run_dir / "response.json").read_text(encoding="utf-8"))
    assert "debug" not in stored_response
    # The payload itself is preserved, in debug.json only.
    debug_doc = json.loads((run_dir / "debug.json").read_text(encoding="utf-8"))
    assert isinstance(debug_doc, dict) and debug_doc

    # Latest-run resolution serves the lean response too.
    resolved = client.get(f"/api/transcriptions/{tid}").json()
    assert "debug" not in resolved
    assert resolved["events"] == body["result"]["events"]


def test_post_run_unknown_transaction_404():
    assert client.post(f"/api/transcriptions/{MISSING_ID}/runs").status_code == 404


def test_post_run_invalid_id_400():
    assert client.post("/api/transcriptions/not-a-uuid/runs").status_code == 400


def test_legacy_response_file_is_immutable_after_run():
    tid = _create_transaction()
    legacy_path = _tx_root() / tid / "response.json"
    before = legacy_path.read_bytes()
    resp = client.post(f"/api/transcriptions/{tid}/runs")
    assert resp.status_code == 200, resp.text
    assert legacy_path.read_bytes() == before


def test_get_transcription_resolves_to_latest_run():
    tid = _create_transaction()
    # A sentinel run distinguishable from the recognizer output.
    storage.create_run(
        tid,
        {"transactionId": tid, "events": [], "warnings": ["sentinel-run"]},
        None,
        commit_sha="x",
        recognizer_fingerprint="ffffffff00000000",
        dsp_fingerprint=None,
    )
    resolved = client.get(f"/api/transcriptions/{tid}").json()
    assert resolved["warnings"] == ["sentinel-run"]
    assert resolved["events"] == []
    # The detail endpoint still injects recording-level timestamps.
    assert "transcribedAt" in resolved


def test_get_runs_lists_history_newest_first():
    tid = _create_transaction()

    first = client.get(f"/api/transcriptions/{tid}/runs")
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["latestRunId"] == "legacy"
    assert [r["runId"] for r in first_body["runs"]] == ["legacy"]
    assert first_body["runs"][0]["isLegacy"] is True

    run_id = client.post(f"/api/transcriptions/{tid}/runs").json()["runId"]

    second = client.get(f"/api/transcriptions/{tid}/runs").json()
    assert second["latestRunId"] == run_id
    assert [r["runId"] for r in second["runs"]] == [run_id, "legacy"]
    assert [r["isLegacy"] for r in second["runs"]] == [False, True]
    for run in second["runs"]:
        assert "eventCount" in run


def test_get_runs_unknown_transaction_404():
    assert client.get(f"/api/transcriptions/{MISSING_ID}/runs").status_code == 404


def _queue_entry(tid: str) -> dict:
    rows = client.get("/api/review-queue?limit=200").json()
    return next(r for r in rows if r["transactionId"] == tid)


def test_review_queue_flags_stale_when_saved_fingerprint_differs():
    """#204 Phase 2: queue rows expose recognizerFingerprint/isStale so a
    "saved != current recognizer" badge can be shown."""
    tid = _create_transaction()
    current_fp = recognizer_fingerprint()

    # A freshly created transaction's request.json records the current
    # fingerprint, so its (legacy) resolved response is not stale.
    entry = _queue_entry(tid)
    assert entry["recognizerFingerprint"] == current_fp
    assert entry["isStale"] is False

    # A sentinel run recorded with a different fingerprint becomes the newest
    # resolved response, flipping isStale to True.
    storage.create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint="deadbeefcafef00d", dsp_fingerprint=None,
    )
    entry = _queue_entry(tid)
    assert entry["recognizerFingerprint"] == "deadbeefcafef00d"
    assert entry["isStale"] is True


def test_review_queue_flags_stale_when_only_dsp_fingerprint_differs():
    """#209 review: a kalimba_dsp-only change leaves recognizer_fingerprint()
    unchanged (it hashes the Python sources) yet still alters output, so isStale
    keys off a recognizer+dsp composite, not the recognizer alone."""
    tid = _create_transaction()
    current_fp = recognizer_fingerprint()
    current_dsp = kalimba_dsp_fingerprint()

    # recognizer matches AND dsp matches → fresh.
    storage.create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint=current_fp, dsp_fingerprint=current_dsp,
    )
    assert _queue_entry(tid)["isStale"] is False

    # recognizer still matches but dsp differs → stale (the gap this fixes).
    storage.create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint=current_fp, dsp_fingerprint="deadbeefdsp00000",
    )
    assert _queue_entry(tid)["isStale"] is True

    # recognizer matches, dsp unknown (run predating dsp fingerprints) → do not
    # fabricate staleness: fresh.
    storage.create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint=current_fp, dsp_fingerprint=None,
    )
    assert _queue_entry(tid)["isStale"] is False


def test_review_queue_fingerprint_resolves_from_readable_run():
    """#209 review: when the newest run's response.json exists but is unreadable,
    display resolution falls back to the previous good run — and the flagged
    fingerprint must come from that same readable run, not the corrupt newest one
    (else a half-written current-fingerprint run makes an older shown response
    look fresh)."""
    tid = _create_transaction()
    # Older, readable run recorded with a stale recognizer fingerprint.
    older = storage.create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint="0ldfp0000stale00", dsp_fingerprint=None,
    )
    # Newest run: meta claims the current fingerprint, but its response.json is
    # corrupt, so the shown response must skip it for the older good run.
    newest = storage.create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint=recognizer_fingerprint(), dsp_fingerprint=None,
    )
    (storage.get_runs_dir(tid) / newest["runId"] / "response.json").write_text(
        "{ not valid json", encoding="utf-8"
    )

    entry = _queue_entry(tid)
    assert entry["recognizerFingerprint"] == "0ldfp0000stale00"
    assert entry["isStale"] is True

    # /runs latestRunId must follow the displayed (readable) run, not the corrupt
    # newest one, so a client selector stays aligned with what is shown (#209).
    runs_body = client.get(f"/api/transcriptions/{tid}/runs").json()
    assert runs_body["latestRunId"] == older["runId"]


def test_review_queue_isstale_none_when_fingerprint_unknown():
    """Pre-#204 recordings have no recognizerFingerprint in request.json; the
    queue must not guess staleness for them (None, not True/False)."""
    tid = _create_transaction()
    request_path = _tx_root() / tid / "request.json"
    request_data = json.loads(request_path.read_text(encoding="utf-8"))
    request_data.pop("recognizerFingerprint", None)
    request_path.write_text(json.dumps(request_data), encoding="utf-8")

    entry = _queue_entry(tid)
    assert entry["recognizerFingerprint"] is None
    assert entry["isStale"] is None


def test_recent_listing_reflects_latest_run_event_count():
    tid = _create_transaction()
    sentinel_events = [
        {"startTimeSec": 0.0, "durationSec": 0.1, "notes": [{"noteName": "C4"}]}
        for _ in range(7)
    ]
    storage.create_run(
        tid,
        {"transactionId": tid, "events": sentinel_events},
        None,
        commit_sha=None,
        recognizer_fingerprint="aaaabbbbccccdddd",
        dsp_fingerprint=None,
    )
    recent = client.get("/api/transcriptions/recent?limit=100").json()
    entry = next(e for e in recent if e["transactionId"] == tid)
    assert entry["eventCount"] == 7


def test_get_specific_run_by_id():
    tid = _create_transaction()
    run_id = client.post(f"/api/transcriptions/{tid}/runs").json()["runId"]

    resp = client.get(f"/api/transcriptions/{tid}/runs/{run_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body["events"], list)
    assert "transcribedAt" in body
    assert "audioFirstSeenAt" in body


def test_get_legacy_run_by_synthetic_id():
    tid = _create_transaction()
    legacy = storage.load_response(tid)
    # Appending a fresh run must not disturb the legacy synthetic id's content.
    client.post(f"/api/transcriptions/{tid}/runs")

    resp = client.get(f"/api/transcriptions/{tid}/runs/legacy")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["events"] == legacy["events"]


def test_get_run_unknown_run_id_404():
    tid = _create_transaction()
    assert client.get(f"/api/transcriptions/{tid}/runs/not-a-real-run").status_code == 404


def test_get_run_unknown_transaction_404():
    assert client.get(f"/api/transcriptions/{MISSING_ID}/runs/legacy").status_code == 404


def test_get_run_invalid_transaction_id_400():
    assert client.get("/api/transcriptions/not-a-uuid/runs/legacy").status_code == 400


def test_dedup_returns_latest_run_content():
    # dedup fires for tunings whose id matches a server default preset.
    preset = client.get("/api/tunings").json()[0]
    tuning = {
        "id": preset["id"],
        "name": preset["name"],
        "notes": [{"noteName": n["noteName"]} for n in preset["notes"]],
    }
    # A unique duration keeps this recording's audio hash distinct from other
    # tests, so the first (non-force) upload creates a fresh transaction.
    audio = wav_bytes(synthesize_note(261.63, duration=0.61))

    first = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("audio.wav", audio, "audio/wav")},
    )
    assert first.status_code == 200, first.text
    tid = first.json()["transactionId"]

    # Append a sentinel run that is a valid TranscriptionResult (legacy + marker).
    legacy = storage.load_response(tid)
    sentinel = dict(legacy)
    sentinel["warnings"] = list(legacy.get("warnings") or []) + ["dedup-sentinel"]
    storage.create_run(
        tid, sentinel, None,
        commit_sha=None, recognizer_fingerprint="1234567890abcdef", dsp_fingerprint=None,
    )

    # Re-uploading the same audio+tuning without force dedups to the same
    # recording and returns its latest run (not the stale upload snapshot).
    second = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning)},
        files={"file": ("audio.wav", audio, "audio/wav")},
    )
    assert second.status_code == 200, second.text
    body = second.json()
    assert body["transactionId"] == tid  # recording-scoped dedup: no new tx
    assert "dedup-sentinel" in body["warnings"]
