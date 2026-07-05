import json
import re

from app.storage import (
    create_run,
    generate_transaction_id,
    get_runs_dir,
    get_transaction_dir,
    latest_run_id,
    list_run_ids,
    list_runs,
    load_audio_path,
    load_latest_response,
    load_response,
    load_run_meta,
    load_run_response,
    save_transaction,
)

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
RUN_ID_RE = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-z]{8}$")

_AUDIO = b"RIFF" + b"\x00" * 100


def test_generate_transaction_id_is_uuid():
    tid = generate_transaction_id()
    assert UUID_RE.match(tid), f"Not a valid UUID: {tid}"


def test_generate_transaction_id_unique():
    ids = {generate_transaction_id() for _ in range(100)}
    assert len(ids) == 100


def test_save_and_load_response():
    tid = generate_transaction_id()
    audio = b"RIFF" + b"\x00" * 100
    request_params = {"tuning": {"id": "test"}, "debug": False}
    response = {"transactionId": tid, "events": []}

    save_transaction(tid, audio, request_params, response, None)

    loaded = load_response(tid)
    assert loaded == response


def test_save_with_debug():
    tid = generate_transaction_id()
    audio = b"RIFF" + b"\x00" * 100
    request_params = {"tuning": {"id": "test"}, "debug": True}
    response = {"transactionId": tid, "events": [], "debug": {"info": "test"}}
    debug_dict = {"info": "test"}

    save_transaction(tid, audio, request_params, response, debug_dict)

    tx_dir = get_transaction_dir(tid)
    assert (tx_dir / "debug.json").exists()
    assert json.loads((tx_dir / "debug.json").read_text()) == debug_dict


def test_load_audio_path():
    tid = generate_transaction_id()
    audio = b"RIFF" + b"\x00" * 100

    save_transaction(tid, audio, {}, {}, None)

    path = load_audio_path(tid)
    assert path is not None
    assert path.read_bytes() == audio


def test_load_response_nonexistent():
    assert load_response("00000000-0000-0000-0000-000000000000") is None


def test_load_audio_path_nonexistent():
    assert load_audio_path("00000000-0000-0000-0000-000000000000") is None


# --- Recognition runs (#204 Phase 1) -------------------------------------


def test_latest_response_falls_back_to_legacy_without_runs():
    tid = generate_transaction_id()
    save_transaction(tid, _AUDIO, {"tuning": {"id": "t"}}, {"events": [{"n": "legacy"}]}, None)

    assert latest_run_id(tid) is None
    assert list_run_ids(tid) == []
    # No runs -> resolution returns the immutable legacy response.
    assert load_latest_response(tid) == {"events": [{"n": "legacy"}]}


def test_create_run_appends_and_becomes_latest():
    tid = generate_transaction_id()
    save_transaction(tid, _AUDIO, {"tuning": {"id": "t"}}, {"events": [{"n": "legacy"}]}, None)

    meta = create_run(
        tid,
        {"events": [{"n": "run1"}]},
        {"trace": 1},
        commit_sha="abc123",
        recognizer_fingerprint="deadbeefcafef00d",
        dsp_fingerprint="absent",
    )

    assert RUN_ID_RE.match(meta["runId"]), meta["runId"]
    # runId = <UTC ts>-<recognizerFingerprint first 8>
    assert meta["runId"].endswith("-deadbeef")
    assert meta["commitSha"] == "abc123"
    assert meta["recognizerFingerprint"] == "deadbeefcafef00d"
    assert meta["dspFingerprint"] == "absent"
    assert meta["ranAt"].endswith("Z")

    # Resolution now returns the run; legacy response is untouched on disk.
    assert latest_run_id(tid) == meta["runId"]
    assert load_latest_response(tid) == {"events": [{"n": "run1"}]}
    assert load_response(tid) == {"events": [{"n": "legacy"}]}

    run_dir = get_runs_dir(tid) / meta["runId"]
    assert json.loads((run_dir / "response.json").read_text()) == {"events": [{"n": "run1"}]}
    assert json.loads((run_dir / "debug.json").read_text()) == {"trace": 1}
    assert load_run_meta(tid, meta["runId"]) == meta


def test_create_run_without_debug_omits_debug_file():
    tid = generate_transaction_id()
    save_transaction(tid, _AUDIO, {"tuning": {"id": "t"}}, {"events": []}, None)
    meta = create_run(
        tid, {"events": []}, None,
        commit_sha=None, recognizer_fingerprint="00112233aabbccdd", dsp_fingerprint=None,
    )
    run_dir = get_runs_dir(tid) / meta["runId"]
    assert (run_dir / "response.json").exists()
    assert not (run_dir / "debug.json").exists()


def test_multiple_runs_latest_wins_and_list_is_newest_first():
    tid = generate_transaction_id()
    save_transaction(
        tid, _AUDIO,
        {
            "tuning": {"id": "t"},
            "commitSha": "c0",
            "recognizerFingerprint": "legacyfp00000000",
            "dspFingerprint": "d0",
        },
        {"events": [1]}, None,
    )
    m1 = create_run(tid, {"events": [1, 2]}, None,
                    commit_sha="c1", recognizer_fingerprint="aaaaaaaaaaaaaaaa", dsp_fingerprint="d1")
    m2 = create_run(tid, {"events": [1, 2, 3]}, None,
                    commit_sha="c2", recognizer_fingerprint="bbbbbbbbbbbbbbbb", dsp_fingerprint="d2")

    # Lexicographic order of run ids is chronological.
    assert m2["runId"] > m1["runId"]
    assert list_run_ids(tid) == [m1["runId"], m2["runId"]]
    assert latest_run_id(tid) == m2["runId"]
    assert load_latest_response(tid) == {"events": [1, 2, 3]}

    runs = list_runs(tid)
    assert [r["runId"] for r in runs] == [m2["runId"], m1["runId"], "legacy"]
    assert [r["eventCount"] for r in runs] == [3, 2, 1]
    assert [r["isLegacy"] for r in runs] == [False, False, True]
    # Legacy synthetic entry carries the version info persisted in request.json.
    assert runs[-1]["recognizerFingerprint"] == "legacyfp00000000"
    assert runs[-1]["commitSha"] == "c0"


def test_runs_with_same_fingerprint_get_unique_ids():
    tid = generate_transaction_id()
    save_transaction(tid, _AUDIO, {"tuning": {"id": "t"}}, {"events": []}, None)
    m1 = create_run(tid, {"events": [1]}, None,
                    commit_sha=None, recognizer_fingerprint="samefp0000000000", dsp_fingerprint=None)
    m2 = create_run(tid, {"events": [2]}, None,
                    commit_sha=None, recognizer_fingerprint="samefp0000000000", dsp_fingerprint=None)
    assert m1["runId"] != m2["runId"]
    assert len(list_run_ids(tid)) == 2
    assert load_latest_response(tid) == {"events": [2]}


def test_latest_response_skips_corrupt_latest_run():
    tid = generate_transaction_id()
    save_transaction(tid, _AUDIO, {"tuning": {"id": "t"}}, {"events": ["legacy"]}, None)
    m1 = create_run(tid, {"events": ["run1"]}, None,
                    commit_sha=None, recognizer_fingerprint="1111111111111111", dsp_fingerprint=None)
    m2 = create_run(tid, {"events": ["run2"]}, None,
                    commit_sha=None, recognizer_fingerprint="2222222222222222", dsp_fingerprint=None)
    # Corrupt the newest run's response.json (e.g. an interrupted write).
    (get_runs_dir(tid) / m2["runId"] / "response.json").write_text("{ not json", encoding="utf-8")
    # Resolution skips the unreadable run and returns the previous good one.
    assert load_latest_response(tid) == {"events": ["run1"]}
    assert load_run_response(tid, m1["runId"]) == {"events": ["run1"]}
