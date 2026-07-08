from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .fingerprints import kalimba_dsp_fingerprint, recognizer_fingerprint


def get_data_dir() -> Path:
    return Path(os.environ.get("KALIMBA_DATA_DIR", "data"))


def get_transactions_dir() -> Path:
    return get_data_dir() / "transactions"


def get_transaction_dir(transaction_id: str) -> Path:
    return get_transactions_dir() / transaction_id


def generate_transaction_id() -> str:
    return str(uuid.uuid4())


def compute_audio_sha256(audio_bytes: bytes) -> str:
    return hashlib.sha256(audio_bytes).hexdigest()


def find_transaction_by_hash_and_tuning(audio_sha256: str, tuning_id: str) -> str | None:
    tx_root = get_transactions_dir()
    if not tx_root.exists():
        return None
    for tx_dir in tx_root.iterdir():
        if not tx_dir.is_dir():
            continue
        request_path = tx_dir / "request.json"
        if not request_path.exists():
            continue
        try:
            data = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stored_hash = data.get("audioSha256")
        stored_tuning_id = (data.get("tuning") or {}).get("id")
        if stored_hash == audio_sha256 and stored_tuning_id == tuning_id:
            return tx_dir.name
    return None


def get_transaction_audio_sha256(transaction_id: str) -> str | None:
    request_path = get_transaction_dir(transaction_id) / "request.json"
    if not request_path.exists():
        return None
    try:
        data = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data.get("audioSha256")


def load_request(transaction_id: str) -> dict | None:
    """The persisted request.json (recording-level, immutable). Used by the
    re-recognition endpoint to reconstruct the tuning + recognition options
    from the stored recording (#204 Phase 1)."""
    request_path = get_transaction_dir(transaction_id) / "request.json"
    if not request_path.exists():
        return None
    try:
        return json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_transactions_by_hash(audio_sha256: str) -> list[dict]:
    tx_root = get_transactions_dir()
    if not tx_root.exists():
        return []
    results: list[dict] = []
    for tx_dir in tx_root.iterdir():
        if not tx_dir.is_dir():
            continue
        request_path = tx_dir / "request.json"
        audio_path = tx_dir / "audio.wav"
        if not request_path.exists() or not audio_path.exists():
            continue
        try:
            data = json.loads(request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("audioSha256") != audio_sha256:
            continue
        entry = _summarize_transaction(tx_dir)
        if entry is not None:
            results.append(entry)
    results.sort(key=lambda e: e["createdAt"], reverse=True)
    return results


def get_transaction_timestamps(transaction_id: str) -> dict | None:
    tx_dir = get_transaction_dir(transaction_id)
    audio_path = tx_dir / "audio.wav"
    if not audio_path.exists():
        return None
    transcribed_at = audio_path.stat().st_mtime
    audio_sha256 = get_transaction_audio_sha256(transaction_id)
    audio_first_seen_at = transcribed_at
    if audio_sha256:
        matches = list_transactions_by_hash(audio_sha256)
        if matches:
            audio_first_seen_at = min(m["createdAt"] for m in matches)
    return {
        "transcribedAt": transcribed_at,
        "audioFirstSeenAt": audio_first_seen_at,
    }


def list_recent_transactions(limit: int) -> list[dict]:
    tx_root = get_transactions_dir()
    if not tx_root.exists():
        return []
    entries: list[tuple[float, Path]] = []
    for tx_dir in tx_root.iterdir():
        if not tx_dir.is_dir():
            continue
        audio_path = tx_dir / "audio.wav"
        if not audio_path.exists():
            continue
        entries.append((audio_path.stat().st_mtime, tx_dir))
    entries.sort(reverse=True)
    results: list[dict] = []
    for _, tx_dir in entries[:limit]:
        entry = _summarize_transaction(tx_dir)
        if entry is not None:
            results.append(entry)
    return results


def list_review_queue(limit: int, status: str | None = None) -> list[dict]:
    """All transactions (newest first) with review/queue metadata, optionally
    filtered by review status. A transaction with no review_status.json is
    treated as ``recorded_only`` for filtering purposes (it was submitted but
    never triaged)."""
    tx_root = get_transactions_dir()
    if not tx_root.exists():
        return []
    entries: list[tuple[float, Path]] = []
    for tx_dir in tx_root.iterdir():
        if not tx_dir.is_dir():
            continue
        audio_path = tx_dir / "audio.wav"
        if not audio_path.exists():
            continue
        entries.append((audio_path.stat().st_mtime, tx_dir))
    entries.sort(reverse=True)
    results: list[dict] = []
    for _, tx_dir in entries:
        entry = _summarize_transaction(tx_dir)
        if entry is None:
            continue
        if status is not None:
            effective = entry.get("reviewStatus") or "recorded_only"
            if effective != status:
                continue
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def resolved_output_fingerprints(transaction_id: str) -> tuple[str | None, str | None]:
    """(recognizerFingerprint, dspFingerprint) of whichever response is currently
    *displayed* for this transaction: the newest run whose response.json actually
    loads, else the legacy request.json values.

    Resolving from the run that ``_load_latest_response_for_dir`` actually reads
    — not merely ``latest_run_id()`` — keeps the flagged fingerprints aligned with
    the response the UI shows (#209 review): a newest run whose response.json is
    present but unreadable must not lend its (fresh) meta to an older response that
    is what actually gets displayed."""
    tx_dir = get_transaction_dir(transaction_id)
    run_id = _latest_readable_run_id_for_dir(tx_dir)
    if run_id is not None:
        meta = load_run_meta(transaction_id, run_id) or {}
        return meta.get("recognizerFingerprint"), meta.get("dspFingerprint")
    request_path = tx_dir / "request.json"
    if not request_path.exists():
        return None, None
    try:
        data = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return data.get("recognizerFingerprint"), data.get("dspFingerprint")


def resolved_recognizer_fingerprint(transaction_id: str) -> str | None:
    """recognizerFingerprint of the displayed response (see
    ``resolved_output_fingerprints``). Kept for external tooling
    (``scripts/audio-analysis/bulk_recognition_runs.py``, #204 Phase 3) that
    reports the recognizer fingerprint on its own."""
    return resolved_output_fingerprints(transaction_id)[0]


def is_response_stale(transaction_id: str) -> bool | None:
    """Whether the displayed response was produced by a different recognizer
    *output* than the one running now — the single source of truth for both the
    review-queue ``isStale`` badge and the bulk re-recognition tool's freshness
    filter (#204 Phase 2 / #209 review).

    Compares a composite of the recognizer (Python transcription sources) and the
    kalimba_dsp (Rust extension) fingerprints, because a DSP-only change leaves
    ``recognizer_fingerprint()`` unchanged yet still alters the output.

    - ``None``  — recognizer fingerprint unknown (pre-#204 recordings); do not guess.
    - ``True``  — recognizer differs, or recognizer matches but a *known* saved DSP
      fingerprint differs from the current one.
    - ``False`` — recognizer matches and the DSP fingerprint matches (or the saved
      DSP fingerprint is unknown, in which case we do not fabricate staleness)."""
    saved_recognizer, saved_dsp = resolved_output_fingerprints(transaction_id)
    if saved_recognizer is None:
        return None
    if saved_recognizer != recognizer_fingerprint():
        return True
    if saved_dsp is not None and saved_dsp != kalimba_dsp_fingerprint():
        return True
    return False


def _summarize_transaction(tx_dir: Path) -> dict | None:
    request_path = tx_dir / "request.json"
    response_path = tx_dir / "response.json"
    if not request_path.exists() or not response_path.exists():
        return None
    try:
        request_data = json.loads(request_path.read_text(encoding="utf-8"))
        response_data = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # #204 Phase 1: listings reflect the latest recognition run when one exists,
    # falling back to the immutable legacy response.json.  Existence of the
    # legacy response is still the gate above (a directory without it is not a
    # transaction), so listing behaviour is unchanged for tx that never had a
    # re-recognition run added.
    latest = _load_latest_response_for_dir(tx_dir)
    if latest is not None:
        response_data = latest
    tuning = request_data.get("tuning") or {}
    audio_path = tx_dir / "audio.wav"
    review_status = _read_review_status(tx_dir)
    corrections_exists = (tx_dir / "corrections.json").exists()
    memo_text = None
    memo_path = tx_dir / "memo.txt"
    if memo_path.exists():
        try:
            memo_text = memo_path.read_text(encoding="utf-8")
        except OSError:
            memo_text = None
    # #204 Phase 2: expose the fingerprint the shown response was produced with,
    # plus whether it differs from the recognizer output running right now, so the
    # review queue can flag re-recognition candidates. isStale is a composite of
    # the recognizer + kalimba_dsp fingerprints (#209 review) and is None when the
    # saved fingerprint is unknown (pre-#204 recordings) rather than guessing stale.
    saved_fingerprint = resolved_recognizer_fingerprint(tx_dir.name)
    is_stale = is_response_stale(tx_dir.name)
    return {
        "transactionId": tx_dir.name,
        "createdAt": audio_path.stat().st_mtime,
        "tuningId": tuning.get("id"),
        "tuningName": tuning.get("name"),
        "eventCount": len(response_data.get("events") or []),
        "audioSha256": request_data.get("audioSha256"),
        "reviewStatus": (review_status or {}).get("status"),
        "reviewStatusUpdatedAt": (review_status or {}).get("updatedAt"),
        "hasCorrections": corrections_exists,
        "hasMemo": bool(memo_text and memo_text.strip()),
        "warningCount": len(response_data.get("warnings") or []),
        "candidateSlotCount": len(response_data.get("candidateSlots") or []),
        # #194 (S6): internal difficulty signal persisted at transcription
        # time.  None for payloads stored before qualityIndicators existed.
        "qualityDifficulty": (response_data.get("qualityIndicators") or {}).get("difficulty"),
        "qualityFlag": (response_data.get("qualityIndicators") or {}).get("flag"),
        "recognizerFingerprint": saved_fingerprint,
        "isStale": is_stale,
    }


def save_transaction(
    transaction_id: str,
    audio_bytes: bytes,
    request_params: dict,
    response_dict: dict,
    debug_dict: dict | None,
) -> None:
    tx_dir = get_transaction_dir(transaction_id)
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "audio.wav").write_bytes(audio_bytes)
    (tx_dir / "request.json").write_text(
        json.dumps(request_params, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tx_dir / "response.json").write_text(
        json.dumps(response_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if debug_dict is not None:
        (tx_dir / "debug.json").write_text(
            json.dumps(debug_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_response(transaction_id: str) -> dict | None:
    """The legacy (upload-time) response.json. Immutable; never overwritten by
    re-recognition. Prefer ``load_latest_response`` for display resolution."""
    response_path = get_transaction_dir(transaction_id) / "response.json"
    if not response_path.exists():
        return None
    return json.loads(response_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Recognition runs (#204 Phase 1)
#
# A recording (audio.wav + request.json) is immutable and can be recognised
# many times as the recognizer improves.  Each re-recognition is appended as a
# new run under ``runs/<runId>/`` without ever touching the legacy
# response.json / debug.json, so pre-#204 clients keep working unchanged while
# reads resolve to the newest run when one exists.
#
#   runId = <UTC ts>-<recognizerFingerprint first 8 chars>
#
# The timestamp is formatted with microsecond precision at fixed width so that
# plain lexicographic sorting of run directory names is chronological (the
# newest run is simply the last one).  ``ranAt`` in meta.json mirrors the same
# instant in ISO-8601.
# ---------------------------------------------------------------------------


def get_runs_dir(transaction_id: str) -> Path:
    return get_transaction_dir(transaction_id) / "runs"


def _format_run_id(ran_at: datetime, recognizer_fingerprint: str | None) -> str:
    stamp = ran_at.strftime("%Y%m%dT%H%M%S.%f") + "Z"
    fingerprint = (recognizer_fingerprint or "unknown")[:8]
    return f"{stamp}-{fingerprint}"


def create_run(
    transaction_id: str,
    response_dict: dict,
    debug_dict: dict | None,
    *,
    commit_sha: str | None,
    recognizer_fingerprint: str | None,
    dsp_fingerprint: str | None,
) -> dict:
    """Append a recognition run and return its meta dict.

    The runId / ranAt are allocated here so the timestamp embedded in the id is
    exactly the recorded ``ranAt``. A uniqueness guard bumps the instant in the
    (practically impossible) event two runs would map to the same id."""
    runs_dir = get_runs_dir(transaction_id)
    ran_at = datetime.now(timezone.utc)
    run_id = _format_run_id(ran_at, recognizer_fingerprint)
    run_dir = runs_dir / run_id
    while run_dir.exists():
        ran_at = datetime.now(timezone.utc)
        run_id = _format_run_id(ran_at, recognizer_fingerprint)
        run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "runId": run_id,
        "commitSha": commit_sha,
        "recognizerFingerprint": recognizer_fingerprint,
        "dspFingerprint": dsp_fingerprint,
        "ranAt": ran_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
    }
    (run_dir / "response.json").write_text(
        json.dumps(response_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if debug_dict is not None:
        (run_dir / "debug.json").write_text(
            json.dumps(debug_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (run_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def list_run_ids(transaction_id: str) -> list[str]:
    """Run ids (ascending / oldest-first) that have a readable response.json.

    Runs missing response.json (e.g. an interrupted write) are skipped so
    resolution never resolves to a half-written run."""
    runs_dir = get_runs_dir(transaction_id)
    if not runs_dir.is_dir():
        return []
    ids = [
        entry.name
        for entry in runs_dir.iterdir()
        if entry.is_dir() and (entry / "response.json").exists()
    ]
    ids.sort()
    return ids


def latest_run_id(transaction_id: str) -> str | None:
    ids = list_run_ids(transaction_id)
    return ids[-1] if ids else None


def load_run_response(transaction_id: str, run_id: str) -> dict | None:
    response_path = get_runs_dir(transaction_id) / run_id / "response.json"
    if not response_path.exists():
        return None
    try:
        return json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_run_meta(transaction_id: str, run_id: str) -> dict | None:
    meta_path = get_runs_dir(transaction_id) / run_id / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _latest_readable_run_id_for_dir(tx_dir: Path) -> str | None:
    """Newest run id whose response.json is present *and* parseable — the run
    ``_load_latest_response_for_dir`` actually displays. None if no run has a
    readable response. Shared so staleness metadata can be read from the same run
    whose response loads (#209 review)."""
    runs_dir = tx_dir / "runs"
    if not runs_dir.is_dir():
        return None
    run_ids = sorted(
        entry.name
        for entry in runs_dir.iterdir()
        if entry.is_dir() and (entry / "response.json").exists()
    )
    for run_id in reversed(run_ids):
        path = runs_dir / run_id / "response.json"
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return run_id
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_latest_response_for_dir(tx_dir: Path) -> dict | None:
    """Newest readable run response for a tx dir, else None. Legacy fallback is
    the caller's responsibility."""
    run_id = _latest_readable_run_id_for_dir(tx_dir)
    if run_id is None:
        return None
    path = tx_dir / "runs" / run_id / "response.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_latest_response(transaction_id: str) -> dict | None:
    """Display-resolution read: newest recognition run, else legacy response.

    This is what pre-#204 clients see through the existing read endpoints, so a
    re-recognised recording shows the newer result without any client change."""
    latest = _load_latest_response_for_dir(get_transaction_dir(transaction_id))
    if latest is not None:
        return latest
    return load_response(transaction_id)


def load_run_or_legacy_response(transaction_id: str, run_id: str) -> dict | None:
    """Full response payload for a specific run id (#204 Phase 2).

    ``run_id == "legacy"`` resolves to the immutable upload-time response, the
    same synthetic id ``list_runs`` uses for it, so a run-switcher UI can treat
    every entry from ``GET .../runs`` uniformly. Returns None when the run (or
    the legacy response) does not exist."""
    if run_id == "legacy":
        return load_response(transaction_id)
    return load_run_response(transaction_id, run_id)


def list_runs(transaction_id: str) -> list[dict]:
    """All recognition runs for a recording, newest-first, each with its meta
    plus a derived ``eventCount``. The immutable legacy response is appended as
    a synthetic ``legacy`` entry (its version info lives in request.json) so the
    full recognition history is visible from one call."""
    runs: list[dict] = []
    for run_id in list_run_ids(transaction_id):
        meta = load_run_meta(transaction_id, run_id) or {}
        response = load_run_response(transaction_id, run_id) or {}
        entry = dict(meta)
        entry["runId"] = run_id
        entry["eventCount"] = len(response.get("events") or [])
        entry["isLegacy"] = False
        runs.append(entry)
    runs.reverse()  # newest first

    legacy_response = load_response(transaction_id)
    if legacy_response is not None:
        request = load_request(transaction_id) or {}
        runs.append(
            {
                "runId": "legacy",
                "commitSha": request.get("commitSha"),
                "recognizerFingerprint": request.get("recognizerFingerprint"),
                "dspFingerprint": request.get("dspFingerprint"),
                "ranAt": None,
                "eventCount": len(legacy_response.get("events") or []),
                "isLegacy": True,
            }
        )
    return runs


def load_audio_path(transaction_id: str) -> Path | None:
    audio_path = get_transaction_dir(transaction_id) / "audio.wav"
    return audio_path if audio_path.exists() else None


def save_memo(transaction_id: str, memo: str) -> None:
    tx_dir = get_transaction_dir(transaction_id)
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "memo.txt").write_text(memo, encoding="utf-8")


def load_memo(transaction_id: str) -> str | None:
    memo_path = get_transaction_dir(transaction_id) / "memo.txt"
    if not memo_path.exists():
        return None
    return memo_path.read_text(encoding="utf-8")


def save_corrections(transaction_id: str, payload: dict) -> None:
    tx_dir = get_transaction_dir(transaction_id)
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "corrections.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_corrections(transaction_id: str) -> dict | None:
    corrections_path = get_transaction_dir(transaction_id) / "corrections.json"
    if not corrections_path.exists():
        return None
    try:
        return json.loads(corrections_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 壊れた JSON (partial write 等) はスキーマ検証層に届かないため、
        # ここで退避しないと次の保存で原本ごと上書きされる
        quarantine_corrections(transaction_id)
        return None


def quarantine_corrections(transaction_id: str) -> None:
    """不正な corrections.json を .invalid に退避する (データは保全しつつ「無し」扱いに)。"""
    corrections_path = get_transaction_dir(transaction_id) / "corrections.json"
    if corrections_path.exists():
        corrections_path.replace(corrections_path.with_suffix(".json.invalid"))


def transaction_exists(transaction_id: str) -> bool:
    return get_transaction_dir(transaction_id).exists()


def save_review_status(transaction_id: str, payload: dict) -> None:
    tx_dir = get_transaction_dir(transaction_id)
    tx_dir.mkdir(parents=True, exist_ok=True)
    (tx_dir / "review_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_review_status(tx_dir: Path) -> dict | None:
    path = tx_dir / "review_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_review_status(transaction_id: str) -> dict | None:
    return _read_review_status(get_transaction_dir(transaction_id))
