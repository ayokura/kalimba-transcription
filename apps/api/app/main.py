from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from .fingerprints import git_head_sha, kalimba_dsp_fingerprint, recognizer_fingerprint
from .models import (
    CorrectionsPayload,
    InstrumentTuning,
    ReviewStatusPayload,
    TranscriptionResult,
)
from .storage import (
    compute_audio_sha256,
    find_transaction_by_hash_and_tuning,
    generate_transaction_id,
    get_data_dir,
    get_transaction_audio_sha256,
    get_transaction_timestamps,
    list_recent_transactions,
    list_review_queue,
    list_transactions_by_hash,
    load_audio_path,
    load_corrections,
    load_memo,
    load_response,
    load_review_status,
    quarantine_corrections,
    save_corrections,
    save_memo,
    save_review_status,
    save_transaction,
    transaction_exists,
)
from .transcription import parse_tuning_json, transcribe_audio
from .transcription.patterns import REPEATED_PATTERN_PASS_IDS
from .tunings import get_default_tunings


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _validate_transaction_id(transaction_id: str) -> None:
    if not _UUID_RE.match(transaction_id):
        raise HTTPException(status_code=400, detail="Invalid transaction ID format.")


def parse_disabled_repeated_pattern_passes(raw_value: str | None) -> frozenset[str]:
    if raw_value is None or not raw_value.strip():
        return frozenset()

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw_value.split(",") if item.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(status_code=400, detail="disabledRepeatedPatternPasses must be a JSON string array or comma-separated string.")

    disabled = frozenset(item.strip() for item in parsed if item.strip())
    unknown = sorted(disabled - set(REPEATED_PATTERN_PASS_IDS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown repeated-pattern pass ids: {', '.join(unknown)}")
    return disabled


app = FastAPI(title="Kalimba Score API", version="0.1.0")


def _parse_allowed_origins(raw: str | None) -> list[str]:
    if raw is None:
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


_ALLOWED_ORIGINS = _parse_allowed_origins(os.environ.get("KALIMBA_ALLOWED_ORIGINS"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tunings", response_model=list[InstrumentTuning])
def list_tunings() -> list[InstrumentTuning]:
    return get_default_tunings()


@app.post("/api/transcriptions", response_model=TranscriptionResult)
async def create_transcription(
    http_request: Request,
    file: UploadFile = File(...),
    tuning: str = Form(...),
    debug: bool = Form(True),
    disabledRepeatedPatternPasses: str | None = Form(None),
    midPerformanceStart: bool = Form(False),
    midPerformanceEnd: bool = Form(False),
    force: bool = Form(False),
    dryRun: bool = Form(False),
    scenario: str | None = Form(None),
    expectedNote: str | None = Form(None),
    expectedPerformance: str | None = Form(None),
    memo: str | None = Form(None),
    captureIntent: str | None = Form(None),
    sourceProfile: str | None = Form(None),
) -> TranscriptionResult:
    audio_bytes = await file.read()
    await file.seek(0)

    parsed_tuning = parse_tuning_json(tuning)
    audio_sha256 = compute_audio_sha256(audio_bytes)

    # capture メタデータ (期待列など) はサーバー側 request.json に永続化する。
    # 2026-07-04 監査: 従来これらはクライアントの Capture Pack ZIP にしか
    # 残らず、data/transactions 側では失われていた (/debug/triage の敵対的
    # 録音で発覚)。GT 化の自動整列は保存された expectedPerformance に依存する。
    expected_performance: dict | None = None
    if expectedPerformance:
        try:
            parsed_expected = json.loads(expectedPerformance)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="expectedPerformance must be valid JSON.") from exc
        if not isinstance(parsed_expected, dict) or not isinstance(parsed_expected.get("events"), list):
            raise HTTPException(
                status_code=400,
                detail="expectedPerformance must be an object with an events array.",
            )
        expected_performance = parsed_expected

    if not dryRun and not force:
        existing_id = find_transaction_by_hash_and_tuning(audio_sha256, parsed_tuning.id)
        if existing_id is not None:
            existing = load_response(existing_id)
            if existing is not None:
                return TranscriptionResult.model_validate(existing)

    disabled_passes = parse_disabled_repeated_pattern_passes(disabledRepeatedPatternPasses)
    result = await transcribe_audio(
        file,
        parsed_tuning,
        debug=debug,
        disabled_repeated_pattern_passes=disabled_passes,
        mid_performance_start=midPerformanceStart,
        mid_performance_end=midPerformanceEnd,
    )

    if dryRun:
        return result

    transaction_id = generate_transaction_id()
    result.transaction_id = transaction_id

    request_params = {
        "tuning": json.loads(tuning),
        "debug": debug,
        "disabledRepeatedPatternPasses": disabledRepeatedPatternPasses,
        "midPerformanceStart": midPerformanceStart,
        "midPerformanceEnd": midPerformanceEnd,
        "audioSha256": audio_sha256,
        "commitSha": git_head_sha(),
        "recognizerFingerprint": recognizer_fingerprint(),
        "dspFingerprint": kalimba_dsp_fingerprint(),
    }
    if scenario and scenario.strip():
        request_params["scenario"] = scenario.strip()
    if expectedNote and expectedNote.strip():
        request_params["expectedNote"] = expectedNote.strip()
    if expected_performance is not None:
        request_params["expectedPerformance"] = expected_performance
    if captureIntent and captureIntent.strip():
        request_params["captureIntent"] = captureIntent.strip()
    if sourceProfile and sourceProfile.strip():
        request_params["sourceProfile"] = sourceProfile.strip()
    # 録音デバイス推定の手がかり (テスター録音の recording-profile 較正用,
    # 2026-07-04 テスターFB)。tunnel/Next proxy は元の User-Agent を素通しする。
    user_agent = (http_request.headers.get("user-agent") or "").strip()
    if user_agent:
        request_params["client"] = {"userAgent": user_agent[:512]}
    response_dict = result.model_dump(by_alias=True)
    debug_dict = response_dict.get("debug")

    save_transaction(transaction_id, audio_bytes, request_params, response_dict, debug_dict)
    if memo and memo.strip():
        save_memo(transaction_id, memo.strip())

    return result


@app.get("/api/transcriptions/by-hash/{audio_sha256}")
def get_transcription_by_hash(audio_sha256: str, tuning: str) -> dict:
    if not _SHA256_RE.match(audio_sha256):
        raise HTTPException(status_code=400, detail="Invalid SHA-256 format.")
    existing_id = find_transaction_by_hash_and_tuning(audio_sha256, tuning)
    if existing_id is None:
        raise HTTPException(status_code=404, detail="No matching transcription found.")
    return {"transactionId": existing_id}


@app.get("/api/transcriptions/recent")
def get_recent_transcriptions(limit: int = 10) -> list[dict]:
    capped = max(1, min(limit, 100))
    return list_recent_transactions(capped)


_REVIEW_STATUS_VALUES = {
    "recorded_only",
    "review_started",
    "review_completed",
    "rerecord_needed",
    "unusable",
    "uncertain",
}


@app.get("/api/review-queue")
def get_review_queue(limit: int = 50, status: str | None = None) -> list[dict]:
    capped = max(1, min(limit, 200))
    if status is not None and status not in _REVIEW_STATUS_VALUES:
        raise HTTPException(status_code=400, detail="Invalid review status filter.")
    return list_review_queue(capped, status=status)


# Dev-only (第 2 期 S1 の計器修理): transactions_triage.py の出力に live の
# review status を重ねて /debug/triage ページへ供給する。temporary — 開発が
# 落ち着いたら /debug/triage と一緒に撤去する (sprint-plan-2026-07b S1)。
@app.get("/api/dev/triage")
def get_dev_triage() -> dict:
    summary_path = get_data_dir() / "triage_summary.json"
    if not summary_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "triage_summary.json not found. Run "
                "`uv run python scripts/audio-analysis/transactions_triage.py` first."
            ),
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for recording in summary.get("recordings", []):
        statuses: dict[str, str | None] = {}
        for tx_id in [recording.get("primaryTx"), *recording.get("duplicateTxs", [])]:
            if not tx_id:
                continue
            payload = load_review_status(tx_id)
            statuses[tx_id] = payload.get("status") if payload else None
        recording["reviewStatuses"] = statuses
    return summary


# Dev-only (第 2 期 S2 の GT 化フロー): gt_draft.py が出力した行データを
# /debug/gt-review ページへ供給し、ユーザーの裁定 (verdict) を保存する。
# temporary — GT 化の運用が落ち着いたら /debug/gt-review と一緒に撤去する。
_GT_DRAFT_TX8_RE = re.compile(r"^[0-9a-f]{8}$")


class GtDraftVerdictPayload(BaseModel):
    # 行 index (文字列化した 1-based) -> {decision?: accept|fix|ignore, notes?, comment?}
    rows: dict[str, dict]
    # 未配置 expected index -> {decision: discard|place, timeSec?: float}
    unplaced: dict[str, dict] = {}
    # 認識器が完全に見逃した音のユーザー手動追加 [{timeSec, notes, comment?}]
    added: list[dict] = []
    # 録音全体への自由コメント
    comment: str = ""
    done: bool = False


@app.get("/api/dev/gt-drafts")
def get_dev_gt_drafts() -> dict:
    drafts_dir = get_data_dir() / "gt_drafts"
    rows_files = sorted(drafts_dir.glob("*.rows.json")) if drafts_dir.is_dir() else []
    if not rows_files:
        raise HTTPException(
            status_code=404,
            detail=(
                "No GT drafts found. Run "
                "`uv run python scripts/audio-analysis/gt_draft.py <tx-prefix>` first."
            ),
        )
    drafts = []
    for path in rows_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        verdict_path = drafts_dir / f"{doc['tx8']}.verdict.json"
        doc["verdict"] = (
            json.loads(verdict_path.read_text(encoding="utf-8"))
            if verdict_path.is_file()
            else None
        )
        drafts.append(doc)
    return {"drafts": drafts}


@app.put("/api/dev/gt-drafts/{tx8}/verdict")
def put_dev_gt_draft_verdict(tx8: str, payload: GtDraftVerdictPayload) -> dict:
    if not _GT_DRAFT_TX8_RE.match(tx8):
        raise HTTPException(status_code=400, detail="Invalid draft id.")
    drafts_dir = get_data_dir() / "gt_drafts"
    if not (drafts_dir / f"{tx8}.rows.json").is_file():
        raise HTTPException(status_code=404, detail="GT draft not found.")
    document = payload.model_dump()
    document["tx8"] = tx8
    document["savedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (drafts_dir / f"{tx8}.verdict.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return {"verdict": document}


@app.get("/api/transcriptions/{transaction_id}")
def get_transcription(transaction_id: str) -> dict:
    _validate_transaction_id(transaction_id)
    data = load_response(transaction_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    timestamps = get_transaction_timestamps(transaction_id)
    if timestamps is not None:
        data["transcribedAt"] = timestamps["transcribedAt"]
        data["audioFirstSeenAt"] = timestamps["audioFirstSeenAt"]
    return data


@app.get("/api/transcriptions/{transaction_id}/alternatives")
def get_transcription_alternatives(transaction_id: str) -> list[dict]:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    audio_sha256 = get_transaction_audio_sha256(transaction_id)
    if audio_sha256 is None:
        return []
    return list_transactions_by_hash(audio_sha256)


@app.get("/api/transcriptions/{transaction_id}/audio")
def get_transcription_audio(transaction_id: str):
    _validate_transaction_id(transaction_id)
    audio_path = load_audio_path(transaction_id)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return FileResponse(audio_path, media_type="audio/wav", filename="audio.wav")


class MemoPayload(BaseModel):
    memo: str


@app.get("/api/transcriptions/{transaction_id}/memo")
def get_transcription_memo(transaction_id: str) -> dict[str, str]:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    memo = load_memo(transaction_id)
    return {"memo": memo if memo is not None else ""}


@app.put("/api/transcriptions/{transaction_id}/memo")
def put_transcription_memo(transaction_id: str, payload: MemoPayload) -> dict[str, str]:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    save_memo(transaction_id, payload.memo)
    return {"memo": payload.memo}


@app.get("/api/transcriptions/{transaction_id}/corrections")
def get_transcription_corrections(transaction_id: str) -> dict:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    raw = load_corrections(transaction_id)
    if raw is None:
        return {"corrections": None}
    try:
        validated = CorrectionsPayload.model_validate(raw)
    except ValidationError:
        # 壊れた/互換性のないファイルを返すとクライアントが誤動作し、次の保存で
        # 原本ごと上書きされる。退避 (.invalid) でデータを保全しつつ「無し」を返す。
        quarantine_corrections(transaction_id)
        return {"corrections": None}
    return {"corrections": validated.model_dump(by_alias=True)}


@app.put("/api/transcriptions/{transaction_id}/corrections")
def put_transcription_corrections(transaction_id: str, payload: CorrectionsPayload) -> dict:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    document = payload.model_dump(by_alias=True)
    document["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_corrections(transaction_id, document)
    return {"corrections": document}


@app.get("/api/transcriptions/{transaction_id}/review-status")
def get_transcription_review_status(transaction_id: str) -> dict:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    raw = load_review_status(transaction_id)
    if raw is None:
        return {"reviewStatus": None}
    try:
        validated = ReviewStatusPayload.model_validate(raw)
    except ValidationError:
        # 互換性のない/壊れた review_status は「未設定」として扱う (誤動作防止)。
        return {"reviewStatus": None}
    return {"reviewStatus": validated.model_dump(by_alias=True)}


@app.put("/api/transcriptions/{transaction_id}/review-status")
def put_transcription_review_status(transaction_id: str, payload: ReviewStatusPayload) -> dict:
    _validate_transaction_id(transaction_id)
    if not transaction_exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transaction not found.")
    document = payload.model_dump(by_alias=True)
    document["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_review_status(transaction_id, document)
    return {"reviewStatus": document}
