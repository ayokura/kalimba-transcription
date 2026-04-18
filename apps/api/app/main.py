from __future__ import annotations

import json
import os
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .fingerprints import git_head_sha, kalimba_dsp_fingerprint, recognizer_fingerprint
from .models import InstrumentTuning, TranscriptionResult
from .storage import (
    compute_audio_sha256,
    find_transaction_by_hash_and_tuning,
    generate_transaction_id,
    get_transaction_audio_sha256,
    get_transaction_timestamps,
    list_recent_transactions,
    list_transactions_by_hash,
    load_audio_path,
    load_memo,
    load_response,
    save_memo,
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
    file: UploadFile = File(...),
    tuning: str = Form(...),
    debug: bool = Form(True),
    disabledRepeatedPatternPasses: str | None = Form(None),
    midPerformanceStart: bool = Form(False),
    midPerformanceEnd: bool = Form(False),
    force: bool = Form(False),
    dryRun: bool = Form(False),
) -> TranscriptionResult:
    audio_bytes = await file.read()
    await file.seek(0)

    parsed_tuning = parse_tuning_json(tuning)
    audio_sha256 = compute_audio_sha256(audio_bytes)

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
    response_dict = result.model_dump(by_alias=True)
    debug_dict = response_dict.get("debug")

    save_transaction(transaction_id, audio_bytes, request_params, response_dict, debug_dict)

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
