from __future__ import annotations

import json
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .models import InstrumentTuning, TranscriptionResult
from .storage import (
    generate_transaction_id,
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    debug: bool = Form(False),
    disabledRepeatedPatternPasses: str | None = Form(None),
    midPerformanceStart: bool = Form(False),
    midPerformanceEnd: bool = Form(False),
) -> TranscriptionResult:
    audio_bytes = await file.read()
    await file.seek(0)

    parsed_tuning = parse_tuning_json(tuning)
    disabled_passes = parse_disabled_repeated_pattern_passes(disabledRepeatedPatternPasses)
    result = await transcribe_audio(
        file,
        parsed_tuning,
        debug=debug,
        disabled_repeated_pattern_passes=disabled_passes,
        mid_performance_start=midPerformanceStart,
        mid_performance_end=midPerformanceEnd,
    )

    transaction_id = generate_transaction_id()
    result.transaction_id = transaction_id

    request_params = {
        "tuning": json.loads(tuning),
        "debug": debug,
        "disabledRepeatedPatternPasses": disabledRepeatedPatternPasses,
        "midPerformanceStart": midPerformanceStart,
        "midPerformanceEnd": midPerformanceEnd,
    }
    response_dict = result.model_dump(by_alias=True)
    debug_dict = response_dict.get("debug") if debug else None

    save_transaction(transaction_id, audio_bytes, request_params, response_dict, debug_dict)

    return result


@app.get("/api/transcriptions/{transaction_id}")
def get_transcription(transaction_id: str) -> dict:
    _validate_transaction_id(transaction_id)
    data = load_response(transaction_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return data


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
