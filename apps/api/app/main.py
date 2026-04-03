from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .models import InstrumentTuning, TranscriptionResult
from .transcription import parse_tuning_json, transcribe_audio
from .transcription.patterns import REPEATED_PATTERN_PASS_IDS
from .tunings import get_default_tunings


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
    parsed_tuning = parse_tuning_json(tuning)
    disabled_passes = parse_disabled_repeated_pattern_passes(disabledRepeatedPatternPasses)
    return await transcribe_audio(
        file,
        parsed_tuning,
        debug=debug,
        disabled_repeated_pattern_passes=disabled_passes,
        mid_performance_start=midPerformanceStart,
        mid_performance_end=midPerformanceEnd,
    )
