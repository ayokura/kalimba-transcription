from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .models import InstrumentTuning, TranscriptionResult
from .transcription import parse_disabled_repeated_pattern_passes, parse_tuning_json, transcribe_audio
from .tunings import get_default_tunings


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
