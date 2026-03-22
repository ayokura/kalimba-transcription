from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TuningNote(BaseModel):
    key: int
    note_name: str = Field(alias="noteName")
    frequency: float

    model_config = {"populate_by_name": True}


class InstrumentTuning(BaseModel):
    id: str
    name: str
    key_count: int = Field(alias="keyCount")
    notes: list[TuningNote]

    model_config = {"populate_by_name": True}


class ScoreNote(BaseModel):
    key: int
    pitch_class: str = Field(alias="pitchClass")
    octave: int
    label_doremi: str = Field(alias="labelDoReMi")
    label_number: str = Field(alias="labelNumber")
    frequency: float

    model_config = {"populate_by_name": True}


class ScoreEvent(BaseModel):
    id: str
    start_beat: float = Field(alias="startBeat")
    duration_beat: float = Field(alias="durationBeat")
    notes: list[ScoreNote]
    is_gliss_like: bool = Field(alias="isGlissLike")
    gesture: str = "ambiguous"

    model_config = {"populate_by_name": True}


class NotationViews(BaseModel):
    western: list[str]
    numbered: list[str]
    vertical_doremi: list[list[str]] = Field(alias="verticalDoReMi")

    model_config = {"populate_by_name": True}



class TranscriptionResult(BaseModel):
    instrument_tuning: InstrumentTuning = Field(alias="instrumentTuning")
    tempo: float
    events: list[ScoreEvent]
    notation_views: NotationViews = Field(alias="notationViews")
    warnings: list[str] = []
    debug: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}
