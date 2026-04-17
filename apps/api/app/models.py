from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TuningNotePartial(BaseModel):
    """A single partial (overtone) of a kalimba tine.

    Beam vibration produces non-integer partials alongside integer harmonics.
    Each partial is defined by its frequency ratio to the fundamental and a
    scoring weight.  When partials are specified on a TuningNote, they replace
    the default integer harmonic comb in scoring and harmonic suppression.
    """
    ratio: float  # frequency ratio to fundamental (e.g. 1.0, 1.5, 2.0)
    weight: float  # scoring weight (1.0 = fundamental, lower for overtones)


class TuningNote(BaseModel):
    key: int
    note_name: str = Field(alias="noteName")
    frequency: float
    layer: int = Field(default=0, alias="layer")
    partials: list[TuningNotePartial] | None = Field(default=None)

    model_config = {"populate_by_name": True}


class InstrumentTuning(BaseModel):
    id: str
    name: str
    key_count: int = Field(alias="keyCount")
    notes: list[TuningNote]
    tonic: str | None = None

    model_config = {"populate_by_name": True}


class ScoreNote(BaseModel):
    key: int
    pitch_class: str = Field(alias="pitchClass")
    octave: int
    label_doremi: str = Field(alias="labelDoReMi")
    label_number: str = Field(alias="labelNumber")
    frequency: float

    model_config = {"populate_by_name": True}


class AlternateGrouping(BaseModel):
    combines_with: list[str] | None = Field(default=None, alias="combinesWith")
    combined_notes: list[ScoreNote] | None = Field(default=None, alias="combinedNotes")
    split_into: list[list[ScoreNote]] | None = Field(default=None, alias="splitInto")
    alternate_note: ScoreNote | None = Field(default=None, alias="alternateNote")
    reason: str
    confidence: float

    model_config = {"populate_by_name": True}


class ScoreEvent(BaseModel):
    id: str
    start_beat: float = Field(alias="startBeat")
    duration_beat: float = Field(alias="durationBeat")
    start_time_sec: float = Field(alias="startTimeSec")
    notes: list[ScoreNote]
    is_gliss_like: bool = Field(alias="isGlissLike")
    gesture: str = "ambiguous"
    alternate_groupings: list[AlternateGrouping] | None = Field(
        default=None, alias="alternateGroupings",
    )

    model_config = {"populate_by_name": True}


class CandidateSlot(BaseModel):
    """A segment dropped by the recognizer, preserved as a low-confidence candidate
    slot for UI presentation (#178 Phase 2). Represents "there might be an event here".
    """
    start_time: float = Field(alias="startTime")
    end_time: float = Field(alias="endTime")
    primary_note: ScoreNote = Field(alias="primaryNote")
    candidates: list[ScoreNote]
    drop_reason: str = Field(alias="dropReason")
    confidence: float

    model_config = {"populate_by_name": True}


class NotationViews(BaseModel):
    western: list[str]
    numbered: list[str]
    vertical_doremi: list[list[str]] = Field(alias="verticalDoReMi")

    model_config = {"populate_by_name": True}



class TranscriptionResult(BaseModel):
    transaction_id: str | None = Field(default=None, alias="transactionId")
    instrument_tuning: InstrumentTuning = Field(alias="instrumentTuning")
    tempo: float
    events: list[ScoreEvent]
    candidate_slots: list[CandidateSlot] = Field(default_factory=list, alias="candidateSlots")
    notation_views: NotationViews = Field(alias="notationViews")
    warnings: list[str] = []
    debug: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}
