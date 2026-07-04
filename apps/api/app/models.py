from __future__ import annotations

from typing import Any, Literal

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
    # Absolute event duration in seconds (#86). Optional for backward compat with
    # transcriptions stored before this field existed; the recognizer always
    # populates it for freshly-computed results so review playback can seek by
    # absolute time instead of approximating from the next event's start.
    duration_sec: float | None = Field(default=None, alias="durationSec")
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


class CorrectionEvent(BaseModel):
    """One event in the user-corrected timeline (review UI).

    Stores note names + absolute seconds so a saved correction can be promoted
    to ground_truth.json (same time/notes vocabulary) without transformation.
    """
    time_sec: float = Field(alias="timeSec")
    notes: list[str] = Field(min_length=1)
    origin: Literal["recognizer", "edited", "inserted-slot", "inserted-manual"] = "recognizer"
    # この onset に主旋律が含まれない (伴奏のみ)。GT 昇格時に role: accompaniment
    # へ変換され、旋律抽出評価の層別に使う (gt-review 側と同じ意味論, 2026-07-05)
    accompaniment_only: bool = Field(default=False, alias="accompanimentOnly")

    model_config = {"populate_by_name": True}


class CorrectionsPayload(BaseModel):
    version: Literal[1] = 1
    updated_at: str | None = Field(default=None, alias="updatedAt")
    events: list[CorrectionEvent]

    model_config = {"populate_by_name": True}


# Review lifecycle status for a recording (tester collection workflow). Kept as a
# small, explicit state set so testers can submit a recording WITHOUT a full
# manual transcription: "recorded_only" is a valid terminal contribution.
# review_completed is the only state that promote_corrections_to_ground_truth.py
# treats as ready for GT-candidate promotion.
ReviewStatusValue = Literal[
    "recorded_only",
    "review_started",
    "review_completed",
    "rerecord_needed",
    "unusable",
    "uncertain",
]


class ReviewStatusPayload(BaseModel):
    version: Literal[1] = 1
    status: ReviewStatusValue
    note: str | None = None
    reviewer: str | None = None
    updated_at: str | None = Field(default=None, alias="updatedAt")

    model_config = {"populate_by_name": True}


class TuningMismatch(BaseModel):
    """Advisory: the recording's spectral peaks fit the selected tuning poorly
    (e.g. a D major recording transcribed with a C major tuning)."""
    selected_coverage: float = Field(alias="selectedCoverage")
    outside_pitch_classes: list[str] = Field(alias="outsidePitchClasses")
    suggested_tuning_id: str | None = Field(default=None, alias="suggestedTuningId")
    suggested_tuning_name: str | None = Field(default=None, alias="suggestedTuningName")
    suggested_coverage: float | None = Field(default=None, alias="suggestedCoverage")

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
    tuning_mismatch: TuningMismatch | None = Field(default=None, alias="tuningMismatch")
    warnings: list[str] = []
    debug: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}
