from __future__ import annotations

from ..models import NotationViews, ScoreEvent
from .constants import PITCH_CLASS_TO_DOREMI, PITCH_CLASS_TO_NUMBER
from .models import NoteCandidate


def quantize_beat(value: float, step: float = 0.25) -> float:
    return round(value / step) * step


def format_doremi(candidate: NoteCandidate) -> str:
    base = PITCH_CLASS_TO_DOREMI[candidate.pitch_class]
    if candidate.octave >= 6:
        return f"{base}.."
    if candidate.octave == 5:
        return f"{base}."
    if candidate.octave == 3:
        return f"_{base}"
    if candidate.octave <= 2:
        return f"__{base}"
    return base


def format_number(candidate: NoteCandidate) -> str:
    base = PITCH_CLASS_TO_NUMBER[candidate.pitch_class]
    if candidate.octave >= 6:
        return f"{base}''"
    if candidate.octave == 5:
        return f"{base}'"
    if candidate.octave == 3:
        return f".{base}"
    if candidate.octave <= 2:
        return f"..{base}"
    return base


def build_notation_views(events: list[ScoreEvent]) -> NotationViews:
    western = [" | ".join(note.pitch_class + str(note.octave) for note in event.notes) for event in events]
    numbered = [" ".join(note.label_number for note in event.notes) for event in events]
    vertical = [[note.label_doremi for note in event.notes] for event in events]
    return NotationViews(western=western, numbered=numbered, verticalDoReMi=vertical)
