from __future__ import annotations

import re
from math import pow

from fastapi import HTTPException

from .models import InstrumentTuning, TuningNote


NOTE_TO_MIDI = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

NOTE_NAME_PATTERN = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")


def parse_note_name(note_name: str) -> tuple[str, int]:
    cleaned = note_name.strip()
    match = NOTE_NAME_PATTERN.match(cleaned)
    if not match:
        raise HTTPException(status_code=400, detail=f"Invalid note name: {note_name}")

    pitch = f"{match.group(1).upper()}{match.group(2)}"
    octave = int(match.group(3))
    return pitch, octave


def note_name_to_frequency(note_name: str) -> float:
    pitch, octave = parse_note_name(note_name)
    midi = NOTE_TO_MIDI[pitch] + (octave + 1) * 12
    return 440.0 * pow(2.0, (midi - 69) / 12.0)


def build_tuning(tuning_id: str, name: str, note_names: list[str]) -> InstrumentTuning:
    notes = []
    for index, note_name in enumerate(note_names):
        pitch, octave = parse_note_name(note_name)
        canonical_name = f"{pitch}{octave}"
        notes.append(TuningNote(key=index + 1, noteName=canonical_name, frequency=note_name_to_frequency(canonical_name)))
    return InstrumentTuning(id=tuning_id, name=name, keyCount=len(notes), notes=notes)


DEFAULT_TUNINGS = [
    build_tuning(
        "kalimba-17-c",
        "17 Key C Major",
        ["D6", "B5", "G5", "E5", "C5", "A4", "F4", "D4", "C4", "E4", "G4", "B4", "D5", "F5", "A5", "C6", "E6"],
    ),
    build_tuning(
        "kalimba-17-g",
        "17 Key G Major",
        ["A6", "F#6", "D6", "B5", "G5", "E5", "C5", "A4", "G4", "B4", "D5", "F#5", "A5", "C6", "E6", "G6", "B6"],
    ),
    build_tuning(
        "kalimba-21-c",
        "21 Key C Major",
        [
            "B6",
            "G6",
            "E6",
            "C6",
            "A5",
            "F5",
            "D5",
            "B4",
            "G4",
            "E4",
            "C4",
            "D4",
            "F4",
            "A4",
            "C5",
            "E5",
            "G5",
            "B5",
            "D6",
            "F6",
            "A6"
        ],
    ),
]


def get_default_tunings() -> list[InstrumentTuning]:
    return DEFAULT_TUNINGS


def build_custom_tuning(name: str, note_names: list[object]) -> InstrumentTuning:
    if not isinstance(name, str):
        raise HTTPException(status_code=400, detail="Tuning name must be a string.")

    clean_notes: list[str] = []
    for note in note_names:
        if not isinstance(note, str):
            raise HTTPException(status_code=400, detail="Each tuning noteName must be a non-empty string.")
        cleaned = note.strip()
        if not cleaned:
            raise HTTPException(status_code=400, detail="Each tuning noteName must be a non-empty string.")
        clean_notes.append(cleaned)

    if not clean_notes:
        raise HTTPException(status_code=400, detail="Tuning must contain at least one valid note.")
    return build_tuning("custom", name or "Custom Tuning", clean_notes)
