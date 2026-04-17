from __future__ import annotations

import re
from math import pow
from typing import Sequence

from fastapi import HTTPException

from .models import InstrumentTuning, TuningNote, TuningNotePartial


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


# Default kalimba partials: integer harmonics + beam vibration partial at 1.5×.
# Beam vibration in rectangular tines produces non-integer partials alongside
# integer ones.  Measured across 3 instruments (17-C, 17-G-low, 34L-C):
# - 1.5× partial is present in most tines (dominant in pickup recordings,
#   substantial in mic recordings)
# - Integer 2×, 3×, 4× are also present
# - The relative strength of 1.5× vs 2.0× varies per tine and recording method
# Weight 0.35 for beam partial is conservative — between P2 (0.55) and P3 (0.3).
KALIMBA_DEFAULT_PARTIALS = [
    TuningNotePartial(ratio=1.0, weight=1.0),
    TuningNotePartial(ratio=1.5, weight=0.35),
    TuningNotePartial(ratio=2.0, weight=0.55),
    TuningNotePartial(ratio=3.0, weight=0.3),
    TuningNotePartial(ratio=4.0, weight=0.15),
]


def build_tuning(
    tuning_id: str,
    name: str,
    note_names: list[str],
    *,
    default_partials: list[TuningNotePartial] | None = None,
    partial_overrides: dict[str, list[TuningNotePartial]] | None = None,
) -> InstrumentTuning:
    notes = []
    for index, note_name in enumerate(note_names):
        pitch, octave = parse_note_name(note_name)
        canonical_name = f"{pitch}{octave}"
        partials = (partial_overrides or {}).get(canonical_name, default_partials)
        notes.append(TuningNote(
            key=index + 1,
            noteName=canonical_name,
            frequency=note_name_to_frequency(canonical_name),
            partials=partials,
        ))
    return InstrumentTuning(id=tuning_id, name=name, keyCount=len(notes), notes=notes)


# Per-tine partial overrides for 17-C, measured from single-note repeat
# fixtures and c4-to-e6 separated notes.
# Only notes with confirmed non-standard partials are overridden; all others
# fall back to KALIMBA_DEFAULT_PARTIALS.
_P = TuningNotePartial
KALIMBA_17C_PARTIAL_OVERRIDES: dict[str, list[TuningNotePartial]] = {
    # C4: P2≈2.0 (1.993), P3=2.908×, P4=3.672× (c4-repeat-01, high confidence)
    "C4": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.35),
        _P(ratio=2.0, weight=0.55),
        _P(ratio=2.908, weight=0.3),    # measured P3 (not 3.0)
        _P(ratio=3.672, weight=0.15),   # measured P4 (not 4.0)
    ],
    # D5: P2=1.508× dominant (d5-repeat-01, high confidence)
    "D5": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.55),     # dominant P2 (swapped with 2.0×)
        _P(ratio=2.0, weight=0.35),
        _P(ratio=3.0, weight=0.3),
        _P(ratio=4.0, weight=0.15),
    ],
    # F5: P2=1.506× dominant (c4-to-e6 separated, confirmed)
    "F5": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.55),     # dominant P2 (swapped with 2.0×)
        _P(ratio=2.0, weight=0.35),
        _P(ratio=3.0, weight=0.3),
        _P(ratio=4.0, weight=0.15),
    ],
    # C5: P2=2.247× (c4-to-e6 separated, non-standard)
    "C5": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.35),
        _P(ratio=2.0, weight=0.3),      # still present but not dominant
        _P(ratio=2.247, weight=0.55),   # measured dominant P2
        _P(ratio=3.0, weight=0.3),
        _P(ratio=4.0, weight=0.15),
    ],
}

# G-low (magnetic pickup): beam partial 1.5× is dominant P2 for most tines.
# Measured from BWV147 solo segments — all tines except G4 show P2≈1.5×.
# Pickup captures tine vibration directly, bypassing the resonance box filter
# that emphasizes integer harmonics in mic recordings.
KALIMBA_GLOW_DEFAULT_PARTIALS = [
    _P(ratio=1.0, weight=1.0),
    _P(ratio=1.5, weight=0.55),     # dominant P2 for pickup recordings
    _P(ratio=2.0, weight=0.35),
    _P(ratio=3.0, weight=0.3),
    _P(ratio=4.0, weight=0.15),
]

KALIMBA_GLOW_PARTIAL_OVERRIDES: dict[str, list[TuningNotePartial]] = {
    # G3: P2=2.020× — G tines show 2.0× dominant (same as G4)
    "G3": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.35),
        _P(ratio=2.0, weight=0.55),     # dominant (G-tine pattern)
        _P(ratio=3.0, weight=0.3),
        _P(ratio=4.0, weight=0.15),
    ],
    # G4: P2=2.019× — same G-tine pattern
    "G4": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.35),
        _P(ratio=2.0, weight=0.55),     # dominant (G-tine pattern)
        _P(ratio=3.0, weight=0.3),
        _P(ratio=4.0, weight=0.15),
    ],
    # B3: P2=2.391× — non-standard, related to #166 (fR=0.599)
    "B3": [
        _P(ratio=1.0, weight=1.0),
        _P(ratio=1.5, weight=0.35),
        _P(ratio=2.0, weight=0.3),      # present but not dominant
        _P(ratio=2.391, weight=0.55),   # measured dominant P2 (BWV147 solo)
        _P(ratio=3.0, weight=0.3),
        _P(ratio=4.0, weight=0.15),
    ],
}

# 17-key kalimba uses a V-shape layout: the tonic sits at center (index 8),
# and notes alternate outward — right side ascending: 3rd, 5th, 7th, 2nd(+oct),
# 4th(+oct), 6th(+oct), 1st(+2oct), 3rd(+2oct); left side mirrors the pattern
# with 2nd, 4th, 6th, 1st(+oct), 3rd(+oct), 5th(+oct), 7th(+oct), 2nd(+2oct).
# The helper below generates this layout for any major-scale tonic.
MAJOR_SCALE_SEMITONES = (0, 2, 4, 5, 7, 9, 11)


def _midi_to_note_name(midi: int, use_flats: bool = False) -> str:
    names_sharp = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    names_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    octave = (midi // 12) - 1
    pitch = (names_flat if use_flats else names_sharp)[midi % 12]
    return f"{pitch}{octave}"


def major_17_key_layout(tonic_note: str, *, use_flats: bool = False) -> list[str]:
    pitch, octave = parse_note_name(tonic_note)
    base_midi = NOTE_TO_MIDI[pitch] + (octave + 1) * 12
    right_offsets = [4, 7, 11, 14, 17, 21, 24, 28]
    left_offsets = [2, 5, 9, 12, 16, 19, 23, 26]
    left_notes = [_midi_to_note_name(base_midi + o, use_flats) for o in left_offsets]
    right_notes = [_midi_to_note_name(base_midi + o, use_flats) for o in right_offsets]
    return list(reversed(left_notes)) + [f"{pitch}{octave}"] + right_notes


DEFAULT_TUNINGS = [
    build_tuning(
        "kalimba-17-c",
        "17 Key C Major",
        ["D6", "B5", "G5", "E5", "C5", "A4", "F4", "D4", "C4", "E4", "G4", "B4", "D5", "F5", "A5", "C6", "E6"],
        default_partials=KALIMBA_DEFAULT_PARTIALS,
        partial_overrides=KALIMBA_17C_PARTIAL_OVERRIDES,
    ),
    build_tuning(
        "kalimba-17-d",
        "17 Key D Major (ニ長調)",
        major_17_key_layout("D4"),
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-e",
        "17 Key E Major (ホ長調)",
        major_17_key_layout("E4"),
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-f",
        "17 Key F Major (ヘ長調)",
        major_17_key_layout("F4", use_flats=True),
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-g",
        "17 Key G Major",
        ["A6", "F#6", "D6", "B5", "G5", "E5", "C5", "A4", "G4", "B4", "D5", "F#5", "A5", "C6", "E6", "G6", "B6"],
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-a",
        "17 Key A Major (イ長調)",
        major_17_key_layout("A4"),
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-b",
        "17 Key B Major (ロ長調)",
        major_17_key_layout("B4"),
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-bb",
        "17 Key Bb Major (変ロ長調)",
        major_17_key_layout("Bb4", use_flats=True),
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
    build_tuning(
        "kalimba-17-g-low",
        "17 Key G Major (Low Octave)",
        ["A5", "F#5", "D5", "B4", "G4", "E4", "C4", "A3", "G3", "B3", "D4", "F#4", "A4", "C5", "E5", "G5", "B5"],
        default_partials=KALIMBA_GLOW_DEFAULT_PARTIALS,
        partial_overrides=KALIMBA_GLOW_PARTIAL_OVERRIDES,
    ),
    build_tuning(
        "kalimba-34l-c",
        "34 Key Lingting C Major",
        [
            # Lower layer (17-key C layout)
            "D6", "B5", "G5", "E5", "C5", "A4", "F4", "D4",
            "C4",
            "E4", "G4", "B4", "D5", "F5", "A5", "C6", "E6",
            # Upper layer (each +1 semitone)
            "D#6", "C6", "G#5", "F5", "C#5", "A#4", "F#4", "D#4",
            "C#4",
            "F4", "G#4", "C5", "D#5", "F#5", "A#5", "C#6", "F6",
        ],
        default_partials=KALIMBA_DEFAULT_PARTIALS,
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
        default_partials=KALIMBA_DEFAULT_PARTIALS,
    ),
]


def get_default_tunings() -> list[InstrumentTuning]:
    return DEFAULT_TUNINGS


def build_custom_tuning(name: str, note_names: Sequence[object]) -> InstrumentTuning:
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
