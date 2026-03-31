from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from ..models import InstrumentTuning
from ..tunings import build_custom_tuning, parse_note_name
from .models import NoteCandidate


def parse_tuning_json(tuning_json: str) -> InstrumentTuning:
    try:
        payload: Any = json.loads(tuning_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid tuning JSON.") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Tuning JSON must be an object.")

    notes = payload.get("notes", [])
    if not isinstance(notes, list) or not notes:
        raise HTTPException(status_code=400, detail="Tuning must contain at least one note.")

    note_names: list[Any] = []
    for note in notes:
        if not isinstance(note, dict):
            raise HTTPException(status_code=400, detail="Each tuning note must be an object.")
        if "noteName" not in note:
            raise HTTPException(status_code=400, detail="Each tuning note must include noteName.")
        note_names.append(note["noteName"])

    name = payload.get("name", "Custom Tuning")
    if not isinstance(name, str):
        raise HTTPException(status_code=400, detail="Tuning name must be a string.")

    return build_custom_tuning(name, note_names)


async def read_audio(upload: UploadFile) -> tuple[np.ndarray, int]:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty.")

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Send WAV audio from the web client.") from exc

    if audio.ndim > 1:
        audio = audio[:, 0]

    if float(np.max(np.abs(audio))) < 1e-4:
        raise HTTPException(status_code=422, detail="Audio appears to be silent.")

    return audio.astype(np.float32), sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    centered = audio - np.mean(audio)
    peak = np.max(np.abs(centered))
    if peak < 1e-6:
        return centered
    return centered / peak


def cents_distance(freq_a: float, freq_b: float) -> float:
    return abs(1200.0 * math.log2(freq_a / freq_b))


def snap_frequency_to_tuning(freq: float, tuning: InstrumentTuning) -> NoteCandidate | None:
    best_note = None
    best_distance = float("inf")

    for note in tuning.notes:
        distance = cents_distance(freq, note.frequency)
        if distance < best_distance:
            best_note = note
            best_distance = distance

    if best_note is None or best_distance > 80:
        return None

    pitch_class, octave = parse_note_name(best_note.note_name)
    return NoteCandidate(
        key=best_note.key,
        note_name=best_note.note_name,
        frequency=best_note.frequency,
        pitch_class=pitch_class,
        octave=octave,
    )
