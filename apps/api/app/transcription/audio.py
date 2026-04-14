from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from ..models import InstrumentTuning
from ..tunings import build_custom_tuning, get_default_tunings
from .models import Note, NoteCandidate


@lru_cache(maxsize=64)
def cached_hanning(length: int) -> np.ndarray:
    window = np.hanning(length)
    window.setflags(write=False)
    return window


@lru_cache(maxsize=64)
def cached_rfftfreq(n_fft: int, sample_rate: int) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    freqs.setflags(write=False)
    return freqs


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

    # If the request's tuning id matches a known default tuning and the note
    # names also match, return the server-side default tuning directly so that
    # per-tine partial configurations (see `apps/api/app/tunings.py`) are
    # applied.  Requests built from the web client typically send standard
    # note sets with the matching id, so this path is the common case.
    tuning_id = payload.get("id")
    if (
        isinstance(tuning_id, str)
        and tuning_id
        and all(isinstance(n, str) for n in note_names)
    ):
        for default in get_default_tunings():
            if default.id != tuning_id:
                continue
            default_names = [n.note_name for n in default.notes]
            if default_names == note_names:
                return default
            # id matched but notes diverged — treat as a custom variant
            break

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

    return NoteCandidate(
        key=best_note.key,
        note=Note.from_name(best_note.note_name),
    )
