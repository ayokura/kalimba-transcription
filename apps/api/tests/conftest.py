from copy import deepcopy
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
MANUAL_CAPTURE_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "manual-captures"


@lru_cache(maxsize=32)
def _load_manual_capture_fixture_inputs(fixture_name: str) -> tuple[dict, Path]:
    fixture_dir = MANUAL_CAPTURE_FIXTURE_ROOT / fixture_name
    request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    return request_payload, fixture_dir / "audio.wav"


@lru_cache(maxsize=32)
def _transcribe_manual_capture_fixture(
    fixture_name: str,
    debug: bool,
    disabled_repeated_pattern_passes_json: str | None,
) -> dict:
    request_payload, audio_path = _load_manual_capture_fixture_inputs(fixture_name)
    audio_bytes = audio_path.read_bytes()
    data = {
        "tuning": json.dumps(request_payload["tuning"]),
        "debug": "true" if debug else "false",
    }
    if disabled_repeated_pattern_passes_json is not None:
        data["disabledRepeatedPatternPasses"] = disabled_repeated_pattern_passes_json

    response = client.post(
        "/api/transcriptions",
        data=data,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200, (
        f"Unexpected status code {response.status_code} for fixture '{fixture_name}' "
        f"(debug={debug}, disabled_repeated_pattern_passes_json={disabled_repeated_pattern_passes_json}). "
        f"Response body: {response.text}"
    )
    return response.json()


def transcribe_manual_capture_fixture(
    fixture_name: str,
    *,
    debug: bool = True,
    disabled_repeated_pattern_passes: tuple[str, ...] | None = None,
) -> dict:
    disabled_passes_json = None
    if disabled_repeated_pattern_passes is not None:
        disabled_passes_json = json.dumps(list(disabled_repeated_pattern_passes))
    return deepcopy(_transcribe_manual_capture_fixture(fixture_name, debug, disabled_passes_json))


def manual_capture_slow(test_func):
    return pytest.mark.slow(pytest.mark.manual_capture(test_func))


def synthesize_note(frequency: float, sample_rate: int = 44100, duration: float = 0.45, harmonics: tuple[float, ...] = (0.7, 0.45, 0.25)) -> np.ndarray:
    times = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    envelope = np.exp(-4.5 * times)
    signal = np.sin(2 * np.pi * frequency * times)
    for index, weight in enumerate(harmonics, start=2):
        signal += weight * np.sin(2 * np.pi * frequency * index * times)
    return (signal * envelope).astype(np.float32)

def synthesize_chord(frequencies: tuple[float, ...], sample_rate: int = 44100, duration: float = 0.5) -> np.ndarray:
    chord = np.zeros(int(sample_rate * duration), dtype=np.float32)
    for frequency in frequencies:
        chord += synthesize_note(frequency, sample_rate=sample_rate, duration=duration, harmonics=(0.8, 0.5, 0.3))
    peak = np.max(np.abs(chord))
    return chord if peak < 1e-6 else (chord / peak).astype(np.float32)

def synthesize_repeated_note(frequency: float, repeats: int = 5, sample_rate: int = 44100) -> np.ndarray:
    note = synthesize_note(frequency, sample_rate=sample_rate)
    silence = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for _ in range(repeats):
        chunks.extend([note, silence])
    return np.concatenate(chunks)

def synthesize_repeated_chord(frequencies: tuple[float, ...], repeats: int = 4, sample_rate: int = 44100) -> np.ndarray:
    chord = synthesize_chord(frequencies, sample_rate=sample_rate, duration=0.42)
    silence = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for _ in range(repeats):
        chunks.extend([chord, silence])
    return np.concatenate(chunks)

def synthesize_note_tail(
    frequency: float,
    *,
    sample_rate: int = 44100,
    duration: float = 0.36,
    offset: float = 0.24,
) -> np.ndarray:
    full = synthesize_note(frequency, sample_rate=sample_rate, duration=duration + offset)
    start = int(sample_rate * offset)
    end = start + int(sample_rate * duration)
    return full[start:end]

def wav_bytes(audio: np.ndarray, sample_rate: int = 44100) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()
