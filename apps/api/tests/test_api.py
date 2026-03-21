import json
from pathlib import Path
from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.tunings import get_default_tunings
from app.transcription import NoteCandidate, RawEvent, segment_peaks, suppress_resonant_carryover, suppress_short_residual_tails, suppress_subset_decay_events

client = TestClient(app)

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

def wav_bytes(audio: np.ndarray, sample_rate: int = 44100) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()

def test_health_check() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_tunings_endpoint_returns_presets() -> None:
    response = client.get("/api/tunings")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 3
    assert payload[0]["notes"]

def test_segment_peaks_prefers_fundamental_for_strong_harmonics() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_note(587.3295, harmonics=(1.0, 0.9, 0.7))
    candidates, debug = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    assert candidates
    assert candidates[0].note_name == "D5"
    assert debug is not None

def test_segment_peaks_detects_d5_and_a5_chord() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_chord((587.3295, 880.0))
    candidates, debug = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    note_names = [candidate.note_name for candidate in candidates]
    assert "D5" in note_names
    assert "A5" in note_names
    assert debug is not None
    assert debug["residualCandidates"]

def test_segment_peaks_allows_true_octave_dyad() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_chord((587.3295, 1174.6591))
    candidates, debug = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    note_names = [candidate.note_name for candidate in candidates]
    assert "D5" in note_names
    assert "D6" in note_names
    assert debug is not None
    assert any(
        item["noteName"] in {"D5", "D6"} and (item.get("accepted") or item.get("octaveDyadAllowed"))
        for item in debug["secondaryDecisionTrail"]
    )

def test_transcription_regression_for_repeated_octave_dyad() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_repeated_chord((587.3295, 1174.6591))
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning.model_dump(by_alias=True)), "debug": "true"},
        files={"file": ("d5-d6.wav", wav_bytes(audio), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    note_sets = [
        sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"])
        for event in payload["events"]
    ]
    octave_hits = sum(1 for note_set in note_sets if note_set == ["D5", "D6"])
    assert octave_hits >= 3
    assert payload["debug"]["segmentCandidates"]

def test_transcription_regression_for_repeated_d5() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_repeated_note(587.3295)
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning.model_dump(by_alias=True)), "debug": "true"},
        files={"file": ("d5.wav", wav_bytes(audio), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert 3 <= len(payload["events"]) <= 7
    detected_d5 = sum(1 for event in payload["events"] if any(note["pitchClass"] == "D" and note["octave"] == 5 for note in event["notes"]))
    assert detected_d5 >= 4
    assert payload["debug"]["segments"]

def test_suppress_resonant_carryover_prefers_fresh_ascending_note() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    c5 = NoteCandidate(key=5, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    f5 = NoteCandidate(key=14, note_name="F5", frequency=698.4564628660078, pitch_class="F", octave=5)
    g5 = NoteCandidate(key=3, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[c4], is_gliss_like=False),
        RawEvent(start_time=0.4, end_time=0.8, notes=[c4, c5], is_gliss_like=False),
        RawEvent(start_time=0.8, end_time=1.2, notes=[c4, d5], is_gliss_like=False),
        RawEvent(start_time=1.2, end_time=1.6, notes=[c5, e5], is_gliss_like=False),
        RawEvent(start_time=1.6, end_time=2.0, notes=[g5], is_gliss_like=False),
        RawEvent(start_time=2.0, end_time=2.1, notes=[f5, g5], is_gliss_like=True),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C4"],
        ["C5"],
        ["D5"],
        ["C5", "E5"],
        ["G5"],
        ["F5"],
    ]

def test_suppress_short_residual_tails_drops_recent_single_note_tail() -> None:
    c5 = NoteCandidate(key=5, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    g5 = NoteCandidate(key=3, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[d5], is_gliss_like=False),
        RawEvent(start_time=0.4, end_time=0.9, notes=[c5, e5], is_gliss_like=False),
        RawEvent(start_time=0.9, end_time=0.99, notes=[d5], is_gliss_like=True),
        RawEvent(start_time=1.01, end_time=1.4, notes=[g5], is_gliss_like=False),
    ]

    cleaned = suppress_short_residual_tails(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["D5"],
        ["C5", "E5"],
        ["G5"],
    ]

def test_transcription_regression_for_manual_mixed_sequence() -> None:
    fixture = Path(__file__).parent / "fixtures" / "manual-captures" / "kalimba-17-c-mixed-sequence-01"
    request_payload = json.loads((fixture / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", (fixture / "audio.wav").read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert [sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]] == [
        ["C4"],
        ["C5"],
        ["D5"],
        ["C5", "E5"],
        ["G5"],
        ["F5"],
    ]

def test_suppress_subset_decay_events_drops_contiguous_subset_tail() -> None:
    c5 = NoteCandidate(key=5, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    g5 = NoteCandidate(key=3, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.5, notes=[c5, e5], is_gliss_like=False),
        RawEvent(start_time=0.5, end_time=1.0, notes=[e5], is_gliss_like=False),
        RawEvent(start_time=1.05, end_time=1.4, notes=[g5], is_gliss_like=False),
    ]

    cleaned = suppress_subset_decay_events(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C5", "E5"],
        ["G5"],
    ]

