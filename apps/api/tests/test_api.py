import json
from pathlib import Path
from io import BytesIO

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.tunings import get_default_tunings
from app.transcription import NoteCandidate, RawEvent, merge_short_chord_clusters, normalize_repeated_triad_patterns, segment_peaks, suppress_leading_gliss_subset_transients, suppress_resonant_carryover, suppress_short_residual_tails, suppress_subset_decay_events

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
    candidates, debug, _ = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    assert candidates
    assert candidates[0].note_name == "D5"
    assert debug is not None

def test_segment_peaks_detects_d5_and_a5_chord() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_chord((587.3295, 880.0))
    candidates, debug, _ = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    note_names = [candidate.note_name for candidate in candidates]
    assert "D5" in note_names
    assert "A5" in note_names
    assert debug is not None
    assert debug["residualCandidates"]

def test_segment_peaks_allows_true_octave_dyad() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_chord((587.3295, 1174.6591))
    candidates, debug, _ = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    note_names = [candidate.note_name for candidate in candidates]
    assert "D5" in note_names
    assert "D6" in note_names
    assert debug is not None
    assert any(
        item["noteName"] in {"D5", "D6"} and (item.get("accepted") or item.get("octaveDyadAllowed"))
        for item in debug["secondaryDecisionTrail"]
    )

def test_segment_peaks_keeps_mono_d4_monophonic() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_note(293.665)
    candidates, debug, _ = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    assert [candidate.note_name for candidate in candidates] == ["D4"]
    assert debug is not None
    assert any(
        item["noteName"] == "D5" and not item.get("accepted")
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

def test_suppress_resonant_carryover_keeps_true_short_octave_dyad() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[d4], is_gliss_like=False),
        RawEvent(start_time=0.6, end_time=0.88, notes=[d4, d5], is_gliss_like=False),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["D4"],
        ["D4", "D5"],
    ]

def test_merge_short_chord_clusters_merges_singleton_and_dyad_into_triad() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.14, notes=[c4], is_gliss_like=True, primary_note_name="C4", primary_score=80.0),
        RawEvent(start_time=0.14, end_time=0.9, notes=[e4, g4], is_gliss_like=False, primary_note_name="G4", primary_score=500.0),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C4", "E4", "G4"]]


def test_merge_short_chord_clusters_merges_subset_into_following_triad() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.8, notes=[d4, f4], is_gliss_like=False, primary_note_name="D4", primary_score=1000.0),
        RawEvent(start_time=0.8, end_time=1.1, notes=[d4, f4, a4], is_gliss_like=True, primary_note_name="A4", primary_score=1200.0),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["D4", "F4", "A4"]]


def test_normalize_repeated_triad_patterns_expands_dominant_subsets() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.7, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=1000.0),
        RawEvent(start_time=0.8, end_time=1.4, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=900.0),
        RawEvent(start_time=1.6, end_time=2.0, notes=[d4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=500.0),
        RawEvent(start_time=2.2, end_time=2.9, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="A4", primary_score=950.0),
    ]

    normalized = normalize_repeated_triad_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
    ]


def test_normalize_repeated_triad_patterns_rewrites_isolated_outlier_triad() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=900.0),
        RawEvent(start_time=0.8, end_time=1.4, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=500.0),
        RawEvent(start_time=1.6, end_time=2.2, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="B4", primary_score=920.0),
        RawEvent(start_time=2.4, end_time=3.0, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="G4", primary_score=940.0),
        RawEvent(start_time=3.2, end_time=3.8, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=910.0),
    ]

    normalized = normalize_repeated_triad_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4"],
        ["E4", "G4", "B4"],
        ["E4", "G4", "B4"],
        ["E4", "G4", "B4"],
        ["E4", "G4", "B4"],
    ]


def test_suppress_leading_gliss_subset_transients_drops_short_prefix() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.09, notes=[c4], is_gliss_like=True, primary_note_name="C4", primary_score=80.0),
        RawEvent(start_time=0.09, end_time=0.5, notes=[c4, e4, g4], is_gliss_like=True, primary_note_name="G4", primary_score=500.0),
    ]

    cleaned = suppress_leading_gliss_subset_transients(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [["C4", "E4", "G4"]]

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

def test_transcription_regression_for_manual_triple_glissando() -> None:
    fixture = Path(__file__).parent / "fixtures" / "manual-captures" / "kalimba-17-c-triple-glissando-ascending-01"
    request_payload = json.loads((fixture / "request.json").read_text(encoding="utf-8"))
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", (fixture / "audio.wav").read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert [sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]] == [
        ["C4", "E4", "G4"],
        ["A4", "D4", "F4"],
        ["B4", "E4", "G4"],
        ["A4", "C5", "F4"],
        ["B4", "D5", "G4"],
        ["A4", "C5", "E5"],
        ["B4", "D5", "F5"],
        ["C5", "E5", "G5"],
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

