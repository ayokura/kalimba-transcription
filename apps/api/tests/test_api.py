import json
from pathlib import Path
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.tunings import get_default_tunings
from app.transcription import REPEATED_PATTERN_PASS_IDS, NoteCandidate, RawEvent, apply_repeated_pattern_passes, build_recent_ascending_primary_run_ceiling, build_recent_note_names, classify_event_gesture, collapse_same_start_primary_singletons, collect_multi_onset_gap_segments, collect_two_onset_gap_segments, detect_segments, merge_four_note_gliss_clusters, merge_short_chord_clusters, merge_short_gliss_clusters, normalize_repeated_explicit_four_note_patterns, normalize_repeated_four_note_family, normalize_repeated_four_note_gliss_patterns, normalize_repeated_triad_patterns, normalize_strict_four_note_subsets, simplify_short_gliss_prefix_to_contiguous_singleton, suppress_isolated_triad_extensions, suppress_leading_gliss_neighbor_noise, suppress_repeated_triad_blips, segment_peaks, suppress_leading_gliss_subset_transients, suppress_resonant_carryover, suppress_short_residual_tails, suppress_subset_decay_events

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

def test_detect_segments_reports_tempo_debug_metrics() -> None:
    sample_rate = 44100
    gap = np.zeros(int(sample_rate * 0.35), dtype=np.float32)
    audio = np.concatenate([
        synthesize_note(329.6275569128699, sample_rate=sample_rate, duration=0.28),
        gap,
        synthesize_note(391.99543598174927, sample_rate=sample_rate, duration=0.28),
        gap,
        synthesize_note(523.2511306011972, sample_rate=sample_rate, duration=0.28),
    ]).astype(np.float32)

    segments, tempo, debug = detect_segments(audio, sample_rate)

    assert len(segments) >= 3
    assert 30.0 <= tempo <= 300.0
    assert debug["tempoHopLength"] == 1024
    assert debug["tempoAudioDurationSec"] > 0
    assert debug["tempoEstimationMs"] >= 0



def test_collect_multi_onset_gap_segments_requires_long_regular_gap() -> None:
    active_ranges = [(0.0, 0.12), (2.5, 2.68)]
    onset_times = [0.62, 0.94, 1.24, 1.82]

    segments = collect_multi_onset_gap_segments(active_ranges, onset_times)

    assert segments == [(0.62, 0.94), (0.94, 1.24), (1.24, 1.82), (1.82, 2.5)]


def test_collect_two_onset_gap_segments_requires_tight_mute_restrike_shape() -> None:
    active_ranges = [(12.2267, 12.5707), (13.1293, 13.448)]
    onset_times = [12.2267, 12.6667, 12.7520, 13.1227, 13.448]

    segments = collect_two_onset_gap_segments(active_ranges, onset_times)

    assert segments == [(12.6667, 12.752)]


def test_detect_segments_collapses_redundant_same_start_segments() -> None:
    sample_rate = 44100
    gap = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
    audio = np.concatenate([
        synthesize_note(261.6255653005986, sample_rate=sample_rate, duration=0.22),
        gap,
        synthesize_note(293.6647679174076, sample_rate=sample_rate, duration=0.22),
        gap,
        synthesize_note(329.6275569128699, sample_rate=sample_rate, duration=0.22),
    ]).astype(np.float32)

    segments, _, _ = detect_segments(audio, sample_rate)

    starts = [round(start, 4) for start, _ in segments]
    assert starts.count(starts[0]) == 1
    assert len(segments) >= 3
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

def test_segment_peaks_replaces_stale_recent_primary_with_fresh_lower_onset() -> None:
    tuning = get_default_tunings()[0]
    total_duration = 1.2
    audio = np.zeros(int(44100 * total_duration), dtype=np.float32)
    stale_e5 = synthesize_note(659.2551138257398, duration=0.9)
    fresh_c4 = synthesize_note(261.6255653005986, duration=0.36)
    audio[:len(stale_e5)] += stale_e5
    start = int(44100 * 0.5)
    audio[start:start + len(fresh_c4)] += fresh_c4
    peak = np.max(np.abs(audio))
    audio = audio if peak < 1e-6 else (audio / peak).astype(np.float32)

    candidates, debug, primary = segment_peaks(
        audio,
        44100,
        0.5,
        0.86,
        tuning,
        debug=True,
        recent_note_names={"E5"},
    )

    assert primary is not None
    assert primary.candidate.note_name == "C4"
    assert [candidate.note_name for candidate in candidates] == ["C4"]
    assert debug is not None
    if debug["primaryPromotion"] is not None:
        assert debug["primaryPromotion"]["replacedPrimaryNote"] == "E5"
        assert debug["primaryPromotion"]["replacementNote"] == "C4"
        assert debug["primaryPromotion"]["replacementOnsetGain"] > debug["primaryPromotion"]["replacedPrimaryOnsetGain"]


def test_segment_peaks_suppresses_recent_upper_carryover_with_weak_onset() -> None:
    tuning = get_default_tunings()[0]
    total_duration = 1.3
    audio = np.zeros(int(44100 * total_duration), dtype=np.float32)
    stale_e5 = synthesize_note(659.2551138257398, duration=1.0)
    fresh_g4 = synthesize_note(391.99543598174927, duration=0.5)
    audio[:len(stale_e5)] += stale_e5
    start = int(44100 * 0.5)
    audio[start:start + len(fresh_g4)] += fresh_g4
    peak = np.max(np.abs(audio))
    audio = audio if peak < 1e-6 else (audio / peak).astype(np.float32)

    candidates, debug, primary = segment_peaks(
        audio,
        44100,
        0.5,
        1.0,
        tuning,
        debug=True,
        recent_note_names={"E5", "G4"},
    )

    assert primary is not None
    assert primary.candidate.note_name == "G4"
    assert [candidate.note_name for candidate in candidates] == ["G4"]
    assert debug is not None
    assert any(
        item["noteName"] == "E5" and not item.get("accepted") and "recent-carryover-candidate" in item.get("reasons", [])
        for item in debug["secondaryDecisionTrail"]
    )


def test_segment_peaks_keeps_fresh_recent_upper_dyad_when_both_notes_attack() -> None:
    tuning = get_default_tunings()[0]
    total_duration = 1.0
    audio = np.zeros(int(44100 * total_duration), dtype=np.float32)
    fresh_e5 = synthesize_note(659.2551138257398, duration=0.45)
    fresh_c5 = synthesize_note(523.2511306011972, duration=0.45)
    start = int(44100 * 0.5)
    audio[start:start + len(fresh_e5)] += fresh_e5
    audio[start:start + len(fresh_c5)] += fresh_c5
    peak = np.max(np.abs(audio))
    audio = audio if peak < 1e-6 else (audio / peak).astype(np.float32)

    candidates, debug, _ = segment_peaks(
        audio,
        44100,
        0.5,
        0.95,
        tuning,
        debug=True,
        recent_note_names={"C5", "E5"},
    )

    assert sorted(candidate.note_name for candidate in candidates) == ["C5", "E5"]
    assert debug is not None

def test_build_recent_ascending_primary_run_ceiling_uses_latest_suffix() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    d6 = NoteCandidate(key=1, note_name="D6", frequency=1174.6590716696303, pitch_class="D", octave=6)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.2, notes=[d6], is_gliss_like=False, primary_note_name="D6", primary_score=500.0),
        RawEvent(start_time=0.2, end_time=0.4, notes=[d4], is_gliss_like=False, primary_note_name="D4", primary_score=120.0),
        RawEvent(start_time=0.4, end_time=0.6, notes=[f4], is_gliss_like=False, primary_note_name="F4", primary_score=130.0),
        RawEvent(start_time=0.6, end_time=0.8, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=140.0),
    ]

    assert build_recent_ascending_primary_run_ceiling(raw_events) == g4.frequency




def test_segment_peaks_suppresses_weak_lower_secondary_without_recent_context() -> None:
    tuning = get_default_tunings()[0]
    total_duration = 1.0
    audio = np.zeros(int(44100 * total_duration), dtype=np.float32)
    stale_e4 = synthesize_note(329.6275569128699, duration=0.9)
    fresh_a4 = synthesize_note(440.0, duration=0.35)
    audio[:len(stale_e4)] += stale_e4
    start = int(44100 * 0.5)
    audio[start:start + len(fresh_a4)] += fresh_a4
    peak = np.max(np.abs(audio))
    audio = audio if peak < 1e-6 else (audio / peak).astype(np.float32)

    candidates, debug, primary = segment_peaks(
        audio,
        44100,
        0.5,
        0.9,
        tuning,
        debug=True,
        recent_note_names={"A4"},
    )

    assert primary is not None
    assert primary.candidate.note_name == "A4"
    assert [candidate.note_name for candidate in candidates] == ["A4"]
    assert debug is not None
    assert any(
        item["noteName"] == "E4" and not item.get("accepted")
        for item in debug["secondaryDecisionTrail"]
    )

def test_transcription_suppresses_repeated_primary_carryover_in_repeat03_fixture() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "manual-captures" / "kalimba-17-c-c4-to-e6-sequence-17-repeat-03-01"
    request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    audio_bytes = (fixture_dir / "audio.wav").read_bytes()

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert 51 <= len(payload["events"]) <= 52
    assert payload["debug"]["multiOnsetGapSegments"] == [
        [7.7013, 8.0373],
        [8.0373, 8.3173],
        [8.3173, 8.8933],
        [8.8933, 9.4973],
    ]
    assert payload["debug"]["leadingOrphanSegments"] == [[1.9893, 2.2293]]
    assert payload["debug"]["singleOnsetGapHeadSegments"] == [[3.3147, 3.5547]]
    assert payload["debug"]["postTailGapHeadSegments"] == [[12.9653, 13.6053], [13.6053, 14.1813]]
    assert note_sets[:5] == ["C4", "D4", "E4", "F4", "G4"]
    assert note_sets[17:22] == ["C4", "D4", "E4", "F4", "G4"]
    assert payload["debug"]["sparseGapTailSegments"] == [
        [11.9067, 12.1467],
        [19.1173, 19.464],
        [19.464, 19.704],
    ]
    assert payload["debug"]["closeTerminalOrphanSegments"] == [[26.0533, 26.2933]]
    gap_onset_sets = [item["gapOnsets"] for item in payload["debug"]["gapIoiDiagnostics"]]
    assert [3.3147] in gap_onset_sets
    assert any(13.6053 in onset_set for onset_set in gap_onset_sets)
    assert "C5+G5" not in note_sets
    assert "E5+E6" not in note_sets
    assert note_sets[31:35] == ["C6", "D6", "E5", "E6"]
    assert note_sets[-2:] == ["D6", "E6"]




def test_build_recent_note_names_collapses_consecutive_duplicates() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[c4], is_gliss_like=False),
        RawEvent(start_time=0.4, end_time=0.8, notes=[d4], is_gliss_like=False),
        RawEvent(start_time=0.8, end_time=1.2, notes=[e4], is_gliss_like=False),
        RawEvent(start_time=1.2, end_time=1.5, notes=[f4], is_gliss_like=False),
        RawEvent(start_time=1.5, end_time=1.7, notes=[f4], is_gliss_like=False),
        RawEvent(start_time=1.7, end_time=1.9, notes=[f4], is_gliss_like=False),
    ]

    assert build_recent_note_names(raw_events) == {"C4", "D4", "E4", "F4"}

def test_transcription_drops_low_register_sparse_gap_tail_helpers_in_d4_d5_fixture() -> None:
    fixture_dir = Path(__file__).parent / "fixtures" / "manual-captures" / "kalimba-17-c-d4-d5-sequence-01"
    request_payload = json.loads((fixture_dir / "request.json").read_text(encoding="utf-8"))
    audio_bytes = (fixture_dir / "audio.wav").read_bytes()

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == ["D4", "D5", "D4", "D5", "D4", "D5"]
    assert payload["debug"]["sparseGapTailSegments"] == [[1.28, 1.52], [5.6827, 5.9227]]
    sparse_candidates = [
        candidate for candidate in payload["debug"]["segmentCandidates"]
        if candidate.get("segmentSource") == "sparse_gap_tail"
    ]
    assert len(sparse_candidates) == 2
    assert all(candidate.get("droppedBy") == "low_register_sparse_gap_tail" for candidate in sparse_candidates)


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

def test_suppress_resonant_carryover_keeps_phrase_reset_ascending_dyad() -> None:
    c5 = NoteCandidate(key=14, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    g5 = NoteCandidate(key=3, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.08, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.08, end_time=0.33, notes=[e5, g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
        RawEvent(start_time=0.92, end_time=1.24, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=260.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C5", "E5"],
        ["E5", "G5"],
        ["G4"],
    ]

def test_collapse_same_start_primary_singletons_prefers_singleton_over_lower_carryover() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.12, notes=[a4, e4], is_gliss_like=False, primary_note_name="A4", primary_score=140.0),
        RawEvent(start_time=0.0, end_time=0.32, notes=[a4], is_gliss_like=False, primary_note_name="A4", primary_score=180.0),
    ]

    cleaned = collapse_same_start_primary_singletons(raw_events)
    assert len(cleaned) == 1
    assert [note.note_name for note in cleaned[0].notes] == ["A4"]
    assert cleaned[0].start_time == 0.0
    assert cleaned[0].end_time == 0.32


def test_collapse_same_start_primary_singletons_prefers_singleton_when_following_primary_is_lower_carryover() -> None:
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.18, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=220.0),
        RawEvent(start_time=0.0, end_time=0.32, notes=[c4, g4], is_gliss_like=False, primary_note_name="C4", primary_score=80.0),
    ]

    cleaned = collapse_same_start_primary_singletons(raw_events)

    assert len(cleaned) == 1
    assert [note.note_name for note in cleaned[0].notes] == ["G4"]
    assert cleaned[0].start_time == 0.0
    assert cleaned[0].end_time == 0.32


def test_merge_short_chord_clusters_merges_singleton_and_dyad_into_triad() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.14, notes=[c4], is_gliss_like=False, primary_note_name="C4", primary_score=80.0),
        RawEvent(start_time=0.14, end_time=0.9, notes=[e4, g4], is_gliss_like=False, primary_note_name="G4", primary_score=500.0),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C4", "E4", "G4"]]


def test_merge_short_chord_clusters_does_not_merge_gliss_like_head_with_dyad() -> None:
    c5 = NoteCandidate(key=13, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e5 = NoteCandidate(key=15, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    g5 = NoteCandidate(key=17, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.0667, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.0667, end_time=0.3174, notes=[g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C5", "E5"], ["G5"]]

def test_merge_short_gliss_clusters_does_not_merge_gliss_head_with_longer_overlapping_dyad() -> None:
    c5 = NoteCandidate(key=13, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e5 = NoteCandidate(key=15, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    g5 = NoteCandidate(key=17, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.0667, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.0667, end_time=0.3174, notes=[e5, g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
    ]

    merged = merge_short_gliss_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C5", "E5"], ["E5", "G5"]]

def test_simplify_short_gliss_prefix_to_contiguous_singleton_handles_dyad_head_before_dyad() -> None:
    c5 = NoteCandidate(key=5, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)
    g5 = NoteCandidate(key=3, note_name="G5", frequency=783.9908719634985, pitch_class="G", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.0667, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.0667, end_time=0.3174, notes=[e5, g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
    ]

    cleaned = simplify_short_gliss_prefix_to_contiguous_singleton(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [["C5"], ["E5", "G5"]]


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


def test_normalize_repeated_four_note_family_promotes_complementary_triads() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.9, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=900.0),
        RawEvent(start_time=1.0, end_time=1.2, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=700.0),
        RawEvent(start_time=1.2, end_time=1.8, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=300.0),
        RawEvent(start_time=2.0, end_time=2.1, notes=[f4], is_gliss_like=False, primary_note_name="F4", primary_score=80.0),
        RawEvent(start_time=2.3, end_time=3.0, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=950.0),
        RawEvent(start_time=3.2, end_time=3.8, notes=[g4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=400.0),
        RawEvent(start_time=4.0, end_time=4.8, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=980.0),
    ]

    normalized = normalize_repeated_four_note_family(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]


@pytest.mark.xfail(
    reason="Global repeated four-note normalization remains intentionally broad until #17 redesign.",
    strict=False,
)
def test_normalize_repeated_four_note_family_stays_within_local_context_gap() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.9, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=900.0),
        RawEvent(start_time=1.0, end_time=1.2, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=700.0),
        RawEvent(start_time=1.2, end_time=1.8, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=300.0),
        RawEvent(start_time=4.0, end_time=4.8, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=950.0),
        RawEvent(start_time=5.0, end_time=5.3, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=320.0),
        RawEvent(start_time=9.0, end_time=9.3, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=330.0),
        RawEvent(start_time=12.0, end_time=12.8, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=980.0),
    ]

    normalized = normalize_repeated_four_note_family(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["D5"],
        ["G4", "B4", "D5"],
    ]


def test_normalize_repeated_explicit_four_note_patterns_drops_terminal_subset_tail() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.9, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=900.0),
        RawEvent(start_time=1.0, end_time=1.8, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=920.0),
        RawEvent(start_time=2.0, end_time=2.8, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=910.0),
        RawEvent(start_time=3.0, end_time=3.8, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=930.0),
        RawEvent(start_time=4.05, end_time=4.32, notes=[g4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=500.0),
    ]

    normalized = normalize_repeated_explicit_four_note_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]


def test_normalize_repeated_explicit_four_note_patterns_cleans_trailing_subsets() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    f5 = NoteCandidate(key=14, note_name="F5", frequency=698.4564628660078, pitch_class="F", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.9, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=900.0),
        RawEvent(start_time=1.0, end_time=1.8, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=920.0),
        RawEvent(start_time=2.0, end_time=2.8, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=910.0),
        RawEvent(start_time=3.0, end_time=3.8, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=930.0),
        RawEvent(start_time=4.0, end_time=4.7, notes=[b4, d5, f5], is_gliss_like=True, primary_note_name="D5", primary_score=760.0),
        RawEvent(start_time=4.7, end_time=4.95, notes=[g4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=350.0),
        RawEvent(start_time=4.95, end_time=5.2, notes=[e4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=320.0),
    ]

    normalized = normalize_repeated_explicit_four_note_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]


def test_suppress_repeated_triad_blips_drops_short_middle_burst() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.9, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=1200.0),
        RawEvent(start_time=1.0, end_time=1.2, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="A4", primary_score=500.0),
        RawEvent(start_time=1.4, end_time=2.2, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="F4", primary_score=1300.0),
        RawEvent(start_time=2.5, end_time=3.1, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=1250.0),
    ]

    cleaned = suppress_repeated_triad_blips(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
    ]


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


def test_simplify_short_gliss_prefix_to_contiguous_singleton_picks_matching_note() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.08, notes=[e4, f4], is_gliss_like=True, primary_note_name="E4", primary_score=120.0),
        RawEvent(start_time=0.08, end_time=0.9, notes=[g4, b4, d5], is_gliss_like=True, primary_note_name="G4", primary_score=500.0),
    ]

    simplified = simplify_short_gliss_prefix_to_contiguous_singleton(raw_events)
    assert [[note.note_name for note in event.notes] for event in simplified] == [["E4"], ["G4", "B4", "D5"]]


def test_merge_four_note_gliss_clusters_merges_triad_and_singleton() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[e4, g4, b4], is_gliss_like=True, primary_note_name="E4", primary_score=500.0),
        RawEvent(start_time=0.6, end_time=0.82, notes=[d5], is_gliss_like=True, primary_note_name="D5", primary_score=420.0),
    ]

    merged = merge_four_note_gliss_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["E4", "G4", "B4", "D5"]]
    assert merged[0].is_gliss_like is True



def test_suppress_leading_gliss_neighbor_noise_drops_short_dyad_before_four_note_gliss() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.08, notes=[e4, f4], is_gliss_like=True, primary_note_name="E4", primary_score=120.0),
        RawEvent(start_time=0.08, end_time=0.9, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=500.0),
    ]

    cleaned = suppress_leading_gliss_neighbor_noise(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [["E4", "G4", "B4", "D5"]]


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

def test_probe_four_note_gliss_pending_capture() -> None:
    fixture = Path(__file__).parent / "fixtures" / "manual-captures" / "kalimba-17-c-e4-g4-b4-d5-four-note-gliss-ascending-01"
    request_payload = json.loads((fixture / "request.json").read_text(encoding="utf-8"))
    audio_bytes = (fixture / "audio.wav").read_bytes()

    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(request_payload["tuning"]), "debug": "true"},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert len(payload["events"]) <= 5
    assert note_sets.count("B4+D5+E4+G4") >= 4


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



def test_classify_event_gesture_strict_chord() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    event = RawEvent(start_time=0.0, end_time=0.7, notes=[c4, e4, g4], is_gliss_like=False)
    assert classify_event_gesture(event, 0, [event], [event]) == "strict_chord"


def test_classify_event_gesture_slide_chord_from_subset_growth() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.5, notes=[g4, b4, d5], is_gliss_like=False),
        RawEvent(start_time=0.5, end_time=0.7, notes=[e4, g4, b4], is_gliss_like=False),
    ]
    merged_event = RawEvent(start_time=0.0, end_time=0.7, notes=[e4, g4, b4, d5], is_gliss_like=False)
    assert classify_event_gesture(merged_event, 0, raw_events, [merged_event]) == "slide_chord"


def test_classify_event_gesture_slide_chord_from_neighbor_progression() -> None:
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    merged_events = [
        RawEvent(start_time=0.0, end_time=0.5, notes=[c4, e4, g4], is_gliss_like=True),
        RawEvent(start_time=0.52, end_time=1.0, notes=[d4, f4, a4], is_gliss_like=True),
    ]
    assert classify_event_gesture(merged_events[0], 0, merged_events, merged_events) == "slide_chord"



def test_normalize_repeated_explicit_four_note_patterns_merges_leading_subset_into_dominant_take() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.8, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=900.0),
        RawEvent(start_time=1.0, end_time=2.05, notes=[g4, b4, d5], is_gliss_like=True, primary_note_name="G4", primary_score=640.0),
        RawEvent(start_time=2.05, end_time=2.8, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=930.0),
        RawEvent(start_time=3.1, end_time=3.9, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="B4", primary_score=910.0),
        RawEvent(start_time=4.2, end_time=5.0, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="D5", primary_score=920.0),
    ]

    normalized = normalize_repeated_explicit_four_note_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]
    assert normalized[1].start_time == pytest.approx(1.0)
    assert normalized[1].end_time == pytest.approx(2.8)


def test_normalize_repeated_explicit_four_note_patterns_merges_adjacent_strict_subsets_into_dominant_take() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.26, notes=[e4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=720.0),
        RawEvent(start_time=0.28, end_time=0.72, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=810.0),
        RawEvent(start_time=1.0, end_time=1.6, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=930.0),
        RawEvent(start_time=1.9, end_time=2.45, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=910.0),
        RawEvent(start_time=2.8, end_time=3.35, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=920.0),
    ]

    normalized = normalize_repeated_explicit_four_note_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]
    assert normalized[0].start_time == pytest.approx(0.0)
    assert normalized[0].end_time == pytest.approx(0.72)
    assert normalized[0].is_gliss_like is False


def test_normalize_repeated_explicit_four_note_patterns_absorbs_short_gliss_prefix_noise() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.8, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=900.0),
        RawEvent(start_time=1.0, end_time=1.08, notes=[f4], is_gliss_like=True, primary_note_name="F4", primary_score=120.0),
        RawEvent(start_time=1.08, end_time=1.9, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=950.0),
        RawEvent(start_time=2.2, end_time=3.0, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="G4", primary_score=920.0),
        RawEvent(start_time=3.3, end_time=4.1, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="D5", primary_score=930.0),
    ]

    normalized = normalize_repeated_explicit_four_note_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]
    assert normalized[1].start_time == pytest.approx(1.0)
    assert normalized[1].end_time == pytest.approx(1.9)



def test_normalize_strict_four_note_subsets_merges_leading_dyad_into_following_dominant() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.42, notes=[e4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=640.0),
        RawEvent(start_time=0.42, end_time=0.84, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=910.0),
        RawEvent(start_time=1.2, end_time=1.75, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=900.0),
        RawEvent(start_time=2.1, end_time=2.7, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=920.0),
        RawEvent(start_time=3.0, end_time=3.6, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=915.0),
    ]

    normalized = normalize_strict_four_note_subsets(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]
    assert normalized[0].start_time == pytest.approx(0.0)
    assert normalized[0].end_time == pytest.approx(0.84)


def test_normalize_repeated_four_note_gliss_patterns_promotes_subsets_and_drops_short_noise() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=1.0, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=900.0),
        RawEvent(start_time=1.3, end_time=2.4, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=600.0),
        RawEvent(start_time=2.7, end_time=3.6, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=920.0),
        RawEvent(start_time=4.0, end_time=4.9, notes=[d5], is_gliss_like=True, primary_note_name="D5", primary_score=580.0),
        RawEvent(start_time=5.2, end_time=5.28, notes=[e4, f4], is_gliss_like=False, primary_note_name="F4", primary_score=110.0),
        RawEvent(start_time=5.28, end_time=6.2, notes=[g4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=560.0),
    ]

    normalized = normalize_repeated_four_note_gliss_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4", "D5"],
    ]





def test_classify_event_gesture_slide_chord_from_gliss_like_family_without_neighbor_shift() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[g4, b4, d5], is_gliss_like=True),
        RawEvent(start_time=0.6, end_time=1.1, notes=[e4, g4, b4, d5], is_gliss_like=True),
    ]
    merged_event = RawEvent(start_time=0.0, end_time=1.1, notes=[e4, g4, b4, d5], is_gliss_like=True)

    assert classify_event_gesture(merged_event, 0, raw_events, [merged_event]) == "slide_chord"

def test_repeated_pattern_pass_ids_are_unique() -> None:
    assert len(REPEATED_PATTERN_PASS_IDS) == len(set(REPEATED_PATTERN_PASS_IDS))


def test_apply_repeated_pattern_passes_can_disable_triad_normalizer() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.75, notes=[d4, f4], is_gliss_like=False, primary_note_name="D4", primary_score=700.0),
        RawEvent(start_time=1.0, end_time=1.7, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=900.0),
        RawEvent(start_time=2.0, end_time=2.7, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="F4", primary_score=920.0),
        RawEvent(start_time=3.0, end_time=3.7, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="A4", primary_score=910.0),
    ]

    enabled, _ = apply_repeated_pattern_passes(raw_events)
    disabled, trace = apply_repeated_pattern_passes(
        raw_events,
        disabled_passes=frozenset({"normalize_repeated_triad_patterns"}),
        debug=True,
    )

    assert sorted(note.note_name for note in enabled[0].notes) == ["A4", "D4", "F4"]
    assert sorted(note.note_name for note in disabled[0].notes) == ["D4", "F4"]
    triad_trace = next(item for item in trace if item["pass"] == "normalize_repeated_triad_patterns")
    assert triad_trace["enabled"] is False


def test_transcription_debug_reports_disabled_repeated_pattern_passes() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_repeated_chord((293.6647679174076, 349.2282314330039, 440.0), repeats=4)

    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(tuning.model_dump(by_alias=True)),
            "debug": "true",
            "disabledRepeatedPatternPasses": json.dumps(["normalize_repeated_triad_patterns"]),
        },
        files={"file": ("d4-f4-a4.wav", wav_bytes(audio), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["disabledRepeatedPatternPasses"] == ["normalize_repeated_triad_patterns"]
    triad_trace = next(item for item in payload["debug"]["repeatedPatternPassTrace"] if item["pass"] == "normalize_repeated_triad_patterns")
    assert triad_trace["enabled"] is False

