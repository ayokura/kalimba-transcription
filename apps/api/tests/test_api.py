from copy import deepcopy
from functools import lru_cache
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.tunings import get_default_tunings
from app.transcription import REPEATED_PATTERN_PASS_IDS, GapAttackCandidates, NoteCandidate, NoteHypothesis, OnsetAttackProfile, RawEvent, apply_repeated_pattern_passes, build_recent_ascending_primary_run_ceiling, build_recent_note_names, classify_event_gesture, collapse_ascending_restart_lower_residue_singletons, collapse_late_descending_step_handoffs, collapse_same_start_primary_singletons, simplify_descending_adjacent_dyad_residue, collect_attack_validated_gap_candidates, collect_multi_onset_gap_segments, collect_terminal_multi_onset_segments, collect_two_onset_terminal_tail_segments, detect_segments, is_adjacent_tuning_step, merge_four_note_gliss_clusters, merge_short_chord_clusters, merge_short_gliss_clusters, normalize_repeated_explicit_four_note_patterns, normalize_repeated_four_note_family, normalize_repeated_triad_patterns, normalize_strict_four_note_subsets, select_contiguous_four_note_cluster, is_slide_playable_contiguous_cluster, should_block_descending_repeated_primary_tertiary_extension, should_keep_dense_trailing_onset, simplify_short_gliss_prefix_to_contiguous_singleton, simplify_short_secondary_bleed, suppress_descending_restart_residual_cluster, suppress_descending_terminal_residual_cluster, suppress_descending_upper_return_overlap, suppress_isolated_triad_extensions, suppress_leading_descending_overlap, suppress_leading_gliss_neighbor_noise, suppress_repeated_triad_blips, segment_peaks, suppress_leading_gliss_subset_transients, suppress_resonant_carryover, suppress_short_residual_tails, suppress_subset_decay_events

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

def test_is_slide_playable_contiguous_cluster_accepts_center_triads() -> None:
    tuning = get_default_tunings()[0]
    notes = [next(note for note in tuning.notes if note.note_name == name) for name in ["C4", "E4", "G4"]]

    assert is_slide_playable_contiguous_cluster(notes, tuning) is True


def test_is_slide_playable_contiguous_cluster_rejects_non_slide_center_crossing_cluster() -> None:
    tuning = get_default_tunings()[0]
    notes = [next(note for note in tuning.notes if note.note_name == name) for name in ["C4", "D4", "F4"]]

    assert is_slide_playable_contiguous_cluster(notes, tuning) is False


def test_suppress_descending_terminal_residual_cluster_drops_rebound_tail() -> None:
    tuning = get_default_tunings()[0]
    note_by_name = {note.note_name: note for note in tuning.notes}
    raw_events = [
        RawEvent(0.0, 0.2, [note_by_name["F4"]], False, "F4", 100.0),
        RawEvent(0.2, 0.4, [note_by_name["E4"]], False, "E4", 100.0),
        RawEvent(0.4, 0.6, [note_by_name["D4"]], False, "D4", 100.0),
        RawEvent(0.6, 0.8, [note_by_name["C4"]], False, "C4", 100.0),
        RawEvent(0.8, 1.0, [note_by_name["D4"], note_by_name["F4"]], False, "D4", 100.0),
    ]

    cleaned = suppress_descending_terminal_residual_cluster(raw_events, tuning)

    assert cleaned == raw_events[:-1]


def test_suppress_descending_terminal_residual_cluster_keeps_non_rebound_tail() -> None:
    tuning = get_default_tunings()[0]
    note_by_name = {note.note_name: note for note in tuning.notes}
    raw_events = [
        RawEvent(0.0, 0.2, [note_by_name["F4"]], False, "F4", 100.0),
        RawEvent(0.2, 0.4, [note_by_name["E4"]], False, "E4", 100.0),
        RawEvent(0.4, 0.6, [note_by_name["D4"]], False, "D4", 100.0),
        RawEvent(0.6, 0.8, [note_by_name["C4"]], False, "C4", 100.0),
        RawEvent(0.8, 1.0, [note_by_name["C4"], note_by_name["E4"], note_by_name["G4"]], True, "C4", 100.0),
    ]

    cleaned = suppress_descending_terminal_residual_cluster(raw_events, tuning)

    assert cleaned == raw_events


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


def test_select_contiguous_four_note_cluster_promotes_dense_local_family() -> None:
    ranked = [
        NoteHypothesis(NoteCandidate(13, 'D5', 587.33, 'D', 5), 1000.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(12, 'B4', 493.88, 'B', 4), 800.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(11, 'G4', 391.99, 'G', 4), 600.0, 0.0, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(10, 'E4', 329.63, 'E', 4), 180.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(9, 'D4', 293.66, 'D', 4), 100.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0),
    ]

    selected = select_contiguous_four_note_cluster(ranked[0], ranked, 0.7)

    assert selected is not None
    assert [candidate.note_name for candidate in selected] == ['E4', 'G4', 'B4', 'D5']


def test_select_contiguous_four_note_cluster_rejects_short_or_ambiguous_window() -> None:
    ranked = [
        NoteHypothesis(NoteCandidate(13, 'D5', 587.33, 'D', 5), 1000.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(12, 'B4', 493.88, 'B', 4), 800.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(11, 'G4', 391.99, 'G', 4), 600.0, 0.0, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(10, 'E4', 329.63, 'E', 4), 120.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(9, 'D4', 293.66, 'D', 4), 110.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0),
    ]

    assert select_contiguous_four_note_cluster(ranked[0], ranked, 0.4) is None
    assert select_contiguous_four_note_cluster(ranked[0], ranked, 0.7) is None



def test_collect_multi_onset_gap_segments_requires_long_regular_gap() -> None:
    active_ranges = [(0.0, 0.12), (2.5, 2.68)]
    onset_times = [0.62, 0.94, 1.24, 1.82]

    segments = collect_multi_onset_gap_segments(active_ranges, onset_times)

    assert segments == [(0.62, 0.94), (0.94, 1.24), (1.24, 1.82), (1.82, 2.5)]


def test_collect_attack_validated_gap_candidates_returns_valid_attack_subset() -> None:
    active_ranges = [(2.4573, 3.0747), (11.3773, 11.9013)]
    onset_times = [3.504, 3.552, 5.0613, 5.68, 6.3013, 7.5467, 8.9467, 8.9893, 9.0667]
    onset_profiles = {
        3.504: OnsetAttackProfile(3.504, 4.219822, 28.350282, 2.38015, True),
        3.552: OnsetAttackProfile(3.552, 1.300026, 0.000596, 0.052879, False),
        5.0613: OnsetAttackProfile(5.0613, 1996.969355, 188.369779, 141.707333, True),
        5.68: OnsetAttackProfile(5.68, 0.807941, 54.555144, 1.22241, True),
        6.3013: OnsetAttackProfile(6.3013, 4.875818, 23.044867, 2.73317, True),
        7.5467: OnsetAttackProfile(7.5467, 35.267216, 67.174492, 8.80443, True),
        8.9467: OnsetAttackProfile(8.9467, 18.432635, 30.794041, 13.678491, True),
        8.9893: OnsetAttackProfile(8.9893, 4.577633, 0.810421, 0.784021, False),
        9.0667: OnsetAttackProfile(9.0667, 53.035509, 10.489827, 7.509409, True),
    }

    candidates = collect_attack_validated_gap_candidates(active_ranges, onset_times, onset_profiles, 12.319)

    assert [[round(value, 4) for value in gap] for gap in candidates.inter_ranges] == [[3.504, 5.0613, 5.68, 6.3013, 7.5467, 8.9467, 9.0667]]
    assert candidates.leading == []
    assert candidates.trailing == []


def test_collect_multi_onset_gap_segments_promotes_attack_validated_run_candidates() -> None:
    active_ranges = [(2.4573, 3.0747), (11.3773, 11.9013)]
    onset_times = [3.504, 3.552, 5.0613, 5.68, 6.3013, 7.5467, 8.9467, 9.0667]
    gap_attack_candidates = GapAttackCandidates(
        inter_ranges=[[3.504, 5.0613, 5.68, 6.3013, 7.5467, 8.9467, 9.0667]],
        leading=[],
        trailing=[],
    )

    segments = collect_multi_onset_gap_segments(active_ranges, onset_times, gap_attack_candidates=gap_attack_candidates)

    assert [tuple(round(value, 4) for value in segment) for segment in segments] == [
        (3.504, 5.0613),
        (5.0613, 5.68),
        (5.68, 6.3013),
        (6.3013, 7.5467),
        (7.5467, 8.9467),
        (8.9467, 9.0667),
        (9.0667, 9.3867),
    ]


def test_collect_terminal_multi_onset_segments_requires_close_orphan_then_regular_run() -> None:
    active_ranges = [(21.692, 21.968)]
    onset_times = [22.112, 23.088, 23.5307, 23.96, 24.3627]

    segments = collect_terminal_multi_onset_segments(active_ranges, onset_times, 24.9)

    assert segments == [(23.088, 23.5307), (23.5307, 23.96), (23.96, 24.3627), (24.3627, 24.6827)]


def test_collect_two_onset_terminal_tail_segments_requires_sparse_two_hit_tail() -> None:
    active_ranges = [(4.2013, 4.6533)]
    onset_times = [2.032, 3.0853, 4.2027, 5.9147, 7.0]

    segments = collect_two_onset_terminal_tail_segments(active_ranges, onset_times, 7.8)

    assert segments == [(5.9147, 6.2347), (7.0, 7.32)]


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


def test_should_keep_dense_trailing_onset_preserves_penultimate_dense_onset() -> None:
    boundary_times = [6.1813, 6.4267, 6.5733, 6.96]

    assert should_keep_dense_trailing_onset(boundary_times, 2, 3.996, 7.424) is True
    assert should_keep_dense_trailing_onset(boundary_times, 1, 3.996, 7.424) is False


def test_detect_segments_does_not_backtrack_into_previous_active_range(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.transcription as transcription

    frame_times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0], dtype=np.float32)
    rms = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    onset_times = np.array([0.45, 0.79], dtype=np.float32)

    monkeypatch.setattr(transcription.librosa.feature, "rms", lambda **kwargs: np.array([rms], dtype=np.float32))
    monkeypatch.setattr(transcription.librosa.onset, "onset_strength", lambda **kwargs: np.zeros_like(rms))
    monkeypatch.setattr(transcription.librosa.onset, "onset_detect", lambda **kwargs: np.array([0, 1], dtype=np.int64))

    def fake_frames_to_time(frames, **kwargs):
        frames = np.asarray(frames)
        if len(frames) == len(rms):
            return frame_times
        return onset_times

    monkeypatch.setattr(transcription.librosa, "frames_to_time", fake_frames_to_time)
    monkeypatch.setattr(transcription.librosa, "get_duration", lambda **kwargs: 1.08)
    monkeypatch.setattr(transcription.librosa.beat, "beat_track", lambda **kwargs: (np.array([90.0]), np.array([], dtype=np.int64)))

    segments, _, _ = detect_segments(np.zeros(44100, dtype=np.float32), 44100)

    late_segments = [(round(start, 2), round(end, 2)) for start, end in segments if start >= 0.58]

    assert len(late_segments) == 1
    assert late_segments[0][0] == pytest.approx(0.79)
    assert late_segments[0][1] == pytest.approx(1.08)

def test_simplify_short_secondary_bleed_strips_restart_stale_upper_note() -> None:
    c4 = NoteCandidate(9, "C4", 261.6255653005986, "C", 4)
    d4 = NoteCandidate(8, "D4", 293.6647679174076, "D", 4)
    e6 = NoteCandidate(17, "E6", 1318.5102276514797, "E", 6)
    events = [
        RawEvent(0.0, 0.18, [e6], False, "E6", 400.0),
        RawEvent(0.18, 0.38, [c4, e6], False, "C4", 350.0),
        RawEvent(0.38, 0.62, [d4], False, "D4", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["C4"]


def test_simplify_short_secondary_bleed_collapses_mirrored_adjacent_run_to_upper_note() -> None:
    d4 = NoteCandidate(8, "D4", 293.6647679174076, "D", 4)
    e4 = NoteCandidate(10, "E4", 329.6275569128699, "E", 4)
    f4 = NoteCandidate(7, "F4", 349.2282314330039, "F", 4)
    events = [
        RawEvent(0.0, 0.24, [d4], False, "D4", 340.0),
        RawEvent(0.24, 0.42, [d4, e4], False, "D4", 300.0),
        RawEvent(0.42, 0.6, [d4, e4], False, "D4", 295.0),
        RawEvent(0.6, 0.86, [f4], False, "F4", 360.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["E4"]
    assert [note.note_name for note in simplified[2].notes] == ["E4"]



def test_simplify_short_secondary_bleed_strips_descending_stale_upper_step() -> None:
    g5 = NoteCandidate(3, "G5", 783.9908719634985, "G", 5)
    f5 = NoteCandidate(14, "F5", 698.4564628660078, "F", 5)
    e5 = NoteCandidate(4, "E5", 659.2551138257398, "E", 5)
    events = [
        RawEvent(0.0, 0.22, [g5], False, "G5", 340.0),
        RawEvent(0.22, 0.4, [f5, g5], False, "F5", 300.0),
        RawEvent(0.4, 0.62, [e5], False, "E5", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["F5"]


def test_simplify_short_secondary_bleed_promotes_descending_lower_step() -> None:
    c5 = NoteCandidate(5, "C5", 523.2511306011972, "C", 5)
    b4 = NoteCandidate(12, "B4", 493.8833012561241, "B", 4)
    a4 = NoteCandidate(6, "A4", 440.0, "A", 4)
    events = [
        RawEvent(0.0, 0.22, [c5], False, "C5", 340.0),
        RawEvent(0.22, 0.4, [b4, c5], False, "C5", 300.0),
        RawEvent(0.4, 0.62, [a4], False, "A4", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["B4"]


def test_simplify_short_secondary_bleed_collapses_descending_upper_residue_to_primary() -> None:
    g4 = NoteCandidate(11, "G4", 391.99543598174927, "G", 4)
    b4 = NoteCandidate(12, "B4", 493.8833012561241, "B", 4)
    f4 = NoteCandidate(7, "F4", 349.2282314330039, "F", 4)
    events = [
        RawEvent(0.0, 0.18, [g4], False, "G4", 320.0),
        RawEvent(0.18, 0.29, [g4, b4], False, "G4", 300.0),
        RawEvent(0.29, 0.5, [f4], False, "F4", 290.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["G4"]


def test_simplify_short_secondary_bleed_collapses_repeated_descending_handoff_to_primary() -> None:
    c5 = NoteCandidate(14, "C5", 523.2511306011972, "C", 5)
    b4 = NoteCandidate(12, "B4", 493.8833012561241, "B", 4)
    a4 = NoteCandidate(10, "A4", 440.0, "A", 4)
    events = [
        RawEvent(0.0, 0.18, [c5], False, "C5", 320.0),
        RawEvent(0.18, 0.31, [b4, c5], False, "B4", 300.0),
        RawEvent(0.31, 0.43, [b4, c5], False, "B4", 290.0),
        RawEvent(0.43, 0.62, [a4], False, "A4", 280.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["B4"]
    assert [note.note_name for note in simplified[2].notes] == ["B4"]


def test_suppress_descending_restart_residual_cluster_drops_repeated_low_register_residue() -> None:
    tuning = get_default_tunings()[0]
    c4 = NoteCandidate(0, "C4", 261.6255653005986, "C", 4)
    d4 = NoteCandidate(1, "D4", 293.6647679174076, "D", 4)
    e4 = NoteCandidate(2, "E4", 329.6275569128699, "E", 4)
    e6 = NoteCandidate(16, "E6", 1318.5102276514797, "E", 6)
    events = [
        RawEvent(0.0, 0.22, [c4], False, "C4", 320.0),
        RawEvent(0.22, 0.34, [d4, e4], False, "D4", 260.0),
        RawEvent(0.62, 0.84, [d4, e4], False, "D4", 240.0),
        RawEvent(1.12, 1.44, [e6], False, "E6", 300.0),
    ]

    cleaned = suppress_descending_restart_residual_cluster(events, tuning)

    assert [[note.note_name for note in event.notes] for event in cleaned] == [["C4"], ["E6"]]


def test_suppress_descending_restart_residual_cluster_drops_repeated_low_register_residue_before_large_restart_gap() -> None:
    tuning = get_default_tunings()[0]
    c4 = NoteCandidate(0, "C4", 261.6255653005986, "C", 4)
    d4 = NoteCandidate(1, "D4", 293.6647679174076, "D", 4)
    e4 = NoteCandidate(2, "E4", 329.6275569128699, "E", 4)
    e6 = NoteCandidate(16, "E6", 1318.5102276514797, "E", 6)
    events = [
        RawEvent(0.0, 0.22, [c4], False, "C4", 320.0),
        RawEvent(0.22, 0.32, [d4, e4], False, "D4", 260.0),
        RawEvent(0.74, 0.84, [d4, e4], False, "D4", 240.0),
        RawEvent(1.55, 1.88, [e6], False, "E6", 300.0),
    ]

    cleaned = suppress_descending_restart_residual_cluster(events, tuning)

    assert [[note.note_name for note in event.notes] for event in cleaned] == [["C4"], ["E6"]]

def test_simplify_descending_adjacent_dyad_residue_collapses_upper_residue_to_lower() -> None:
    a4 = NoteCandidate(10, "A4", 440.0, "A", 4)
    g4 = NoteCandidate(11, "G4", 391.99543598174927, "G", 4)
    f4 = NoteCandidate(7, "F4", 349.2282314330039, "F", 4)
    events = [
        RawEvent(0.0, 0.18, [a4], False, "A4", 320.0),
        RawEvent(0.18, 0.31, [g4, a4], False, "G4", 300.0),
        RawEvent(0.31, 0.5, [f4], False, "F4", 290.0),
    ]

    simplified = simplify_descending_adjacent_dyad_residue(events)

    assert [note.note_name for note in simplified[1].notes] == ["G4"]


def test_collapse_ascending_restart_lower_residue_singletons_keeps_previous_step() -> None:
    tuning = get_default_tunings()[0]
    d6 = NoteCandidate(15, "D6", 1174.6590716696303, "D", 6)
    b5 = NoteCandidate(13, "B5", 987.7666025122483, "B", 5)
    e6 = NoteCandidate(16, "E6", 1318.5102276514797, "E", 6)
    events = [
        RawEvent(0.0, 0.28, [d6], False, "D6", 320.0),
        RawEvent(0.28, 0.40, [b5], False, "B5", 260.0),
        RawEvent(0.40, 0.72, [e6], False, "E6", 340.0),
    ]

    cleaned = collapse_ascending_restart_lower_residue_singletons(events, tuning)

    assert [note.note_name for note in cleaned[1].notes] == ["D6"]


def test_simplify_short_secondary_bleed_keeps_primary_for_restart_upper_sandwich() -> None:
    b5 = NoteCandidate(2, "B5", 987.7666025122483, "B", 5)
    d6 = NoteCandidate(1, "D6", 1174.6590716696303, "D", 6)
    e6 = NoteCandidate(17, "E6", 1318.5102276514797, "E", 6)
    events = [
        RawEvent(0.0, 0.18, [e6], False, "E6", 320.0),
        RawEvent(0.18, 0.3, [b5, d6], False, "D6", 300.0),
        RawEvent(0.3, 0.54, [e6], False, "E6", 310.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["D6"]


def test_simplify_short_secondary_bleed_keeps_primary_for_restart_followed_by_upper_extension() -> None:
    b5 = NoteCandidate(2, "B5", 987.7666025122483, "B", 5)
    d6 = NoteCandidate(1, "D6", 1174.6590716696303, "D", 6)
    e6 = NoteCandidate(17, "E6", 1318.5102276514797, "E", 6)
    c6 = NoteCandidate(16, "C6", 1046.5022612023945, "C", 6)
    events = [
        RawEvent(0.0, 0.18, [e6], False, "E6", 320.0),
        RawEvent(0.18, 0.3, [b5, d6], False, "D6", 300.0),
        RawEvent(0.3, 0.52, [d6, e6], False, "D6", 310.0),
        RawEvent(0.52, 0.74, [c6], False, "C6", 290.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["D6"]


def test_collapse_late_descending_step_handoffs_keeps_lower_note() -> None:
    a4 = NoteCandidate(13, "A4", 440.0, "A", 4)
    g4 = NoteCandidate(11, "G4", 391.99543598174927, "G", 4)
    f4 = NoteCandidate(7, "F4", 349.2282314330039, "F", 4)
    events = [
        RawEvent(0.0, 0.22, [a4], False, "A4", 320.0),
        RawEvent(0.22, 0.34, [g4, a4], False, "G4", 280.0),
        RawEvent(0.34, 0.58, [f4], False, "F4", 300.0),
    ]

    cleaned = collapse_late_descending_step_handoffs(events)

    assert [note.note_name for note in cleaned[1].notes] == ["G4"]


def test_collapse_late_descending_step_handoffs_keeps_lower_note_when_middle_is_short_gliss_like() -> None:
    a4 = NoteCandidate(13, "A4", 440.0, "A", 4)
    g4 = NoteCandidate(11, "G4", 391.99543598174927, "G", 4)
    f4 = NoteCandidate(7, "F4", 349.2282314330039, "F", 4)
    events = [
        RawEvent(0.0, 0.22, [a4], False, "A4", 320.0),
        RawEvent(0.22, 0.34, [g4, a4], True, "G4", 280.0),
        RawEvent(0.34, 0.58, [f4], False, "F4", 300.0),
    ]

    cleaned = collapse_late_descending_step_handoffs(events)

    assert [note.note_name for note in cleaned[1].notes] == ["G4"]


def test_simplify_short_secondary_bleed_collapses_descending_bridge_to_upper() -> None:
    c6 = NoteCandidate(16, "C6", 1046.5022612023945, "C", 6)
    b5 = NoteCandidate(2, "B5", 987.7666025122483, "B", 5)
    a5 = NoteCandidate(13, "A5", 880.0, "A", 5)
    g5 = NoteCandidate(3, "G5", 783.9908719634985, "G", 5)
    events = [
        RawEvent(0.0, 0.22, [c6], False, "C6", 340.0),
        RawEvent(0.22, 0.4, [b5, g5], False, "B5", 300.0),
        RawEvent(0.4, 0.62, [a5], False, "A5", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["B5"]


def test_suppress_resonant_carryover_keeps_lower_note_in_descending_adjacent_chain() -> None:
    f4 = NoteCandidate(7, "F4", 349.2282314330039, "F", 4)
    g4 = NoteCandidate(11, "G4", 391.99543598174927, "G", 4)
    e4 = NoteCandidate(2, "E4", 329.6275569128699, "E", 4)
    events = [
        RawEvent(0.0, 0.18, [f4], False, "F4", 320.0),
        RawEvent(0.18, 0.29, [f4, g4], False, "F4", 300.0),
        RawEvent(0.29, 0.5, [e4], False, "E4", 290.0),
    ]

    cleaned = suppress_resonant_carryover(events)

    assert [note.note_name for note in cleaned[1].notes] == ["F4"]


def test_suppress_descending_upper_return_overlap_drops_residual_dyad() -> None:
    e6 = NoteCandidate(17, "E6", 1318.5102276514797, "E", 6)
    d6 = NoteCandidate(1, "D6", 1174.6590716696303, "D", 6)
    c6 = NoteCandidate(16, "C6", 1046.5022612023945, "C", 6)
    events = [
        RawEvent(0.0, 0.14, [e6], False, "E6", 320.0),
        RawEvent(0.14, 0.25, [d6], False, "D6", 300.0),
        RawEvent(0.25, 0.36, [d6, e6], False, "D6", 280.0),
        RawEvent(0.36, 0.58, [c6], False, "C6", 290.0),
    ]

    cleaned = suppress_descending_upper_return_overlap(events)

    assert [[note.note_name for note in event.notes] for event in cleaned] == [["E6"], ["D6"], ["C6"]]


def test_suppress_leading_descending_overlap_collapses_first_bridge() -> None:
    tuning = get_default_tunings()[0]
    e6 = NoteCandidate(17, "E6", 1318.5102276514797, "E", 6)
    d6 = NoteCandidate(1, "D6", 1174.6590716696303, "D", 6)
    c6 = NoteCandidate(16, "C6", 1046.5022612023945, "C", 6)
    events = [
        RawEvent(0.0, 0.12, [c6, e6], False, "E6", 400.0),
        RawEvent(0.12, 0.36, [d6], False, "D6", 320.0),
        RawEvent(0.36, 0.58, [c6], False, "C6", 300.0),
    ]

    simplified = suppress_leading_descending_overlap(events, tuning)

    assert [note.note_name for note in simplified[0].notes] == ["E6"]


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


def test_should_block_descending_repeated_primary_tertiary_extension_requires_descending_suffix_context() -> None:
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    assert should_block_descending_repeated_primary_tertiary_extension(
        selected=[g4, b4],
        extension=d5,
        segment_duration=0.116,
        previous_primary_was_singleton=True,
        descending_primary_suffix_floor=g4.frequency,
        descending_primary_suffix_ceiling=440.0,
        descending_primary_suffix_note_names={"G4", "A4"},
    ) is True

    assert should_block_descending_repeated_primary_tertiary_extension(
        selected=[g4, b4],
        extension=d5,
        segment_duration=0.116,
        previous_primary_was_singleton=False,
        descending_primary_suffix_floor=g4.frequency,
        descending_primary_suffix_ceiling=440.0,
        descending_primary_suffix_note_names={"G4", "A4"},
    ) is False

    assert should_block_descending_repeated_primary_tertiary_extension(
        selected=[g4, b4],
        extension=d5,
        segment_duration=0.116,
        previous_primary_was_singleton=True,
        descending_primary_suffix_floor=None,
        descending_primary_suffix_ceiling=440.0,
        descending_primary_suffix_note_names={"G4", "A4"},
    ) is False


@manual_capture_slow
def test_transcription_blocks_descending_repeated_primary_tertiary_extension_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    target_segment = next(
        segment
        for segment in payload["debug"]["segmentCandidates"]
        if abs(segment["startTime"] - 7.308) < 0.02 and abs(segment["endTime"] - 7.424) < 0.02
    )
    assert target_segment["selectedNotes"] == ["G4", "B4"]
    assert any(
        decision["noteName"] == "D5"
        and decision["accepted"] is False
        and "descending-repeated-primary-tertiary-blocked" in decision["reasons"]
        for decision in target_segment["secondaryDecisionTrail"]
    )


def test_segment_peaks_suppresses_descending_stale_upper_adjacent_carryover(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    a5 = NoteCandidate(key=16, note_name="A5", frequency=880.0, pitch_class="A", octave=5)
    b5 = NoteCandidate(key=15, note_name="B5", frequency=987.7666025122483, pitch_class="B", octave=5)

    ranked_calls = 0

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        nonlocal ranked_calls
        ranked_calls += 1
        if ranked_calls == 1:
            return [
                NoteHypothesis(a5, 100.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(b5, 40.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
            ]
        return [
            NoteHypothesis(b5, 40.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_energy_gain(_audio, _sample_rate, _start_time, _end_time, frequency):
        if abs(frequency - a5.frequency) < 1e-6:
            return 6.0
        if abs(frequency - b5.frequency) < 1e-6:
            return 0.5
        return 0.0

    monkeypatch.setattr(transcription, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription, "suppress_harmonics", lambda spectrum, frequencies, _frequency: spectrum)
    monkeypatch.setattr(transcription, "onset_energy_gain", fake_onset_energy_gain)

    candidates, debug, primary = segment_peaks(
        synthesize_note(880.0, duration=0.2),
        44100,
        0.0,
        0.2,
        tuning,
        debug=True,
        previous_primary_note_name="B5",
        previous_primary_was_singleton=True,
    )

    assert primary is not None
    assert primary.candidate.note_name == "A5"
    assert [candidate.note_name for candidate in candidates] == ["A5"]
    assert debug is not None
    assert any(
        item["noteName"] == "B5"
        and not item.get("accepted")
        and "descending-adjacent-upper-carryover" in item.get("reasons", [])
        for item in debug["secondaryDecisionTrail"]
    )


def test_segment_peaks_replaces_descending_repeated_stale_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(f4, 100.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
            NoteHypothesis(e4, 60.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_energy_gain(_audio, _sample_rate, _start_time, _end_time, frequency):
        if abs(frequency - f4.frequency) < 1e-6:
            return 0.9
        if abs(frequency - e4.frequency) < 1e-6:
            return 10.0
        return 0.0

    monkeypatch.setattr(transcription, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription, "onset_energy_gain", fake_onset_energy_gain)

    candidates, debug, primary = segment_peaks(
        synthesize_note(f4.frequency, duration=0.3),
        44100,
        0.0,
        0.3,
        tuning,
        debug=True,
        recent_note_names={"F4"},
        previous_primary_note_name="F4",
        previous_primary_frequency=f4.frequency,
        previous_primary_was_singleton=True,
    )

    assert primary is not None
    assert primary.candidate.note_name == "E4"
    assert [candidate.note_name for candidate in candidates][0] == "E4"
    assert debug is not None
    assert debug["primaryPromotion"]["reason"] == "descending-repeated-primary"


def test_segment_peaks_suppresses_descending_restart_upper_carryover(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)

    ranked_calls = 0

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        nonlocal ranked_calls
        ranked_calls += 1
        if ranked_calls == 1:
            return [
                NoteHypothesis(g4, 100.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(e5, 15.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
            ]
        return [
            NoteHypothesis(e5, 15.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_energy_gain(_audio, _sample_rate, _start_time, _end_time, frequency):
        if abs(frequency - g4.frequency) < 1e-6:
            return 60.0
        if abs(frequency - e5.frequency) < 1e-6:
            return 0.8
        return 0.0

    monkeypatch.setattr(transcription, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription, "suppress_harmonics", lambda spectrum, frequencies, _frequency: spectrum)
    monkeypatch.setattr(transcription, "onset_energy_gain", fake_onset_energy_gain)

    candidates, debug, primary = segment_peaks(
        synthesize_note(g4.frequency, duration=0.24),
        44100,
        0.0,
        0.24,
        tuning,
        debug=True,
        previous_primary_note_name="A4",
        previous_primary_frequency=440.0,
        previous_primary_was_singleton=True,
    )

    assert primary is not None
    assert primary.candidate.note_name == "G4"
    assert [candidate.note_name for candidate in candidates] == ["G4"]
    assert debug is not None
    assert any(
        item["noteName"] == "E5"
        and not item.get("accepted")
        and "descending-restart-upper-carryover" in item.get("reasons", [])
        for item in debug["secondaryDecisionTrail"]
    )


@manual_capture_slow
def test_transcription_suppresses_repeated_primary_carryover_in_repeat03_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-c4-to-e6-sequence-17-repeat-03-01")
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
    assert note_sets[17:22] == ["C4", "D4", "E4", "F4", "G4"]
    assert note_sets[31:34] == ["C6", "D6", "E6"]
    assert note_sets[-2:] == ["D6", "E6"]
    assert all("+" not in note_set for note_set in note_sets)




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

@manual_capture_slow
def test_transcription_d4_d5_sequence_fixture_is_pending() -> None:
    """d4-d5-sequence-01 is pending: chair noise masking and G5+B5 misdetection."""
    payload = transcribe_manual_capture_fixture("kalimba-17-c-d4-d5-sequence-01")
    assert len(payload["events"]) >= 1


@manual_capture_slow
def test_transcription_regression_for_e6_to_g4_restart_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-g4-sequence-06-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == ["E6", "C4", "D4", "E4", "F4", "G4"]



@manual_capture_slow
def test_transcription_regression_for_d6_to_e6_sequence_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-d6-to-e6-sequence-10-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == ["D6", "E6", "D6", "E6", "D6", "E6", "D6", "E6", "D6", "E6"]


@manual_capture_slow
def test_transcription_regression_for_e6_to_c6_sequence_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c6-sequence-15-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == ["E6", "D6", "C6", "E6", "D6", "C6", "E6", "D6", "C6", "E6", "D6", "C6", "E6", "D6", "C6"]



@manual_capture_slow
def test_transcription_regression_for_c6_to_e6_sequence_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-c6-to-e6-sequence-15-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == ["C6", "D6", "E6", "C6", "D6", "E6", "C6", "D6", "E6", "C6", "D6", "E6", "C6", "D6", "E6"]
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


def test_suppress_resonant_carryover_keeps_lower_note_when_high_return_is_stale() -> None:
    e6 = NoteCandidate(key=17, note_name="E6", frequency=1318.5102276514797, pitch_class="E", octave=6)
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[e6], is_gliss_like=False, primary_note_name="E6", primary_score=700.0),
        RawEvent(start_time=0.4, end_time=0.78, notes=[c4], is_gliss_like=False, primary_note_name="C4", primary_score=380.0),
        RawEvent(start_time=0.78, end_time=0.98, notes=[c4, e6], is_gliss_like=False, primary_note_name="E6", primary_score=220.0),
        RawEvent(start_time=0.98, end_time=1.36, notes=[d4], is_gliss_like=False, primary_note_name="D4", primary_score=340.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["E6"],
        ["C4"],
        ["C4"],
        ["D4"],
    ]

def test_suppress_resonant_carryover_keeps_repeated_note_for_short_restart_overlap() -> None:
    c5 = NoteCandidate(key=5, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.32, notes=[c5], is_gliss_like=False, primary_note_name="C5", primary_score=320.0),
        RawEvent(start_time=0.32, end_time=0.438, notes=[c5, e4], is_gliss_like=False, primary_note_name="C5", primary_score=180.0),
        RawEvent(start_time=0.438, end_time=0.62, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=260.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C5"],
        ["C5"],
        ["D5"],
    ]


def test_suppress_resonant_carryover_keeps_repeated_note_for_short_post_triad_upper_tail() -> None:
    tuning = next(tuning for tuning in get_default_tunings() if tuning.id == "kalimba-17-c")
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    e5 = NoteCandidate(key=4, note_name="E5", frequency=659.2551138257398, pitch_class="E", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.44, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=420.0),
        RawEvent(start_time=0.44, end_time=0.556, notes=[d5, e5], is_gliss_like=False, primary_note_name="D5", primary_score=210.0),
        RawEvent(start_time=0.556, end_time=0.76, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=260.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events, tuning)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["G4", "B4", "D5"],
        ["D5"],
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


def test_normalize_repeated_four_note_family_merges_local_slide_extension() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.7, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=900.0),
        RawEvent(start_time=0.7, end_time=0.82, notes=[e4], is_gliss_like=True, primary_note_name="E4", primary_score=400.0),
        RawEvent(start_time=1.3, end_time=1.9, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=880.0),
    ]

    normalized = normalize_repeated_four_note_family(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "G4", "B4", "D5"],
        ["G4", "B4", "D5"],
    ]



def test_normalize_repeated_four_note_family_stays_within_local_context_gap() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.7, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="G4", primary_score=900.0),
        RawEvent(start_time=1.6, end_time=1.72, notes=[e4], is_gliss_like=True, primary_note_name="E4", primary_score=400.0),
        RawEvent(start_time=2.2, end_time=2.8, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=880.0),
    ]

    normalized = normalize_repeated_four_note_family(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["G4", "B4", "D5"],
        ["E4"],
        ["G4", "B4", "D5"],
    ]

def test_normalize_repeated_explicit_four_note_patterns_requires_explicit_four_note_anchor() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.26, notes=[e4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=720.0),
        RawEvent(start_time=0.28, end_time=0.72, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=810.0),
        RawEvent(start_time=1.0, end_time=1.44, notes=[e4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=700.0),
        RawEvent(start_time=1.46, end_time=1.9, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="E4", primary_score=790.0),
    ]

    normalized = normalize_repeated_explicit_four_note_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "D5"],
        ["E4", "G4", "B4"],
        ["E4", "D5"],
        ["E4", "G4", "B4"],
    ]


def test_normalize_repeated_explicit_four_note_patterns_keeps_terminal_subset_tail_without_future_anchor() -> None:
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
        ["G4", "D5"],
    ]

def test_normalize_repeated_explicit_four_note_patterns_keeps_off_family_gliss_tail_outside_local_run() -> None:
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
        ["B4", "D5", "F5"],
        ["G4", "D5"],
        ["E4", "D5"],
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


def test_suppress_repeated_triad_blips_keeps_non_identical_anchor_context() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.9, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=1200.0),
        RawEvent(start_time=1.0, end_time=1.2, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="A4", primary_score=500.0),
        RawEvent(start_time=1.4, end_time=2.2, notes=[c4, e4, g4], is_gliss_like=False, primary_note_name="C4", primary_score=1300.0),
    ]

    cleaned = suppress_repeated_triad_blips(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
        ["C4", "E4", "G4"],
    ]


def test_suppress_isolated_triad_extensions_rewrites_local_extension_between_dyad_anchors() -> None:
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    f5 = NoteCandidate(key=14, note_name="F5", frequency=698.4564628660078, pitch_class="F", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.35, notes=[b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=700.0),
        RawEvent(start_time=0.55, end_time=0.9, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=320.0),
        RawEvent(start_time=1.2, end_time=1.85, notes=[b4, d5, f5], is_gliss_like=False, primary_note_name="B4", primary_score=760.0),
        RawEvent(start_time=2.1, end_time=2.8, notes=[b4, d5], is_gliss_like=False, primary_note_name="B4", primary_score=710.0),
        RawEvent(start_time=3.0, end_time=3.7, notes=[b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=720.0),
    ]

    cleaned = suppress_isolated_triad_extensions(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["B4", "D5"],
        ["D5"],
        ["B4", "D5"],
        ["B4", "D5"],
        ["B4", "D5"],
    ]



def test_suppress_isolated_triad_extensions_does_not_rewrite_without_bidirectional_dyad_support() -> None:
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)
    f5 = NoteCandidate(key=14, note_name="F5", frequency=698.4564628660078, pitch_class="F", octave=5)
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    c5 = NoteCandidate(key=15, note_name="C5", frequency=523.2511306011972, pitch_class="C", octave=5)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[c4, g4], is_gliss_like=False, primary_note_name="C4", primary_score=680.0),
        RawEvent(start_time=0.65, end_time=0.95, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=310.0),
        RawEvent(start_time=1.2, end_time=1.85, notes=[b4, d5, f5], is_gliss_like=False, primary_note_name="B4", primary_score=760.0),
        RawEvent(start_time=2.15, end_time=2.6, notes=[e4, c5], is_gliss_like=False, primary_note_name="E4", primary_score=690.0),
    ]

    cleaned = suppress_isolated_triad_extensions(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C4", "G4"],
        ["D5"],
        ["B4", "D5", "F5"],
        ["E4", "C5"],
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


def test_normalize_repeated_triad_patterns_rewrites_terminal_dyad_with_anchor_run() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.7, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=1000.0),
        RawEvent(start_time=0.9, end_time=1.5, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="F4", primary_score=980.0),
        RawEvent(start_time=1.7, end_time=2.3, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="A4", primary_score=1020.0),
        RawEvent(start_time=2.5, end_time=2.9, notes=[d4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=420.0),
    ]

    normalized = normalize_repeated_triad_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
        ["D4", "F4", "A4"],
    ]



def test_normalize_repeated_triad_patterns_does_not_rewrite_without_local_anchor_support() -> None:
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)
    f4 = NoteCandidate(key=7, note_name="F4", frequency=349.2282314330039, pitch_class="F", octave=4)
    a4 = NoteCandidate(key=6, note_name="A4", frequency=440.0, pitch_class="A", octave=4)
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[d4, f4, a4], is_gliss_like=False, primary_note_name="D4", primary_score=980.0),
        RawEvent(start_time=0.8, end_time=1.3, notes=[d4], is_gliss_like=False, primary_note_name="D4", primary_score=180.0),
        RawEvent(start_time=1.5, end_time=2.1, notes=[c4, e4, g4], is_gliss_like=False, primary_note_name="C4", primary_score=990.0),
        RawEvent(start_time=4.0, end_time=4.5, notes=[a4], is_gliss_like=False, primary_note_name="A4", primary_score=160.0),
    ]

    normalized = normalize_repeated_triad_patterns(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["D4", "F4", "A4"],
        ["D4"],
        ["C4", "E4", "G4"],
        ["A4"],
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

@manual_capture_slow
def test_transcription_regression_for_manual_mixed_sequence() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-mixed-sequence-01")
    assert [sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]] == [
        ["C4"],
        ["C5"],
        ["D5"],
        ["C5", "E5"],
        ["G5"],
        ["F5"],
    ]

@manual_capture_slow
def test_transcription_regression_for_manual_four_note_gliss_ascending() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e4-g4-b4-d5-four-note-gliss-ascending-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == [
        "B4+D5+E4+G4",
        "B4+D5+E4+G4",
        "B4+D5+E4+G4",
        "B4+D5+E4+G4",
        "B4+D5+E4+G4",
    ]


@manual_capture_slow
def test_transcription_regression_for_manual_triad_repeat() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e4-g4-b4-triad-repeat-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == [
        "B4+E4+G4",
        "B4+E4+G4",
        "B4+E4+G4",
        "B4+E4+G4",
        "B4+E4+G4",
    ]


@manual_capture_slow
def test_transcription_regression_for_manual_e6_to_c4_sequence_17() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-17-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    assert note_sets == [
        "E6", "D6", "C6", "B5", "A5", "G5", "F5", "E5", "D5",
        "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4",
    ]


@manual_capture_slow
def test_transcription_regression_for_manual_e6_to_c4_sequence_51() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    note_sets = [
        "+".join(sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]))
        for event in payload["events"]
    ]
    expected_cycle = [
        "E6", "D6", "C6", "B5", "A5", "G5", "F5", "E5", "D5",
        "C5", "B4", "A4", "G4", "F4", "E4", "D4", "C4",
    ]
    assert note_sets == [*expected_cycle, *expected_cycle, *expected_cycle]


@manual_capture_slow
def test_transcription_regression_for_manual_triple_glissando() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-triple-glissando-ascending-01")
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
        RawEvent(start_time=1.0, end_time=1.08, notes=[e4], is_gliss_like=True, primary_note_name="E4", primary_score=120.0),
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


def test_normalize_strict_four_note_subsets_keeps_one_off_prefix_without_local_support() -> None:
    e4 = NoteCandidate(key=10, note_name="E4", frequency=329.6275569128699, pitch_class="E", octave=4)
    g4 = NoteCandidate(key=11, note_name="G4", frequency=391.99543598174927, pitch_class="G", octave=4)
    b4 = NoteCandidate(key=12, note_name="B4", frequency=493.8833012561241, pitch_class="B", octave=4)
    d5 = NoteCandidate(key=13, note_name="D5", frequency=587.3295358348151, pitch_class="D", octave=5)

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.42, notes=[e4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=640.0),
        RawEvent(start_time=0.42, end_time=0.84, notes=[e4, g4, b4, d5], is_gliss_like=False, primary_note_name="E4", primary_score=910.0),
        RawEvent(start_time=1.2, end_time=1.75, notes=[e4, g4, b4], is_gliss_like=False, primary_note_name="B4", primary_score=900.0),
    ]

    normalized = normalize_strict_four_note_subsets(raw_events)
    assert [[note.note_name for note in event.notes] for event in normalized] == [
        ["E4", "D5"],
        ["E4", "G4", "B4", "D5"],
        ["E4", "G4", "B4"],
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


@manual_capture_slow
def test_transcription_recovers_terminal_descending_onset_run_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    assert payload["debug"]["terminalMultiOnsetSegments"]
    assert payload["events"][-1]["notes"][0]["pitchClass"] == "C"
    assert payload["events"][-1]["notes"][0]["octave"] == 4
    assert len(payload["events"]) >= 50


@manual_capture_slow
def test_transcription_drops_descending_restart_bridge_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    note_sets = {"+".join(note["pitchClass"] + str(note["octave"]) for note in event["notes"]) for event in payload["events"]}
    assert "D4+E4" not in note_sets


@manual_capture_slow
def test_transcription_eliminates_second_cycle_g4_a4_merged_dyad_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    merged_note_sets = ["+".join(event["notes"]) for event in payload["debug"]["mergedEvents"]]
    assert "G4+A4" not in merged_note_sets


@manual_capture_slow
def test_transcription_eliminates_first_cycle_f4_g4_merged_dyad_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    merged_note_sets = ["+".join(event["notes"]) for event in payload["debug"]["mergedEvents"]]
    assert "F4+G4" not in merged_note_sets


@manual_capture_slow
def test_transcription_eliminates_b5_d6_restart_residue_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    merged_note_sets = ["+".join(event["notes"]) for event in payload["debug"]["mergedEvents"]]
    assert "B5+D6" not in merged_note_sets


@manual_capture_slow
def test_transcription_reaches_exact_event_count_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    assert len(payload["events"]) == 51


@manual_capture_slow
def test_transcription_recovers_third_cycle_prefix_in_51_note_fixture() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-e6-to-c4-sequence-51-01")
    note_sets = ["+".join("{}{}".format(note["pitchClass"], note["octave"]) for note in event["notes"]) for event in payload["events"]]
    assert note_sets[33:37] == ["C4", "E6", "D6", "C6"]
    assert [round(start, 4) for start, _ in payload["debug"]["shortBridgeActiveRanges"]] == [20.6013]





@manual_capture_slow
def test_bwv147_restart_prefix_recovers_scoped_opening_phrase() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-restart-prefix-01")
def test_bwv147_restart_high_register_collapses_short_repeated_overlap() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-restart-high-register-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets == ["C4+E5", "C5", "D5", "C5+E5", "G5"]
    assert "E4" not in note_sets


@manual_capture_slow
def test_bwv147_mid_cluster_rebundles_short_upper_tail_into_triad() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-mid-gesture-cluster-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets[:3] == ["A4+F5", "B4", "C5+E5"]
    assert set(note_sets[3].split("+")) == {"B4", "D5", "G4"}
    assert "E5" not in note_sets


@manual_capture_slow
def test_bwv147_upper_cluster_recovers_delayed_terminal_e5() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-upper-mixed-cluster-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets == ["B5", "A4+C6", "G5", "E5", "A4+C5", "D5", "E5"]
    assert [round(start, 4) for start, _ in payload["debug"]["delayedTerminalOrphanSegments"]] == [5.416]



@manual_capture_slow
def test_bwv147_restart_tail_promotes_recent_upper_octave_alias() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-restart-tail-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets == ["G5", "E4+G5", "C6", "B5", "A4+C6"]
    assert payload["debug"]["segmentCandidates"][-1]["primaryPromotion"]["reason"] == "recent-upper-octave-alias-primary"
    trail = payload["debug"]["segmentCandidates"][-1]["secondaryDecisionTrail"]
    assert trail[0]["reasons"] == ["recent-upper-octave-alias-secondary-blocked"]
    assert trail[1]["noteName"] == "A4" and trail[1]["accepted"] is True


@manual_capture_slow
def test_bwv147_late_upper_tail_recovers_sparse_terminal_d5_e5_tail() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-late-upper-tail-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets[-5:] == ["G5", "E5", "A4+C5", "D5", "E5"]
    assert [tuple(round(value, 4) for value in segment) for segment in payload["debug"]["terminalTwoOnsetTailSegments"]] == [
        (5.9147, 6.2347),
        (7.0, 7.32),
    ]


@manual_capture_slow
def test_bwv147_lower_mixed_roll_recovers_opening_mixed_dyad_and_long_gap_run() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-lower-mixed-roll-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets == ["C5", "D4+G4+B4", "C5", "D5", "D4+G4", "B4", "D5", "G4+B4+D5+F5", "E5"]
    opening_segment = next(segment for segment in payload["debug"]["segmentCandidates"] if abs(segment["startTime"] - 1.0667) < 0.02)
    assert opening_segment["selectedNotes"] == ["D4", "B4", "G4"]
    assert any(decision["noteName"] == "G4" and decision["accepted"] and decision["reasons"] == ["lower-mixed-roll-extension"] for decision in opening_segment["secondaryDecisionTrail"])


@manual_capture_slow
def test_bwv147_lower_context_roll_matches_completed_nine_event_phrase() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-lower-context-roll-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets == ["C5", "D4+G4+B4", "C5", "D5", "D4+G4", "B4", "D5", "G4+B4+D5+F5", "E5"]


@manual_capture_slow
def test_bwv147_lower_f4_mixed_run_is_a_clean_pending_child_with_late_tail_miss() -> None:
    payload = transcribe_manual_capture_fixture("kalimba-17-c-bwv147-lower-f4-mixed-run-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert len(note_sets) == 4
    assert set(note_sets[0].split("+")) == {"C5", "G4"}
    assert note_sets[1:3] == ["D5", "E5"]
    assert set(note_sets[3].split("+")) == {"A4", "F4"}
    assert [round(onset, 4) for onset in payload["debug"]["onsetTimes"][-4:]] == [5.848, 6.9813, 9.568, 9.76]
    assert [tuple(round(value, 4) for value in segment) for segment in payload["debug"]["segments"]] == [
        (0.0107, 0.9653),
        (1.0667, 1.72),
        (2.072, 2.8613),
        (4.0747, 4.7067),
    ]


