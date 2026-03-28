import numpy as np
import pytest

from app.transcription import (
    GapAttackCandidates,
    OnsetAttackProfile,
    collect_attack_validated_gap_candidates,
    collect_multi_onset_gap_segments,
    collect_terminal_multi_onset_segments,
    collect_two_onset_terminal_tail_segments,
    detect_segments,
)
from conftest import synthesize_note


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

    assert [[round(value, 4) for value in gap] for gap in candidates.inter_ranges] == [[3.504, 5.0613, 6.3013, 7.5467, 8.9467, 9.0667]]
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
