import numpy as np
import pytest

from app.tunings import get_default_tunings
from app.transcription import (
    NoteCandidate,
    NoteHypothesis,
    PRIMARY_REJECTION_MAX_SCORE,
    PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO,
    segment_peaks,
)
from conftest import synthesize_note, synthesize_chord


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

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda spectrum, frequencies, _frequency: spectrum)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

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

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

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

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda spectrum, frequencies, _frequency: spectrum)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

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


def test_segment_peaks_rejects_weak_primary_with_low_score_and_fundamental_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary with both low score and low fundamental ratio is rejected."""
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    e6 = NoteCandidate(key=0, note_name="E6", frequency=1318.5102276514797, pitch_class="E", octave=6)

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(e6, PRIMARY_REJECTION_MAX_SCORE - 1, 0.0, 0.0,
                           PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO - 0.1, 0.0, 0.0, 0.0, 0.0),
        ]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)

    candidates, debug, primary = segment_peaks(
        synthesize_note(e6.frequency, duration=0.2),
        44100,
        0.0,
        0.2,
        tuning,
        debug=True,
    )

    assert candidates == []
    assert primary is None
    assert debug is None


def test_segment_peaks_keeps_low_score_primary_with_high_fundamental_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low score alone does not trigger rejection when fundamental ratio is adequate."""
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    c4 = NoteCandidate(key=9, note_name="C4", frequency=261.6255653005986, pitch_class="C", octave=4)

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(c4, PRIMARY_REJECTION_MAX_SCORE - 1, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 0.0),
        ]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)

    candidates, debug, primary = segment_peaks(
        synthesize_note(c4.frequency, duration=0.2),
        44100,
        0.0,
        0.2,
        tuning,
        debug=True,
    )

    assert primary is not None
    assert primary.candidate.note_name == "C4"
    assert len(candidates) >= 1


def test_segment_peaks_keeps_high_score_primary_with_low_fundamental_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low fundamental ratio alone does not trigger rejection when score is adequate."""
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    d4 = NoteCandidate(key=8, note_name="D4", frequency=293.6647679174076, pitch_class="D", octave=4)

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(d4, 50.0, 0.0, 0.0,
                           PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO - 0.1, 0.0, 0.0, 0.0, 0.0),
        ]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)

    candidates, debug, primary = segment_peaks(
        synthesize_note(d4.frequency, duration=0.2),
        44100,
        0.0,
        0.2,
        tuning,
        debug=True,
    )

    assert primary is not None
    assert primary.candidate.note_name == "D4"
    assert len(candidates) >= 1
