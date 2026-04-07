import numpy as np
import pytest

from app.tunings import get_default_tunings
from app.transcription import (
    Note,
    NoteCandidate,
    NoteHypothesis,
    PRIMARY_REJECTION_MAX_SCORE,
    PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO,
    maybe_promote_lower_secondary_to_recent_upper_octave,
    segment_peaks,
    settings,
)
from conftest import synthesize_note, synthesize_chord


def test_segment_peaks_prefers_fundamental_for_strong_harmonics() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_note(587.3295, harmonics=(1.0, 0.9, 0.7))
    candidates, debug, _, _trace = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    assert candidates
    assert candidates[0].note_name == "D5"
    assert debug is not None

def test_segment_peaks_detects_d5_and_a5_chord() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_chord((587.3295, 880.0))
    candidates, debug, _, _trace = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
    note_names = [candidate.note_name for candidate in candidates]
    assert "D5" in note_names
    assert "A5" in note_names
    assert debug is not None
    assert debug["residualCandidates"]

def test_segment_peaks_allows_true_octave_dyad() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_chord((587.3295, 1174.6591))
    candidates, debug, _, _trace = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
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
    candidates, debug, _, _trace = segment_peaks(audio, 44100, 0.0, len(audio) / 44100, tuning, debug=True)
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

    candidates, debug, primary, _trace = segment_peaks(
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

    candidates, debug, primary, _trace = segment_peaks(
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


def test_maybe_promote_lower_secondary_to_recent_upper_octave_has_reachable_interval_window() -> None:
    primary = NoteHypothesis(
        NoteCandidate(14, Note.from_name("F5")),
        100.0,
        0.0,
        0.0,
        0.98,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    accepted_secondary = NoteHypothesis(
        NoteCandidate(10, Note.from_name("E4")),
        10.0,
        0.0,
        0.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    residual_ranked = [
        NoteHypothesis(
            NoteCandidate(4, Note.from_name("E5")),
            20.0,
            0.0,
            0.0,
            0.95,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    ]

    promoted, promoted_from = maybe_promote_lower_secondary_to_recent_upper_octave(
        primary,
        accepted_secondary,
        residual_ranked,
        segment_duration=0.2,
    )

    assert promoted.candidate.note_name == "E5"
    assert promoted_from == "E4"


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

    candidates, debug, _, _trace = segment_peaks(
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

    candidates, debug, primary, _trace = segment_peaks(
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
    a5 = NoteCandidate(key=16, note=Note.from_name("A5"))
    b5 = NoteCandidate(key=15, note=Note.from_name("B5"))

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

    def fake_onset_energy_gain(_audio, _sample_rate, _start_time, _end_time, frequency, **_kwargs):
        if abs(frequency - a5.frequency) < 1e-6:
            return 6.0
        if abs(frequency - b5.frequency) < 1e-6:
            return 0.5
        return 0.0

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda spectrum, frequencies, _frequency: spectrum)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

    candidates, debug, primary, _trace = segment_peaks(
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
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(f4, 100.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
            NoteHypothesis(e4, 60.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_energy_gain(_audio, _sample_rate, _start_time, _end_time, frequency, **_kwargs):
        if abs(frequency - f4.frequency) < 1e-6:
            return 0.9
        if abs(frequency - e4.frequency) < 1e-6:
            return 10.0
        return 0.0

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

    candidates, debug, primary, _trace = segment_peaks(
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
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))

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

    def fake_onset_energy_gain(_audio, _sample_rate, _start_time, _end_time, frequency, **_kwargs):
        if abs(frequency - g4.frequency) < 1e-6:
            return 60.0
        if abs(frequency - e5.frequency) < 1e-6:
            return 0.8
        return 0.0

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda spectrum, frequencies, _frequency: spectrum)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

    candidates, debug, primary, _trace = segment_peaks(
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
    e6 = NoteCandidate(key=0, note=Note.from_name("E6"))

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(e6, PRIMARY_REJECTION_MAX_SCORE - 1, 0.0, 0.0,
                           PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO - 0.1, 0.0, 0.0, 0.0, 0.0),
        ]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)

    candidates, debug, primary, _trace = segment_peaks(
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


def test_rejected_primary_rescued_by_alternative_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When primary is rejected, an alternative primary can rescue the segment."""
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    # Primary: E6 — will be rejected (low score + low FR)
    e6 = NoteCandidate(key=0, note=Note.from_name("E6"))
    # Alternative: D5 — strong candidate, not harmonically related to E6
    d5 = NoteCandidate(key=3, note=Note.from_name("D5"))

    ranked_calls = 0

    def fake_rank(frequencies, spectrum, _tuning, debug=False):
        nonlocal ranked_calls
        ranked_calls += 1
        if ranked_calls == 1:
            # Initial ranking: E6 top (will be rejected), D5 second
            return [
                NoteHypothesis(e6, PRIMARY_REJECTION_MAX_SCORE - 1, 0.0, 0.0,
                               PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO - 0.1, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(d5, 200.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
            ]
        # Residual ranking (after suppressing alternative primary D5)
        return [
            NoteHypothesis(e6, 5.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_energy_gain(_audio, _sr, _start, _end, frequency, **_kwargs):
        if abs(frequency - d5.frequency) < 1:
            return 15.0  # Strong onset for D5
        return 0.5

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda s, f, _freq: s)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

    candidates, _debug, result_primary, trace = segment_peaks(
        synthesize_note(d5.frequency, duration=0.3),
        44100, 0.0, 0.3, tuning, debug=True,
    )

    assert len(candidates) > 0, "alternative branch should rescue the segment"
    assert result_primary is not None
    assert result_primary.candidate.note_name == "D5"
    assert trace is not None
    assert "multi-primary-rescue" in trace.primary.promotions


def test_rejected_primary_not_rescued_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With use_multi_primary_branching=False, rejected primary stays rejected."""
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    e6 = NoteCandidate(key=0, note=Note.from_name("E6"))
    d5 = NoteCandidate(key=3, note=Note.from_name("D5"))

    ranked_calls = 0

    def fake_rank(frequencies, spectrum, _tuning, debug=False):
        nonlocal ranked_calls
        ranked_calls += 1
        if ranked_calls == 1:
            return [
                NoteHypothesis(e6, PRIMARY_REJECTION_MAX_SCORE - 1, 0.0, 0.0,
                               PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO - 0.1, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(d5, 200.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
            ]
        return [
            NoteHypothesis(e6, 5.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_energy_gain(_audio, _sr, _start, _end, frequency, **_kwargs):
        if abs(frequency - d5.frequency) < 1:
            return 15.0
        return 0.5

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda s, f, _freq: s)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset_energy_gain)

    with settings.override(use_multi_primary_branching=False):
        candidates, _debug, primary, _trace = segment_peaks(
            synthesize_note(d5.frequency, duration=0.3),
            44100, 0.0, 0.3, tuning, debug=True,
        )

    assert candidates == [], "should remain rejected when flag is off"
    assert primary is None


def test_segment_peaks_keeps_low_score_primary_with_high_fundamental_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low score alone does not trigger rejection when fundamental ratio is adequate."""
    import app.transcription as transcription

    tuning = get_default_tunings()[0]
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(c4, PRIMARY_REJECTION_MAX_SCORE - 1, 0.0, 0.0,
                           1.0, 0.0, 0.0, 0.0, 0.0),
        ]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)

    candidates, debug, primary, _trace = segment_peaks(
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
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))

    def fake_rank_tuning_candidates(_frequencies, _spectrum, _tuning, debug=False):
        return [
            NoteHypothesis(d4, 50.0, 0.0, 0.0,
                           PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO - 0.1, 0.0, 0.0, 0.0, 0.0),
        ]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank_tuning_candidates)

    candidates, debug, primary, _trace = segment_peaks(
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


# --- Iterative harmonic suppression mechanism tests ---


def _build_iterative_suppression_fakes():
    """Build fake functions for iterative suppression tests.

    Scenario: primary=C5, secondary=A4, tertiary candidate=C4 (octave of C5).
    C4 is rejected in the 1st pass for harmonic-related but should be
    recovered by the 2nd pass with iterative suppression.
    """
    c5 = NoteCandidate(key=5, note=Note.from_name("C5"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))

    rank_calls = 0

    def fake_rank(frequencies, spectrum, tuning, debug=False):
        nonlocal rank_calls
        rank_calls += 1
        if rank_calls == 1:
            # Original ranking: C5 dominant
            return [
                NoteHypothesis(c5, 800.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(a4, 200.0, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(c4, 150.0, 0.0, 0.0, 0.40, 0.0, 0.0, 0.0, 0.0),
            ]
        if rank_calls == 2:
            # Residual after C5 suppression: A4 dominates, C4 has low FR
            return [
                NoteHypothesis(a4, 180.0, 0.0, 0.0, 0.98, 0.0, 0.0, 0.0, 0.0),
                NoteHypothesis(c4, 120.0, 0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0),
            ]
        # Iterative residual (C5+A4 suppressed): C4 fundamental stands out
        return [
            NoteHypothesis(c4, 110.0, 0.0, 0.0, 0.80, 0.0, 0.0, 0.0, 0.0),
        ]

    def fake_onset_gain(_audio, _sr, _start, _end, freq, **_kwargs):
        # High onset_gain so iterative suppression's tertiary gates pass via
        # the onset side (TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE).
        return 25.0

    def fake_backward_gain(_audio, _sr, _start, freq):
        # Low backward_gain (< TERTIARY_MIN_BACKWARD_ATTACK_GAIN=20) so the
        # #152 harmonic-related bypass does not trigger; this scenario must
        # validate iterative suppression as the only recovery path for C4.
        return 5.0

    return c4, a4, c5, fake_rank, fake_onset_gain, fake_backward_gain


def test_iterative_suppression_recovers_octave_tertiary(monkeypatch: pytest.MonkeyPatch) -> None:
    """2nd pass recovers C4 in C5+A4+C4 chord (C4 is octave of C5)."""
    import app.transcription as transcription

    c4, a4, c5, fake_rank, fake_onset, fake_backward = _build_iterative_suppression_fakes()
    tuning = get_default_tunings()[0]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda s, f, _: s)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset)
    monkeypatch.setattr(transcription.peaks, "onset_backward_attack_gain", fake_backward)

    from app.transcription import settings
    with settings.override(use_iterative_harmonic_suppression=True):
        candidates, debug, primary, _trace = segment_peaks(
            synthesize_note(523.2511, duration=0.4), 44100, 0.0, 0.4, tuning, debug=True,
        )

    note_names = {c.note_name for c in candidates}
    assert "C5" in note_names, f"expected C5 in {note_names}"
    assert "A4" in note_names, f"expected A4 in {note_names}"
    assert "C4" in note_names, f"expected C4 in {note_names} (iterative suppression recovery)"

    trail = debug["secondaryDecisionTrail"]
    iter_accepted = [
        e for e in trail
        if e["accepted"] and "iterative-suppression-tertiary" in e.get("reasons", [])
    ]
    assert len(iter_accepted) == 1
    assert iter_accepted[0]["noteName"] == "C4"


def test_iterative_suppression_ablation_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With USE_ITERATIVE_HARMONIC_SUPPRESSION=False, 2nd pass does not run."""
    import app.transcription as transcription

    c4, a4, c5, fake_rank, fake_onset, fake_backward = _build_iterative_suppression_fakes()
    tuning = get_default_tunings()[0]

    monkeypatch.setattr(transcription.peaks, "rank_tuning_candidates", fake_rank)
    monkeypatch.setattr(transcription.peaks, "suppress_harmonics", lambda s, f, _: s)
    monkeypatch.setattr(transcription.peaks, "onset_energy_gain", fake_onset)
    monkeypatch.setattr(transcription.peaks, "onset_backward_attack_gain", fake_backward)

    from app.transcription import settings
    with settings.override(use_iterative_harmonic_suppression=False):
        candidates, debug, primary, _trace = segment_peaks(
            synthesize_note(523.2511, duration=0.4), 44100, 0.0, 0.4, tuning, debug=True,
        )

    note_names = {c.note_name for c in candidates}
    assert "C5" in note_names
    assert "A4" in note_names
    assert "C4" not in note_names, "C4 should NOT be recovered with flag off"

    trail = debug["secondaryDecisionTrail"]
    iter_entries = [
        e for e in trail
        if any("iterative" in r for r in e.get("reasons", []))
    ]
    assert len(iter_entries) == 0, "no iterative trail entries when flag is off"
