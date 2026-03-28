"""Exploratory full-clip integration tests for pending/review_needed fixtures.

These tests cover fixtures where expected.json assertions are not yet
strong enough for the parameterized regression test. They should be
converted to scoped expected.json assertions and deleted when the fixture
stabilizes. They intentionally bypass evaluationWindows/ignoredRanges to
capture whole-clip behavior that is still under review.
"""
from conftest import manual_capture_slow, transcribe_manual_capture_fixture_full_audio


@manual_capture_slow
def test_bwv147_mid_cluster_rebundles_short_upper_tail_into_triad() -> None:
    payload = transcribe_manual_capture_fixture_full_audio("kalimba-17-c-bwv147-mid-gesture-cluster-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert note_sets[:3] == ["A4+F5", "B4", "C5+E5"]
    assert set(note_sets[3].split("+")) == {"B4", "D5", "G4"}
    assert "E5" not in note_sets

@manual_capture_slow
def test_bwv147_lower_f4_mixed_run_is_a_clean_pending_child_with_late_tail_miss() -> None:
    payload = transcribe_manual_capture_fixture_full_audio("kalimba-17-c-bwv147-lower-f4-mixed-run-01")
    note_sets = ["+".join(f"{note['pitchClass']}{note['octave']}" for note in event["notes"]) for event in payload["events"]]
    assert len(note_sets) == 4
    assert set(note_sets[0].split("+")) == {"C5", "G4"}
    assert note_sets[1:3] == ["D5", "E5"]
    assert set(note_sets[3].split("+")) == {"A4", "F4"}
