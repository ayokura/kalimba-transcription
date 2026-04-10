"""Mechanism tests for #151 alternateGroupings (dissonance-aware merge guard).

When use_alternate_groupings is enabled, merge passes should suppress merges
that would create dissonant intervals (minor/major 2nd) and record the
hypothetical merge as an alternateGrouping on the event.
"""

import pytest

from app.transcription.events import (
    has_dissonant_interval,
    merge_short_chord_clusters,
    merge_short_gliss_clusters,
    min_semitone_distance,
)
from app.transcription.models import Note, NoteCandidate, RawEvent
from app.transcription.settings import override


def _nc(name: str, key: int = 1) -> NoteCandidate:
    return NoteCandidate(key=key, note=Note.from_name(name))


# ---------------------------------------------------------------------------
# min_semitone_distance / has_dissonant_interval
# ---------------------------------------------------------------------------


class TestDissonanceHelper:
    def test_minor_second(self) -> None:
        assert min_semitone_distance([_nc("B4"), _nc("C5")]) == 1

    def test_major_second(self) -> None:
        assert min_semitone_distance([_nc("C4"), _nc("D4")]) == 2

    def test_minor_third(self) -> None:
        assert min_semitone_distance([_nc("C4"), _nc("Eb4")]) == 3

    def test_major_third_not_dissonant(self) -> None:
        assert not has_dissonant_interval([_nc("C4"), _nc("E4")])

    def test_triad_with_minor_second(self) -> None:
        # B4 + C5 + E5 contains B4-C5 = minor 2nd
        assert has_dissonant_interval([_nc("B4"), _nc("C5"), _nc("E5")])

    def test_consonant_triad(self) -> None:
        # C4 + E4 + G4 — all intervals >= minor 3rd
        assert not has_dissonant_interval([_nc("C4"), _nc("E4"), _nc("G4")])

    def test_single_note(self) -> None:
        assert not has_dissonant_interval([_nc("C4")])

    def test_empty(self) -> None:
        assert not has_dissonant_interval([])


# ---------------------------------------------------------------------------
# merge_short_chord_clusters with dissonance guard
# ---------------------------------------------------------------------------


class TestChordClusterDissonanceGuard:
    def test_dissonant_merge_suppressed_when_flag_on(self) -> None:
        """B4 + [C5, E5] would create B4-C5 minor 2nd — merge suppressed.
        Keys must be contiguous (1,2,3) for merge eligibility."""
        events = [
            RawEvent(0.0, 0.12, [_nc("B4", key=1)], False, "B4", 300.0),
            RawEvent(0.13, 0.30, [_nc("C5", key=2), _nc("E5", key=3)], False, "C5", 280.0),
        ]
        with override(use_alternate_groupings=True):
            result = merge_short_chord_clusters(events)

        # Should NOT merge (kept as 2 events)
        assert len(result) == 2
        # First event should have an alternate grouping recorded
        assert len(result[0].alternate_groupings) == 1
        alt = result[0].alternate_groupings[0]
        assert alt.reason == "dissonant_merge_suppressed"
        assert {n.note_name for n in alt.combined_notes} == {"B4", "C5", "E5"}

    def test_consonant_merge_proceeds_when_flag_on(self) -> None:
        """C4 + [E4, G4] — consonant triad, merge should proceed."""
        events = [
            RawEvent(0.0, 0.12, [_nc("C4", key=1)], False, "C4", 300.0),
            RawEvent(0.13, 0.30, [_nc("E4", key=2), _nc("G4", key=3)], False, "E4", 280.0),
        ]
        with override(use_alternate_groupings=True):
            result = merge_short_chord_clusters(events)

        # Should merge (1 event with 3 notes)
        assert len(result) == 1
        assert {n.note_name for n in result[0].notes} == {"C4", "E4", "G4"}

    def test_dissonant_merge_proceeds_when_flag_off(self) -> None:
        """Without the flag, dissonant merges proceed as before."""
        events = [
            RawEvent(0.0, 0.12, [_nc("B4", key=1)], False, "B4", 300.0),
            RawEvent(0.13, 0.30, [_nc("C5", key=2), _nc("E5", key=3)], False, "C5", 280.0),
        ]
        # Default: use_alternate_groupings=False
        result = merge_short_chord_clusters(events)

        # Should merge (original behavior)
        assert len(result) == 1
        assert {n.note_name for n in result[0].notes} == {"B4", "C5", "E5"}


# ---------------------------------------------------------------------------
# merge_short_gliss_clusters with dissonance guard
# ---------------------------------------------------------------------------


class TestGlissClusterDissonanceGuard:
    def test_dissonant_gliss_merge_suppressed(self) -> None:
        """Adjacent gliss-like events with dissonant combined result.
        Keys must be contiguous for merge eligibility."""
        events = [
            RawEvent(0.0, 0.15, [_nc("B4", key=1)], True, "B4", 300.0),
            RawEvent(0.16, 0.30, [_nc("C5", key=2), _nc("E5", key=3)], True, "C5", 280.0),
        ]
        with override(use_alternate_groupings=True):
            result = merge_short_gliss_clusters(events)

        assert len(result) == 2
        assert len(result[0].alternate_groupings) == 1

    def test_consonant_gliss_merge_proceeds(self) -> None:
        """Consonant gliss cluster should still merge."""
        events = [
            RawEvent(0.0, 0.15, [_nc("E4", key=1)], True, "E4", 300.0),
            RawEvent(0.16, 0.30, [_nc("G4", key=2), _nc("B4", key=3)], True, "G4", 280.0),
        ]
        with override(use_alternate_groupings=True):
            result = merge_short_gliss_clusters(events)

        assert len(result) == 1
        assert {n.note_name for n in result[0].notes} == {"E4", "G4", "B4"}
