"""Mechanism tests for #166 onset-strong sibling primary rescue.

When the top-ranked candidate has weak attack evidence (low onset_gain) and
a diluted fundamentalRatio, a top-K sibling with clean fR and strong attack
should be promoted to primary.
"""

from unittest.mock import MagicMock

import pytest

from app.transcription.models import Note, NoteCandidate, NoteHypothesis
from app.transcription.peaks import maybe_promote_onset_strong_sibling


def _hyp(name: str, score: float, fR: float, key: int = 1) -> NoteHypothesis:
    candidate = NoteCandidate(key=key, note=Note.from_name(name), score=score)
    return NoteHypothesis(
        candidate=candidate,
        score=score,
        fundamental_energy=100.0,
        overtone_energy=50.0,
        fundamental_ratio=fR,
        subharmonic_alias_energy=0.0,
        octave_alias_energy=0.0,
        octave_alias_ratio=0.0,
        octave_alias_penalty=0.0,
    )


def _make_evidence(gains_by_freq: dict[float, float]) -> MagicMock:
    evidence = MagicMock()
    def lookup(freq: float) -> float:
        for f, g in gains_by_freq.items():
            if abs(f - freq) < 1.0:
                return g
        return 0.0
    evidence.onset_gain.side_effect = lookup
    return evidence


class TestOnsetStrongSiblingRescue:
    """Narrow rescue targeting the #166 E100 B3/B4 pattern."""

    def test_e100_b3_b4_pattern_promotes_sibling(self) -> None:
        """The canonical #166 case: B3 (fR=0.599, gain=1.8) loses primary
        to B4 (fR=0.979, gain=62) despite B3 having the higher raw score."""
        b3 = _hyp("B3", 270.7, 0.599)
        b4 = _hyp("B4", 229.4, 0.979)
        d4 = _hyp("D4", 214.0, 0.974)
        a4 = _hyp("A4", 168.7, 0.996)
        ranked = [b3, b4, d4, a4]
        evidence = _make_evidence({
            b3.candidate.frequency: 1.8,
            b4.candidate.frequency: 62.0,
            d4.candidate.frequency: 12.5,
            a4.candidate.frequency: 1.1,
        })

        _r = maybe_promote_onset_strong_sibling(
            b3, 1.8, ranked, evidence,
        )
        new_primary, new_gain, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug

        assert new_primary.candidate.note_name == "B4"
        assert new_gain == 62.0
        assert debug is not None
        assert debug["reason"] == "onset-strong-sibling"
        assert debug["replacedPrimaryNote"] == "B3"
        assert debug["replacementNote"] == "B4"

    def test_fresh_primary_with_no_cached_gain_is_evaluated(self) -> None:
        """When primary_onset_gain is None (e.g. primary not in
        recent_note_names), the rescue should compute it via evidence
        and still evaluate."""
        b3 = _hyp("B3", 270.7, 0.599)
        b4 = _hyp("B4", 229.4, 0.979)
        ranked = [b3, b4]
        evidence = _make_evidence({
            b3.candidate.frequency: 1.8,
            b4.candidate.frequency: 62.0,
        })

        _r = maybe_promote_onset_strong_sibling(
            b3, None, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "B4"
        assert debug is not None

    def test_clean_primary_is_preserved(self) -> None:
        """When the primary has high fR, no rescue should fire."""
        c4 = _hyp("C4", 300.0, 0.95)
        e4 = _hyp("E4", 250.0, 0.98)
        ranked = [c4, e4]
        evidence = _make_evidence({
            c4.candidate.frequency: 2.0,
            e4.candidate.frequency: 80.0,
        })

        _r = maybe_promote_onset_strong_sibling(
            c4, 2.0, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "C4"
        assert debug is None

    def test_strong_primary_is_preserved(self) -> None:
        """When the primary has strong onset_gain, no rescue should fire."""
        c4 = _hyp("C4", 300.0, 0.60)
        e4 = _hyp("E4", 250.0, 0.98)
        ranked = [c4, e4]
        evidence = _make_evidence({
            c4.candidate.frequency: 15.0,   # >= 5.0 threshold
            e4.candidate.frequency: 80.0,
        })

        _r = maybe_promote_onset_strong_sibling(
            c4, 15.0, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "C4"
        assert debug is None

    def test_sibling_with_weak_gain_does_not_promote(self) -> None:
        """Sibling must have onset_gain >= 20 AND >= 10x primary gain."""
        c4 = _hyp("C4", 300.0, 0.60)
        e4 = _hyp("E4", 250.0, 0.98)
        ranked = [c4, e4]
        evidence = _make_evidence({
            c4.candidate.frequency: 2.0,
            e4.candidate.frequency: 15.0,   # < 20.0 min
        })

        _r = maybe_promote_onset_strong_sibling(
            c4, 2.0, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "C4"
        assert debug is None

    def test_sibling_with_low_fr_does_not_promote(self) -> None:
        """Sibling must have fR >= 0.85."""
        c4 = _hyp("C4", 300.0, 0.60)
        e4 = _hyp("E4", 250.0, 0.80)   # < 0.85 min
        ranked = [c4, e4]
        evidence = _make_evidence({
            c4.candidate.frequency: 2.0,
            e4.candidate.frequency: 80.0,
        })

        _r = maybe_promote_onset_strong_sibling(
            c4, 2.0, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "C4"
        assert debug is None

    def test_sibling_with_very_low_score_does_not_promote(self) -> None:
        """Sibling score must be >= 0.70 of primary score."""
        c4 = _hyp("C4", 300.0, 0.60)
        e4 = _hyp("E4", 180.0, 0.98)   # 180/300 = 0.60 < 0.70
        ranked = [c4, e4]
        evidence = _make_evidence({
            c4.candidate.frequency: 2.0,
            e4.candidate.frequency: 80.0,
        })

        _r = maybe_promote_onset_strong_sibling(
            c4, 2.0, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "C4"
        assert debug is None

    def test_picks_highest_gain_among_eligible_siblings(self) -> None:
        """When multiple siblings qualify, the one with the highest gain wins."""
        c4 = _hyp("C4", 300.0, 0.60)
        e4 = _hyp("E4", 250.0, 0.90)
        g4 = _hyp("G4", 240.0, 0.95)
        ranked = [c4, e4, g4]
        evidence = _make_evidence({
            c4.candidate.frequency: 2.0,
            e4.candidate.frequency: 25.0,
            g4.candidate.frequency: 50.0,   # highest
        })

        _r = maybe_promote_onset_strong_sibling(
            c4, 2.0, ranked, evidence,
        )
        new_primary, _, debug = _r.primary, _r.primary_onset_gain, _r.promotion_debug
        assert new_primary.candidate.note_name == "G4"
