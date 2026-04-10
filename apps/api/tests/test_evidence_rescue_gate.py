"""Direct unit tests for _evidence_rescue_gate.

Tests the rescue dispatch logic using constructed NoteHypothesis +
a simple mock for _NoteEvidenceCache.  Each test targets a specific
rescue path or rejection condition.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

from app.transcription import (
    Note,
    NoteCandidate,
    NoteHypothesis,
)
from app.transcription.peaks import _evidence_rescue_gate, _CandidateDecision
from app.transcription.constants import (
    RESCUE_CARRYOVER_MAX_AS_RATIO,
    RESCUE_CARRYOVER_MIN_FUNDAMENTAL_RATIO,
    RESCUE_CARRYOVER_MIN_ONSET_GAIN,
    RESCUE_CARRYOVER_MIN_SCORE_RATIO,
    RESCUE_MIN_BACKWARD_GAIN,
    RESCUE_SCORE_FR_OVERRIDE_MIN_FR,
    RESCUE_SCORE_FR_OVERRIDE_MIN_SCORE_RATIO,
    RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO,
    RESCUE_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO,
)


def _hyp(
    note_name: str,
    key: int,
    score: float = 100.0,
    fundamental_ratio: float = 0.95,
) -> NoteHypothesis:
    return NoteHypothesis(
        candidate=NoteCandidate(key, Note.from_name(note_name)),
        score=score,
        fundamental_energy=0.0,
        overtone_energy=0.0,
        fundamental_ratio=fundamental_ratio,
        subharmonic_alias_energy=0.0,
        octave_alias_energy=0.0,
        octave_alias_ratio=0.0,
        octave_alias_penalty=0.0,
    )


def _decision(reasons: list[str]) -> _CandidateDecision:
    return _CandidateDecision(
        note_name="X",
        frequency=440.0,
        score=100.0,
        fundamental_ratio=0.95,
        onset_gain=None,
        accepted=False,
        reasons=list(reasons),
        octave_dyad_allowed=False,
        source="test",
    )


def _mock_evidence(
    *,
    backward_gain: float = 0.0,
    onset_gain: float = 0.0,
    attack_to_sustain_ratio: float = 1.0,
) -> MagicMock:
    """Build a mock _NoteEvidenceCache returning fixed values."""
    ev = MagicMock()
    ev.backward_attack_gain.return_value = backward_gain
    ev.onset_gain.return_value = onset_gain
    ev.attack_to_sustain_ratio.return_value = attack_to_sustain_ratio
    return ev


# ══ recent-carryover-candidate path ══════════════════════════════

class TestCarryoverRescue:

    def test_rescued_by_high_backward_gain(self) -> None:
        """High backward_gain alone rescues recent-carryover."""
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            _hyp("F#4", 12),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result == "evidence-rescue-recent-carryover"

    def test_rescued_by_onset_gain_and_score_and_fr(self) -> None:
        """Low backward + high onset + score + fR rescues."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp(
            "F#4", 12,
            score=200.0 * RESCUE_CARRYOVER_MIN_SCORE_RATIO + 1,
            fundamental_ratio=RESCUE_CARRYOVER_MIN_FUNDAMENTAL_RATIO + 0.01,
        )
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result == "evidence-rescue-recent-carryover"

    def test_blocked_when_onset_gain_too_low(self) -> None:
        """Low backward + low onset → not rescued."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp("F#4", 12, score=100.0, fundamental_ratio=0.95)
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN - 0.1,
            ),
        )
        assert result is None

    def test_blocked_when_fr_too_low(self) -> None:
        """Onset OK but fR below threshold → not rescued (alias guard)."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp(
            "E4", 10,
            score=100.0,
            fundamental_ratio=RESCUE_CARRYOVER_MIN_FUNDAMENTAL_RATIO - 0.01,
        )
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result is None

    def test_blocked_when_score_ratio_too_low(self) -> None:
        """Onset + fR OK but score too low vs primary → not rescued."""
        primary = _hyp("D4", 11, score=1000.0)
        hypothesis = _hyp(
            "F#4", 12,
            score=1000.0 * RESCUE_CARRYOVER_MIN_SCORE_RATIO * 0.5,
            fundamental_ratio=0.95,
        )
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result is None

    def test_blocked_by_high_as_ratio(self) -> None:
        """Broadband transient (high AS ratio) blocks carryover rescue."""
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            _hyp("D4", 11, score=200.0),
            _hyp("B4", 5, score=300.0),
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN + 5,
                attack_to_sustain_ratio=RESCUE_CARRYOVER_MAX_AS_RATIO + 0.1,
            ),
        )
        assert result is None

    def test_as_ratio_below_threshold_allows_rescue(self) -> None:
        """AS ratio just below cap allows normal backward-gain rescue."""
        result = _evidence_rescue_gate(
            _decision(["recent-carryover-candidate"]),
            _hyp("D4", 11),
            _hyp("B4", 5, score=200.0),
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1,
                attack_to_sustain_ratio=RESCUE_CARRYOVER_MAX_AS_RATIO - 0.1,
            ),
        )
        assert result == "evidence-rescue-recent-carryover"


# ══ tertiary score+FR override path ═══════════════════════════════

class TestTertiaryScoreOverride:

    def test_tertiary_weak_onset_rescued(self) -> None:
        """tertiary-weak-onset with high onset + score + fR → rescued."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp(
            "G4", 9,
            score=200.0 * RESCUE_SCORE_FR_OVERRIDE_MIN_SCORE_RATIO + 1,
            fundamental_ratio=RESCUE_SCORE_FR_OVERRIDE_MIN_FR + 0.01,
        )
        result = _evidence_rescue_gate(
            _decision(["tertiary-weak-onset"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result == "evidence-rescue-tertiary-score-override"

    def test_tertiary_weak_backward_rescued(self) -> None:
        """tertiary-weak-backward-attack also rescued by same path."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp(
            "G4", 9,
            score=200.0 * RESCUE_SCORE_FR_OVERRIDE_MIN_SCORE_RATIO + 1,
            fundamental_ratio=RESCUE_SCORE_FR_OVERRIDE_MIN_FR + 0.01,
        )
        result = _evidence_rescue_gate(
            _decision(["tertiary-weak-backward-attack"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result == "evidence-rescue-tertiary-score-override"

    def test_tertiary_not_rescued_when_fr_too_low(self) -> None:
        """Tertiary override blocked when fR below threshold."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp(
            "G4", 9,
            score=100.0,
            fundamental_ratio=RESCUE_SCORE_FR_OVERRIDE_MIN_FR - 0.01,
        )
        result = _evidence_rescue_gate(
            _decision(["tertiary-weak-onset"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result is None

    def test_non_tertiary_reason_not_rescued_by_override(self) -> None:
        """Score+FR override only fires for tertiary-weak-* reasons."""
        primary = _hyp("D4", 11, score=200.0)
        hypothesis = _hyp(
            "G4", 9,
            score=100.0,
            fundamental_ratio=RESCUE_SCORE_FR_OVERRIDE_MIN_FR + 0.01,
        )
        result = _evidence_rescue_gate(
            _decision(["score-below-threshold"]),
            hypothesis,
            primary,
            _mock_evidence(
                backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1,
                onset_gain=RESCUE_CARRYOVER_MIN_ONSET_GAIN + 0.1,
            ),
        )
        assert result is None


# ══ weak-lower-secondary path ═════════════════════════════════════

class TestWeakLowerSecondaryRescue:

    def test_rescued_with_high_fr(self) -> None:
        result = _evidence_rescue_gate(
            _decision(["weak-lower-secondary"]),
            _hyp("G3", 9, fundamental_ratio=RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO + 0.01),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result == "evidence-rescue-weak-lower-secondary"

    def test_not_rescued_with_low_fr(self) -> None:
        result = _evidence_rescue_gate(
            _decision(["weak-lower-secondary"]),
            _hyp("G3", 9, fundamental_ratio=RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO - 0.01),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result is None

    def test_not_rescued_with_low_backward_gain(self) -> None:
        """Even high fR doesn't help if backward_gain is too low."""
        result = _evidence_rescue_gate(
            _decision(["weak-lower-secondary"]),
            _hyp("G3", 9, fundamental_ratio=0.95),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN - 1),
        )
        assert result is None


# ══ weak-secondary-onset path ═════════════════════════════════════

class TestWeakSecondaryOnsetRescue:

    def test_rescued_with_high_fr(self) -> None:
        result = _evidence_rescue_gate(
            _decision(["weak-secondary-onset"]),
            _hyp("A4", 13, fundamental_ratio=RESCUE_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO + 0.01),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result == "evidence-rescue-weak-secondary-onset"

    def test_not_rescued_with_low_fr(self) -> None:
        result = _evidence_rescue_gate(
            _decision(["weak-secondary-onset"]),
            _hyp("A4", 13, fundamental_ratio=RESCUE_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO - 0.01),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result is None


# ══ Priority: weak-lower-secondary checked before weak-secondary-onset ══

class TestReasonPriority:

    def test_weak_lower_takes_priority_when_both_present(self) -> None:
        """When both reasons present, weak-lower-secondary is checked first."""
        result = _evidence_rescue_gate(
            _decision(["weak-lower-secondary", "weak-secondary-onset"]),
            _hyp("G3", 9, fundamental_ratio=RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO + 0.01),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result == "evidence-rescue-weak-lower-secondary"

    def test_weak_lower_blocks_when_fr_too_low_even_if_onset_would_pass(self) -> None:
        """If weak-lower-secondary check fails (low fR), returns None
        even if weak-secondary-onset would have passed."""
        result = _evidence_rescue_gate(
            _decision(["weak-lower-secondary", "weak-secondary-onset"]),
            _hyp("G3", 9, fundamental_ratio=RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO - 0.01),
            _hyp("D4", 11, score=200.0),
            _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 1),
        )
        assert result is None


# ══ Unrecognized reasons → no rescue ═════════════════════════════

def test_unrecognized_reason_returns_none() -> None:
    """Reasons not handled by any rescue path → None."""
    result = _evidence_rescue_gate(
        _decision(["some-unknown-gate"]),
        _hyp("A4", 13),
        _hyp("D4", 11, score=200.0),
        _mock_evidence(backward_gain=RESCUE_MIN_BACKWARD_GAIN + 10),
    )
    assert result is None
