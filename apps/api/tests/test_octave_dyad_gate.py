"""Direct unit tests for allow_octave_secondary.

Tests the octave dyad acceptance gate using constructed NoteHypothesis
objects — no synthesized audio or full pipeline needed.  Each test
targets a specific branch/condition in the function.
"""

import pytest

from app.transcription import (
    Note,
    NoteCandidate,
    NoteHypothesis,
)
from app.transcription.peaks import allow_octave_secondary
from app.transcription.constants import (
    OCTAVE_DYAD_LOWER_OCTAVE4_MIN_FUNDAMENTAL_RATIO,
    OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO,
    OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO,
    OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO,
    OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO,
)


def _hyp(
    note_name: str,
    key: int,
    score: float = 100.0,
    fundamental_ratio: float = 0.95,
    fundamental_energy: float = 100.0,
    second_harmonic_energy: float = 0.0,
) -> NoteHypothesis:
    """Convenience builder for NoteHypothesis with sensible defaults."""
    return NoteHypothesis(
        candidate=NoteCandidate(key, Note.from_name(note_name)),
        score=score,
        fundamental_energy=fundamental_energy,
        overtone_energy=0.0,
        fundamental_ratio=fundamental_ratio,
        subharmonic_alias_energy=0.0,
        octave_alias_energy=0.0,
        octave_alias_ratio=0.0,
        octave_alias_penalty=0.0,
        second_harmonic_energy=second_harmonic_energy,
    )


# ── No harmonic relation → returns False ──────────────────────────

def test_non_harmonic_pair_returns_false() -> None:
    primary = _hyp("D4", 11)
    hypothesis = _hyp("A4", 13)
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Non-octave harmonic (3rd, 4th) → returns False ───────────────

def test_third_harmonic_relation_rejected() -> None:
    # A4 (440) and E6 (1318.5) are ~3:1 ratio
    primary = _hyp("A4", 13, fundamental_ratio=0.95)
    hypothesis = _hyp("E6", 4, fundamental_ratio=0.95)
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Upper octave: basic acceptance ────────────────────────────────

def test_upper_octave_accepted_with_high_fr() -> None:
    """D4 primary → D5 secondary with high fR should be accepted."""
    primary = _hyp("D4", 11, second_harmonic_energy=50.0)
    hypothesis = _hyp("D5", 4, fundamental_ratio=0.98, fundamental_energy=100.0)
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is True


# ── Upper octave: low fR → rejected ──────────────────────────────

def test_upper_octave_rejected_with_low_fr() -> None:
    """Upper octave with fR below threshold should be rejected."""
    primary = _hyp("D4", 11, second_harmonic_energy=50.0)
    hypothesis = _hyp(
        "D5", 4,
        fundamental_ratio=OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO - 0.01,
        fundamental_energy=100.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


def test_upper_octave_at_exact_fr_threshold_accepted() -> None:
    """Upper octave with fR exactly at threshold should pass."""
    primary = _hyp("D4", 11, second_harmonic_energy=0.0)
    hypothesis = _hyp(
        "D5", 4,
        fundamental_ratio=OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO,
        fundamental_energy=100.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is True


# ── Upper octave: harmonic energy check ───────────────────────────

def test_upper_octave_rejected_when_energy_below_harmonic_ratio() -> None:
    """Upper octave rejected when its energy is tiny vs primary's 2nd harmonic."""
    primary = _hyp("D4", 11, second_harmonic_energy=1000.0)
    hypothesis = _hyp(
        "D5", 4,
        fundamental_ratio=0.98,
        fundamental_energy=1000.0 * OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO * 0.5,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


def test_upper_octave_accepted_when_energy_above_harmonic_ratio() -> None:
    """Upper octave accepted when its energy exceeds the harmonic energy ratio."""
    primary = _hyp("D4", 11, second_harmonic_energy=1000.0)
    hypothesis = _hyp(
        "D5", 4,
        fundamental_ratio=0.98,
        fundamental_energy=1000.0 * OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO * 2.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is True


def test_upper_octave_bypasses_energy_check_when_no_second_harmonic() -> None:
    """When primary has zero second_harmonic_energy, energy check is skipped."""
    primary = _hyp("D4", 11, second_harmonic_energy=0.0)
    hypothesis = _hyp("D5", 4, fundamental_ratio=0.98, fundamental_energy=1.0)
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is True


# ── Upper octave: existing octave > 4 → rejected ─────────────────

def test_upper_octave_rejected_when_existing_above_octave4() -> None:
    """Upper octave of an octave-5+ note is rejected (too high)."""
    primary = _hyp("D5", 4)
    hypothesis = _hyp("D6", 1, fundamental_ratio=0.98)
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Lower octave: basic acceptance ────────────────────────────────

def test_lower_octave_accepted_octave5() -> None:
    """D5 primary → D4 secondary (octave 4) with decent fR should be accepted."""
    primary = _hyp("D5", 4, fundamental_energy=100.0)
    hypothesis = _hyp(
        "D4", 11,
        fundamental_ratio=OCTAVE_DYAD_LOWER_OCTAVE4_MIN_FUNDAMENTAL_RATIO + 0.01,
        fundamental_energy=100.0 * OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO * 2.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is True


# ── Lower octave: octave ≤ 3 → rejected ──────────────────────────

def test_lower_octave_rejected_at_octave3() -> None:
    """Lower octave in octave 3 or below is rejected."""
    primary = _hyp("D4", 11, fundamental_energy=100.0)
    hypothesis = _hyp("D3", 20, fundamental_ratio=0.95, fundamental_energy=50.0)
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Lower octave: octave 4 fR gate ───────────────────────────────

def test_lower_octave4_rejected_with_low_fr() -> None:
    """Lower octave at octave 4 rejected when fR below stricter threshold."""
    primary = _hyp("D5", 4, fundamental_energy=100.0)
    hypothesis = _hyp(
        "D4", 11,
        fundamental_ratio=OCTAVE_DYAD_LOWER_OCTAVE4_MIN_FUNDAMENTAL_RATIO - 0.01,
        fundamental_energy=50.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Lower octave: octave 5+ fR gate (more relaxed) ───────────────

def test_lower_octave5_accepted_with_moderate_fr() -> None:
    """Lower octave at octave 5 uses the more relaxed fR threshold."""
    primary = _hyp("D6", 1, fundamental_energy=100.0)
    hypothesis = _hyp(
        "D5", 4,
        fundamental_ratio=OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO + 0.01,
        fundamental_energy=100.0 * OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO * 2.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is True


def test_lower_octave5_rejected_with_very_low_fr() -> None:
    """Lower octave at octave 5 rejected when fR below relaxed threshold."""
    primary = _hyp("D6", 1, fundamental_energy=100.0)
    hypothesis = _hyp(
        "D5", 4,
        fundamental_ratio=OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO - 0.01,
        fundamental_energy=50.0,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Lower octave: energy ratio gate ──────────────────────────────

def test_lower_octave_rejected_with_insufficient_energy() -> None:
    """Lower octave rejected when its energy is too small vs primary."""
    primary = _hyp("D5", 4, fundamental_energy=1000.0)
    hypothesis = _hyp(
        "D4", 11,
        fundamental_ratio=0.80,
        fundamental_energy=1000.0 * OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO * 0.5,
    )
    selected = [primary.candidate]
    assert allow_octave_secondary(primary, hypothesis, selected) is False


# ── Empty selected list → returns False ───────────────────────────

def test_empty_selected_returns_false() -> None:
    """With no existing selected notes, no harmonic relation can exist."""
    primary = _hyp("D4", 11)
    hypothesis = _hyp("D5", 4)
    assert allow_octave_secondary(primary, hypothesis, []) is False
