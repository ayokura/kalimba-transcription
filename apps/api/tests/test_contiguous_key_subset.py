"""Direct unit tests for _extract_contiguous_key_subset.

Tests contiguous-key subset extraction using constructed NoteCandidate
objects.  The function strips noise notes from a merged note set by
finding the best contiguous-key window.
"""

from app.transcription import Note, NoteCandidate
from app.transcription.events import _extract_contiguous_key_subset


def _nc(note_name: str, key: int) -> NoteCandidate:
    return NoteCandidate(key, Note.from_name(note_name))


# ── Basic contiguous run extraction ──────────────────────────────

def test_extracts_4_from_5_contiguous() -> None:
    """E133: union {C4(7), D4(11), F#4(12), A4(13), C5(14)} → best {11-14}."""
    merged = [_nc("C4", 7), _nc("D4", 11), _nc("F#4", 12), _nc("A4", 13), _nc("C5", 14)]
    longer = [_nc("D4", 11), _nc("F#4", 12), _nc("A4", 13), _nc("C5", 14)]
    result = _extract_contiguous_key_subset(merged, max_count=4, longer_notes=longer)
    assert result is not None
    keys = sorted(n.key for n in result)
    assert keys == [11, 12, 13, 14]


def test_extracts_3_from_5_with_gap() -> None:
    """E82: union {A3(8), G3(9), D4(11), F#4(12), A4(13)} → best {11-13}."""
    merged = [_nc("A3", 8), _nc("G3", 9), _nc("D4", 11), _nc("F#4", 12), _nc("A4", 13)]
    longer = [_nc("D4", 11), _nc("F#4", 12), _nc("A4", 13)]
    result = _extract_contiguous_key_subset(merged, max_count=4, longer_notes=longer)
    assert result is not None
    keys = sorted(n.key for n in result)
    assert keys == [11, 12, 13]


# ── Prefers longer_notes coverage ────────────────────────────────

def test_prefers_window_with_more_longer_coverage() -> None:
    """Two valid runs: one has more longer_notes overlap."""
    merged = [_nc("A3", 8), _nc("G3", 9), _nc("B3", 10),
              _nc("D4", 11), _nc("F#4", 12), _nc("A4", 13)]
    longer = [_nc("D4", 11), _nc("F#4", 12), _nc("A4", 13)]
    result = _extract_contiguous_key_subset(merged, max_count=3, longer_notes=longer)
    assert result is not None
    keys = sorted(n.key for n in result)
    assert keys == [11, 12, 13]


# ── Returns None when no run meets min_run_length ────────────────

def test_returns_none_when_no_contiguous_run() -> None:
    """All notes are isolated (no 2 adjacent keys)."""
    merged = [_nc("C4", 7), _nc("G3", 9), _nc("D4", 11), _nc("A4", 13), _nc("C5", 15)]
    longer = [_nc("G3", 9), _nc("D4", 11)]
    result = _extract_contiguous_key_subset(merged, max_count=4, longer_notes=longer)
    assert result is None


def test_returns_none_when_runs_too_short() -> None:
    """Runs of length 2 exist but min_run_length=3 by default."""
    merged = [_nc("G3", 9), _nc("B3", 10), _nc("F#4", 12), _nc("A4", 13)]
    longer = [_nc("G3", 9), _nc("B3", 10)]
    result = _extract_contiguous_key_subset(merged, max_count=4, longer_notes=longer)
    assert result is None


# ── min_run_length override ──────────────────────────────────────

def test_custom_min_run_length_2() -> None:
    """With min_run_length=2, dyads are valid subsets."""
    merged = [_nc("G3", 9), _nc("B3", 10), _nc("F#4", 12), _nc("A4", 13)]
    longer = [_nc("F#4", 12), _nc("A4", 13)]
    result = _extract_contiguous_key_subset(
        merged, max_count=4, longer_notes=longer, min_run_length=2,
    )
    assert result is not None
    keys = sorted(n.key for n in result)
    assert keys == [12, 13]


# ── max_count limits window size ─────────────────────────────────

def test_max_count_caps_window() -> None:
    """Long contiguous run capped to max_count."""
    merged = [_nc("B3", 10), _nc("D4", 11), _nc("F#4", 12), _nc("A4", 13), _nc("C5", 14)]
    longer = [_nc("D4", 11), _nc("F#4", 12), _nc("A4", 13), _nc("C5", 14)]
    result = _extract_contiguous_key_subset(merged, max_count=3, longer_notes=longer)
    assert result is not None
    assert len(result) == 3
    keys = sorted(n.key for n in result)
    # Multiple 3-windows have coverage=3: {11,12,13} and {12,13,14}
    # First found wins (iteration starts from largest then sweeps left-to-right)
    assert all(k in range(10, 15) for k in keys)
    # Verify the window is actually contiguous
    assert keys[-1] - keys[0] == 2


# ── Tie-breaking: larger window wins ─────────────────────────────

def test_larger_window_preferred_on_coverage_tie() -> None:
    """When coverage is equal, larger window wins."""
    # Keys 10,11,12,13 all contiguous; longer has 11,12
    merged = [_nc("B3", 10), _nc("D4", 11), _nc("F#4", 12), _nc("A4", 13)]
    longer = [_nc("D4", 11), _nc("F#4", 12)]
    result = _extract_contiguous_key_subset(merged, max_count=4, longer_notes=longer)
    assert result is not None
    # 4-window has coverage=2, 3-windows {10,11,12} or {11,12,13} also coverage=2
    # score = (coverage=2, size=4) beats (coverage=2, size=3)
    assert len(result) == 4


# ── Single contiguous run exactly at max_count ───────────────────

def test_exact_max_count_returned_as_is() -> None:
    """When the contiguous run is exactly max_count, return all."""
    merged = [_nc("D4", 11), _nc("F#4", 12), _nc("A4", 13), _nc("C5", 14)]
    longer = [_nc("D4", 11), _nc("F#4", 12), _nc("A4", 13), _nc("C5", 14)]
    result = _extract_contiguous_key_subset(merged, max_count=4, longer_notes=longer)
    assert result is not None
    assert len(result) == 4
