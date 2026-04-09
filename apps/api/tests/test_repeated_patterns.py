import pytest

from app.transcription import (
    REPEATED_PATTERN_PASS_IDS,
    Note,
    NoteCandidate,
    RawEvent,
    apply_repeated_pattern_passes,
    normalize_repeated_explicit_four_note_patterns,
    normalize_repeated_four_note_family,
    normalize_repeated_triad_patterns,
    normalize_strict_four_note_subsets,
)


def test_normalize_repeated_four_note_family_merges_local_slide_extension() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))
    f5 = NoteCandidate(key=14, note=Note.from_name("F5"))

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


def test_normalize_repeated_triad_patterns_expands_dominant_subsets() -> None:
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))

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
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))

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


def test_normalize_repeated_explicit_four_note_patterns_merges_leading_subset_into_dominant_take() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

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


def test_repeated_pattern_pass_ids_are_unique() -> None:
    assert len(REPEATED_PATTERN_PASS_IDS) == len(set(REPEATED_PATTERN_PASS_IDS))


def test_apply_repeated_pattern_passes_can_disable_triad_normalizer() -> None:
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))

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
