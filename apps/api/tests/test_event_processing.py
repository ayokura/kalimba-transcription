from app.tunings import get_default_tunings
from app.transcription import (
    Note,
    NoteCandidate,
    NoteHypothesis,
    RawEvent,
    build_recent_ascending_primary_run_ceiling,
    build_recent_note_names,
    classify_event_gesture,
    collapse_ascending_restart_lower_residue_singletons,
    collapse_late_descending_step_handoffs,
    collapse_same_start_primary_singletons,
    is_slide_playable_contiguous_cluster,
    merge_four_note_gliss_clusters,
    merge_short_chord_clusters,
    merge_short_gliss_clusters,
    select_contiguous_four_note_cluster,
    should_block_descending_repeated_primary_tertiary_extension,
    should_keep_dense_trailing_onset,
    simplify_descending_adjacent_dyad_residue,
    simplify_short_gliss_prefix_to_contiguous_singleton,
    simplify_short_secondary_bleed,
    suppress_descending_terminal_residual_cluster,
    suppress_descending_upper_return_overlap,
    suppress_leading_descending_overlap,
    suppress_leading_gliss_neighbor_noise,
    suppress_leading_gliss_subset_transients,
    suppress_resonant_carryover,
    suppress_short_residual_tails,
    suppress_subset_decay_events,
)


def test_is_slide_playable_contiguous_cluster_accepts_center_triads() -> None:
    tuning = get_default_tunings()[0]
    notes = [next(note for note in tuning.notes if note.note_name == name) for name in ["C4", "E4", "G4"]]

    assert is_slide_playable_contiguous_cluster(notes, tuning) is True


def test_is_slide_playable_contiguous_cluster_rejects_non_slide_center_crossing_cluster() -> None:
    tuning = get_default_tunings()[0]
    notes = [next(note for note in tuning.notes if note.note_name == name) for name in ["C4", "D4", "F4"]]

    assert is_slide_playable_contiguous_cluster(notes, tuning) is False


def test_suppress_descending_terminal_residual_cluster_drops_rebound_tail() -> None:
    tuning = get_default_tunings()[0]
    note_by_name = {note.note_name: note for note in tuning.notes}
    raw_events = [
        RawEvent(0.0, 0.2, [note_by_name["F4"]], False, "F4", 100.0),
        RawEvent(0.2, 0.4, [note_by_name["E4"]], False, "E4", 100.0),
        RawEvent(0.4, 0.6, [note_by_name["D4"]], False, "D4", 100.0),
        RawEvent(0.6, 0.8, [note_by_name["C4"]], False, "C4", 100.0),
        RawEvent(0.8, 1.0, [note_by_name["D4"], note_by_name["F4"]], False, "D4", 100.0),
    ]

    cleaned = suppress_descending_terminal_residual_cluster(raw_events, tuning)

    assert cleaned == raw_events[:-1]


def test_suppress_descending_terminal_residual_cluster_keeps_non_rebound_tail() -> None:
    tuning = get_default_tunings()[0]
    note_by_name = {note.note_name: note for note in tuning.notes}
    raw_events = [
        RawEvent(0.0, 0.2, [note_by_name["F4"]], False, "F4", 100.0),
        RawEvent(0.2, 0.4, [note_by_name["E4"]], False, "E4", 100.0),
        RawEvent(0.4, 0.6, [note_by_name["D4"]], False, "D4", 100.0),
        RawEvent(0.6, 0.8, [note_by_name["C4"]], False, "C4", 100.0),
        RawEvent(0.8, 1.0, [note_by_name["C4"], note_by_name["E4"], note_by_name["G4"]], True, "C4", 100.0),
    ]

    cleaned = suppress_descending_terminal_residual_cluster(raw_events, tuning)

    assert cleaned == raw_events


def test_select_contiguous_four_note_cluster_promotes_dense_local_family() -> None:
    ranked = [
        NoteHypothesis(NoteCandidate(13, Note.from_name('D5')), 1000.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(12, Note.from_name('B4')), 800.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(11, Note.from_name('G4')), 600.0, 0.0, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(10, Note.from_name('E4')), 180.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(9, Note.from_name('D4')), 100.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0),
    ]

    selected = select_contiguous_four_note_cluster(ranked[0], ranked, 0.7)

    assert selected is not None
    assert [candidate.note_name for candidate in selected] == ['E4', 'G4', 'B4', 'D5']


def test_select_contiguous_four_note_cluster_rejects_short_or_ambiguous_window() -> None:
    ranked = [
        NoteHypothesis(NoteCandidate(13, Note.from_name('D5')), 1000.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(12, Note.from_name('B4')), 800.0, 0.0, 0.0, 0.99, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(11, Note.from_name('G4')), 600.0, 0.0, 0.0, 0.97, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(10, Note.from_name('E4')), 120.0, 0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 0.0),
        NoteHypothesis(NoteCandidate(9, Note.from_name('D4')), 110.0, 0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 0.0),
    ]

    assert select_contiguous_four_note_cluster(ranked[0], ranked, 0.4) is None
    assert select_contiguous_four_note_cluster(ranked[0], ranked, 0.7) is None


def test_should_keep_dense_trailing_onset_preserves_penultimate_dense_onset() -> None:
    boundary_times = [6.1813, 6.4267, 6.5733, 6.96]

    assert should_keep_dense_trailing_onset(boundary_times, 2, 3.996, 7.424) is True
    assert should_keep_dense_trailing_onset(boundary_times, 1, 3.996, 7.424) is False


def test_simplify_short_secondary_bleed_strips_restart_stale_upper_note() -> None:
    c4 = NoteCandidate(9, Note.from_name("C4"))
    d4 = NoteCandidate(8, Note.from_name("D4"))
    e6 = NoteCandidate(17, Note.from_name("E6"))
    events = [
        RawEvent(0.0, 0.18, [e6], False, "E6", 400.0),
        RawEvent(0.18, 0.38, [c4, e6], False, "C4", 350.0),
        RawEvent(0.38, 0.62, [d4], False, "D4", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["C4"]


def test_simplify_short_secondary_bleed_collapses_mirrored_adjacent_run_to_upper_note() -> None:
    d4 = NoteCandidate(8, Note.from_name("D4"))
    e4 = NoteCandidate(10, Note.from_name("E4"))
    f4 = NoteCandidate(7, Note.from_name("F4"))
    events = [
        RawEvent(0.0, 0.24, [d4], False, "D4", 340.0),
        RawEvent(0.24, 0.42, [d4, e4], False, "D4", 300.0),
        RawEvent(0.42, 0.6, [d4, e4], False, "D4", 295.0),
        RawEvent(0.6, 0.86, [f4], False, "F4", 360.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["E4"]
    assert [note.note_name for note in simplified[2].notes] == ["E4"]



def test_simplify_short_secondary_bleed_strips_descending_stale_upper_step() -> None:
    g5 = NoteCandidate(3, Note.from_name("G5"))
    f5 = NoteCandidate(14, Note.from_name("F5"))
    e5 = NoteCandidate(4, Note.from_name("E5"))
    events = [
        RawEvent(0.0, 0.22, [g5], False, "G5", 340.0),
        RawEvent(0.22, 0.4, [f5, g5], False, "F5", 300.0),
        RawEvent(0.4, 0.62, [e5], False, "E5", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["F5"]


def test_simplify_short_secondary_bleed_promotes_descending_lower_step() -> None:
    c5 = NoteCandidate(5, Note.from_name("C5"))
    b4 = NoteCandidate(12, Note.from_name("B4"))
    a4 = NoteCandidate(6, Note.from_name("A4"))
    events = [
        RawEvent(0.0, 0.22, [c5], False, "C5", 340.0),
        RawEvent(0.22, 0.4, [b4, c5], False, "C5", 300.0),
        RawEvent(0.4, 0.62, [a4], False, "A4", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["B4"]


def test_simplify_short_secondary_bleed_collapses_descending_upper_residue_to_primary() -> None:
    g4 = NoteCandidate(11, Note.from_name("G4"))
    b4 = NoteCandidate(12, Note.from_name("B4"))
    f4 = NoteCandidate(7, Note.from_name("F4"))
    events = [
        RawEvent(0.0, 0.18, [g4], False, "G4", 320.0),
        RawEvent(0.18, 0.29, [g4, b4], False, "G4", 300.0),
        RawEvent(0.29, 0.5, [f4], False, "F4", 290.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["G4"]


def test_simplify_short_secondary_bleed_collapses_repeated_descending_handoff_to_primary() -> None:
    c5 = NoteCandidate(14, Note.from_name("C5"))
    b4 = NoteCandidate(12, Note.from_name("B4"))
    a4 = NoteCandidate(10, Note.from_name("A4"))
    events = [
        RawEvent(0.0, 0.18, [c5], False, "C5", 320.0),
        RawEvent(0.18, 0.31, [b4, c5], False, "B4", 300.0),
        RawEvent(0.31, 0.43, [b4, c5], False, "B4", 290.0),
        RawEvent(0.43, 0.62, [a4], False, "A4", 280.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["B4"]
    assert [note.note_name for note in simplified[2].notes] == ["B4"]


def test_simplify_descending_adjacent_dyad_residue_collapses_upper_residue_to_lower() -> None:
    a4 = NoteCandidate(10, Note.from_name("A4"))
    g4 = NoteCandidate(11, Note.from_name("G4"))
    f4 = NoteCandidate(7, Note.from_name("F4"))
    events = [
        RawEvent(0.0, 0.18, [a4], False, "A4", 320.0),
        RawEvent(0.18, 0.31, [g4, a4], False, "G4", 300.0),
        RawEvent(0.31, 0.5, [f4], False, "F4", 290.0),
    ]

    simplified = simplify_descending_adjacent_dyad_residue(events)

    assert [note.note_name for note in simplified[1].notes] == ["G4"]


def test_collapse_ascending_restart_lower_residue_singletons_keeps_previous_step() -> None:
    tuning = get_default_tunings()[0]
    d6 = NoteCandidate(15, Note.from_name("D6"))
    b5 = NoteCandidate(13, Note.from_name("B5"))
    e6 = NoteCandidate(16, Note.from_name("E6"))
    events = [
        RawEvent(0.0, 0.28, [d6], False, "D6", 320.0),
        RawEvent(0.28, 0.40, [b5], False, "B5", 260.0),
        RawEvent(0.40, 0.72, [e6], False, "E6", 340.0),
    ]

    cleaned = collapse_ascending_restart_lower_residue_singletons(events, tuning)

    assert [note.note_name for note in cleaned[1].notes] == ["D6"]


def test_simplify_short_secondary_bleed_keeps_primary_for_restart_upper_sandwich() -> None:
    b5 = NoteCandidate(2, Note.from_name("B5"))
    d6 = NoteCandidate(1, Note.from_name("D6"))
    e6 = NoteCandidate(17, Note.from_name("E6"))
    events = [
        RawEvent(0.0, 0.18, [e6], False, "E6", 320.0),
        RawEvent(0.18, 0.3, [b5, d6], False, "D6", 300.0),
        RawEvent(0.3, 0.54, [e6], False, "E6", 310.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["D6"]


def test_simplify_short_secondary_bleed_keeps_primary_for_restart_followed_by_upper_extension() -> None:
    b5 = NoteCandidate(2, Note.from_name("B5"))
    d6 = NoteCandidate(1, Note.from_name("D6"))
    e6 = NoteCandidate(17, Note.from_name("E6"))
    c6 = NoteCandidate(16, Note.from_name("C6"))
    events = [
        RawEvent(0.0, 0.18, [e6], False, "E6", 320.0),
        RawEvent(0.18, 0.3, [b5, d6], False, "D6", 300.0),
        RawEvent(0.3, 0.52, [d6, e6], False, "D6", 310.0),
        RawEvent(0.52, 0.74, [c6], False, "C6", 290.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["D6"]


def test_collapse_late_descending_step_handoffs_keeps_lower_note() -> None:
    a4 = NoteCandidate(13, Note.from_name("A4"))
    g4 = NoteCandidate(11, Note.from_name("G4"))
    f4 = NoteCandidate(7, Note.from_name("F4"))
    events = [
        RawEvent(0.0, 0.22, [a4], False, "A4", 320.0),
        RawEvent(0.22, 0.34, [g4, a4], False, "G4", 280.0),
        RawEvent(0.34, 0.58, [f4], False, "F4", 300.0),
    ]

    cleaned = collapse_late_descending_step_handoffs(events)

    assert [note.note_name for note in cleaned[1].notes] == ["G4"]


def test_collapse_late_descending_step_handoffs_keeps_lower_note_when_middle_is_short_gliss_like() -> None:
    a4 = NoteCandidate(13, Note.from_name("A4"))
    g4 = NoteCandidate(11, Note.from_name("G4"))
    f4 = NoteCandidate(7, Note.from_name("F4"))
    events = [
        RawEvent(0.0, 0.22, [a4], False, "A4", 320.0),
        RawEvent(0.22, 0.34, [g4, a4], True, "G4", 280.0),
        RawEvent(0.34, 0.58, [f4], False, "F4", 300.0),
    ]

    cleaned = collapse_late_descending_step_handoffs(events)

    assert [note.note_name for note in cleaned[1].notes] == ["G4"]


def test_simplify_short_secondary_bleed_collapses_descending_bridge_to_upper() -> None:
    c6 = NoteCandidate(16, Note.from_name("C6"))
    b5 = NoteCandidate(2, Note.from_name("B5"))
    a5 = NoteCandidate(13, Note.from_name("A5"))
    g5 = NoteCandidate(3, Note.from_name("G5"))
    events = [
        RawEvent(0.0, 0.22, [c6], False, "C6", 340.0),
        RawEvent(0.22, 0.4, [b5, g5], False, "B5", 300.0),
        RawEvent(0.4, 0.62, [a5], False, "A5", 320.0),
    ]

    simplified = simplify_short_secondary_bleed(events)

    assert [note.note_name for note in simplified[1].notes] == ["B5"]


def test_suppress_resonant_carryover_keeps_lower_note_in_descending_adjacent_chain() -> None:
    f4 = NoteCandidate(7, Note.from_name("F4"))
    g4 = NoteCandidate(11, Note.from_name("G4"))
    e4 = NoteCandidate(2, Note.from_name("E4"))
    events = [
        RawEvent(0.0, 0.18, [f4], False, "F4", 320.0),
        RawEvent(0.18, 0.29, [f4, g4], False, "F4", 300.0),
        RawEvent(0.29, 0.5, [e4], False, "E4", 290.0),
    ]

    cleaned = suppress_resonant_carryover(events)

    assert [note.note_name for note in cleaned[1].notes] == ["F4"]


def test_suppress_descending_upper_return_overlap_drops_residual_dyad() -> None:
    e6 = NoteCandidate(17, Note.from_name("E6"))
    d6 = NoteCandidate(1, Note.from_name("D6"))
    c6 = NoteCandidate(16, Note.from_name("C6"))
    events = [
        RawEvent(0.0, 0.14, [e6], False, "E6", 320.0),
        RawEvent(0.14, 0.25, [d6], False, "D6", 300.0),
        RawEvent(0.25, 0.36, [d6, e6], False, "D6", 280.0),
        RawEvent(0.36, 0.58, [c6], False, "C6", 290.0),
    ]

    cleaned = suppress_descending_upper_return_overlap(events)

    assert [[note.note_name for note in event.notes] for event in cleaned] == [["E6"], ["D6"], ["C6"]]


def test_suppress_leading_descending_overlap_collapses_first_bridge() -> None:
    tuning = get_default_tunings()[0]
    e6 = NoteCandidate(17, Note.from_name("E6"))
    d6 = NoteCandidate(1, Note.from_name("D6"))
    c6 = NoteCandidate(16, Note.from_name("C6"))
    events = [
        RawEvent(0.0, 0.12, [c6, e6], False, "E6", 400.0),
        RawEvent(0.12, 0.36, [d6], False, "D6", 320.0),
        RawEvent(0.36, 0.58, [c6], False, "C6", 300.0),
    ]

    simplified = suppress_leading_descending_overlap(events, tuning)

    assert [note.note_name for note in simplified[0].notes] == ["E6"]


def test_build_recent_ascending_primary_run_ceiling_uses_latest_suffix() -> None:
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    d6 = NoteCandidate(key=1, note=Note.from_name("D6"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.2, notes=[d6], is_gliss_like=False, primary_note_name="D6", primary_score=500.0),
        RawEvent(start_time=0.2, end_time=0.4, notes=[d4], is_gliss_like=False, primary_note_name="D4", primary_score=120.0),
        RawEvent(start_time=0.4, end_time=0.6, notes=[f4], is_gliss_like=False, primary_note_name="F4", primary_score=130.0),
        RawEvent(start_time=0.6, end_time=0.8, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=140.0),
    ]

    assert build_recent_ascending_primary_run_ceiling(raw_events) == g4.frequency


def test_should_block_descending_repeated_primary_tertiary_extension_requires_descending_suffix_context() -> None:
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    assert should_block_descending_repeated_primary_tertiary_extension(
        selected=[g4, b4],
        extension=d5,
        segment_duration=0.116,
        previous_primary_was_singleton=True,
        descending_primary_suffix_floor=g4.frequency,
        descending_primary_suffix_ceiling=440.0,
        descending_primary_suffix_note_names={"G4", "A4"},
    ) is True

    assert should_block_descending_repeated_primary_tertiary_extension(
        selected=[g4, b4],
        extension=d5,
        segment_duration=0.116,
        previous_primary_was_singleton=False,
        descending_primary_suffix_floor=g4.frequency,
        descending_primary_suffix_ceiling=440.0,
        descending_primary_suffix_note_names={"G4", "A4"},
    ) is False

    assert should_block_descending_repeated_primary_tertiary_extension(
        selected=[g4, b4],
        extension=d5,
        segment_duration=0.116,
        previous_primary_was_singleton=True,
        descending_primary_suffix_floor=None,
        descending_primary_suffix_ceiling=440.0,
        descending_primary_suffix_note_names={"G4", "A4"},
    ) is False


def test_build_recent_note_names_collapses_consecutive_duplicates() -> None:
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[c4], is_gliss_like=False),
        RawEvent(start_time=0.4, end_time=0.8, notes=[d4], is_gliss_like=False),
        RawEvent(start_time=0.8, end_time=1.2, notes=[e4], is_gliss_like=False),
        RawEvent(start_time=1.2, end_time=1.5, notes=[f4], is_gliss_like=False),
        RawEvent(start_time=1.5, end_time=1.7, notes=[f4], is_gliss_like=False),
        RawEvent(start_time=1.7, end_time=1.9, notes=[f4], is_gliss_like=False),
    ]

    assert build_recent_note_names(raw_events) == {"C4", "D4", "E4", "F4"}


def test_suppress_resonant_carryover_prefers_fresh_ascending_note() -> None:
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    c5 = NoteCandidate(key=5, note=Note.from_name("C5"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))
    f5 = NoteCandidate(key=14, note=Note.from_name("F5"))
    g5 = NoteCandidate(key=3, note=Note.from_name("G5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[c4], is_gliss_like=False),
        RawEvent(start_time=0.4, end_time=0.8, notes=[c4, c5], is_gliss_like=False),
        RawEvent(start_time=0.8, end_time=1.2, notes=[c4, d5], is_gliss_like=False),
        RawEvent(start_time=1.2, end_time=1.6, notes=[c5, e5], is_gliss_like=False),
        RawEvent(start_time=1.6, end_time=2.0, notes=[g5], is_gliss_like=False),
        RawEvent(start_time=2.0, end_time=2.1, notes=[f5, g5], is_gliss_like=True),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C4"],
        ["C5"],
        ["D5"],
        ["C5", "E5"],
        ["G5"],
        ["F5"],
    ]

def test_suppress_resonant_carryover_keeps_true_short_octave_dyad() -> None:
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[d4], is_gliss_like=False),
        RawEvent(start_time=0.6, end_time=0.88, notes=[d4, d5], is_gliss_like=False),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["D4"],
        ["D4", "D5"],
    ]

def test_suppress_resonant_carryover_keeps_phrase_reset_ascending_dyad() -> None:
    c5 = NoteCandidate(key=14, note=Note.from_name("C5"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))
    g5 = NoteCandidate(key=3, note=Note.from_name("G5"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.08, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.08, end_time=0.33, notes=[e5, g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
        RawEvent(start_time=0.92, end_time=1.24, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=260.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C5", "E5"],
        ["E5", "G5"],
        ["G4"],
    ]


def test_suppress_resonant_carryover_keeps_lower_note_when_high_return_is_stale() -> None:
    e6 = NoteCandidate(key=17, note=Note.from_name("E6"))
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[e6], is_gliss_like=False, primary_note_name="E6", primary_score=700.0),
        RawEvent(start_time=0.4, end_time=0.78, notes=[c4], is_gliss_like=False, primary_note_name="C4", primary_score=380.0),
        RawEvent(start_time=0.78, end_time=0.98, notes=[c4, e6], is_gliss_like=False, primary_note_name="E6", primary_score=220.0),
        RawEvent(start_time=0.98, end_time=1.36, notes=[d4], is_gliss_like=False, primary_note_name="D4", primary_score=340.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["E6"],
        ["C4"],
        ["C4"],
        ["D4"],
    ]

def test_suppress_resonant_carryover_keeps_repeated_note_for_short_restart_overlap() -> None:
    c5 = NoteCandidate(key=5, note=Note.from_name("C5"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.32, notes=[c5], is_gliss_like=False, primary_note_name="C5", primary_score=320.0),
        RawEvent(start_time=0.32, end_time=0.438, notes=[c5, e4], is_gliss_like=False, primary_note_name="C5", primary_score=180.0),
        RawEvent(start_time=0.438, end_time=0.62, notes=[d5], is_gliss_like=False, primary_note_name="D5", primary_score=260.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C5"],
        ["C5"],
        ["D5"],
    ]


def test_suppress_resonant_carryover_keeps_repeated_note_for_short_post_triad_upper_tail() -> None:
    tuning = next(tuning for tuning in get_default_tunings() if tuning.id == "kalimba-17-c")
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.44, notes=[g4, b4, d5], is_gliss_like=False, primary_note_name="D5", primary_score=420.0),
        RawEvent(start_time=0.44, end_time=0.556, notes=[d5, e5], is_gliss_like=False, primary_note_name="D5", primary_score=210.0),
        RawEvent(start_time=0.556, end_time=0.76, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=260.0),
    ]

    cleaned = suppress_resonant_carryover(raw_events, tuning)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["G4", "B4", "D5"],
        ["D5"],
        ["G4"],
    ]

def test_collapse_same_start_primary_singletons_prefers_singleton_over_lower_carryover() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.12, notes=[a4, e4], is_gliss_like=False, primary_note_name="A4", primary_score=140.0),
        RawEvent(start_time=0.0, end_time=0.32, notes=[a4], is_gliss_like=False, primary_note_name="A4", primary_score=180.0),
    ]

    cleaned = collapse_same_start_primary_singletons(raw_events)
    assert len(cleaned) == 1
    assert [note.note_name for note in cleaned[0].notes] == ["A4"]
    assert cleaned[0].start_time == 0.0
    assert cleaned[0].end_time == 0.32


def test_collapse_same_start_primary_singletons_prefers_singleton_when_following_primary_is_lower_carryover() -> None:
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.18, notes=[g4], is_gliss_like=False, primary_note_name="G4", primary_score=220.0),
        RawEvent(start_time=0.0, end_time=0.32, notes=[c4, g4], is_gliss_like=False, primary_note_name="C4", primary_score=80.0),
    ]

    cleaned = collapse_same_start_primary_singletons(raw_events)

    assert len(cleaned) == 1
    assert [note.note_name for note in cleaned[0].notes] == ["G4"]
    assert cleaned[0].start_time == 0.0
    assert cleaned[0].end_time == 0.32


def test_merge_short_chord_clusters_merges_singleton_and_dyad_into_triad() -> None:
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.14, notes=[c4], is_gliss_like=False, primary_note_name="C4", primary_score=80.0),
        RawEvent(start_time=0.14, end_time=0.9, notes=[e4, g4], is_gliss_like=False, primary_note_name="G4", primary_score=500.0),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C4", "E4", "G4"]]


def test_merge_short_chord_clusters_does_not_merge_gliss_like_head_with_dyad() -> None:
    c5 = NoteCandidate(key=13, note=Note.from_name("C5"))
    e5 = NoteCandidate(key=15, note=Note.from_name("E5"))
    g5 = NoteCandidate(key=17, note=Note.from_name("G5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.0667, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.0667, end_time=0.3174, notes=[g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C5", "E5"], ["G5"]]

def test_merge_short_gliss_clusters_does_not_merge_gliss_head_with_longer_overlapping_dyad() -> None:
    c5 = NoteCandidate(key=13, note=Note.from_name("C5"))
    e5 = NoteCandidate(key=15, note=Note.from_name("E5"))
    g5 = NoteCandidate(key=17, note=Note.from_name("G5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.0667, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.0667, end_time=0.3174, notes=[e5, g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
    ]

    merged = merge_short_gliss_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["C5", "E5"], ["E5", "G5"]]

def test_simplify_short_gliss_prefix_to_contiguous_singleton_handles_dyad_head_before_dyad() -> None:
    c5 = NoteCandidate(key=5, note=Note.from_name("C5"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))
    g5 = NoteCandidate(key=3, note=Note.from_name("G5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.0667, notes=[c5, e5], is_gliss_like=True, primary_note_name="C5", primary_score=121.0),
        RawEvent(start_time=0.0667, end_time=0.3174, notes=[e5, g5], is_gliss_like=False, primary_note_name="G5", primary_score=554.7),
    ]

    cleaned = simplify_short_gliss_prefix_to_contiguous_singleton(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [["C5"], ["E5", "G5"]]


def test_merge_short_chord_clusters_merges_subset_into_following_triad() -> None:
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.8, notes=[d4, f4], is_gliss_like=False, primary_note_name="D4", primary_score=1000.0),
        RawEvent(start_time=0.8, end_time=1.1, notes=[d4, f4, a4], is_gliss_like=True, primary_note_name="A4", primary_score=1200.0),
    ]

    merged = merge_short_chord_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["D4", "F4", "A4"]]


def test_simplify_short_gliss_prefix_to_contiguous_singleton_picks_matching_note() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.08, notes=[e4, f4], is_gliss_like=True, primary_note_name="E4", primary_score=120.0),
        RawEvent(start_time=0.08, end_time=0.9, notes=[g4, b4, d5], is_gliss_like=True, primary_note_name="G4", primary_score=500.0),
    ]

    simplified = simplify_short_gliss_prefix_to_contiguous_singleton(raw_events)
    assert [[note.note_name for note in event.notes] for event in simplified] == [["E4"], ["G4", "B4", "D5"]]


def test_merge_four_note_gliss_clusters_merges_triad_and_singleton() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[e4, g4, b4], is_gliss_like=True, primary_note_name="E4", primary_score=500.0),
        RawEvent(start_time=0.6, end_time=0.82, notes=[d5], is_gliss_like=True, primary_note_name="D5", primary_score=420.0),
    ]

    merged = merge_four_note_gliss_clusters(raw_events)
    assert [[note.note_name for note in event.notes] for event in merged] == [["E4", "G4", "B4", "D5"]]
    assert merged[0].is_gliss_like is True



def test_suppress_leading_gliss_neighbor_noise_drops_short_dyad_before_four_note_gliss() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.08, notes=[e4, f4], is_gliss_like=True, primary_note_name="E4", primary_score=120.0),
        RawEvent(start_time=0.08, end_time=0.9, notes=[e4, g4, b4, d5], is_gliss_like=True, primary_note_name="E4", primary_score=500.0),
    ]

    cleaned = suppress_leading_gliss_neighbor_noise(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [["E4", "G4", "B4", "D5"]]


def test_suppress_leading_gliss_subset_transients_drops_short_prefix() -> None:
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.09, notes=[c4], is_gliss_like=True, primary_note_name="C4", primary_score=80.0),
        RawEvent(start_time=0.09, end_time=0.5, notes=[c4, e4, g4], is_gliss_like=True, primary_note_name="G4", primary_score=500.0),
    ]

    cleaned = suppress_leading_gliss_subset_transients(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [["C4", "E4", "G4"]]

def test_suppress_short_residual_tails_drops_recent_single_note_tail() -> None:
    c5 = NoteCandidate(key=5, note=Note.from_name("C5"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))
    g5 = NoteCandidate(key=3, note=Note.from_name("G5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.4, notes=[d5], is_gliss_like=False),
        RawEvent(start_time=0.4, end_time=0.9, notes=[c5, e5], is_gliss_like=False),
        RawEvent(start_time=0.9, end_time=0.99, notes=[d5], is_gliss_like=True),
        RawEvent(start_time=1.01, end_time=1.4, notes=[g5], is_gliss_like=False),
    ]

    cleaned = suppress_short_residual_tails(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["D5"],
        ["C5", "E5"],
        ["G5"],
    ]


def test_suppress_subset_decay_events_drops_contiguous_subset_tail() -> None:
    c5 = NoteCandidate(key=5, note=Note.from_name("C5"))
    e5 = NoteCandidate(key=4, note=Note.from_name("E5"))
    g5 = NoteCandidate(key=3, note=Note.from_name("G5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.5, notes=[c5, e5], is_gliss_like=False),
        RawEvent(start_time=0.5, end_time=1.0, notes=[e5], is_gliss_like=False),
        RawEvent(start_time=1.05, end_time=1.4, notes=[g5], is_gliss_like=False),
    ]

    cleaned = suppress_subset_decay_events(raw_events)
    assert [[note.note_name for note in event.notes] for event in cleaned] == [
        ["C5", "E5"],
        ["G5"],
    ]


def test_classify_event_gesture_strict_chord() -> None:
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))

    event = RawEvent(start_time=0.0, end_time=0.7, notes=[c4, e4, g4], is_gliss_like=False)
    assert classify_event_gesture(event, 0, [event], [event]) == "strict_chord"


def test_classify_event_gesture_slide_chord_from_subset_growth() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.5, notes=[g4, b4, d5], is_gliss_like=False),
        RawEvent(start_time=0.5, end_time=0.7, notes=[e4, g4, b4], is_gliss_like=False),
    ]
    merged_event = RawEvent(start_time=0.0, end_time=0.7, notes=[e4, g4, b4, d5], is_gliss_like=False)
    assert classify_event_gesture(merged_event, 0, raw_events, [merged_event]) == "slide_chord"


def test_classify_event_gesture_slide_chord_from_neighbor_progression() -> None:
    c4 = NoteCandidate(key=9, note=Note.from_name("C4"))
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    d4 = NoteCandidate(key=8, note=Note.from_name("D4"))
    f4 = NoteCandidate(key=7, note=Note.from_name("F4"))
    a4 = NoteCandidate(key=6, note=Note.from_name("A4"))

    merged_events = [
        RawEvent(start_time=0.0, end_time=0.5, notes=[c4, e4, g4], is_gliss_like=True),
        RawEvent(start_time=0.52, end_time=1.0, notes=[d4, f4, a4], is_gliss_like=True),
    ]
    assert classify_event_gesture(merged_events[0], 0, merged_events, merged_events) == "slide_chord"


def test_classify_event_gesture_slide_chord_from_gliss_like_family_without_neighbor_shift() -> None:
    e4 = NoteCandidate(key=10, note=Note.from_name("E4"))
    g4 = NoteCandidate(key=11, note=Note.from_name("G4"))
    b4 = NoteCandidate(key=12, note=Note.from_name("B4"))
    d5 = NoteCandidate(key=13, note=Note.from_name("D5"))

    raw_events = [
        RawEvent(start_time=0.0, end_time=0.6, notes=[g4, b4, d5], is_gliss_like=True),
        RawEvent(start_time=0.6, end_time=1.1, notes=[e4, g4, b4, d5], is_gliss_like=True),
    ]
    merged_event = RawEvent(start_time=0.0, end_time=1.1, notes=[e4, g4, b4, d5], is_gliss_like=True)

    assert classify_event_gesture(merged_event, 0, raw_events, [merged_event]) == "slide_chord"
