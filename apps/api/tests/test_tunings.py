from app.tunings import build_custom_tuning, get_default_tunings, note_name_to_frequency, parse_note_name


def test_default_tunings_are_available() -> None:
    tunings = get_default_tunings()
    assert len(tunings) >= 3
    assert tunings[0].key_count == len(tunings[0].notes)


def test_custom_tuning_uses_provided_notes() -> None:
    tuning = build_custom_tuning("My Kalimba", ["C4", "E4", "G4"])
    assert tuning.name == "My Kalimba"
    assert [note.note_name for note in tuning.notes] == ["C4", "E4", "G4"]


def test_note_name_parser_supports_accidentals_and_multi_digit_octave() -> None:
    assert parse_note_name("bb10") == ("Bb", 10)


def test_note_name_to_frequency_matches_a440_reference() -> None:
    assert round(note_name_to_frequency("A4"), 2) == 440.00