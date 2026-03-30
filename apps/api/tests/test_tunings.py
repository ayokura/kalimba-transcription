import pytest
from fastapi import HTTPException

from app.tunings import build_custom_tuning, get_default_tunings, note_name_to_frequency, parse_note_name


def test_default_tunings_are_available() -> None:
    tunings = get_default_tunings()
    assert len(tunings) >= 3
    assert tunings[0].key_count == len(tunings[0].notes)


def test_custom_tuning_uses_provided_notes() -> None:
    tuning = build_custom_tuning("My Kalimba", ["C4", "E4", "G4"])
    assert tuning.name == "My Kalimba"
    assert [note.note_name for note in tuning.notes] == ["C4", "E4", "G4"]


def test_custom_tuning_preserves_duplicate_note_names() -> None:
    tuning = build_custom_tuning("Chromatic 34", ["C4", "D4", "C4"])
    assert [note.note_name for note in tuning.notes] == ["C4", "D4", "C4"]


def test_custom_tuning_rejects_non_string_note_names() -> None:
    with pytest.raises(HTTPException) as exc_info:
        build_custom_tuning("My Kalimba", ["C4", 7])  # type: ignore[list-item]

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Each tuning noteName must be a non-empty string."


def test_custom_tuning_rejects_blank_note_names() -> None:
    with pytest.raises(HTTPException) as exc_info:
        build_custom_tuning("My Kalimba", ["C4", "  "])

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Each tuning noteName must be a non-empty string."


def test_custom_tuning_rejects_non_string_name() -> None:
    with pytest.raises(HTTPException) as exc_info:
        build_custom_tuning(17, ["C4"])  # type: ignore[arg-type]

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Tuning name must be a string."


def test_note_name_parser_supports_accidentals_and_multi_digit_octave() -> None:
    assert parse_note_name("bb10") == ("Bb", 10)


def test_note_name_to_frequency_matches_a440_reference() -> None:
    assert round(note_name_to_frequency("A4"), 2) == 440.00
