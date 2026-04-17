import pytest
from fastapi import HTTPException

from app.tunings import build_custom_tuning, get_default_tunings, major_17_key_layout, note_name_to_frequency, parse_note_name


def test_default_tunings_are_available() -> None:
    tunings = get_default_tunings()
    assert len(tunings) >= 3
    assert tunings[0].key_count == len(tunings[0].notes)


def test_major_17_key_layout_matches_c_major_preset() -> None:
    # The helper must reproduce the hand-authored C major layout exactly.
    assert major_17_key_layout("C4") == [
        "D6", "B5", "G5", "E5", "C5", "A4", "F4", "D4",
        "C4",
        "E4", "G4", "B4", "D5", "F5", "A5", "C6", "E6",
    ]


def test_major_17_key_layout_d_major() -> None:
    # D major: F#, C# as accidentals.
    layout = major_17_key_layout("D4")
    assert layout[8] == "D4"
    assert layout[9] == "F#4"  # 3rd
    assert layout[-1] == "F#6"  # highest (3rd +2oct)


def test_major_17_key_layout_f_major_uses_flats() -> None:
    # F major naturally uses Bb (not A#).
    layout = major_17_key_layout("F4", use_flats=True)
    assert layout[8] == "F4"
    assert "Bb4" in layout
    assert "A#4" not in layout


def test_new_japanese_key_presets_are_registered() -> None:
    ids = {t.id for t in get_default_tunings()}
    expected = {
        "kalimba-17-d",
        "kalimba-17-e",
        "kalimba-17-f",
        "kalimba-17-a",
        "kalimba-17-b",
        "kalimba-17-bb",
    }
    assert expected.issubset(ids)


def test_all_default_presets_have_tonic() -> None:
    for tuning in get_default_tunings():
        assert tuning.tonic is not None, f"{tuning.id} is missing tonic"


def test_tonic_matches_expected_per_preset() -> None:
    expected = {
        "kalimba-17-c": "C",
        "kalimba-17-d": "D",
        "kalimba-17-e": "E",
        "kalimba-17-f": "F",
        "kalimba-17-g": "G",
        "kalimba-17-a": "A",
        "kalimba-17-b": "B",
        "kalimba-17-bb": "Bb",
        "kalimba-17-g-low": "G",
        "kalimba-34l-c": "C",
        "kalimba-21-c": "C",
    }
    by_id = {t.id: t for t in get_default_tunings()}
    for tid, tonic in expected.items():
        assert by_id[tid].tonic == tonic, f"{tid} tonic mismatch"


def test_custom_tuning_has_no_tonic() -> None:
    tuning = build_custom_tuning("My Kalimba", ["C4", "E4", "G4"])
    assert tuning.tonic is None


def test_new_presets_all_have_17_keys() -> None:
    ids_to_check = {
        "kalimba-17-d", "kalimba-17-e", "kalimba-17-f",
        "kalimba-17-a", "kalimba-17-b", "kalimba-17-bb",
    }
    for tuning in get_default_tunings():
        if tuning.id in ids_to_check:
            assert tuning.key_count == 17, f"{tuning.id} has {tuning.key_count} keys"
            assert len(tuning.notes) == 17


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
