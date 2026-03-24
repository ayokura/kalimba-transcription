from scripts.audit_manual_captures import fixture_taxonomy


def _request_payload(events: list[dict]) -> dict:
    return {
        "expectedPerformance": {
            "events": events,
        }
    }


def _event(note_names: list[str], intent: str | None = None) -> dict:
    payload = {"keys": [{"noteName": note_name} for note_name in note_names]}
    if intent is not None:
        payload["intent"] = intent
    return payload


def test_fixture_taxonomy_classifies_single_event_repeat() -> None:
    request_payload = _request_payload([
        _event(["C4"]),
        _event(["C4"]),
        _event(["C4"]),
        _event(["C4"]),
    ])
    assert fixture_taxonomy(request_payload) == "single_event_repeat"


def test_fixture_taxonomy_classifies_small_repeated_phrase() -> None:
    request_payload = _request_payload([
        _event(["C4"]),
        _event(["E4"]),
        _event(["C4"]),
        _event(["E4"]),
    ])
    assert fixture_taxonomy(request_payload) == "small_repeated_phrase"


def test_fixture_taxonomy_classifies_free_performance() -> None:
    request_payload = _request_payload([
        _event(["C4"]),
        _event(["D4"]),
        _event(["E4"]),
        _event(["F4"]),
        _event(["G4"]),
        _event(["A4"]),
        _event(["B4"]),
        _event(["C5"]),
    ])
    assert fixture_taxonomy(request_payload) == "free_performance"


def test_fixture_taxonomy_classifies_mixed_phrase_from_intents() -> None:
    request_payload = _request_payload([
        _event(["C4"], intent="separated_notes"),
        _event(["E4", "G4"], intent="strict_chord"),
        _event(["A4", "C5", "E5"], intent="slide_chord"),
    ])
    assert fixture_taxonomy(request_payload) == "mixed_phrase"
