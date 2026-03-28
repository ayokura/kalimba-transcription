import json

from app.tunings import get_default_tunings
from conftest import (
    client,
    synthesize_repeated_chord,
    synthesize_repeated_note,
    wav_bytes,
)


def test_transcription_regression_for_repeated_octave_dyad() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_repeated_chord((587.3295, 1174.6591))
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning.model_dump(by_alias=True)), "debug": "true"},
        files={"file": ("d5-d6.wav", wav_bytes(audio), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    note_sets = [
        sorted(f"{note['pitchClass']}{note['octave']}" for note in event["notes"])
        for event in payload["events"]
    ]
    octave_hits = sum(1 for note_set in note_sets if note_set == ["D5", "D6"])
    assert octave_hits >= 3
    assert payload["debug"]["segmentCandidates"]

def test_transcription_regression_for_repeated_d5() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_repeated_note(587.3295)
    response = client.post(
        "/api/transcriptions",
        data={"tuning": json.dumps(tuning.model_dump(by_alias=True)), "debug": "true"},
        files={"file": ("d5.wav", wav_bytes(audio), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert 3 <= len(payload["events"]) <= 7
    detected_d5 = sum(1 for event in payload["events"] if any(note["pitchClass"] == "D" and note["octave"] == 5 for note in event["notes"]))
    assert detected_d5 >= 4
    assert payload["debug"]["segments"]

def test_transcription_debug_reports_disabled_repeated_pattern_passes() -> None:
    tuning = get_default_tunings()[0]
    audio = synthesize_repeated_chord((293.6647679174076, 349.2282314330039, 440.0), repeats=4)

    response = client.post(
        "/api/transcriptions",
        data={
            "tuning": json.dumps(tuning.model_dump(by_alias=True)),
            "debug": "true",
            "disabledRepeatedPatternPasses": json.dumps(["normalize_repeated_triad_patterns"]),
        },
        files={"file": ("d4-f4-a4.wav", wav_bytes(audio), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"]["disabledRepeatedPatternPasses"] == ["normalize_repeated_triad_patterns"]
    triad_trace = next(item for item in payload["debug"]["repeatedPatternPassTrace"] if item["pass"] == "normalize_repeated_triad_patterns")
    assert triad_trace["enabled"] is False
