from __future__ import annotations

from unittest.mock import patch

import numpy as np

from app.models import InstrumentTuning, TuningNote
from app.transcription.peaks import measure_narrow_fft_note_scores


def _tuning() -> InstrumentTuning:
    return InstrumentTuning(
        id="cache-test",
        name="cache-test",
        key_count=1,
        notes=[TuningNote(key=1, note_name="A4", frequency=440.0)],
    )


def test_measure_narrow_fft_note_scores_reuses_time_cache():
    audio = np.zeros(44100, dtype=np.float32)
    tuning = _tuning()
    cache: dict[tuple[int, float], dict[str, tuple[float, float, float]] | None] = {}
    fake_result = {"A4": (42.0, 7.0, 0.99)}

    class _Hypothesis:
        def __init__(self):
            self.candidate = type("Candidate", (), {"note_name": "A4"})()
            self.fundamental_energy = 42.0
            self.score = 7.0
            self.fundamental_ratio = 0.99

    fake_spectral = type("Spectral", (), {"ranked": [_Hypothesis()]})()
    with patch(
        "app.transcription.peaks._narrow_fft_at_sub_onset",
        return_value=fake_spectral,
    ) as mock_fft:
        first = measure_narrow_fft_note_scores(
            audio, 44100, 0.5, tuning, cache=cache,
        )
        second = measure_narrow_fft_note_scores(
            audio, 44100, 0.5, tuning, cache=cache,
        )

    assert first == fake_result
    assert second == fake_result
    assert mock_fft.call_count == 1
