"""Differential tests: kalimba_dsp segment-stage port vs the Python source.

B1 slice 1 (sprint-plan 2026-07 S5): the Rust shared core gained
rms_threshold / raw_active_ranges / merge_time_ranges, ported from the
active-range head of detect_segments (segments.py). Python remains the
production implementation; these tests pin the pyo3 binding against the
Python semantics on constructed inputs (the wasm side is pinned on real
fixture audio by crates/kalimba-dsp/check_wasm.sh step 5).
"""

from __future__ import annotations

import numpy as np
import pytest

import kalimba_dsp
from app.transcription.segments import merge_time_ranges


def python_threshold(rms: np.ndarray) -> float:
    """Replica of the detect_segments threshold line (segments.py)."""
    max_rms = float(np.max(rms))
    median_rms = float(np.median(rms))
    return max(max_rms * 0.18, min(median_rms * 2.2, max_rms * 0.45), 0.01)


def python_raw_active_ranges(
    rms: np.ndarray, sample_rate: int, hop_length: int, duration_sec: float
) -> list[tuple[float, float]]:
    """Replica of the detect_segments active-range scan (segments.py)."""
    threshold = python_threshold(rms)
    frame_times = np.arange(len(rms)) * hop_length / float(sample_rate)
    active_frames = rms >= threshold

    ranges: list[tuple[float, float]] = []
    active_start = None
    for index, is_active in enumerate(active_frames):
        if is_active and active_start is None:
            active_start = index
        elif not is_active and active_start is not None:
            start_time = max(float(frame_times[active_start]) - 0.02, 0.0)
            end_time = float(frame_times[min(index, len(frame_times) - 1)]) + 0.08
            ranges.append((start_time, end_time))
            active_start = None
    if active_start is not None:
        ranges.append((max(float(frame_times[active_start]) - 0.02, 0.0), duration_sec))
    return ranges


def flat_to_pairs(flat: np.ndarray) -> list[tuple[float, float]]:
    values = np.asarray(flat, dtype=np.float64)
    return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]


@pytest.fixture
def noisy_rms() -> np.ndarray:
    rng = np.random.default_rng(42)
    quiet = rng.uniform(0.0, 0.02, size=64).astype(np.float32)
    loud = rng.uniform(0.3, 1.0, size=48).astype(np.float32)
    return np.concatenate([quiet, loud, quiet, loud[:12], quiet[:8]]).astype(np.float32)


def test_rms_threshold_matches_python(noisy_rms: np.ndarray) -> None:
    assert kalimba_dsp.rms_threshold(noisy_rms) == pytest.approx(
        python_threshold(noisy_rms), abs=1e-12
    )


def test_rms_threshold_floor_and_median_cap() -> None:
    # All-quiet input: 0.01 floor wins.
    quiet = np.full(10, 0.001, dtype=np.float32)
    assert kalimba_dsp.rms_threshold(quiet) == pytest.approx(python_threshold(quiet), abs=1e-12)
    # High median: max_rms * 0.45 cap wins over median * 2.2.
    high = np.full(11, 0.8, dtype=np.float32)
    high[5] = 1.0
    assert kalimba_dsp.rms_threshold(high) == pytest.approx(python_threshold(high), abs=1e-12)


def test_raw_active_ranges_matches_python(noisy_rms: np.ndarray) -> None:
    duration = len(noisy_rms) * 256 / 48000.0 + 0.01
    got = flat_to_pairs(kalimba_dsp.raw_active_ranges(noisy_rms, 48000, 256, duration))
    expected = python_raw_active_ranges(noisy_rms, 48000, 256, duration)
    assert got == pytest.approx(expected, abs=1e-12)


def test_raw_active_ranges_trailing_open_range_uses_duration() -> None:
    rms = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    got = flat_to_pairs(kalimba_dsp.raw_active_ranges(rms, 100, 10, 0.5))
    expected = python_raw_active_ranges(rms, 100, 10, 0.5)
    assert got == pytest.approx(expected, abs=1e-12)
    assert got[-1][1] == 0.5  # trailing open range extends to duration, no +0.08


def test_merge_time_ranges_matches_python() -> None:
    ranges = [(0.0, 1.0), (1.05, 2.0), (2.5, 3.0), (3.05, 4.0)]
    flat = np.array([v for pair in ranges for v in pair], dtype=np.float64)
    got = flat_to_pairs(kalimba_dsp.merge_time_ranges(flat, 0.06))
    assert got == pytest.approx(
        [tuple(pair) for pair in merge_time_ranges(ranges)], abs=1e-12
    )
    got_tight = flat_to_pairs(kalimba_dsp.merge_time_ranges(flat, 0.01))
    assert got_tight == pytest.approx(
        [tuple(pair) for pair in merge_time_ranges(ranges, gap_tolerance=0.01)], abs=1e-12
    )
