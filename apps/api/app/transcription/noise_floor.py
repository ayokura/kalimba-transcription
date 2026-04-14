"""Per-recording per-band noise floor calibration (#154 / #153 Phase B).

The Phase A narrow-FFT merge passes (``merge_short_segment_guard_via_narrow_fft``,
``recover_masked_reattack_via_narrow_fft``) compare a candidate note's
``fundamental_energy`` against a fixed absolute threshold.  This works for the
Phase A motivating cases (E148 C6, E97/E133 D5) but breaks down on much
weaker attacks such as 17-key BWV147 E97 G4 (peak band energy on the order
of single digits, ~1/6500 of a clean G4 attack).

Phase B replaces the fixed thresholds with ``noise_floor[note] * factor``.
The noise floor is measured by sampling silent regions of the recording
(gaps between detected segments, plus the leading/trailing tail) with the
same narrow-FFT window the merge passes use, then taking the median
``fundamental_energy`` per tuning note.  The result adapts automatically
to mic gain, room noise, and per-band differences across recordings.

Design constraints (AGENTS.md "Free Performance" / "Browser-side"):
    * No score knowledge — silent region detection only uses segment
      timing already produced by ``detect_segments`` / ``rescue_gap_mute_dips``.
    * Streaming-friendly — once the leading silence has been measured the
      same noise floor can be reused for the rest of the stream.
    * Portable arithmetic — the FFT itself is delegated to numpy /
      ``batch_peak_energies`` (already imported by peaks.py); no librosa
      coupling here.

The module exposes :func:`measure_noise_floor` and the
:class:`NoiseFloorMeasurement` value object that the merge passes consume
via :meth:`NoiseFloorMeasurement.threshold_for`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..models import InstrumentTuning
from .constants import (
    HARMONIC_BAND_CENTS,
    NARROW_FFT_WINDOW_SECONDS,
    NOISE_FLOOR_EDGE_PADDING_SECONDS,
    NOISE_FLOOR_MAX_SAMPLES,
    NOISE_FLOOR_MIN_SILENT_GAP_SECONDS,
)
from .models import Segment
from .audio import cached_hanning, cached_rfftfreq
from .peaks import _adaptive_n_fft, batch_peak_energies


@dataclass(frozen=True)
class NoiseFloorMeasurement:
    """Result of per-recording per-band noise floor calibration.

    ``per_note`` maps tuning note name to the median ``fundamental_energy``
    measured at that note's frequency across all silent narrow-FFT
    samples.  Notes with no measurable signal in any silent slice
    (numerically zero energy in every sample) are absent.

    ``silent_regions`` lists the time windows that were sampled and is
    surfaced in the debug payload so investigations can verify the
    calibration used real silence.

    ``sample_count`` is the number of silent slices that successfully
    contributed to the median.
    """

    per_note: dict[str, float]
    silent_regions: tuple[tuple[float, float], ...]
    sample_count: int

    @property
    def is_empty(self) -> bool:
        return self.sample_count == 0 or not self.per_note

    def threshold_for(
        self,
        note_name: str,
        *,
        factor: float,
        fallback: float,
        hard_floor: float = 0.0,
    ) -> float:
        """Compute a noise-floor-aware energy threshold for *note_name*.

        Returns ``noise_floor[note_name] * factor`` clamped to at least
        *hard_floor* when a measurement is available, else *fallback*.

        The fallback path lets callers stay safe when noise floor
        calibration produced no useful samples (e.g., a synthetic
        fixture with no silent region) by reusing their Phase A
        absolute threshold.
        """
        floor = self.per_note.get(note_name)
        if floor is None or floor <= 0.0:
            return fallback
        return max(floor * factor, hard_floor)

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "perNote": {
                name: round(energy, 6) for name, energy in sorted(self.per_note.items())
            },
            "silentRegions": [
                {"startTime": round(start, 4), "endTime": round(end, 4)}
                for start, end in self.silent_regions
            ],
            "sampleCount": self.sample_count,
        }


_EMPTY_MEASUREMENT = NoiseFloorMeasurement(per_note={}, silent_regions=(), sample_count=0)


def _collect_silent_regions(
    audio_duration: float,
    segments: Sequence[Segment],
    *,
    min_gap_seconds: float,
    edge_padding_seconds: float,
) -> list[tuple[float, float]]:
    """Return ``(start, end)`` windows that sit between segments.

    The leading region (recording start to the first segment) and the
    trailing region (last segment to recording end) are included.
    Each candidate is shrunk by ``edge_padding_seconds`` on both sides
    so the silent slice does not pick up segment attack rises or decay
    tails, then dropped if the remaining length is below
    ``min_gap_seconds``.
    """
    regions: list[tuple[float, float]] = []
    if audio_duration <= 0.0:
        return regions

    sorted_segments = sorted(segments, key=lambda s: s.start_time)
    boundaries: list[tuple[float, float]] = []
    if sorted_segments:
        boundaries.append((0.0, sorted_segments[0].start_time))
        for prev, nxt in zip(sorted_segments, sorted_segments[1:]):
            boundaries.append((prev.end_time, nxt.start_time))
        boundaries.append((sorted_segments[-1].end_time, audio_duration))
    else:
        boundaries.append((0.0, audio_duration))

    for raw_start, raw_end in boundaries:
        start = max(raw_start + edge_padding_seconds, 0.0)
        end = min(raw_end - edge_padding_seconds, audio_duration)
        if end - start >= min_gap_seconds:
            regions.append((start, end))

    return regions


def _select_samples(
    regions: Sequence[tuple[float, float]],
    *,
    window_seconds: float,
    max_samples: int,
) -> list[float]:
    """Pick centre times for narrow-FFT slices.

    Long regions are preferred (sorted by length, descending) so the
    median is dominated by real silence rather than tight inter-attack
    pauses that may still contain decay tails.  Each selected region
    contributes a single slice centred on its midpoint.

    The window itself is opened by the FFT helper around the centre
    time; here we only return the centres.
    """
    if not regions or max_samples <= 0:
        return []
    sortable = sorted(
        regions,
        key=lambda r: r[1] - r[0],
        reverse=True,
    )
    centres: list[float] = []
    for start, end in sortable[:max_samples]:
        if end - start < window_seconds:
            # Tighter than one FFT window — skip rather than pad with
            # silence-padded edges that would bias the energy estimate.
            continue
        centres.append(0.5 * (start + end))
    return centres


def _narrow_fft_band_energies(
    audio: np.ndarray,
    sample_rate: int,
    centre_time: float,
    note_freqs: np.ndarray,
    *,
    window_seconds: float,
    min_frequency: float,
) -> np.ndarray | None:
    """Compute per-note ``fundamental_energy`` at *centre_time*.

    Mirrors the FFT setup in :func:`peaks._narrow_fft_at_sub_onset` (same
    window, same Hann taper, same adaptive ``n_fft`` floor) but skips
    the full ranking pipeline because we only need raw band energies
    at the tuning frequencies — silent slices have no ranked
    candidates and would otherwise return ``None``.

    Returns ``None`` if the window cannot be filled (recording too
    short at the requested centre).
    """
    window_samples = max(int(sample_rate * window_seconds), 256)
    centre_sample = int(centre_time * sample_rate)
    half = window_samples // 2
    start = max(centre_sample - half, 0)
    end = min(start + window_samples, len(audio))
    chunk = audio[start:end]
    if len(chunk) < 256:
        return None
    n_fft = _adaptive_n_fft(sample_rate, min_frequency, len(chunk))
    window = cached_hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(chunk * window, n=n_fft))
    frequencies = cached_rfftfreq(n_fft, sample_rate)
    return batch_peak_energies(
        frequencies, spectrum, note_freqs, band_cents=HARMONIC_BAND_CENTS,
    )


def measure_noise_floor(
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    segments: Sequence[Segment],
    *,
    window_seconds: float = NARROW_FFT_WINDOW_SECONDS,
    min_silent_gap_seconds: float = NOISE_FLOOR_MIN_SILENT_GAP_SECONDS,
    edge_padding_seconds: float = NOISE_FLOOR_EDGE_PADDING_SECONDS,
    max_samples: int = NOISE_FLOOR_MAX_SAMPLES,
) -> NoiseFloorMeasurement:
    """Calibrate per-tuning-note noise floors from silent slices.

    Returns an empty :class:`NoiseFloorMeasurement` (callers will fall
    back to Phase A absolute thresholds) when:

    * the recording has zero duration,
    * no silent gap between segments is long enough to fit a window,
    * every selected slice fails the FFT window length check.

    The same narrow-FFT window seconds used by the Phase A merge passes
    must be passed in (default :data:`NARROW_FFT_WINDOW_SECONDS`) so the
    measured noise floor is directly comparable to the
    ``fundamental_energy`` values produced by
    :func:`peaks.measure_narrow_fft_note_scores`.
    """
    if len(audio) == 0 or sample_rate <= 0:
        return _EMPTY_MEASUREMENT
    audio_duration = len(audio) / float(sample_rate)
    regions = _collect_silent_regions(
        audio_duration,
        segments,
        min_gap_seconds=min_silent_gap_seconds,
        edge_padding_seconds=edge_padding_seconds,
    )
    centres = _select_samples(
        regions,
        window_seconds=window_seconds,
        max_samples=max_samples,
    )
    if not centres:
        return _EMPTY_MEASUREMENT

    note_freqs = np.array([note.frequency for note in tuning.notes], dtype=float)
    note_names = [note.note_name for note in tuning.notes]
    min_frequency = float(note_freqs.min()) if len(note_freqs) > 0 else 40.0

    samples: list[np.ndarray] = []
    used_centres: list[float] = []
    for centre in centres:
        energies = _narrow_fft_band_energies(
            audio,
            sample_rate,
            centre,
            note_freqs,
            window_seconds=window_seconds,
            min_frequency=min_frequency,
        )
        if energies is None:
            continue
        samples.append(energies)
        used_centres.append(centre)

    if not samples:
        return _EMPTY_MEASUREMENT

    stacked = np.vstack(samples)
    medians = np.median(stacked, axis=0)
    per_note: dict[str, float] = {}
    for name, energy in zip(note_names, medians):
        if energy > 0.0:
            per_note[name] = float(energy)

    half_window = window_seconds / 2.0
    sampled_regions = tuple(
        (max(centre - half_window, 0.0), min(centre + half_window, audio_duration))
        for centre in used_centres
    )
    return NoiseFloorMeasurement(
        per_note=per_note,
        silent_regions=sampled_regions,
        sample_count=len(samples),
    )
