from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace as dataclass_replace
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from ..models import InstrumentTuning
from . import settings
from .constants import *
from .models import GapAttackCandidates, NoteCandidate, OnsetAttackProfile, OnsetWaveformStats, Segment
from .profiles import (
    GAP_ONSET_KURTOSIS_OVERRIDE_MIN_GAIN,
    GAP_ONSET_MAX_KURTOSIS,
    GAP_ONSET_MAX_POST_CREST,
    GAP_ONSET_MIN_BROADBAND_GAIN,
    GAP_ONSET_MIN_POST_SUSTAIN_RATIO,
    LEADING_GAP_START_MARGIN,
    _lookup_onset_attack_profile,
    filter_gap_onsets_by_attack,
    precompute_onset_attack_profiles,
    precompute_onset_waveform_stats,
    refine_onset_times_by_attack_profile,
)


# LRU cache for librosa results keyed by (audio hash, sample rate, hpss flag).
# Same audio is reprocessed across ablation variants and eval-window/full-audio
# pairs; caching the deterministic librosa outputs avoids several seconds of
# recomputation per variant.
_LIBROSA_CACHE: "OrderedDict[tuple[str, int, bool], dict[str, Any]]" = OrderedDict()
_LIBROSA_CACHE_MAX = 8


# ---------------------------------------------------------------------------
# librosa-derived code (ISC License). The following functions in this module are
# ported from or closely derived from librosa source and are covered by the ISC
# notice below (see also THIRD_PARTY_NOTICES.md at the repository root):
#   _peak_pick_numpy        — port of librosa's __peak_pick kernel (util/utils.py)
#   _onset_detect_numpy     — replacement for librosa.onset.onset_detect
#   _onset_backtrack_numpy  — port of librosa.onset.onset_backtrack
#   _mel_filterbank         — derived from librosa.filters.mel (Slaney norm)
#   _onset_strength_numpy   — derived from librosa.onset.onset_strength
#
# Original source: librosa (https://librosa.org), >= 0.10
# ISC License — Copyright (c) 2013--2023, librosa development team.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# ---------------------------------------------------------------------------


def _peak_pick_numpy(
    x: np.ndarray,
    *,
    pre_max: int = 0,
    post_max: int = 1,
    pre_avg: int = 0,
    post_avg: int = 1,
    delta: float = 0.0,
    wait: int = 0,
) -> np.ndarray:
    """Pure-numpy reimplementation of ``librosa.util.peak_pick``.

    Avoids ``numba.np.ufunc.gufunc`` which intermittently segfaults on
    GitHub Actions (Python 3.12 + numba 0.64).  Semantics match librosa's
    ``sparse=True, axis=-1`` mode: returns a 1-D int array of peak indices.

    Ported from librosa's ``__peak_pick`` guvectorize kernel (utils.py):
    frame 0 is special-cased, then a while-loop scans n=1..N-1 with
    greedy wait-skip.
    """
    sz = len(x)
    if sz == 0:
        return np.array([], dtype=np.intp)

    peaks: list[int] = []

    # Special case: frame 0
    max0 = np.max(x[: min(post_max, sz)])
    avg0 = np.mean(x[: min(post_avg, sz)])
    if x[0] >= max0 and x[0] >= avg0 + delta:
        peaks.append(0)
        n = wait + 1
    else:
        n = 1

    while n < sz:
        lo = max(0, n - pre_max)
        hi = min(n + post_max, sz)
        if x[n] != np.max(x[lo:hi]):
            n += 1
            continue
        avg_lo = max(0, n - pre_avg)
        avg_hi = min(n + post_avg, sz)
        if x[n] < np.mean(x[avg_lo:avg_hi]) + delta:
            n += 1
            continue
        peaks.append(n)
        n += wait + 1

    return np.array(peaks, dtype=np.intp)


def _onset_detect_numpy(
    onset_envelope: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    backtrack: bool = False,
) -> np.ndarray:
    """Pure-numpy replacement for ``librosa.onset.onset_detect``.

    Replaces only the ``peak_pick`` call (the numba gufunc that segfaults
    on GitHub Actions Python 3.12 runners) with ``_peak_pick_numpy``.
    All other steps — max-normalisation and ``onset_backtrack`` — use
    librosa's own implementations, which are numba-free.
    """
    env = onset_envelope.copy()
    env_max = env.max()
    if env_max > 0:
        env /= env_max

    pre_max = int(0.03 * sr // hop_length)
    post_max = int(0.00 * sr // hop_length) + 1
    pre_avg = int(0.10 * sr // hop_length)
    post_avg = int(0.10 * sr // hop_length) + 1
    wait = int(0.03 * sr // hop_length)
    delta = 0.07

    frames = _peak_pick_numpy(
        env,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
    )
    if backtrack and len(frames) > 0:
        frames = _onset_backtrack_numpy(frames, onset_envelope)
    return frames


def _frames_to_time_numpy(frames: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Pure-numpy ``librosa.frames_to_time`` (n_fft=None: no centering offset).

    Verified bit-exact against librosa 0.11 (frames * hop_length / sr)."""
    return np.asarray(frames) * hop_length / float(sr)


def _audio_duration_sec(audio: np.ndarray, sample_rate: int) -> float:
    """Pure-numpy ``librosa.get_duration(y=audio, sr=sr)`` for a time-domain signal."""
    return audio.shape[-1] / float(sample_rate)


def _rms_numpy(audio: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Pure-numpy ``librosa.feature.rms`` (center=True, pad_mode='constant').

    Verified against librosa 0.11 to within float32 epsilon (~2e-7) on fixture
    audio; the fixture regression suite is the authoritative equivalence gate."""
    pad = frame_length // 2
    padded = np.pad(audio, pad, mode="constant")
    n_frames = 1 + (len(padded) - frame_length) // hop_length
    indices = np.arange(frame_length)[:, None] + hop_length * np.arange(n_frames)[None, :]
    frames = padded[indices]
    power = np.mean(np.abs(frames) ** 2, axis=0)
    return np.sqrt(power).astype(np.float32)


def _onset_backtrack_numpy(events: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Pure-numpy ``librosa.onset.onset_backtrack``.

    Snaps each onset event back to the nearest preceding local minimum of the
    energy envelope (energy[i] <= energy[i-1] and energy[i] < energy[i+1]), with
    frame 0 always available as a fallback. Verified frame-exact against
    librosa 0.11."""
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:])) + 1
    minima = np.unique(np.concatenate([[0], minima]))
    idx = np.clip(np.searchsorted(minima, events, side="right") - 1, 0, len(minima) - 1)
    return minima[idx]


@lru_cache(maxsize=8)
def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Slaney-normalised mel filterbank, bit-exact to librosa.filters.mel
    (fmin=0, fmax=sr/2, htk=False, norm='slaney'). Cached per (sr, n_fft, n_mels).
    Verified against librosa 0.11 to ~2.6e-9 (float rounding)."""
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    logstep = np.log(6.4) / 27.0
    min_log_mel = min_log_hz / f_sp

    def hz_to_mel(freqs: np.ndarray) -> np.ndarray:
        freqs = np.asarray(freqs, dtype=float)
        mels = freqs / f_sp
        log_region = freqs >= min_log_hz
        mels[log_region] = min_log_mel + np.log(freqs[log_region] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        mels = np.asarray(mels, dtype=float)
        freqs = f_sp * mels
        log_region = mels >= min_log_mel
        freqs[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
        return freqs

    mel_min, mel_max = hz_to_mel(np.array([0.0, sample_rate / 2.0]))
    mel_f = mel_to_hz(np.linspace(mel_min, mel_max, n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        weights[i] = np.maximum(0.0, np.minimum(-ramps[i] / fdiff[i], ramps[i + 2] / fdiff[i + 1]))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def _onset_strength_numpy(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int,
    n_fft: int = FRAME_LENGTH,
    n_mels: int = 128,
) -> np.ndarray:
    """Pure-numpy ``librosa.onset.onset_strength`` (mel spectral flux), default path.

    power mel-spectrogram -> power_to_db (ref=1.0, amin=1e-10, top_db=80) ->
    lag-1 positive difference -> mean over mel bands -> left-pad by
    lag + n_fft//(2*hop), trimmed to frame count. Verified equivalent to
    librosa 0.11 within float32 epsilon (~4e-6, frame count exact) on fixture
    audio at hop 256/1024 -- peak-picked onset frames are identical."""
    audio = np.asarray(audio, dtype=np.float32)
    pad = n_fft // 2
    padded = np.pad(audio, pad, mode="constant")
    n_frames = 1 + (len(padded) - n_fft) // hop_length
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    indices = np.arange(n_fft)[:, None] + hop_length * np.arange(n_frames)[None, :]
    frames = padded[indices] * window[:, None]
    power = (np.abs(np.fft.rfft(frames, axis=0)) ** 2).astype(np.float32)
    mel = _mel_filterbank(sample_rate, n_fft, n_mels) @ power
    log_mel = 10.0 * np.log10(np.maximum(1e-10, mel))
    log_mel = np.maximum(log_mel, log_mel.max() - 80.0)
    onset_env = np.maximum(0.0, log_mel[:, 1:] - log_mel[:, :-1]).mean(axis=0)
    pad_width = 1 + n_fft // (2 * hop_length)
    onset_env = np.pad(onset_env, (pad_width, 0), mode="constant")
    return onset_env[: power.shape[1]].astype(np.float32)


def _compute_librosa_features(
    audio: np.ndarray, sample_rate: int, use_hpss_onset: bool
) -> dict[str, Any]:
    """Run librosa rms/onset routines and return cacheable outputs."""
    rms = _rms_numpy(audio, FRAME_LENGTH, HOP_LENGTH)
    frame_times = _frames_to_time_numpy(np.arange(len(rms)), sample_rate, HOP_LENGTH)
    onset_env = _onset_strength_numpy(audio, sample_rate, HOP_LENGTH)
    onset_frames = _onset_detect_numpy(
        onset_env, sample_rate, HOP_LENGTH, backtrack=True,
    )
    if use_hpss_onset:
        # Lazy import: HPSS is the only remaining librosa dependency, gated behind
        # this default-off research flag (#148). Importing it here keeps the
        # module's primary (default) path fully librosa-free for WASM portability.
        # Remaining #193 work: port hpss to numpy or drop the flag entirely.
        import librosa

        _, percussive = librosa.effects.hpss(
            audio, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH,
        )
        perc_env = _onset_strength_numpy(percussive, sample_rate, HOP_LENGTH)
        perc_frames = _onset_detect_numpy(
            perc_env, sample_rate, HOP_LENGTH, backtrack=True,
        )
        onset_frames = np.unique(np.concatenate([onset_frames, perc_frames]))
    return {"rms": rms, "frame_times": frame_times, "onset_frames": onset_frames}


def _get_cached_librosa_features(
    audio: np.ndarray, sample_rate: int, use_hpss_onset: bool
) -> dict[str, Any]:
    """Return cached librosa outputs for ``audio``, computing them on miss."""
    contiguous = np.ascontiguousarray(audio)
    audio_hash = hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()[:16]
    key = (audio_hash, int(sample_rate), bool(use_hpss_onset))
    cached = _LIBROSA_CACHE.get(key)
    if cached is not None:
        _LIBROSA_CACHE.move_to_end(key)
        return cached
    # Pass the contiguous buffer (not the original `audio`) so the bytes that
    # were hashed and the bytes librosa consumes are guaranteed to be the same,
    # and any non-contiguous input is converted exactly once.
    features = _compute_librosa_features(contiguous, sample_rate, use_hpss_onset)
    _LIBROSA_CACHE[key] = features
    if len(_LIBROSA_CACHE) > _LIBROSA_CACHE_MAX:
        _LIBROSA_CACHE.popitem(last=False)
    return features


def merge_time_ranges(ranges: list[tuple[float, float]], gap_tolerance: float = 0.06) -> list[tuple[float, float]]:
    if not ranges:
        return []

    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + gap_tolerance:
            merged[-1] = (previous_start, max(previous_end, end))
            continue
        merged.append((start, end))

    return merged


def _segment_leaves(seg: Segment) -> tuple[Segment, ...]:
    """Return leaf segments (originals before any merge)."""
    return seg.merged_from if seg.merged_from else (seg,)


def _merge_segments(a: Segment, b: Segment, start: float, end: float, reason: str = "") -> Segment:
    """Create a merged segment preserving provenance from both inputs."""
    if end == a.end_time and end != b.end_time:
        merged_end_estimated = a.end_estimated
    elif end == b.end_time and end != a.end_time:
        merged_end_estimated = b.end_estimated
    else:
        merged_end_estimated = a.end_estimated or b.end_estimated
    # Preserve per-note provenance: keep if both agree or only one is set
    confirmed = a.confirmed_primary if a.confirmed_primary == b.confirmed_primary else (a.confirmed_primary or b.confirmed_primary)
    hint = a.hint_primary if a.hint_primary == b.hint_primary else (a.hint_primary or b.hint_primary)
    return Segment(
        start_time=start,
        end_time=end,
        sources=a.sources | b.sources,
        merged_from=_segment_leaves(a) + _segment_leaves(b),
        merge_reason=reason,
        end_estimated=merged_end_estimated,
        confirmed_primary=confirmed,
        hint_primary=hint,
    )


def dedupe_nested_segments(segments: list[Segment]) -> list[Segment]:
    if len(segments) < 2:
        return segments

    deduped: list[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start_time, s.end_time)):
        if deduped:
            prev = deduped[-1]
            same_start = abs(seg.start_time - prev.start_time) <= NESTED_SEGMENT_DEDUP_MAX_START_DELTA
            if same_start:
                if seg.end_time <= prev.end_time:
                    deduped[-1] = _merge_segments(prev, seg, prev.start_time, prev.end_time, reason="nested")
                    continue
                deduped[-1] = _merge_segments(prev, seg, prev.start_time, seg.end_time, reason="nested")
                continue
        deduped.append(seg)

    return deduped


def dedupe_cross_collector_segments(segments: list[Segment]) -> list[Segment]:
    if len(segments) < 2:
        return segments

    deduped: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = deduped[-1]
        overlap = min(prev.end_time, seg.end_time) - max(prev.start_time, seg.start_time)
        if overlap > 0:
            shorter_duration = min(prev.end_time - prev.start_time, seg.end_time - seg.start_time)
            if shorter_duration > 0 and overlap >= shorter_duration * CROSS_COLLECTOR_DEDUP_MIN_OVERLAP_RATIO:
                if prev.end_estimated and not seg.end_estimated:
                    trimmed_end = seg.start_time
                    if trimmed_end - prev.start_time >= 0.08:
                        deduped[-1] = dataclass_replace(prev, end_time=trimmed_end, end_estimated=False, trimmed_from=prev)
                        deduped.append(seg)
                    else:
                        deduped[-1] = seg
                elif seg.end_estimated and not prev.end_estimated:
                    trimmed_start_for_seg = prev.end_time
                    if seg.end_time - trimmed_start_for_seg >= 0.08:
                        deduped.append(dataclass_replace(seg, start_time=trimmed_start_for_seg, trimmed_from=seg))
                else:
                    deduped[-1] = _merge_segments(
                        prev,
                        seg,
                        min(prev.start_time, seg.start_time),
                        max(prev.end_time, seg.end_time),
                        reason="cross_collector_overlap",
                    )
                continue
        deduped.append(seg)
    return deduped


def trim_small_overlapping_segments(segments: list[Segment]) -> list[Segment]:
    if len(segments) < 2:
        return segments

    trimmed: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = trimmed[-1]
        overlap = prev.end_time - seg.start_time
        duration = seg.end_time - seg.start_time
        if (
            overlap > 0
            and overlap <= SEGMENT_OVERLAP_TRIM_MAX_OVERLAP
            and duration >= SEGMENT_OVERLAP_TRIM_MIN_DURATION
        ):
            adjusted_start = prev.end_time
            if seg.end_time - adjusted_start >= 0.08:
                trimmed.append(dataclass_replace(seg, start_time=adjusted_start))
                continue
        trimmed.append(seg)

    return trimmed


def should_keep_dense_trailing_onset(
    boundary_times: list[float],
    index: int,
    range_start: float,
    range_end: float,
) -> bool:
    if index <= 0 or index >= len(boundary_times) - 1:
        return False

    previous_time = boundary_times[index - 1]
    current_time = boundary_times[index]
    next_time = boundary_times[index + 1]
    return (
        len(boundary_times) >= 4
        and range_end - range_start >= 2.5
        and current_time - previous_time >= 0.1
        and current_time - previous_time < 0.18
        and next_time - current_time >= 0.28
        and current_time - range_start >= 1.5
        and range_end - current_time <= 1.0
        and range_end - current_time >= 0.18
    )


def should_keep_short_range_trailing_onset(
    boundary_times: list[float],
    index: int,
    range_start: float,
    range_end: float,
) -> bool:
    if index < 1 or index != len(boundary_times) - 1:
        return False

    previous_time = boundary_times[index - 1]
    current_time = boundary_times[index]
    previous_previous_time = range_start if index == 1 else boundary_times[index - 2]
    range_duration = range_end - range_start
    return (
        0.9 <= range_duration <= 1.5
        and current_time - previous_time >= 0.1
        and current_time - previous_time <= 0.16
        and previous_time - previous_previous_time >= 0.25
        and previous_time - previous_previous_time <= 0.45
        and range_end - current_time >= 0.28
        and range_end - current_time <= 0.45
    )


def should_snap_range_start_to_first_onset(
    range_start: float,
    first_onset: float,
    onset_profiles: dict[float, OnsetAttackProfile],
) -> bool:
    if first_onset - range_start > ATTACK_REFINED_ONSET_MAX_INTERVAL:
        return False
    first_profile = _lookup_onset_attack_profile(onset_profiles, first_onset)
    return first_profile is not None and first_profile.is_valid_attack


def collapse_active_range_head_onsets(
    effective_range_start: float,
    range_end: float,
    range_onsets: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
) -> list[float]:
    if not range_onsets:
        return range_onsets
    if range_end - effective_range_start < ACTIVE_RANGE_HEAD_CLUSTER_MIN_DURATION:
        return range_onsets

    head_cluster: list[float] = []
    previous_time = effective_range_start
    for onset_time in range_onsets:
        if onset_time - effective_range_start > ACTIVE_RANGE_HEAD_CLUSTER_MAX_OFFSET:
            break
        if onset_time - previous_time > ACTIVE_RANGE_HEAD_CLUSTER_MAX_INTERVAL:
            break
        profile = _lookup_onset_attack_profile(onset_profiles, onset_time)
        if profile is None:
            break
        if not profile.is_valid_attack and not head_cluster:
            break
        head_cluster.append(onset_time)
        previous_time = onset_time

    anchor_profile = _lookup_onset_attack_profile(onset_profiles, effective_range_start)
    anchor_valid = anchor_profile is not None and anchor_profile.is_valid_attack
    if len(head_cluster) + (1 if anchor_valid else 0) < 2:
        return range_onsets
    if len(range_onsets) > 3:
        return range_onsets

    head_cluster_set = {round(onset_time, 4) for onset_time in head_cluster}
    return [onset_time for onset_time in range_onsets if round(onset_time, 4) not in head_cluster_set]


def collect_prior_backtrack_onsets(
    range_start: float,
    previous_range_end: float | None,
    backtrack_onset_times: list[float],
) -> list[float]:
    return [
        time
        for time in backtrack_onset_times
        if range_start - PRIOR_ONSET_BACKTRACK_SECONDS <= time <= range_start + 0.005
        and (previous_range_end is None or time >= previous_range_end + 0.005)
    ]


def collect_range_prior_backtrack_onsets(
    range_start: float,
    range_end: float,
    previous_range_end: float | None,
    onset_times: list[float],
    filtered_backtrack_onset_times: list[float] | None = None,
) -> list[float]:
    if range_end - range_start <= LONG_RANGE_BACKTRACK_MIN_DURATION:
        return collect_prior_backtrack_onsets(range_start, previous_range_end, onset_times)

    backtrack_source = filtered_backtrack_onset_times if filtered_backtrack_onset_times is not None else onset_times
    return collect_prior_backtrack_onsets(range_start, previous_range_end, backtrack_source)


def _valid_attack_gap_onsets(
    gap_start: float,
    gap_end: float,
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
    start_margin: float = 0.05,
    waveform_stats: dict[float, OnsetWaveformStats] | None = None,
) -> list[float]:
    valid: list[float] = []
    for time in onset_times:
        if not (gap_start + start_margin < time < gap_end - 0.05):
            continue
        profile = onset_profiles.get(round(time, 4))
        if profile is None or not profile.is_valid_attack:
            continue
        if profile.broadband_onset_gain < GAP_ONSET_MIN_BROADBAND_GAIN:
            continue
        if waveform_stats is not None:
            ws = waveform_stats.get(round(time, 4))
            if ws is not None:
                strong_attack = profile.broadband_onset_gain >= GAP_ONSET_KURTOSIS_OVERRIDE_MIN_GAIN
                if GAP_ONSET_MAX_KURTOSIS > 0 and ws.kurtosis > GAP_ONSET_MAX_KURTOSIS and not strong_attack:
                    continue
                if GAP_ONSET_MAX_POST_CREST > 0 and ws.crest > GAP_ONSET_MAX_POST_CREST:
                    continue
                if GAP_ONSET_MIN_POST_SUSTAIN_RATIO > 0 and ws.post_sustain_ratio < GAP_ONSET_MIN_POST_SUSTAIN_RATIO:
                    continue
        valid.append(time)
    return valid


def collect_attack_validated_gap_candidates(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
    audio_duration: float,
    waveform_stats: dict[float, OnsetWaveformStats] | None = None,
) -> GapAttackCandidates:
    inter_ranges: list[list[float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        inter_ranges.append(
            _valid_attack_gap_onsets(previous_end, next_start, onset_times, onset_profiles, waveform_stats=waveform_stats)
        )

    leading: list[float] = []
    trailing: list[float] = []
    if active_ranges:
        leading = _valid_attack_gap_onsets(
            0.0,
            active_ranges[0][0],
            onset_times,
            onset_profiles,
            start_margin=LEADING_GAP_START_MARGIN,
            waveform_stats=waveform_stats,
        )
        trailing = _valid_attack_gap_onsets(
            active_ranges[-1][1],
            audio_duration + 0.06,
            onset_times,
            onset_profiles,
            waveform_stats=waveform_stats,
        )

    return GapAttackCandidates(inter_ranges=inter_ranges, leading=leading, trailing=trailing)


CANDIDATE_PROMOTION_MIN_CANDIDATES = 1
CANDIDATE_PROMOTION_MIN_EDGE_DISTANCE = 0.3
CANDIDATE_PROMOTION_MIN_SEGMENT_DURATION = 0.08
CANDIDATE_PROMOTION_SEGMENT_DURATION = 0.32
CANDIDATE_PROMOTION_MAX_SEGMENT_DURATION = 0.8
CANDIDATE_PROMOTION_CLUSTER_MAX_INTERVAL = 0.1


def _cluster_gap_candidates(candidate_onsets: list[float]) -> list[float]:
    if not candidate_onsets:
        return []
    clusters: list[list[float]] = [[candidate_onsets[0]]]
    for onset_time in candidate_onsets[1:]:
        if onset_time - clusters[-1][-1] <= CANDIDATE_PROMOTION_CLUSTER_MAX_INTERVAL:
            clusters[-1].append(onset_time)
        else:
            clusters.append([onset_time])
    return [cluster[-1] for cluster in clusters]


def _promote_gap_candidates_by_structure(
    candidate_onsets: list[float],
    gap_start: float,
    gap_end: float,
) -> list[tuple[float, float]]:
    clustered = _cluster_gap_candidates(candidate_onsets)
    if len(clustered) < CANDIDATE_PROMOTION_MIN_CANDIDATES:
        return []

    eligible = [
        onset_time
        for onset_time in clustered
        if onset_time - gap_start >= CANDIDATE_PROMOTION_MIN_EDGE_DISTANCE
        and gap_end - onset_time >= CANDIDATE_PROMOTION_MIN_EDGE_DISTANCE
    ]
    if len(eligible) < CANDIDATE_PROMOTION_MIN_CANDIDATES:
        return []

    segments: list[tuple[float, float]] = []
    for index, onset_time in enumerate(eligible):
        if index + 1 < len(eligible):
            end_time = min(eligible[index + 1], onset_time + CANDIDATE_PROMOTION_MAX_SEGMENT_DURATION)
        else:
            end_time = min(onset_time + CANDIDATE_PROMOTION_SEGMENT_DURATION, gap_end - 0.08)
        if end_time - onset_time >= CANDIDATE_PROMOTION_MIN_SEGMENT_DURATION:
            segments.append((onset_time, end_time))

    return segments


def collect_multi_onset_gap_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
    gap_attack_candidates: GapAttackCandidates | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        gap_duration = next_start - previous_end
        if gap_duration < MULTI_ONSET_GAP_MIN_DURATION:
            continue

        gap_onsets = [time for time in onset_times if previous_end + 0.05 < time < next_start - 0.05]
        if len(gap_onsets) < 3:
            continue

        if (
            gap_onsets[0] - previous_end < MULTI_ONSET_GAP_MIN_EDGE_SPACING
            or next_start - gap_onsets[-1] < MULTI_ONSET_GAP_MIN_EDGE_SPACING
        ):
            continue

        intervals = [gap_onsets[i + 1] - gap_onsets[i] for i in range(len(gap_onsets) - 1)]
        short_interval_count = sum(
            1
            for interval in intervals
            if MULTI_ONSET_GAP_MIN_INTERVAL <= interval <= MULTI_ONSET_GAP_MAX_INTERVAL
        )
        if short_interval_count >= MULTI_ONSET_GAP_MIN_SHORT_INTERVALS:
            for start_time, end_time in zip(gap_onsets, gap_onsets[1:]):
                if end_time - start_time >= 0.08:
                    segments.append((start_time, end_time))

            trailing_gap = next_start - gap_onsets[-1]
            if trailing_gap > PRIOR_ONSET_BACKTRACK_SECONDS and trailing_gap >= 0.08:
                segments.append((gap_onsets[-1], next_start))
            continue

        candidate_onsets = (
            gap_attack_candidates.inter_ranges[index]
            if gap_attack_candidates is not None and index < len(gap_attack_candidates.inter_ranges)
            else []
        )
        if candidate_onsets:
            segments.extend(_promote_gap_candidates_by_structure(candidate_onsets, previous_end, next_start))

    return segments




def collect_sparse_gap_tail_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start = active_ranges[index + 1][0]
        gap_duration = next_start - previous_end
        if gap_duration < SPARSE_GAP_TAIL_MIN_DURATION:
            continue

        gap_onsets = [
            onset_time
            for onset_time in onset_times
            if previous_end + SPARSE_GAP_TAIL_MIN_PREVIOUS_EDGE < onset_time < next_start - 0.05
        ]
        early_gap_onsets = [
            onset_time
            for onset_time in gap_onsets
            if onset_time - previous_end <= SPARSE_GAP_TAIL_MAX_ONSET_OFFSET
        ]
        if not (1 <= len(early_gap_onsets) <= 2):
            continue
        if next_start - early_gap_onsets[-1] < SPARSE_GAP_TAIL_MIN_TRAILING_EDGE:
            continue

        if len(early_gap_onsets) == 1:
            orphan_start = early_gap_onsets[0]
            orphan_end = min(orphan_start + SPARSE_GAP_TAIL_SEGMENT_DURATION, next_start - 0.08)
            if orphan_end - orphan_start >= 0.08:
                segments.append((orphan_start, orphan_end))
            continue

        onset_interval = early_gap_onsets[1] - early_gap_onsets[0]
        if not (SPARSE_GAP_TAIL_MIN_INTERVAL <= onset_interval <= SPARSE_GAP_TAIL_MAX_INTERVAL):
            continue

        if early_gap_onsets[1] - early_gap_onsets[0] >= 0.08:
            segments.append((early_gap_onsets[0], early_gap_onsets[1]))
        orphan_start = early_gap_onsets[1]
        orphan_end = min(orphan_start + SPARSE_GAP_TAIL_SEGMENT_DURATION, next_start - 0.08)
        if orphan_end - orphan_start >= 0.08:
            segments.append((orphan_start, orphan_end))

    return segments



def collect_attack_validated_gap_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile],
    audio_duration: float,
    gap_attack_candidates: GapAttackCandidates | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    candidates = gap_attack_candidates or collect_attack_validated_gap_candidates(
        active_ranges,
        onset_times,
        onset_profiles,
        audio_duration,
    )

    for index in range(len(active_ranges) - 1):
        next_start = active_ranges[index + 1][0]
        valid_onsets = candidates.inter_ranges[index] if index < len(candidates.inter_ranges) else []
        for onset_index, onset_time in enumerate(valid_onsets):
            if onset_index + 1 < len(valid_onsets):
                end_time = min(valid_onsets[onset_index + 1], onset_time + CANDIDATE_PROMOTION_MAX_SEGMENT_DURATION)
            else:
                end_time = min(onset_time + ATTACK_VALIDATED_GAP_SEGMENT_DURATION, next_start)
            if end_time - onset_time >= 0.08:
                segments.append((onset_time, end_time))

    if active_ranges:
        first_start = active_ranges[0][0]
        for onset_index, onset_time in enumerate(candidates.leading):
            if onset_index + 1 < len(candidates.leading):
                end_time = min(candidates.leading[onset_index + 1], onset_time + CANDIDATE_PROMOTION_MAX_SEGMENT_DURATION)
            else:
                end_time = min(onset_time + ATTACK_VALIDATED_GAP_SEGMENT_DURATION, first_start)
            if end_time - onset_time >= 0.08:
                segments.append((onset_time, end_time))

    if active_ranges:
        for onset_index, onset_time in enumerate(candidates.trailing):
            if onset_index + 1 < len(candidates.trailing):
                end_time = min(candidates.trailing[onset_index + 1], onset_time + CANDIDATE_PROMOTION_MAX_SEGMENT_DURATION)
            else:
                end_time = min(onset_time + ATTACK_VALIDATED_GAP_SEGMENT_DURATION, audio_duration)
            if end_time - onset_time >= 0.08:
                segments.append((onset_time, end_time))

    return segments


def build_gap_ioi_diagnostics(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start, next_end = active_ranges[index + 1]
        gap_onsets = [onset_time for onset_time in onset_times if previous_end + 0.05 < onset_time < next_start - 0.05]
        if not gap_onsets:
            continue

        previous_context = [onset_time for onset_time in onset_times if onset_time < gap_onsets[0]]
        next_context = [onset_time for onset_time in onset_times if onset_time > gap_onsets[-1]]
        previous_interval = gap_onsets[0] - previous_context[-1] if previous_context else None
        next_interval = next_context[0] - gap_onsets[-1] if next_context else None
        diagnostics.append(
            {
                "previousEnd": round(previous_end, 4),
                "nextStart": round(next_start, 4),
                "nextEnd": round(next_end, 4),
                "gapDuration": round(next_start - previous_end, 4),
                "nextRangeDuration": round(next_end - next_start, 4),
                "gapOnsets": [round(onset_time, 4) for onset_time in gap_onsets],
                "previousInterval": None if previous_interval is None else round(previous_interval, 4),
                "nextInterval": None if next_interval is None else round(next_interval, 4),
                "previousEdgeDistance": round(gap_onsets[0] - previous_end, 4),
                "nextEdgeDistance": round(next_start - gap_onsets[-1], 4),
            }
        )
    return diagnostics


def simplify_sparse_gap_tail_high_octave_dyad(candidates: list[NoteCandidate]) -> list[NoteCandidate]:
    if len(candidates) != 2:
        return candidates

    ordered = sorted(candidates, key=lambda candidate: candidate.frequency)
    lower, upper = ordered
    if upper.octave < 6:
        return candidates
    if lower.pitch_class != upper.pitch_class:
        return candidates
    if upper.octave - lower.octave != 1:
        return candidates
    return [upper]


def suppress_short_bridge_active_ranges(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(active_ranges) < 3:
        return active_ranges, []

    filtered_ranges: list[tuple[float, float]] = []
    suppressed_ranges: list[tuple[float, float]] = []

    for index, current_range in enumerate(active_ranges):
        if index == 0 or index == len(active_ranges) - 1:
            filtered_ranges.append(current_range)
            continue

        current_start, current_end = current_range
        current_duration = current_end - current_start
        if current_duration > SHORT_BRIDGE_ACTIVE_RANGE_MAX_DURATION:
            filtered_ranges.append(current_range)
            continue

        current_onsets = [time for time in onset_times if current_start <= time <= current_end]
        if len(current_onsets) != 1:
            filtered_ranges.append(current_range)
            continue

        current_onset = current_onsets[0]
        if current_onset - current_start > SHORT_BRIDGE_ACTIVE_RANGE_MAX_ONSET_OFFSET:
            filtered_ranges.append(current_range)
            continue

        next_start, _ = active_ranges[index + 1]
        next_prior_onsets = [
            time
            for time in onset_times
            if next_start - PRIOR_ONSET_BACKTRACK_SECONDS <= time <= next_start + 0.005
            and time >= current_end + 0.005
        ]
        if not next_prior_onsets:
            filtered_ranges.append(current_range)
            continue

        next_backtracked_onset = next_prior_onsets[-1]
        onset_gap = next_backtracked_onset - current_onset
        edge_gap = next_backtracked_onset - current_end
        if not (
            SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_ONSET_GAP
            <= onset_gap
            <= SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_ONSET_GAP
            and SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_EDGE_GAP <= edge_gap <= SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_EDGE_GAP
        ):
            filtered_ranges.append(current_range)
            continue

        suppressed_ranges.append(current_range)

    return filtered_ranges, suppressed_ranges


def should_keep_low_register_sparse_gap_tail(
    candidates: list[NoteCandidate],
    tuning: InstrumentTuning,
    descending_primary_suffix_floor: float | None,
    descending_primary_suffix_note_names: frozenset[str],
) -> bool:
    if len(candidates) != 1 or descending_primary_suffix_floor is None:
        return False

    candidate = candidates[0]
    if candidate.octave >= 6:
        return False
    if candidate.note_name in descending_primary_suffix_note_names:
        return False

    sorted_notes = sorted(tuning.notes, key=lambda item: item.frequency)
    rank_by_name = {note.note_name: index for index, note in enumerate(sorted_notes)}
    suffix_floor_name = next(
        (note.note_name for note in sorted_notes if abs(note.frequency - descending_primary_suffix_floor) < 1e-6),
        None,
    )
    candidate_rank = rank_by_name.get(candidate.note_name)
    suffix_rank = rank_by_name.get(suffix_floor_name) if suffix_floor_name is not None else None
    if candidate_rank is None or suffix_rank is None:
        return False
    return candidate_rank == suffix_rank - 1


def _active_range_debug_context(
    range_index: int,
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    backtrack_onset_times: list[float] | None = None,
) -> dict[str, Any]:
    range_start, range_end = active_ranges[range_index]
    effective_range_start = range_start
    previous_range_end = active_ranges[range_index - 1][1] if range_index > 0 else None
    prior_onsets = collect_range_prior_backtrack_onsets(
        range_start,
        range_end,
        previous_range_end,
        onset_times,
        backtrack_onset_times,
    )
    if prior_onsets:
        effective_range_start = prior_onsets[-1]
        if (
            previous_range_end is not None
            and range_start - previous_range_end >= ACTIVE_RANGE_START_CLUSTER_MIN_GAP
            and range_end - range_start <= ACTIVE_RANGE_START_CLUSTER_MAX_DURATION
        ):
            trailing_cluster = [
                time for time in prior_onsets if effective_range_start - time <= ACTIVE_RANGE_START_CLUSTER_MAX_SPAN
            ]
            if len(trailing_cluster) >= 2:
                effective_range_start = trailing_cluster[0]

    range_onsets = [time for time in onset_times if effective_range_start + 0.005 < time < range_end - 0.05]
    return {
        "activeRangeStart": round(range_start, 4),
        "activeRangeEnd": round(range_end, 4),
        "backtrackedStartTime": round(effective_range_start, 4),
        "activeRangeOnsets": [round(time, 4) for time in range_onsets],
        "activeRangeOnsetCount": len(range_onsets),
    }


def build_segment_debug_contexts(
    segments: list[Segment],
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    backtrack_onset_times: list[float] | None = None,
) -> dict[tuple[float, float], dict[str, Any]]:
    active_contexts = [
        _active_range_debug_context(index, active_ranges, onset_times, backtrack_onset_times)
        for index in range(len(active_ranges))
    ]
    segment_contexts: dict[tuple[float, float], dict[str, Any]] = {}
    for index, (start_time, end_time) in enumerate(segments):
        segment_key = (round(start_time, 4), round(end_time, 4))
        segment_onsets = [time for time in onset_times if start_time <= time <= end_time]
        context = {
            "previousGapSec": None if index == 0 else round(start_time - segments[index - 1][1], 4),
            "nextGapSec": None if index + 1 >= len(segments) else round(segments[index + 1][0] - end_time, 4),
            "segmentOnsets": [round(time, 4) for time in segment_onsets],
            "localOnsetCount": len(segment_onsets),
        }
        active_index = next(
            (
                range_index
                for range_index, (range_start, range_end) in enumerate(active_ranges)
                if start_time < range_end and end_time > range_start
            ),
            None,
        )
        if active_index is not None:
            context.update(active_contexts[active_index])
        else:
            context.update(
                {
                    "activeRangeStart": None,
                    "activeRangeEnd": None,
                    "backtrackedStartTime": None,
                    "activeRangeOnsets": [],
                    "activeRangeOnsetCount": 0,
                }
            )
        segment_contexts[segment_key] = context
    return segment_contexts


def _estimate_tempo_autocorr(
    onset_env: np.ndarray,
    sample_rate: int,
    hop_length: int,
) -> float:
    """Estimate global tempo (BPM) via onset-envelope autocorrelation.

    Pure-numpy replacement for ``librosa.beat.beat_track`` — avoids the
    numba gufunc that intermittently segfaults on GitHub Actions runners
    (see ``numba/np/ufunc/gufunc.py:263`` in CI logs).

    Known limitations vs librosa.beat_track (which uses a tempogram +
    dynamic-programming beat tracker):

    * Sub-harmonic / octave ambiguity: simple global autocorrelation can
      lock onto a sub-harmonic of the true tempo (e.g., 40 BPM when the
      real tempo is ~128 BPM).  librosa's DP tracker avoids this by
      penalising tempo deviations frame-to-frame.
    * No beat positions: we only extract BPM; beat positions are discarded
      by the caller anyway (``detect_segments`` uses only ``tempo``).

    These limitations are acceptable because:

    1. Tempo is used *only* for ``startBeat`` rendering (beat-time
       quantisation of events).  Note detection is unaffected.
    2. No test asserts exact ``startBeat`` values; the only assertion is
       ``30 <= tempo <= 300``.
    3. Streaming transcription (#141 / AGENTS.md vision) will redesign
       tempo estimation entirely (online tracker or post-hoc), so this
       batch implementation is intentionally minimal.
    """
    n = len(onset_env)
    if n < 2:
        return TEMPO_FALLBACK_BPM
    oe = onset_env - onset_env.mean()
    if oe.std() < 1e-8:
        return TEMPO_FALLBACK_BPM
    spec = np.fft.rfft(oe, n=2 * n)
    acf = np.fft.irfft(np.abs(spec) ** 2)[:n]

    lag_min = max(1, int(np.ceil(60.0 / TEMPO_BPM_MAX * sample_rate / hop_length)))
    lag_max = min(int(np.floor(60.0 / TEMPO_BPM_MIN * sample_rate / hop_length)), n - 1)
    if lag_max <= lag_min:
        return TEMPO_FALLBACK_BPM

    peak_lag = lag_min + int(np.argmax(acf[lag_min : lag_max + 1]))
    if peak_lag <= 0:
        return TEMPO_FALLBACK_BPM
    bpm = 60.0 * sample_rate / (hop_length * peak_lag)
    return max(bpm, 1.0)


@dataclass(frozen=True, slots=True)
class SegmentDetectionResult:
    segments: list[Segment]
    tempo: float
    debug: dict[str, Any]


def detect_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    mid_performance_start: bool = False,
    mid_performance_end: bool = False,
) -> SegmentDetectionResult:
    cfg = settings.get()
    cached_features = _get_cached_librosa_features(audio, sample_rate, cfg.use_hpss_onset)
    rms = cached_features["rms"]
    frame_times = cached_features["frame_times"]
    max_rms = float(np.max(rms))
    median_rms = float(np.median(rms))
    threshold = max(max_rms * 0.18, min(median_rms * 2.2, max_rms * RMS_MEDIAN_THRESHOLD_MAX_PEAK_RATIO), 0.01)
    active_frames = rms >= threshold

    active_ranges: list[tuple[float, float]] = []
    active_start = None
    for index, is_active in enumerate(active_frames):
        if is_active and active_start is None:
            active_start = index
        elif not is_active and active_start is not None:
            start_time = max(float(frame_times[active_start]) - 0.02, 0.0)
            end_time = float(frame_times[min(index, len(frame_times) - 1)]) + 0.08
            active_ranges.append((start_time, end_time))
            active_start = None

    if active_start is not None:
        active_ranges.append((max(float(frame_times[active_start]) - 0.02, 0.0), _audio_duration_sec(audio, sample_rate)))

    raw_active_ranges = active_ranges.copy()
    active_ranges = merge_time_ranges(active_ranges)

    onset_frames = cached_features["onset_frames"]
    onset_times = [float(value) for value in _frames_to_time_numpy(onset_frames, sample_rate, HOP_LENGTH)]
    onset_attack_profiles = precompute_onset_attack_profiles(audio, sample_rate, onset_times)
    onset_times = refine_onset_times_by_attack_profile(onset_times, onset_attack_profiles)
    onset_waveform_stats = (
        precompute_onset_waveform_stats(audio, sample_rate, onset_times)
        if cfg.filter_gap_onsets_by_attack_profile or cfg.use_attack_validated_gap_collector
        else {}
    )
    active_ranges, short_bridge_active_ranges = suppress_short_bridge_active_ranges(active_ranges, onset_times)
    gap_onset_times = (
        filter_gap_onsets_by_attack(onset_times, active_ranges, onset_attack_profiles, onset_waveform_stats)
        if cfg.filter_gap_onsets_by_attack_profile
        else onset_times
    )
    gap_ioi_diagnostics = build_gap_ioi_diagnostics(active_ranges, onset_times)

    audio_duration = float(_audio_duration_sec(audio, sample_rate))
    gap_onset_keys = {round(onset_time, 4) for onset_time in gap_onset_times}
    waveform_stats = {
        onset_time: stats
        for onset_time, stats in onset_waveform_stats.items()
        if onset_time in gap_onset_keys
    }
    attack_validated_gap_candidates = collect_attack_validated_gap_candidates(
        active_ranges,
        gap_onset_times,
        onset_attack_profiles,
        audio_duration,
    )
    if mid_performance_start or mid_performance_end:
        attack_validated_gap_candidates = dataclass_replace(
            attack_validated_gap_candidates,
            **({"leading": []} if mid_performance_start else {}),
            **({"trailing": []} if mid_performance_end else {}),
        )
    filtered_gap_candidates = (
        collect_attack_validated_gap_candidates(
            active_ranges,
            gap_onset_times,
            onset_attack_profiles,
            audio_duration,
            waveform_stats=waveform_stats,
        )
        if cfg.use_attack_validated_gap_collector
        else None
    )
    if filtered_gap_candidates is not None and (mid_performance_start or mid_performance_end):
        filtered_gap_candidates = dataclass_replace(
            filtered_gap_candidates,
            **({"leading": []} if mid_performance_start else {}),
            **({"trailing": []} if mid_performance_end else {}),
        )

    multi_onset_gap_segments = (
        [] if cfg.ablate_multi_onset_gap else collect_multi_onset_gap_segments(active_ranges, gap_onset_times, onset_attack_profiles, attack_validated_gap_candidates)
    )
    sparse_gap_tail_segments = (
        [] if cfg.ablate_sparse_gap_tail else collect_sparse_gap_tail_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    )
    attack_validated_gap_segments: list[tuple[float, float]] = []
    if active_ranges and not mid_performance_end:
        if cfg.use_attack_validated_gap_collector:
            attack_validated_gap_segments = collect_attack_validated_gap_segments(
                active_ranges,
                gap_onset_times,
                onset_attack_profiles,
                audio_duration,
                filtered_gap_candidates,
            )

    segments: list[Segment] = []
    active_range_segments: list[tuple[float, float]] = []
    for range_index, (range_start, range_end) in enumerate(active_ranges):
        effective_range_start = range_start
        previous_range_end = active_ranges[range_index - 1][1] if range_index > 0 else None
        prior_onsets = collect_range_prior_backtrack_onsets(
            range_start,
            range_end,
            previous_range_end,
            onset_times,
            gap_onset_times,
        )
        relaxed_head_segment = False
        if prior_onsets:
            effective_range_start = prior_onsets[-1]
            if (
                previous_range_end is not None
                and range_start - previous_range_end >= ACTIVE_RANGE_START_CLUSTER_MIN_GAP
                and range_end - range_start <= ACTIVE_RANGE_START_CLUSTER_MAX_DURATION
            ):
                trailing_cluster = [
                    time for time in prior_onsets if effective_range_start - time <= ACTIVE_RANGE_START_CLUSTER_MAX_SPAN
                ]
                if len(trailing_cluster) >= 2:
                    effective_range_start = trailing_cluster[0]
                    relaxed_head_segment = True

        range_onsets = [time for time in onset_times if effective_range_start + 0.005 < time < range_end - 0.05]
        if not cfg.ablate_collapse_active_range_head:
            range_onsets = collapse_active_range_head_onsets(
                effective_range_start,
                range_end,
                range_onsets,
                onset_attack_profiles,
            )
        if not cfg.ablate_snap_range_start_to_onset and not prior_onsets and not relaxed_head_segment and range_onsets:
            first_range_onset = range_onsets[0]
            if should_snap_range_start_to_first_onset(effective_range_start, first_range_onset, onset_attack_profiles):
                effective_range_start = first_range_onset
                range_onsets = [time for time in range_onsets if effective_range_start + 0.005 < time < range_end - 0.05]
        boundary_times = sorted(range_onsets)
        deduped_onsets: list[float] = []
        for boundary_index, time in enumerate(boundary_times):
            if not deduped_onsets:
                deduped_onsets.append(time)
                continue

            previous_time = deduped_onsets[-1]
            if (
                time - previous_time >= 0.18
                or should_keep_dense_trailing_onset(boundary_times, boundary_index, effective_range_start, range_end)
                or should_keep_short_range_trailing_onset(boundary_times, boundary_index, effective_range_start, range_end)
            ):
                deduped_onsets.append(time)

        starts = [effective_range_start, *deduped_onsets]
        for index, start_time in enumerate(starts):
            end_time = starts[index + 1] if index + 1 < len(starts) else range_end
            min_duration = CLUSTERED_RANGE_HEAD_MIN_DURATION if relaxed_head_segment and index == 0 else 0.08
            if end_time - start_time >= min_duration:
                seg = Segment(start_time, end_time, sources=frozenset({"activeRange"}))
                segments.append(seg)
                active_range_segments.append((start_time, end_time))

    collector_sources: list[tuple[list[tuple[float, float]], str, bool]] = [
        (multi_onset_gap_segments, "multiOnsetGap", False),
        (sparse_gap_tail_segments, "sparseGapTail", True),
        (attack_validated_gap_segments, "attackValidatedGap", True),
    ]
    for collector_segments, source_name, estimated in collector_sources:
        source_tag = frozenset({source_name})
        for start_time, end_time in collector_segments:
            if end_time - start_time >= 0.08:
                segments.append(Segment(start_time, end_time, sources=source_tag, end_estimated=estimated))

    segments = dedupe_nested_segments(segments)
    segments = dedupe_cross_collector_segments(segments)
    segments = trim_small_overlapping_segments(segments)

    if not segments:
        duration = _audio_duration_sec(audio, sample_rate)
        segments = [Segment(0.0, duration, sources=frozenset({"fallback"}))]

    tempo_audio_duration_sec = float(_audio_duration_sec(audio, sample_rate))
    tempo_start = perf_counter()
    tempo_onset_env = _onset_strength_numpy(audio, sample_rate, TEMPO_ESTIMATION_HOP_LENGTH)
    tempo = _estimate_tempo_autocorr(tempo_onset_env, sample_rate, TEMPO_ESTIMATION_HOP_LENGTH)
    tempo_estimation_ms = (perf_counter() - tempo_start) * 1000.0

    debug_info = {
        "onsetTimes": onset_times,
        "gapValidatedOnsetTimes": gap_onset_times if cfg.filter_gap_onsets_by_attack_profile else None,
        "attackValidatedGapCandidates": {
            "interRanges": [[round(time, 4) for time in gap] for gap in attack_validated_gap_candidates.inter_ranges],
            "leading": [round(time, 4) for time in attack_validated_gap_candidates.leading],
            "trailing": [round(time, 4) for time in attack_validated_gap_candidates.trailing],
        },
        "gapIoiDiagnostics": gap_ioi_diagnostics,
        "activeRanges": [[round(start, 4), round(end, 4)] for start, end in active_ranges],
        "rawActiveRanges": [[round(start, 4), round(end, 4)] for start, end in raw_active_ranges],
        "shortBridgeActiveRanges": [[round(start, 4), round(end, 4)] for start, end in short_bridge_active_ranges],
        "activeRangeSegments": [[round(start, 4), round(end, 4)] for start, end in active_range_segments],
        "multiOnsetGapSegments": [[round(start, 4), round(end, 4)] for start, end in multi_onset_gap_segments],
        "sparseGapTailSegments": [[round(start, 4), round(end, 4)] for start, end in sparse_gap_tail_segments],
        "attackValidatedGapSegments": [[round(start, 4), round(end, 4)] for start, end in attack_validated_gap_segments],
        "segments": [[round(start, 4), round(end, 4)] for start, end in segments],
        "rmsThreshold": round(threshold, 6),
        "tempoRaw": round(tempo, 4),
        "tempoHopLength": TEMPO_ESTIMATION_HOP_LENGTH,
        "tempoAudioDurationSec": round(tempo_audio_duration_sec, 4),
        "tempoEstimationMs": round(tempo_estimation_ms, 3),
        "onsetAttackProfiles": {
            str(key): {
                "broadbandOnsetGain": round(profile.broadband_onset_gain, 6),
                "highBandSpectralFlux": round(profile.high_band_spectral_flux, 6),
                "broadbandSpectralFlux": round(profile.broadband_spectral_flux, 6),
                "isValidAttack": profile.is_valid_attack,
            }
            for key, profile in onset_attack_profiles.items()
        },
    }
    return SegmentDetectionResult(segments=segments, tempo=tempo, debug=debug_info)
