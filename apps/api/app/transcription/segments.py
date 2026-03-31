from __future__ import annotations

from dataclasses import replace as dataclass_replace
from time import perf_counter
from typing import Any

import librosa
import numpy as np

from ..models import InstrumentTuning
from .constants import *
from .models import GapAttackCandidates, NoteCandidate, OnsetAttackProfile, OnsetWaveformStats, Segment
from .profiles import (
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
    return Segment(
        start_time=start,
        end_time=end,
        sources=a.sources | b.sources,
        merged_from=_segment_leaves(a) + _segment_leaves(b),
        merge_reason=reason,
        end_estimated=merged_end_estimated,
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
                if GAP_ONSET_MAX_KURTOSIS > 0 and ws.kurtosis > GAP_ONSET_MAX_KURTOSIS:
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


def collect_post_tail_gap_head_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start, next_end = active_ranges[index + 1]
        gap_duration = next_start - previous_end
        if gap_duration < POST_TAIL_GAP_HEAD_MIN_DURATION:
            continue

        gap_onsets = [
            onset_time
            for onset_time in onset_times
            if previous_end + SPARSE_GAP_TAIL_MIN_PREVIOUS_EDGE < onset_time < next_start - 0.05
        ]
        if len(gap_onsets) < 4:
            continue

        early_gap_onsets = [
            onset_time
            for onset_time in gap_onsets
            if onset_time - previous_end <= POST_TAIL_GAP_HEAD_MAX_EARLY_ONSET_OFFSET
        ]
        late_gap_onsets = [
            onset_time
            for onset_time in gap_onsets
            if onset_time - previous_end >= POST_TAIL_GAP_HEAD_MIN_LATE_ONSET_OFFSET
        ]
        if not (1 <= len(early_gap_onsets) <= 2):
            continue
        if len(late_gap_onsets) < 3:
            continue
        if next_end - next_start > POST_TAIL_GAP_HEAD_MAX_NEXT_RANGE_DURATION:
            continue
        if next_start - late_gap_onsets[-1] < POST_TAIL_GAP_HEAD_MIN_TRAILING_EDGE:
            continue

        late_intervals = [late_gap_onsets[i + 1] - late_gap_onsets[i] for i in range(len(late_gap_onsets) - 1)]
        if not all(
            POST_TAIL_GAP_HEAD_MIN_INTERVAL <= interval <= POST_TAIL_GAP_HEAD_MAX_INTERVAL
            for interval in late_intervals
        ):
            continue

        for start_time, end_time in zip(late_gap_onsets, late_gap_onsets[1:]):
            if end_time - start_time >= 0.08:
                segments.append((start_time, end_time))

    return segments


def collect_leading_orphan_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    first_range_start = active_ranges[0][0]
    leading_onsets = [
        onset_time
        for onset_time in onset_times
        if first_range_start - LEADING_ORPHAN_ONSET_MAX_GAP_BEFORE_ACTIVE
        <= onset_time
        <= first_range_start - LEADING_ORPHAN_ONSET_MIN_GAP_BEFORE_ACTIVE
    ]
    if len(leading_onsets) != 1:
        return []

    orphan_start = leading_onsets[0]
    orphan_end = min(orphan_start + LEADING_ORPHAN_SEGMENT_DURATION, first_range_start - 0.08)
    if orphan_end - orphan_start < 0.08:
        return []
    return [(orphan_start, orphan_end)]


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


def collect_single_onset_gap_head_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    for index in range(len(active_ranges) - 1):
        previous_end = active_ranges[index][1]
        next_start, next_end = active_ranges[index + 1]
        gap_duration = next_start - previous_end
        if not (SINGLE_ONSET_GAP_HEAD_MIN_DURATION <= gap_duration < SINGLE_ONSET_GAP_HEAD_MAX_DURATION):
            continue

        gap_onsets = [onset_time for onset_time in onset_times if previous_end + 0.05 < onset_time < next_start - 0.05]
        if len(gap_onsets) != 1:
            continue

        gap_onset = gap_onsets[0]
        if gap_onset - previous_end < SINGLE_ONSET_GAP_HEAD_MIN_PREVIOUS_EDGE:
            continue
        if next_start - gap_onset < SINGLE_ONSET_GAP_HEAD_MIN_TRAILING_EDGE:
            continue
        if next_end - next_start > SINGLE_ONSET_GAP_HEAD_MAX_NEXT_RANGE_DURATION:
            continue

        segment_start = gap_onset
        segment_end = min(segment_start + SINGLE_ONSET_GAP_HEAD_SEGMENT_DURATION, next_start - 0.08)
        if segment_end - segment_start >= 0.08:
            segments.append((segment_start, segment_end))

    return segments


def collect_two_onset_terminal_tail_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    last_range_end = active_ranges[-1][1]
    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if last_range_end + TERMINAL_TWO_ONSET_TAIL_MIN_GAP_AFTER_ACTIVE <= onset_time <= audio_duration - 0.08
    ]
    if len(trailing_onsets) != 2:
        return []

    first_onset, second_onset = trailing_onsets
    if first_onset > last_range_end + TERMINAL_TWO_ONSET_TAIL_MAX_GAP_AFTER_ACTIVE:
        return []
    if not (TERMINAL_TWO_ONSET_TAIL_MIN_INTERVAL <= second_onset - first_onset <= TERMINAL_TWO_ONSET_TAIL_MAX_INTERVAL):
        return []

    segments: list[tuple[float, float]] = []
    first_end = min(first_onset + TERMINAL_TWO_ONSET_TAIL_SEGMENT_DURATION, second_onset - 0.08)
    if first_end - first_onset >= 0.08:
        segments.append((first_onset, first_end))

    second_end = min(second_onset + TERMINAL_TWO_ONSET_TAIL_SEGMENT_DURATION, audio_duration)
    if second_end - second_onset >= 0.08:
        segments.append((second_onset, second_end))

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


def collect_close_terminal_orphan_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    last_range_end = active_ranges[-1][1]
    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE
        <= onset_time
        <= min(last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE, audio_duration - 0.08)
    ]
    if len(trailing_onsets) != 1:
        return []

    orphan_start = trailing_onsets[0]
    orphan_end = min(orphan_start + CLOSE_TERMINAL_ORPHAN_SEGMENT_DURATION, audio_duration)
    if orphan_end - orphan_start < 0.08:
        return []
    return [(orphan_start, orphan_end)]


def collect_delayed_terminal_orphan_segments(
    base_segment: tuple[float, float] | None,
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if base_segment is None:
        return []

    base_start, base_end = base_segment
    base_duration = base_end - base_start
    if not (DELAYED_TERMINAL_ORPHAN_MIN_BASE_DURATION <= base_duration <= DELAYED_TERMINAL_ORPHAN_MAX_BASE_DURATION):
        return []

    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if base_end + DELAYED_TERMINAL_ORPHAN_MIN_GAP_AFTER_ACTIVE
        <= onset_time
        <= min(base_end + DELAYED_TERMINAL_ORPHAN_MAX_GAP_AFTER_ACTIVE, audio_duration - 0.08)
    ]
    if len(trailing_onsets) != 1:
        return []

    orphan_start = trailing_onsets[0]
    orphan_end = min(orphan_start + DELAYED_TERMINAL_ORPHAN_SEGMENT_DURATION, audio_duration)
    if orphan_end - orphan_start < 0.08:
        return []
    return [(orphan_start, orphan_end)]


def collect_terminal_multi_onset_segments(
    active_ranges: list[tuple[float, float]],
    onset_times: list[float],
    audio_duration: float,
    onset_profiles: dict[float, OnsetAttackProfile] | None = None,
) -> list[tuple[float, float]]:
    if not active_ranges:
        return []

    last_range_end = active_ranges[-1][1]
    trailing_onsets = [
        onset_time
        for onset_time in onset_times
        if onset_time >= last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE
        and onset_time <= audio_duration - 0.08
    ]
    if len(trailing_onsets) < TERMINAL_MULTI_ONSET_MIN_COUNT:
        return []
    if trailing_onsets[0] > last_range_end + CLOSE_TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE:
        return []

    run_onsets = trailing_onsets[1:]
    if len(run_onsets) < 3:
        return []

    intervals = [run_onsets[i + 1] - run_onsets[i] for i in range(len(run_onsets) - 1)]
    if not intervals:
        return []
    if not all(TERMINAL_MULTI_ONSET_MIN_INTERVAL <= interval <= TERMINAL_MULTI_ONSET_MAX_INTERVAL for interval in intervals):
        return []

    segments: list[tuple[float, float]] = []
    for start_time, end_time in zip(run_onsets, run_onsets[1:]):
        if end_time - start_time >= 0.08:
            segments.append((start_time, end_time))

    tail_start = run_onsets[-1]
    tail_end = min(tail_start + TERMINAL_MULTI_ONSET_TAIL_DURATION, audio_duration)
    if tail_end - tail_start >= 0.08:
        segments.append((tail_start, tail_end))
    return segments


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
    descending_primary_suffix_note_names: set[str],
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


def detect_segments(
    audio: np.ndarray,
    sample_rate: int,
    *,
    mid_performance_start: bool = False,
    mid_performance_end: bool = False,
) -> tuple[list[Segment], float, dict[str, Any]]:
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=HOP_LENGTH)
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
        active_ranges.append((max(float(frame_times[active_start]) - 0.02, 0.0), librosa.get_duration(y=audio, sr=sample_rate)))

    raw_active_ranges = active_ranges.copy()
    active_ranges = merge_time_ranges(active_ranges)

    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=HOP_LENGTH)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        backtrack=True,
    )
    onset_times = [float(value) for value in librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=HOP_LENGTH)]
    onset_attack_profiles = precompute_onset_attack_profiles(audio, sample_rate, onset_times)
    onset_times = refine_onset_times_by_attack_profile(onset_times, onset_attack_profiles)
    onset_waveform_stats = (
        precompute_onset_waveform_stats(audio, sample_rate, onset_times)
        if FILTER_GAP_ONSETS_BY_ATTACK_PROFILE or USE_ATTACK_VALIDATED_GAP_COLLECTOR
        else {}
    )
    active_ranges, short_bridge_active_ranges = suppress_short_bridge_active_ranges(active_ranges, onset_times)
    gap_onset_times = (
        filter_gap_onsets_by_attack(onset_times, active_ranges, onset_attack_profiles, onset_waveform_stats)
        if FILTER_GAP_ONSETS_BY_ATTACK_PROFILE
        else onset_times
    )
    gap_ioi_diagnostics = build_gap_ioi_diagnostics(active_ranges, onset_times)

    audio_duration = float(librosa.get_duration(y=audio, sr=sample_rate))
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
        if USE_ATTACK_VALIDATED_GAP_COLLECTOR
        else None
    )
    if filtered_gap_candidates is not None and (mid_performance_start or mid_performance_end):
        filtered_gap_candidates = dataclass_replace(
            filtered_gap_candidates,
            **({"leading": []} if mid_performance_start else {}),
            **({"trailing": []} if mid_performance_end else {}),
        )

    leading_orphan_segments = (
        [] if (ABLATE_LEADING_ORPHAN or mid_performance_start) else collect_leading_orphan_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    )
    gap_injected_segments: list[tuple[float, float]] = []
    multi_onset_gap_segments = (
        [] if ABLATE_MULTI_ONSET_GAP else collect_multi_onset_gap_segments(active_ranges, gap_onset_times, onset_attack_profiles, attack_validated_gap_candidates)
    )
    post_tail_gap_head_segments = (
        [] if ABLATE_POST_TAIL_GAP_HEAD else collect_post_tail_gap_head_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    )
    single_onset_gap_head_segments = (
        [] if ABLATE_SINGLE_ONSET_GAP_HEAD else collect_single_onset_gap_head_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    )
    sparse_gap_tail_segments = (
        [] if ABLATE_SPARSE_GAP_TAIL else collect_sparse_gap_tail_segments(active_ranges, gap_onset_times, onset_attack_profiles)
    )
    if not ABLATE_GAP_INJECTED:
        qualifying_gap_run: list[tuple[float, float]] = []
        for index in range(len(active_ranges) - 1):
            previous_end = active_ranges[index][1]
            next_start = active_ranges[index + 1][0]
            gap_onsets = [time for time in gap_onset_times if previous_end + 0.05 < time < next_start - 0.05]
            qualifies = (
                len(gap_onsets) == 1
                and gap_onsets[0] - previous_end >= 0.35
                and next_start - gap_onsets[0] >= 0.35
            )
            if qualifies:
                qualifying_gap_run.append((gap_onsets[0], next_start))
                continue
            if len(qualifying_gap_run) >= 3:
                gap_injected_segments.extend(qualifying_gap_run)
            qualifying_gap_run = []

        if len(qualifying_gap_run) >= 3:
            gap_injected_segments.extend(qualifying_gap_run)

    terminal_orphan_segments: list[tuple[float, float]] = []
    close_terminal_orphan_segments: list[tuple[float, float]] = []
    delayed_terminal_orphan_segments: list[tuple[float, float]] = []
    terminal_multi_onset_segments: list[tuple[float, float]] = []
    terminal_two_onset_tail_segments: list[tuple[float, float]] = []
    attack_validated_gap_segments: list[tuple[float, float]] = []
    if active_ranges and not mid_performance_end:
        last_range_end = active_ranges[-1][1]
        trailing_onsets = [
            onset_time
            for onset_time in gap_onset_times
            if last_range_end + TERMINAL_ORPHAN_ONSET_MIN_GAP_AFTER_ACTIVE
            <= onset_time
            <= min(last_range_end + TERMINAL_ORPHAN_ONSET_MAX_GAP_AFTER_ACTIVE, audio_duration - 0.08)
        ]
        if len(trailing_onsets) == 1:
            orphan_start = trailing_onsets[0]
            orphan_end = min(orphan_start + TERMINAL_ORPHAN_SEGMENT_DURATION, audio_duration)
            if orphan_end - orphan_start >= 0.08:
                terminal_orphan_segments.append((orphan_start, orphan_end))
        close_terminal_orphan_segments = (
            [] if ABLATE_CLOSE_TERMINAL_ORPHAN else collect_close_terminal_orphan_segments(active_ranges, gap_onset_times, audio_duration, onset_attack_profiles)
        )
        delayed_base_segment = (
            close_terminal_orphan_segments[-1]
            if close_terminal_orphan_segments
            else (terminal_orphan_segments[-1] if terminal_orphan_segments else None)
        )
        delayed_terminal_orphan_segments = (
            [] if ABLATE_DELAYED_TERMINAL_ORPHAN else collect_delayed_terminal_orphan_segments(delayed_base_segment, gap_onset_times, audio_duration, onset_attack_profiles)
        )
        terminal_multi_onset_segments = (
            [] if ABLATE_TERMINAL_MULTI_ONSET else collect_terminal_multi_onset_segments(active_ranges, gap_onset_times, audio_duration, onset_attack_profiles)
        )
        if not terminal_orphan_segments and not close_terminal_orphan_segments and not delayed_terminal_orphan_segments and not terminal_multi_onset_segments:
            terminal_two_onset_tail_segments = (
                [] if ABLATE_TWO_ONSET_TERMINAL_TAIL else collect_two_onset_terminal_tail_segments(active_ranges, gap_onset_times, audio_duration, onset_attack_profiles)
            )
        if USE_ATTACK_VALIDATED_GAP_COLLECTOR:
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
        if not ABLATE_COLLAPSE_ACTIVE_RANGE_HEAD:
            range_onsets = collapse_active_range_head_onsets(
                effective_range_start,
                range_end,
                range_onsets,
                onset_attack_profiles,
            )
        if not ABLATE_SNAP_RANGE_START_TO_ONSET and not prior_onsets and not relaxed_head_segment and range_onsets:
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
        (gap_injected_segments, "gapInjected", False),
        (leading_orphan_segments, "leadingOrphan", False),
        (multi_onset_gap_segments, "multiOnsetGap", False),
        (post_tail_gap_head_segments, "postTailGapHead", False),
        (single_onset_gap_head_segments, "singleOnsetGapHead", True),
        (sparse_gap_tail_segments, "sparseGapTail", True),
        (terminal_orphan_segments, "terminalOrphan", True),
        (close_terminal_orphan_segments, "closeTerminalOrphan", True),
        (delayed_terminal_orphan_segments, "delayedTerminalOrphan", True),
        (terminal_multi_onset_segments, "terminalMultiOnset", False),
        (terminal_two_onset_tail_segments, "terminalTwoOnsetTail", True),
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
        duration = librosa.get_duration(y=audio, sr=sample_rate)
        segments = [Segment(0.0, duration, sources=frozenset({"fallback"}))]

    tempo_audio_duration_sec = float(librosa.get_duration(y=audio, sr=sample_rate))
    tempo_start = perf_counter()
    tempo_onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=TEMPO_ESTIMATION_HOP_LENGTH)
    tempo_array, _ = librosa.beat.beat_track(
        onset_envelope=tempo_onset_env,
        sr=sample_rate,
        hop_length=TEMPO_ESTIMATION_HOP_LENGTH,
    )
    tempo_estimation_ms = (perf_counter() - tempo_start) * 1000.0
    tempo = float(np.asarray(tempo_array).reshape(-1)[0]) if np.asarray(tempo_array).size else 90.0
    if tempo <= 1.0:
        tempo = 90.0

    debug_info = {
        "onsetTimes": onset_times,
        "gapValidatedOnsetTimes": gap_onset_times if FILTER_GAP_ONSETS_BY_ATTACK_PROFILE else None,
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
        "gapInjectedSegments": [[round(start, 4), round(end, 4)] for start, end in gap_injected_segments],
        "leadingOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in leading_orphan_segments],
        "multiOnsetGapSegments": [[round(start, 4), round(end, 4)] for start, end in multi_onset_gap_segments],
        "postTailGapHeadSegments": [[round(start, 4), round(end, 4)] for start, end in post_tail_gap_head_segments],
        "singleOnsetGapHeadSegments": [[round(start, 4), round(end, 4)] for start, end in single_onset_gap_head_segments],
        "sparseGapTailSegments": [[round(start, 4), round(end, 4)] for start, end in sparse_gap_tail_segments],
        "terminalOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in terminal_orphan_segments],
        "closeTerminalOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in close_terminal_orphan_segments],
        "delayedTerminalOrphanSegments": [[round(start, 4), round(end, 4)] for start, end in delayed_terminal_orphan_segments],
        "terminalMultiOnsetSegments": [[round(start, 4), round(end, 4)] for start, end in terminal_multi_onset_segments],
        "terminalTwoOnsetTailSegments": [[round(start, 4), round(end, 4)] for start, end in terminal_two_onset_tail_segments],
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
    return segments, tempo, debug_info
