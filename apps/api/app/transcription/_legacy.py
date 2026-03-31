from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import numpy as np
from fastapi import HTTPException, UploadFile

from ..models import InstrumentTuning, NotationViews, ScoreEvent, ScoreNote, TranscriptionResult
from .audio import cents_distance, normalize_audio, read_audio
from .constants import *
from .models import NoteCandidate, RawEvent, RepeatedPatternPass
from .peaks import are_harmonic_related, harmonic_relation_multiple, is_adjacent_tuning_step, segment_peaks
from .segments import build_segment_debug_contexts, detect_segments, should_keep_low_register_sparse_gap_tail, simplify_sparse_gap_tail_high_octave_dyad
from .events import (
    collapse_ascending_restart_lower_residue_singletons,
    collapse_high_register_adjacent_bridge_dyads,
    collapse_late_descending_step_handoffs,
    collapse_restart_tail_subset_into_following_chord,
    collapse_same_start_primary_singletons,
    merge_adjacent_events,
    merge_four_note_gliss_clusters,
    merge_short_chord_clusters,
    merge_short_gliss_clusters,
    simplify_descending_adjacent_dyad_residue,
    simplify_short_gliss_prefix_to_contiguous_singleton,
    simplify_short_secondary_bleed,
    split_adjacent_step_dyads_in_ascending_runs,
    split_ambiguous_upper_octave_pairs,
    suppress_bridging_octave_pairs,
    suppress_descending_restart_residual_cluster,
    suppress_descending_terminal_residual_cluster,
    suppress_descending_upper_return_overlap,
    suppress_descending_upper_singleton_spikes,
    suppress_leading_descending_overlap,
    suppress_leading_gliss_neighbor_noise,
    suppress_leading_gliss_subset_transients,
    suppress_leading_single_transient,
    suppress_low_confidence_dyad_transients,
    suppress_post_tail_gap_bridge_dyads,
    suppress_resonant_carryover,
    suppress_short_descending_return_singletons,
    suppress_short_residual_tails,
    build_recent_ascending_primary_run_ceiling,
    build_recent_ascending_singleton_suffix,
    build_recent_descending_primary_suffix,
    build_recent_note_names,
    classify_event_gesture,
    contiguous_note_cluster,
    suppress_subset_decay_events,
)

def parse_disabled_repeated_pattern_passes(raw_value: str | None) -> frozenset[str]:
    if raw_value is None or not raw_value.strip():
        return frozenset()

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in raw_value.split(",") if item.strip()]

    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise HTTPException(status_code=400, detail="disabledRepeatedPatternPasses must be a JSON string array or comma-separated string.")

    disabled = frozenset(item.strip() for item in parsed if item.strip())
    unknown = sorted(disabled - set(REPEATED_PATTERN_PASS_IDS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown repeated-pattern pass ids: {', '.join(unknown)}")
    return disabled



def normalize_repeated_four_note_family(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        if len(event_set) == 3:
            family_notes_by_name = {note.note_name: note for note in event.notes}
            family_set = set(event_set)
            merged_end_time = event.end_time
            merged_primary_name = event.primary_note_name
            merged_primary_score = event.primary_score
            merged_is_gliss_like = event.is_gliss_like
            consumed = 1
            added_new_note = False
            merged_family = False
            step_index = index + 1

            while step_index < len(raw_events) and consumed < 3:
                following = raw_events[step_index]
                following_set = frozenset(note.note_name for note in following.notes)
                gap = following.start_time - merged_end_time
                new_notes = following_set - family_set

                if gap > GLISS_CLUSTER_MAX_GAP or len(following_set) != 1:
                    break
                if len(new_notes) != 1 or len(family_set | following_set) != 4:
                    break

                if new_notes:
                    following_duration = following.end_time - following.start_time
                    if following_duration > 0.18 and not following.is_gliss_like:
                        break
                    added_new_note = True

                for note in following.notes:
                    family_notes_by_name.setdefault(note.note_name, note)
                family_set |= following_set
                merged_end_time = following.end_time
                merged_is_gliss_like = merged_is_gliss_like or following.is_gliss_like
                if merged_primary_name not in family_set and following.primary_note_name in family_set:
                    merged_primary_name = following.primary_note_name
                merged_primary_score = max(merged_primary_score, following.primary_score)
                consumed += 1
                step_index += 1

                if added_new_note and len(family_set) == 4:
                    family_notes = sorted(
                        (family_notes_by_name[name] for name in family_set),
                        key=lambda note: note.frequency,
                    )
                    normalized.append(
                        RawEvent(
                            start_time=event.start_time,
                            end_time=merged_end_time,
                            notes=family_notes,
                            is_gliss_like=merged_is_gliss_like,
                            primary_note_name=merged_primary_name if merged_primary_name in family_set else family_notes[0].note_name,
                            primary_score=merged_primary_score,
                        )
                    )
                    index += consumed
                    merged_family = True
                    break

            if merged_family:
                continue

        normalized.append(event)
        index += 1

    return normalized

def normalize_repeated_four_note_gliss_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    # Legacy no-op. Practical gliss behavior is now recovered by earlier local
    # front-end logic and by the anchor-bounded explicit four-note pass below.
    return raw_events

def collect_local_four_note_family_runs(raw_events: list[RawEvent]) -> list[dict[str, Any]]:
    exact_family_events: dict[frozenset[str], list[tuple[int, RawEvent]]] = {}
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        if len(event_set) == 4:
            exact_family_events.setdefault(event_set, []).append((index, event))

    candidate_runs: list[dict[str, Any]] = []
    for family_set, anchors in exact_family_events.items():
        if len(anchors) < 2:
            continue

        family_notes_by_name: dict[str, NoteCandidate] = {}
        for _, anchor in anchors:
            for note in anchor.notes:
                family_notes_by_name.setdefault(note.note_name, note)
        if len(family_notes_by_name) != 4:
            continue

        family_notes = sorted((family_notes_by_name[name] for name in family_set), key=lambda note: note.frequency)
        anchor_indices = [index for index, _ in anchors]
        dominant_score = float(np.median([anchor.primary_score for _, anchor in anchors]))
        runs_for_family: list[dict[str, Any]] = []

        for anchor_index in anchor_indices:
            start_index = anchor_index
            while start_index > 0:
                previous_event = raw_events[start_index - 1]
                gap = raw_events[start_index].start_time - previous_event.end_time
                previous_set = frozenset(note.note_name for note in previous_event.notes)
                previous_duration = previous_event.end_time - previous_event.start_time
                if previous_set <= family_set and previous_duration <= 1.25:
                    start_index -= 1
                    continue
                if (
                    previous_set
                    and not previous_set <= family_set
                    and len(previous_event.notes) <= 2
                    and previous_duration <= 0.18
                    and gap <= 0.12
                ):
                    start_index -= 1
                    continue
                break

            end_index = anchor_index
            while end_index + 1 < len(raw_events):
                next_event = raw_events[end_index + 1]
                gap = next_event.start_time - raw_events[end_index].end_time
                next_set = frozenset(note.note_name for note in next_event.notes)
                next_duration = next_event.end_time - next_event.start_time
                if next_set <= family_set and next_duration <= 1.25:
                    end_index += 1
                    continue
                if (
                    next_set
                    and not next_set <= family_set
                    and len(next_event.notes) <= 2
                    and next_duration <= 0.18
                    and gap <= 0.12
                ):
                    end_index += 1
                    continue
                break

            if runs_for_family and start_index <= runs_for_family[-1]["end_index"] + 1:
                runs_for_family[-1]["end_index"] = max(runs_for_family[-1]["end_index"], end_index)
            else:
                runs_for_family.append(
                    {
                        "family_set": family_set,
                        "family_notes": family_notes,
                        "dominant_score": dominant_score,
                        "start_index": start_index,
                        "end_index": end_index,
                    }
                )

        for run in runs_for_family:
            run_anchor_indices = [
                index for index in anchor_indices if run["start_index"] <= index <= run["end_index"]
            ]
            if len(run_anchor_indices) < 2:
                continue
            run["anchor_indices"] = run_anchor_indices
            candidate_runs.append(run)

    claimed_indices: dict[int, frozenset[str]] = {}
    ambiguous_indices: set[int] = set()
    for run in candidate_runs:
        for index in range(run["start_index"], run["end_index"] + 1):
            existing_family = claimed_indices.get(index)
            if existing_family is None:
                claimed_indices[index] = run["family_set"]
            elif existing_family != run["family_set"]:
                ambiguous_indices.add(index)

    resolved_runs: list[dict[str, Any]] = []
    for run in candidate_runs:
        if any(index in ambiguous_indices for index in range(run["start_index"], run["end_index"] + 1)):
            continue
        resolved_runs.append(run)

    return resolved_runs


def find_local_four_note_family_run(
    index: int,
    family_runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for run in family_runs:
        if run["start_index"] <= index <= run["end_index"]:
            return run
    return None


def normalize_repeated_explicit_four_note_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    family_runs = collect_local_four_note_family_runs(raw_events)
    if not family_runs:
        return raw_events

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        run = find_local_four_note_family_run(index, family_runs)
        if run is None:
            normalized.append(event)
            index += 1
            continue

        dominant_set = run["family_set"]
        dominant_notes = run["family_notes"]
        anchor_indices = run["anchor_indices"]
        event_set = frozenset(note.note_name for note in event.notes)
        duration = event.end_time - event.start_time
        lower_anchor_name = dominant_notes[0].note_name

        if event_set == dominant_set:
            normalized.append(event)
            index += 1
            continue

        previous_exact_distance = min((index - anchor_index for anchor_index in anchor_indices if anchor_index < index), default=999)
        next_exact_distance = min((anchor_index - index for anchor_index in anchor_indices if anchor_index > index), default=999)
        exact_support_before = previous_exact_distance <= 2
        exact_support_after = next_exact_distance <= 2

        next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
        next_set = frozenset(note.note_name for note in next_event.notes) if next_event is not None else frozenset()
        next_gap = next_event.start_time - event.end_time if next_event is not None else 1.0

        if next_event is not None and find_local_four_note_family_run(index + 1, family_runs) == run:
            next_exact_after = min((anchor_index - (index + 1) for anchor_index in anchor_indices if anchor_index > index + 1), default=999) <= 2
            if (
                next_set == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and event_set < dominant_set
                and len(event_set) == 1
                and event.primary_note_name == lower_anchor_name
                and duration <= 0.12
                and event.is_gliss_like
                and (exact_support_before or next_exact_after)
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=True,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

            if (
                next_set == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and event_set < dominant_set
                and len(event_set) == 3
                and duration <= 1.25
                and event.is_gliss_like
                and (exact_support_before or next_exact_after)
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=True,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

            if (
                next_set < dominant_set
                and len(event_set) >= 2
                and len(next_set) >= 2
                and event_set < dominant_set
                and (event_set | next_set) == dominant_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and max(duration, next_event.end_time - next_event.start_time) <= 1.25
                and not event.is_gliss_like
                and not next_event.is_gliss_like
                and next_exact_after
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=dominant_notes,
                        is_gliss_like=False,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue

        if (
            event_set < dominant_set
            and len(event_set) in {2, 3}
            and duration <= 1.25
            and exact_support_before
            and exact_support_after
        ):
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=dominant_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name if event.primary_note_name in dominant_set else dominant_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            index += 1
            continue

        normalized.append(event)
        index += 1

    return normalized


def normalize_strict_four_note_subsets(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    normalized: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        event = raw_events[index]
        event_set = frozenset(note.note_name for note in event.notes)
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            next_set = frozenset(note.note_name for note in next_event.notes)
            next_gap = next_event.start_time - event.end_time
            previous_set = frozenset(note.note_name for note in normalized[-1].notes) if normalized else frozenset()
            future_support = any(
                frozenset(note.note_name for note in raw_events[future_index].notes) == next_set
                for future_index in range(index + 2, min(len(raw_events), index + 4))
            )
            if (
                len(event_set) == 2
                and len(next_set) == 4
                and event_set < next_set
                and next_gap <= GLISS_CLUSTER_MAX_GAP
                and not event.is_gliss_like
                and not next_event.is_gliss_like
                and (previous_set == next_set or future_support)
            ):
                normalized.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=next_event.end_time,
                        notes=next_event.notes,
                        is_gliss_like=event.is_gliss_like or next_event.is_gliss_like,
                        primary_note_name=next_event.primary_note_name if next_event.primary_note_name in next_set else next_event.notes[0].note_name,
                        primary_score=max(event.primary_score, next_event.primary_score),
                    )
                )
                index += 2
                continue
        normalized.append(event)
        index += 1

    return normalized


def normalize_repeated_triad_patterns(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    note_set_counts: dict[frozenset[str], int] = {}
    anchor_indices_by_set: dict[frozenset[str], list[int]] = {}
    anchor_notes_by_set: dict[frozenset[str], dict[str, NoteCandidate]] = {}
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        note_set_counts[event_set] = note_set_counts.get(event_set, 0) + 1
        if len(event_set) != 3:
            continue
        anchor_indices_by_set.setdefault(event_set, []).append(index)
        notes_by_name = anchor_notes_by_set.setdefault(event_set, {})
        for note in event.notes:
            notes_by_name.setdefault(note.note_name, note)

    if not anchor_indices_by_set:
        return raw_events

    def event_set_at(index: int) -> frozenset[str]:
        return frozenset(note.note_name for note in raw_events[index].notes)

    def is_local_fragment(index: int, anchor_set: frozenset[str]) -> bool:
        event_set = event_set_at(index)
        overlap = len(event_set & anchor_set)
        if overlap == 0:
            return False
        if event_set < anchor_set and len(event_set) <= 2:
            return True
        if len(event_set) == 2 and overlap >= 1:
            return True
        if len(event_set) == 3 and overlap == 2 and note_set_counts.get(event_set, 0) <= 1:
            return True
        return False

    local_actions: dict[int, tuple[str, frozenset[str]]] = {}
    conflicting_indices: set[int] = set()

    def assign_local_action(index: int, action: str, anchor_set: frozenset[str]) -> None:
        existing = local_actions.get(index)
        if existing is None:
            local_actions[index] = (action, anchor_set)
            return
        if existing != (action, anchor_set):
            conflicting_indices.add(index)

    def schedule_region(indices: list[int], anchor_set: frozenset[str], *, region: str, anchor_count: int) -> None:
        if not indices:
            return
        if region in {'head', 'tail'} and anchor_count < 3:
            return
        if not all(is_local_fragment(index, anchor_set) for index in indices):
            return
        for index in indices:
            event = raw_events[index]
            event_set = event_set_at(index)
            overlap = len(event_set & anchor_set)
            duration = event.end_time - event.start_time
            if event_set < anchor_set and len(event_set) <= 2:
                if region == 'head' and len(event_set) == 1 and duration <= 0.18:
                    assign_local_action(index, 'drop', anchor_set)
                else:
                    assign_local_action(index, 'rewrite', anchor_set)
                continue
            if len(event_set) == 3 and overlap == 2 and note_set_counts.get(event_set, 0) <= 1:
                assign_local_action(index, 'rewrite', anchor_set)
                continue
            if len(event_set) == 2 and overlap >= 1 and duration <= 0.2:
                assign_local_action(index, 'drop', anchor_set)

    candidate_sets = sorted(
        (
            (anchor_set, indices)
            for anchor_set, indices in anchor_indices_by_set.items()
            if len(indices) >= 2 and len(anchor_notes_by_set.get(anchor_set, {})) == 3
        ),
        key=lambda item: (-len(item[1]), item[1][0]),
    )

    for anchor_set, anchor_indices in candidate_sets:
        for left_anchor_index, right_anchor_index in zip(anchor_indices, anchor_indices[1:]):
            interior_indices = list(range(left_anchor_index + 1, right_anchor_index))
            schedule_region(interior_indices, anchor_set, region='between', anchor_count=len(anchor_indices))

        first_anchor_index = anchor_indices[0]
        head_indices = list(range(0, first_anchor_index))
        if first_anchor_index <= 2:
            schedule_region(head_indices, anchor_set, region='head', anchor_count=len(anchor_indices))

        last_anchor_index = anchor_indices[-1]
        tail_indices = list(range(last_anchor_index + 1, len(raw_events)))
        if 0 < len(tail_indices) <= 2:
            schedule_region(tail_indices, anchor_set, region='tail', anchor_count=len(anchor_indices))

    normalized: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if index in conflicting_indices:
            normalized.append(event)
            continue

        local_action = local_actions.get(index)
        if not local_action:
            normalized.append(event)
            continue

        action, anchor_set = local_action
        if action == 'drop':
            continue
        if action == 'rewrite':
            anchor_notes = anchor_notes_by_set.get(anchor_set)
            if anchor_notes is None or len(anchor_notes) != 3:
                normalized.append(event)
                continue
            dominant_notes = sorted((anchor_notes[name] for name in anchor_set), key=lambda note: note.frequency)
            normalized.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=dominant_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name if event.primary_note_name in anchor_set else dominant_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            )
            continue

        normalized.append(event)

    return normalized


def suppress_repeated_triad_blips(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        event_set = frozenset(note.note_name for note in event.notes)
        if len(event_set) == 3 and 0 < index < len(raw_events) - 1:
            previous_event = raw_events[index - 1]
            next_event = raw_events[index + 1]
            previous_set = frozenset(note.note_name for note in previous_event.notes)
            next_set = frozenset(note.note_name for note in next_event.notes)
            duration = event.end_time - event.start_time
            previous_duration = previous_event.end_time - previous_event.start_time
            next_duration = next_event.end_time - next_event.start_time
            if (
                previous_set == event_set
                and next_set == event_set
                and duration <= 0.35
                and duration <= min(previous_duration, next_duration) * 0.45
            ):
                continue
        cleaned.append(event)
    return cleaned


def suppress_isolated_triad_extensions(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if event.is_gliss_like:
            cleaned.append(event)
            continue
        event_note_set = frozenset(note.note_name for note in event.notes)
        if len(event.notes) == 3:
            candidate_subsets = [
                frozenset(event_note_set - {note.note_name})
                for note in event.notes
                if len(event_note_set - {note.note_name}) == 2
            ]
            best_subset: frozenset[str] | None = None
            best_support: tuple[int, float] | None = None
            for subset in candidate_subsets:
                if event.primary_note_name not in subset:
                    continue
                previous_support: tuple[int, float] | None = None
                for offset in range(max(0, index - 2), index):
                    previous_event = raw_events[offset]
                    previous_set = frozenset(note.note_name for note in previous_event.notes)
                    if previous_event.is_gliss_like or previous_set != subset:
                        continue
                    previous_support = (index - offset, event.start_time - previous_event.end_time)
                next_support: tuple[int, float] | None = None
                for offset in range(index + 1, min(len(raw_events), index + 3)):
                    next_event = raw_events[offset]
                    next_set = frozenset(note.note_name for note in next_event.notes)
                    if next_event.is_gliss_like or next_set != subset:
                        continue
                    next_support = (offset - index, next_event.start_time - event.end_time)
                    break
                if previous_support is None or next_support is None:
                    continue
                support_score = (
                    previous_support[0] + next_support[0],
                    previous_support[1] + next_support[1],
                )
                if best_support is None or support_score < best_support:
                    best_support = support_score
                    best_subset = subset
            if best_subset is not None:
                target_notes = [note for note in event.notes if note.note_name in best_subset]
                cleaned.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=sorted(target_notes, key=lambda note: note.frequency),
                        is_gliss_like=False,
                        primary_note_name=event.primary_note_name if event.primary_note_name in best_subset else target_notes[0].note_name,
                        primary_score=event.primary_score,
                    )
                )
                continue
        cleaned.append(event)

    return cleaned


def suppress_recent_upper_echo_mixed_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 5:
        return raw_events

    without_echo: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if 2 <= index < len(raw_events) - 2 and len(event.notes) == 1:
            previous_previous_event = raw_events[index - 2]
            previous_event = raw_events[index - 1]
            next_event = raw_events[index + 1]
            next_next_event = raw_events[index + 2]
            current_note = event.notes[0]
            if (
                len(previous_previous_event.notes) == 1
                and len(previous_event.notes) == 1
                and len(next_event.notes) == 3
                and len(next_next_event.notes) == 1
                and event.is_gliss_like
                and event.end_time - event.start_time <= 0.14
                and current_note.note_name == previous_previous_event.notes[0].note_name
                and previous_event.notes[0].frequency < current_note.frequency
                and 250.0 <= abs(cents_distance(previous_event.notes[0].frequency, current_note.frequency)) <= 350.0
                and previous_event.notes[0].note_name in {note.note_name for note in next_event.notes}
                and all(note.frequency < current_note.frequency for note in next_event.notes)
                and max(note.frequency for note in next_event.notes if note.note_name != previous_event.notes[0].note_name)
                < next_next_event.notes[0].frequency
                < previous_event.notes[0].frequency
            ):
                continue
        without_echo.append(event)

    cleaned: list[RawEvent] = []
    for index, event in enumerate(without_echo):
        updated_event = event
        if 2 <= index < len(without_echo) - 1 and len(event.notes) == 3:
            previous_previous_event = without_echo[index - 2]
            previous_event = without_echo[index - 1]
            next_event = without_echo[index + 1]
            if (
                len(previous_previous_event.notes) == 1
                and len(previous_event.notes) == 1
                and len(next_event.notes) == 1
            ):
                repeated_notes = [note for note in event.notes if note.note_name == previous_event.notes[0].note_name]
                fresh_notes = [note for note in event.notes if note.note_name != previous_event.notes[0].note_name]
                if len(repeated_notes) == 1 and len(fresh_notes) == 2:
                    repeated_note = repeated_notes[0]
                    if (
                        repeated_note.frequency == max(note.frequency for note in event.notes)
                        and previous_previous_event.notes[0].frequency > repeated_note.frequency
                        and 250.0 <= abs(cents_distance(previous_previous_event.notes[0].frequency, repeated_note.frequency)) <= 350.0
                        and max(note.frequency for note in fresh_notes) < next_event.notes[0].frequency < repeated_note.frequency
                        and event.end_time - event.start_time <= 0.55
                    ):
                        sorted_fresh_notes = sorted(fresh_notes, key=lambda note: note.frequency)
                        updated_event = RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=sorted_fresh_notes,
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=sorted_fresh_notes[-1].note_name,
                            primary_score=event.primary_score,
                        )
        cleaned.append(updated_event)

    return cleaned


def _debug_event_signature(raw_event: RawEvent) -> tuple[float, float, tuple[str, ...], bool]:
    return (
        round(raw_event.start_time, 4),
        round(raw_event.end_time, 4),
        tuple(sorted(note.note_name for note in raw_event.notes)),
        raw_event.is_gliss_like,
    )


def _debug_event_payload_from_signature(signature: tuple[float, float, tuple[str, ...], bool]) -> dict[str, Any]:
    start_time, end_time, notes, is_gliss_like = signature
    return {
        "startTime": start_time,
        "endTime": end_time,
        "notes": list(notes),
        "isGlissLike": is_gliss_like,
    }


def repeated_pattern_passes() -> tuple[RepeatedPatternPass, ...]:
    return (
        RepeatedPatternPass("normalize_repeated_explicit_four_note_patterns", normalize_repeated_explicit_four_note_patterns, merge_after=True),
        RepeatedPatternPass("normalize_repeated_triad_patterns", normalize_repeated_triad_patterns, merge_after=True),
        RepeatedPatternPass("normalize_strict_four_note_subsets", normalize_strict_four_note_subsets, merge_after=True),
        RepeatedPatternPass("suppress_repeated_triad_blips", suppress_repeated_triad_blips, merge_after=False),
        RepeatedPatternPass("suppress_isolated_triad_extensions", suppress_isolated_triad_extensions, merge_after=False),
    )


REPEATED_PATTERN_PASS_IDS = tuple(pass_entry.name for pass_entry in repeated_pattern_passes())


def apply_repeated_pattern_passes(
    raw_events: list[RawEvent],
    *,
    disabled_passes: frozenset[str] | None = None,
    debug: bool = False,
) -> tuple[list[RawEvent], list[dict[str, Any]]]:
    disabled = disabled_passes or frozenset()
    trace: list[dict[str, Any]] = []
    current = raw_events

    for pass_entry in repeated_pattern_passes():
        before_signatures = [_debug_event_signature(event) for event in current]
        next_events = current if pass_entry.name in disabled else pass_entry.fn(current)
        if pass_entry.merge_after:
            next_events = merge_adjacent_events(next_events)
        after_signatures = [_debug_event_signature(event) for event in next_events]

        if debug:
            before_counter = Counter(before_signatures)
            after_counter = Counter(after_signatures)
            removed: list[dict[str, Any]] = []
            added: list[dict[str, Any]] = []
            for signature, count in (before_counter - after_counter).items():
                removed.extend(_debug_event_payload_from_signature(signature) for _ in range(count))
            for signature, count in (after_counter - before_counter).items():
                added.extend(_debug_event_payload_from_signature(signature) for _ in range(count))
            trace.append(
                {
                    "pass": pass_entry.name,
                    "enabled": pass_entry.name not in disabled,
                    "mergeAfter": pass_entry.merge_after,
                    "beforeEventCount": len(before_signatures),
                    "afterEventCount": len(after_signatures),
                    "changed": before_signatures != after_signatures,
                    "removed": removed,
                    "added": added,
                }
            )

        current = next_events

    return current, trace


def quantize_beat(value: float, step: float = 0.25) -> float:
    return round(value / step) * step

def format_doremi(candidate: NoteCandidate) -> str:
    base = PITCH_CLASS_TO_DOREMI[candidate.pitch_class]
    if candidate.octave >= 6:
        return f"{base}.."
    if candidate.octave == 5:
        return f"{base}."
    if candidate.octave == 3:
        return f"_{base}"
    if candidate.octave <= 2:
        return f"__{base}"
    return base

def format_number(candidate: NoteCandidate) -> str:
    base = PITCH_CLASS_TO_NUMBER[candidate.pitch_class]
    if candidate.octave >= 6:
        return f"{base}''"
    if candidate.octave == 5:
        return f"{base}'"
    if candidate.octave == 3:
        return f".{base}"
    if candidate.octave <= 2:
        return f"..{base}"
    return base

def build_notation_views(events: list[ScoreEvent]) -> NotationViews:
    western = [" | ".join(note.pitch_class + str(note.octave) for note in event.notes) for event in events]
    numbered = [" ".join(note.label_number for note in event.notes) for event in events]
    vertical = [[note.label_doremi for note in event.notes] for event in events]
    return NotationViews(western=western, numbered=numbered, verticalDoReMi=vertical)

async def transcribe_audio(
    upload: UploadFile,
    tuning: InstrumentTuning,
    *,
    debug: bool = False,
    disabled_repeated_pattern_passes: frozenset[str] | None = None,
    mid_performance_start: bool = False,
    mid_performance_end: bool = False,
) -> TranscriptionResult:
    audio, sample_rate = await read_audio(upload)
    normalized = normalize_audio(audio)
    segments, tempo, segment_debug = detect_segments(
        normalized, sample_rate,
        mid_performance_start=mid_performance_start,
        mid_performance_end=mid_performance_end,
    )

    raw_events: list[RawEvent] = []
    segment_candidates_debug: list[dict[str, Any]] = []
    segment_contexts = build_segment_debug_contexts(
        segments,
        [tuple(item) for item in segment_debug.get("activeRanges", [])],
        [float(value) for value in segment_debug.get("onsetTimes", [])],
        [float(value) for value in segment_debug.get("gapValidatedOnsetTimes", [])] if segment_debug.get("gapValidatedOnsetTimes") else None,
    ) if debug else {}
    def _segment_keys(key: str) -> set[tuple[float, float]]:
        return {
            (round(float(s), 4), round(float(e), 4))
            for s, e in segment_debug.get(key, [])
        }

    sparse_gap_tail_segment_keys = _segment_keys("sparseGapTailSegments")
    multi_onset_gap_segments = [
        (float(start_time), float(end_time))
        for start_time, end_time in segment_debug.get("multiOnsetGapSegments", [])
    ]
    multi_onset_gap_segment_keys = _segment_keys("multiOnsetGapSegments")
    single_onset_gap_head_segment_keys = _segment_keys("singleOnsetGapHeadSegments")
    for segment in segments:
        start_time, end_time = segment
        duration = max(end_time - start_time, 0.08)
        recent_note_names = build_recent_note_names(raw_events)
        ascending_primary_run_ceiling = build_recent_ascending_primary_run_ceiling(raw_events)
        ascending_singleton_suffix_ceiling, ascending_singleton_suffix_note_names = build_recent_ascending_singleton_suffix(raw_events)
        descending_primary_suffix_floor, descending_primary_suffix_ceiling, descending_primary_suffix_note_names = build_recent_descending_primary_suffix(raw_events)
        previous_primary_note_name = raw_events[-1].primary_note_name if raw_events else None
        previous_primary_frequency = None
        if raw_events and raw_events[-1].primary_note_name is not None:
            previous_primary = next((note for note in raw_events[-1].notes if note.note_name == raw_events[-1].primary_note_name), None)
            previous_primary_frequency = previous_primary.frequency if previous_primary is not None else None
        previous_primary_was_singleton = bool(raw_events and len(raw_events[-1].notes) == 1)
        candidates, candidate_debug, primary = segment_peaks(
            normalized,
            sample_rate,
            start_time,
            end_time,
            tuning,
            debug=debug,
            recent_note_names=recent_note_names,
            ascending_primary_run_ceiling=ascending_primary_run_ceiling,
            ascending_singleton_suffix_ceiling=ascending_singleton_suffix_ceiling,
            ascending_singleton_suffix_note_names=ascending_singleton_suffix_note_names,
            descending_primary_suffix_floor=descending_primary_suffix_floor,
            descending_primary_suffix_ceiling=descending_primary_suffix_ceiling,
            descending_primary_suffix_note_names=descending_primary_suffix_note_names,
            previous_primary_note_name=previous_primary_note_name,
            previous_primary_frequency=previous_primary_frequency,
            previous_primary_was_singleton=previous_primary_was_singleton,
            raw_audio=audio,
        )
        if not candidates or primary is None:
            if debug and candidate_debug:
                segment_candidates_debug.append(candidate_debug)
            continue

        segment_key = (round(start_time, 4), round(end_time, 4))
        if debug and candidate_debug:
            candidate_debug.update(segment_contexts.get(segment_key, {}))
            candidate_debug["segmentSource"] = sorted(segment.sources) if segment.sources else ["unknown"]
            if segment.end_estimated:
                candidate_debug["endEstimated"] = True
            if segment.trimmed_from:
                candidate_debug["trimmedFrom"] = {
                    "startTime": round(segment.trimmed_from.start_time, 6),
                    "endTime": round(segment.trimmed_from.end_time, 6),
                    "sources": sorted(segment.trimmed_from.sources),
                }
            if segment.merged_from:
                candidate_debug["mergeReason"] = segment.merge_reason
                candidate_debug["mergedFrom"] = [
                    {"startTime": round(s.start_time, 6), "endTime": round(s.end_time, 6), "sources": sorted(s.sources)}
                    for s in segment.merged_from
                ]
        if segment_key in sparse_gap_tail_segment_keys and max(note.octave for note in candidates) < 6:
            keep_gap_run_lead_in = any(
                later_start >= end_time + GAP_RUN_LEAD_IN_MIN_FOLLOWUP_GAP
                for later_start, _ in multi_onset_gap_segments
            )
            if not keep_gap_run_lead_in and not should_keep_low_register_sparse_gap_tail(
                candidates,
                tuning,
                descending_primary_suffix_floor,
                descending_primary_suffix_note_names,
            ):
                if debug and candidate_debug:
                    candidate_debug["droppedBy"] = "low_register_sparse_gap_tail"
                    segment_candidates_debug.append(candidate_debug)
                continue
            if debug and candidate_debug:
                candidate_debug["sparseGapTailAdjustment"] = (
                    "gap_run_lead_in_kept"
                    if keep_gap_run_lead_in
                    else "descending_restart_kept"
                )
        if segment_key in sparse_gap_tail_segment_keys:
            simplified_candidates = simplify_sparse_gap_tail_high_octave_dyad(candidates)
            if len(simplified_candidates) != len(candidates):
                candidates = simplified_candidates
                if debug and candidate_debug:
                    candidate_debug["selectedNotes"] = [candidate.note_name for candidate in candidates]
                    candidate_debug["sparseGapTailAdjustment"] = "high_register_octave_simplified"

        raw_events.append(
            RawEvent(
                start_time=start_time,
                end_time=end_time,
                notes=candidates,
                is_gliss_like=duration < 0.18,
                primary_note_name=primary.candidate.note_name,
                primary_score=primary.score,
            )
        )
        if debug and candidate_debug:
            segment_candidates_debug.append(candidate_debug)

    processed_events = suppress_low_confidence_dyad_transients(raw_events)
    processed_events = suppress_resonant_carryover(processed_events, tuning)
    processed_events = collapse_restart_tail_subset_into_following_chord(processed_events, tuning)
    processed_events = collapse_same_start_primary_singletons(processed_events)
    processed_events = simplify_short_secondary_bleed(processed_events)
    processed_events = suppress_post_tail_gap_bridge_dyads(processed_events)
    processed_events = suppress_leading_descending_overlap(processed_events, tuning)
    processed_events = simplify_descending_adjacent_dyad_residue(processed_events)
    processed_events = collapse_high_register_adjacent_bridge_dyads(processed_events, tuning)
    processed_events = suppress_descending_upper_singleton_spikes(processed_events)
    processed_events = suppress_short_descending_return_singletons(processed_events, tuning)
    processed_events = suppress_descending_upper_return_overlap(processed_events)
    processed_events = merge_short_gliss_clusters(processed_events)
    processed_events = simplify_short_gliss_prefix_to_contiguous_singleton(processed_events)
    processed_events = merge_four_note_gliss_clusters(processed_events)
    processed_events = suppress_leading_gliss_subset_transients(processed_events)
    processed_events = suppress_leading_gliss_neighbor_noise(processed_events)
    processed_events = suppress_leading_single_transient(processed_events)
    processed_events = suppress_subset_decay_events(processed_events)
    processed_events = split_ambiguous_upper_octave_pairs(processed_events)
    processed_events = suppress_bridging_octave_pairs(processed_events)
    processed_events = suppress_short_residual_tails(processed_events)
    processed_events = suppress_descending_terminal_residual_cluster(processed_events, tuning)
    processed_events = suppress_descending_restart_residual_cluster(processed_events, tuning)
    processed_events = collapse_late_descending_step_handoffs(processed_events)
    merged_events = merge_adjacent_events(processed_events)
    merged_events = collapse_late_descending_step_handoffs(merged_events)
    merged_events = merge_short_chord_clusters(merged_events)
    merged_events = merge_adjacent_events(merged_events)
    merged_events = suppress_recent_upper_echo_mixed_clusters(merged_events)
    merged_events = collapse_ascending_restart_lower_residue_singletons(merged_events, tuning)
    merged_events = merge_adjacent_events(merged_events)
    merged_events, repeated_pattern_pass_trace = apply_repeated_pattern_passes(
        merged_events,
        disabled_passes=disabled_repeated_pattern_passes,
        debug=debug,
    )
    merged_events = split_adjacent_step_dyads_in_ascending_runs(merged_events, tuning)
    if not merged_events:
        raise HTTPException(status_code=422, detail="No musical notes were detected. Try a clearer recording or a different tuning.")

    beat_seconds = 60.0 / tempo
    events: list[ScoreEvent] = []
    warnings: list[str] = []

    for index, event in enumerate(merged_events, start=1):
        start_beat = quantize_beat(event.start_time / beat_seconds)
        duration_beat = max(quantize_beat((event.end_time - event.start_time) / beat_seconds), 0.25)
        notes = [
            ScoreNote(
                key=candidate.key,
                pitchClass=candidate.pitch_class,
                octave=candidate.octave,
                labelDoReMi=format_doremi(candidate),
                labelNumber=format_number(candidate),
                frequency=round(candidate.frequency, 3),
            )
            for candidate in event.notes
        ]
        events.append(
            ScoreEvent(
                id=f"evt-{index}",
                startBeat=start_beat,
                durationBeat=duration_beat,
                notes=notes,
                isGlissLike=event.is_gliss_like,
                gesture=classify_event_gesture(event, index - 1, raw_events, merged_events),
            )
        )

    if len(events) < 3:
        warnings.append("Only a small number of note events were detected.")

    result_debug = None
    if debug:
        result_debug = {
            **segment_debug,
            "segmentCandidates": segment_candidates_debug,
            "mergedEvents": [
                {
                    "startTime": round(event.start_time, 4),
                    "endTime": round(event.end_time, 4),
                    "notes": [candidate.note_name for candidate in event.notes],
                    "gesture": classify_event_gesture(event, index, raw_events, merged_events),
                }
                for index, event in enumerate(merged_events)
            ],
            "disabledRepeatedPatternPasses": sorted(disabled_repeated_pattern_passes or ()),
            "repeatedPatternPassTrace": repeated_pattern_pass_trace,
        }

    return TranscriptionResult(
        instrumentTuning=tuning,
        tempo=round(tempo, 2),
        events=events,
        notationViews=build_notation_views(events),
        warnings=warnings,
        debug=result_debug,
    )
