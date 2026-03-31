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



def suppress_leading_descending_overlap(raw_events: list[RawEvent], tuning: InstrumentTuning) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    first_event = raw_events[0]
    second_event = raw_events[1]
    third_event = raw_events[2]
    first_duration = first_event.end_time - first_event.start_time
    if (
        len(first_event.notes) != 2
        or len(second_event.notes) != 1
        or len(third_event.notes) != 1
        or first_duration > 0.16
    ):
        return raw_events

    sorted_notes = sorted(first_event.notes, key=lambda note: note.frequency)
    lower_note = sorted_notes[0]
    upper_note = sorted_notes[1]
    middle_note = second_event.notes[0]
    tail_note = third_event.notes[0]
    if (
        tail_note.note_name != lower_note.note_name
        or middle_note.frequency >= upper_note.frequency
        or middle_note.frequency <= lower_note.frequency
        or not is_adjacent_tuning_step(upper_note, middle_note, tuning)
        or not is_adjacent_tuning_step(middle_note, lower_note, tuning)
        or first_event.primary_note_name != upper_note.note_name
    ):
        return raw_events

    updated_first = RawEvent(
        start_time=first_event.start_time,
        end_time=first_event.end_time,
        notes=[upper_note],
        is_gliss_like=first_event.is_gliss_like,
        primary_note_name=upper_note.note_name,
        primary_score=first_event.primary_score,
    )
    return [updated_first, *raw_events[1:]]


def simplify_descending_adjacent_dyad_residue(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        updated_event = event
        if 0 < index < len(raw_events) - 1 and len(event.notes) == 2 and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION:
            previous_event = raw_events[index - 1]
            next_event = raw_events[index + 1]
            if len(previous_event.notes) == 1 and len(next_event.notes) == 1:
                lower_note, upper_note = sorted(event.notes, key=lambda note: note.frequency)
                if (
                    previous_event.notes[0].note_name == upper_note.note_name
                    and next_event.notes[0].frequency < lower_note.frequency
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= cents_distance(lower_note.frequency, upper_note.frequency)
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                ):
                    updated_event = RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=[lower_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=lower_note.note_name,
                        primary_score=event.primary_score,
                    )
        cleaned.append(updated_event)

    return cleaned




def collapse_ascending_restart_lower_residue_singletons(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned = list(raw_events)
    for index in range(1, len(cleaned) - 1):
        previous_event = cleaned[index - 1]
        event = cleaned[index]
        next_event = cleaned[index + 1]
        duration = event.end_time - event.start_time
        if (
            len(previous_event.notes) != 1
            or len(event.notes) != 1
            or len(next_event.notes) != 1
            or previous_event.is_gliss_like
            or next_event.is_gliss_like
            or duration > SHORT_SECONDARY_STRIP_MAX_DURATION
        ):
            continue

        previous_note = previous_event.notes[0]
        residue_note = event.notes[0]
        next_note = next_event.notes[0]
        if (
            previous_note.frequency >= next_note.frequency
            or not is_adjacent_tuning_step(previous_note, next_note, tuning)
            or residue_note.frequency >= previous_note.frequency
            or is_adjacent_tuning_step(residue_note, previous_note, tuning)
            or is_adjacent_tuning_step(residue_note, next_note, tuning)
            or abs(cents_distance(residue_note.frequency, previous_note.frequency)) < 250.0
            or abs(cents_distance(residue_note.frequency, previous_note.frequency)) > 550.0
        ):
            continue

        cleaned[index] = RawEvent(
            start_time=event.start_time,
            end_time=event.end_time,
            notes=[previous_note],
            is_gliss_like=event.is_gliss_like,
            primary_note_name=previous_note.note_name,
            primary_score=event.primary_score,
        )

    return cleaned


def collapse_high_register_adjacent_bridge_dyads(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    cleaned = list(raw_events)
    for index in range(1, len(cleaned) - 2):
        previous_event = cleaned[index - 1]
        event = cleaned[index]
        next_event = cleaned[index + 1]
        next_next_event = cleaned[index + 2]
        duration = event.end_time - event.start_time
        if (
            len(previous_event.notes) != 1
            or len(event.notes) != 2
            or len(next_event.notes) != 1
            or len(next_next_event.notes) != 1
            or previous_event.is_gliss_like
            or next_event.is_gliss_like
            or next_next_event.is_gliss_like
            or duration > ADJACENT_SEPARATED_DYAD_MAX_DURATION
        ):
            continue

        lower_note, upper_note = sorted(event.notes, key=lambda note: note.frequency)
        if (
            lower_note.octave < 6
            or upper_note.octave < 6
            or previous_event.notes[0].note_name != upper_note.note_name
            or next_event.notes[0].note_name != upper_note.note_name
            or next_next_event.notes[0].note_name != lower_note.note_name
            or event.primary_note_name != lower_note.note_name
            or not is_adjacent_tuning_step(lower_note, upper_note, tuning)
        ):
            continue

        cleaned[index] = RawEvent(
            start_time=event.start_time,
            end_time=event.end_time,
            notes=[lower_note],
            is_gliss_like=event.is_gliss_like,
            primary_note_name=lower_note.note_name,
            primary_score=event.primary_score,
        )

    return cleaned
def suppress_descending_upper_singleton_spikes(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        if index == 0 or index + 1 >= len(raw_events):
            cleaned.append(raw_events[index])
            index += 1
            continue

        previous_event = raw_events[index - 1]
        event = raw_events[index]
        next_event = raw_events[index + 1]

        if (
            len(previous_event.notes) == 2
            and len(event.notes) == 1
            and len(next_event.notes) == 1
            and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION
        ):
            previous_sorted = sorted(previous_event.notes, key=lambda note: note.frequency)
            lower_note = previous_sorted[0]
            upper_note = previous_sorted[1]
            current_note = event.notes[0]
            next_note = next_event.notes[0]
            is_descending_upper_spike = (
                current_note.frequency > upper_note.frequency > lower_note.frequency > next_note.frequency
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(current_note.frequency, upper_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(upper_note.frequency, lower_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(lower_note.frequency, next_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
            )
            if is_descending_upper_spike:
                index += 1
                continue

        cleaned.append(event)
        index += 1

    return cleaned



def suppress_short_descending_return_singletons(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if index == 0 or index + 1 >= len(raw_events):
            cleaned.append(event)
            continue

        previous_event = raw_events[index - 1]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time
        gap_before = event.start_time - previous_event.end_time
        gap_after = next_event.start_time - event.end_time
        if (
            len(previous_event.notes) == 1
            and len(event.notes) == 1
            and len(next_event.notes) == 1
            and not previous_event.is_gliss_like
            and not next_event.is_gliss_like
            and duration <= 0.14
            and gap_before <= 0.02
            and gap_after >= 0.18
        ):
            previous_note = previous_event.notes[0]
            current_note = event.notes[0]
            next_note = next_event.notes[0]
            if (
                previous_note.octave >= 6
                and current_note.octave >= 6
                and next_note.octave >= 6
                and previous_note.frequency < current_note.frequency < next_note.frequency
                and is_adjacent_tuning_step(previous_note, current_note, tuning)
                and is_adjacent_tuning_step(current_note, next_note, tuning)
            ):
                continue

        cleaned.append(event)

    return cleaned
def suppress_descending_upper_return_overlap(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        if index < 2 or index + 1 >= len(raw_events):
            cleaned.append(raw_events[index])
            index += 1
            continue

        previous_previous_event = raw_events[index - 2]
        previous_event = raw_events[index - 1]
        event = raw_events[index]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time

        if (
            len(previous_previous_event.notes) == 1
            and len(previous_event.notes) == 1
            and len(event.notes) == 2
            and len(next_event.notes) == 1
            and not previous_previous_event.is_gliss_like
            and not previous_event.is_gliss_like
            and not event.is_gliss_like
            and not next_event.is_gliss_like
            and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
        ):
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                upper_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency > primary_note.frequency]
                if len(upper_notes) == 1:
                    upper_note = upper_notes[0]
                    if (
                        previous_previous_event.notes[0].note_name == upper_note.note_name
                        and previous_event.notes[0].note_name == primary_note.note_name
                        and next_event.notes[0].frequency < primary_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= cents_distance(primary_note.frequency, upper_note.frequency)
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    ):
                        index += 1
                        continue

        cleaned.append(event)
        index += 1

    return cleaned

def split_ambiguous_upper_octave_pairs(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    split_events: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        duration = event.end_time - event.start_time
        if (
            len(event.notes) == 2
            and event.primary_score <= SPLIT_UPPER_OCTAVE_PAIR_PRIMARY_SCORE_MAX
            and harmonic_relation_multiple(event.notes[0], event.notes[1]) == 2.0
        ):
            higher_note = max(event.notes, key=lambda note: note.frequency)
            lower_note = min(event.notes, key=lambda note: note.frequency)
            next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
            previous_event = split_events[-1] if split_events else None
            current_note_names = {note.note_name for note in event.notes}
            previous_note_names = {note.note_name for note in previous_event.notes} if previous_event else set()
            next_note_names = {note.note_name for note in next_event.notes} if next_event else set()
            terminal_reset_octave_pair = (
                previous_event is not None
                and next_event is not None
                and len(previous_event.notes) == 1
                and len(next_event.notes) == 1
                and next_event.notes[0].frequency < lower_note.frequency
                and lower_note.frequency < previous_event.notes[0].frequency < higher_note.frequency
                and duration >= max(0.22, SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION - 0.08)
            )
            if (
                (SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION <= duration <= SPLIT_UPPER_OCTAVE_PAIR_MAX_DURATION or terminal_reset_octave_pair)
                and event.primary_note_name == higher_note.note_name
                and current_note_names != previous_note_names
                and current_note_names != next_note_names
            ):
                split_offset = max(0.08, min(duration * SPLIT_UPPER_OCTAVE_PAIR_FRACTION, duration - 0.08))
                split_time = event.start_time + split_offset
                split_events.append(
                    RawEvent(
                        start_time=event.start_time,
                        end_time=split_time,
                        notes=[lower_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=lower_note.note_name,
                        primary_score=event.primary_score,
                    )
                )
                split_events.append(
                    RawEvent(
                        start_time=split_time,
                        end_time=event.end_time,
                        notes=[higher_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=higher_note.note_name,
                        primary_score=event.primary_score,
                    )
                )
                continue
        split_events.append(event)

    return split_events

def suppress_resonant_carryover(raw_events: list[RawEvent], tuning: InstrumentTuning | None = None) -> list[RawEvent]:
    if not raw_events:
        return []

    cleaned = [raw_events[0]]
    for index, event in enumerate(raw_events[1:], start=1):
        immediate_previous_notes = cleaned[-1].notes
        immediate_recent = {note.note_name for note in immediate_previous_notes}
        older_recent: set[str] = set()
        if len(cleaned) > 1:
            older_recent = {
                note.note_name
                for note in cleaned[-2].notes
                if any(are_harmonic_related(note, previous_note) for previous_note in immediate_previous_notes)
            }
        recent_notes = immediate_recent | older_recent

        updated_event = event
        if len(event.notes) == 2:
            repeated_notes = [note for note in event.notes if note.note_name in recent_notes]
            fresh_notes = [note for note in event.notes if note.note_name not in recent_notes]
            duration = event.end_time - event.start_time
            keep_short_octave_dyad = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1:
                keep_short_octave_dyad = (
                    harmonic_relation_multiple(repeated_notes[0], fresh_notes[0]) == 2.0
                    and duration <= 0.35
                )
            keep_phrase_reset_ascending_dyad = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                next_event = raw_events[index + 1]
                next_gap = next_event.start_time - event.end_time
                if (
                    len(next_event.notes) == 1
                    and next_gap >= RESONANT_CARRYOVER_PHRASE_RESET_MIN_GAP
                    and next_event.notes[0].frequency < repeated_notes[0].frequency
                ):
                    keep_phrase_reset_ascending_dyad = True
            keep_phrase_reset_lower_repeated = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                next_event = raw_events[index + 1]
                next_gap = next_event.start_time - event.end_time
                if (
                    len(next_event.notes) == 1
                    and duration <= RESONANT_CARRYOVER_HIGH_RETURN_MAX_DURATION
                    and next_gap <= RESONANT_CARRYOVER_HIGH_RETURN_MAX_NEXT_GAP
                    and repeated_notes[0].frequency < next_event.notes[0].frequency < fresh_notes[0].frequency
                    and cents_distance(repeated_notes[0].frequency, fresh_notes[0].frequency) >= RESONANT_CARRYOVER_HIGH_RETURN_MIN_INTERVAL_CENTS
                ):
                    keep_phrase_reset_lower_repeated = True
            keep_repeated_short_restart_overlap = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                previous_event = cleaned[-1]
                next_event = raw_events[index + 1]
                if (
                    len(previous_event.notes) == 1
                    and previous_event.notes[0].note_name == repeated_notes[0].note_name
                    and len(next_event.notes) == 1
                    and fresh_notes[0].frequency < repeated_notes[0].frequency < next_event.notes[0].frequency
                    and duration <= 0.14
                    and cents_distance(fresh_notes[0].frequency, repeated_notes[0].frequency) >= 300.0
                ):
                    keep_repeated_short_restart_overlap = True
            keep_descending_lower_repeated = False
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and index + 1 < len(raw_events):
                next_event = raw_events[index + 1]
                if (
                    len(next_event.notes) == 1
                    and repeated_notes[0].frequency < fresh_notes[0].frequency
                    and next_event.notes[0].frequency < repeated_notes[0].frequency
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= cents_distance(repeated_notes[0].frequency, fresh_notes[0].frequency)
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(next_event.notes[0].frequency, repeated_notes[0].frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                ):
                    keep_descending_lower_repeated = True
            keep_high_register_repeated_lower_restart = False
            if (
                tuning is not None
                and len(repeated_notes) == 1
                and len(fresh_notes) == 1
                and index > 1
                and index + 1 < len(raw_events)
            ):
                previous_event = cleaned[-1]
                previous_previous_event = cleaned[-2]
                next_event = raw_events[index + 1]
                repeated_note = repeated_notes[0]
                fresh_note = fresh_notes[0]
                if (
                    len(previous_event.notes) == 1
                    and len(previous_previous_event.notes) == 1
                    and len(next_event.notes) == 1
                    and previous_event.notes[0].note_name == repeated_note.note_name
                    and previous_previous_event.notes[0].note_name == fresh_note.note_name
                    and repeated_note.octave >= 6
                    and fresh_note.octave >= 6
                    and next_event.notes[0].octave >= 6
                    and repeated_note.frequency < fresh_note.frequency < next_event.notes[0].frequency
                    and is_adjacent_tuning_step(repeated_note, fresh_note, tuning)
                    and is_adjacent_tuning_step(fresh_note, next_event.notes[0], tuning)
                    and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION
                ):
                    keep_high_register_repeated_lower_restart = True
            keep_post_triad_upper_tail_repeated = False
            if (
                tuning is not None
                and len(repeated_notes) == 1
                and len(fresh_notes) == 1
                and index + 1 < len(raw_events)
            ):
                previous_event = cleaned[-1]
                next_event = raw_events[index + 1]
                repeated_note = repeated_notes[0]
                fresh_note = fresh_notes[0]
                if (
                    len(previous_event.notes) >= 3
                    and len(next_event.notes) == 1
                    and repeated_note.note_name in {note.note_name for note in previous_event.notes}
                    and event.primary_note_name == repeated_note.note_name
                    and repeated_note.frequency < fresh_note.frequency
                    and next_event.notes[0].frequency < repeated_note.frequency
                    and is_adjacent_tuning_step(repeated_note, fresh_note, tuning)
                    and duration <= 0.14
                ):
                    keep_post_triad_upper_tail_repeated = True
            if (
                keep_phrase_reset_lower_repeated
                or keep_repeated_short_restart_overlap
                or keep_descending_lower_repeated
                or keep_high_register_repeated_lower_restart
                or keep_post_triad_upper_tail_repeated
            ):
                updated_event = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=repeated_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=repeated_notes[0].note_name,
                    primary_score=event.primary_score,
                )
            if len(repeated_notes) == 1 and len(fresh_notes) == 1 and (
                repeated_notes[0].frequency < fresh_notes[0].frequency
                or duration <= 0.14
            ) and not keep_short_octave_dyad and not keep_phrase_reset_ascending_dyad and not keep_phrase_reset_lower_repeated and not keep_repeated_short_restart_overlap and not keep_descending_lower_repeated and not keep_high_register_repeated_lower_restart and not keep_post_triad_upper_tail_repeated:
                updated_event = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=fresh_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=fresh_notes[0].note_name,
                    primary_score=event.primary_score,
                )
        cleaned.append(updated_event)

    return cleaned



def collapse_same_start_primary_singletons(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            cleaned.append(current)
            break

        following = raw_events[index + 1]
        shared_start = abs(current.start_time - following.start_time) <= SAME_START_PRIMARY_SINGLETON_MAX_START_DELTA
        if not shared_start:
            cleaned.append(current)
            index += 1
            continue

        current_primary = next((note for note in current.notes if note.note_name == current.primary_note_name), None)
        following_primary = next((note for note in following.notes if note.note_name == following.primary_note_name), None)
        if current_primary is None or following_primary is None:
            cleaned.append(current)
            index += 1
            continue

        current_non_primary = [note for note in current.notes if note.note_name != current.primary_note_name]
        following_non_primary = [note for note in following.notes if note.note_name != following.primary_note_name]
        current_is_primary_singleton = len(current.notes) == 1
        following_is_primary_singleton = len(following.notes) == 1
        current_note_names = {note.note_name for note in current.notes}
        following_note_names = {note.note_name for note in following.notes}
        shared_primary = current.primary_note_name and current.primary_note_name == following.primary_note_name

        prefer_current = (
            shared_primary
            and current_is_primary_singleton
            and following_non_primary
            and all(note.frequency < current_primary.frequency for note in following_non_primary)
        ) or (
            current_is_primary_singleton
            and current.primary_note_name in following_note_names
            and any(note.note_name != current.primary_note_name for note in following.notes)
            and all(note.note_name == current.primary_note_name or note.frequency < current_primary.frequency for note in following.notes)
        )
        prefer_following = (
            shared_primary
            and following_is_primary_singleton
            and current_non_primary
            and all(note.frequency < following_primary.frequency for note in current_non_primary)
        ) or (
            following_is_primary_singleton
            and following.primary_note_name in current_note_names
            and any(note.note_name != following.primary_note_name for note in current.notes)
            and all(note.note_name == following.primary_note_name or note.frequency < following_primary.frequency for note in current.notes)
        )

        if not prefer_current and not prefer_following:
            cleaned.append(current)
            index += 1
            continue

        phrase_reset_lower: NoteCandidate | None = None
        if (
            prefer_following
            and shared_primary
            and len(current_non_primary) == 1
            and following_is_primary_singleton
            and index + 2 < len(raw_events)
        ):
            lower_note = current_non_primary[0]
            next_event = raw_events[index + 2]
            next_primary = next((note for note in next_event.notes if note.note_name == next_event.primary_note_name), None)
            if (
                next_primary is not None
                and len(next_event.notes) == 1
                and next_primary.frequency > following_primary.frequency
                and cents_distance(lower_note.frequency, following_primary.frequency) >= 350.0
            ):
                phrase_reset_lower = lower_note

        if phrase_reset_lower is not None:
            preferred_notes = [phrase_reset_lower]
            preferred_primary = phrase_reset_lower.note_name
        else:
            preferred = current if prefer_current else following
            preferred_notes = preferred.notes
            preferred_primary = preferred.primary_note_name
        cleaned.append(
            RawEvent(
                start_time=min(current.start_time, following.start_time),
                end_time=max(current.end_time, following.end_time),
                notes=preferred_notes,
                is_gliss_like=current.is_gliss_like or following.is_gliss_like,
                primary_note_name=preferred_primary,
                primary_score=max(current.primary_score, following.primary_score),
            )
        )
        index += 2

    return cleaned


def collapse_restart_tail_subset_into_following_chord(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 2 >= len(raw_events):
            cleaned.append(current)
            index += 1
            continue

        following = raw_events[index + 1]
        trailing = raw_events[index + 2]
        if len(current.notes) == 1 and len(following.notes) == 1 and len(trailing.notes) == 2:
            current_note = current.notes[0]
            following_note = following.notes[0]
            trailing_note_names = {note.note_name for note in trailing.notes}
            trailing_lower_notes = [note for note in trailing.notes if note.frequency < following_note.frequency]
            following_duration = following.end_time - following.start_time
            gap_to_trailing = trailing.start_time - following.end_time
            if (
                following_note.note_name in trailing_note_names
                and trailing.primary_note_name == following_note.note_name
                and current_note.frequency < following_note.frequency
                and is_adjacent_tuning_step(current_note, following_note, tuning)
                and trailing_lower_notes
                and gap_to_trailing <= 0.04
                and following_duration <= 0.32
                and all(note.frequency < current_note.frequency for note in trailing_lower_notes)
            ):
                cleaned.append(current)
                cleaned.append(trailing)
                index += 3
                continue

        cleaned.append(current)
        index += 1

    return cleaned

def suppress_leading_single_transient(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    first_event = raw_events[0]
    next_event = raw_events[1]
    first_duration = first_event.end_time - first_event.start_time
    next_note_names = {note.note_name for note in next_event.notes}
    if (
        len(first_event.notes) == 1
        and first_event.notes[0].note_name in next_note_names
        and len(next_event.notes) >= 1
        and first_duration <= LEADING_SINGLE_TRANSIENT_MAX_DURATION
        and first_event.primary_score <= LEADING_SINGLE_TRANSIENT_MAX_SCORE
        and next_event.primary_score >= first_event.primary_score * LEADING_SINGLE_TRANSIENT_NEXT_SCORE_RATIO
    ):
        return raw_events[1:]

    return raw_events



def suppress_leading_gliss_subset_transients(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            current_duration = current.end_time - current.start_time
            gap = next_event.start_time - current.end_time
            current_note_names = {note.note_name for note in current.notes}
            next_note_names = {note.note_name for note in next_event.notes}
            if (
                len(current.notes) == 1
                and len(next_event.notes) >= 3
                and current_note_names < next_note_names
                and current_duration <= GLISS_LEADING_SUBSET_MAX_DURATION
                and gap <= GLISS_CLUSTER_MAX_GAP
                and next_event.primary_score >= current.primary_score * GLISS_LEADING_SUBSET_SCORE_RATIO
            ):
                index += 1
                continue
        cleaned.append(current)
        index += 1

    return cleaned

def suppress_low_confidence_dyad_transients(raw_events: list[RawEvent]) -> list[RawEvent]:
    cleaned: list[RawEvent] = []
    for event in raw_events:
        duration = event.end_time - event.start_time
        if len(event.notes) == 2 and ((duration <= LOW_CONFIDENCE_DYAD_MAX_DURATION and event.primary_score <= LOW_CONFIDENCE_DYAD_MAX_SCORE) or event.primary_score <= VERY_LOW_CONFIDENCE_EVENT_MAX_SCORE):
            continue
        cleaned.append(event)
    return cleaned

def simplify_short_secondary_bleed(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        updated_event = event
        duration = event.end_time - event.start_time
        if (
            len(event.notes) == 2
            and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
            and event.primary_score >= SHORT_SECONDARY_STRIP_MIN_SCORE
        ):
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                previous_event = raw_events[index - 1] if index > 0 else None
                next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
                lower_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency < primary_note.frequency]
                upper_notes = [note for note in event.notes if note.note_name != primary_note.note_name and note.frequency > primary_note.frequency]
                sorted_notes = sorted(event.notes, key=lambda note: note.frequency)
                descending_bridge_to_upper = False
                upper_bridge_note: NoteCandidate | None = None
                if (
                    len(sorted_notes) == 2
                    and previous_event is not None
                    and len(previous_event.notes) == 1
                    and next_event is not None
                    and len(next_event.notes) == 1
                ):
                    lower_bridge_note = sorted_notes[0]
                    upper_bridge_note = sorted_notes[1]
                    previous_note = previous_event.notes[0]
                    next_note = next_event.notes[0]
                    descending_bridge_to_upper = (
                        previous_note.frequency > upper_bridge_note.frequency > next_note.frequency > lower_bridge_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(previous_note.frequency, upper_bridge_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(upper_bridge_note.frequency, next_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_note.frequency, lower_bridge_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    )
                if descending_bridge_to_upper and upper_bridge_note is not None:
                    updated_event = RawEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        notes=[upper_bridge_note],
                        is_gliss_like=event.is_gliss_like,
                        primary_note_name=upper_bridge_note.note_name,
                        primary_score=event.primary_score,
                    )
                    cleaned.append(updated_event)
                    continue
                if len(lower_notes) == 1:
                    next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
                    if (
                        next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].note_name == primary_note.note_name
                        and next_event.primary_score >= event.primary_score * SHORT_SECONDARY_STRIP_NEXT_SCORE_RATIO
                    ):
                        updated_event = RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=[primary_note],
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=primary_note.note_name,
                            primary_score=event.primary_score,
                        )
                    if previous_event is not None and next_event is not None and updated_event is event:
                        lower_note = lower_notes[0]
                    if previous_event is not None and next_event is not None and updated_event is event:
                        lower_note = lower_notes[0]
                        descending_restart_primary_repeat = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].frequency > primary_note.frequency
                            and next_event.primary_note_name == primary_note.note_name
                            and any(note.note_name == primary_note.note_name for note in next_event.notes)
                            and all(note.note_name == primary_note.note_name or note.frequency > primary_note.frequency for note in next_event.notes)
                        )
                        descending_step_handoff = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].note_name == primary_note.note_name
                            and len(next_event.notes) == 1
                            and next_event.notes[0].frequency < lower_note.frequency
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(lower_note.frequency, primary_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        )
                        descending_upper_residue = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].note_name == lower_note.note_name
                            and len(next_event.notes) == 1
                            and next_event.notes[0].frequency < lower_note.frequency
                            and 250.0
                            <= abs(cents_distance(lower_note.frequency, primary_note.frequency))
                            <= 550.0
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        )
                        descending_restart_upper_sandwich = (
                            len(previous_event.notes) == 1
                            and len(next_event.notes) == 1
                            and previous_event.notes[0].note_name == next_event.notes[0].note_name
                            and previous_event.notes[0].frequency > primary_note.frequency
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(lower_note.frequency, primary_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                            <= abs(cents_distance(previous_event.notes[0].frequency, primary_note.frequency))
                            <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        )
                        if (descending_step_handoff or descending_upper_residue or descending_restart_upper_sandwich or descending_restart_primary_repeat) and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION:
                            kept_note = lower_note
                            if descending_restart_upper_sandwich or descending_restart_primary_repeat:
                                kept_note = primary_note
                            updated_event = RawEvent(
                                start_time=event.start_time,
                                end_time=event.end_time,
                                notes=[kept_note],
                                is_gliss_like=event.is_gliss_like,
                                primary_note_name=kept_note.note_name,
                                primary_score=event.primary_score,
                            )
                elif len(upper_notes) == 1:
                    upper_note = upper_notes[0]
                    descending_restart_bridge = (
                        previous_event is not None
                        and len(previous_event.notes) == 1
                        and previous_event.notes[0].frequency < primary_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(previous_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency > upper_note.frequency
                        and cents_distance(upper_note.frequency, next_event.notes[0].frequency) >= 1200.0
                    )
                    if descending_restart_bridge:
                        continue
                    repeated_high_return = (
                        duration <= SHORT_SECONDARY_STRIP_MAX_DURATION
                        and previous_event is not None
                        and any(note.note_name == upper_note.note_name for note in previous_event.notes)
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and primary_note.frequency < next_event.notes[0].frequency < upper_note.frequency
                        and cents_distance(primary_note.frequency, upper_note.frequency) >= RESONANT_CARRYOVER_HIGH_RETURN_MIN_INTERVAL_CENTS
                    )
                    descending_step_handoff = (
                        previous_event is not None
                        and len(previous_event.notes) == 1
                        and previous_event.notes[0].note_name == upper_note.note_name
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency < primary_note.frequency
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= cents_distance(primary_note.frequency, upper_note.frequency)
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    )
                    descending_upper_residue = (
                        previous_event is not None
                        and len(previous_event.notes) == 1
                        and previous_event.notes[0].note_name == primary_note.note_name
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency < primary_note.frequency
                        and 250.0
                        <= cents_distance(primary_note.frequency, upper_note.frequency)
                        <= 550.0
                        and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                        <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                        <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    )
                    mirrored_adjacent_run = False
                    if duration <= SHORT_SECONDARY_STRIP_MAX_DURATION and previous_event is not None and next_event is not None and index + 2 < len(raw_events):
                        next_next_event = raw_events[index + 2]
                        mirrored_adjacent_run = (
                            len(previous_event.notes) == 1
                            and previous_event.notes[0].note_name == primary_note.note_name
                            and len(next_event.notes) == 2
                            and {note.note_name for note in next_event.notes} == {primary_note.note_name, upper_note.note_name}
                            and len(next_next_event.notes) == 1
                            and next_next_event.notes[0].frequency > upper_note.frequency
                            and cents_distance(primary_note.frequency, upper_note.frequency) <= 250.0
                        )
                    repeated_descending_handoff = False
                    if duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION and previous_event is not None and next_event is not None:
                        repeated_descending_handoff_head = False
                        if index + 2 < len(raw_events):
                            next_next_event = raw_events[index + 2]
                            repeated_descending_handoff_head = (
                                len(previous_event.notes) == 1
                                and previous_event.notes[0].note_name == upper_note.note_name
                                and {note.note_name for note in next_event.notes} == {primary_note.note_name, upper_note.note_name}
                                and len(next_next_event.notes) == 1
                                and next_next_event.notes[0].frequency < primary_note.frequency
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= cents_distance(primary_note.frequency, upper_note.frequency)
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= abs(cents_distance(next_next_event.notes[0].frequency, primary_note.frequency))
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            )
                        repeated_descending_handoff_tail = False
                        if index >= 2:
                            previous_previous_event = raw_events[index - 2]
                            repeated_descending_handoff_tail = (
                                {note.note_name for note in previous_event.notes} == {primary_note.note_name, upper_note.note_name}
                                and len(previous_previous_event.notes) == 1
                                and previous_previous_event.notes[0].note_name == upper_note.note_name
                                and len(next_event.notes) == 1
                                and next_event.notes[0].frequency < primary_note.frequency
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= cents_distance(primary_note.frequency, upper_note.frequency)
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                                <= abs(cents_distance(next_event.notes[0].frequency, primary_note.frequency))
                                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                            )
                        repeated_descending_handoff = repeated_descending_handoff_head or repeated_descending_handoff_tail
                    repeated_adjacent_tail = (
                        duration <= SHORT_SECONDARY_STRIP_MAX_DURATION
                        and previous_event is not None
                        and {note.note_name for note in previous_event.notes} == {primary_note.note_name, upper_note.note_name}
                        and next_event is not None
                        and len(next_event.notes) == 1
                        and next_event.notes[0].frequency > upper_note.frequency
                        and cents_distance(primary_note.frequency, upper_note.frequency) <= 250.0
                    )
                    if repeated_high_return or repeated_descending_handoff or ((descending_step_handoff or descending_upper_residue) and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION) or mirrored_adjacent_run or repeated_adjacent_tail:
                        kept_note = primary_note if repeated_high_return else upper_note
                        if repeated_descending_handoff or ((descending_step_handoff or descending_upper_residue) and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION):
                            kept_note = primary_note
                        updated_event = RawEvent(
                            start_time=event.start_time,
                            end_time=event.end_time,
                            notes=[kept_note],
                            is_gliss_like=event.is_gliss_like,
                            primary_note_name=kept_note.note_name,
                            primary_score=event.primary_score,
                        )
        cleaned.append(updated_event)

    return cleaned


def suppress_post_tail_gap_bridge_dyads(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        previous_event = raw_events[index - 1] if index > 0 else None
        next_event = raw_events[index + 1] if index + 1 < len(raw_events) else None
        if (
            previous_event is not None
            and next_event is not None
            and len(event.notes) == 2
            and len(previous_event.notes) == 1
            and len(next_event.notes) == 2
            and {note.note_name for note in next_event.notes} == {note.note_name for note in event.notes}
        ):
            duration = event.end_time - event.start_time
            gap_after = next_event.start_time - event.end_time
            primary_note = next((note for note in event.notes if note.note_name == event.primary_note_name), None)
            if primary_note is not None:
                upper_note = next((note for note in event.notes if note.note_name != primary_note.note_name and note.frequency > primary_note.frequency), None)
                if upper_note is not None and (
                    duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
                    and gap_after >= 0.35
                    and previous_event.notes[0].frequency < primary_note.frequency
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(previous_event.notes[0].frequency, primary_note.frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                    and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                    <= abs(cents_distance(primary_note.frequency, upper_note.frequency))
                    <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                ):
                    continue
        cleaned.append(event)

    return cleaned


def merge_short_gliss_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            merged.append(current)
            break

        following = raw_events[index + 1]
        gap = following.start_time - current.end_time
        current_duration = current.end_time - current.start_time
        following_duration = following.end_time - following.start_time
        combined_notes: dict[str, NoteCandidate] = {note.note_name: note for note in current.notes}
        for note in following.notes:
            combined_notes.setdefault(note.note_name, note)
        combined_keys = sorted(note.key for note in combined_notes.values())
        contiguous_keys = bool(combined_keys) and (combined_keys[-1] - combined_keys[0] + 1 == len(combined_keys))
        overlap_count = len(current.notes) + len(following.notes) - len(combined_notes)
        merge_pattern_ok = (
            (
                {len(current.notes), len(following.notes)} == {1, 2}
                and max(current_duration, following_duration) <= 0.18
            )
            or (
                len(current.notes) == 2
                and len(following.notes) == 2
                and overlap_count == 1
                and (following.is_gliss_like or following_duration <= 0.18)
            )
            or (min(len(current.notes), len(following.notes)) == 1 and max(len(current.notes), len(following.notes)) == 3 and overlap_count >= 1)
        )
        can_merge = (
            gap <= GLISS_CLUSTER_MAX_GAP
            and current_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and following_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and (following.end_time - current.start_time) <= GLISS_CLUSTER_MAX_TOTAL_DURATION
            and len(combined_notes) == GLISS_CLUSTER_TARGET_NOTE_COUNT
            and contiguous_keys
            and merge_pattern_ok
        )
        if can_merge:
            merged.append(
                RawEvent(
                    start_time=current.start_time,
                    end_time=following.end_time,
                    notes=sorted(combined_notes.values(), key=lambda note: note.frequency),
                    is_gliss_like=True,
                    primary_note_name=current.primary_note_name,
                    primary_score=max(current.primary_score, following.primary_score),
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def simplify_short_gliss_prefix_to_contiguous_singleton(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        updated_event = current
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            duration = current.end_time - current.start_time
            gap = next_event.start_time - current.end_time
            if len(current.notes) == 2 and duration <= 0.14 and gap <= GLISS_CLUSTER_MAX_GAP:
                candidate_unions: list[tuple[NoteCandidate, list[int]]] = []
                next_keys = sorted(note.key for note in next_event.notes)
                target_union_size = 4 if len(next_event.notes) >= 3 else 3
                if len(next_event.notes) >= 2 and not next_event.is_gliss_like:
                    for note in current.notes:
                        merged_keys = sorted(set(next_keys + [note.key]))
                        if merged_keys[-1] - merged_keys[0] + 1 == len(merged_keys) == target_union_size:
                            candidate_unions.append((note, merged_keys))
                if len(next_event.notes) >= 3:
                    candidate_unions = []
                    for note in current.notes:
                        merged_keys = sorted(set(next_keys + [note.key]))
                        if merged_keys[-1] - merged_keys[0] + 1 == len(merged_keys) == 4:
                            candidate_unions.append((note, merged_keys))
                if len(candidate_unions) == 1:
                    chosen_note = candidate_unions[0][0]
                    updated_event = RawEvent(
                        start_time=current.start_time,
                        end_time=current.end_time,
                        notes=[chosen_note],
                        is_gliss_like=True,
                        primary_note_name=chosen_note.note_name,
                        primary_score=current.primary_score,
                    )
        cleaned.append(updated_event)
        index += 1

    return cleaned


def merge_four_note_gliss_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            merged.append(current)
            break

        following = raw_events[index + 1]
        gap = following.start_time - current.end_time
        current_duration = current.end_time - current.start_time
        following_duration = following.end_time - following.start_time
        combined_notes: dict[str, NoteCandidate] = {note.note_name: note for note in current.notes}
        for note in following.notes:
            combined_notes.setdefault(note.note_name, note)
        combined_keys = sorted(note.key for note in combined_notes.values())
        contiguous_keys = bool(combined_keys) and (combined_keys[-1] - combined_keys[0] + 1 == len(combined_keys))
        overlap_count = len(current.notes) + len(following.notes) - len(combined_notes)
        merge_pattern_ok = (
            {len(current.notes), len(following.notes)} == {1, 3}
            or ({len(current.notes), len(following.notes)} == {2, 3} and overlap_count >= 1)
        )
        can_merge = (
            gap <= GLISS_CLUSTER_MAX_GAP
            and current_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and following_duration <= GLISS_CLUSTER_MAX_EVENT_DURATION
            and (following.end_time - current.start_time) <= GLISS_CLUSTER_MAX_TOTAL_DURATION
            and len(combined_notes) == 4
            and contiguous_keys
            and merge_pattern_ok
        )
        if can_merge:
            merged.append(
                RawEvent(
                    start_time=current.start_time,
                    end_time=following.end_time,
                    notes=sorted(combined_notes.values(), key=lambda note: note.frequency),
                    is_gliss_like=True,
                    primary_note_name=current.primary_note_name,
                    primary_score=max(current.primary_score, following.primary_score),
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


def suppress_leading_gliss_neighbor_noise(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 < len(raw_events):
            next_event = raw_events[index + 1]
            current_duration = current.end_time - current.start_time
            gap = next_event.start_time - current.end_time
            current_note_names = {note.note_name for note in current.notes}
            next_note_names = {note.note_name for note in next_event.notes}
            if (
                len(current.notes) == 2
                and len(next_event.notes) >= 4
                and len(current_note_names & next_note_names) == 1
                and current_duration <= 0.14
                and gap <= GLISS_CLUSTER_MAX_GAP
                and next_event.primary_score >= current.primary_score * 1.2
            ):
                index += 1
                continue
        cleaned.append(current)
        index += 1

    return cleaned


def suppress_bridging_octave_pairs(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = [raw_events[0]]
    for index in range(1, len(raw_events) - 1):
        event = raw_events[index]
        previous = cleaned[-1]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time
        if (
            len(event.notes) == 2
            and len(previous.notes) == 1
            and len(next_event.notes) == 1
            and duration <= BRIDGING_OCTAVE_PAIR_MAX_DURATION
            and event.primary_score <= BRIDGING_OCTAVE_PAIR_MAX_SCORE
            and harmonic_relation_multiple(event.notes[0], event.notes[1]) == 2.0
        ):
            event_note_names = {note.note_name for note in event.notes}
            neighbor_note_names = {previous.notes[0].note_name, next_event.notes[0].note_name}
            if event_note_names == neighbor_note_names:
                continue
        cleaned.append(event)

    cleaned.append(raw_events[-1])
    return cleaned

def suppress_subset_decay_events(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    cleaned: list[RawEvent] = [raw_events[0]]
    for event in raw_events[1:]:
        previous = cleaned[-1]
        gap = event.start_time - previous.end_time
        previous_note_names = {note.note_name for note in previous.notes}
        event_note_names = {note.note_name for note in event.notes}
        if gap <= 0.02 and event_note_names < previous_note_names:
            continue
        cleaned.append(event)

    return cleaned

def suppress_short_residual_tails(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned: list[RawEvent] = [raw_events[0]]
    for index in range(1, len(raw_events) - 1):
        event = raw_events[index]
        next_event = raw_events[index + 1]
        duration = event.end_time - event.start_time
        gap_to_next = next_event.start_time - event.end_time
        if len(event.notes) == 1 and duration <= 0.14 and gap_to_next <= 0.12:
            recent_notes = {note.note_name for note in cleaned[-1].notes}
            if len(cleaned) > 1:
                recent_notes |= {note.note_name for note in cleaned[-2].notes}
            current_note = event.notes[0]
            next_note_names = {note.note_name for note in next_event.notes}
            if current_note.note_name in recent_notes and current_note.note_name not in next_note_names:
                continue
        cleaned.append(event)

    cleaned.append(raw_events[-1])
    return cleaned

def suppress_descending_terminal_residual_cluster(raw_events: list[RawEvent], tuning: InstrumentTuning) -> list[RawEvent]:
    if len(raw_events) < 5:
        return raw_events

    trailing = raw_events[-1]
    previous = raw_events[-2]
    if trailing.is_gliss_like or previous.is_gliss_like or len(trailing.notes) < 2 or len(trailing.notes) > 3:
        return raw_events
    if len(previous.notes) != 1 or (trailing.end_time - trailing.start_time) > 0.28:
        return raw_events

    rank_by_name = {
        note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))
    }
    previous_rank = rank_by_name.get(previous.notes[0].note_name)
    primary_rank = rank_by_name.get(trailing.primary_note_name)
    if previous_rank is None or primary_rank is None or primary_rank != previous_rank + 1:
        return raw_events

    suffix_names: set[str] = {previous.notes[0].note_name}
    expected_rank = previous_rank + 1
    for event in reversed(raw_events[:-2]):
        if len(event.notes) != 1 or event.is_gliss_like:
            break
        note_name = event.notes[0].note_name
        note_rank = rank_by_name.get(note_name)
        if note_rank != expected_rank:
            break
        suffix_names.add(note_name)
        expected_rank += 1
        if len(suffix_names) >= 4:
            break

    trailing_names = {note.note_name for note in trailing.notes}
    if len(suffix_names) < 3 or not trailing_names.issubset(suffix_names):
        return raw_events
    if not any(rank_by_name.get(name, -1) > primary_rank for name in trailing_names):
        return raw_events

    return raw_events[:-1]


def suppress_descending_restart_residual_cluster(raw_events: list[RawEvent], tuning: InstrumentTuning) -> list[RawEvent]:
    if len(raw_events) < 4:
        return raw_events

    rank_by_name = {
        note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))
    }
    cleaned: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        previous_event = cleaned[-1] if cleaned else None
        event = raw_events[index]
        if (
            previous_event is not None
            and len(previous_event.notes) == 1
            and not previous_event.is_gliss_like
            and len(event.notes) == 2
            and not event.is_gliss_like
            and (event.end_time - event.start_time) <= DESCENDING_STEP_HANDOFF_MAX_DURATION
            and index + 1 < len(raw_events)
        ):
            event_names = {note.note_name for note in event.notes}
            event_ranks = sorted(rank_by_name.get(note.note_name, -1) for note in event.notes)
            previous_rank = rank_by_name.get(previous_event.notes[0].note_name)
            if previous_rank is not None and len(event_ranks) == 2 and event_ranks[0] == previous_rank + 1 and event_ranks[1] == previous_rank + 2:
                run_end = index
                while run_end + 1 < len(raw_events) and {note.note_name for note in raw_events[run_end + 1].notes} == event_names:
                    repeated_event = raw_events[run_end + 1]
                    if repeated_event.is_gliss_like or (repeated_event.end_time - repeated_event.start_time) > DESCENDING_STEP_HANDOFF_MAX_DURATION:
                        break
                    run_end += 1
                if run_end + 1 < len(raw_events):
                    restart_event = raw_events[run_end + 1]
                    restart_rank = rank_by_name.get(restart_event.primary_note_name)
                    restart_gap = restart_event.start_time - raw_events[run_end].end_time
                    restart_gap_limit = 0.8
                    if restart_rank is not None and restart_rank >= event_ranks[1] + 10:
                        restart_gap_limit = 1.5
                    if (
                        len(restart_event.notes) == 1
                        and not restart_event.is_gliss_like
                        and restart_rank is not None
                        and restart_rank >= event_ranks[1] + 6
                        and restart_gap <= restart_gap_limit
                    ):
                        index = run_end + 1
                        continue
        cleaned.append(event)
        index += 1

    return cleaned


def collapse_late_descending_step_handoffs(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    cleaned = list(raw_events)
    for index in range(1, len(cleaned) - 1):
        previous_event = cleaned[index - 1]
        event = cleaned[index]
        next_event = cleaned[index + 1]
        duration = event.end_time - event.start_time
        if (
            len(previous_event.notes) == 1
            and len(event.notes) == 2
            and len(next_event.notes) == 1
            and not previous_event.is_gliss_like
            and not next_event.is_gliss_like
            and duration <= DESCENDING_STEP_HANDOFF_MAX_DURATION
        ):
            lower_note, upper_note = sorted(event.notes, key=lambda note: note.frequency)
            if (
                previous_event.notes[0].note_name == upper_note.note_name
                and next_event.notes[0].frequency < lower_note.frequency
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= cents_distance(lower_note.frequency, upper_note.frequency)
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
                and ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS
                <= abs(cents_distance(next_event.notes[0].frequency, lower_note.frequency))
                <= ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS
            ):
                cleaned[index] = RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=[lower_note],
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=lower_note.note_name,
                    primary_score=event.primary_score,
                )

    return cleaned


def merge_adjacent_events(raw_events: list[RawEvent]) -> list[RawEvent]:
    if not raw_events:
        return []

    merged = [raw_events[0]]
    for event in raw_events[1:]:
        previous = merged[-1]
        gap = event.start_time - previous.end_time
        same_notes = [note.note_name for note in previous.notes] == [note.note_name for note in event.notes]
        if same_notes and gap <= 0.12:
            merged[-1] = RawEvent(
                start_time=previous.start_time,
                end_time=event.end_time,
                notes=previous.notes,
                is_gliss_like=previous.is_gliss_like or event.is_gliss_like,
                primary_note_name=previous.primary_note_name,
                primary_score=max(previous.primary_score, event.primary_score),
            )
            continue
        merged.append(event)

    return merged


def split_adjacent_step_dyads_in_ascending_runs(
    raw_events: list[RawEvent],
    tuning: InstrumentTuning,
) -> list[RawEvent]:
    if len(raw_events) < 3:
        return raw_events

    note_rank = {note.note_name: index for index, note in enumerate(sorted(tuning.notes, key=lambda item: item.frequency))}

    def event_rank_range(event: RawEvent) -> tuple[int, int] | None:
        ranks = sorted({note_rank.get(note.note_name, -1) for note in event.notes})
        if not ranks or ranks[0] < 0:
            return None
        return ranks[0], ranks[-1]

    def is_split_candidate(event: RawEvent) -> tuple[int, int] | None:
        if event.is_gliss_like or len(event.notes) != 2 or (event.end_time - event.start_time) > ADJACENT_SEPARATED_DYAD_MAX_DURATION:
            return None
        rank_range = event_rank_range(event)
        if rank_range is None:
            return None
        low_rank, high_rank = rank_range
        if high_rank != low_rank + 1:
            return None
        return low_rank, high_rank

    def forward_support(index: int, current_high_rank: int) -> int:
        support = 0
        previous_high_rank = current_high_rank
        for next_index in range(index + 1, len(raw_events)):
            next_event = raw_events[next_index]
            if len(next_event.notes) > 2:
                break
            if next_event.is_gliss_like and len(next_event.notes) != 1:
                break
            next_range = event_rank_range(next_event)
            if next_range is None:
                break
            next_low_rank, next_high_rank = next_range
            if next_high_rank <= previous_high_rank or next_low_rank < previous_high_rank:
                break
            support += 1
            previous_high_rank = next_high_rank
            if support >= ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT:
                break
        return support

    split_events: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        split_candidate = is_split_candidate(event)
        if split_candidate is None:
            split_events.append(event)
            continue

        low_rank, high_rank = split_candidate
        if forward_support(index, high_rank) < ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT:
            split_events.append(event)
            continue

        previous_same_dyad = False
        if index > 0:
            previous_split_candidate = is_split_candidate(raw_events[index - 1])
            previous_same_dyad = previous_split_candidate == split_candidate
        previous_step_sequence = False
        if index > 1 and len(raw_events[index - 1].notes) == 1 and len(raw_events[index - 2].notes) == 1:
            previous_singleton_ranks = event_rank_range(raw_events[index - 1])
            older_singleton_ranks = event_rank_range(raw_events[index - 2])
            previous_step_sequence = (
                previous_singleton_ranks is not None
                and older_singleton_ranks is not None
                and previous_singleton_ranks[0] == previous_singleton_ranks[1] == high_rank
                and older_singleton_ranks[0] == older_singleton_ranks[1] == low_rank
            )
        next_is_higher_singleton = False
        if index + 1 < len(raw_events):
            next_range = event_rank_range(raw_events[index + 1])
            next_is_higher_singleton = next_range is not None and next_range[0] == next_range[1] and next_range[0] > high_rank
        if previous_step_sequence and next_is_higher_singleton:
            continue
        if previous_same_dyad and next_is_higher_singleton:
            upper_note = max(event.notes, key=lambda note: note.frequency)
            split_events.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=[upper_note],
                    is_gliss_like=False,
                    primary_note_name=upper_note.note_name,
                    primary_score=event.primary_score,
                )
            )
            continue

        ordered_notes = sorted(event.notes, key=lambda note: note.frequency)
        midpoint = event.start_time + ((event.end_time - event.start_time) * 0.5)
        split_events.append(
            RawEvent(
                start_time=event.start_time,
                end_time=midpoint,
                notes=[ordered_notes[0]],
                is_gliss_like=False,
                primary_note_name=ordered_notes[0].note_name,
                primary_score=event.primary_score,
            )
        )
        split_events.append(
            RawEvent(
                start_time=midpoint,
                end_time=event.end_time,
                notes=[ordered_notes[1]],
                is_gliss_like=False,
                primary_note_name=ordered_notes[1].note_name,
                primary_score=event.primary_score,
            )
        )

    return split_events

def merge_short_chord_clusters(raw_events: list[RawEvent]) -> list[RawEvent]:
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    index = 0
    while index < len(raw_events):
        current = raw_events[index]
        if index + 1 >= len(raw_events):
            merged.append(current)
            break

        following = raw_events[index + 1]
        gap = following.start_time - current.end_time
        current_names = {note.note_name for note in current.notes}
        following_names = {note.note_name for note in following.notes}
        combined: dict[str, NoteCandidate] = {note.note_name: note for note in current.notes}
        for note in following.notes:
            combined.setdefault(note.note_name, note)
        combined_keys = sorted(note.key for note in combined.values())
        contiguous_keys = bool(combined_keys) and (combined_keys[-1] - combined_keys[0] + 1 == len(combined_keys))
        combined_count = len(combined)
        total_duration = following.end_time - current.start_time
        is_singleton_plus_dyad = (
            {len(current.notes), len(following.notes)} == {1, 2}
            and not current_names.intersection(following_names)
            and not current.is_gliss_like
            and not following.is_gliss_like
            and min(current.end_time - current.start_time, following.end_time - following.start_time) <= CHORD_CLUSTER_MAX_SINGLETON_DURATION
        )
        is_subset_to_triad = (
            combined_count == 3
            and (current_names < following_names or following_names < current_names)
            and max(len(current.notes), len(following.notes)) == 3
        )
        if (
            gap <= CHORD_CLUSTER_MAX_GAP
            and total_duration <= CHORD_CLUSTER_MAX_TOTAL_DURATION
            and combined_count == 3
            and contiguous_keys
            and (is_singleton_plus_dyad or is_subset_to_triad)
        ):
            merged.append(
                RawEvent(
                    start_time=current.start_time,
                    end_time=following.end_time,
                    notes=sorted(combined.values(), key=lambda note: note.frequency),
                    is_gliss_like=current.is_gliss_like or following.is_gliss_like,
                    primary_note_name=following.primary_note_name,
                    primary_score=max(current.primary_score, following.primary_score),
                )
            )
            index += 2
            continue

        merged.append(current)
        index += 1

    return merged


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

def contiguous_note_cluster(raw_event: RawEvent) -> bool:
    keys = sorted(note.key for note in raw_event.notes)
    if len(keys) < 2:
        return False
    return keys[-1] - keys[0] + 1 == len(keys)
def classify_event_gesture(event: RawEvent, index: int, raw_events: list[RawEvent], merged_events: list[RawEvent]) -> str:
    if len(event.notes) <= 1:
        return "ambiguous"

    support_events = [
        raw_event
        for raw_event in raw_events
        if raw_event.start_time <= event.end_time + 0.03 and raw_event.end_time >= event.start_time - 0.03
    ]
    support_count = len(support_events)
    event_note_names = {note.note_name for note in event.notes}
    support_subsets = [
        {note.note_name for note in raw_event.notes}
        for raw_event in support_events
        if {note.note_name for note in raw_event.notes} < event_note_names
    ]
    all_support_gliss_like = bool(support_events) and all(raw_event.is_gliss_like for raw_event in support_events)
    support_key_ranges = [
        (min(note.key for note in raw_event.notes), max(note.key for note in raw_event.notes), raw_event.start_time)
        for raw_event in support_events
        if raw_event.notes
    ]
    support_key_ranges.sort(key=lambda item: item[2])

    previous_event = merged_events[index - 1] if index > 0 else None
    next_event = merged_events[index + 1] if index + 1 < len(merged_events) else None
    neighbor_progression = False
    for neighbor in (previous_event, next_event):
        if neighbor is None or len(neighbor.notes) != len(event.notes):
            continue
        if not contiguous_note_cluster(event) or not contiguous_note_cluster(neighbor):
            continue
        event_keys = sorted(note.key for note in event.notes)
        neighbor_keys = sorted(note.key for note in neighbor.notes)
        average_shift = float(np.mean([abs(a - b) for a, b in zip(event_keys, neighbor_keys)]))
        if 0.4 <= average_shift <= 3.5 and set(event_keys) != set(neighbor_keys):
            neighbor_progression = True
            break

    if event.is_gliss_like and contiguous_note_cluster(event) and neighbor_progression:
        return "slide_chord"

    if event.is_gliss_like and contiguous_note_cluster(event) and support_subsets:
        event_keys = sorted(note.key for note in event.notes)
        event_max_key = event_keys[-1]
        if support_key_ranges:
            first_min_key, first_max_key, _ = support_key_ranges[0]
            last_min_key, last_max_key, _ = support_key_ranges[-1]
            has_ascending_support_progression = (
                support_count >= 3
                and first_max_key < event_max_key
                and last_max_key >= event_max_key
                and last_min_key > first_min_key
            )
            if has_ascending_support_progression:
                return "slide_chord"
        if all_support_gliss_like:
            return "slide_chord"

    if support_count >= 2 and support_subsets:
        return "slide_chord"

    if not event.is_gliss_like:
        return "strict_chord"

    return "ambiguous"

def build_notation_views(events: list[ScoreEvent]) -> NotationViews:
    western = [" | ".join(note.pitch_class + str(note.octave) for note in event.notes) for event in events]
    numbered = [" ".join(note.label_number for note in event.notes) for event in events]
    vertical = [[note.label_doremi for note in event.notes] for event in events]
    return NotationViews(western=western, numbered=numbered, verticalDoReMi=vertical)

def build_recent_note_names(raw_events: list[RawEvent]) -> set[str] | None:
    if not raw_events:
        return None

    deduped_recent_events: list[RawEvent] = []
    last_note_set: tuple[str, ...] | None = None
    for recent_event in reversed(raw_events):
        note_set = tuple(note.note_name for note in recent_event.notes)
        if note_set == last_note_set:
            continue
        deduped_recent_events.append(recent_event)
        last_note_set = note_set
        if len(deduped_recent_events) >= 4:
            break

    recent_note_names: set[str] = set()
    for recent_event in deduped_recent_events:
        recent_note_names |= {note.note_name for note in recent_event.notes}
    return recent_note_names

def build_recent_ascending_primary_run_ceiling(raw_events: list[RawEvent]) -> float | None:
    if not raw_events:
        return None

    recent_primary_frequencies: list[float] = []
    for recent_event in reversed(raw_events):
        if recent_event.is_gliss_like or len(recent_event.notes) > 2 or recent_event.primary_note_name is None:
            break
        primary_note = next((note for note in recent_event.notes if note.note_name == recent_event.primary_note_name), None)
        if primary_note is None:
            break
        if recent_primary_frequencies and primary_note.frequency > recent_primary_frequencies[-1]:
            break
        recent_primary_frequencies.append(primary_note.frequency)
        if len(recent_primary_frequencies) >= 4:
            break

    if len(recent_primary_frequencies) < ASCENDING_PRIMARY_RUN_MIN_LENGTH:
        return None

    recent_primary_frequencies.reverse()
    if any(current < previous for previous, current in zip(recent_primary_frequencies, recent_primary_frequencies[1:])):
        return None
    if recent_primary_frequencies[-1] <= recent_primary_frequencies[0]:
        return None
    return recent_primary_frequencies[-1]



def build_recent_ascending_singleton_suffix(raw_events: list[RawEvent]) -> tuple[float | None, set[str]]:
    if not raw_events:
        return None, set()

    recent_primary_notes: list[NoteCandidate] = []
    for recent_event in reversed(raw_events):
        if len(recent_event.notes) != 1 or recent_event.primary_note_name is None:
            break
        primary_note = recent_event.notes[0]
        if primary_note.note_name != recent_event.primary_note_name:
            break
        if recent_primary_notes and primary_note.frequency > recent_primary_notes[-1].frequency:
            break
        recent_primary_notes.append(primary_note)
        if len(recent_primary_notes) >= 4:
            break

    if len(recent_primary_notes) < ASCENDING_PRIMARY_RUN_MIN_LENGTH:
        return None, set()

    recent_primary_notes.reverse()
    recent_primary_frequencies = [note.frequency for note in recent_primary_notes]
    if any(current < previous for previous, current in zip(recent_primary_frequencies, recent_primary_frequencies[1:])):
        return None, set()
    if recent_primary_frequencies[-1] <= recent_primary_frequencies[0]:
        return None, set()

    return recent_primary_frequencies[-1], {note.note_name for note in recent_primary_notes}


def build_recent_descending_primary_suffix(raw_events: list[RawEvent]) -> tuple[float | None, float | None, set[str]]:
    if not raw_events:
        return None, None, set()

    recent_primary_notes: list[NoteCandidate] = []
    for recent_event in reversed(raw_events):
        if recent_event.is_gliss_like or len(recent_event.notes) > 2 or recent_event.primary_note_name is None:
            break
        primary_note = next((note for note in recent_event.notes if note.note_name == recent_event.primary_note_name), None)
        if primary_note is None:
            break
        if recent_primary_notes and primary_note.frequency < recent_primary_notes[-1].frequency:
            break
        recent_primary_notes.append(primary_note)
        if len(recent_primary_notes) >= 4:
            break

    if len(recent_primary_notes) < DESCENDING_PRIMARY_SUFFIX_MIN_LENGTH:
        return None, None, set()

    recent_primary_notes.reverse()
    recent_primary_frequencies = [note.frequency for note in recent_primary_notes]
    if any(current > previous for previous, current in zip(recent_primary_frequencies, recent_primary_frequencies[1:])):
        return None, None, set()
    if recent_primary_frequencies[-1] >= recent_primary_frequencies[0]:
        return None, None, set()

    return (
        recent_primary_frequencies[-1],
        recent_primary_frequencies[0],
        {note.note_name for note in recent_primary_notes},
    )


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
