from __future__ import annotations

from typing import Sequence

import numpy as np

from ..models import InstrumentTuning
from .audio import cents_distance
from .constants import *
from .models import Note, NoteCandidate, RawEvent
from .noise_floor import NoiseFloorMeasurement
from .peaks import (
    are_harmonic_related,
    harmonic_relation_multiple,
    is_adjacent_tuning_step,
    measure_narrow_fft_note_scores,
    onset_backward_attack_gain,
    pick_matching_sub_onset,
)

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

def suppress_onset_decaying_carryover(raw_events: list[RawEvent]) -> list[RawEvent]:
    """Remove secondary notes whose energy is decaying from the previous primary.

    When a note appears as a non-primary secondary with onset_gain < 1.0
    (energy strictly decreasing at the segment onset) and matches the
    previous event's primary, it is residual resonance rather than a new
    attack.  Removing it here avoids the cascade risk of rejecting inside
    segment_peaks (see #107 analysis).
    """
    if not raw_events:
        return []

    result = [raw_events[0]]
    for event in raw_events[1:]:
        prev_primary = result[-1].primary_note_name
        prev_note_names = {n.note_name for n in result[-1].notes}
        curr_primary = event.primary_note_name

        # If current primary was also in the previous event, this looks like
        # a chord re-attack / continuation — don't strip secondaries.
        if curr_primary in prev_note_names:
            result.append(event)
            continue

        filtered = [
            note for note in event.notes
            if not (
                note.note_name != curr_primary
                and note.note_name == prev_primary
                and note.onset_gain is not None
                and note.onset_gain < 1.0
            )
        ]

        if len(filtered) == len(event.notes) or len(filtered) == 0:
            result.append(event)
        else:
            result.append(RawEvent(
                start_time=event.start_time,
                end_time=event.end_time,
                notes=filtered,
                is_gliss_like=event.is_gliss_like,
                primary_note_name=event.primary_note_name,
                primary_score=event.primary_score,
            ))

    return result


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
        # Short-segment-guarded primaries are tentative single-note attacks
        # from windows too narrow for full FFT analysis.  They must not be
        # treated as residual tails of recently-played same-name notes,
        # because the carryover-overlap signature this function uses cannot
        # distinguish a genuine new attack from a decay tail in such windows.
        # Future per-sub-onset narrow FFT (#141 follow-up) will provide a
        # better recovery path for these.
        if event.from_short_segment_guard:
            cleaned.append(event)
            continue
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


def recover_masked_reattack_via_narrow_fft(
    raw_events: list[RawEvent],
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    *,
    noise_floor: NoiseFloorMeasurement | None = None,
    min_fundamental_energy: float = NARROW_FFT_REATTACK_MIN_ENERGY,
    noise_floor_factor: float = NARROW_FFT_REATTACK_NOISE_FACTOR,
    noise_floor_hard_floor: float = NARROW_FFT_NOISE_THRESHOLD_HARD_FLOOR,
    min_fundamental_ratio: float = NARROW_FFT_REATTACK_MIN_FR,
    min_score_dominance_ratio: float = NARROW_FFT_REATTACK_DOMINANCE_RATIO,
    dissonance_neighbor_cents: float = NARROW_FFT_REATTACK_DISSONANCE_CENTS,
) -> list[RawEvent]:
    """Recover a chord note that was rejected as ``weak-secondary-onset`` /
    ``descending-restart-upper-carryover`` because the segment-wide FFT
    could not separate its re-attack from prior sustain (#153 Phase A.4).

    Motivating cases (34-key BWV147):
        - R1 E97 ``<F5,D5,B4,G4>``: D5 is rank 1 in narrow FFT at the
          segment start (fund_e ≈ 36, fr ≈ 0.99) but rejected by the
          weak-secondary gate; pick_matching_sub_onset still locates a
          real D5 attack at a later sub-onset of the segment.
        - R4 E133 ``<F5,D5,B4,G4>``: same shape, with D5 rejected via
          ``score-below-threshold,descending-restart-upper-carryover``.

    Multiple complementary safeguards prevent false positives:

    1. The segment must have ≥2 sub-onsets.  A single-sub-onset segment
       cannot distinguish a re-attack from sustain residue, since the
       only available pre/post comparison sits at the segment boundary.

    2. The candidate must be the highest-scoring narrow-FFT note that
       is not already in the event AND its score must be at least
       ``min_score_dominance_ratio`` (default 1.5) times the highest
       score among the event's existing notes in the same narrow FFT.
       This signals "this rejected note dominates the early window
       even though the segment-wide FFT picked something else".

    3. fundamental_ratio ≥ 0.95 — a sharp fundamental, not spectral
       leakage from a near-frequency neighbour.

    4. fundamental_energy ≥ 30 — a meaningful absolute presence.

    5. Not within ``dissonance_neighbor_cents`` (default 200 cents = whole
       step) of any existing event note.  This also rejects whole-step
       dissonance such as the d6-to-e6 alternating sequence where E6 from
       a previous event would otherwise be added on top of a D6 attack.

    6. ``pick_matching_sub_onset`` returns a non-first sub-onset — the
       attack rise must occur *after* the segment start, because the
       start sub-onset is where the segment's existing primaries already
       attack and any post/pre rise there could be a sustained-residue
       artefact.

    Only one note is added per event per pass.
    """
    if not raw_events:
        return raw_events

    recovered: list[RawEvent] = []
    for event in raw_events:
        if event.from_short_segment_guard:
            recovered.append(event)
            continue
        if len(event.notes) >= MAX_POLYPHONY:
            recovered.append(event)
            continue
        # Require at least 3 sub-onsets in the segment.  This is a strong
        # physical signal that the segmenter identified multiple attack
        # moments inside the segment (gliss-like, slide chord, etc.) — a
        # 2-sub-onset segment (just start + end) is a normal single-attack
        # event where masked re-attack recovery has no temporal handle.
        # Empirically all observed false positives in 17-key + 34-key
        # BWV147 had 2 sub-onsets while the true E97 / E133 D5 cases had 3.
        if len(event.sub_onsets) < 3:
            recovered.append(event)
            continue
        narrow_scores = measure_narrow_fft_note_scores(
            audio, sample_rate, event.start_time, tuning,
        )
        if narrow_scores is None:
            recovered.append(event)
            continue
        existing_names = {note.note_name for note in event.notes}
        existing_freqs = [note.frequency for note in event.notes]
        # Highest score among existing notes in this narrow FFT — the
        # bar a recovered candidate must clearly clear.
        existing_max_score = 0.0
        for note in event.notes:
            entry = narrow_scores.get(note.note_name)
            if entry is None:
                continue
            existing_max_score = max(existing_max_score, entry[1])
        # The candidate must be the rank-1 narrow-FFT entry that is not
        # already in the event.  Iterate by descending score.
        ranked = sorted(
            narrow_scores.items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
        added = False
        for note_name, (fund_e, score, fr) in ranked:
            if note_name in existing_names:
                # Skip notes already present, but they still count toward
                # rank: the first non-existing entry is "rank 1 among
                # candidates", which is what we want.
                continue
            # Phase B (#154): noise-floor-aware energy threshold; see
            # merge_short_segment_guard_via_narrow_fft for the rationale.
            if noise_floor is not None:
                min_energy = noise_floor.threshold_for(
                    note_name,
                    factor=noise_floor_factor,
                    fallback=min_fundamental_energy,
                    hard_floor=noise_floor_hard_floor,
                )
            else:
                min_energy = min_fundamental_energy
            if fund_e < min_energy:
                break
            if fr < min_fundamental_ratio:
                # Spectral leakage signature — skip; downstream entries
                # are weaker, so break.
                break
            if (
                existing_max_score > 0.0
                and score < existing_max_score * min_score_dominance_ratio
            ):
                # The candidate is not dominantly stronger than the
                # event's existing notes — likely the segment-wide
                # selection is correct, no recovery needed.
                break
            tuning_match = next(
                (n for n in tuning.notes if n.note_name == note_name),
                None,
            )
            if tuning_match is None:
                continue
            candidate_freq = tuning_match.frequency
            if any(
                cents_distance(candidate_freq, existing_freq)
                <= dissonance_neighbor_cents
                for existing_freq in existing_freqs
            ):
                continue
            matched_sub_onset = pick_matching_sub_onset(
                audio, sample_rate, candidate_freq, event.sub_onsets,
            )
            if matched_sub_onset is None:
                continue
            # Re-attacks occur after the segment start, not at it.
            if matched_sub_onset <= event.sub_onsets[0] + 1e-6:
                continue
            new_candidate = NoteCandidate(
                key=tuning_match.key,
                note=Note.from_name(note_name),
                score=score,
            )
            updated_notes = sorted(
                [*event.notes, new_candidate],
                key=lambda candidate: candidate.frequency,
            )
            recovered.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=updated_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name,
                    primary_score=event.primary_score,
                    from_short_segment_guard=event.from_short_segment_guard,
                    sub_onsets=event.sub_onsets,
                )
            )
            added = True
            break
        if not added:
            recovered.append(event)

    return recovered


def recover_pre_segment_attack_via_narrow_fft(
    raw_events: list[RawEvent],
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    onset_times: "Sequence[float]",
    *,
    noise_floor: NoiseFloorMeasurement | None = None,
    lookback_seconds: float = NARROW_FFT_PRE_SEGMENT_LOOKBACK_SECONDS,
    min_fundamental_energy: float = NARROW_FFT_PRE_SEGMENT_MIN_ENERGY,
    noise_floor_factor: float = NARROW_FFT_PRE_SEGMENT_NOISE_FACTOR,
    noise_floor_hard_floor: float = NARROW_FFT_PRE_SEGMENT_HARD_FLOOR,
    min_fundamental_ratio: float = NARROW_FFT_PRE_SEGMENT_MIN_FR,
    min_backward_attack_gain: float = NARROW_FFT_PRE_SEGMENT_MIN_BACKWARD_GAIN,
    dissonance_neighbor_cents: float = NARROW_FFT_PRE_SEGMENT_DISSONANCE_CENTS,
    onset_consumed_tolerance: float = NARROW_FFT_PRE_SEGMENT_ONSET_CONSUMED_TOLERANCE,
    decay_min_ratio: float = NARROW_FFT_PRE_SEGMENT_DECAY_MIN_RATIO,
    bg_dominance_ratio: float = NARROW_FFT_PRE_SEGMENT_BG_DOMINANCE_RATIO,
) -> list[RawEvent]:
    """Recover a chord note that attacks in the gap before an event from
    a broadband onset that the segmenter did not materialize (#153
    Phase B / #154 lookback rescue).

    Motivating case: 17-key BWV147 E97 ``<F5,D5,B4,G4>``.  The broadband
    onset detector reports an onset at 168.0827s — well inside the G4
    attack window (167.98–168.10s).  However the segmenter starts the
    next segment at 168.152s, after G4 has decayed, so segment_peaks
    rejects G4 as ``score-below-threshold,recent-carryover-candidate``
    and the merge / re-attack rescue passes never see it because no
    segment contains the G4 attack.

    The pass walks unconsumed onsets (those that do not fall inside any
    existing event), runs a narrow FFT centred on each, and adds the
    highest-scoring candidate not already in the event to the
    immediately following event when every gate passes:

    Candidates are evaluated in **descending backward_attack_gain
    order** rather than narrow-FFT score order.  Score-ordered
    iteration biases toward whatever is loudest in the early window
    (often a sympathetic-resonance peak from an incoming attack);
    bg-ordered iteration puts the strongest *fresh-attack signature*
    first, which is exactly what the rescue is looking for.  The
    first candidate that passes every gate below is added to the
    event; later candidates are not considered.

    Gates (in evaluation order):

    1. ``backward_attack_gain ≥ min_backward_attack_gain`` —
       primary fresh-attack-vs-sustain discriminator.  Iteration
       breaks once this falls below the bar because lower-bg
       candidates are ranked even worse.
    2. ``fundamental_ratio ≥ min_fundamental_ratio`` — clean
       fundamental, not spectral leakage.
    3. ``fundamental_energy ≥ noise_floor[note] * noise_floor_factor``
       clamped to ``noise_floor_hard_floor`` (or
       ``min_fundamental_energy`` when calibration is unavailable).
    4. The candidate is not within ``dissonance_neighbor_cents`` of
       any existing event note (whole-step rejection).
    5. ``fund_e_onset / fund_e_segment_start ≥ decay_min_ratio``:
       the candidate's energy must not be RISING into the segment.
       A rising candidate is part of the upcoming chord and was
       already evaluated by segment_peaks.  Example: 34-key R5 E154
       D4 with fund_e 1.5 → 8.3 (ratio 0.18).

    6. ``rescue_bg ≥ max_in_event_bg * bg_dominance_ratio``:
       distinguishes a *separate pre-segment attack* (in-event
       notes have low bg, the rescue note's bg dominates) from an
       *early sample of the same chord attack* (in-event notes are
       themselves mid-attack with very high bg, the rescue note is
       a sympathetic-resonance neighbour pumped up by the incoming
       attack).  Example: 17-key d4-d5 octave-dyad fixture
       18.1173s where D5 (in-event) has bg 17037 because the
       segmenter is about to detect it 35 ms later — any rescue at
       that onset would be promoting a resonance peak.

    7. The candidate is not already present in the event and the
       event has room for another note under :data:`MAX_POLYPHONY`.

    The lookback window is bounded above by the previous event's
    ``end_time`` so the rescue cannot reach back across an earlier
    chord and steal one of its tail notes.  Within the window the
    *latest* unconsumed onset is preferred because narrow FFT energy
    is highest near the actual attack moment.
    """
    if not raw_events or not onset_times:
        return raw_events

    sorted_onsets = sorted({float(t) for t in onset_times})

    def _is_consumed(onset: float) -> bool:
        for event in raw_events:
            if (
                event.start_time - onset_consumed_tolerance
                <= onset
                <= event.end_time + onset_consumed_tolerance
            ):
                return True
        return False

    unconsumed = [o for o in sorted_onsets if not _is_consumed(o)]
    if not unconsumed:
        return raw_events

    recovered: list[RawEvent] = []
    for index, event in enumerate(raw_events):
        if event.from_short_segment_guard or len(event.notes) >= MAX_POLYPHONY:
            recovered.append(event)
            continue
        # Bound the lookback above by the previous event's end so the
        # rescue cannot reach across an earlier chord and adopt one of
        # its tail-decay onsets.
        prev_event_end = raw_events[index - 1].end_time if index > 0 else 0.0
        floor = max(event.start_time - lookback_seconds, prev_event_end)
        candidates_in_window = [
            o for o in unconsumed if floor < o < event.start_time
        ]
        if not candidates_in_window:
            recovered.append(event)
            continue
        # Latest unconsumed onset within the window — narrow-FFT energy
        # is highest near the attack moment, so picking the latest
        # gives the strongest signal for the closest pre-event note.
        onset_time = candidates_in_window[-1]

        narrow_scores = measure_narrow_fft_note_scores(
            audio, sample_rate, onset_time, tuning,
        )
        if narrow_scores is None:
            recovered.append(event)
            continue
        # Computed once for the decay-pattern discriminator.  The
        # pass should only rescue notes whose energy is DECAYING
        # SLOWLY from the unconsumed onset to the segment start.
        segment_start_scores = measure_narrow_fft_note_scores(
            audio, sample_rate, event.start_time, tuning,
        )

        existing_names = {note.note_name for note in event.notes}
        existing_freqs = [note.frequency for note in event.notes]

        # Pre-compute backward_attack_gain for every tuning note
        # that surfaced in the narrow FFT.  In-event bgs feed the
        # bg-dominance gate (gate 6); non-event bgs are sorted in
        # descending order so the iteration tries the strongest
        # fresh-attack candidate first regardless of narrow-FFT
        # score (which can be dominated by an incoming chord's
        # loudest sustain).
        max_in_event_bg = 0.0
        bg_candidates: list[tuple[float, str, float, float, float, object, float]] = []
        for note_name, (fund_e, score, fr) in narrow_scores.items():
            tuning_match = next(
                (n for n in tuning.notes if n.note_name == note_name),
                None,
            )
            if tuning_match is None:
                continue
            candidate_freq = tuning_match.frequency
            backward_gain = onset_backward_attack_gain(
                audio, sample_rate, onset_time, candidate_freq,
            )
            if note_name in existing_names:
                if backward_gain > max_in_event_bg:
                    max_in_event_bg = backward_gain
                continue
            bg_candidates.append(
                (backward_gain, note_name, fund_e, score, fr, tuning_match, candidate_freq)
            )
        bg_candidates.sort(key=lambda c: c[0], reverse=True)

        added = False
        for backward_gain, note_name, fund_e, score, fr, tuning_match, candidate_freq in bg_candidates:
            # Lower-bg candidates only get worse — bail out instead of
            # walking through a long tail of sustain residue.
            if backward_gain < min_backward_attack_gain:
                break
            # Bg dominance: if any in-event note is itself mid-attack
            # at the unconsumed onset, the onset is just an early
            # detection of the same chord attack and the rescue
            # would only promote a sympathetic-resonance neighbour.
            if (
                max_in_event_bg > 0.0
                and backward_gain < max_in_event_bg * bg_dominance_ratio
            ):
                break
            if fr < min_fundamental_ratio:
                continue
            if noise_floor is not None:
                min_energy = noise_floor.threshold_for(
                    note_name,
                    factor=noise_floor_factor,
                    fallback=min_fundamental_energy,
                    hard_floor=noise_floor_hard_floor,
                )
            else:
                min_energy = min_fundamental_energy
            if fund_e < min_energy:
                continue
            if any(
                cents_distance(candidate_freq, existing_freq)
                <= dissonance_neighbor_cents
                for existing_freq in existing_freqs
            ):
                continue
            if segment_start_scores is not None:
                start_fund_e = segment_start_scores.get(
                    note_name, (0.0, 0.0, 0.0),
                )[0]
                if start_fund_e > 0.0:
                    ratio = fund_e / start_fund_e
                    if ratio < decay_min_ratio:
                        continue
            new_candidate = NoteCandidate(
                key=tuning_match.key,
                note=Note.from_name(note_name),
                score=score,
            )
            updated_notes = sorted(
                [*event.notes, new_candidate],
                key=lambda candidate: candidate.frequency,
            )
            recovered.append(
                RawEvent(
                    start_time=event.start_time,
                    end_time=event.end_time,
                    notes=updated_notes,
                    is_gliss_like=event.is_gliss_like,
                    primary_note_name=event.primary_note_name,
                    primary_score=event.primary_score,
                    from_short_segment_guard=event.from_short_segment_guard,
                    sub_onsets=event.sub_onsets,
                )
            )
            added = True
            break
        if not added:
            recovered.append(event)

    return recovered


def merge_gliss_split_segments(
    raw_events: list[RawEvent],
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    *,
    max_gap_seconds: float = GLISS_SPLIT_MERGE_MAX_GAP,
    max_first_duration: float = GLISS_SPLIT_MERGE_MAX_FIRST_DURATION,
    semitone_neighbor_cents: float = GLISS_SPLIT_MERGE_SEMITONE_CENTS,
) -> list[RawEvent]:
    """Merge two adjacent non-guarded segments that are different parts of
    the same gliss/chord (#153 Phase A.3).

    Two patterns are unified by a single union-with-semitone-dedup rule:

    Pattern A — gliss prefix splitting (e.g., 34-key R3 E121, E127):
        A short prefix segment (~50 ms) of a chord attack is detected
        independently with a few of the chord's notes plus spectral-leakage
        semitone artifacts (e.g., D#4 leaking from E4, or C5 leaking from
        B4).  The longer main segment that follows correctly captures the
        chord.  Merging the prefix into the main, then dropping any prefix
        notes that have a semitone neighbor in the union, eliminates the
        spurious extras while preserving the main segment's exact match.

    Pattern B — gliss late-note splitting (e.g., 34-key R1 E97, R4 E133):
        The early head of a 4-note gliss is detected as a chord (e.g.,
        ``[B4,G4]``) and the later F5 attack lands as a separate adjacent
        segment ``[F5,B4]``.  The expected chord ``<F5,D5,B4,G4>`` cannot
        match the head alone, but merging the F5 into the head segment
        recovers a 3/4 superset (D5 still requires Phase A.4).

    Both patterns share:
        - Adjacent in time (gap below ``max_gap_seconds``).
        - At least one shared note between the two segments — confirms
          they are parts of the same chord/gliss rather than two unrelated
          neighbouring events.
        - Neither segment is short-segment-guarded (Phase A.2 handles
          guarded singletons separately).
        - The earlier segment is gliss-like (duration below
          ``max_first_duration``).

    Notes from the longer segment always win in semitone conflicts: the
    longer segment has better FFT resolution and more reliable per-note
    energies, so its primary set is the trustworthy ground.  A note from
    the shorter segment is added only when (a) its frequency is more than
    ``semitone_neighbor_cents`` from any note already in the merged set
    and (b) its onset_backward_attack_gain at the segment start exceeds
    ``new_note_min_backward_gain`` — the same residual-sustain guard used
    in Phase A.2.
    """
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    skip_next = False
    for index, event in enumerate(raw_events):
        if skip_next:
            skip_next = False
            continue
        if index == len(raw_events) - 1:
            merged.append(event)
            continue
        next_event = raw_events[index + 1]
        if event.from_short_segment_guard or next_event.from_short_segment_guard:
            merged.append(event)
            continue
        gap = next_event.start_time - event.end_time
        if gap < 0 or gap > max_gap_seconds:
            merged.append(event)
            continue
        first_duration = event.end_time - event.start_time
        if first_duration > max_first_duration:
            merged.append(event)
            continue
        # Build the merged note set by union with semitone dedup.  Notes
        # from the longer event win in semitone conflicts (better FFT
        # resolution → more reliable per-note energies).  Both events were
        # already validated by segment_peaks, so no extra per-note narrow
        # FFT confirmation is needed here — the goal is to consolidate
        # gliss-split fragments back into a single event.
        second_duration = next_event.end_time - next_event.start_time
        if second_duration >= first_duration:
            longer_event, shorter_event = next_event, event
        else:
            longer_event, shorter_event = event, next_event
        merged_notes = list(longer_event.notes)
        merged_names = {note.note_name for note in merged_notes}
        for note in shorter_event.notes:
            if note.note_name in merged_names:
                continue
            if any(
                cents_distance(note.frequency, existing.frequency)
                <= semitone_neighbor_cents
                for existing in merged_notes
            ):
                continue
            merged_notes.append(note)
            merged_names.add(note.note_name)
        if len(merged_notes) > MAX_POLYPHONY:
            merged.append(event)
            continue
        # Dissonance guard: reject the merge if the resulting note set
        # contains any whole-step (≤200 cents) pair, which is the dominant
        # signature of a false merge between two unrelated short events
        # (e.g., end of one melodic phrase running into the start of the
        # next).  Genuine gliss splits and chord recoveries (E97, E121,
        # E148, etc.) all produce 3rds, 4ths, octaves and wider intervals.
        rejected_by_dissonance = False
        for i, note_i in enumerate(merged_notes):
            for note_j in merged_notes[i + 1:]:
                if cents_distance(note_i.frequency, note_j.frequency) <= 200.0:
                    rejected_by_dissonance = True
                    break
            if rejected_by_dissonance:
                break
        if rejected_by_dissonance:
            merged.append(event)
            continue
        merged_notes.sort(key=lambda candidate: candidate.frequency)
        merged.append(
            RawEvent(
                start_time=event.start_time,
                end_time=next_event.end_time,
                notes=merged_notes,
                is_gliss_like=event.is_gliss_like or next_event.is_gliss_like,
                primary_note_name=longer_event.primary_note_name,
                primary_score=longer_event.primary_score,
                from_short_segment_guard=False,
                sub_onsets=tuple(sorted(set(event.sub_onsets) | set(next_event.sub_onsets))),
            )
        )
        skip_next = True

    return merged


def suppress_unmerged_guarded_singletons(
    raw_events: list[RawEvent],
) -> list[RawEvent]:
    """Drop short-segment-guarded singletons that no upstream pass merged
    into a real chord (#153 cosmetic extras follow-up).

    Pre-condition: this pass runs *after*
    :func:`merge_short_segment_guard_via_narrow_fft` (Phase A.2), which
    rejoins guarded primaries into adjacent chords whenever the four
    narrow-FFT safeguards (energy / fundamental_ratio / backward_gain /
    next-primary dominance) confirm a real attack.  Any RawEvent that
    still carries ``from_short_segment_guard=True`` at this point failed
    that cross-validation, which is strong evidence the singleton is a
    spectral artefact of a nearby real attack — a 6-16 ms FFT window
    cannot resolve secondaries reliably, so the guard preserved a
    tentative primary that none of the recovery passes could attach to
    a real chord.

    The drop is unconditional within the guard scope: the four A.2
    disambiguators have already separated real cases (which became
    chord notes) from artefact cases (which still have the flag).
    """
    if not raw_events:
        return raw_events
    return [event for event in raw_events if not event.from_short_segment_guard]


def merge_short_segment_guard_via_narrow_fft(
    raw_events: list[RawEvent],
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    *,
    noise_floor: NoiseFloorMeasurement | None = None,
    max_gap_seconds: float = 0.05,
    narrow_fft_min_energy: float = NARROW_FFT_GUARD_MERGE_MIN_ENERGY,
    noise_floor_factor: float = NARROW_FFT_GUARD_MERGE_NOISE_FACTOR,
    noise_floor_hard_floor: float = NARROW_FFT_NOISE_THRESHOLD_HARD_FLOOR,
    next_primary_dominance_ratio: float = NARROW_FFT_GUARD_MERGE_DOMINANCE_RATIO,
    min_fundamental_ratio: float = NARROW_FFT_GUARD_MERGE_MIN_FR,
    min_backward_attack_gain: float = NARROW_FFT_GUARD_MERGE_MIN_BACKWARD_GAIN,
) -> list[RawEvent]:
    """Merge a short-segment-guarded singleton into the following event when
    its primary note is independently confirmed by a narrow FFT centred on
    the next event's start (#153 Phase A.2).

    Motivating case: 17-key BWV147 E148 ``[C6,<C5,A4>]`` produces two
    consecutive segments — ``[*-*]`` (≈7 ms guarded singleton with C6) and
    ``[*-*]`` (≈1.13 s with C5+A4) — and the C6 singleton must rejoin the
    chord to recover an exact match.  The cross-segment merge passes that
    follow only merge identical note sets, so a dedicated pass is needed.

    The narrow FFT confirmation is essential because the C6 fundamental at
    1046 Hz is identical to C5's 2nd harmonic.  Inside the segment-wide FFT
    of the next event the bin is already dominated by C5 sustain; only the
    early ~30 ms window centred on the second event's start time still
    shows the C6 attack as an independent peak.

    The merge fires only when:
        1. The first event is short-segment-guarded.
        2. The first event has exactly one note (the guarded primary).
        3. The gap to the next event is below ``max_gap_seconds`` (default 50 ms).
        4. The merged note set stays within ``MAX_POLYPHONY``.
        5. The guarded primary is not already in the next event's notes.
        6. A narrow FFT centred on the next event's start time reports the
           guarded primary's fundamental energy at or above
           ``narrow_fft_min_energy`` (Phase A fixed threshold; Phase B
           replaces this with a per-band noise floor multiplier — #154).
        7. The guarded primary's narrow-FFT score is not dwarfed by the
           next event's primary at the same instant — specifically,
           ``guarded_score >= next_primary_score * next_primary_dominance_ratio``.
           This is the key disambiguator between a fresh octave-coincident
           attack (E148 C6: at the merge point C5 has not yet started, so
           C5 is absent from the narrow FFT and C6 dominates as the only
           upper-register peak) and a sustain leakage from a recently
           played same-name note (R6 E161 cosmetic A4: at the merge point
           C5 is already at full energy in the narrow FFT, so any A4
           presence must be carryover, not a fresh attack).
    """
    if len(raw_events) < 2:
        return raw_events

    merged: list[RawEvent] = []
    skip_next = False
    for index, event in enumerate(raw_events):
        if skip_next:
            skip_next = False
            continue
        if index == len(raw_events) - 1:
            merged.append(event)
            continue
        next_event = raw_events[index + 1]
        if not event.from_short_segment_guard:
            merged.append(event)
            continue
        if len(event.notes) != 1:
            merged.append(event)
            continue
        gap = next_event.start_time - event.end_time
        if gap < 0 or gap > max_gap_seconds:
            merged.append(event)
            continue
        guarded_note = event.notes[0]
        next_note_names = {note.note_name for note in next_event.notes}
        if guarded_note.note_name in next_note_names:
            merged.append(event)
            continue
        if len(next_event.notes) + 1 > MAX_POLYPHONY:
            merged.append(event)
            continue
        narrow_scores = measure_narrow_fft_note_scores(
            audio, sample_rate, next_event.start_time, tuning,
        )
        if narrow_scores is None:
            merged.append(event)
            continue
        guarded_energy, guarded_score, guarded_fr = narrow_scores.get(
            guarded_note.note_name, (0.0, 0.0, 0.0),
        )
        # Phase B (#154): noise-floor-aware energy threshold.  When a
        # per-recording per-band noise floor is available, use
        # ``floor[note] * noise_floor_factor`` (clamped to a hard
        # floor) instead of the Phase A absolute threshold.  This lets
        # the same factor adapt to mic gain and per-band noise across
        # recordings while still rejecting trivial noise picks.
        if noise_floor is not None:
            min_energy = noise_floor.threshold_for(
                guarded_note.note_name,
                factor=noise_floor_factor,
                fallback=narrow_fft_min_energy,
                hard_floor=noise_floor_hard_floor,
            )
        else:
            min_energy = narrow_fft_min_energy
        if guarded_energy < min_energy:
            merged.append(event)
            continue
        # A real fresh attack has fundamental >> harmonics in the narrow
        # window.  Spectral leakage from a nearby semitone shows a much
        # lower fundamental_ratio (e.g., 34-key L1 6.298s B4 fr=0.674
        # leaking from A4 area, vs E148 C6 fr=0.903 as a real attack).
        if guarded_fr < min_fundamental_ratio:
            merged.append(event)
            continue
        # Sustain-vs-attack disambiguation.  Even when fundamental_ratio is
        # high, the guarded note can still be a residual sustain from a
        # recently played same-name note (e.g., 34-key R3 64.418s B4 with
        # fr=0.980 is leftover sustain from E113 [<B4,G4>,D4] ~1.4s prior).
        # onset_backward_attack_gain compares early-window energy to a
        # 200ms-prior reference; a fresh attack shows a high ratio while a
        # decayed-but-still-ringing sustain shows a low ratio.
        backward_gain = onset_backward_attack_gain(
            audio, sample_rate, next_event.start_time,
            guarded_note.frequency,
        )
        if backward_gain < min_backward_attack_gain:
            merged.append(event)
            continue
        next_primary_name = next_event.primary_note_name
        _, next_primary_score, _ = narrow_scores.get(
            next_primary_name, (0.0, 0.0, 0.0),
        )
        if (
            next_primary_score > 0.0
            and guarded_score < next_primary_score * next_primary_dominance_ratio
        ):
            merged.append(event)
            continue
        combined_notes = sorted(
            [*next_event.notes, guarded_note],
            key=lambda candidate: candidate.frequency,
        )
        merged.append(
            RawEvent(
                start_time=event.start_time,
                end_time=next_event.end_time,
                notes=combined_notes,
                is_gliss_like=next_event.is_gliss_like,
                primary_note_name=next_event.primary_note_name,
                primary_score=next_event.primary_score,
                from_short_segment_guard=False,
                sub_onsets=tuple(sorted(set(event.sub_onsets) | set(next_event.sub_onsets))),
            )
        )
        skip_next = True

    return merged

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
