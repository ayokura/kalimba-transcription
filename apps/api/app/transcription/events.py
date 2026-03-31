from __future__ import annotations

from ..models import InstrumentTuning
from .audio import cents_distance
from .constants import *
from .models import NoteCandidate, RawEvent
from .peaks import are_harmonic_related, harmonic_relation_multiple, is_adjacent_tuning_step

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
