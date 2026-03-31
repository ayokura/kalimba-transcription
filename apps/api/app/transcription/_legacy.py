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
from .patterns import (
    REPEATED_PATTERN_PASS_IDS,
    apply_repeated_pattern_passes,
    suppress_recent_upper_echo_mixed_clusters,
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
