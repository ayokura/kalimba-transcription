from __future__ import annotations

from typing import Any

from fastapi import HTTPException, UploadFile

from ..models import InstrumentTuning, ScoreEvent, ScoreNote, TranscriptionResult
from .audio import read_audio
from .constants import GAP_RUN_LEAD_IN_MIN_FOLLOWUP_GAP, SHORT_SEGMENT_SECONDARY_GUARD_DURATION
from .events import (
    build_recent_ascending_primary_run_ceiling,
    build_recent_ascending_singleton_suffix,
    build_recent_descending_primary_suffix,
    build_recent_note_names,
    classify_event_gesture,
    collapse_ascending_restart_lower_residue_singletons,
    collapse_high_register_adjacent_bridge_dyads,
    collapse_late_descending_step_handoffs,
    collapse_same_start_primary_singletons,
    merge_adjacent_events,
    merge_four_note_gliss_clusters,
    merge_gliss_split_segments,
    merge_short_chord_clusters,
    merge_short_gliss_clusters,
    merge_short_segment_guard_via_narrow_fft,
    recover_masked_reattack_via_narrow_fft,
    recover_pre_segment_attack_via_narrow_fft,
    suppress_unmerged_guarded_singletons,
    simplify_descending_adjacent_dyad_residue,
    simplify_short_gliss_prefix_to_contiguous_singleton,
    simplify_short_secondary_bleed,
    split_adjacent_step_dyads_in_ascending_runs,
    split_ambiguous_upper_octave_pairs,
    suppress_bridging_octave_pairs,
    suppress_descending_upper_return_overlap,
    suppress_descending_upper_singleton_spikes,
    suppress_leading_descending_overlap,
    suppress_leading_gliss_neighbor_noise,
    suppress_leading_gliss_subset_transients,
    suppress_leading_single_transient,
    suppress_low_confidence_dyad_transients,
    suppress_onset_decaying_carryover,
    suppress_post_tail_gap_bridge_dyads,
    suppress_resonant_carryover,
    suppress_short_descending_return_singletons,
    suppress_short_residual_tails,
    suppress_subset_decay_events,
)
from .models import NoteCandidate, RawEvent
from .noise_floor import measure_noise_floor
from .notation import build_notation_views, format_doremi, format_number, quantize_beat
from .patterns import apply_repeated_pattern_passes
from .peaks import segment_peaks
from .per_note import rescue_gap_mute_dips
from .segments import (
    build_segment_debug_contexts,
    detect_segments,
    should_keep_low_register_sparse_gap_tail,
    simplify_sparse_gap_tail_high_octave_dyad,
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
    segments, tempo, segment_debug = detect_segments(
        audio, sample_rate,
        mid_performance_start=mid_performance_start,
        mid_performance_end=mid_performance_end,
    )
    segments = rescue_gap_mute_dips(segments, audio, sample_rate, tuning)

    # #154 / #153 Phase B: per-recording per-band noise floor calibration.
    # Sample silent gaps between segments with the same narrow-FFT window
    # used by the Phase A merge passes so the result is directly
    # comparable to ``measure_narrow_fft_note_scores`` output.  Returns an
    # empty measurement on synthetic / silence-poor fixtures, in which
    # case the merge passes fall back to their Phase A absolute thresholds.
    noise_floor = measure_noise_floor(audio, sample_rate, tuning, segments)

    raw_events: list[RawEvent] = []
    segment_candidates_debug: list[dict[str, Any]] = []
    all_onset_times = [float(value) for value in segment_debug.get("onsetTimes", [])]
    segment_contexts = build_segment_debug_contexts(
        segments,
        [tuple(item) for item in segment_debug.get("activeRanges", [])],
        all_onset_times,
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
    for segment in segments:
        start_time, end_time = segment
        duration = max(end_time - start_time, 0.08)
        sub_onsets = tuple(
            t for t in all_onset_times if start_time <= t <= end_time
        )
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
        candidates, candidate_debug, primary, _trace = segment_peaks(
            audio,
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
            confirmed_primary=segment.confirmed_primary,
            sub_onsets=sub_onsets,
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
                # Use raw segment duration (not the clamped pipeline `duration`)
                # so the guard fires for the actual short windows.
                from_short_segment_guard=(end_time - start_time) < SHORT_SEGMENT_SECONDARY_GUARD_DURATION,
                sub_onsets=sub_onsets,
            )
        )
        if debug and candidate_debug:
            segment_candidates_debug.append(candidate_debug)

    processed_events = suppress_low_confidence_dyad_transients(raw_events)
    processed_events = suppress_onset_decaying_carryover(processed_events)
    processed_events = suppress_resonant_carryover(processed_events, tuning)
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
    # #153 Phase A.2: rejoin short-segment-guarded primaries (e.g., E148 C6)
    # into adjacent chords when narrow FFT cross-validation confirms the
    # guarded note is independently present in the next event's attack
    # window.  Must run before merge_adjacent_events / merge_short_chord_clusters
    # because those passes only merge identical or compatible note sets.
    processed_events = merge_short_segment_guard_via_narrow_fft(
        processed_events, audio, sample_rate, tuning,
        noise_floor=noise_floor,
    )
    # #153 cosmetic extras follow-up: any short-segment-guarded singleton
    # that A.2 could not merge into a real chord is a spectral artefact
    # of the 6-16 ms FFT window, not a played note.  Drop it before the
    # downstream merge passes can promote it into a final event.
    processed_events = suppress_unmerged_guarded_singletons(processed_events)
    # #153 Phase A.3: rejoin gliss-split adjacent segments (e.g., E121
    # prefix splitting; E97 / E133 F5 trailing) by union with semitone
    # dedup.  Operates on non-guarded segments only.
    processed_events = merge_gliss_split_segments(
        processed_events, audio, sample_rate, tuning,
    )
    # #153 Phase A.4: recover a chord note rejected by the weak-secondary
    # gate when its narrow-FFT presence + a real attack rise within the
    # segment's sub-onsets jointly confirm it (e.g., E97 / E133 D5 masked
    # by prior D5 sustain).
    processed_events = recover_masked_reattack_via_narrow_fft(
        processed_events, audio, sample_rate, tuning,
        noise_floor=noise_floor,
    )
    # #154 Phase B lookback rescue: when the broadband onset detector
    # reports an onset that the segmenter did not materialize (the
    # onset sits in a gap between segments), narrow FFT at the
    # unconsumed onset time can reveal a chord note that attacks in
    # the gap and decays before the next segment starts.  E.g.,
    # 17-key BWV147 E97 G4 attacks at 167.98s but the next segment
    # starts at 168.152s; an unconsumed onset at 168.0827s sits in
    # the gap with G4 as the rank-1 narrow-FFT candidate.
    processed_events = recover_pre_segment_attack_via_narrow_fft(
        processed_events, audio, sample_rate, tuning, all_onset_times,
        noise_floor=noise_floor,
    )
    processed_events = collapse_late_descending_step_handoffs(processed_events)
    merged_events = merge_adjacent_events(processed_events)
    merged_events = collapse_late_descending_step_handoffs(merged_events)
    merged_events = merge_short_chord_clusters(merged_events)
    merged_events = merge_adjacent_events(merged_events)
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
            "noiseFloor": noise_floor.to_debug_dict(),
        }

    return TranscriptionResult(
        instrumentTuning=tuning,
        tempo=round(tempo, 2),
        events=events,
        notationViews=build_notation_views(events),
        warnings=warnings,
        debug=result_debug,
    )
