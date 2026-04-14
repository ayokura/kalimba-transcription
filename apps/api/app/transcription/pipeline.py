from __future__ import annotations

from typing import Any

from fastapi import HTTPException, UploadFile

from ..models import AlternateGrouping, CandidateSlot, InstrumentTuning, ScoreEvent, ScoreNote, TranscriptionResult
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
    recover_spread_chord_via_segment_start_probe,
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
    suppress_short_descending_return_singletons,
    suppress_short_residual_tails,
    suppress_subset_decay_events,
)
from .models import NoteCandidate, RawCandidateSlot, RawEvent
from .noise_floor import measure_noise_floor
from .notation import build_notation_views, format_doremi, format_number, quantize_beat
from .patterns import apply_repeated_pattern_passes
from .peaks import analyze_spectrum_at_onset, has_kalimba_sustain_profile, segment_peaks
from .per_note import rescue_gap_mute_dips
from .segments import (
    build_segment_debug_contexts,
    detect_segments,
    should_keep_low_register_sparse_gap_tail,
    simplify_sparse_gap_tail_high_octave_dyad,
)


def _event_sig(ev: RawEvent) -> tuple[float, float, tuple[str, ...]]:
    return (round(ev.start_time, 4), round(ev.end_time, 4),
            tuple(sorted(n.note_name for n in ev.notes)))


# #178 Phase 2: base confidence per drop reason for dropped-segment slots.
_DROP_REASON_BASE_CONFIDENCE: dict[str, float] = {
    # High confidence: mute-dip re-attack detected at sub-onset inside rejected segment.
    # This is a strong physical signal that a fresh attack occurred.
    "sub-onset-mute-dip-reattack": 0.80,
    # Medium-high confidence: onset detected (broadband + gap-validated) but no
    # segment constructed because RMS-based active range didn't cover it.
    # Physical attack signal exists; just the segmenter missed it.
    "orphan-onset-no-segment": 0.50,
    "residual-decay-no-reattack": 0.15,
    "low_register_sparse_gap_tail": 0.10,
    "primary-score-too-low": 0.05,
    "onset-gate-no-evidence": 0.05,
}

# Thresholds for promoting an orphan-onset candidate to a primary RawEvent
# instead of a candidate_slot.  Derived from cross-fixture orphan
# distribution: recording-edge transients and non-pluck noise typically
# show very high og but very low score (og >> 500, score < 10); genuine
# missed plucks show both high og and high score (Free Perf 10.94s D5:
# og=3268, score=237).  The sustain profile check is the final guard —
# non-pluck noise decays to noise floor within tens of ms even when it
# happens to land on a tuning frequency.
_ORPHAN_PROMOTE_MIN_OG = 500.0
_ORPHAN_PROMOTE_MIN_SCORE = 50.0


def _build_candidate_slot(
    start_time: float,
    end_time: float,
    primary: NoteCandidate,
    ranked_notes: list[NoteCandidate],
    drop_reason: str,
) -> RawCandidateSlot:
    """Build a RawCandidateSlot from a dropped segment's primary and top-ranked notes."""
    base_conf = _DROP_REASON_BASE_CONFIDENCE.get(drop_reason, 0.05)
    # Boost confidence if onset evidence is present on the primary.
    og = primary.onset_gain
    if og is not None and og >= 10.0:
        base_conf = min(0.35, base_conf * 2.0)
    # Top 3 alternative candidates (excluding duplicates of primary).
    seen = {primary.note_name}
    alts: list[NoteCandidate] = []
    for cand in ranked_notes:
        if cand.note_name in seen:
            continue
        seen.add(cand.note_name)
        alts.append(cand)
        if len(alts) >= 3:
            break
    return RawCandidateSlot(
        start_time=start_time,
        end_time=end_time,
        primary_note=primary,
        candidates=alts,
        drop_reason=drop_reason,
        confidence=round(base_conf, 3),
    )


def _trace_post_processing_step(
    name: str,
    before: list[RawEvent],
    after: list[RawEvent],
    trace: list[dict[str, Any]],
) -> None:
    """Record removed/added events when a post-processing step changes the list."""
    before_sigs = [_event_sig(e) for e in before]
    after_sigs = [_event_sig(e) for e in after]
    if before_sigs == after_sigs:
        return
    after_set = set(after_sigs)
    before_set = set(before_sigs)
    removed = [{"startTime": s[0], "endTime": s[1], "notes": list(s[2])}
               for s in before_sigs if s not in after_set]
    added = [{"startTime": s[0], "endTime": s[1], "notes": list(s[2])}
             for s in after_sigs if s not in before_set]
    trace.append({"pass": name, "removed": removed, "added": added})


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
    dropped_slots: list[RawCandidateSlot] = []  # #178 Phase 2: preserved dropped segments
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
        candidates, candidate_debug, primary, _trace, _soft_alts, _dropped_primary, _dropped_ranked, _dropped_reason, _dropped_rescues = segment_peaks(
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
            segment_sources=segment.sources,
        )
        if not candidates or primary is None:
            if debug and candidate_debug:
                segment_candidates_debug.append(candidate_debug)
            # #178 Phase 2: preserve dropped segment as candidate slot
            if _dropped_primary is not None:
                dropped_slots.append(_build_candidate_slot(
                    start_time=start_time,
                    end_time=end_time,
                    primary=_dropped_primary,
                    ranked_notes=_dropped_ranked,
                    drop_reason=_dropped_reason,
                ))
                # #178 Phase 2: sub-onset rescues — mute-dip re-attack found
                # within a rejected segment.  The mute-dip envelope (ringing
                # followed by energy drop and fresh spike) is a strong
                # physical signal of a genuine re-attack on the same pitch,
                # so we promote to a primary-bearing RawEvent rather than a
                # candidate_slot.  Secondary-only; no harmonic/chord rerun
                # is attempted at the rescue time.
                for rescue_time in _dropped_rescues:
                    rescue_end = min(rescue_time + 0.3, end_time)
                    rescue_duration = rescue_end - rescue_time
                    raw_events.append(
                        RawEvent(
                            start_time=rescue_time,
                            end_time=rescue_end,
                            notes=[_dropped_primary],
                            is_gliss_like=rescue_duration < 0.18,
                            primary_note_name=_dropped_primary.note_name,
                            primary_score=0.0,
                            from_short_segment_guard=rescue_duration < SHORT_SEGMENT_SECONDARY_GUARD_DURATION,
                            sub_onsets=[],
                            alternate_groupings=[],
                        )
                    )
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
                # #178 Phase 2: preserve as candidate slot.
                # candidates is non-empty here; use primary's note as primary_note.
                primary_cand = next(
                    (c for c in candidates if c.note_name == primary.candidate.note_name),
                    candidates[0],
                )
                dropped_slots.append(_build_candidate_slot(
                    start_time=start_time,
                    end_time=end_time,
                    primary=primary_cand,
                    ranked_notes=[c for c in candidates if c.note_name != primary_cand.note_name],
                    drop_reason="low_register_sparse_gap_tail",
                ))
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
                alternate_groupings=list(_soft_alts),
            )
        )
        if debug and candidate_debug:
            segment_candidates_debug.append(candidate_debug)

    post_processing_trace: list[dict[str, Any]] = []
    _t = post_processing_trace  # short alias for trace list

    def _pp(name: str, fn, *args, **kwargs) -> list[RawEvent]:
        """Run a post-processing step and trace changes."""
        nonlocal processed_events
        before = processed_events
        result = fn(before, *args, **kwargs)
        _trace_post_processing_step(name, before, result, _t)
        processed_events = result
        return result

    # #178 Phase 2.5: orphan onset recovery — gap-validated onsets that fall
    # outside all segments (active ranges didn't cover them). Recovered as
    # medium-confidence candidate slots via lightweight FFT analysis.
    gap_validated_onsets = segment_debug.get("gapValidatedOnsetTimes")
    if gap_validated_onsets:
        segment_ranges = [(s.start_time, s.end_time) for s in segments]
        # Stricter "near-segment-boundary" threshold — 100ms catches pre-onset
        # artifacts that create near-duplicate slots next to real events.
        _ORPHAN_BOUNDARY_SKIP_SEC = 0.10
        for onset_time in gap_validated_onsets:
            onset_time = float(onset_time)
            # Skip if within any existing segment
            if any(s <= onset_time <= e for s, e in segment_ranges):
                continue
            # Skip if too close to segment boundary (pre-onset / post-onset artifact)
            if any(abs(onset_time - s) < _ORPHAN_BOUNDARY_SKIP_SEC or abs(onset_time - e) < _ORPHAN_BOUNDARY_SKIP_SEC
                   for s, e in segment_ranges):
                continue
            orphan_candidates = analyze_spectrum_at_onset(
                audio, sample_rate, onset_time, tuning,
            )
            if not orphan_candidates:
                continue
            # Require meaningful attack evidence: the top candidate's onset_gain
            # should be clearly above the noise floor. Spurious trailing-silence
            # detections and pure noise have og < ~10.
            top = orphan_candidates[0]
            top_og = top.onset_gain or 0
            if top_og < 10.0:
                continue
            # Sustain profile filter: drop orphans without kalimba-like
            # ringing entirely (no slot, no alt).  Recording-edge clicks
            # and non-pluck noise can hit a pitch bin on onset but decay
            # to noise floor in ~40 ms; a genuine pluck sustains for
            # hundreds of ms.  See Free Perf 10.94s D5 (sustain/og ~2%)
            # vs triple-glissando 0.03s C5 (sustain/og ~0.2%).
            if not has_kalimba_sustain_profile(audio, sample_rate, onset_time, top.frequency):
                continue
            # Promote to RawEvent when onset + score are both strong;
            # otherwise surface as a candidate_slot for UI review.
            # Thresholds calibrated against cross-fixture orphan
            # distribution (see #178 Phase 2.5 follow-up investigation).
            top_score = top.score or 0
            promote = (
                top_og >= _ORPHAN_PROMOTE_MIN_OG
                and top_score >= _ORPHAN_PROMOTE_MIN_SCORE
            )
            if promote:
                rescue_end = onset_time + 0.2
                rescue_duration = rescue_end - onset_time
                raw_events.append(
                    RawEvent(
                        start_time=onset_time,
                        end_time=rescue_end,
                        notes=[top],
                        is_gliss_like=rescue_duration < 0.18,
                        primary_note_name=top.note_name,
                        primary_score=top_score,
                        from_short_segment_guard=rescue_duration < SHORT_SEGMENT_SECONDARY_GUARD_DURATION,
                        sub_onsets=[],
                        alternate_groupings=[],
                    )
                )
            else:
                dropped_slots.append(_build_candidate_slot(
                    start_time=onset_time,
                    end_time=onset_time + 0.2,
                    primary=top,
                    ranked_notes=orphan_candidates[1:],
                    drop_reason="orphan-onset-no-segment",
                ))

    processed_events = raw_events
    _pp("suppress_low_confidence_dyad_transients", suppress_low_confidence_dyad_transients)
    _pp("suppress_onset_decaying_carryover", suppress_onset_decaying_carryover)
    _pp("collapse_same_start_primary_singletons", collapse_same_start_primary_singletons)
    _pp("simplify_short_secondary_bleed", simplify_short_secondary_bleed)
    _pp("suppress_post_tail_gap_bridge_dyads", suppress_post_tail_gap_bridge_dyads)
    _pp("suppress_leading_descending_overlap", suppress_leading_descending_overlap, tuning)
    _pp("simplify_descending_adjacent_dyad_residue", simplify_descending_adjacent_dyad_residue)
    _pp("collapse_high_register_adjacent_bridge_dyads", collapse_high_register_adjacent_bridge_dyads, tuning)
    _pp("suppress_descending_upper_singleton_spikes", suppress_descending_upper_singleton_spikes)
    _pp("suppress_short_descending_return_singletons", suppress_short_descending_return_singletons, tuning)
    _pp("suppress_descending_upper_return_overlap", suppress_descending_upper_return_overlap)
    _pp("merge_short_gliss_clusters", merge_short_gliss_clusters)
    _pp("simplify_short_gliss_prefix_to_contiguous_singleton", simplify_short_gliss_prefix_to_contiguous_singleton)
    _pp("merge_four_note_gliss_clusters", merge_four_note_gliss_clusters)
    _pp("suppress_leading_gliss_subset_transients", suppress_leading_gliss_subset_transients)
    _pp("suppress_leading_gliss_neighbor_noise", suppress_leading_gliss_neighbor_noise)
    _pp("suppress_leading_single_transient", suppress_leading_single_transient)
    _pp("suppress_subset_decay_events", suppress_subset_decay_events)
    _pp("split_ambiguous_upper_octave_pairs", split_ambiguous_upper_octave_pairs)
    _pp("suppress_bridging_octave_pairs", suppress_bridging_octave_pairs)
    _pp("suppress_short_residual_tails", suppress_short_residual_tails)
    # #153 Phase A.2: rejoin short-segment-guarded primaries (e.g., E148 C6)
    # into adjacent chords when narrow FFT cross-validation confirms the
    # guarded note is independently present in the next event's attack
    # window.  Must run before merge_adjacent_events / merge_short_chord_clusters
    # because those passes only merge identical or compatible note sets.
    _pp("merge_short_segment_guard_via_narrow_fft",
        merge_short_segment_guard_via_narrow_fft,
        audio, sample_rate, tuning, noise_floor=noise_floor)
    # #153 cosmetic extras follow-up: any short-segment-guarded singleton
    # that A.2 could not merge into a real chord is a spectral artefact
    # of the 6-16 ms FFT window, not a played note.  Drop it before the
    # downstream merge passes can promote it into a final event.
    _pp("suppress_unmerged_guarded_singletons", suppress_unmerged_guarded_singletons)
    # #153 Phase A.3: rejoin gliss-split adjacent segments (e.g., E121
    # prefix splitting; E97 / E133 F5 trailing) by union with semitone
    # dedup.  Operates on non-guarded segments only.
    _pp("merge_gliss_split_segments",
        merge_gliss_split_segments, audio, sample_rate, tuning)
    # #153 Phase A.4: recover a chord note rejected by the weak-secondary
    # gate when its narrow-FFT presence + a real attack rise within the
    # segment's sub-onsets jointly confirm it (e.g., E97 / E133 D5 masked
    # by prior D5 sustain).
    _pp("recover_masked_reattack_via_narrow_fft",
        recover_masked_reattack_via_narrow_fft,
        audio, sample_rate, tuning, noise_floor=noise_floor)
    # #154 Phase B lookback rescue: when the broadband onset detector
    # reports an onset that the segmenter did not materialize (the
    # onset sits in a gap between segments), narrow FFT at the
    # unconsumed onset time can reveal a chord note that attacks in
    # the gap and decays before the next segment starts.
    _pp("recover_pre_segment_attack_via_narrow_fft",
        recover_pre_segment_attack_via_narrow_fft,
        audio, sample_rate, tuning, all_onset_times, noise_floor=noise_floor)
    # #167 Phase C spread-chord rescue: recover notes that attacked
    # between the broadband onset and the segment start (rolled /
    # arpeggiated chords).  Unlike Phase B, these notes are RISING
    # into the segment, so the probe point is segment start, not
    # onset time.  Run twice to recover up to two notes per event
    # (e.g. G-low E136: B3 then G3).
    _pp("recover_spread_chord_via_segment_start_probe",
        recover_spread_chord_via_segment_start_probe,
        audio, sample_rate, tuning, all_onset_times, noise_floor=noise_floor)
    _pp("recover_spread_chord_via_segment_start_probe_2",
        recover_spread_chord_via_segment_start_probe,
        audio, sample_rate, tuning, all_onset_times, noise_floor=noise_floor)
    _pp("collapse_late_descending_step_handoffs", collapse_late_descending_step_handoffs)

    def _mp(name: str, fn, *args, **kwargs) -> list[RawEvent]:
        """Run a merge-phase post-processing step and trace changes."""
        nonlocal merged_events
        before = merged_events
        result = fn(before, *args, **kwargs)
        _trace_post_processing_step(name, before, result, _t)
        merged_events = result
        return result

    merged_events = merge_adjacent_events(processed_events)
    _trace_post_processing_step("merge_adjacent_events", processed_events, merged_events, _t)
    _mp("collapse_late_descending_step_handoffs_2", collapse_late_descending_step_handoffs)
    _mp("merge_short_chord_clusters", merge_short_chord_clusters)
    _mp("merge_adjacent_events_2", merge_adjacent_events)
    _mp("collapse_ascending_restart_lower_residue_singletons",
        collapse_ascending_restart_lower_residue_singletons, tuning)
    _mp("merge_adjacent_events_3", merge_adjacent_events)

    merged_events, repeated_pattern_pass_trace = apply_repeated_pattern_passes(
        merged_events,
        disabled_passes=disabled_repeated_pattern_passes,
        debug=debug,
    )
    _mp("split_adjacent_step_dyads_in_ascending_runs",
        split_adjacent_step_dyads_in_ascending_runs, tuning)
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
        alt_groupings = None
        if event.alternate_groupings:
            alt_groupings = []
            for alt in event.alternate_groupings:
                if alt.split_into is not None:
                    # Split mode (B2 gap ambiguity): merged event records the
                    # original separate note groups as an alternative.
                    split_groups = [
                        [
                            ScoreNote(
                                key=c.key, pitchClass=c.pitch_class, octave=c.octave,
                                labelDoReMi=format_doremi(c), labelNumber=format_number(c),
                                frequency=round(c.frequency, 3),
                            )
                            for c in group
                        ]
                        for group in alt.split_into
                    ]
                    alt_groupings.append(
                        AlternateGrouping(
                            splitInto=split_groups,
                            reason=alt.reason,
                            confidence=alt.confidence,
                        )
                    )
                elif alt.alternate_note is not None:
                    # AlternateNote mode (C soft candidate): a secondary candidate
                    # that was conservatively rejected but could be valid.
                    c = alt.alternate_note
                    alt_groupings.append(
                        AlternateGrouping(
                            alternateNote=ScoreNote(
                                key=c.key, pitchClass=c.pitch_class, octave=c.octave,
                                labelDoReMi=format_doremi(c), labelNumber=format_number(c),
                                frequency=round(c.frequency, 3),
                            ),
                            reason=alt.reason,
                            confidence=alt.confidence,
                        )
                    )
                else:
                    # Combine mode (B1 dissonance guard): separate event records
                    # the hypothetical merged result as an alternative.
                    partner_idx = alt.combine_with_index or 0
                    partner_id = f"evt-{partner_idx + 1}" if partner_idx < len(merged_events) else "evt-?"
                    for mi, me in enumerate(merged_events):
                        if mi != index - 1 and me.start_time >= event.start_time - 0.01 and alt.combined_notes and any(
                            n.note_name in {c.note_name for c in alt.combined_notes}
                            for n in me.notes
                        ):
                            partner_id = f"evt-{mi + 1}"
                            break
                    alt_groupings.append(
                        AlternateGrouping(
                            combinesWith=[partner_id],
                            combinedNotes=[
                                ScoreNote(
                                    key=c.key, pitchClass=c.pitch_class, octave=c.octave,
                                    labelDoReMi=format_doremi(c), labelNumber=format_number(c),
                                    frequency=round(c.frequency, 3),
                                )
                                for c in (alt.combined_notes or [])
                            ],
                            reason=alt.reason,
                            confidence=alt.confidence,
                        )
                    )

        events.append(
            ScoreEvent(
                id=f"evt-{index}",
                startBeat=start_beat,
                durationBeat=duration_beat,
                notes=notes,
                isGlissLike=event.is_gliss_like,
                gesture=classify_event_gesture(event, index - 1, raw_events, merged_events),
                alternateGroupings=alt_groupings,
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
            "postProcessingTrace": post_processing_trace,
            "disabledRepeatedPatternPasses": sorted(disabled_repeated_pattern_passes or ()),
            "repeatedPatternPassTrace": repeated_pattern_pass_trace,
            "noiseFloor": noise_floor.to_debug_dict(),
        }

    # #178 Phase 2: convert dropped slots to API model
    candidate_slots_api: list[CandidateSlot] = []
    for slot in sorted(dropped_slots, key=lambda s: s.start_time):
        def _to_score_note(c: NoteCandidate) -> ScoreNote:
            return ScoreNote(
                key=c.key, pitchClass=c.pitch_class, octave=c.octave,
                labelDoReMi=format_doremi(c), labelNumber=format_number(c),
                frequency=round(c.frequency, 3),
            )
        candidate_slots_api.append(CandidateSlot(
            startTime=round(slot.start_time, 4),
            endTime=round(slot.end_time, 4),
            primaryNote=_to_score_note(slot.primary_note),
            candidates=[_to_score_note(c) for c in slot.candidates],
            dropReason=slot.drop_reason,
            confidence=slot.confidence,
        ))

    return TranscriptionResult(
        instrumentTuning=tuning,
        tempo=round(tempo, 2),
        events=events,
        candidateSlots=candidate_slots_api,
        notationViews=build_notation_views(events),
        warnings=warnings,
        debug=result_debug,
    )
