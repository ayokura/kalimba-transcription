from __future__ import annotations

# Audio framing and collector geometry.
FRAME_LENGTH = 2048
HOP_LENGTH = 256
TEMPO_ESTIMATION_HOP_LENGTH = 1024
RMS_MEDIAN_THRESHOLD_MAX_PEAK_RATIO = 0.45
NESTED_SEGMENT_DEDUP_MAX_START_DELTA = 0.02
CROSS_COLLECTOR_DEDUP_MIN_OVERLAP_RATIO = 0.5
SEGMENT_OVERLAP_TRIM_MAX_OVERLAP = 0.18
SEGMENT_OVERLAP_TRIM_MIN_DURATION = 0.18
MULTI_ONSET_GAP_MIN_DURATION = 1.0
MULTI_ONSET_GAP_MIN_EDGE_SPACING = 0.18
MULTI_ONSET_GAP_MIN_INTERVAL = 0.18
MULTI_ONSET_GAP_MAX_INTERVAL = 0.42
MULTI_ONSET_GAP_MIN_SHORT_INTERVALS = 2
GAP_RUN_LEAD_IN_MIN_FOLLOWUP_GAP = 0.9
SPARSE_GAP_TAIL_MIN_DURATION = 1.0
SPARSE_GAP_TAIL_MIN_PREVIOUS_EDGE = 0.04
SPARSE_GAP_TAIL_MAX_ONSET_OFFSET = 0.45
SPARSE_GAP_TAIL_MIN_TRAILING_EDGE = 0.6
SPARSE_GAP_TAIL_MIN_INTERVAL = 0.08
SPARSE_GAP_TAIL_MAX_INTERVAL = 0.45
SPARSE_GAP_TAIL_SEGMENT_DURATION = 0.24
SHORT_BRIDGE_ACTIVE_RANGE_MAX_DURATION = 0.16
SHORT_BRIDGE_ACTIVE_RANGE_MAX_ONSET_OFFSET = 0.03
SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_ONSET_GAP = 0.1
SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_ONSET_GAP = 0.3
SHORT_BRIDGE_ACTIVE_RANGE_MIN_NEXT_EDGE_GAP = 0.08
SHORT_BRIDGE_ACTIVE_RANGE_MAX_NEXT_EDGE_GAP = 0.2

# Gap-head collectors.
ACTIVE_RANGE_START_CLUSTER_MIN_GAP = 0.45
ACTIVE_RANGE_START_CLUSTER_MAX_SPAN = 0.09
ACTIVE_RANGE_START_CLUSTER_MAX_DURATION = 0.35
LONG_RANGE_BACKTRACK_MIN_DURATION = 0.35
CLUSTERED_RANGE_HEAD_MIN_DURATION = 0.05
ACTIVE_RANGE_HEAD_CLUSTER_MAX_OFFSET = 0.3
ACTIVE_RANGE_HEAD_CLUSTER_MAX_INTERVAL = 0.14
ACTIVE_RANGE_HEAD_CLUSTER_MIN_DURATION = 0.35

# Onset attack analysis and gate thresholds.
ATTACK_ANALYSIS_SECONDS = 0.16
ATTACK_ANALYSIS_RATIO = 0.35
ONSET_ENERGY_WINDOW_SECONDS = 0.08
SPECTRAL_FLUX_HIGH_BAND_MIN_FREQUENCY = 2000.0

# Sub-onset based per-note attack window anchoring (#152).
# When a segment contains multiple sub-onsets, per-note attack measurement
# (onset_energy_gain) anchors its window at the sub-onset showing the
# strongest energy increase for the target note, instead of always using
# segment start.  This handles slide chords where notes attack staggered
# within a single segment.
SUB_ONSET_ANCHOR_ENABLED = True
SUB_ONSET_ANCHOR_MEASUREMENT_WINDOW = 0.04
SUB_ONSET_ANCHOR_MIN_RATIO = 2.0
# Minimum post-onset peak energy required to consider a sub-onset a real
# attack for the target note.  Filters out FFT spectral leakage from
# attacks of nearby (different) notes, which produce huge post/pre ratios
# from near-zero values.  Calibrated against fixture per-note energies.
SUB_ONSET_ANCHOR_MIN_POST_ENERGY = 1.0
ONSET_ATTACK_MIN_BROADBAND_GAIN = 10.0
ONSET_ATTACK_MIN_HIGH_BAND_FLUX = 1.5
ONSET_ATTACK_GAIN_REQUIRES_MIN_FLUX = 0.5
ONSET_ATTACK_MODERATE_GAIN = 3.0
ONSET_ATTACK_MODERATE_GAIN_MIN_FLUX = 0.8
ATTACK_VALIDATED_GAP_SEGMENT_DURATION = 0.24
ATTACK_REFINED_ONSET_MAX_INTERVAL = 0.15

# Short-segment secondary guard.
# When segment duration is below this threshold, FFT analysis cannot reliably
# distinguish secondary peaks from spectral leakage / pre-attack noise.
# Restrict candidate selection to primary only and mark skipped notes in the
# trail so downstream logic (e.g., per-sub-onset narrow FFT, future #141 work)
# can recover them via a different analysis path.
# Threshold rationale: kalimba C4 (~261 Hz) needs ~30 ms (8 fundamental periods)
# for minimal FFT resolution; below this, secondary detections are unreliable.
SHORT_SEGMENT_SECONDARY_GUARD_DURATION = 0.030

# Sub-onset narrow FFT (#153 Phase A).
# Used to detect notes spectrally hidden in the segment-wide FFT but visible
# in a narrow attack window centered on a sub-onset.  Motivating cases:
#   - octave-coincident chord (e.g., C6 fundamental at 1046 Hz collides with
#     C5 2nd harmonic; only separable in early ~30 ms before C5 sustain
#     dominates the bin)
#   - gliss prefix splitting where a gliss head/tail is detected as an
#     independent short segment that should merge with the main chord
#   - early-window primary masking (e.g., C4 attack at 56.10s clearly dominant
#     for ~80 ms before E5 takes over; segment-wide FFT shows E5 only)
#
# Phase A uses fixed absolute energy floors below.  Phase B (#154) replaces
# these with per-recording per-band noise floor calibration; the thresholds
# are accessed via function arguments so the Phase B swap is straightforward.
NARROW_FFT_WINDOW_SECONDS = 0.030
NARROW_FFT_PRIMARY_THRESHOLD = 50.0
# Used by merge_short_segment_guard_via_narrow_fft (Phase A.2).  Calibrated
# from 17-key BWV147 E148 (C6 fundamental_energy ≈ 26 in narrow FFT).
NARROW_FFT_GUARD_MERGE_MIN_ENERGY = 20.0
# Disambiguator A: the guarded note's narrow-FFT score must be at least this
# fraction of the next event's primary score in the same window.  E148 C6
# (score 36) vs the next event's primary C5 (score < 5 because C5 has not
# yet started ringing) → ratio is high, merge fires.  R6 E161 cosmetic A4
# (score 27) vs C5 (score 98 because C5 is already dominant) → ratio 0.27,
# merge is suppressed (carryover, not fresh attack).
NARROW_FFT_GUARD_MERGE_DOMINANCE_RATIO = 0.5
# Disambiguator B: the guarded note's fundamental_ratio in the narrow FFT
# must be at least this value.  A real fresh attack has the fundamental
# clearly dominant over its harmonics; a spectral leakage artifact from a
# semitone-adjacent note has a depressed fundamental_ratio (e.g., 34-key
# L1 6.298s B4 fr=0.674 leaking from A4 vs E148 C6 fr=0.903 real attack).
NARROW_FFT_GUARD_MERGE_MIN_FR = 0.85
# Disambiguator C: onset_backward_attack_gain (early-window vs 200ms-prior
# reference) must exceed this threshold.  This separates a fresh attack
# from a residual sustain — a same-name note from ~1-2s prior can still
# show high fundamental_energy and high fundamental_ratio if its sustain
# has not fully decayed (e.g., 34-key R3 64.418s B4 sustain from E113 at
# ~63s shows backward gain ≈ 6.0; 34-key L1 6.298s B4 leakage shows ≈ 0.9;
# 17-key R6 275.520s cosmetic A4 shows ≈ 6.8 — all rejected at 10).  A
# genuine new attack like E148 256.304s C6 shows backward gain ≈ 139.
# Threshold matches RESCUE_MIN_BACKWARD_GAIN.
NARROW_FFT_GUARD_MERGE_MIN_BACKWARD_GAIN = 10.0

# Phase A.3: gliss-split segment merge (#153).
# Two non-guarded adjacent segments belonging to the same gliss/chord are
# unified by a union-with-semitone-dedup rule.  Patterns covered:
#   - Gliss prefix splitting (E121, E127): a short prefix segment with
#     spectral-leakage semitone artifacts gets absorbed into the longer
#     main segment; the leakage notes are dropped because they sit within
#     a semitone of a main-segment note.
#   - Gliss late-note splitting (E97, E133): the F5 attack of a 4-note
#     gliss lands as a separate adjacent segment and gets merged into the
#     gliss head as a new chord note.
# A dissonance guard rejects merges whose result contains any whole-step
# (≤200 cents) pair, blocking false merges between unrelated short events.
GLISS_SPLIT_MERGE_MAX_GAP = 0.05
GLISS_SPLIT_MERGE_MAX_FIRST_DURATION = 0.20
GLISS_SPLIT_MERGE_SEMITONE_CENTS = 100.0
# Contiguous-key subset extraction: when a naive gliss-split merge exceeds
# MAX_POLYPHONY, attempt to find a contiguous-key subset that strips noise
# notes (broadband leak / sympathetic) while preserving the real chord.
# Minimum run length of 3 avoids merging into trivial dyads.
# Calibrated on kalimba-17-g-low BWV147:
#   E133: union {C4(7),D4(11),F#4(12),A4(13),C5(14)} → best subset {11-14} (4 notes)
#   E82:  union {A3(8),G3(9),D4(11),F#4(12),A4(13)} → best subset {11-13} (3 notes)
GLISS_SPLIT_CONTIGUOUS_SUBSET_MIN_RUN = 3

# Phase A.4: masked re-attack recovery via narrow FFT (#153).
# When the segment-wide FFT rejects a chord note as weak-secondary-onset
# because the note's prior sustain masks its re-attack signature (e.g.,
# 34-key R1 E97 D5 with previous D5 sustain in the same R1 line), narrow
# FFT at the segment start can still surface it as the rank-1 candidate
# with a high fundamental_ratio.  pick_matching_sub_onset then confirms
# the same note has a real attack rise within the segment's sub-onsets.
# Both signals are required: spectral dominance alone could be sustained
# residue, and onset rise alone could be a near-frequency neighbour.
NARROW_FFT_REATTACK_MIN_ENERGY = 30.0
NARROW_FFT_REATTACK_MIN_FR = 0.95
# Recovery is gated on the candidate clearly dominating the event's own
# selected notes in the early-window narrow FFT.  Without this gate the
# pass adds a "best of the rest" note to almost every multi-sub-onset
# event and produces tens of cosmetic regressions across both fixtures.
NARROW_FFT_REATTACK_DOMINANCE_RATIO = 1.5
# Reject candidates that sit within this many cents of any existing event
# note (whole step = 200 cents).  Without the whole-step guard the pass
# wrongly adds E6 to D6 events in the d6-to-e6 alternating sequence.
NARROW_FFT_REATTACK_DISSONANCE_CENTS = 200.0

# Phase B: per-recording per-band noise floor calibration (#154).
# Silent regions in the recording are sampled with the same narrow-FFT
# parameters used by the Phase A merge passes; the median fundamental
# energy at each tuning note becomes a per-note noise floor.  Phase A
# absolute thresholds are then expressed as ``noise_floor[note] * factor``
# so the same factor adapts to mic gain, room noise, and per-band
# differences across recordings.
#
# Silent region selection — gaps between segments must be at least
# this long, and a small padding is trimmed from each edge so the
# silent slice does not include the segment's attack/release tail.
NOISE_FLOOR_MIN_SILENT_GAP_SECONDS = 0.10
NOISE_FLOOR_EDGE_PADDING_SECONDS = 0.02
# Cap on the number of silent slices; the longest regions are
# preferred so the median is computed over real silence rather than
# tight inter-attack pauses that may still contain decay tails.
NOISE_FLOOR_MAX_SAMPLES = 12
# Multipliers applied to the per-note noise floor when computing the
# Phase B replacement thresholds.  Calibrated so:
#   - 17-key BWV147 E148 C6 (Phase A.2 motivating case, narrow-FFT
#     fund_e ≈ 26) still passes the guard merge with comfortable margin.
#   - 17-key BWV147 E97 G4 (Phase B target case, very weak attack with
#     narrow-FFT fund_e on the order of single digits) clears the lower
#     bound when the noise floor measurement is in the expected range.
NARROW_FFT_GUARD_MERGE_NOISE_FACTOR = 10.0
NARROW_FFT_REATTACK_NOISE_FACTOR = 12.0
# Hard floor under the noise-floor-derived thresholds.  Defends against
# pathologically low noise floor measurements (e.g., heavily compressed
# recordings or numerical underflow on a near-silent fixture).  Set
# below the Phase A absolute thresholds (20 / 30) so the Phase B swap
# can lower the bar for genuinely weak attacks while still rejecting
# trivial noise picks.
NARROW_FFT_NOISE_THRESHOLD_HARD_FLOOR = 5.0

# Phase B pre-segment lookback rescue (#154 / #153 Phase B).
# When the broadband onset detector reports an onset that the
# segmenter did not materialize (the onset sits in a gap between
# segments), narrow FFT at the unconsumed onset time can reveal a
# chord note that attacks in the gap and decays before the next
# segment starts.  The pass walks unconsumed onsets and adds the
# rank-1 candidate to the immediately following event when a stack
# of physical safeguards passes.
#
# Motivating case: 17-key BWV147 E97 <F5,D5,B4,G4>.  G4 attacks at
# ~167.98s (narrow FFT fund_e peaks ~1.0).  The next segment starts
# at 168.152s — well after G4 decayed — so segment_peaks rejects G4
# as score-below-threshold,recent-carryover-candidate.  An unconsumed
# broadband onset at 168.0827s sits in the gap; narrow FFT there
# shows G4 as the rank-1 candidate with backward_attack_gain ~119
# (a clean fresh attack, not sustain residue).
NARROW_FFT_PRE_SEGMENT_LOOKBACK_SECONDS = 0.20
# Noise floor multiplier kept low because (a) the lookback narrow FFT
# runs in a clean gap with no competing sustain so the floor is mostly
# a sanity check, not the primary discriminator, and (b) noise floor
# itself varies meaningfully between full-audio and evaluation-scope
# transcription passes (eval-scope trims silence and produces a higher
# floor estimate), so a tight factor would gate the same fund_e in
# only one mode.  The real discriminator is ``backward_attack_gain``
# below — noise floor only protects against pathological undershoot.
#
# Calibration: 17-key BWV147 E97 G4 narrow-FFT fund_e at the unconsumed
# onset 168.0827s is ~0.637.  noise_floor[G4] is ~0.116 in full-audio
# and ~0.181 in evaluation-scope passes; factor=3 yields thresholds
# 0.348 / 0.543 respectively, both below 0.637 so the rescue fires in
# either mode.
NARROW_FFT_PRE_SEGMENT_NOISE_FACTOR = 3.0
NARROW_FFT_PRE_SEGMENT_HARD_FLOOR = 0.3
# Phase A absolute fallback used when noise floor calibration is
# unavailable (e.g., synthetic fixture with no silent gap).  Set well
# below the merge-pass fallbacks because pre-segment rescue runs in
# clean gaps where weak attacks are physically meaningful.
NARROW_FFT_PRE_SEGMENT_MIN_ENERGY = 5.0
NARROW_FFT_PRE_SEGMENT_MIN_FR = 0.85
# Backward attack gain is the primary fresh-attack-vs-sustain
# discriminator and the gate that catches false positives where the
# rank-1 narrow-FFT note at the unconsumed onset is actually decay
# from a recently played note.  Calibration data from 17-key BWV147:
#
#   * 168.0827s G4 (true positive, E97 rescue target) → 119.0
#   * 116.176s  A4 (false positive, L5 E64) → 12.38
#   * 264.2613s top-6 candidates (false positive, R5 E154) → 0.5–0.9
#
# Threshold 50 cleanly separates the true rescue from the false
# positives while leaving E97 G4 with comfortable margin.
NARROW_FFT_PRE_SEGMENT_MIN_BACKWARD_GAIN = 50.0
# Reject candidates within this many cents of any existing event
# note.  Slightly larger than the merge-pass 200-cent (whole step)
# threshold for two reasons: (a) cents_distance returns 200.000…06
# for an exact whole step under standard tuning frequencies, so the
# strict 200 boundary suffers from a float-precision miss, and (b)
# pre-segment rescue is more permissive on weak attacks where
# spectral leakage from a sympathetic-resonance neighbour is more
# likely to look like a fresh note.  250 still allows minor thirds
# (300 cents) and any wider chord interval through.
NARROW_FFT_PRE_SEGMENT_DISSONANCE_CENTS = 250.0
# Decay-pattern discriminator: the candidate's fund_e at the
# unconsumed onset must be at least this fraction of its fund_e at
# the segment start.  Rejects notes whose energy is RISING into the
# segment (= part of the upcoming chord and already evaluated by
# segment_peaks).  Example: 34-key R5 E154 D4 with fund_e 1.509 at
# onset / 8.287 at segment start = 0.18.
NARROW_FFT_PRE_SEGMENT_DECAY_MIN_RATIO = 0.8
# Backward-gain dominance discriminator: the candidate's bg must be
# at least this fraction of the maximum bg among the EVENT'S OWN
# notes at the unconsumed onset.  Physical meaning: if the in-event
# notes are themselves mid-attack at the unconsumed onset (high bg),
# then the unconsumed onset is just an early detection of the same
# chord attack rather than a separate pre-segment event, so a
# rescue would be promoting a sympathetic-resonance neighbour
# instead of recovering a missed note.  When the in-event notes
# have low bg (= they have not started ringing yet), any candidate
# with a meaningful fresh-attack signature is a plausible
# pre-segment recovery.
#
# Calibration:
#   * E97 G4: rescue_bg 119 / max_event_bg 63 (B4) = 1.89  → keep
#   * E100 C4: rescue_bg 267 / max_event_bg 260 (E4) = 1.03 → keep
#   * d4-d5 18.1173 G5: rescue_bg 465 / max_event_bg 17037 (D5) =
#     0.027 → drop  (D5 is itself attacking; the unconsumed onset
#     is the leading edge of the same D4+D5 attack the segmenter
#     captures 35 ms later, not a separate pre-segment event)
#   * R5 E154 D4: rescue_bg 57 / max_event_bg 1051 (A4) = 0.054 → drop
#
# Threshold 0.5 cleanly separates true rescues (ratio ≥ 1.0) from
# the sympathetic-resonance false positives (ratio < 0.1).
NARROW_FFT_PRE_SEGMENT_BG_DOMINANCE_RATIO = 0.5
# Onsets that fall inside an event's [start_time, end_time] are
# already represented in segment processing and are skipped by the
# rescue.  A small slack on the end accommodates floating-point
# rounding when comparing onset times to segment boundaries.
NARROW_FFT_PRE_SEGMENT_ONSET_CONSUMED_TOLERANCE = 0.005

# Phase C: spread-chord segment-start rescue (#167).
# Recovers notes that attack between the broadband onset and the segment
# start (rolled / arpeggiated chords).  Unlike Phase B, these notes are
# *rising* into the segment, so probing at onset_time shows nothing; the
# correct probe point is event.start_time with a lookback long enough to
# reach before the earliest individual attack.
#
# Motivating case: G-low BWV147 E136 <B4,D4,B3,G3>.  B3/G3 attack at
# ~81.15-81.20s, segment starts at 81.276s; backward_attack_gain at
# segment start with 0.20s lookback (reference: 81.076s, before all
# attacks) is 83 / 48 — strong fresh-attack signal.
# Calibrated on G-low E136: G3 bg=48.2, B3 bg=83.1.  45.0 admits G3
# with 7% margin while Phase B's 50.0 (calibrated for onset-time probe)
# would miss it.  The rise discriminator (gate 5) provides additional
# false-positive protection absent from Phase B.
NARROW_FFT_SPREAD_CHORD_MIN_BACKWARD_GAIN = 45.0
NARROW_FFT_SPREAD_CHORD_MIN_ENERGY = 5.0
NARROW_FFT_SPREAD_CHORD_NOISE_FACTOR = 3.0
NARROW_FFT_SPREAD_CHORD_HARD_FLOOR = 0.3
# Sustain-phase fr is lower than attack-phase fr; G3 at segment start
# has fr=0.738.  0.70 gives 5% margin.
NARROW_FFT_SPREAD_CHORD_MIN_FR = 0.70
# Rise discriminator: fund_e_segment / fund_e_onset >= 2.0.
# Separates rising notes (ratio >> 2, e.g. G3=23.8, B3=33.2) from
# notes already sustaining at onset (ratio ~1.0).
NARROW_FFT_SPREAD_CHORD_RISE_MIN_RATIO = 2.0
NARROW_FFT_SPREAD_CHORD_DISSONANCE_CENTS = 250.0
NARROW_FFT_SPREAD_CHORD_LOOKBACK_SECONDS = 0.20
# Bg dominance guard: relaxed vs Phase B (0.25 vs 0.5) because the
# rise discriminator (gate 5) already rejects resonance that was
# present at onset time.  Calibrated: G3 bg=46.9, B4 (in-event)
# bg=174.6 → ratio 0.269; 0.25 admits G3 with ~8% margin.
NARROW_FFT_SPREAD_CHORD_BG_DOMINANCE_RATIO = 0.25

# Residual-decay and recent-primary replacement thresholds.
MIN_RECENT_NOTE_ONSET_GAIN = 2.5
RESIDUAL_DECAY_MIN_ONSET_GAIN = 1.5
RECENT_PRIMARY_REPLACEMENT_MIN_SCORE_RATIO = 0.18
RECENT_PRIMARY_REPLACEMENT_MIN_FUNDAMENTAL_RATIO = 0.6
RECENT_PRIMARY_REPLACEMENT_RELAXED_FUNDAMENTAL_RATIO = 0.45
RECENT_PRIMARY_REPLACEMENT_STRONG_ONSET_GAIN = 100.0
RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_GAIN = 20.0
RECENT_PRIMARY_REPLACEMENT_MIN_ONSET_RATIO = 8.0
RECENT_PRIMARY_REPLACEMENT_MAX_DURATION = 0.47
DESCENDING_REPEATED_PRIMARY_MAX_DURATION = 0.47
DESCENDING_REPEATED_PRIMARY_MAX_PRIMARY_ONSET_GAIN = 2.5
DESCENDING_REPEATED_PRIMARY_MIN_REPLACEMENT_ONSET_GAIN = 5.0
DESCENDING_REPEATED_PRIMARY_MIN_REPLACEMENT_ONSET_RATIO = 4.0
DESCENDING_REPEATED_PRIMARY_MIN_SCORE_RATIO = 0.35
DESCENDING_REPEATED_PRIMARY_MIN_FUNDAMENTAL_RATIO = 0.9
DESCENDING_REPEATED_PRIMARY_MIN_INTERVAL_CENTS = 80.0
DESCENDING_REPEATED_PRIMARY_MAX_INTERVAL_CENTS = 220.0

# Secondary promotion heuristics.
RECENT_UPPER_SECONDARY_MIN_DURATION = 0.22
RECENT_UPPER_SECONDARY_PRIMARY_ONSET_GAIN = 20.0
UPPER_SECONDARY_WEAK_ONSET_MIN_DURATION = 0.4
UPPER_SECONDARY_WEAK_ONSET_MAX_GAIN = 30.0
UPPER_SECONDARY_WEAK_ONSET_SCORE_RATIO = 0.14
SHORT_SECONDARY_WEAK_ONSET_MAX_DURATION = 0.36
SHORT_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN = 15.0
SHORT_SECONDARY_WEAK_ONSET_MAX_GAIN = 2.5
SHORT_SECONDARY_WEAK_ONSET_MAX_RATIO = 0.08
SHORT_SECONDARY_WEAK_ONSET_SCORE_RATIO = 0.55
LOWER_SECONDARY_WEAK_ONSET_MAX_DURATION = 0.45
LOWER_SECONDARY_WEAK_ONSET_MIN_PRIMARY_GAIN = 20.0
LOWER_SECONDARY_WEAK_ONSET_MAX_GAIN = 2.5
LOWER_SECONDARY_WEAK_ONSET_MAX_RATIO = 0.08
LOWER_SECONDARY_WEAK_ONSET_SCORE_RATIO = 0.35
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_DURATION = 0.32
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_DURATION = 0.12
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_PRIMARY_ONSET_GAIN = 1.5
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MAX_PRIMARY_FUNDAMENTAL_RATIO = 0.85
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO = 0.95
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_SCORE_RATIO = 0.25
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_UPPER_ALIAS_RATIO = 1.2
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_SUPPORTING_SCORE_RATIO = 0.75
STALE_PRIMARY_UPPER_OCTAVE_PROMOTION_MIN_SCORE = 121.0
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_LOWER_SCORE_RATIO = 0.4
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_LOWER_FUNDAMENTAL_RATIO = 0.85
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_UPPER_FUNDAMENTAL_RATIO = 0.94
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_UPPER_SCORE_RATIO = 0.15
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_PRIMARY_SCORE_RATIO = 0.04
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_FUNDAMENTAL_RATIO_DELTA = 0.12
LOWER_SECONDARY_OCTAVE_PROMOTION_MIN_PRIMARY_INTERVAL_CENTS = 1000.0
LOWER_SECONDARY_OCTAVE_PROMOTION_MAX_PRIMARY_INTERVAL_CENTS = 700.0

# Primary-run and suffix heuristics.
ASCENDING_PRIMARY_RUN_MIN_LENGTH = 3
ASCENDING_PRIMARY_RUN_MAX_DURATION = 0.45
ASCENDING_PRIMARY_RUN_SECONDARY_SCORE_RATIO = 0.35
ASCENDING_PRIMARY_RUN_RECENT_SECONDARY_ONSET_GAIN = 8.0
DESCENDING_PRIMARY_SUFFIX_MIN_LENGTH = 2
DESCENDING_PRIMARY_SUFFIX_MAX_DURATION = 0.5
DESCENDING_PRIMARY_SUFFIX_PRIMARY_ONSET_GAIN = 10.0
DESCENDING_PRIMARY_SUFFIX_UPPER_SCORE_RATIO = 0.6
DESCENDING_REPEATED_PRIMARY_STALE_UPPER_MAX_ONSET_GAIN = 1.5
DESCENDING_REPEATED_PRIMARY_STALE_UPPER_SCORE_RATIO = 0.95
ADJACENT_SEPARATED_DYAD_MAX_DURATION = 0.6
ADJACENT_SEPARATED_DYAD_RUN_MIN_FORWARD_SUPPORT = 2
PRIOR_ONSET_BACKTRACK_SECONDS = 0.55

# Harmonic scoring and repeated-pattern suppression.
HARMONIC_WEIGHTS = [1.0, 0.55, 0.3, 0.15]
HARMONIC_BAND_CENTS = 40.0
SUPPRESSION_BAND_CENTS = 45.0
MAX_POLYPHONY = 4
PRIMARY_REJECTION_MAX_SCORE = 30.0
PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO = 0.97
# #166: narrow onset-based primary rescue.  When the top-ranked candidate
# has very weak attack evidence AND a diluted fundamentalRatio, a top-K
# sibling with clean fR and strong attack is promoted to primary.
# Calibrated against E100 in kalimba-17-g-low-bwv147-sequence-163-01:
#   B3 (primary): score=270.7, fR=0.599, onsetGain=1.8
#   B4 (sibling): score=229.4, fR=0.979, onsetGain=62.0
# See #166 for context.
ONSET_RESCUE_PRIMARY_MAX_ONSET_GAIN = 5.0
ONSET_RESCUE_PRIMARY_MAX_FUNDAMENTAL_RATIO = 0.75
ONSET_RESCUE_SIBLING_MIN_FUNDAMENTAL_RATIO = 0.85
ONSET_RESCUE_SIBLING_MIN_ONSET_GAIN = 20.0
ONSET_RESCUE_SIBLING_GAIN_RATIO = 10.0
ONSET_RESCUE_SIBLING_MIN_SCORE_RATIO = 0.70
ONSET_RESCUE_MAX_SIBLING_RANK = 4
TERTIARY_MIN_SCORE_RATIO = 0.12
TERTIARY_MIN_FUNDAMENTAL_RATIO = 0.85
TERTIARY_MIN_SCORE = 20.0
ITERATIVE_TERTIARY_OCTAVE_MIN_FUNDAMENTAL_RATIO = 0.55
ITERATIVE_RESCUE_MIN_FUNDAMENTAL_RATIO = 0.9
TERTIARY_MIN_ONSET_GAIN = 1.8
TERTIARY_BACKWARD_LOOKBACK_SECONDS = 0.2
TERTIARY_MIN_BACKWARD_ATTACK_GAIN = 20.0
TERTIARY_BACKWARD_GATE_ONSET_OVERRIDE = 20.0
SEMITONE_LEAKAGE_MAX_CENTS = 150.0
SEMITONE_LEAKAGE_MAX_SCORE_RATIO = 0.20
GLISS_CLUSTER_MAX_GAP = 0.06
GLISS_CLUSTER_MAX_EVENT_DURATION = 0.85
GLISS_CLUSTER_MAX_TOTAL_DURATION = 1.2
GLISS_CLUSTER_TARGET_NOTE_COUNT = 3
GLISS_LEADING_SUBSET_MAX_DURATION = 0.18
GLISS_LEADING_SUBSET_SCORE_RATIO = 4.0
GLISS_TERTIARY_MAX_DURATION = 1.35
GLISS_TERTIARY_SCORE_RATIO = 0.02
GLISS_TERTIARY_MIN_SCORE = 20.0
GLISS_TERTIARY_STRONG_ONSET_GAIN = 5.0
GLISS_TERTIARY_WEAK_ONSET_GAIN = 2.0
GLISS_TERTIARY_MIN_FUNDAMENTAL_RATIO = 0.9
# Per-tine attack-to-sustain ratio gate added 2026-04-10 (G3d).  See
# `_extend_gliss_tertiary` in peaks.py.  Distinguishes spurious tertiary
# candidates (where the fundamental band shows a brief noise-like spike with
# little sustain — high a/s ratio) from true tertiary notes (where the
# candidate is a real plucked note with comparable sustain energy — low a/s
# ratio).  Calibrated against:
#   - b4-d5-double-notes-01 E2 F5: a/s = 6.7 (false positive, must reject)
#   - bwv147 E95 G4: a/s = 1.28, E111 B4: 2.36, E155 B4: 1.45 (true)
#   - triple-glissando E4 E4: 2.17, E6 F4: 1.29, E10 C5: 1.48 (true)
# A threshold of 3.0 provides margin: max true a/s = 2.36 (margin 0.64 below)
# while F5 6.7 is well above (margin 3.7).
GLISS_TERTIARY_MAX_ATTACK_TO_SUSTAIN_RATIO = 3.0
# Broadband transient leak gate (Phase A secondary): short segments where a
# low-frequency secondary shows high attack energy but no sustain.  Signature
# of a broadband strike transient leaking into adjacent frequency bands, not a
# real played note.  Calibrated against kalimba-17-g-low E82:
#   B3: AS=2.03 score_ratio=0.387 sustain=21 (false positive, broadband leak)
#   C4: AS=2.12 score_ratio=0.309 sustain=15 (false positive, broadband leak)
#   A3: AS=0.57 (true positive, sustain=80 — must NOT be rejected)
#   D4: AS=0.88 (true positive, primary — not evaluated)
BROADBAND_TRANSIENT_LEAK_MAX_DURATION = 0.15
BROADBAND_TRANSIENT_LEAK_MIN_AS_RATIO = 1.8
BROADBAND_TRANSIENT_LEAK_MAX_SCORE_RATIO = 0.45
LOWER_ROLL_TAIL_EXTENSION_MAX_DURATION = 0.4
LOWER_ROLL_TAIL_EXTENSION_MIN_FUNDAMENTAL_RATIO = 0.95
LOWER_ROLL_TAIL_EXTENSION_MIN_PRIMARY_ONSET_GAIN = 25.0
LOWER_ROLL_TAIL_EXTENSION_MAX_ONSET_GAIN = 5.0
LOWER_ROLL_TAIL_EXTENSION_MIN_SCORE_RATIO = 0.9
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_DURATION = 0.55
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_SCORE_RATIO = 0.08
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_FUNDAMENTAL_RATIO = 0.9
FOUR_NOTE_CONTIGUOUS_CLUSTER_MIN_MARGIN_RATIO = 1.15
FOUR_NOTE_GLISS_EXTENSION_MAX_DURATION = 1.0
FOUR_NOTE_GLISS_EXTENSION_SCORE_RATIO = 0.02
FOUR_NOTE_GLISS_EXTENSION_MIN_SCORE = 18.0
FOUR_NOTE_GLISS_EXTENSION_MIN_FUNDAMENTAL_RATIO = 0.82
CHORD_CLUSTER_MAX_GAP = 0.08
CHORD_CLUSTER_MAX_SINGLETON_DURATION = 0.22
CHORD_CLUSTER_MAX_TOTAL_DURATION = 1.6
REPEATED_PATTERN_LOCAL_CONTEXT_MAX_GAP = 0.35
MAX_HARMONIC_MULTIPLE = 4
SECONDARY_SCORE_RATIO = 0.12
SECONDARY_MIN_FUNDAMENTAL_RATIO = 0.18
SECONDARY_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO = 0.5
# G-low E137 calibration: a secondary candidate whose +1 octave is in
# recent_note_names and whose fR is below this threshold likely represents
# subharmonic-alias leakage from the decaying upper note rather than a fresh
# attack.  Force onset_gain verification in that band.
SECONDARY_SUBHARMONIC_ALIAS_MAX_FUNDAMENTAL_RATIO = 0.75
SHORT_SEGMENT_SECONDARY_SCORE_RATIO = 0.06
OVERTONE_DOMINANT_FUNDAMENTAL_RATIO = 0.18
OVERTONE_DOMINANT_PENALTY_WEIGHT = 0.0
OCTAVE_ALIAS_RATIO_THRESHOLD = 1.15
OCTAVE_ALIAS_MAX_FUNDAMENTAL_RATIO = 0.34
OCTAVE_ALIAS_PENALTY = 0.85
OCTAVE_DYAD_MIN_FUNDAMENTAL_RATIO = 0.32
OCTAVE_DYAD_LOWER_OCTAVE4_MIN_FUNDAMENTAL_RATIO = 0.75
# Calibrated: ghost octave leakage has ratio ≤0.16 (E6→E5: 0.075,
# 0.158; G5→G4: 0.0); genuine dyads have ratio ≥0.60.  0.20 rejects
# all observed ghosts with 3× margin to the nearest genuine case.
OCTAVE_DYAD_MIN_PRIMARY_ENERGY_RATIO = 0.20
OCTAVE_DYAD_UPPER_MIN_FUNDAMENTAL_RATIO = 0.95
OCTAVE_DYAD_UPPER_HARMONIC_ENERGY_RATIO = 0.05
OCTAVE_DYAD_UPPER_SCORE_RATIO = 0.03
LOW_CONFIDENCE_DYAD_MAX_DURATION = 0.25
LOW_CONFIDENCE_DYAD_MAX_SCORE = 120.0
SHORT_SECONDARY_STRIP_MAX_DURATION = 0.28
SHORT_SECONDARY_STRIP_MIN_SCORE = 60.0
SHORT_SECONDARY_STRIP_NEXT_SCORE_RATIO = 5.0
RESTART_STALE_UPPER_STRIP_MIN_INTERVAL_CENTS = 1800.0
RESTART_STALE_UPPER_STRIP_MAX_DURATION = 0.24
ADJACENT_RUN_STRIP_MIN_INTERVAL_CENTS = 80.0
ADJACENT_RUN_STRIP_MAX_INTERVAL_CENTS = 350.0
DESCENDING_STEP_HANDOFF_MAX_DURATION = 0.46
DESCENDING_ADJACENT_UPPER_CARRYOVER_MAX_DURATION = 0.24
DESCENDING_ADJACENT_UPPER_PRIMARY_ONSET_GAIN = 5.0
DESCENDING_ADJACENT_UPPER_SCORE_RATIO = 1.5
DESCENDING_RESTART_UPPER_CARRYOVER_MAX_DURATION = 0.45
DESCENDING_RESTART_UPPER_PRIMARY_ONSET_GAIN = 20.0
DESCENDING_RESTART_UPPER_SCORE_RATIO = 0.2
DESCENDING_PRIMARY_BAND_MIN_LENGTH = 4
DESCENDING_PRIMARY_SUFFIX_UPPER_CARRYOVER_MAX_DURATION = 0.45
DESCENDING_PRIMARY_BAND_PRIMARY_ONSET_GAIN = 5.0
DESCENDING_PRIMARY_SUFFIX_UPPER_SCORE_RATIO = 0.75
RESONANT_CARRYOVER_HIGH_RETURN_MIN_INTERVAL_CENTS = 1800.0
LEADING_SINGLE_TRANSIENT_MAX_DURATION = 0.3
LEADING_SINGLE_TRANSIENT_MAX_SCORE = 150.0
LEADING_SINGLE_TRANSIENT_NEXT_SCORE_RATIO = 8.0
VERY_LOW_CONFIDENCE_EVENT_MAX_SCORE = 2.0

# Evidence gate rescue thresholds (Layer 3.5: _apply_final_decisions)
# Rescue carryover AS ratio cap: don't rescue candidates whose attack energy
# dwarfs sustain (broadband transient blip, not a real re-attack).  Calibrated:
#   kalimba-17-g-low E55 D4: AS=4.52 (false rescue, broadband transient)
#   GLISS_TERTIARY true positives max AS=2.36 — 3.5 gives safe margin.
RESCUE_CARRYOVER_MAX_AS_RATIO = 3.5
RESCUE_MIN_BACKWARD_GAIN = 10.0
RESCUE_MIN_SCORE_RATIO = 0.12
# Onset gain for carryover rescue: a recently-played note that is re-attacked
# (e.g., in a slide_chord) has elevated pre-attack energy from the prior
# sustain, making onset_gain lower than a cold attack.  Threshold 1.8 (was 2.0)
# accommodates this while requiring fR guard to block alias rescue.
# Calibrated: E97 F#4 OG=1.9 is the tightest true positive; lowering from 2.0
# to 1.8 with fR≥0.80 guard selects only 1 new rescue from 27 OG[1.5,2.0] candidates.
RESCUE_CARRYOVER_MIN_ONSET_GAIN = 1.8
RESCUE_CARRYOVER_MIN_SCORE_RATIO = 0.15
# fR guard: block rescue of alias candidates (e.g., E4 fR=0.13) that happen
# to have moderate onset_gain from broadband transient.  Threshold 0.60 is
# conservative: real low-register notes typically have fR≥0.60 (e.g., 34l-c
# E130 D4 fR=0.642), while alias candidates are much lower (0.13, 0.34).
RESCUE_CARRYOVER_MIN_FUNDAMENTAL_RATIO = 0.60
RESCUE_WEAK_ONSET_MIN_FUNDAMENTAL_RATIO = 0.45
RESCUE_WEAK_LOWER_MIN_FUNDAMENTAL_RATIO = 0.50
RESCUE_SCORE_FR_OVERRIDE_MIN_SCORE_RATIO = 0.15
RESCUE_SCORE_FR_OVERRIDE_MIN_FR = 0.75
BRIDGING_OCTAVE_PAIR_MAX_DURATION = 0.4
BRIDGING_OCTAVE_PAIR_MAX_SCORE = 600.0
SPLIT_UPPER_OCTAVE_PAIR_MIN_DURATION = 0.28
SPLIT_UPPER_OCTAVE_PAIR_MAX_DURATION = 0.7
SPLIT_UPPER_OCTAVE_PAIR_PRIMARY_SCORE_MAX = 800.0
SPLIT_UPPER_OCTAVE_PAIR_FRACTION = 0.45
SAME_START_PRIMARY_SINGLETON_MAX_START_DELTA = 0.02
OVERLAPPING_PRIMARY_SINGLETON_MIN_START_DELTA = 0.05
OVERLAPPING_PRIMARY_SINGLETON_MIN_OVERLAP = 0.08
OVERLAPPING_PRIMARY_SINGLETON_MAX_DURATION = 0.4

# Direct UTF-8 note labels keep the mapping readable.
PITCH_CLASS_TO_DOREMI = {
    "C": "ド",
    "C#": "ド#",
    "Db": "レb",
    "D": "レ",
    "D#": "レ#",
    "Eb": "ミb",
    "E": "ミ",
    "F": "ファ",
    "F#": "ファ#",
    "Gb": "ソb",
    "G": "ソ",
    "G#": "ソ#",
    "Ab": "ラb",
    "A": "ラ",
    "A#": "ラ#",
    "Bb": "シb",
    "B": "シ",
}

PITCH_CLASS_TO_NUMBER = {
    "C": "1",
    "C#": "#1",
    "Db": "b2",
    "D": "2",
    "D#": "#2",
    "Eb": "b3",
    "E": "3",
    "F": "4",
    "F#": "#4",
    "Gb": "b5",
    "G": "5",
    "G#": "#5",
    "Ab": "b6",
    "A": "6",
    "A#": "#6",
    "Bb": "b7",
    "B": "7",
}
