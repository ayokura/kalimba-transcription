# Strategy B: Gap Candidates And Promotion

## Findings

- `collect_multi_onset_gap_segments()` rejects `kalimba-17-c-bwv147-lower-context-roll-01` because its long gap does not satisfy the collector's `0.18s-0.42s` short-interval gate. The gap has onset intervals `0.048, 1.5093, 0.6187, 0.6213, 1.2454, 1.4, 0.12`, so `short_interval_count` is `0` and the collector returns no segments.
- Directly enabling `collect_attack_validated_gap_segments()` improves that BWV gap, but it over-detects on 10/32 completed fixtures because it promotes every valid attack onset in leading, inter-range, and trailing gaps.
- The common Strategy A failure mode is not false attack validation. It is over-promotion. Leading and trailing housekeeping onsets become full segments, and inter-gap singleton attacks split otherwise stable phrases.

## Strategy A Failure Set (2026-03-27)

Direct attack-validated emission currently regresses these completed fixtures:

- `kalimba-17-c-a4-d4-f4-triad-repeat-01`
- `kalimba-17-c-bwv147-restart-prefix-01`
- `kalimba-17-c-c4-e4-g4-triad-repeat-01`
- `kalimba-17-c-c4-to-e6-sequence-17-repeat-03-01`
- `kalimba-17-c-c4-to-g4-sequence-17-01`
- `kalimba-17-c-d4-d5-octave-dyad-01`
- `kalimba-17-c-d4-d5-sequence-01`
- `kalimba-17-c-d4-repeat-01`
- `kalimba-17-c-e4-g4-b4-triad-repeat-01`
- `kalimba-17-c-mixed-sequence-01`

The dominant failure classes are:

- repeated-note or repeated-triad fixtures gaining extra leading/trailing singleton segments
- scoped practical fixtures gaining an extra inter-gap singleton that steals a note from a neighboring chord
- longer sequence fixtures over-splitting one phrase and then missing a later required occurrence because note counts drift

## Prototype Shape

The prototype keeps `collect_post_sparse_gap_run_segments()` in place, but introduces a reusable candidate/promotion path:

1. `collect_attack_validated_gap_candidates()` derives valid-attack onset pools for leading, inter-range, and trailing gaps.
2. `collect_attack_validated_gap_segments()` becomes a thin segment emitter over that candidate pool.
3. `collect_multi_onset_gap_segments()` gains a candidate-consuming fallback. When its legacy short-interval gate fails, it can promote a regular run of valid attack candidates into gap segments.

In the current prototype, the candidate fallback emits:

- the first regular run interval,
- a short tail segment anchored at the last run onset,
- an optional follow-up promoted onset after a large post-run gap,
- an optional dense-cluster tail using the relaxed gap onset list.

This is enough to make `kalimba-17-c-bwv147-lower-context-roll-01` pass even when `ABLATE_POST_SPARSE_GAP_RUN = True`.

## Strategy A vs Strategy B

### Strategy A: Direct attack-validated segment emission

Pros:
- Smallest implementation delta.
- Easy to reason about locally.

Cons:
- Couples evidence (`is_valid_attack`) with promotion (`make a segment now`).
- Leading/trailing residual attacks become full events.
- Tuning thresholds becomes brittle because the same knob must solve both "is this a real onset?" and "should this onset become a segment?".
- More likely to break on new instruments or recording environments where attack density changes.

### Strategy B: Candidate pool plus structural promotion

Pros:
- Separates onset evidence from segmentation policy.
- Existing collectors keep their structural logic and can consume better gap evidence.
- Over-detection control becomes collector-specific instead of threshold-only.
- Easier to extend for other instruments because candidate generation and promotion policy can evolve independently.

Cons:
- Slightly more architecture.
- Requires deciding which collectors should consume candidate pools and which should stay raw-onset based.

Recommendation:
- Prefer Strategy B. The current failures show that the main problem is promotion policy, not candidate validity.

## 3a vs 3b

### 3a: librosa gap onsets with attack-candidate supplementation

Pros:
- Lowest migration cost.
- Keeps existing collectors almost unchanged.

Cons:
- Collectors still need to reason about two onset populations.
- Raw gap onsets remain the default even when their validation quality is lower.
- Harder to explain which source produced a segment.

### 3b: collectors consume attack-derived candidates directly

Pros:
- Cleaner contract: structural collectors operate on a single canonical gap candidate list.
- Better separation between evidence and promotion.
- Easier debug story and future generalization.

Cons:
- Some residual shapes still benefit from relaxed raw gap onsets, especially dense tail clusters.
- In the current codebase, attack profiles are computed only for detected onset times, so a true librosa-free 3b would require a separate onset proposal stage before validation.

Recommendation:
- Use 3b as the default for structural promotion, but allow narrow raw-onset fallback for residual cluster detection.
- The current prototype is intentionally hybrid in that sense: valid attack candidates drive the run/follow-up promotions, while the dense-cluster tail still looks at relaxed gap onsets.

## Current Gate

The intended gate for this direction is:

- `kalimba-17-c-bwv147-lower-context-roll-01` passes with `ABLATE_POST_SPARSE_GAP_RUN = True`.
- All completed manual-capture fixtures still pass in the default configuration.

That does not yet justify deleting `collect_post_sparse_gap_run_segments()`, but it is enough to show that a generic candidate/promotion path can cover the BWV lower-context gap without re-enabling direct attack-validated emission.
