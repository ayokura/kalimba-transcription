# Recognizer Local Rules Ledger

## Purpose

This document tracks recognizer rules that are either:
- effectively fixture-specific, or
- narrow enough that they may become debt for future free-performance transcription without expected metadata.

The goal is not to remove these rules immediately. The goal is to keep them visible so they can be revisited when:
- BWV147 practical coverage is broader,
- scoped-fixture debt is reduced,
- or free-performance recognition becomes a first-class target.

## Classification

### Keep For Now

These are local rules, but they currently protect multiple practical fixtures or broad onset quality rather than a single clip.

| Rule / helper | Code path | Current role | Protected practical fixtures / tests | Why keep for now | Future removal trigger |
| --- | --- | --- | --- | --- | --- |
| `collapse_active_range_head_onsets` | `apps/api/app/transcription/segments.py` | Collapses dense head-onset clusters inside an active range to reduce over-segmentation at phrase starts. | `#30` practical recovery set, verified through `apps/api/tests/test_api.py` and current completed/manual-capture suites | This is a cross-fixture onset-shape fix, not a single BWV rule. | Remove only after a more principled onset-boundary model replaces head-cluster collapse. |
| `refine_onset_times_by_attack_profile` | `apps/api/app/transcription/profiles.py` | Replaces a nearby weak / invalid onset with a stronger valid-attack onset. | `#30` practical recovery set and current manual-capture regressions | This is still local, but it is attack-profile based and broadly useful. | Remove only after onset detection itself reliably resolves close invalid-valid pairs. |

### Candidate For Later Removal

These rules are useful today, but they are strongly tied to particular practical failure shapes. They should be treated as debt and periodically re-evaluated.

| Rule / helper | Code path | Processing summary | Primary purpose | Protected practical fixtures / tests | Debt signal | Preferred long-term replacement |
| --- | --- | --- | --- | --- | --- | --- |
| ~~`collect_two_onset_terminal_tail_segments`~~ | **削除済み** (#122) — AVC trailing collector で代替。詳細は [deferred-ideas.md](deferred-ideas.md) の「Terminal collector 5種」を参照。 | — | — | — | — | — |
| `recent-upper-octave-alias-primary` promotion | `apps/api/app/transcription/peaks.py` (`maybe_promote_recent_upper_octave_alias_primary`) | Promotes an upper octave candidate to primary when the current primary looks like a recent lower alias and support is present. | Recover the correct high-register note at restart tails. | `kalimba-17-c-bwv147-restart-tail-01`; `test_bwv147_restart_tail_promotes_recent_upper_octave_alias` | Strongly tied to a high-register alias failure at the tail of a restart phrase. | Better alias modeling at candidate ranking time, instead of post-hoc primary replacement. |
| `collapse_restart_tail_subset_into_following_chord` | `apps/api/app/transcription/events.py` | Drops a transient subset event when a following chord clearly absorbs it. | Suppress restart-tail subset blips before the final chord. | `kalimba-17-c-bwv147-restart-tail-01` practical regression path | Pattern-specific post-merge cleanup for one restart-tail failure mode. | Better event bundling around restart tails so subset blips are not emitted. |
| `suppress_recent_upper_echo_mixed_clusters` | `apps/api/app/transcription/patterns.py` | Removes a short upper echo and strips repeated upper carryover from a following mixed cluster. | Prevent extra upper-note echoes / over-bundling in mixed upper BWV phrases. | `kalimba-17-c-bwv147-upper-mixed-cluster-01`; `test_bwv147_upper_cluster_recovers_delayed_terminal_e5` | Highly shaped around one mixed upper echo pattern. | Better carryover / resonance modeling before post-merge cleanup. |
| `lower-mixed-roll-extension` | `apps/api/app/transcription/peaks.py` (inline in `segment_peaks`) | Adds one missing lower/mixed note from residual candidates when onset and score constraints are met. | Recover missing `G4`-class lower notes inside mixed lower-roll phrases. | `kalimba-17-c-bwv147-lower-mixed-roll-01`; `kalimba-17-c-bwv147-lower-context-roll-01`; `test_bwv147_lower_mixed_roll_recovers_opening_mixed_dyad_and_long_gap_run`; `test_bwv147_lower_context_roll_matches_completed_nine_event_phrase` | Narrowly tuned to BWV lower-roll vocabulary and onset shape. | Better multi-note selection for lower/mixed phrases without a dedicated extension rule. |
| ~~`collect_post_sparse_gap_run_segments`~~ | **Removed** — replaced by `_promote_gap_candidates_by_structure` via candidate fallback in `collect_multi_onset_gap_segments`. | — | — | — | — | — |

## Fixture Coverage Map

### BWV147 fixtures currently protected by local rules

| Fixture | Status | Local rules that materially protect it | Notes |
| --- | --- | --- | --- |
| `kalimba-17-c-bwv147-restart-prefix-01` | `pending` | none that are clearly fixture-specific today | 4-note chord (C4+E4+G4+E5) の検出が未完。比較的クリーンな restart baseline。 |
| `kalimba-17-c-bwv147-upper-mixed-cluster-01` | `completed` | `suppress_recent_upper_echo_mixed_clusters`, delayed-terminal-orphan path | Main risk is upper echo / bundle overgrowth. |
| `kalimba-17-c-bwv147-restart-tail-01` | `pending` | `recent-upper-octave-alias-primary`, `collapse_restart_tail_subset_into_following_chord` | 3-note chord の検出が未完。restart-tail-specific cleanup debt の典型例。 |
| `kalimba-17-c-bwv147-late-upper-tail-01` | `completed` | AVC trailing collector (terminal collectors は #122 で削除済み) | Sparse terminal-tail rescue. |
| `kalimba-17-c-bwv147-lower-context-roll-01` | `completed` | `lower-mixed-roll-extension`, candidate promotion via `_promote_gap_candidates_by_structure` | Context-preserved lower-roll phrase. |
| `kalimba-17-c-bwv147-lower-mixed-roll-01` | `completed` | `lower-mixed-roll-extension`, candidate promotion via `_promote_gap_candidates_by_structure` | Scoped-evaluation 解消により completed に昇格。 |
| `kalimba-17-c-bwv147-lower-f4-mixed-run-01` | `pending` | none yet beyond general recognizer behavior | Added as a non-overlapping lower/F4 coverage fixture; current miss shape is still open. |
| `kalimba-17-c-bwv147-mid-gesture-cluster-01` | `review_needed` | none should be added currently | Main blocker is boundary contamination and failed re-windowing, not recognizer logic. |
| `kalimba-17-c-bwv147-upper-transition-01` | `reference_only` | none should be added currently | Provenance/scoping mismatch, not an active recognizer target. |
| `kalimba-17-c-bwv147-restart-high-register-01` | `reference_only` | none should be added currently | Superseded by cleaner restart subsets. |

### Non-BWV practical coverage touched by local onset logic

| Rule / area | Protected practical scope |
| --- | --- |
| `collapse_active_range_head_onsets` | Current `#30` practical fixture recovery set and associated completed/manual-capture regressions |
| `refine_onset_times_by_attack_profile` | Current `#30` practical fixture recovery set and associated completed/manual-capture regressions |

## Review Guidance

When adding another local rule, record all of the following here:
- exact helper / promotion / suppression name
- whether it is "Keep For Now" or "Candidate For Later Removal"
- the fixture(s) that require it
- the concrete failure shape it patches
- the long-term replacement that would make it removable

When free-performance recognition becomes an active goal, review the "Candidate For Later Removal" table first.
