"""Shared runtime settings for the recognizer pipeline.

Values here are intended to be overridden in tests, diagnostic scripts,
and future side-by-side experiments.  Structural constants that should
never change at runtime stay in ``constants.py``.

Usage in production code::

    from . import settings
    cfg = settings.get()
    if cfg.use_attack_validated_gap_collector:
        ...

Usage in tests::

    from app.transcription.settings import override

    with override(ablate_multi_onset_gap=True):
        ...
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator

from .constants import (
    PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO,
    PRIMARY_REJECTION_MAX_SCORE,
)


@dataclass(frozen=True, slots=True)
class RecognizerSettings:
    """Recognizer feature flags and ablation switches.

    All fields must have defaults matching production behaviour.
    """

    # Feature flags
    use_attack_validated_gap_collector: bool = True
    filter_gap_onsets_by_attack_profile: bool = True
    use_iterative_harmonic_suppression: bool = True
    use_evidence_gate_rescue: bool = True
    use_multi_primary_branching: bool = True
    use_onset_gate: bool = True  # #141: reject primary with no onset evidence
    use_alternate_groupings: bool = True  # #151: dissonance-aware merge guard
    use_soft_candidate_alternates: bool = True  # #178: preserve soft-rejected candidates as alternates
    # Phase C: context-aware rescue pass after Phase A/B and evidence rescue.
    # Revisits candidates rejected in Phase A with score-below-threshold only,
    # where allow_octave_secondary(primary, hypothesis, selected_final) now
    # returns True.  Catches upper-octave notes in octave-dyad chords whose
    # subharmonic_alias_energy penalty was driven by a legitimate lower-octave
    # partner that is itself in the final selected set (see Free Performance
    # E1 C4+C5+E5: C5 score=12 due to C4 real-note sub-alias).
    use_phase_c_octave_dyad_rescue: bool = True
    # #149: per-tine partial scoring in rank_tuning_candidates.
    # Currently disabled because kalimba tunings contain fifths as adjacent
    # tines, so beam partials (e.g. 1.5×) collide with other notes' fundamentals.
    # The existing fR-based alias detection is not designed for this collision
    # case and mis-penalises legitimate chords.  Per-tine partials remain
    # active in suppress_harmonics (which handles collisions via the
    # fundamental guard) — see 2cd5a7a.
    use_per_tine_partial_scoring: bool = False
    # #141 S5 round 2: per-tine phase-tracking rescue judge (research line,
    # dual-run). Post-stage only — proposes carryover re-strike events and
    # low-confidence candidate slots, never removes broadband events.
    # Default ON on the research branch so the fixture suite and the corpus
    # benchmark measure the integrated recognizer (kill criteria C1/K2-K4);
    # main is the dual-run baseline.
    use_pertine_tracker_rescue: bool = True
    # #206 / #141 S6 round 3: per-tine veto over the residual-decay bulk
    # rejection — dropped residual-decay-no-reattack slots are adjudicated by
    # the tracker's detection core (strict slot-window semantics) and firing
    # tines are promoted to events. Nested inside use_pertine_tracker_rescue.
    # DEFAULT OFF — round-3 measurement (2026-07-06) returned a clean
    # negative: the 2x2 ablation isolated the integrated net effect to +1 GT
    # FP on 70cc6637 with zero recall gain (probe-projected recoveries are
    # suppressed by the calibration guards or already covered by the round-2
    # rescue / forward-scan), and the #206 metamorphic WARN did not resolve
    # (dropped 5 -> 4+2 time-shifted, criterion was ->0). Kept as a measured
    # negative-result asset (docs/research/pertine-round3-ablation.json).
    use_pertine_residual_autopsy: bool = False

    # Ablation switches (True = disable the feature)
    ablate_sparse_gap_tail: bool = False
    # #206 round 3 2x2 ablation: disable the residual-forward-scan (recent-
    # note mute-dip scan + octave-up rescue) inside _resolve_primary, so a
    # residual-decay segment rejects without the recent-note-memory rescue.
    # The replacement claim "fscan off + autopsy on >= fscan on + autopsy off"
    # is measured over this switch x use_pertine_residual_autopsy.
    ablate_residual_forward_scan: bool = False
    ablate_multi_onset_gap: bool = False
    # #197: trailing strummed-chord cluster rescue (one segment per cluster of
    # >=2 gap-validated trailing onsets with a valid-attack anchor).
    ablate_trailing_chord_cluster: bool = False
    ablate_collapse_active_range_head: bool = False
    ablate_snap_range_start_to_onset: bool = False

    # Gate-level ablation: set of gate reason strings to skip.
    # See GATE_CATEGORIES in peaks.py for the full list.
    # Example: frozenset({"recent-carryover-candidate", "weak-upper-secondary"})
    disabled_gates: frozenset[str] = frozenset()

    # Tunable thresholds (#131 Phase 2). Defaults are sourced from constants.py,
    # where their calibration is documented; exposing them here makes them
    # overridable in mechanism tests and in-process diagnostics via override()
    # without mutating source. The fixture_rejection_sweep.py source-rewrite path
    # still works (a subprocess re-imports constants, which flows into these
    # defaults). Migrated incrementally — see #131.
    primary_rejection_max_score: float = PRIMARY_REJECTION_MAX_SCORE
    primary_rejection_max_fundamental_ratio: float = PRIMARY_REJECTION_MAX_FUNDAMENTAL_RATIO


def _apply_env_overrides(base: RecognizerSettings) -> RecognizerSettings:
    """KALIMBA_SETTINGS_OVERRIDES (JSON) をプロセス起動時に反映する。

    ablation observatory (第 2 期 S4) が実 pytest スイート / benchmark の
    サブプロセスへトグルを渡すための唯一の経路。ランタイム中の変更は
    見ない (import 時に一度だけ読む)。未知キーは無視、disabled_gates は
    list → frozenset に変換。prod では未設定なので no-op。
    """
    raw = os.environ.get("KALIMBA_SETTINGS_OVERRIDES")
    if not raw:
        return base
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return base
    if not isinstance(data, dict):
        return base
    valid = {f.name for f in fields(RecognizerSettings)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in valid:
            continue
        if key == "disabled_gates" and isinstance(value, list):
            value = frozenset(value)
        kwargs[key] = value
    return replace(base, **kwargs) if kwargs else base


_DEFAULTS = _apply_env_overrides(RecognizerSettings())
_current: RecognizerSettings = _DEFAULTS


def get() -> RecognizerSettings:
    """Return the active settings snapshot."""
    return _current


@contextmanager
def override(**kwargs: Any) -> Iterator[RecognizerSettings]:
    """Temporarily replace settings for the duration of a ``with`` block.

    Only fields defined on :class:`RecognizerSettings` are accepted;
    unknown keys raise ``TypeError``.

    ::

        with override(ablate_multi_onset_gap=True):
            # ablation is active inside this block
            ...
        # original settings restored here
    """
    global _current
    previous = _current
    _current = replace(_current, **kwargs)
    try:
        yield _current
    finally:
        _current = previous


def reset() -> None:
    """Restore production defaults.  Useful in test teardown."""
    global _current
    _current = _DEFAULTS
