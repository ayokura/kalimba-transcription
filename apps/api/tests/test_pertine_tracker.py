"""Mechanism tests for the per-tine phase-tracking rescue judge (#141 S5).

Constructed-input tests only (no full pipeline): synthetic decaying tones
with controlled re-strikes exercise the physical conditions (pre-ring,
re-injection, phase break), and Rescue dataclasses exercise the tier rule.
"""
from __future__ import annotations

import numpy as np

from app.transcription.pertine import (
    Rescue,
    _dedup_same_note,
    load_tables,
    tier_of,
    track_and_rescue,
)

SR = 16000


def _strike(t_axis: np.ndarray, t0: float, freq: float, amp: float,
            phase: float, tau: float = 0.5) -> np.ndarray:
    """Exponentially decaying tone starting at t0 with a ~5 ms attack ramp.

    tau=0.5 puts the 1.5 s-later re-strike at ~20x the residual ring — the
    regime the detector is calibrated for (the demod low-pass smears the
    attack over ~30 ms, so jerk >= JERK_BAR needs a strike well above the
    residual, matching real carryover re-strikes)."""
    dt = t_axis - t0
    env = np.where(dt >= 0, amp * np.exp(-np.maximum(dt, 0.0) / tau), 0.0)
    ramp = np.clip(dt / 0.005, 0.0, 1.0)
    return env * ramp * np.sin(2 * np.pi * freq * t_axis + phase)


def test_restrike_of_ringing_tine_is_rescued_at_event_tier():
    t = np.arange(int(SR * 3.2)) / SR
    audio = _strike(t, 0.5, 440.0, 1.0, 0.0) + _strike(t, 2.0, 440.0, 1.0, np.pi / 2)
    rescues = track_and_rescue(
        audio, SR, [("A4", 440.0)], existing=[(0.5, "A4")],
    )
    assert any(abs(r.time - 2.0) <= 0.06 and r.note == "A4" for r in rescues), rescues
    hit = next(r for r in rescues if abs(r.time - 2.0) <= 0.06)
    assert tier_of(hit) == "event"
    # the re-strike is the only proposal — the first strike is covered
    assert all(abs(r.time - 2.0) <= 0.06 for r in rescues), rescues


def test_fresh_strike_of_quiet_tine_is_not_rescued():
    # No pre-ring: a fresh strike is broadband's job (condition 4).
    t = np.arange(int(SR * 3.2)) / SR
    audio = _strike(t, 2.0, 440.0, 1.0, 0.0)
    rescues = track_and_rescue(audio, SR, [("A4", 440.0)], existing=[])
    assert rescues == []


def test_duplicate_of_existing_event_is_suppressed():
    # Same synthetic re-strike, but the recognizer already has an event
    # within EXISTING_TOL (recognizer starts deviate up to ~0.1 s).
    t = np.arange(int(SR * 3.2)) / SR
    audio = _strike(t, 0.5, 440.0, 1.0, 0.0) + _strike(t, 2.0, 440.0, 1.0, np.pi / 2)
    rescues = track_and_rescue(
        audio, SR, [("A4", 440.0)], existing=[(0.5, "A4"), (2.1, "A4")],
    )
    assert rescues == []


def test_tier_requires_pre_ring_margin():
    weak = Rescue(1.0, "A4", 2.0, 300.0, 0.5, pre_ring_ratio=1.2,
                  reinject_ratio=3.0, attackers=[])
    assert tier_of(weak) == "candidate"
    strong = Rescue(1.0, "A4", 2.0, 300.0, 0.5, pre_ring_ratio=2.0,
                    reinject_ratio=1.1, attackers=[])
    assert tier_of(strong) == "event"  # isolated: no re-inject margin needed


def test_tier_demotes_rescue_dominated_by_coincident_attack():
    atk = [{"note": "D5", "dt": 0.01, "kind": "event", "cents": 100.0,
            "envRatio": 0.5}]
    dominated = Rescue(1.0, "D#5", 2.0, 300.0, 0.5, pre_ring_ratio=2.0,
                       reinject_ratio=3.0, attackers=atk)
    assert tier_of(dominated) == "candidate"
    atk_weak = [{"note": "D5", "dt": 0.01, "kind": "event", "cents": 100.0,
                 "envRatio": 2.5}]
    dominant = Rescue(1.0, "D#5", 2.0, 300.0, 0.5, pre_ring_ratio=2.0,
                      reinject_ratio=3.0, attackers=atk_weak)
    assert tier_of(dominant) == "event"
    low_reinject = Rescue(1.0, "D#5", 2.0, 300.0, 0.5, pre_ring_ratio=2.0,
                          reinject_ratio=1.2, attackers=atk_weak)
    assert tier_of(low_reinject) == "candidate"


def test_same_note_double_fire_keeps_stronger_jerk():
    a = Rescue(1.000, "A4", 2.0, 300.0, 0.5)
    b = Rescue(1.060, "A4", 2.0, 200.0, 0.4)
    c = Rescue(1.500, "A4", 2.0, 250.0, 0.4)
    kept = _dedup_same_note([a, b, c])
    assert [(r.time, r.jerk) for r in kept] == [(1.000, 300.0), (1.500, 250.0)]


def test_load_tables_unknown_tuning_is_empty():
    partial, coupling = load_tables("no-such-tuning")
    assert partial == {} and coupling == {}
