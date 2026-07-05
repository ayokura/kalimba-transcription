"""per-tine tracker v0 — rescue-only judge (S5 round 1, #141 research line).

Minimal-coupling form from docs/research/2026-07-per-tine-tracker-design.md:
the tracker does NOT replace broadband detection; it proposes *additional*
note onsets (carryover-mask rescue) that the main recognizer missed, using
continuous per-tine state instead of the probe's winner-take-all guards.

Detection core (shared with the calibrated ROC probe): heterodyne
demodulation per tine, phase-trend RMS error >= PHASE_BAR and relative
envelope jerk >= JERK_BAR at the calibrated reference combo (0.7 / 150),
with the fixed gates (absolute env floor, sustain-holds-10% ring test).

Explaining-away (replaces dominance / harmonic-parent guards):
1. self ring-out — inherent: a ringing tine only fires on a genuine phase
   break + fresh jerk (this is what the guards could not express).
2. measured-partial bleed — a candidate at note k coincident (+-30 ms) with
   a much stronger candidate at note j is rejected only if j's *measured*
   partial (in-band collision table) lands on k within +-50 cents AND k's
   envelope is explainable by j's envelope x measured bleed amplitude x
   safety factor. No integer-harmonic assumption; instruments without a
   measured table simply skip this term (partial term is optional by design).
3. skirt bleed — quantified: neighbour within SKIRT_CENTS whose envelope at
   the candidate instant exceeds the candidate's by >= SKIRT_ENV_RATIO.
   (The demod low-pass half-band is ~43 cents, so at >=250 cents a genuine
   skirt leak requires a hugely louder neighbour — winner-take-all by jerk
   is replaced by an amplitude ratio with a physical rationale.)

The module is evaluation-driven (round 1 = offline dual-run); pipeline
integration is round 2, gated on the offline numbers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_tracking_roc import (  # noqa: F401  (shared calibrated core)
    ENV_GATE_FRAC,
    HOP_SEC,
    candidates_for_track,
    demodulate,
    nms,
)

REPO = Path(__file__).resolve().parents[3]

PHASE_BAR = 0.7
JERK_BAR = 150.0
COINCIDENT_SEC = 0.03
PARTIAL_CENTS = 50.0
PARTIAL_SAFETY = 3.0      # env_k <= relAmp * env_j * SAFETY -> explainable as bleed
SKIRT_CENTS = 300.0
SKIRT_ENV_RATIO = 20.0
EXISTING_TOL = 0.08       # candidate within this of an existing same-note event -> not a rescue

# Round-1.5 state conditions (dual-run round 1 showed the calibrated bars
# alone leak FP 53->1678; the state model has to carry the precision):
# (4) carryover-class restriction — rescue only a tine that was already
#     ringing just before the candidate (median env over PRE_RING_WIN >=
#     the absolute gate). The K2 quarry is exactly this class; a quiet
#     tine's fresh strike is broadband's job, not the tracker's.
PRE_RING_WIN = (-0.25, -0.05)
# (5) energy re-injection — a genuine re-strike pushes the envelope at or
#     above its recent ring level; beating/vibrato oscillates strictly
#     below the decaying peak. Attack peak must reach REINJECT_FRAC x the
#     recent max (REINJECT_LOOKBACK window).
REINJECT_FRAC = 1.0
REINJECT_LOOKBACK = 0.5
# (6) coincident-attack bleed bound — round-1.5 leak analysis: residual FPs
#     cluster at other tines' strike instants (dense BWV sequences, 125/133
#     fixture additions). Any strike's bleed into another band was measured
#     at <= 0.352 x striker (A4->E6, the largest fresh collision), so a
#     candidate coincident with a stronger attack and below COINC_BLEED_MAX
#     of it is explainable as bleed regardless of instrument/partial table.
#     Amplitude-quantified — not the jerk winner-take-all this replaces.
COINC_ATTACK_SEC = 0.05
COINC_BLEED_MAX = 0.4
# (7) measured per-pair coupling — per instrument group, the fresh energy a
#     strike of tine j injects at tine k's own fundamental (mount coupling /
#     skirt / unlabelled partials), measured from fully-labelled GT strikes
#     (build_coupling_table.py). A candidate coincident with j's attack whose
#     envelope is within COUPLING_SAFETY x the measured p90 transfer is
#     explainable without invoking a fresh strike of k.
COUPLING_SAFETY = 2.0

PARTIAL_TABLE_PATH = REPO / "docs" / "research" / "per-tine-partial-table.json"
COUPLING_TABLE_PATH = REPO / "docs" / "research" / "per-tine-coupling-table.json"


def load_coupling_table(group: str) -> dict[str, dict[str, float]]:
    """striker -> {victim: attack-window p90 transfer ratio}; {} if absent.

    attackP90 (transfer during the strike transient) is the right bound for
    tracker candidates, which fire on attacks; body-window p90 is the
    fallback for pairs where the attack measurement is missing."""
    if not COUPLING_TABLE_PATH.is_file():
        return {}
    data = json.loads(COUPLING_TABLE_PATH.read_text())
    g = data.get("groups", {}).get(group, {})
    return {j: {k: (v.get("attackP90") or v["p90"]) for k, v in d.items()}
            for j, d in g.items()}


@dataclass
class Rescue:
    time: float
    note: str
    phase_err: float
    jerk: float
    env: float


def load_partial_table(group: str) -> dict[str, list[tuple[float, float]]]:
    """note -> [(ratio, medianRelAmp)] for an instrument group; {} if absent."""
    if not PARTIAL_TABLE_PATH.is_file():
        return {}
    data = json.loads(PARTIAL_TABLE_PATH.read_text())
    g = data.get("groups", {}).get(group)
    if not g:
        return {}
    out = {}
    for note, entry in g.get("notes", {}).items():
        # weak clusters flagged relaxedOnly are excluded from load-bearing use
        parts = [(p["ratio"], p["medianRelAmp"]) for p in entry.get("partials", [])
                 if not p.get("relaxedOnly")]
        if parts:
            out[note] = parts
    return out


def track_and_rescue(
    audio: np.ndarray,
    sr: int,
    tuning: list[tuple[str, float]],
    existing: list[tuple[float, str]],
    partial_table: dict[str, list[tuple[float, float]]] | None = None,
    coupling_table: dict[str, dict[str, float]] | None = None,
) -> list[Rescue]:
    """Propose rescue onsets the existing event list does not cover."""
    hop = int(sr * HOP_SEC)
    hop_sec = hop / sr
    tracks = {}
    for name, freq in tuning:
        env, phase = demodulate(audio, sr, freq, hop)
        tracks[name] = (freq, env, phase)
    global_peak = max(t[1].max() for t in tracks.values()) or 1.0
    abs_gate = global_peak * ENV_GATE_FRAC
    # calibrated candidate pass + per-tine NMS
    cands = []  # (t, note, err, jerk, freq, env_at)
    for name, (freq, env, phase) in tracks.items():
        hits = candidates_for_track(env, phase, hop_sec, abs_gate, JERK_BAR)
        sel = [(i, e, j) for i, e, j in hits if e >= PHASE_BAR]
        for i, e, j in nms(sel, hop_sec):
            cands.append((i * hop_sec, name, e, j, freq, float(env[i])))
    cands.sort()
    freqs = {n: f for n, f in tuning}

    def env_at(note: str, t: float) -> float:
        _f, env, _p = tracks[note]
        i = min(len(env) - 1, max(0, int(t / hop_sec)))
        return float(env[i])

    hop_len = hop_sec

    def env_window_stat(note: str, t0: float, t1: float, fn) -> float:
        _f, env, _p = tracks[note]
        a = max(0, int(t0 / hop_len)); b = min(len(env), max(a + 1, int(t1 / hop_len)))
        return float(fn(env[a:b]))

    rescues: list[Rescue] = []
    for t, name, err, jerk, freq, e_self in cands:
        # already covered by an existing event of the same note?
        if any(n == name and abs(t0 - t) <= EXISTING_TOL for t0, n in existing):
            continue
        # (4) carryover-class restriction: the tine must already be ringing
        pre_ring = env_window_stat(name, t + PRE_RING_WIN[0], t + PRE_RING_WIN[1], np.median)
        if pre_ring < abs_gate:
            continue
        # (5) energy re-injection: attack peak reaches its recent ring maximum
        attack_peak = env_window_stat(name, t, t + 0.06, np.max)
        recent_max = env_window_stat(name, t - REINJECT_LOOKBACK, t - 0.01, np.max)
        if attack_peak < recent_max * REINJECT_FRAC:
            continue
        # (2) measured-partial bleed: candidate k explainable as j's partial?
        explained = False
        if partial_table:
            for t2, n2, _e2, _j2, f2, e2 in cands:
                if n2 == name or abs(t2 - t) > COINCIDENT_SEC:
                    continue
                for ratio, rel_amp in partial_table.get(n2, []):
                    pf = f2 * ratio
                    if abs(1200 * np.log2(freq / pf)) <= PARTIAL_CENTS \
                            and e_self <= rel_amp * e2 * PARTIAL_SAFETY:
                        explained = True
                        break
                if explained:
                    break
        if explained:
            continue
        # (3) quantified skirt bleed from a near-frequency neighbour
        skirt = False
        for n2, f2 in freqs.items():
            if n2 == name:
                continue
            if abs(1200 * np.log2(f2 / freq)) <= SKIRT_CENTS \
                    and env_at(n2, t) >= e_self * SKIRT_ENV_RATIO:
                skirt = True
                break
        if skirt:
            continue
        # (6) coincident-attack bleed bound: strongest simultaneous attack on
        # any other tine (candidate or existing event)
        coinc_max = 0.0
        for t2, n2, _e2, _j2, _f2, e2 in cands:
            if n2 != name and abs(t2 - t) <= COINC_ATTACK_SEC:
                coinc_max = max(coinc_max, e2)
        for t0, n0 in existing:
            if n0 != name and n0 in tracks and abs(t0 - t) <= COINC_ATTACK_SEC:
                coinc_max = max(coinc_max, env_at(n0, min(t0 + 0.03, t + 0.03)))
        if coinc_max > 0 and e_self < coinc_max * COINC_BLEED_MAX:
            continue
        # (7) measured coupling from a coincident striker j
        coupled = False
        if coupling_table:
            attackers = [(n2, e2) for t2, n2, _e2j, _j2, _f2, e2 in cands
                         if n2 != name and abs(t2 - t) <= COINC_ATTACK_SEC]
            attackers += [(n0, env_at(n0, min(t0 + 0.03, t + 0.03))) for t0, n0 in existing
                          if n0 != name and n0 in tracks and abs(t0 - t) <= COINC_ATTACK_SEC]
            for n2, e2 in attackers:
                ratio = coupling_table.get(n2, {}).get(name)
                if ratio is not None and e_self <= ratio * e2 * COUPLING_SAFETY:
                    coupled = True
                    break
        if coupled:
            continue
        rescues.append(Rescue(round(t, 4), name, round(err, 3), round(jerk, 1), e_self))
    return rescues
