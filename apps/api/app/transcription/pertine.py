"""Per-tine phase-tracking rescue judge (#141 research line, S5 round 2).

Proposes re-strike onsets of already-ringing tines that broadband detection
discarded (the carryover-mask class): a tine's own ring hides the fresh
attack from broadband gates, but the strike breaks the tine's phase trend
and re-injects envelope energy, both visible in a per-tine heterodyne
demodulation. This is deliberately a *post-stage judge*: it never removes
or reorders broadband events, only proposes additions, and it is the
minimal-coupling integration form chosen to avoid kill-criteria C2 (no new
constants or passes in constants.py / events.py — see
docs/research/2026-07-per-tine-tracker-design.md §5).

Two output tiers (docs/research/pertine-tier-analysis.json, 97 labelled
rescues over 15 GT recordings + 111 completed fixtures):
- "event": isolated, clearly-ringing re-strikes -> become single-note events
  (the K2 quarry; offline round-2 projection adds +9 TP / 0 FP).
- "candidate": everything that passed the physical rejections but not the
  tier margins -> preserved as low-confidence candidate slots (#178).

Round 3 (#206) adds a second judge, `adjudicate_residual_slots`: a *veto*
over the residual-decay bulk rejection. Where the rescue path proposes
onsets broadband never asserted (and therefore needs the carryover
conditions pre-ring / re-injection), the veto adjudicates a segment
broadband DID assert and the bulk rejection dropped whole — it only asks
"which tine shows a fresh attack inside the dropped window", so pre-ring
and re-injection are deliberately waived while the bleed explaining-away
battery and the duplicate guard stay (probe: fires on 16/21 FN notes in
dropped segments, 1/18 on clean suppressions, and that one false fire sat
outside the segment window — docs/research/
2026-07-pertine-round3-residual-decay-replacement.md §3).

Signal chain (portable by design — heterodyne mix + 3rd-order butterworth
low-pass + hop decimation; the same kernel is expressible in the kalimba-dsp
Rust crate for the browser/WASM track):
  per tine: env/phase = |LPF(audio * e^{-j2pi f t})|, unwrap(angle(...))
  candidate: envelope jerk >= JERK_BAR and phase-trend RMS error >= PHASE_BAR
  state conditions: pre-ring (carryover class), energy re-injection
  explaining-away: measured partial bleed, quantified skirt, coincident
  attack bleed bound, measured per-pair coupling (attack window)

Calibration provenance: PHASE_BAR/JERK_BAR from the ROC sweep on the two
spectral-pinned Twinkle takes (docs/research/2026-07-phase-tracking-roc.md);
state/explaining-away constants from dual-run round 1; tier margins and
EXISTING_TOL from the labelled round-2 analysis. Tables are measured, not
believed (guardrail 2): docs/research/per-tine-partial-table.json and
per-tine-coupling-table.json, merged across performer groups per tuning
(the pipeline has no performer identity; per-pair max transfer is the
conservative merge). Missing tables simply disable those terms.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from ..models import InstrumentTuning

# --- detection core (shared calibration with the ROC probe) ---
HOP_SEC = 0.005
PAST_SEC = 0.10
FUT_SEC = 0.06
MIN_SEP_SEC = 0.06
ENV_GATE_FRAC = 0.02
SUSTAIN_FROM_SEC = 0.10
SUSTAIN_TO_SEC = 0.25
SUSTAIN_MIN_FRAC = 0.10
PHASE_BAR = 0.7
JERK_BAR = 150.0

# --- explaining-away (round 1) ---
COINCIDENT_SEC = 0.03
PARTIAL_CENTS = 50.0
PARTIAL_SAFETY = 3.0      # env_k <= relAmp * env_j * SAFETY -> explainable as bleed
SKIRT_CENTS = 300.0
SKIRT_ENV_RATIO = 20.0
# A candidate within this of an existing same-note event is a duplicate, not
# a rescue: recognizer event starts deviate up to ~0.1 s from spectral onsets
# (corpus-management.md), and duplicates at dt 0.088/0.101 each displaced the
# baseline prediction in greedy GT matching (net +1 FP, ebecf0c6 round 2).
EXISTING_TOL = 0.15
# (4) carryover-class restriction — rescue only a tine already ringing just
# before the candidate. A quiet tine's fresh strike is broadband's job.
PRE_RING_WIN = (-0.25, -0.05)
# (5) energy re-injection — a genuine re-strike pushes the envelope to at
# least its recent ring maximum; beating/vibrato stays below the decay peak.
REINJECT_FRAC = 1.0
REINJECT_LOOKBACK = 0.5
# (6) coincident-attack bleed bound — largest measured fresh-collision bleed
# was 0.352x the striker (A4->E6), so a candidate below COINC_BLEED_MAX of a
# coincident stronger attack is explainable as bleed without any table.
COINC_ATTACK_SEC = 0.05
COINC_BLEED_MAX = 0.4
# (7) measured per-pair coupling (attack-window p90 transfer, safety 2x).
COUPLING_SAFETY = 2.0

# --- tier rule (round 2, labelled-rescue analysis) ---
# pre-ring >= 1.5 on both branches: the mandate is re-strikes of *clearly*
# ringing tines; a ring barely above gate was the signature of the late
# phantoms (preRing 1.29/1.36 vs TP minimum 1.65). An isolated rescue (no
# coincident attack within ATTACKER_WIN) cannot be attack bleed or mount
# coupling; with coincident attacks the rescue must dominate the strongest
# attack (>= 1.5x) and clearly re-inject (>= 1.5x recent max).
ATTACKER_WIN = 0.08
TIER_MIN_PRE_RING = 1.5
TIER_MIN_ATTACKER_RATIO = 1.5
TIER_MIN_REINJECT = 1.5
# Same-note double-fire dedup: the per-tine NMS (60 ms) lets one physical
# re-strike fire twice at the window edge; keep the stronger jerk.
RESCUE_SAME_NOTE_SEP = 0.12

# Measured tables (research-line data assets; relocate on merge).
_RESEARCH_DIR = Path(__file__).resolve().parents[4] / "docs" / "research"
PARTIAL_TABLE_PATH = _RESEARCH_DIR / "per-tine-partial-table.json"
COUPLING_TABLE_PATH = _RESEARCH_DIR / "per-tine-coupling-table.json"
_TABLE_CACHE: dict[str, tuple[dict, dict]] = {}


@dataclass
class Rescue:
    time: float
    note: str
    phase_err: float
    jerk: float
    env: float
    pre_ring_ratio: float = 0.0    # median pre-window env / abs gate
    reinject_ratio: float = 0.0    # attack peak / recent ring max
    attackers: list[dict] | None = None


def load_tables(tuning_id: str) -> tuple[dict, dict]:
    """(partial_table, coupling_table) for a tuning, merged across performer
    groups (per-pair max transfer, union of partials); ({}, {}) if absent."""
    cached = _TABLE_CACHE.get(tuning_id)
    if cached is not None:
        return cached
    partial: dict[str, list[tuple[float, float]]] = {}
    coupling: dict[str, dict[str, float]] = {}
    try:
        if PARTIAL_TABLE_PATH.is_file():
            data = json.loads(PARTIAL_TABLE_PATH.read_text())
            for group, g in (data.get("groups") or {}).items():
                if not group.startswith(tuning_id + "|"):
                    continue
                for note, entry in (g.get("notes") or {}).items():
                    # weak clusters flagged relaxedOnly are not load-bearing
                    parts = [(p["ratio"], p["medianRelAmp"])
                             for p in entry.get("partials", []) if not p.get("relaxedOnly")]
                    if parts:
                        partial.setdefault(note, []).extend(parts)
        if COUPLING_TABLE_PATH.is_file():
            data = json.loads(COUPLING_TABLE_PATH.read_text())
            for group, g in (data.get("groups") or {}).items():
                if not group.startswith(tuning_id + "|"):
                    continue
                for j, d in g.items():
                    tgt = coupling.setdefault(j, {})
                    for k, v in d.items():
                        # attack-window transfer bounds tracker candidates
                        # (they fire on attacks); body p90 is the fallback.
                        tgt[k] = max(tgt.get(k, 0.0), v.get("attackP90") or v["p90"])
    except (OSError, ValueError, KeyError, TypeError):
        partial, coupling = {}, {}
    _TABLE_CACHE[tuning_id] = (partial, coupling)
    return partial, coupling


def _demodulate(audio: np.ndarray, sr: int, freq: float, hop: int,
                t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = audio * np.exp(-2j * np.pi * freq * t)
    bw = max(freq * 0.025, 12.0)
    sos = butter(3, bw, btype="low", fs=sr, output="sos")
    zf = sosfilt(sos, z.real) + 1j * sosfilt(sos, z.imag)
    zh = zf[::hop]
    return np.abs(zh), np.unwrap(np.angle(zh))


def _build_tracks(
    audio: np.ndarray, sr: int, tuning: list[tuple[str, float]],
) -> tuple[dict[str, tuple[float, np.ndarray, np.ndarray]], float, float] | None:
    """Demodulate every tine once: (tracks, hop_sec, abs_gate); None if unusable."""
    hop = int(sr * HOP_SEC)
    if hop <= 0 or len(audio) < sr // 2:
        return None
    hop_sec = hop / sr
    t_axis = np.arange(len(audio)) / sr
    tracks: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name, freq in tuning:
        if name in tracks or freq <= 0:
            continue
        env, phase = _demodulate(audio, sr, freq, hop, t_axis)
        tracks[name] = (freq, env, phase)
    if not tracks:
        return None
    abs_gate = (max(tr[1].max() for tr in tracks.values()) or 1.0) * ENV_GATE_FRAC
    return tracks, hop_sec, abs_gate


def _core_candidates(tracks, hop_sec, abs_gate):
    """Detection-core hits across all tines: (t, note, err, jerk, freq, env)."""
    cands = []
    for name, (freq, env, phase) in tracks.items():
        hits = _candidates_for_track(env, phase, hop_sec, abs_gate, JERK_BAR)
        sel = [(i, e, j) for i, e, j in hits if e >= PHASE_BAR]
        for i, e, j in _nms(sel, hop_sec):
            cands.append((i * hop_sec, name, e, j, freq, float(env[i])))
    cands.sort()
    return cands


def _env_at(tracks, hop_sec, note: str, t: float) -> float:
    _f, env, _p = tracks[note]
    i = min(len(env) - 1, max(0, int(t / hop_sec)))
    return float(env[i])


def _env_window_stat(tracks, hop_sec, note: str, t0: float, t1: float, fn) -> float:
    _f, env, _p = tracks[note]
    a = max(0, int(t0 / hop_sec)); b = min(len(env), max(a + 1, int(t1 / hop_sec)))
    return float(fn(env[a:b]))


def _bleed_explained(t, name, freq, e_self, cands, existing, tracks, freqs,
                     hop_sec, partial_table, coupling_table) -> bool:
    """Explaining-away battery (2)(3)(6)(7): is e_self at (t, name) explainable
    as bleed from another tine (measured partial / filter skirt / coincident
    attack bound / measured coupling)? Shared by the rescue and veto paths."""
    # (2) measured-partial bleed: candidate k explainable as j's partial?
    if partial_table:
        for t2, n2, _e2, _j2, f2, e2 in cands:
            if n2 == name or abs(t2 - t) > COINCIDENT_SEC:
                continue
            for ratio, rel_amp in partial_table.get(n2, []):
                pf = f2 * ratio
                if abs(1200 * np.log2(freq / pf)) <= PARTIAL_CENTS \
                        and e_self <= rel_amp * e2 * PARTIAL_SAFETY:
                    return True
    # (3) quantified skirt bleed from a near-frequency neighbour
    for n2, f2 in freqs.items():
        if n2 == name:
            continue
        if abs(1200 * np.log2(f2 / freq)) <= SKIRT_CENTS \
                and _env_at(tracks, hop_sec, n2, t) >= e_self * SKIRT_ENV_RATIO:
            return True
    # (6) coincident-attack bleed bound
    coinc_max = 0.0
    for t2, n2, _e2, _j2, _f2, e2 in cands:
        if n2 != name and abs(t2 - t) <= COINC_ATTACK_SEC:
            coinc_max = max(coinc_max, e2)
    for t0, n0 in existing:
        if n0 != name and n0 in tracks and abs(t0 - t) <= COINC_ATTACK_SEC:
            coinc_max = max(coinc_max, _env_at(tracks, hop_sec, n0, min(t0 + 0.03, t + 0.03)))
    if coinc_max > 0 and e_self < coinc_max * COINC_BLEED_MAX:
        return True
    # (7) measured coupling from a coincident striker j
    if coupling_table:
        attackers = [(n2, e2) for t2, n2, _e2j, _j2, _f2, e2 in cands
                     if n2 != name and abs(t2 - t) <= COINC_ATTACK_SEC]
        attackers += [(n0, _env_at(tracks, hop_sec, n0, min(t0 + 0.03, t + 0.03)))
                      for t0, n0 in existing
                      if n0 != name and n0 in tracks and abs(t0 - t) <= COINC_ATTACK_SEC]
        for n2, e2 in attackers:
            ratio = coupling_table.get(n2, {}).get(name)
            if ratio is not None and e_self <= ratio * e2 * COUPLING_SAFETY:
                return True
    return False


def _attacker_context(t, name, freq, e_self, cands, existing, tracks, freqs,
                      hop_sec) -> list[dict]:
    """Every coincident attack within +-ATTACKER_WIN (tier context)."""
    ctx = []
    for t2, n2, _e2j, _j2, f2, e2 in cands:
        if n2 != name and abs(t2 - t) <= ATTACKER_WIN:
            ctx.append({
                "note": n2, "dt": round(t2 - t, 4), "kind": "candidate",
                "cents": round(1200 * float(np.log2(f2 / freq)), 1),
                "envRatio": round(e_self / e2, 3) if e2 > 0 else None,
            })
    for t0, n0 in existing:
        if n0 != name and n0 in tracks and abs(t0 - t) <= ATTACKER_WIN:
            e0 = _env_at(tracks, hop_sec, n0, min(t0 + 0.03, t + 0.03))
            ctx.append({
                "note": n0, "dt": round(t0 - t, 4), "kind": "event",
                "cents": round(1200 * float(np.log2(freqs[n0] / freq)), 1),
                "envRatio": round(e_self / e0, 3) if e0 > 0 else None,
            })
    return ctx


def _candidates_for_track(env, phase, hop_sec, abs_gate, jerk_floor):
    """All hops passing the fixed gates + jerk floor, with features."""
    n = len(env)
    past = int(PAST_SEC / hop_sec)
    fut = int(FUT_SEC / hop_sec)
    s_from = int(SUSTAIN_FROM_SEC / hop_sec)
    s_to = int(SUSTAIN_TO_SEC / hop_sec)
    tt = np.arange(n) * hop_sec
    out = []
    for i in range(past + 2, n - s_to - 1):
        seg = env[i - 2:i + fut]
        denv = np.diff(seg) / hop_sec
        ref = np.median(env[i - past:i - 2]) + 1e-9
        jerk = float(denv.max() / ref) if len(denv) else 0.0
        if jerk < jerk_floor:
            continue
        attack_peak = env[i:i + fut].max()
        if attack_peak < abs_gate:
            continue
        sustain = np.median(env[i + s_from:i + s_to])
        if sustain < attack_peak * SUSTAIN_MIN_FRAC:
            continue
        p_t = tt[i - past:i - 2]
        p_y = phase[i - past:i - 2]
        coef = np.polyfit(p_t, p_y, 1)
        pred = np.polyval(coef, tt[i + 1:i + fut])
        err = float(np.sqrt(np.mean((phase[i + 1:i + fut] - pred) ** 2)))
        out.append((i, err, jerk))
    return out


def _nms(hits, hop_sec):
    if not hits:
        return []
    min_sep = int(MIN_SEP_SEC / hop_sec)
    hits = sorted(hits, key=lambda h: -h[2])
    kept, taken = [], []
    for h in hits:
        if all(abs(h[0] - t) >= min_sep for t in taken):
            kept.append(h)
            taken.append(h[0])
    return sorted(kept)


def _dedup_same_note(rescues: list[Rescue]) -> list[Rescue]:
    kept: list[Rescue] = []
    for r in sorted(rescues, key=lambda r: -r.jerk):
        if any(k.note == r.note and abs(k.time - r.time) <= RESCUE_SAME_NOTE_SEP
               for k in kept):
            continue
        kept.append(r)
    return sorted(kept, key=lambda r: r.time)


def _attacker_tier(rescue: Rescue) -> str:
    """Dominance margins vs coincident attackers (shared tier core).

    An isolated rescue (no coincident attack within ATTACKER_WIN) cannot be
    attack bleed or mount coupling; with coincident attacks the rescue must
    dominate the strongest attack (>= 1.5x) and clearly re-inject (>= 1.5x
    recent max)."""
    attackers = rescue.attackers or []
    if not attackers:
        return "event"
    ratios = [a["envRatio"] for a in attackers if a["envRatio"] is not None]
    if not ratios:  # attackers present but unmeasurable -> conservative
        return "candidate"
    if min(ratios) < TIER_MIN_ATTACKER_RATIO:
        return "candidate"
    if rescue.reinject_ratio < TIER_MIN_REINJECT:
        return "candidate"
    return "event"


def tier_of(rescue: Rescue) -> str:
    """"event" (strong) or "candidate" (weak) — see TIER_* rationale."""
    if rescue.pre_ring_ratio < TIER_MIN_PRE_RING:
        return "candidate"
    return _attacker_tier(rescue)


def veto_tier(rescue: Rescue) -> str:
    """Tier rule for residual-autopsy fires (#206 round 3).

    Pre-ring is waived — the #206 class is a masked tine that was NOT
    ringing — but the round-2 dominance margins stay: every false fire
    remaining after the re-injection guard (fixtures 2026-07-06: bwv147 G5
    envRatio 1.21 vs F5 / triad-02 G4+C4 mutual pair / c4-to-e6 D5 under
    E5+F5 strikes / ebecf0c6 C6 reinject 1.27) is a quiet tine firing
    coincident with a stronger strike whose bleed the measured tables fail
    to explain, exactly the regime the round-2 tier margins were labelled
    on. Demoted fires surface as low-confidence candidate slots (#178)."""
    return _attacker_tier(rescue)


def track_and_rescue(
    audio: np.ndarray,
    sr: int,
    tuning: list[tuple[str, float]],
    existing: list[tuple[float, str]],
    partial_table: dict[str, list[tuple[float, float]]] | None = None,
    coupling_table: dict[str, dict[str, float]] | None = None,
) -> list[Rescue]:
    """Propose rescue onsets the existing event list does not cover."""
    built = _build_tracks(audio, sr, tuning)
    if built is None:
        return []
    tracks, hop_sec, abs_gate = built
    cands = _core_candidates(tracks, hop_sec, abs_gate)
    freqs = {n: f for n, f in tuning}

    rescues: list[Rescue] = []
    for t, name, err, jerk, freq, e_self in cands:
        # already covered by an existing event of the same note?
        if any(n == name and abs(t0 - t) <= EXISTING_TOL for t0, n in existing):
            continue
        # (4) carryover-class restriction: the tine must already be ringing
        pre_ring = _env_window_stat(tracks, hop_sec, name,
                                    t + PRE_RING_WIN[0], t + PRE_RING_WIN[1], np.median)
        if pre_ring < abs_gate:
            continue
        # (5) energy re-injection: attack peak reaches its recent ring maximum
        attack_peak = _env_window_stat(tracks, hop_sec, name, t, t + 0.06, np.max)
        recent_max = _env_window_stat(tracks, hop_sec, name,
                                      t - REINJECT_LOOKBACK, t - 0.01, np.max)
        if attack_peak < recent_max * REINJECT_FRAC:
            continue
        # (2)(3)(6)(7) bleed explaining-away battery (shared with the veto path)
        if _bleed_explained(t, name, freq, e_self, cands, existing, tracks,
                            freqs, hop_sec, partial_table, coupling_table):
            continue
        rescues.append(Rescue(
            round(t, 4), name, round(err, 3), round(jerk, 1), e_self,
            pre_ring_ratio=round(pre_ring / abs_gate, 2),
            reinject_ratio=round(attack_peak / max(recent_max, 1e-12), 2),
            attackers=_attacker_context(t, name, freq, e_self, cands, existing,
                                        tracks, freqs, hop_sec),
        ))
    return _dedup_same_note(rescues)


def adjudicate_residual_slots(
    audio: np.ndarray,
    sr: int,
    tuning: list[tuple[str, float]],
    windows: list[tuple[float, float]],
    existing: list[tuple[float, str]],
    partial_table: dict[str, list[tuple[float, float]]] | None = None,
    coupling_table: dict[str, dict[str, float]] | None = None,
) -> list[Rescue]:
    """Veto judge over residual-decay bulk rejections (#206, round 3).

    `windows` are the [start, end] spans of segments dropped with reason
    residual-decay-no-reattack. Broadband already asserted an onset inside
    each window (the segment existed), so unlike track_and_rescue this
    judge waives the carryover conditions (pre-ring, re-injection) and only
    asks which tine shows a fresh attack *strictly inside* the window.
    Strict membership is the window-semantics calibration chosen from the
    round-3 design's options (§3): the probe's single false veto sat
    +0.029 s past its segment end, and a segment's evidence ends at its
    boundary — the next onset belongs to the next segment. Cost measured on
    the probe: one boundary recovery at +0.017 s (fires 16 -> 15 of 21 FN
    notes). Bleed explaining-away and the duplicate guard are shared with
    the rescue path.

    Condition (5) re-injection is RETAINED, unlike the probe's raw bars
    (implementation-round calibration, 2026-07-06): without it, chord-repeat
    fixtures over-promote non-played tines wholesale — a4-d4-f4-triad-01
    alone gained 16 events, all on unplayed C4/E4, every one with reinject
    0.32-0.86 — because simultaneous ringing tines beat against each other
    and the beating wobble passes the phase/jerk bars. Re-injection is the
    calibrated guard for exactly that regime ("beating stays below the decay
    peak", round 1), and a genuinely masked fresh strike passes it easily
    (#206 F5: reinject 8.84 / 2.48; a quiet tine's recent max is near zero).
    Probe-side cost: fires 16 -> 10, recoverable slots 12 -> 6 — but the
    C5-class losses are the carryover-mask quarry the round-2 rescue path
    already recovers in the integrated pipeline, so the marginal loss is
    measured (not assumed) by the 2x2 ablation's autopsy arms. Pre-ring
    stays waived: the #206 class is a masked tine that was NOT ringing.
    """
    if not windows:
        return []
    built = _build_tracks(audio, sr, tuning)
    if built is None:
        return []
    tracks, hop_sec, abs_gate = built
    cands = _core_candidates(tracks, hop_sec, abs_gate)
    freqs = {n: f for n, f in tuning}
    rescues: list[Rescue] = []
    for t, name, err, jerk, freq, e_self in cands:
        win = next(((w0, w1) for w0, w1 in windows if w0 <= t <= w1), None)
        if win is None:
            continue
        if any(n == name and abs(t0 - t) <= EXISTING_TOL for t0, n in existing):
            continue
        # A same-note event at the window's edge means the dropped segment
        # abuts that note's own recognized segment — the fire inside the
        # window is the same attack seen through the segment split, and
        # promoting it double-emits the note (c4-to-e6 fixture: D5 fired at
        # 4.535 inside the dropped C5-residual window [4.432, 4.859] whose
        # end IS the recognized D5 event's start).
        if any(n == name and min(abs(t0 - win[0]), abs(t0 - win[1])) <= EXISTING_TOL
               for t0, n in existing):
            continue
        # A recognized event inside the attacker window means broadband saw
        # this instant and the full scoring stack already adjudicated its
        # note set — the veto's mandate is instants the bulk rejection
        # silenced entirely, not a re-score of decided attacks (17-c bwv147:
        # a quiet-tine D5 was promoted inside a recognized chord attack that
        # the chord selector had already rejected it from).
        if any(abs(t0 - t) <= ATTACKER_WIN for t0, _n0 in existing):
            continue
        if _bleed_explained(t, name, freq, e_self, cands, existing, tracks,
                            freqs, hop_sec, partial_table, coupling_table):
            continue
        # (5) energy re-injection — the beating guard (see docstring).
        attack_peak = _env_window_stat(tracks, hop_sec, name, t, t + 0.06, np.max)
        recent_max = _env_window_stat(tracks, hop_sec, name,
                                      t - REINJECT_LOOKBACK, t - 0.01, np.max)
        if attack_peak < recent_max * REINJECT_FRAC:
            continue
        pre_ring = _env_window_stat(tracks, hop_sec, name,
                                    t + PRE_RING_WIN[0], t + PRE_RING_WIN[1], np.median)
        rescues.append(Rescue(
            round(t, 4), name, round(err, 3), round(jerk, 1), e_self,
            pre_ring_ratio=round(pre_ring / abs_gate, 2),
            reinject_ratio=round(attack_peak / max(recent_max, 1e-12), 2),
            attackers=_attacker_context(t, name, freq, e_self, cands, existing,
                                        tracks, freqs, hop_sec),
        ))
    return _dedup_same_note(rescues)


def propose_rescues(
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    existing: list[tuple[float, str]],
) -> tuple[list[Rescue], list[Rescue]]:
    """(strong, weak) rescue proposals for the pipeline integration point."""
    partial_table, coupling_table = load_tables(tuning.id)
    notes = [(n.note_name, float(n.frequency)) for n in tuning.notes]
    rescues = track_and_rescue(
        np.asarray(audio, dtype=np.float64), sample_rate, notes, existing,
        partial_table=partial_table, coupling_table=coupling_table,
    )
    strong = [r for r in rescues if tier_of(r) == "event"]
    weak = [r for r in rescues if tier_of(r) == "candidate"]
    return strong, weak


def build_residual_oracle(
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
):
    """Fresh-attack oracle for the in-stage residual-decay decision
    (#206 round-4 design, kill-count round 3).

    Returns ``oracle(t0, t1, existing) -> bool`` — True iff some tine shows
    a fresh attack strictly inside [t0, t1]: detection core (phase/jerk
    bars, NMS) + re-injection (>= REINJECT_FRAC of the recent ring max) +
    the bleed explaining-away battery. Same semantics as the round-4 full
    dump (pertine_round4_oracle_dump.py; recovery 9/15 fnOverlap slots,
    false fires 2/18 clean slots — gate <=3/18 passed 2026-07-06, C2
    approved). No thresholds beyond pertine.py's existing calibrated
    constants; ``existing`` supplies coincident-event context for
    conditions (6)/(7) and is bound per segment by the pipeline.

    Returns None when the audio is unusable (caller falls back to the
    mute-dip condition).
    """
    built = _build_tracks(np.asarray(audio, dtype=np.float64), sample_rate,
                          [(n.note_name, float(n.frequency)) for n in tuning.notes])
    if built is None:
        return None
    tracks, hop_sec, abs_gate = built
    cands = _core_candidates(tracks, hop_sec, abs_gate)
    freqs = {n.note_name: float(n.frequency) for n in tuning.notes}
    partial_table, coupling_table = load_tables(tuning.id)

    def oracle(t0: float, t1: float,
               existing: list[tuple[float, str]] = ()) -> str | None:
        """Strongest-fresh-tine note name inside [t0, t1], or None.

        Option-i wiring (C2 update 2026-07-06): the caller promotes the
        returned tine via the existing forward-scan promotion plumbing —
        keeping a saved segment as kept-residual measured as harmful in
        rounds 3-4 (timing drift / structural disruption), so the fresh
        note itself must take the segment over.
        """
        best_note, best_jerk, best_t = None, -1.0, 0.0
        for t, name, _err, jerk, freq, e_self in cands:
            if not (t0 <= t <= t1):
                continue
            attack_peak = _env_window_stat(tracks, hop_sec, name, t, t + 0.06, np.max)
            recent_max = _env_window_stat(tracks, hop_sec, name,
                                          t - REINJECT_LOOKBACK, t - 0.01, np.max)
            if attack_peak < recent_max * REINJECT_FRAC:
                continue
            if _bleed_explained(t, name, freq, e_self, cands, existing, tracks,
                                freqs, hop_sec, partial_table, coupling_table):
                continue
            if jerk > best_jerk:
                best_note, best_jerk, best_t = name, jerk, t
        if best_note is None:
            return None
        # Parent-alias redirect (#149-class collision, measured on ebecf0c6:
        # an F5 strike pumps its quint partial at C6, C6 wins the jerk race
        # while F5's own track misses the phase bars). If the winner sits on
        # a measured partial of another tine whose envelope also re-injects
        # at the same instant, the energy belongs to the parent's strike —
        # redirect to the parent. Measured tables + existing constants only.
        bf = freqs[best_note]
        for parent, parts in (partial_table or {}).items():
            if parent == best_note or parent not in tracks or parent not in freqs:
                continue
            pf = freqs[parent]
            for ratio, _amp in parts:
                if abs(1200 * np.log2(bf / (pf * ratio))) <= PARTIAL_CENTS:
                    a_pk = _env_window_stat(tracks, hop_sec, parent,
                                            best_t, best_t + 0.06, np.max)
                    r_mx = _env_window_stat(tracks, hop_sec, parent,
                                            best_t - REINJECT_LOOKBACK,
                                            best_t - 0.01, np.max)
                    if a_pk >= r_mx * REINJECT_FRAC:
                        return parent
                    break
        return best_note

    return oracle


def propose_residual_autopsy(
    audio: np.ndarray,
    sample_rate: int,
    tuning: InstrumentTuning,
    windows: list[tuple[float, float]],
    existing: list[tuple[float, str]],
) -> list[Rescue]:
    """Adjudicated promotions for residual-decay dropped slots (pipeline hook)."""
    partial_table, coupling_table = load_tables(tuning.id)
    notes = [(n.note_name, float(n.frequency)) for n in tuning.notes]
    return adjudicate_residual_slots(
        np.asarray(audio, dtype=np.float64), sample_rate, notes, windows, existing,
        partial_table=partial_table, coupling_table=coupling_table,
    )
