"""Round-4 design probe (#206 in-stage replacement): oracle full-count dump.

Analysis only — no recognizer changes. Implements step 1 of the verification
protocol in docs/research/2026-07-pertine-round4-instage-replacement.md §5:
"現行認識の residual-decay 棄却 33 slot (probe A) に対し oracle 判定を照合 —
FN 重なり slot の回収可能数 / clean suppression の誤発火数を実装前ベース
ラインとして記録". The measured numbers double as the §6 kill-gate check
("clean suppression 誤発火 > 3/18 で設計見直し").

The "oracle" here is the round-4 design's proposed replacement judge (§3):
detection core (phase RMS error >= PHASE_BAR and jerk >= JERK_BAR, both
already applied inside pertine._core_candidates, plus its per-tine NMS) +
energy re-injection (>= REINJECT_FRAC) + the shared bleed explaining-away
battery (_bleed_explained) — with the round-3 veto's *existing-event*
guards (duplicate-of-existing, edge-abutting, attacker-window-has-existing-
event) deliberately REMOVED, per the round-4 design's stated replacement
target. A candidate must fall strictly inside the dropped segment's own
[startTime, endTime] window (no pad) to count; a wider +-0.10 s pad is used
only to classify a slot as "fnOverlap" (a GT false negative sits near the
slot) vs "clean" (no nearby FN) — this matches probe A's classification so
the two probes' slot counts are directly comparable.

For an fnOverlap slot, an oracle fire is recall recovered (good) — the
design's whole point is a masked tine's fresh attack surfacing again. For a
clean slot, any oracle fire is a false alarm (bad) — the kill-gate
numerator. Fire-detail rows also record whether the firing tine matches one
of the slot's nearby FN notes (a stricter signal: did the oracle wake up
the *right* tine, not just some tine in the window).

Usage: uv run python scripts/audio-analysis/research/pertine_round4_oracle_dump.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

import note_f1_benchmark as nfb  # noqa: E402
from pertine_dualrun import gt_tx_ids  # noqa: E402
from tine_partial_collision_probe import audio_for, load_audio, request_for  # noqa: E402

from app.transcription.pertine import (  # noqa: E402
    JERK_BAR,
    PHASE_BAR,
    REINJECT_FRAC,
    REINJECT_LOOKBACK,
    _bleed_explained,
    _build_tracks,
    _core_candidates,
    _env_window_stat,
    load_tables,
)

OUT = REPO / "docs" / "research" / "pertine-round4-oracle-dump.json"
DROP_REASON = "residual-decay-no-reattack"
# Pad used only to classify a slot as fnOverlap vs clean (matches probe A's
# WIN_PAD in pertine_round3_probe.py so slot counts are comparable). Oracle
# candidate membership itself uses the slot's own [startTime, endTime]
# strictly, no pad (round-4 design §3 window-semantics choice).
FN_WIN_PAD = 0.10
CLEAN_MISFIRE_KILL_THRESHOLD = 3


def _note_name(note: dict) -> str:
    return f"{note['pitchClass']}{note['octave']}"


def oracle_fires_in_window(cands, t0, t1, tracks, hop_sec, existing, freqs,
                            partial_table, coupling_table):
    """Core candidates strictly inside [t0, t1] passing reinject + bleed
    explaining-away (round-4 oracle; no existing-event guards)."""
    fires = []
    for t, name, err, jerk, freq, e_self in cands:
        if not (t0 <= t <= t1):
            continue
        attack_peak = _env_window_stat(tracks, hop_sec, name, t, t + 0.06, np.max)
        recent_max = _env_window_stat(
            tracks, hop_sec, name, t - REINJECT_LOOKBACK, t - 0.01, np.max)
        reinject = attack_peak / max(recent_max, 1e-12)
        if reinject < REINJECT_FRAC:
            continue
        if _bleed_explained(t, name, freq, e_self, cands, existing, tracks, freqs,
                            hop_sec, partial_table, coupling_table):
            continue
        fires.append({
            "note": name, "t": round(t, 3), "phaseErr": round(err, 3),
            "jerk": round(jerk, 1), "reinject": round(reinject, 2),
        })
    return fires


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.transcription import settings as recognizer_settings
    client = TestClient(app)

    recordings_out = []
    fire_rows = []
    tot = {
        "slots": 0,
        "fnOverlapSlots": 0, "fnOverlapOracleFires": 0, "fnOverlapOracleFiresMatchFn": 0,
        "cleanSlots": 0, "cleanOracleFires": 0,
    }

    for tx in gt_tx_ids():
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        with recognizer_settings.override(use_pertine_tracker_rescue=False):
            payload = nfb.transcribe_payload(client, tx, debug=True)
        match = nfb.match_pairs(truth, nfb.collect_one_best(payload))
        fns = match["falseNegatives"]
        slots = [s for s in payload.get("candidateSlots") or []
                 if s.get("dropReason") == DROP_REASON]
        if not slots:
            continue

        ap = audio_for(tx)
        req = request_for(tx)
        if ap is None or req is None:
            print(f"{tx[:8]}: SKIP (missing audio/request)")
            continue
        audio, sr = load_audio(ap)
        tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
        freqs = {n: f for n, f in tuning}
        partial_table, coupling_table = load_tables(req["tuning"]["id"])

        built = _build_tracks(np.asarray(audio, dtype=np.float64), sr, tuning)
        if built is None:
            print(f"{tx[:8]}: SKIP (tracks unusable)")
            continue
        tracks, hop_sec, abs_gate = built
        cands = _core_candidates(tracks, hop_sec, abs_gate)

        existing = [(float(e["startTimeSec"]), _note_name(n))
                    for e in payload["events"] for n in e["notes"]]

        slot_rows = []
        for s in slots:
            t0, t1 = float(s["startTime"]), float(s["endTime"])
            fns_in = [fn for fn in fns if t0 - FN_WIN_PAD <= fn["time"] <= t1 + FN_WIN_PAD]
            fn_notes = {fn["note"] for fn in fns_in}
            primary = s["primaryNote"]
            pname = f"{primary['pitchClass']}{primary['octave']}"

            fires = oracle_fires_in_window(cands, t0, t1, tracks, hop_sec, existing,
                                           freqs, partial_table, coupling_table)
            matches_fn = any(f["note"] in fn_notes for f in fires)

            row = {
                "win": [round(t0, 3), round(t1, 3)],
                "droppedPrimary": pname,
                "fnNotesInWindow": [{"t": round(fn["time"], 3), "note": fn["note"]} for fn in fns_in],
                "oracleFires": bool(fires),
                "firingCandidates": fires,
                "matchesFn": matches_fn,
            }
            slot_rows.append(row)
            tot["slots"] += 1
            if fns_in:
                tot["fnOverlapSlots"] += 1
                if fires:
                    tot["fnOverlapOracleFires"] += 1
                if matches_fn:
                    tot["fnOverlapOracleFiresMatchFn"] += 1
            else:
                tot["cleanSlots"] += 1
                if fires:
                    tot["cleanOracleFires"] += 1

            for f in fires:
                fire_rows.append({
                    "tx": tx[:8], "win": row["win"], "note": f["note"], "t": f["t"],
                    "phaseErr": f["phaseErr"], "jerk": f["jerk"], "reinject": f["reinject"],
                    "classification": "fnOverlap" if fns_in else "clean",
                    "matchesFn": f["note"] in fn_notes,
                })

        recordings_out.append({"tx": tx[:8], "slots": slot_rows})
        n_fn = sum(1 for r in slot_rows if r["fnNotesInWindow"])
        n_fires = sum(1 for r in slot_rows if r["oracleFires"])
        print(f"{tx[:8]}: residual-decay slots={len(slots)} fnOverlap={n_fn} oracleFires={n_fires}")

    print("\n=== totals ===")
    print(json.dumps(tot, indent=1))
    gate_triggered = tot["cleanOracleFires"] > CLEAN_MISFIRE_KILL_THRESHOLD
    print(
        f"\nkill-gate (clean misfires > {CLEAN_MISFIRE_KILL_THRESHOLD}/{tot['cleanSlots']}): "
        f"{'TRIGGERED -- design review required' if gate_triggered else 'not triggered'} "
        f"({tot['cleanOracleFires']}/{tot['cleanSlots']})"
    )
    print(
        f"fnOverlap recovery: {tot['fnOverlapOracleFires']}/{tot['fnOverlapSlots']} slots fire "
        f"({tot['fnOverlapOracleFiresMatchFn']}/{tot['fnOverlapSlots']} fire on the missing note itself)"
    )

    print("\n=== firing detail rows ===")
    for r in fire_rows:
        print(
            f"  {r['tx']} win={r['win']} {r['classification']:9s} note={r['note']:4s} "
            f"t={r['t']:.3f} phase={r['phaseErr']:.2f} jerk={r['jerk']:.1f} "
            f"reinject={r['reinject']:.2f} matchesFn={r['matchesFn']}"
        )

    OUT.write_text(json.dumps({
        "dropReason": DROP_REASON,
        "coreBars": {"phase": PHASE_BAR, "jerk": JERK_BAR},
        "reinjectFrac": REINJECT_FRAC, "reinjectLookback": REINJECT_LOOKBACK,
        "fnClassificationWinPad": FN_WIN_PAD,
        "totals": tot,
        "killGate": {
            "cleanMisfireThreshold": CLEAN_MISFIRE_KILL_THRESHOLD,
            "triggered": gate_triggered,
            "cleanOracleFires": tot["cleanOracleFires"],
            "cleanSlots": tot["cleanSlots"],
        },
        "recordings": recordings_out,
        "fireRows": fire_rows,
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
