"""Tier-design analysis for per-tine tracker rescues (S5 round 2, #141).

Round 1 measured the rescue pool: +16 TP / +16 FP on the non-saturated GT
set, +30 FP on the two saturated magnetic-pickup recordings, and 35
would-be exact-match additions on 6 completed fixtures. Round 2 integrates
with a tier: strong rescues become events (count toward K2/K3), weak ones
become low-confidence candidate slots (#178 infrastructure, no fixture or
K4 impact). This script labels every rescue TP/FP against GT (fixture
additions are presumptive FP) and dumps per-rescue features so the tier
rule is chosen from data, per the onset-classification methodology
(separate fake types, no mixed analysis).

Usage: uv run python scripts/audio-analysis/research/pertine_tier_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

import numpy as np  # noqa: E402

import note_f1_benchmark as nfb  # noqa: E402
from pertine_tracker import load_coupling_table, load_partial_table, track_and_rescue  # noqa: E402
from tine_partial_collision_probe import audio_for, load_audio, request_for, resolve_source  # noqa: E402
from pertine_dualrun import gt_tx_ids, K2_RECORDINGS, MANUAL_CAPTURES  # noqa: E402

OUT = REPO / "docs" / "research" / "pertine-tier-analysis.json"


def rescue_row(r, label: str, tx: str, group: str) -> dict:
    """Flatten a Rescue + label into an analysis row."""
    atk = r.attackers or []
    # strongest coincident attack = smallest envRatio (self env / attacker env)
    strongest = min((a for a in atk if a["envRatio"] is not None),
                    key=lambda a: a["envRatio"], default=None)
    # nearest-in-pitch attacker among those at least comparable in level
    semitone = [a for a in atk if abs(a["cents"]) <= 130 and abs(a["dt"]) <= 0.05]
    sem_strong = min((a for a in semitone if a["envRatio"] is not None),
                     key=lambda a: a["envRatio"], default=None)
    return {
        "tx": tx, "group": group, "label": label,
        "t": r.time, "note": r.note,
        "phaseErr": r.phase_err, "jerk": r.jerk, "env": r.env,
        "preRingRatio": r.pre_ring_ratio, "reinjectRatio": r.reinject_ratio,
        "nAttackers": len(atk),
        "strongestEnvRatio": strongest["envRatio"] if strongest else None,
        "strongestCents": strongest["cents"] if strongest else None,
        "semitoneCoincident": bool(semitone),
        "semitoneEnvRatio": sem_strong["envRatio"] if sem_strong else None,
        "attackers": atk,
    }


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    rows: list[dict] = []

    for tx in gt_tx_ids():
        payload = nfb.transcribe_payload(client, tx, debug=False)
        baseline = nfb.collect_one_best(payload)
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        src = resolve_source(tx)
        ap = audio_for(tx)
        if src is None or ap is None:
            continue
        _tuning_id, group, _freqs = src
        audio, sr = load_audio(ap)
        req = request_for(tx)
        tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
        rescues = track_and_rescue(
            audio, sr, tuning,
            existing=[(p["time"], p["note"]) for p in baseline],
            partial_table=load_partial_table(group),
            coupling_table=load_coupling_table(group),
        )
        rescue_dicts = [{"time": r.time, "note": r.note} for r in rescues]
        ma = nfb.match_pairs(truth, baseline + rescue_dicts)
        fp_ids = {id(p) for p in ma["falsePositives"]}
        for r, rd in zip(rescues, rescue_dicts):
            rows.append(rescue_row(r, "fp" if id(rd) in fp_ids else "tp", tx[:8], group))
        print(f"{tx[:8]} [{group}] rescues={len(rescues)}")

    for d in sorted(MANUAL_CAPTURES.iterdir()):
        wav, reqp = d / "audio.wav", d / "request.json"
        if not (wav.is_file() and reqp.is_file()):
            continue
        try:
            req = json.loads(reqp.read_text())
            import wave as wavmod
            with wavmod.open(str(wav), "rb") as w:
                sr = w.getframerate(); raw = w.readframes(w.getnframes())
                ch = w.getnchannels(); width = w.getsampwidth()
            dt = {2: np.int16, 4: np.int32}[width]
            a = np.frombuffer(raw, dtype=dt).astype(np.float64)
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            a = a / (np.max(np.abs(a)) or 1.0)
            resp = client.post("/api/transcriptions", data={
                "tuning": json.dumps(req["tuning"]), "debug": "false",
                "dryRun": "true", "force": "true"},
                files={"file": ("audio.wav", wav.read_bytes(), "audio/wav")})
            payload = resp.json()
            existing = [(float(e["startTimeSec"]), nfb._note_name(n)) for e in payload["events"] for n in e["notes"]]
            tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
            fx_group = f"{req['tuning']['id']}|author"
            rescues = track_and_rescue(a, sr, tuning, existing,
                                       partial_table=load_partial_table(fx_group),
                                       coupling_table=load_coupling_table(fx_group))
            for r in rescues:
                rows.append(rescue_row(r, "fixture", d.name, fx_group))
            if rescues:
                print(f"fixture {d.name}: {len(rescues)}")
        except Exception as exc:  # noqa: BLE001 — survey run
            print(f"fixture {d.name}: ERROR {str(exc)[:100]}")

    OUT.write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}  ({len(rows)} rescues)")

    # --- separation summary per label ---
    def dist(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "n=0"
        q = np.percentile(vals, [0, 25, 50, 75, 100])
        return f"n={len(vals)} min={q[0]:.2f} q25={q[1]:.2f} med={q[2]:.2f} q75={q[3]:.2f} max={q[4]:.2f}"

    print("\n=== separation summary ===")
    for label in ("tp", "fp", "fixture"):
        sub = [r for r in rows if r["label"] == label]
        print(f"\n[{label}] n={len(sub)}")
        if not sub:
            continue
        sem = sum(1 for r in sub if r["semitoneCoincident"])
        iso = sum(1 for r in sub if r["strongestEnvRatio"] is None or r["strongestEnvRatio"] > 1.0)
        print(f"  semitoneCoincident: {sem}/{len(sub)}   isolated(no attacker >= self): {iso}/{len(sub)}")
        print(f"  strongestEnvRatio: {dist([r['strongestEnvRatio'] for r in sub])}")
        print(f"  phaseErr: {dist([r['phaseErr'] for r in sub])}")
        print(f"  jerk:     {dist([r['jerk'] for r in sub])}")
        print(f"  reinject: {dist([r['reinjectRatio'] for r in sub])}")
        print(f"  preRing:  {dist([r['preRingRatio'] for r in sub])}")

    # --- candidate tier rules, evaluated directly ---
    def eval_rule(name, demote_fn):
        tp_kept = [r for r in rows if r["label"] == "tp" and not demote_fn(r)]
        fp_kept = [r for r in rows if r["label"] == "fp" and not demote_fn(r)]
        fx_kept = [r for r in rows if r["label"] == "fixture" and not demote_fn(r)]
        k2_kept = sum(1 for r in tp_kept if any(r["tx"] == k[:8] for k in K2_RECORDINGS))
        k2_all = sum(1 for r in rows if r["label"] == "tp" and any(r["tx"] == k[:8] for k in K2_RECORDINGS))
        print(f"{name:55s} TP {len(tp_kept):2d}/16  FP {len(fp_kept):2d}  fixtureAdds {len(fx_kept):2d}/35  K2-TP {k2_kept}/{k2_all}")

    print("\n=== tier rule sweep (kept at event tier) ===")
    def sem_rule(max_ratio):
        def demote(r):
            return r["semitoneCoincident"] and (
                r["semitoneEnvRatio"] is not None and r["semitoneEnvRatio"] <= max_ratio)
        return demote
    eval_rule("demote: semitone attacker (any level)", lambda r: r["semitoneCoincident"])
    eval_rule("demote: semitone attacker, self<=1x attacker", sem_rule(1.0))
    eval_rule("demote: semitone attacker, self<=2x attacker", sem_rule(2.0))
    eval_rule("demote: any attacker stronger than self",
              lambda r: r["strongestEnvRatio"] is not None and r["strongestEnvRatio"] <= 1.0)
    eval_rule("demote: any attacker >= 0.5x self",
              lambda r: r["strongestEnvRatio"] is not None and r["strongestEnvRatio"] <= 2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
