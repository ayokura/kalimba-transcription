"""Offline dual-run: main recognizer vs main + per-tine tracker rescues (S5 round 1).

Round-1 form of the kill-criteria dual-run: the tracker runs as a
post-processing judge on the recognizer's output (no pipeline surgery), so
K2/K3/K4-shaped numbers and the fixture-addition risk can be measured before
committing to integration. Kill-relevant 巡 counting starts at the
*integrated* dual-run (round 2+); this run gates whether integration is
worth doing at all.

Outputs:
- per-GT-recording baseline vs augmented P/R/F1
- K2: combined recall on 4e1ae5c6 + 9ce7df83 (target >= 14/16 = 0.875)
- K3-shaped: non-saturated (n=9) micro R delta + bootstrap CI (S4 baseline R=0.726)
- K4-shaped: non-saturated pooled FP delta (S4 baseline FP=53, ceiling 61)
- fixture-addition risk: rescues proposed on completed manual-capture
  fixtures (each would be an exact-match regression after integration)

Usage: uv run python scripts/audio-analysis/research/pertine_dualrun.py [--skip-fixtures]
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

import note_f1_benchmark as nfb  # noqa: E402
from pertine_tracker import load_coupling_table, load_partial_table, track_and_rescue  # noqa: E402
from tine_partial_collision_probe import audio_for, load_audio, request_for, resolve_source  # noqa: E402

K2_RECORDINGS = ["4e1ae5c6-df9a-4876-917d-b7e47699c8e5", "9ce7df83-33a0-455d-bf86-c9392ce6f777"]
S4_BASELINE = {"microR": 0.726, "pooledFP": 53, "k4Ceiling": 61}
MANUAL_CAPTURES = REPO / "apps" / "api" / "tests" / "fixtures" / "manual-captures"
OUT = REPO / "docs" / "research" / "pertine-dualrun-round1.json"


def gt_tx_ids() -> list[str]:
    ids = []
    for base in (REPO / "apps/api/tests/fixtures/free-performance-corpus",
                 REPO / "apps/api/tests/fixtures/transaction-captures"):
        for d in sorted(base.iterdir()):
            if (d / "ground_truth.json").is_file() and d.name not in ids:
                ids.append(d.name)
    return ids


def main() -> int:
    skip_fixtures = "--skip-fixtures" in sys.argv
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    results_base, results_aug, rows = [], [], []
    k2 = {"tp_base": 0, "tp_aug": 0, "truth": 0}
    for tx in gt_tx_ids():
        payload = nfb.transcribe_payload(client, tx, debug=False)
        baseline = nfb.collect_one_best(payload)
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        src = resolve_source(tx)
        ap = audio_for(tx)
        if src is None or ap is None:
            continue
        _tuning_id, group, freqs_map = src
        audio, sr = load_audio(ap)
        req = request_for(tx)
        tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
        rescues = track_and_rescue(
            audio, sr, tuning,
            existing=[(p["time"], p["note"]) for p in baseline],
            partial_table=load_partial_table(group),
            coupling_table=load_coupling_table(group),
        )
        augmented = baseline + [{"time": r.time, "note": r.note} for r in rescues]
        mb = nfb.match_pairs(truth, baseline)
        ma = nfb.match_pairs(truth, augmented)
        results_base.append(mb); results_aug.append(ma)
        rows.append({"tx": tx[:8], "group": group, "rescues": len(rescues),
                     "base": {k: mb[k] for k in ("tp", "precision", "recall", "f1")},
                     "aug": {k: ma[k] for k in ("tp", "precision", "recall", "f1")},
                     "fpBase": len(mb["falsePositives"]), "fpAug": len(ma["falsePositives"]),
                     "rescueList": [{"t": r.time, "note": r.note, "err": r.phase_err, "jerk": r.jerk} for r in rescues]})
        if tx in K2_RECORDINGS:
            k2["tp_base"] += mb["tp"]; k2["tp_aug"] += ma["tp"]; k2["truth"] += mb["truthNotes"]
        print(f"{tx[:8]} [{group}] rescues={len(rescues):3d}  base F1={mb['f1']:.3f} (tp={mb['tp']} fp={len(mb['falsePositives'])})  aug F1={ma['f1']:.3f} (tp={ma['tp']} fp={len(ma['falsePositives'])})")

    def agg(results):
        ns = [r for r in results if r["f1"] < 1.0]
        tp = sum(r["tp"] for r in ns); tr = sum(r["truthNotes"] for r in ns)
        pr = sum(r["predictedNotes"] for r in ns)
        p = tp / pr if pr else 0.0; rc = tp / tr if tr else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        fp = sum(len(r["falsePositives"]) for r in ns)
        return {"n": len(ns), "tp": tp, "fp": fp, "P": round(p, 3), "R": round(rc, 3), "F1": round(f1, 3),
                "ci": nfb.bootstrap_micro_f1_ci(ns)}

    # NOTE: non-saturated set is frozen to the BASELINE's non-saturated
    # recordings so both sides are measured on the same set (kill-criteria
    # S4 addendum requirement).
    ns_ids = {id(r) for r in results_base if r["f1"] < 1.0}
    ns_base = [r for r in results_base if r["f1"] < 1.0]
    ns_aug = [ra for rb, ra in zip(results_base, results_aug) if rb["f1"] < 1.0]

    def agg_fixed(ns):
        tp = sum(r["tp"] for r in ns); tr = sum(r["truthNotes"] for r in ns)
        pr = sum(r["predictedNotes"] for r in ns)
        p = tp / pr if pr else 0.0; rc = tp / tr if tr else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        fp = sum(len(r["falsePositives"]) for r in ns)
        return {"n": len(ns), "tp": tp, "fp": fp, "P": round(p, 3), "R": round(rc, 3), "F1": round(f1, 3),
                "ci": nfb.bootstrap_micro_f1_ci(ns)}

    base_h, aug_h = agg_fixed(ns_base), agg_fixed(ns_aug)
    k2_base = k2["tp_base"] / k2["truth"] if k2["truth"] else 0.0
    k2_aug = k2["tp_aug"] / k2["truth"] if k2["truth"] else 0.0
    print("\n=== headline (non-saturated set frozen on baseline) ===")
    print(f"base: n={base_h['n']} P={base_h['P']} R={base_h['R']} F1={base_h['F1']} FP={base_h['fp']} CI={base_h['ci']['microF1CI95']}")
    print(f"aug : n={aug_h['n']} P={aug_h['P']} R={aug_h['R']} F1={aug_h['F1']} FP={aug_h['fp']} CI={aug_h['ci']['microF1CI95']}")
    print(f"K2 (4e1ae5c6+9ce7df83): base {k2['tp_base']}/{k2['truth']}={k2_base:.3f} -> aug {k2['tp_aug']}/{k2['truth']}={k2_aug:.3f} (target >=0.875)")
    print(f"K3-shape: dR={aug_h['R']-base_h['R']:+.3f} (need >= +0.015)   K4-shape: FP {base_h['fp']} -> {aug_h['fp']} (ceiling {S4_BASELINE['k4Ceiling']})")

    fixture_risk = None
    if not skip_fixtures:
        adds = []
        for d in sorted(MANUAL_CAPTURES.iterdir()):
            wav, reqp = d / "audio.wav", d / "request.json"
            if not (wav.is_file() and reqp.is_file()):
                continue
            payload = None
            try:
                req = json.loads(reqp.read_text())
                import wave as wavmod
                import numpy as np
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
                fx_group = f"{req['tuning']['id']}|author"  # manual captures are the author's
                rescues = track_and_rescue(a, sr, tuning, existing,
                                           partial_table=load_partial_table(fx_group),
                                           coupling_table=load_coupling_table(fx_group))
                if rescues:
                    adds.append({"fixture": d.name, "rescues": [{"t": r.time, "note": r.note} for r in rescues]})
            except Exception as exc:  # noqa: BLE001 — survey run, report and continue
                adds.append({"fixture": d.name, "error": str(exc)[:120]})
        n_bad = sum(len(x.get("rescues", [])) for x in adds)
        fixture_risk = {"fixturesWithAdditions": len([x for x in adds if x.get("rescues")]),
                        "totalAdditions": n_bad, "detail": adds}
        print(f"\n=== fixture-addition risk: {fixture_risk['fixturesWithAdditions']} fixtures, {n_bad} additions (each = would-be exact-match regression) ===")
        for x in adds:
            if x.get("rescues"):
                print(f"  {x['fixture']}: {[(r['t'], r['note']) for r in x['rescues']]}")

    OUT.write_text(json.dumps({
        "round": 1, "form": "offline post-processing (pre-integration)",
        "params": {"phaseBar": 0.7, "jerkBar": 150.0},
        "s4Baseline": S4_BASELINE,
        "recordings": rows,
        "headline": {"base": base_h, "aug": aug_h},
        "k2": {"base": round(k2_base, 3), "aug": round(k2_aug, 3), "truth": k2["truth"]},
        "fixtureRisk": {k: v for k, v in (fixture_risk or {}).items() if k != "detail"},
        "fixtureRiskDetail": (fixture_risk or {}).get("detail"),
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
