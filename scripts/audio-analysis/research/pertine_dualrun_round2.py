"""Offline round-2 projection: tiered rescues (event tier only) vs baseline.

Two questions before pipeline wiring (the real dual-run is pytest + benchmark
on the integrated branch, per AGENTS.md test policy):
1. Do the tier rule + same-note dedup keep K2=14/16, K3 >= +0.015, K4 <= 61,
   fixture additions 0 — including per-recording corpus floors (9ce7df83
   dropped below its floor in round 1)?
2. The app cannot resolve performer identity, so tables must be keyed by
   tuning only. Does merging author/tester tables (per-pair max transfer,
   union of partials) change the outcome?

Usage: uv run python scripts/audio-analysis/research/pertine_dualrun_round2.py [--per-group]
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
import pertine_tracker as pt  # noqa: E402
from pertine_dualrun import K2_RECORDINGS, MANUAL_CAPTURES, S4_BASELINE, gt_tx_ids  # noqa: E402
from tine_partial_collision_probe import audio_for, load_audio, request_for, resolve_source  # noqa: E402

OUT = REPO / "docs" / "research" / "pertine-dualrun-round2-offline.json"


def merged_tables(tuning_id: str) -> tuple[dict, dict]:
    """(partial_table, coupling_table) merged across performer groups of a
    tuning — per-pair max transfer, union of partials — matching what the
    pipeline (which has no performer identity) will see."""
    partial: dict[str, list[tuple[float, float]]] = {}
    coupling: dict[str, dict[str, float]] = {}
    data_p = json.loads(pt.PARTIAL_TABLE_PATH.read_text()) if pt.PARTIAL_TABLE_PATH.is_file() else {}
    data_c = json.loads(pt.COUPLING_TABLE_PATH.read_text()) if pt.COUPLING_TABLE_PATH.is_file() else {}
    for group in (data_p.get("groups") or {}):
        if not group.startswith(tuning_id + "|"):
            continue
        for note, parts in pt.load_partial_table(group).items():
            partial.setdefault(note, []).extend(parts)
    for group in (data_c.get("groups") or {}):
        if not group.startswith(tuning_id + "|"):
            continue
        for j, d in pt.load_coupling_table(group).items():
            tgt = coupling.setdefault(j, {})
            for k, v in d.items():
                tgt[k] = max(tgt.get(k, 0.0), v)
    return partial, coupling


def main() -> int:
    per_group = "--per-group" in sys.argv
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    results_base, results_aug, rows = [], [], []
    k2 = {"tp_base": 0, "tp_aug": 0, "truth": 0}
    weak_total = 0
    for tx in gt_tx_ids():
        payload = nfb.transcribe_payload(client, tx, debug=False)
        baseline = nfb.collect_one_best(payload)
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        src = resolve_source(tx)
        ap = audio_for(tx)
        if src is None or ap is None:
            continue
        tuning_id, group, _freqs = src
        audio, sr = load_audio(ap)
        req = request_for(tx)
        tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
        if per_group:
            tbl_p, tbl_c = pt.load_partial_table(group), pt.load_coupling_table(group)
        else:
            tbl_p, tbl_c = merged_tables(tuning_id)
        rescues = pt.track_and_rescue(
            audio, sr, tuning,
            existing=[(p["time"], p["note"]) for p in baseline],
            partial_table=tbl_p, coupling_table=tbl_c,
        )
        strong = [r for r in rescues if pt.tier_of(r) == "event"]
        weak = [r for r in rescues if pt.tier_of(r) == "candidate"]
        weak_total += len(weak)
        augmented = baseline + [{"time": r.time, "note": r.note} for r in strong]
        mb = nfb.match_pairs(truth, baseline)
        ma = nfb.match_pairs(truth, augmented)
        results_base.append(mb); results_aug.append(ma)
        rows.append({"tx": tx[:8], "group": group, "strong": len(strong), "weak": len(weak),
                     "base": {k: round(mb[k], 3) if isinstance(mb[k], float) else mb[k] for k in ("tp", "precision", "recall", "f1")},
                     "aug": {k: round(ma[k], 3) if isinstance(ma[k], float) else ma[k] for k in ("tp", "precision", "recall", "f1")},
                     "fpBase": len(mb["falsePositives"]), "fpAug": len(ma["falsePositives"]),
                     "strongList": [{"t": r.time, "note": r.note} for r in strong]})
        if tx in K2_RECORDINGS:
            k2["tp_base"] += mb["tp"]; k2["tp_aug"] += ma["tp"]; k2["truth"] += mb["truthNotes"]
        flag = " *F1 DOWN*" if ma["f1"] < mb["f1"] - 1e-9 else ""
        print(f"{tx[:8]} [{group}] strong={len(strong):2d} weak={len(weak):2d}  base F1={mb['f1']:.3f} (tp={mb['tp']} fp={len(mb['falsePositives'])})  aug F1={ma['f1']:.3f} (tp={ma['tp']} fp={len(ma['falsePositives'])}){flag}")

    def agg_fixed(ns):
        tp = sum(r["tp"] for r in ns); tr = sum(r["truthNotes"] for r in ns)
        pr = sum(r["predictedNotes"] for r in ns)
        p = tp / pr if pr else 0.0; rc = tp / tr if tr else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        fp = sum(len(r["falsePositives"]) for r in ns)
        return {"n": len(ns), "tp": tp, "fp": fp, "P": round(p, 3), "R": round(rc, 3), "F1": round(f1, 3),
                "ci": nfb.bootstrap_micro_f1_ci(ns)}

    ns_base = [r for r in results_base if r["f1"] < 1.0]
    ns_aug = [ra for rb, ra in zip(results_base, results_aug) if rb["f1"] < 1.0]
    base_h, aug_h = agg_fixed(ns_base), agg_fixed(ns_aug)
    k2_base = k2["tp_base"] / k2["truth"] if k2["truth"] else 0.0
    k2_aug = k2["tp_aug"] / k2["truth"] if k2["truth"] else 0.0
    mode = "per-group tables" if per_group else "merged (tuning-only) tables"
    print(f"\n=== headline, event tier only ({mode}) ===")
    print(f"base: n={base_h['n']} P={base_h['P']} R={base_h['R']} F1={base_h['F1']} FP={base_h['fp']} CI={base_h['ci']['microF1CI95']}")
    print(f"aug : n={aug_h['n']} P={aug_h['P']} R={aug_h['R']} F1={aug_h['F1']} FP={aug_h['fp']} CI={aug_h['ci']['microF1CI95']}")
    print(f"K2: base {k2['tp_base']}/{k2['truth']}={k2_base:.3f} -> aug {k2['tp_aug']}/{k2['truth']}={k2_aug:.3f} (target >=0.875)")
    print(f"K3-shape: dR={aug_h['R']-base_h['R']:+.3f} (need >= +0.015)   K4-shape: FP {base_h['fp']} -> {aug_h['fp']} (ceiling {S4_BASELINE['k4Ceiling']})   weak-tier slots: {weak_total}")

    adds = []
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
            if per_group:
                tbl_p, tbl_c = pt.load_partial_table(f"{req['tuning']['id']}|author"), pt.load_coupling_table(f"{req['tuning']['id']}|author")
            else:
                tbl_p, tbl_c = merged_tables(req["tuning"]["id"])
            rescues = pt.track_and_rescue(a, sr, tuning, existing, partial_table=tbl_p, coupling_table=tbl_c)
            strong = [r for r in rescues if pt.tier_of(r) == "event"]
            weak = [r for r in rescues if pt.tier_of(r) == "candidate"]
            if strong or weak:
                adds.append({"fixture": d.name,
                             "strong": [{"t": r.time, "note": r.note} for r in strong],
                             "weak": len(weak)})
        except Exception as exc:  # noqa: BLE001 — survey run
            adds.append({"fixture": d.name, "error": str(exc)[:120]})
    n_strong = sum(len(x.get("strong", [])) for x in adds)
    print(f"\n=== fixture additions at event tier: {n_strong} (must be 0)  weak-tier slots: {sum(x.get('weak', 0) for x in adds)} ===")
    for x in adds:
        if x.get("strong"):
            print(f"  {x['fixture']}: {[(r['t'], r['note']) for r in x['strong']]}")

    OUT.write_text(json.dumps({
        "round": "2-offline", "tables": mode,
        "tier": {"minAttackerRatio": pt.TIER_MIN_ATTACKER_RATIO, "minPreRing": pt.TIER_MIN_PRE_RING,
                 "minReinject": pt.TIER_MIN_REINJECT, "sameNoteSep": pt.RESCUE_SAME_NOTE_SEP},
        "s4Baseline": S4_BASELINE,
        "recordings": rows,
        "headline": {"base": base_h, "aug": aug_h},
        "k2": {"base": round(k2_base, 3), "aug": round(k2_aug, 3), "truth": k2["truth"]},
        "fixtureAdditionsEventTier": n_strong,
        "fixtureDetail": adds,
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
