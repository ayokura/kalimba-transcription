"""Integrated dual-run (S5 round 2): main recognizer vs per-tine tracker ON.

This is the kill-criteria evaluation run (K2/K3/K4). Unlike the offline
round-1/round-2 projections, both sides here are the *real pipeline* —
base = settings.override(use_pertine_tracker_rescue=False) (main-equivalent),
aug = branch default (tracker ON) — so the numbers are what the fixture
suite and prod would see. Non-saturated set is frozen on the base side
(kill-criteria S4 addendum), K3 carries a paired bootstrap ΔR CI.

Usage: uv run python scripts/audio-analysis/research/pertine_dualrun_integrated.py
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
from pertine_dualrun import K2_RECORDINGS, S4_BASELINE, gt_tx_ids  # noqa: E402

OUT = REPO / "docs" / "research" / "pertine-dualrun-round2-integrated.json"
BOOT_ITERS = 2000
BOOT_SEED = 42


def agg(ns: list[dict]) -> dict:
    tp = sum(r["tp"] for r in ns); tr = sum(r["truthNotes"] for r in ns)
    pr = sum(r["predictedNotes"] for r in ns)
    p = tp / pr if pr else 0.0; rc = tp / tr if tr else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    fp = sum(len(r["falsePositives"]) for r in ns)
    return {"n": len(ns), "tp": tp, "fp": fp, "truth": tr, "pred": pr,
            "P": round(p, 3), "R": round(rc, 3), "F1": round(f1, 3),
            "ci": nfb.bootstrap_micro_f1_ci(ns)}


def paired_delta_r_ci(base: list[dict], aug: list[dict]) -> dict:
    """Paired bootstrap over recordings: CI95 of micro-R(aug) - micro-R(base)."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(base)
    deltas = []
    for _ in range(BOOT_ITERS):
        idx = rng.integers(0, n, n)
        tb = sum(base[i]["tp"] for i in idx); rb = sum(base[i]["truthNotes"] for i in idx)
        ta = sum(aug[i]["tp"] for i in idx); ra = sum(aug[i]["truthNotes"] for i in idx)
        deltas.append((ta / ra if ra else 0.0) - (tb / rb if rb else 0.0))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"iterations": BOOT_ITERS, "seed": BOOT_SEED,
            "deltaRCI95": [round(float(lo), 4), round(float(hi), 4)]}


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.transcription import settings as recognizer_settings
    client = TestClient(app)

    rows = []
    results_base, results_aug = [], []
    k2 = {"tp_base": 0, "tp_aug": 0, "truth": 0}
    for tx in gt_tx_ids():
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        with recognizer_settings.override(use_pertine_tracker_rescue=False):
            payload_base = nfb.transcribe_payload(client, tx, debug=False)
        payload_aug = nfb.transcribe_payload(client, tx, debug=False)
        mb = nfb.match_pairs(truth, nfb.collect_one_best(payload_base))
        ma = nfb.match_pairs(truth, nfb.collect_one_best(payload_aug))
        results_base.append(mb); results_aug.append(ma)
        slots_aug = sum(1 for s in payload_aug.get("candidateSlots") or []
                        if s.get("dropReason") == "pertine-weak-rescue")
        rows.append({"tx": tx[:8],
                     "base": {"tp": mb["tp"], "fp": len(mb["falsePositives"]), "f1": round(mb["f1"], 3)},
                     "aug": {"tp": ma["tp"], "fp": len(ma["falsePositives"]), "f1": round(ma["f1"], 3)},
                     "pertineWeakSlots": slots_aug})
        if tx in K2_RECORDINGS:
            k2["tp_base"] += mb["tp"]; k2["tp_aug"] += ma["tp"]; k2["truth"] += mb["truthNotes"]
        flag = " *F1 DOWN*" if ma["f1"] < mb["f1"] - 1e-9 else ""
        print(f"{tx[:8]} base F1={mb['f1']:.3f} (tp={mb['tp']} fp={len(mb['falsePositives'])})  "
              f"aug F1={ma['f1']:.3f} (tp={ma['tp']} fp={len(ma['falsePositives'])})  weakSlots={slots_aug}{flag}")

    ns_base = [r for r in results_base if r["f1"] < 1.0]
    ns_aug = [ra for rb, ra in zip(results_base, results_aug) if rb["f1"] < 1.0]
    base_h, aug_h = agg(ns_base), agg(ns_aug)
    delta_ci = paired_delta_r_ci(ns_base, ns_aug)
    k2_base = k2["tp_base"] / k2["truth"] if k2["truth"] else 0.0
    k2_aug = k2["tp_aug"] / k2["truth"] if k2["truth"] else 0.0
    d_r = aug_h["R"] - base_h["R"]

    print("\n=== integrated dual-run (non-saturated set frozen on base) ===")
    print(f"base: n={base_h['n']} P={base_h['P']} R={base_h['R']} F1={base_h['F1']} FP={base_h['fp']} CI={base_h['ci']['microF1CI95']}")
    print(f"aug : n={aug_h['n']} P={aug_h['P']} R={aug_h['R']} F1={aug_h['F1']} FP={aug_h['fp']} CI={aug_h['ci']['microF1CI95']}")
    if abs(base_h["tp"] - 304) + abs(base_h["fp"] - 53) > 0:
        print(f"NOTE: base side deviates from S4 doc baseline (tp=304/fp=53) — set changed? tp={base_h['tp']} fp={base_h['fp']}")
    k2_verdict = "PASS" if k2_aug >= 0.875 else "FAIL"
    k3_verdict = "PASS" if (d_r >= 0.015 and delta_ci["deltaRCI95"][0] > 0) else (
        "PASS (dR>=+0.015, CI includes 0)" if d_r >= 0.015 else "FAIL")
    k4_verdict = "PASS" if aug_h["fp"] <= S4_BASELINE["k4Ceiling"] else "FAIL"
    print(f"K2: {k2['tp_base']}/{k2['truth']} -> {k2['tp_aug']}/{k2['truth']} = {k2_aug:.3f} (>=0.875) [{k2_verdict}]")
    print(f"K3: dR={d_r:+.3f} (>=+0.015), paired bootstrap CI95={delta_ci['deltaRCI95']} [{k3_verdict}]")
    print(f"K4: FP {base_h['fp']} -> {aug_h['fp']} (ceiling {S4_BASELINE['k4Ceiling']}) [{k4_verdict}]")

    OUT.write_text(json.dumps({
        "round": 2, "form": "integrated pipeline dual-run (settings-flag toggle)",
        "s4Baseline": S4_BASELINE,
        "recordings": rows,
        "headline": {"base": base_h, "aug": aug_h},
        "k2": {"base": round(k2_base, 3), "aug": round(k2_aug, 3), "truth": k2["truth"],
               "verdict": k2_verdict},
        "k3": {"deltaR": round(d_r, 4), "pairedBootstrap": delta_ci, "verdict": k3_verdict},
        "k4": {"fpBase": base_h["fp"], "fpAug": aug_h["fp"],
               "ceiling": S4_BASELINE["k4Ceiling"], "verdict": k4_verdict},
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
