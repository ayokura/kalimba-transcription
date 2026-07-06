"""Round-3 2x2 ablation: residual-forward-scan x residual autopsy (#206).

The replacement claim of merge condition (3) — "the per-tine veto can retire
the recent-note-memory forward-scan" — is measured as a 2x2 over the real
pipeline (tracker rescue ON on all four arms, so this is the round-3
integrated recognizer):

  A fscan on  / autopsy off   (round-2 state = control)
  B fscan off / autopsy on    (the replacement)
  C fscan on  / autopsy on    (branch default while C2 consult is pending)
  D fscan off / autopsy off   (harm of removal alone)

plus base = tracker OFF (main-equivalent), which anchors the round-3
kill-criteria continuation (K2/K3/K4 vs the S4 addendum baseline, counted
on arm C).  Success requires B >= A on every indicator (non-saturated
headline P/R/F1, pooled FP, K2 recall, and no per-recording F1 regression).
Non-saturated set is frozen on the base side (kill-criteria S4 addendum).

Usage: uv run python scripts/audio-analysis/research/pertine_round3_ablation.py
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

OUT = REPO / "docs" / "research" / "pertine-round3-ablation.json"
BOOT_ITERS = 2000
BOOT_SEED = 42

ARMS: dict[str, dict] = {
    "base": {"use_pertine_tracker_rescue": False},
    "A_fscan_on_autopsy_off": {"use_pertine_residual_autopsy": False},
    "B_fscan_off_autopsy_on": {"ablate_residual_forward_scan": True},
    "C_fscan_on_autopsy_on": {},
    "D_fscan_off_autopsy_off": {
        "ablate_residual_forward_scan": True,
        "use_pertine_residual_autopsy": False,
    },
}


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

    txs = gt_tx_ids()
    truths = {tx: nfb.load_ground_truth(nfb.ground_truth_path_for(tx)) for tx in txs}
    matches: dict[str, dict[str, dict]] = {arm: {} for arm in ARMS}
    for arm, overrides in ARMS.items():
        print(f"--- arm {arm} ---")
        for tx in txs:
            if overrides:
                with recognizer_settings.override(**overrides):
                    payload = nfb.transcribe_payload(client, tx, debug=False)
            else:
                payload = nfb.transcribe_payload(client, tx, debug=False)
            m = nfb.match_pairs(truths[tx], nfb.collect_one_best(payload))
            matches[arm][tx] = m
            print(f"  {tx[:8]} F1={m['f1']:.3f} tp={m['tp']} fp={len(m['falsePositives'])}")

    # non-saturated set frozen on the base (tracker OFF) side
    ns_txs = [tx for tx in txs if matches["base"][tx]["f1"] < 1.0]
    headline = {arm: agg([matches[arm][tx] for tx in ns_txs]) for arm in ARMS}
    k2 = {arm: {"tp": sum(matches[arm][tx]["tp"] for tx in K2_RECORDINGS),
                "truth": sum(matches[arm][tx]["truthNotes"] for tx in K2_RECORDINGS)}
          for arm in ARMS}

    print("\n=== headline (non-saturated, frozen on base) ===")
    for arm in ARMS:
        h = headline[arm]
        k2r = k2[arm]["tp"] / k2[arm]["truth"] if k2[arm]["truth"] else 0.0
        print(f"{arm:24s} P={h['P']:.3f} R={h['R']:.3f} F1={h['F1']:.3f} FP={h['fp']:3d} "
              f"K2={k2[arm]['tp']}/{k2[arm]['truth']}={k2r:.3f} CI={h['ci']['microF1CI95']}")

    # replacement claim: B >= A on every indicator
    a, b = headline["A_fscan_on_autopsy_off"], headline["B_fscan_off_autopsy_on"]
    k2_a = k2["A_fscan_on_autopsy_off"]; k2_b = k2["B_fscan_off_autopsy_on"]
    per_rec_regressions = [
        {"tx": tx[:8],
         "A": round(matches["A_fscan_on_autopsy_off"][tx]["f1"], 3),
         "B": round(matches["B_fscan_off_autopsy_on"][tx]["f1"], 3)}
        for tx in txs
        if matches["B_fscan_off_autopsy_on"][tx]["f1"]
        < matches["A_fscan_on_autopsy_off"][tx]["f1"] - 1e-9
    ]
    claim = {
        "R": b["R"] >= a["R"], "P": b["P"] >= a["P"], "F1": b["F1"] >= a["F1"],
        "FP": b["fp"] <= a["fp"],
        "K2": k2_b["tp"] >= k2_a["tp"],
        "noPerRecordingF1Regression": not per_rec_regressions,
    }
    claim_verdict = "HOLDS" if all(claim.values()) else "FAILS"
    print(f"\n=== replacement claim (B >= A): {claim_verdict} ===")
    print(json.dumps(claim))
    if per_rec_regressions:
        print("per-recording regressions (B < A):", json.dumps(per_rec_regressions))

    # round-3 kill-criteria continuation on arm C (branch default) vs base
    base_ns = [matches["base"][tx] for tx in ns_txs]
    c_ns = [matches["C_fscan_on_autopsy_on"][tx] for tx in ns_txs]
    d_r = headline["C_fscan_on_autopsy_on"]["R"] - headline["base"]["R"]
    delta_ci = paired_delta_r_ci(base_ns, c_ns)
    k2_c = k2["C_fscan_on_autopsy_on"]
    k2_rate = k2_c["tp"] / k2_c["truth"] if k2_c["truth"] else 0.0
    k2_verdict = "PASS" if k2_rate >= 0.875 else "FAIL"
    k3_verdict = "PASS" if (d_r >= 0.015 and delta_ci["deltaRCI95"][0] > 0) else (
        "PASS (dR>=+0.015, CI includes 0)" if d_r >= 0.015 else "FAIL")
    k4_verdict = ("PASS" if headline["C_fscan_on_autopsy_on"]["fp"] <= S4_BASELINE["k4Ceiling"]
                  else "FAIL")
    print(f"\n=== round-3 kill continuation (arm C vs base, S4 baseline) ===")
    print(f"K2: {k2_c['tp']}/{k2_c['truth']} = {k2_rate:.3f} (>=0.875) [{k2_verdict}]")
    print(f"K3: dR={d_r:+.3f} (>=+0.015), paired bootstrap CI95={delta_ci['deltaRCI95']} [{k3_verdict}]")
    print(f"K4: FP base {headline['base']['fp']} -> C {headline['C_fscan_on_autopsy_on']['fp']} "
          f"(ceiling {S4_BASELINE['k4Ceiling']}) [{k4_verdict}]")

    OUT.write_text(json.dumps({
        "round": 3, "form": "2x2 fscan x autopsy + base (settings-flag arms)",
        "s4Baseline": S4_BASELINE,
        "arms": {arm: sorted(ov.items()) for arm, ov in ARMS.items()},
        "nonSaturatedTxs": [tx[:8] for tx in ns_txs],
        "headline": headline,
        "k2": {arm: k2[arm] for arm in ARMS},
        "replacementClaim": {"verdict": claim_verdict, "indicators": claim,
                             "perRecordingRegressions": per_rec_regressions},
        "killContinuationArmC": {
            "k2": {"rate": round(k2_rate, 3), "verdict": k2_verdict},
            "k3": {"deltaR": round(d_r, 4), "pairedBootstrap": delta_ci,
                   "verdict": k3_verdict},
            "k4": {"fp": headline["C_fscan_on_autopsy_on"]["fp"],
                   "ceiling": S4_BASELINE["k4Ceiling"], "verdict": k4_verdict},
        },
        "perRecording": {
            arm: {tx[:8]: {"f1": round(matches[arm][tx]["f1"], 3),
                           "tp": matches[arm][tx]["tp"],
                           "fp": len(matches[arm][tx]["falsePositives"])}
                  for tx in txs}
            for arm in ARMS
        },
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
