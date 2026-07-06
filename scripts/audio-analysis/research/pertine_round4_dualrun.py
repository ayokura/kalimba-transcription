"""Round-4 integrated dual-run — kill-count round 3 (#206 in-stage oracle).

Arms (all real pipeline, settings-flag toggles):
  base        = tracker rescue OFF + oracle OFF   (main-equivalent)
  rescue_only = rescue ON, oracle OFF             (round-2 state)
  full        = branch defaults (rescue ON + oracle ON, OR wiring)
  oracle_only = full + fscan ablated              (fscan marginal arm)
  no_mutedip  = full + mute-dip OR-backup ablated (mute-dip marginal arm)

Kill continuation (K2/K3/K4, S4 denominators) is judged on full vs base.
Marginal contributions (merge condition 3 material): oracle = full vs
rescue_only, fscan = full vs oracle_only, mute-dip backup = full vs
no_mutedip. Non-saturated set is frozen on the base side (kill-criteria
S4 addendum).

Usage: uv run python scripts/audio-analysis/research/pertine_round4_dualrun.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

import note_f1_benchmark as nfb  # noqa: E402
from pertine_dualrun import K2_RECORDINGS, S4_BASELINE, gt_tx_ids  # noqa: E402

OUT = REPO / "docs" / "research" / "pertine-round4-dualrun.json"
BOOT_ITERS = 2000
BOOT_SEED = 42

ARMS = {
    "base": {"use_pertine_tracker_rescue": False,
             "use_pertine_residual_oracle": False},
    "rescue_only": {"use_pertine_residual_oracle": False},
    "full": {},
    "oracle_only": {"ablate_residual_forward_scan": True},
    "no_mutedip": {"ablate_residual_mute_dip_backup": True},
}


def agg(ns):
    tp = sum(r["tp"] for r in ns); tr = sum(r["truthNotes"] for r in ns)
    pr = sum(r["predictedNotes"] for r in ns)
    p = tp / pr if pr else 0.0; rc = tp / tr if tr else 0.0
    f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    fp = sum(len(r["falsePositives"]) for r in ns)
    return {"n": len(ns), "tp": tp, "fp": fp, "truth": tr, "pred": pr,
            "P": round(p, 3), "R": round(rc, 3), "F1": round(f1, 3),
            "ci": nfb.bootstrap_micro_f1_ci(ns)}


def paired_delta_r_ci(base, aug):
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
    matches = {arm: {} for arm in ARMS}
    for arm, ov in ARMS.items():
        print(f"--- arm {arm} ---")
        for tx in txs:
            if ov:
                with recognizer_settings.override(**ov):
                    payload = nfb.transcribe_payload(client, tx, debug=False)
            else:
                payload = nfb.transcribe_payload(client, tx, debug=False)
            m = nfb.match_pairs(truths[tx], nfb.collect_one_best(payload))
            matches[arm][tx] = m
            print(f"  {tx[:8]} F1={m['f1']:.3f} tp={m['tp']} fp={len(m['falsePositives'])}")

    ns_txs = [tx for tx in txs if matches["base"][tx]["f1"] < 1.0]
    headline = {arm: agg([matches[arm][tx] for tx in ns_txs]) for arm in ARMS}
    k2 = {arm: {"tp": sum(matches[arm][tx]["tp"] for tx in K2_RECORDINGS),
                "truth": sum(matches[arm][tx]["truthNotes"] for tx in K2_RECORDINGS)}
          for arm in ARMS}

    print("\n=== headline (non-saturated, frozen on base) ===")
    for arm in ARMS:
        h = headline[arm]
        k2r = k2[arm]["tp"] / k2[arm]["truth"] if k2[arm]["truth"] else 0.0
        print(f"{arm:12s} P={h['P']:.3f} R={h['R']:.3f} F1={h['F1']:.3f} FP={h['fp']:3d} "
              f"K2={k2[arm]['tp']}/{k2[arm]['truth']}={k2r:.3f} CI={h['ci']['microF1CI95']}")

    base_ns = [matches["base"][tx] for tx in ns_txs]
    full_ns = [matches["full"][tx] for tx in ns_txs]
    d_r = headline["full"]["R"] - headline["base"]["R"]
    delta_ci = paired_delta_r_ci(base_ns, full_ns)
    k2_full = k2["full"]
    k2_rate = k2_full["tp"] / k2_full["truth"] if k2_full["truth"] else 0.0
    k2_v = "PASS" if k2_rate >= 0.875 else "FAIL"
    k3_v = "PASS" if (d_r >= 0.015 and delta_ci["deltaRCI95"][0] > 0) else (
        "PASS (dR>=+0.015, CI includes 0)" if d_r >= 0.015 else "FAIL")
    k4_v = "PASS" if headline["full"]["fp"] <= S4_BASELINE["k4Ceiling"] else "FAIL"
    print(f"\n=== kill-count round 3 (full vs base, S4 denominators) ===")
    print(f"K2: {k2_full['tp']}/{k2_full['truth']} = {k2_rate:.3f} (>=0.875) [{k2_v}]")
    print(f"K3: dR={d_r:+.3f} (>=+0.015), paired bootstrap CI95={delta_ci['deltaRCI95']} [{k3_v}]")
    print(f"K4: FP base {headline['base']['fp']} -> full {headline['full']['fp']} "
          f"(ceiling {S4_BASELINE['k4Ceiling']}) [{k4_v}]")

    def marginal(other_arm):
        return [
            {"tx": tx[:8],
             other_arm: round(matches[other_arm][tx]["f1"], 3),
             "full": round(matches["full"][tx]["f1"], 3)}
            for tx in txs
            if abs(matches["full"][tx]["f1"] - matches[other_arm][tx]["f1"]) > 1e-9
        ]

    oracle_marg = marginal("rescue_only")
    fscan_marg = marginal("oracle_only")
    mutedip_marg = marginal("no_mutedip")
    print("\noracle marginal (full vs rescue_only), changed recordings:",
          json.dumps(oracle_marg) if oracle_marg else "none")
    print("fscan marginal (full vs oracle_only), changed recordings:",
          json.dumps(fscan_marg) if fscan_marg else "none")
    print("mute-dip backup marginal (full vs no_mutedip), changed recordings:",
          json.dumps(mutedip_marg) if mutedip_marg else "none")

    OUT.write_text(json.dumps({
        "round": "kill-count 3", "form": "in-stage oracle OR mute-dip (round 4)",
        "s4Baseline": S4_BASELINE,
        "nonSaturatedTxs": [tx[:8] for tx in ns_txs],
        "headline": headline,
        "k2": k2,
        "kill": {"k2": {"rate": round(k2_rate, 3), "verdict": k2_v},
                 "k3": {"deltaR": round(d_r, 4), "pairedBootstrap": delta_ci,
                        "verdict": k3_v},
                 "k4": {"fp": headline["full"]["fp"],
                        "ceiling": S4_BASELINE["k4Ceiling"], "verdict": k4_v}},
        "oracleMarginalChanged": oracle_marg,
        "fscanMarginalChanged": fscan_marg,
        "muteDipBackupMarginalChanged": mutedip_marg,
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
