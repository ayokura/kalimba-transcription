"""FN-overlap artifact: recognizer FNs vs phase-tracking detections (S4 audit item 4).

The S4 gate materials cited "67%/50% of current recognizer FNs are already
detected by the phase-tracking probe" without persisting the computation.
This script regenerates it reproducibly and saves everything the audit asked
for: the combo used, the recognizer FN list, the phase hits, per-FN matches,
and an analytic chance baseline (probability that random placement of the
same per-note prediction counts would match each FN within tolerance).

Output: docs/research/fn-overlap-artifact.json

Usage: uv run python scripts/audio-analysis/research/fn_overlap_artifact.py <bench_json>
  bench_json: note_f1_benchmark.py --json output (for the FN lists)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase_tracking_roc import (  # noqa: E402
    ENV_GATE_FRAC,
    HOP_SEC,
    REF_COMBO,
    candidates_for_track,
    cross_tine_guards,
    demodulate,
    nms,
)
from tine_partial_collision_probe import REPO, audio_for, load_audio, request_for  # noqa: E402
from spectral_pin import resolve_tx  # noqa: E402

TOL = 0.08
RECORDINGS = ["70cc6637", "47902d34"]
OUT = REPO / "docs" / "research" / "fn-overlap-artifact.json"


def main() -> int:
    bench = json.loads(Path(sys.argv[1]).read_text())
    fns = {r["txId"][:8]: r["falseNegatives"] for r in bench["results"]}
    pb, jb = REF_COMBO
    out = {"combo": {"phaseBar": pb, "jerkBar": jb, "guardMode": "full"},
           "toleranceSec": TOL,
           "recognizerFingerprint": bench["summary"]["recognizerFingerprint"],
           "recordings": {}}
    for prefix in RECORDINGS:
        tx = resolve_tx(prefix)
        audio, sr = load_audio(audio_for(tx))
        duration = len(audio) / sr
        hop = int(sr * HOP_SEC)
        hop_sec = hop / sr
        req = request_for(tx)
        tracks = [(n["noteName"], float(n["frequency"]),
                   *demodulate(audio, sr, float(n["frequency"]), hop))
                  for n in req["tuning"]["notes"]]
        gp = max(t[2].max() for t in tracks) or 1.0
        events = []
        for nm_, f, env, ph in tracks:
            hits = candidates_for_track(env, ph, hop_sec, gp * ENV_GATE_FRAC, jb)
            sel = [(i, e, j) for i, e, j in hits if e >= pb]
            for i, e, j in nms(sel, hop_sec):
                events.append((i * hop_sec, nm_, e, j, f))
        events.sort()
        pred = [(round(t, 3), n) for t, n, _e, _j, _f in cross_tine_guards(events, "full")]
        fn_list = fns.get(prefix, [])
        per_note_pred: dict[str, list[float]] = {}
        for t, n in pred:
            per_note_pred.setdefault(n, []).append(t)
        matches, chance_sum = [], 0.0
        for x in fn_list:
            cands = per_note_pred.get(x["note"], [])
            hit = [t for t in cands if abs(t - x["time"]) <= TOL]
            # chance a uniformly random placement of the same k same-note
            # predictions would land at least one within +-TOL of this FN
            k = len(cands)
            p_chance = 1.0 - (1.0 - min(1.0, 2 * TOL / duration)) ** k
            chance_sum += p_chance
            matches.append({"time": x["time"], "note": x["note"],
                            "phaseHits": hit, "matched": bool(hit),
                            "chanceP": round(p_chance, 4)})
        n_matched = sum(1 for m in matches if m["matched"])
        out["recordings"][prefix] = {
            "durationSec": round(duration, 1),
            "recognizerFN": len(fn_list),
            "phasePred": len(pred),
            "matched": n_matched,
            "matchRate": round(n_matched / max(1, len(fn_list)), 3),
            "chanceExpectedMatches": round(chance_sum, 2),
            "perFn": matches,
            "phasePredictions": pred,
        }
        print(f"{prefix}: FN={len(fn_list)} matched={n_matched} "
              f"({100*n_matched/max(1,len(fn_list)):.0f}%) chance≈{chance_sum:.1f} pred={len(pred)}")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
