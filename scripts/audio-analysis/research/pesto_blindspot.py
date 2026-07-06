"""PESTO blind-spot overlap measurement (S6 bets #3 next step, #203).

Judgement rules were FIXED BEFORE MEASUREMENT (#203 comment, user-approved
2026-07-06); changing them after this run requires decision-log + external-
model audit (guardrail 12 procedure):

- measurement set: every GT-complete recording at run time (tx ids frozen
  into the output JSON)
- blind spot: recognizer 1-best FN against current GT; hard miss (absent
  from candidates too) reported as reference only
- matching: note identity + the GT's toleranceSec (default +-0.08 s).
  Basic Pitch is recomputed under the SAME rule (not its older 0.15 s)
- primary verdict: PESTO-unique verified new finds (absent from BP, GT and
  recognizer 1-best, human-verified) < 5 notes -> drop PESTO
- secondary (report-only): blind-spot overlap |PESTO∩BP| / |PESTO| >= 80%
  -> "redundant-leaning" note
- human cost cap: verify top 20 PESTO-unique candidates by confidence

Isolation: PESTO runs via pesto_infer.py in a `uv run --no-project`
subprocess (guardrail 3: external AMT is a dev instrument only).

Usage: uv run python scripts/audio-analysis/research/pesto_blindspot.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

import note_f1_benchmark as nfb  # noqa: E402
from basic_pitch_disagreement import (  # noqa: E402
    midi_to_name,
    run_basic_pitch,
)
from pertine_dualrun import gt_tx_ids  # noqa: E402
from tine_partial_collision_probe import audio_for  # noqa: E402

PESTO_CACHE = REPO / "data" / "pesto_cache"
OUT = REPO / "docs" / "research" / "pesto-blindspot.json"
VERIFY_OUT = REPO / "docs" / "research" / "pesto-verify-candidates.json"
VERIFY_CAP = 20
DEFAULT_TOL = 0.08


def run_pesto(tx_id: str, audio_path: Path) -> list[dict]:
    PESTO_CACHE.mkdir(parents=True, exist_ok=True)
    cache = PESTO_CACHE / f"{tx_id}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    result = subprocess.run(
        [
            "uv", "run", "--python", "3.11", "--no-project",
            "--with", "pesto-pitch", "--with", "soundfile",
            "python", str(REPO / "scripts/audio-analysis/research/pesto_infer.py"),
            str(audio_path),
        ],
        capture_output=True, text=True, timeout=900, cwd=str(REPO),
    )
    if result.returncode != 0:
        raise RuntimeError(f"pesto failed for {tx_id}: {result.stderr[-500:]}")
    rows = json.loads(result.stdout)
    cache.write_text(json.dumps(rows))
    return rows


def _hits(events: list[tuple[float, str]], note: str, t: float, tol: float,
          used: list[bool]) -> int | None:
    """Greedy nearest-unused same-note match (the GT matching rule)."""
    best, best_d = None, None
    for i, (et, en) in enumerate(events):
        if used[i] or en != note:
            continue
        d = abs(et - t)
        if d <= tol and (best_d is None or d < best_d):
            best, best_d = i, d
    return best


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    txs = gt_tx_ids()
    rows = []
    tot = {"fn": 0, "pestoSpots": 0, "bpSpots": 0, "overlap": 0, "hardMiss": 0}
    unique_candidates: list[dict] = []
    for tx in txs:
        # load_ground_truth returns [{"time", "note", "tol"}, ...] (per-note tol)
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        payload = nfb.transcribe_payload(client, tx, debug=False)
        one_best = nfb.collect_one_best(payload)
        match = nfb.match_pairs(truth, one_best)
        fns = match["falseNegatives"]

        ap = audio_for(tx)
        if ap is None:
            print(f"{tx[:8]}: no audio, skipped", file=sys.stderr)
            continue
        pesto = [(r["start"], midi_to_name(r["midi"]), r["confidence"])
                 for r in run_pesto(tx, ap)]
        bp = [(r["start"], midi_to_name(r["midi"])) for r in run_basic_pitch(tx, ap)]
        rec_events = [(p["time"], p["note"]) for p in one_best]
        # candidate pool for the hard-miss reference count: 1-best + slots
        cand_pool = list(rec_events)
        for slot in payload.get("candidateSlots") or []:
            pn = slot.get("primaryNote") or {}
            if pn:
                cand_pool.append((float(slot["startTime"]),
                                  f"{pn['pitchClass']}{pn['octave']}"))
            for c in slot.get("candidates") or []:
                cand_pool.append((float(slot["startTime"]),
                                  f"{c['pitchClass']}{c['octave']}"))

        pesto_events = [(t, n) for t, n, _c in pesto]
        pesto_used = [False] * len(pesto_events)
        bp_used = [False] * len(bp)
        pool_used = [False] * len(cand_pool)
        pesto_spots, bp_spots, both_spots, hard_miss = [], [], 0, 0
        for fn in fns:
            t, note = float(fn["time"]), fn["note"]
            tol = float(fn.get("tol", DEFAULT_TOL))
            pi = _hits(pesto_events, note, t, tol, pesto_used)
            bi = _hits(bp, note, t, tol, bp_used)
            ci = _hits(cand_pool, note, t, tol, pool_used)
            if pi is not None:
                pesto_used[pi] = True
                pesto_spots.append({"t": round(t, 3), "note": note})
            if bi is not None:
                bp_used[bi] = True
                bp_spots.append({"t": round(t, 3), "note": note})
            if pi is not None and bi is not None:
                both_spots += 1
            if ci is not None:
                pool_used[ci] = True
            else:
                hard_miss += 1

        # PESTO-unique new-find candidates: not matching GT (any onset note,
        # each at its own tol), not matching BP, not matching recognizer 1-best.
        gt_used = [False] * len(truth)
        bp_used2 = [False] * len(bp)
        rec_used2 = [False] * len(rec_events)

        def _gt_hit(note: str, t: float) -> bool:
            for i, gt in enumerate(truth):
                if not gt_used[i] and gt["note"] == note \
                        and abs(gt["time"] - t) <= float(gt.get("tol", DEFAULT_TOL)):
                    gt_used[i] = True
                    return True
            return False

        for (t, n, conf) in pesto:
            if _gt_hit(n, t):
                continue
            if _hits(bp, n, t, DEFAULT_TOL, bp_used2) is not None:
                continue
            if _hits(rec_events, n, t, DEFAULT_TOL, rec_used2) is not None:
                continue
            unique_candidates.append(
                {"tx": tx, "t": round(t, 3), "note": n, "confidence": conf})

        rows.append({
            "tx": tx[:8], "fn": len(fns),
            "pestoSpots": len(pesto_spots), "bpSpots": len(bp_spots),
            "overlap": both_spots, "hardMiss": hard_miss,
            "pestoSpotList": pesto_spots,
        })
        tot["fn"] += len(fns); tot["pestoSpots"] += len(pesto_spots)
        tot["bpSpots"] += len(bp_spots); tot["overlap"] += both_spots
        tot["hardMiss"] += hard_miss
        print(f"{tx[:8]} FN={len(fns):3d} pesto={len(pesto_spots):3d} "
              f"bp={len(bp_spots):3d} overlap={both_spots:3d} hardMiss={hard_miss:3d}")

    overlap_rate = (tot["overlap"] / tot["pestoSpots"]) if tot["pestoSpots"] else None
    unique_candidates.sort(key=lambda r: -r["confidence"])
    top = unique_candidates[:VERIFY_CAP]
    print(f"\nTOTAL FN={tot['fn']} pestoSpots={tot['pestoSpots']} bpSpots={tot['bpSpots']} "
          f"overlap={tot['overlap']} rate={overlap_rate if overlap_rate is None else round(overlap_rate, 3)}")
    print(f"PESTO-unique candidates: {len(unique_candidates)} (verify top {len(top)})")
    for r in top:
        print(f"  {r['tx'][:8]} {r['t']:8.3f}s {r['note']:4s} conf={r['confidence']:.3f}")

    OUT.write_text(json.dumps({
        "rulesFixed": "#203 2026-07-06 (pre-measurement)",
        "frozenTxIds": txs,
        "totals": {**tot, "overlapRate": overlap_rate},
        "secondaryVerdict": (
            "redundant-leaning (>=0.80)" if overlap_rate is not None and overlap_rate >= 0.80
            else "not redundant by overlap"),
        "recordings": rows,
        "uniqueCandidateCount": len(unique_candidates),
    }, indent=1) + "\n")
    VERIFY_OUT.write_text(json.dumps({
        "cap": VERIFY_CAP,
        "instruction": "human verify: is each a real played note? primary "
                       "verdict = drop PESTO if verified-real < 5",
        "candidates": top,
        "remainingBeyondCap": len(unique_candidates) - len(top),
    }, indent=1) + "\n")
    print(f"\nwrote {OUT.relative_to(REPO)} / {VERIFY_OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
