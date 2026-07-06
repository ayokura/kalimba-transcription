"""Rule-based first-pass adjudication over gt_agent_observations.py output.

Evidence per (slot, tine):
  E1 pitch identity: resolvable long-window peak within +/-35 cents and
     peakOverFloor >= 5 (the only instrument that separates semitones)
  E2 attack: short-window band-energy step >= 2.5 (decay never rises)
  E3 Basic Pitch agrees (independent detector, +/-90ms)
  E4 recognizer draft/top-candidate agrees

Tiers:
  confirmed  = E1 && E2, surviving harmonic explaining-away
  suspect    = E1 && E2 but explainable as a 2x/3x partial (+/-60 cents) of
               a much stronger simultaneous confirmed tine -> needs manual
               review (34L-C has no measured partial table yet, so the
               generic ratio test is conservative both ways)
  candidate  = E2 && E3 but E1 unresolvable (window clipped below the
               semitone separation limit) -> needs manual review
  none       = everything else (attack absent -> residual ring, or no peak)

Slots with no confirmed/suspect/candidate at all are proposed as
decision=ignore (spurious onset slot). This script proposes; the final
verdict is reviewed and written by the adjudicating agent, and the GT is
finalized with method=spectrogram_verified (gt_finalize.py agentVerified
route) per guardrail 8.

Usage:
    uv run python scripts/audio-analysis/research/gt_agent_adjudicate.py 1955b5bd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

E1_MAX_CENTS = 35.0
E1_MIN_FLOOR = 5.0
E2_MIN_GAIN = 2.5
# 34L-C is fully chromatic, so the beam partials land ON real tines:
# A4 x3 -> E6 (+2.6c), D#5 x2 -> D#6 (0c), C4 x3 -> G5 (+1.8c),
# F5 x1.5 -> C6 (+2c). The 1.5x beam partial (measured weight ~0.55 on
# 17-C) must be in the ratio set, and a partial can approach the parent's
# strength, so dominance is near-1 and the attack-step ratio arbitrates:
# a leaked partial rises together with its parent but weaker, a genuinely
# struck tine attacks on its own strength.
HARMONIC_RATIOS = (1.5, 2.0, 3.0, 4.0)
HARMONIC_TOL_CENTS = 40.0
HARMONIC_DOMINANCE = 1.2
HARMONIC_ATTACK_CLEAR = 1.0  # child attackGain >= parent's -> keep despite ratio

NOTE_INDEX = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def note_freq(name: str) -> float:
    pc, octave = name[:-1], int(name[-1])
    midi = 12 * (octave + 1) + NOTE_INDEX[pc]
    return 440.0 * 2 ** ((midi - 69) / 12)


def cents(a: float, b: float) -> float:
    import math
    return abs(1200 * math.log2(a / b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tx8")
    args = parser.parse_args()
    obs_path = REPO_ROOT / "data" / "gt_drafts" / f"gt-agent-observations-{args.tx8}.json"
    doc = json.loads(obs_path.read_text(encoding="utf-8"))

    out = []
    for r in doc["rows"]:
        bp_notes = {b["note"] for b in r["bpNear"]}
        rec_notes = set(r["draftNotes"]) | {c["note"] for c in r["topCandidates"][:3]}
        confirmed: list[dict] = []
        candidate: list[dict] = []
        for name, e in r["tines"].items():
            gain = e.get("attackGain") or 0.0
            pk_cents = e.get("peakCents")
            pk_floor = e.get("peakOverFloor") or 0.0
            e1 = (e["resolvable"] and pk_cents is not None
                  and abs(pk_cents) <= E1_MAX_CENTS and pk_floor >= E1_MIN_FLOOR)
            e2 = gain >= E2_MIN_GAIN
            e3 = name in bp_notes
            e4 = name in rec_notes
            entry = {"note": name, "cents": pk_cents, "floorX": pk_floor,
                     "gain": round(gain, 1), "bp": e3, "rec": e4}
            if e1 and e2:
                confirmed.append(entry)
            elif e2 and e3 and not e["resolvable"]:
                candidate.append(entry)

        # Harmonic explaining-away among confirmed, ascending by frequency
        # so a potential parent is always adjudicated before its partials
        # (floorX order is biased: quiet high bands inflate the ratio).
        # floorX is floor-relative and not comparable across bands, so the
        # dominance test is dropped; the ratio nominates, and the
        # attack-step + Basic-Pitch arbitration decides.
        confirmed.sort(key=lambda x: note_freq(x["note"]))
        kept: list[dict] = []
        suspect: list[dict] = []
        for c in confirmed:
            fx = note_freq(c["note"])
            parent = None
            for k in kept:
                fy = note_freq(k["note"])
                if fy >= fx:
                    continue
                if not any(cents(fx, ratio * fy) <= HARMONIC_TOL_CENTS
                           for ratio in HARMONIC_RATIOS):
                    continue
                if c["gain"] >= k["gain"] * HARMONIC_ATTACK_CLEAR and c["bp"]:
                    continue  # attacks on its own strength + independent detector
                parent = k["note"]
                break
            if parent is not None:
                suspect.append({**c, "parent": parent})
            else:
                kept.append(c)

        if kept and not suspect and not candidate:
            proposal = "accept" if set(n["note"] for n in kept) == set(r["draftNotes"]) else "fix"
        elif not kept and not suspect and not candidate:
            proposal = "ignore"
        else:
            proposal = "review"
        out.append({
            "index": r["index"], "timeSec": r["timeSec"], "gapToNextSec": r["gapToNextSec"],
            "draftNotes": r["draftNotes"], "recognized": r["recognized"],
            "confirmed": kept, "harmonicSuspect": suspect, "windowClippedCandidate": candidate,
            "bpNear": sorted(bp_notes), "proposal": proposal,
        })

    out_path = REPO_ROOT / "data" / "gt_drafts" / f"gt-agent-adjudication-{args.tx8}.json"
    out_path.write_text(json.dumps({
        "source": obs_path.name,
        "rules": {"e1": f"|cents|<={E1_MAX_CENTS} & floor x{E1_MIN_FLOOR} & resolvable",
                  "e2": f"attackGain>={E2_MIN_GAIN}",
                  "harmonic": f"{HARMONIC_RATIOS} +/-{HARMONIC_TOL_CENTS}c, ascending-frequency pass, attack-step + BP arbitrate"},
        "rows": out,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from collections import Counter
    dist = Counter(r["proposal"] for r in out)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    print("proposal distribution:", dict(dist))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
