"""Round-3 design probes: can per-tine state replace the residual-decay
bulk rejection chain (#206)? Analysis only — no recognizer changes.

Probe A (stakes): across GT recordings, how many residual-decay-no-reattack
segment drops overlap a GT false negative (recall the mechanism costs) vs
drop cleanly (precision it protects)? Also counts residual-forward-scan
promotions (the recent-note-memory rescue the replacement would retire).

Probe B (discrimination): for each dropped segment, detection-core features
(phase RMS error / jerk / re-injection — NO carryover restriction, because a
veto judge adjudicates a segment where broadband already asserted an onset)
for (i) the dropped primary's tine (correct suppressions should show no
fresh-attack evidence) and (ii) any overlapping FN note's tine (a
replacement must fire here).

Probe C (robustness, #206 reproduction): ebecf0c6 baseline vs lowpass 8 kHz
(metamorphic alarm transform): F5 fresh-attack features at the two dropped
strikes (2.603 / 3.243) and D5 residual features must be ~invariant,
demonstrating the WARN would not occur under a tracker-informed judge.

Usage: uv run python scripts/audio-analysis/research/pertine_round3_probe.py
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
from augmentation_robustness import transform_lowpass  # noqa: E402
from pertine_dualrun import gt_tx_ids  # noqa: E402
from tine_partial_collision_probe import audio_for, load_audio, request_for  # noqa: E402

from app.transcription.pertine import (  # noqa: E402
    ENV_GATE_FRAC,
    HOP_SEC,
    JERK_BAR,
    PHASE_BAR,
    REINJECT_LOOKBACK,
    _candidates_for_track,
    _demodulate,
)

OUT = REPO / "docs" / "research" / "pertine-round3-probe.json"
DROP_REASON = "residual-decay-no-reattack"
WIN_PAD = 0.10
PROBE_JERK_FLOOR = 10.0


class TineProbe:
    """Full-tuning demodulation for one audio buffer, with window features."""

    def __init__(self, audio: np.ndarray, sr: int, tuning: list[tuple[str, float]]):
        self.sr = sr
        hop = int(sr * HOP_SEC)
        self.hop_sec = hop / sr
        t_axis = np.arange(len(audio)) / sr
        self.tracks = {}
        for name, freq in tuning:
            if name in self.tracks or freq <= 0:
                continue
            env, phase = _demodulate(np.asarray(audio, dtype=np.float64), sr, freq, hop, t_axis)
            self.tracks[name] = (freq, env, phase)
        self.abs_gate = (max(tr[1].max() for tr in self.tracks.values()) or 1.0) * ENV_GATE_FRAC
        self._cands: dict[str, list] = {}

    def _hits(self, note: str) -> list:
        if note not in self._cands:
            _f, env, phase = self.tracks[note]
            self._cands[note] = _candidates_for_track(
                env, phase, self.hop_sec, self.abs_gate, PROBE_JERK_FLOOR)
        return self._cands[note]

    def features(self, note: str, t0: float, t1: float) -> dict:
        """Strongest detection-core evidence for `note` inside [t0, t1]."""
        if note not in self.tracks:
            return {"note": note, "error": "not-in-tuning"}
        _f, env, _p = self.tracks[note]
        hits = [(i, e, j) for i, e, j in self._hits(note)
                if t0 <= i * self.hop_sec <= t1]
        out = {"note": note, "win": [round(t0, 3), round(t1, 3)], "nHits": len(hits)}
        if not hits:
            out["fires"] = False
            return out
        i, e, j = max(hits, key=lambda h: h[2])
        t = i * self.hop_sec
        a = int(t / self.hop_sec)
        attack_peak = float(env[a:a + int(0.06 / self.hop_sec)].max())
        lb = max(0, int((t - REINJECT_LOOKBACK) / self.hop_sec))
        recent_max = float(env[lb:max(lb + 1, a - 2)].max()) if a > 2 else 0.0
        out.update({
            "t": round(t, 3), "phaseErr": round(e, 2), "jerk": round(j, 1),
            "reinject": round(attack_peak / max(recent_max, 1e-12), 2),
            "envPeakVsGate": round(attack_peak / self.abs_gate, 2),
            "fires": bool(e >= PHASE_BAR and j >= JERK_BAR),
        })
        return out


def main() -> int:
    from fastapi.testclient import TestClient
    from app.main import app
    from app.transcription import settings as recognizer_settings
    client = TestClient(app)

    recordings_out = []
    tot = {"slots": 0, "fnOverlapSlots": 0, "fnNotes": 0, "cleanSlots": 0,
           "forwardScanPromotions": 0,
           "vetoFiresOnFn": 0, "vetoFiresOnCleanPrimary": 0}
    for tx in gt_tx_ids():
        truth = nfb.load_ground_truth(nfb.ground_truth_path_for(tx))
        with recognizer_settings.override(use_pertine_tracker_rescue=False):
            payload = nfb.transcribe_payload(client, tx, debug=True)
        match = nfb.match_pairs(truth, nfb.collect_one_best(payload))
        fns = match["falseNegatives"]
        slots = [s for s in payload.get("candidateSlots") or []
                 if s.get("dropReason") == DROP_REASON]
        fscan = json.dumps(payload.get("debug", {}).get("segmentCandidates", [])).count(
            "residual-forward-scan")
        tot["forwardScanPromotions"] += fscan
        if not slots:
            if fscan:
                recordings_out.append({"tx": tx[:8], "slots": [], "forwardScan": fscan})
            continue
        ap = audio_for(tx)
        req = request_for(tx)
        if ap is None or req is None:
            continue
        audio, sr = load_audio(ap)
        tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
        probe = TineProbe(audio, sr, tuning)
        slot_rows = []
        for s in slots:
            t0, t1 = float(s["startTime"]) - WIN_PAD, float(s["endTime"]) + WIN_PAD
            fns_in = [fn for fn in fns if t0 <= fn["time"] <= t1]
            primary = s["primaryNote"]
            pname = f"{primary['pitchClass']}{primary['octave']}"
            row = {
                "win": [round(float(s["startTime"]), 3), round(float(s["endTime"]), 3)],
                "droppedPrimary": pname,
                "fnNotesInWindow": [{"t": round(fn["time"], 3), "note": fn["note"]} for fn in fns_in],
                "primaryFeatures": probe.features(pname, t0, t1),
                "fnFeatures": [probe.features(fn["note"], max(t0, fn["time"] - WIN_PAD),
                                              fn["time"] + WIN_PAD) for fn in fns_in],
            }
            slot_rows.append(row)
            tot["slots"] += 1
            if fns_in:
                tot["fnOverlapSlots"] += 1
                tot["fnNotes"] += len(fns_in)
                tot["vetoFiresOnFn"] += sum(1 for f in row["fnFeatures"] if f.get("fires"))
            else:
                tot["cleanSlots"] += 1
                if row["primaryFeatures"].get("fires"):
                    tot["vetoFiresOnCleanPrimary"] += 1
        recordings_out.append({"tx": tx[:8], "forwardScan": fscan, "slots": slot_rows})
        print(f"{tx[:8]}: residual-decay slots={len(slots)} fnOverlap={sum(1 for r in slot_rows if r['fnNotesInWindow'])} forwardScan={fscan}")

    # --- Probe C: ebecf0c6 lowpass invariance ---
    eb = next(tx for tx in gt_tx_ids() if tx.startswith("ebecf0c6"))
    audio, sr = load_audio(audio_for(eb))
    req = request_for(eb)
    tuning = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
    lp_audio, lp_meta = transform_lowpass(np.asarray(audio, dtype=np.float64), sr, cutoff_hz=8000.0)
    probes = {"baseline": TineProbe(audio, sr, tuning),
              "lowpass8k": TineProbe(np.asarray(lp_audio, dtype=np.float64), sr, tuning)}
    c_rows = []
    for label, pr in probes.items():
        for t_strike in (2.603, 3.243):
            c_rows.append({"variant": label, "target": f"F5@{t_strike}",
                           **pr.features("F5", t_strike - 0.12, t_strike + 0.12)})
            c_rows.append({"variant": label, "target": f"D5(residual)@{t_strike}",
                           **pr.features("D5", t_strike - 0.12, t_strike + 0.12)})
    print("\n=== Probe C: ebecf0c6 lowpass invariance (F5 must fire in both; D5 must not) ===")
    for r in c_rows:
        print(f"  {r['variant']:9s} {r['target']:18s} fires={r.get('fires')} phaseErr={r.get('phaseErr')} jerk={r.get('jerk')} reinject={r.get('reinject')}")

    print("\n=== totals ===")
    print(json.dumps(tot, indent=1))
    OUT.write_text(json.dumps({
        "dropReason": DROP_REASON, "probeJerkFloor": PROBE_JERK_FLOOR,
        "coreBars": {"phase": PHASE_BAR, "jerk": JERK_BAR},
        "totals": tot, "recordings": recordings_out,
        "lowpassInvariance": {"meta": lp_meta, "rows": c_rows},
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
