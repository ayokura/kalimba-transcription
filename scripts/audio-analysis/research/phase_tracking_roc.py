"""ROC calibration of the phase-tracking onset detector (third-term S3).

The S0 probe (branch claude/s5-agenda4-bg-reattack-rescue, fcedeb3) showed
the phase-tracking detector reaches 92-96% inclusion of broadband onsets and
6x detection on the fast take, but over-detects 2-5x. That probe had no GT;
now both Twinkle takes carry ear_verified GT + spectral pins (90a8c0f), so
the PHASE_BAR x JERK_BAR operating surface can be measured properly.

Method:
- Per tine: heterodyne demodulation identical to the probe (5 ms hop).
- Candidate pass: every hop clearing the *loosest* swept jerk bar, with the
  fixed gates (absolute env gate, sustain-holds-10% ring test) applied and
  phase RMS error computed. This factors the expensive work out of the sweep.
- Sweep: filter candidates by (jerk >= J, phase_err >= P), per-tine NMS,
  then the probe's cross-tine guards (near-frequency dominance, harmonic
  parent) — these run per combo because they depend on the surviving set.
- Score: note-level greedy 1:1 match vs GT (note must match; |dt| <= 60 ms
  vs pinned time, 80 ms vs unpinned GT time). unpinnable notes stay in GT —
  recall over them is exactly the carryover-mask capability we care about.

Output: docs/research/phase-tracking-roc.json + stdout table.

Usage: uv run python scripts/audio-analysis/research/phase_tracking_roc.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tine_partial_collision_probe import (  # noqa: E402
    REPO,
    audio_for,
    load_audio,
    request_for,
)
from spectral_pin import gt_path_for, resolve_tx  # noqa: E402

HOP_SEC = 0.005
PAST_SEC = 0.10
FUT_SEC = 0.06
MIN_SEP_SEC = 0.06
DOM_RATIO = 5.0
DOM_CENTS = 350.0
ENV_GATE_FRAC = 0.02
SUSTAIN_FROM_SEC = 0.10
SUSTAIN_TO_SEC = 0.25
SUSTAIN_MIN_FRAC = 0.10

PHASE_BARS = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
JERK_BARS = [20.0, 35.0, 50.0, 75.0, 100.0, 150.0]

TOL_PINNED = 0.06
TOL_UNPINNED = 0.08

RECORDINGS = ["70cc6637", "47902d34"]
OUT_PATH = REPO / "docs" / "research" / "phase-tracking-roc.json"


def demodulate(audio, sr, freq, hop):
    t = np.arange(len(audio)) / sr
    z = audio * np.exp(-2j * np.pi * freq * t)
    bw = max(freq * 0.025, 12.0)
    sos = butter(3, bw, btype="low", fs=sr, output="sos")
    zf = sosfilt(sos, z.real) + 1j * sosfilt(sos, z.imag)
    zh = zf[::hop]
    return np.abs(zh), np.unwrap(np.angle(zh))


def candidates_for_track(env, phase, hop_sec, abs_gate, jerk_floor):
    """All hops passing fixed gates + the loosest jerk bar, with features."""
    n = len(env)
    past = int(PAST_SEC / hop_sec)
    fut = int(FUT_SEC / hop_sec)
    s_from = int(SUSTAIN_FROM_SEC / hop_sec)
    s_to = int(SUSTAIN_TO_SEC / hop_sec)
    tt = np.arange(n) * hop_sec
    out = []
    for i in range(past + 2, n - s_to - 1):
        seg = env[i - 2:i + fut]
        denv = np.diff(seg) / hop_sec
        ref = np.median(env[i - past:i - 2]) + 1e-9
        jerk = float(denv.max() / ref) if len(denv) else 0.0
        if jerk < jerk_floor:
            continue
        attack_peak = env[i:i + fut].max()
        if attack_peak < abs_gate:
            continue
        sustain = np.median(env[i + s_from:i + s_to])
        if sustain < attack_peak * SUSTAIN_MIN_FRAC:
            continue
        p_t = tt[i - past:i - 2]
        p_y = phase[i - past:i - 2]
        coef = np.polyfit(p_t, p_y, 1)
        pred = np.polyval(coef, tt[i + 1:i + fut])
        err = float(np.sqrt(np.mean((phase[i + 1:i + fut] - pred) ** 2)))
        out.append((i, err, jerk))
    return out


def nms(hits, hop_sec):
    if not hits:
        return []
    min_sep = int(MIN_SEP_SEC / hop_sec)
    hits = sorted(hits, key=lambda h: -h[2])
    kept, taken = [], []
    for h in hits:
        if all(abs(h[0] - t) >= min_sep for t in taken):
            kept.append(h)
            taken.append(h[0])
    return sorted(kept)


def cross_tine_guards(events):
    """events: [(t, note, err, jerk, freq)] sorted by t. Probe's two guards."""
    kept = []
    for e in events:
        t, name, err, jerk, freq = e
        dominated = any(
            abs(o[0] - t) <= 0.03 and o[1] != name and (
                (abs(1200 * np.log2(o[4] / freq)) <= 250 and o[3] > jerk)
                or (abs(1200 * np.log2(o[4] / freq)) <= DOM_CENTS and o[3] >= jerk * DOM_RATIO)
            )
            for o in events
        )
        if dominated:
            continue
        harmonic_parent = any(
            abs(o[0] - t) <= 0.03 and o[1] != name and any(
                abs(1200 * np.log2(freq / (o[4] * m))) <= 50 for m in (2.0, 3.0, 4.0)
            )
            for o in events
        )
        if not harmonic_parent:
            kept.append(e)
    return kept


def load_gt_notes(tx_full):
    """[(time, note, tol)] — pinned time when available."""
    gt = json.loads(gt_path_for(tx_full).read_text())
    pins_p = gt_path_for(tx_full).parent / "spectral_pins.json"
    pin_map = {}
    if pins_p.is_file():
        for p in json.loads(pins_p.read_text())["pins"]:
            if p.get("status") == "pinned":
                pin_map[(p["index"], p["note"])] = p["pinSec"]
    out = []
    for i, onset in enumerate(gt["onsets"]):
        for note in onset.get("notes") or []:
            pin = pin_map.get((i, note))
            if pin is not None:
                out.append((pin, note, TOL_PINNED))
            else:
                out.append((float(onset["timeSec"]), note, TOL_UNPINNED))
    return out


def score(pred, gt_notes):
    """Greedy 1:1 note-level matching. pred: [(t, note)], gt: [(t, note, tol)]."""
    used = [False] * len(pred)
    tp = 0
    for t_gt, note, tol in gt_notes:
        best, best_dt = -1, 1e9
        for j, (t_p, n_p) in enumerate(pred):
            if used[j] or n_p != note:
                continue
            dt = abs(t_p - t_gt)
            if dt <= tol and dt < best_dt:
                best, best_dt = j, dt
        if best >= 0:
            used[best] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt_notes) - tp
    return tp, fp, fn


def main() -> int:
    jerk_floor = min(JERK_BARS)
    results = {"generated": str(date.today()), "phaseBars": PHASE_BARS, "jerkBars": JERK_BARS,
               "tolerances": {"pinned": TOL_PINNED, "unpinned": TOL_UNPINNED},
               "fixedGates": {"envGateFrac": ENV_GATE_FRAC, "sustainMinFrac": SUSTAIN_MIN_FRAC},
               "recordings": {}, "pooled": {}}
    per_combo_pool = {}
    for prefix in RECORDINGS:
        tx = resolve_tx(prefix)
        audio, sr = load_audio(audio_for(tx))
        hop = int(sr * HOP_SEC)
        hop_sec = hop / sr
        req = request_for(tx)
        notes = [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]
        tracks = []
        for name, freq in notes:
            env, phase = demodulate(audio, sr, freq, hop)
            tracks.append((name, freq, env, phase))
        global_peak = max(t[2].max() for t in tracks) or 1.0
        abs_gate = global_peak * ENV_GATE_FRAC
        cand = {}
        for name, freq, env, phase in tracks:
            cand[name] = (freq, candidates_for_track(env, phase, hop_sec, abs_gate, jerk_floor))
        gt_notes = load_gt_notes(tx)
        print(f"\n=== {prefix} GT notes={len(gt_notes)} candidates={sum(len(c[1]) for c in cand.values())} ===")
        print("PHASE  JERK   pred    P      R      F1")
        rec_out = {}
        for pb in PHASE_BARS:
            for jb in JERK_BARS:
                events = []
                for name, (freq, hits) in cand.items():
                    sel = [(i, e, j) for i, e, j in hits if e >= pb and j >= jb]
                    for i, e, j in nms(sel, hop_sec):
                        events.append((i * hop_sec, name, e, j, freq))
                events.sort()
                kept = cross_tine_guards(events)
                pred = [(t, n) for t, n, _e, _j, _f in kept]
                tp, fp, fn = score(pred, gt_notes)
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                key = f"{pb}/{jb}"
                rec_out[key] = {"pred": len(pred), "tp": tp, "fp": fp, "fn": fn,
                                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
                agg = per_combo_pool.setdefault(key, [0, 0, 0])
                agg[0] += tp; agg[1] += fp; agg[2] += fn
                print(f"{pb:5.2f} {jb:6.0f} {len(pred):5d}  {p:.3f}  {r:.3f}  {f1:.3f}")
        results["recordings"][prefix] = {"gtNotes": len(gt_notes), "combos": rec_out}
    print("\n=== pooled (2 recordings) ===")
    print("PHASE/JERK   P      R      F1")
    for key, (tp, fp, fn) in sorted(per_combo_pool.items()):
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        results["pooled"][key] = {"tp": tp, "fp": fp, "fn": fn,
                                  "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}
        print(f"{key:10s}  {p:.3f}  {r:.3f}  {f1:.3f}")
    OUT_PATH.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {OUT_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
