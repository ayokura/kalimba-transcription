"""Second-pass probes for the review rows of gt_agent_adjudicate.py.

Two targeted measurements:

A. Harmonic-suspect arbitration — for each (child, parent) suspect pair,
   trace both note-band envelopes at 12.5ms steps over onset-0.1s..+0.5s
   and report (1) the Pearson correlation of the normalized envelopes and
   (2) each side's own attack step. A leaked partial rides its parent's
   envelope (r ~ 0.9+, no independent attack); a genuinely struck tine
   moves on its own.

B. Window-clipped candidates — consecutive slots closer than the low-band
   separation window form one strum/roll cluster. Re-measure the long
   FFT window over the whole cluster span (start = first onset + 20ms,
   end = cluster end + tail up to the next distinct onset), which restores
   the frequency resolution the per-slot clipping destroyed. Report each
   unresolved tine's peak against that extended window.

Output feeds the agent's manual verdict; no decisions are made here.

Usage:
    uv run python scripts/audio-analysis/research/gt_agent_probe2.py 1955b5bd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import kalimba_dsp as K  # noqa: E402
from app.transcription.constants import HARMONIC_BAND_CENTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt_agent_observations import (  # noqa: E402
    band_window_sec, hann_spectrum, peak_near, separation_limit_sec,
)

DRAFTS_DIR = REPO_ROOT / "data" / "gt_drafts"
TX_DIR = REPO_ROOT / "data" / "transactions"

ENV_STEP = 0.0125
ENV_PRE = 0.100
ENV_POST = 0.500
ENV_WIN = 0.025
CLUSTER_TAIL_GUARD = 0.010

NOTE_INDEX = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def note_freq(name: str) -> float:
    pc, octave = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((12 * (octave + 1) + NOTE_INDEX[pc] - 69) / 12)


def envelope(audio, sr, freq, t0, t1):
    ts = np.arange(t0, t1, ENV_STEP)
    vals = []
    for t in ts:
        if t - ENV_WIN / 2 < 0 or t + ENV_WIN / 2 > len(audio) / sr:
            vals.append(0.0)
        else:
            vals.append(K.note_band_energy(audio, sr, float(t), freq, ENV_WIN,
                                           HARMONIC_BAND_CENTS))
    return ts, np.asarray(vals)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tx8")
    args = parser.parse_args()
    adj = json.loads((DRAFTS_DIR / f"gt-agent-adjudication-{args.tx8}.json").read_text(encoding="utf-8"))
    rows_doc = json.loads((DRAFTS_DIR / f"{args.tx8}.rows.json").read_text(encoding="utf-8"))
    tx_id = rows_doc["txId"]
    audio, sr = sf.read(TX_DIR / tx_id / "audio.wav", dtype="float32")
    if audio.ndim > 1:
        rms = [(float(np.mean(audio[:: max(1, len(audio) // 5000), ch] ** 2)), ch)
               for ch in range(audio.shape[1])]
        audio = audio[:, max(rms)[1]]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    duration = len(audio) / sr

    rows = adj["rows"]
    onsets = [r["timeSec"] for r in rows]

    # --- A: suspect pairs ---
    suspects_out = []
    for r in rows:
        for s in r["harmonicSuspect"]:
            t = r["timeSec"]
            fc, fp = note_freq(s["note"]), note_freq(s["parent"])
            t0, t1 = max(0.0, t - ENV_PRE), min(duration, t + ENV_POST)
            _, env_c = envelope(audio, sr, fc, t0, t1)
            _, env_p = envelope(audio, sr, fp, t0, t1)
            if env_c.std() > 0 and env_p.std() > 0:
                corr = float(np.corrcoef(env_c, env_p)[0, 1])
            else:
                corr = None
            suspects_out.append({
                "index": r["index"], "timeSec": t,
                "child": s["note"], "parent": s["parent"],
                "envCorr": round(corr, 3) if corr is not None else None,
                "childGain": s["gain"], "childBp": s["bp"], "childCents": s["cents"],
            })

    # --- B: clusters around window-clipped candidates ---
    cluster_out = []
    cand_rows = [r for r in rows if r["windowClippedCandidate"]]
    for r in cand_rows:
        i = r["index"] - 1
        # Grow the cluster forward while gaps stay below the candidate's
        # own required separation window
        j = i
        while j + 1 < len(rows) and onsets[j + 1] - onsets[j] < 0.13:
            j += 1
        t_start = r["timeSec"] + 0.020
        t_end = (onsets[j + 1] - CLUSTER_TAIL_GUARD) if j + 1 < len(rows) else duration
        t_end = min(t_end, t_start + 0.200, duration)
        eff = t_end - t_start
        seg = audio[int(t_start * sr): int(t_end * sr)]
        freqs, spec = hann_spectrum(seg, sr)
        checks = []
        for c in r["windowClippedCandidate"]:
            f = note_freq(c["note"])
            p = peak_near(freqs, spec, f)
            checks.append({
                "note": c["note"], "gain": c["gain"],
                "extendedWindowMs": round(eff * 1000),
                "resolvable": bool(eff >= separation_limit_sec(f)),
                "peakCents": round(p[0], 1) if p else None,
                "peakAmp": round(p[1], 1) if p else None,
            })
        cluster_out.append({
            "index": r["index"], "timeSec": r["timeSec"],
            "clusterEndIndex": rows[j]["index"], "candidates": checks,
        })

    out_path = DRAFTS_DIR / f"gt-agent-probe2-{args.tx8}.json"
    out_path.write_text(json.dumps({
        "suspectPairs": suspects_out,
        "clusters": cluster_out,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO_ROOT)} "
          f"({len(suspects_out)} suspect pairs, {len(cluster_out)} clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
