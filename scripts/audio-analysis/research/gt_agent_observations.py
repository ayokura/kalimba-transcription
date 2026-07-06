"""Per-slot multi-instrument observations for agent-side GT adjudication.

Built for held-out 1955b5bd (34L-C, 106 draft slots) where human ear
adjudication was declared infeasible (2026-07-06) and the user approved
agent-led adjudication under the standing permission "real-note calls are
agent-decidable when backed by multiple independent observations".

Two-instrument design (user-flagged FFT-window leakage constraint: 34L-C
neighbors are one semitone apart, so a fixed 50ms window cannot separate
C4 from C#4):

1. Pitch identity — long-window FFT, band-dependent length (130/90/50ms
   for <350 / <600 / >=600 Hz; Hann main-lobe separation ~1.44/T vs the
   local semitone spacing), window start onset+20ms (skips the broadband
   attack transient), end clipped to next_onset-10ms. If the clipped
   window drops below the separation limit for that band the tine is
   reported resolvable=false instead of guessing. Peak location via
   parabolic interpolation; identity is the cents offset to the tine.
2. Attack existence — short-window (25ms) band-energy step around the
   onset (pre at -35ms vs post at +35ms), the same measurement family the
   recognizer itself uses. Short windows smear semitone neighbors in the
   low band, so this instrument only nominates candidate tines; identity
   always comes from instrument 1.

Independent cross-checks recorded per slot: Basic Pitch notes (isolated
run, dev-instrument-only per guardrail 3) within +/-90ms, and the draft's
recognizer top candidates. Adjudication itself happens downstream; this
script only measures.

Usage (from repo root):
    uv run python scripts/audio-analysis/research/gt_agent_observations.py \
        1955b5bd --bp <bp_notes.json> \
        [--out data/gt_drafts/gt-agent-observations-1955b5bd.json]  # default: data/gt_drafts/ (local-only — GT 相当の note 列を含むため権利未確認録音でも安全)
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

DRAFTS_DIR = REPO_ROOT / "data" / "gt_drafts"
TX_DIR = REPO_ROOT / "data" / "transactions"

ATTACK_WIN_SEC = 0.025
ATTACK_PRE_SEC = 0.035
ATTACK_POST_SEC = 0.035
WINDOW_START_OFFSET_SEC = 0.020
NEXT_ONSET_GUARD_SEC = 0.010
PEAK_SEARCH_CENTS = 60.0
NOISE_FLOOR_STEP_SEC = 0.5
ZERO_PAD_FACTOR = 4


def band_window_sec(freq: float) -> float:
    if freq < 350.0:
        return 0.130
    if freq < 600.0:
        return 0.090
    return 0.050


def separation_limit_sec(freq: float) -> float:
    """Minimum Hann window to split this tine from a semitone neighbor."""
    semitone_hz = freq * (2 ** (1 / 12) - 1)
    return 1.44 / semitone_hz


def hann_spectrum(seg: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    w = np.hanning(len(seg))
    n_fft = int(2 ** np.ceil(np.log2(len(seg) * ZERO_PAD_FACTOR)))
    spec = np.abs(np.fft.rfft(seg * w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    return freqs, spec


def peak_near(freqs: np.ndarray, spec: np.ndarray, target: float) -> tuple[float, float] | None:
    """Local peak within +/-PEAK_SEARCH_CENTS of target -> (cents offset, amplitude)."""
    lo = target * 2 ** (-PEAK_SEARCH_CENTS / 1200)
    hi = target * 2 ** (PEAK_SEARCH_CENTS / 1200)
    idx = np.where((freqs >= lo) & (freqs <= hi))[0]
    if len(idx) < 3:
        return None
    sub = spec[idx]
    k = int(np.argmax(sub))
    i = idx[k]
    if i <= 0 or i >= len(spec) - 1:
        return None
    # Parabolic interpolation on log-amplitude
    a, b, c = np.log(spec[i - 1] + 1e-12), np.log(spec[i] + 1e-12), np.log(spec[i + 1] + 1e-12)
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    f_peak = freqs[i] + delta * (freqs[1] - freqs[0])
    cents = 1200 * np.log2(f_peak / target)
    return float(cents), float(spec[i])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tx8")
    parser.add_argument("--bp", type=Path, required=True,
                        help="basic_pitch_infer.py output JSON (list of {start,end,midi,amplitude})")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows_doc = json.loads((DRAFTS_DIR / f"{args.tx8}.rows.json").read_text(encoding="utf-8"))
    tx_id = rows_doc["txId"]
    req = json.loads((TX_DIR / tx_id / "request.json").read_text(encoding="utf-8"))
    audio, sr = sf.read(TX_DIR / tx_id / "audio.wav", dtype="float32")
    if audio.ndim > 1:
        # Keep the louder channel (one-sided-silent stereo recordings exist)
        rms = [(float(np.mean(audio[:: max(1, len(audio) // 5000), ch] ** 2)), ch)
               for ch in range(audio.shape[1])]
        audio = audio[:, max(rms)[1]]
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    duration = len(audio) / sr

    tines: dict[str, float] = {}
    for n in sorted(req["tuning"]["notes"], key=lambda n: n["frequency"]):
        tines.setdefault(n["noteName"], float(n["frequency"]))

    midi_to_name = {}
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for m in range(21, 109):
        midi_to_name[m] = f"{names[m % 12]}{m // 12 - 1}"
    bp_notes = [
        {"t": float(n["start"]), "note": midi_to_name.get(int(n["midi"]), f"?{n['midi']}"),
         "amp": round(float(n["amplitude"]), 3)}
        for n in json.loads(args.bp.read_text(encoding="utf-8"))
    ]

    # Noise floor per tine per band-window: median long-window peak amplitude
    # sampled across the recording (robust to the few real onsets per tine).
    floor: dict[str, float] = {}
    win_groups: dict[float, list[str]] = {}
    for name, f in tines.items():
        win_groups.setdefault(band_window_sec(f), []).append(name)
    for win_sec, group in win_groups.items():
        n_win = int(win_sec * sr)
        t = 0.0
        amps: dict[str, list[float]] = {g: [] for g in group}
        while t + win_sec < duration:
            seg = audio[int(t * sr): int(t * sr) + n_win]
            freqs, spec = hann_spectrum(seg, sr)
            for g in group:
                p = peak_near(freqs, spec, tines[g])
                if p is not None:
                    amps[g].append(p[1])
            t += NOISE_FLOOR_STEP_SEC
        for g in group:
            floor[g] = float(np.median(amps[g])) if amps[g] else 1e-9

    onsets = [r["timeSec"] for r in rows_doc["rows"]]
    out_rows = []
    for i, row in enumerate(rows_doc["rows"]):
        t = row["timeSec"]
        t_next = onsets[i + 1] if i + 1 < len(onsets) else duration
        gap = t_next - t
        obs: dict[str, dict] = {}
        for name, f in tines.items():
            # Instrument 2: attack step (candidate nomination only)
            pre_c = t - ATTACK_PRE_SEC
            post_c = t + ATTACK_POST_SEC
            attack_gain = None
            if pre_c - ATTACK_WIN_SEC / 2 >= 0 and post_c + ATTACK_WIN_SEC / 2 <= duration:
                pre = K.note_band_energy(audio, sr, pre_c, f, ATTACK_WIN_SEC, HARMONIC_BAND_CENTS)
                post = K.note_band_energy(audio, sr, post_c, f, ATTACK_WIN_SEC, HARMONIC_BAND_CENTS)
                attack_gain = round((post + 1e-6) / (pre + 1e-6), 2)

            # Instrument 1: pitch identity via clipped long window
            win_sec = band_window_sec(f)
            w_start = t + WINDOW_START_OFFSET_SEC
            w_end = min(w_start + win_sec, t_next - NEXT_ONSET_GUARD_SEC, duration)
            eff = w_end - w_start
            resolvable = eff >= separation_limit_sec(f)
            entry: dict = {"attackGain": attack_gain, "windowMs": round(eff * 1000),
                           "resolvable": bool(resolvable)}
            if eff > 0.02:
                seg = audio[int(w_start * sr): int(w_end * sr)]
                freqs, spec = hann_spectrum(seg, sr)
                p = peak_near(freqs, spec, f)
                if p is not None:
                    cents, amp = p
                    entry["peakCents"] = round(cents, 1)
                    entry["peakOverFloor"] = round(amp / max(floor[name], 1e-9), 1)
            obs[name] = entry

        out_rows.append({
            "index": row["index"],
            "timeSec": t,
            "gapToNextSec": round(gap, 3),
            "draftNotes": row.get("draftNotes") or [],
            "recognized": bool(row.get("recognized")),
            "topCandidates": [
                {"note": c["note"], "share": round(c["share"], 3)}
                for c in (row.get("top") or [])[:5]
            ],
            "bpNear": [b for b in bp_notes if abs(b["t"] - t) <= 0.09],
            "tines": obs,
        })

    out_path = args.out or (DRAFTS_DIR / f"gt-agent-observations-{args.tx8}.json")
    out_path.write_text(json.dumps({
        "txId": tx_id,
        "windowDesign": {
            "bands": "130ms <350Hz / 90ms <600Hz / 50ms >=600Hz (Hann, 4x zero-pad)",
            "clip": "start=onset+20ms, end=nextOnset-10ms",
            "resolvableRule": "effective window >= 1.44 / local semitone spacing",
            "attack": "25ms band-energy step at -35/+35ms (nomination only)",
        },
        "rows": out_rows,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    n_unres = sum(1 for r in out_rows for e in r["tines"].values() if not e["resolvable"])
    print(f"wrote {out_path.relative_to(REPO_ROOT)} ({len(out_rows)} rows; "
          f"unresolvable tine-slots: {n_unres})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
