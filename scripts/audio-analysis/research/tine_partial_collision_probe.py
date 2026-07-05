"""#149 collision probe (S0, third-term plan / per-tine GO condition G1).

Question: do measured per-tine partials land on other tines' fundamental
bands closely enough that a per-tine partial table would self-poison
(mis-attribute partial energy as the neighbour's fundamental)?

Method:
1. Collect isolated single-note GT events (no other GT event within
   +-ISOLATION_SEC) from GT'd recordings.
2. For each, FFT a body window and pick spectral peaks in the 1.2x-4.4x
   band of the played note's fundamental.
3. Aggregate per tine: median partial ratios (the measured, non-integer
   partial structure).
4. Collision map: measured partial vs every tine fundamental, flag pairs
   within COLLISION_CENTS.
5. For flagged pairs, quantify contamination: energy at the victim tine's
   fundamental band while only the striker sounds, relative to the
   striker's own fundamental (contamination ratio). High ratio + collision
   = the pair a tracker must explain away.

Usage: uv run python scripts/audio-analysis/research/tine_partial_collision_probe.py
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps" / "api"))

ISOLATION_SEC = 0.35
BODY_OFFSET = 0.05
BODY_WINDOW = 0.20
PARTIAL_BAND = (1.2, 4.4)   # x fundamental
COLLISION_CENTS = 50.0
MIN_EVENTS_PER_TINE = 3
MIN_PEAK_REL = 0.02         # peak must be >=2% of fundamental energy

GT_SOURCES = [
    ("apps/api/tests/fixtures/free-performance-corpus", True),
    ("apps/api/tests/fixtures/transaction-captures", False),
]
DATA_DIR = REPO / "data" / "transactions"


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
        ch = w.getnchannels(); width = w.getsampwidth()
    dt = {2: np.int16, 4: np.int32}[width]
    d = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d / (np.max(np.abs(d)) or 1.0), sr


def audio_for(tx: str) -> Path | None:
    for base, _ in GT_SOURCES:
        p = REPO / base / tx / "audio.wav"
        if p.is_file():
            return p
    p = DATA_DIR / tx / "audio.wav"
    return p if p.is_file() else None


def collect_gt() -> dict[str, list[tuple[float, list[str]]]]:
    out: dict[str, list[tuple[float, list[str]]]] = {}
    for base, _ in GT_SOURCES:
        for d in sorted((REPO / base).iterdir()):
            gt = d / "ground_truth.json"
            if not gt.is_file() or d.name in out:
                continue
            j = json.loads(gt.read_text())
            onsets = [(float(o["timeSec"]), o.get("notes") or [o.get("note")]) for o in j["onsets"]]
            out[d.name] = onsets
    return out


def note_freqs_17c() -> dict[str, float]:
    # any 17-C request.json carries the tuning table
    for tx in collect_gt():
        p = None
        for base, _ in GT_SOURCES:
            q = REPO / base / tx / "request.json"
            if q.is_file():
                p = q; break
        if p is None:
            q = DATA_DIR / tx / "request.json"
            p = q if q.is_file() else None
        if p is None:
            continue
        req = json.loads(p.read_text())
        if req["tuning"]["id"].startswith("kalimba-17-c"):
            return {n["noteName"]: float(n["frequency"]) for n in req["tuning"]["notes"]}
    raise RuntimeError("no 17-C tuning found")


def body_spectrum(audio: np.ndarray, sr: int, t: float) -> tuple[np.ndarray, np.ndarray] | None:
    a = int((t + BODY_OFFSET) * sr)
    b = a + int(BODY_WINDOW * sr)
    if a < 0 or b > len(audio):
        return None
    seg = audio[a:b] * np.hanning(b - a)
    spec = np.abs(np.fft.rfft(seg, n=1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1.0 / sr)
    return freqs, spec


def peak_near(freqs: np.ndarray, spec: np.ndarray, f: float, cents: float = 60.0) -> float:
    lo, hi = f * 2 ** (-cents / 1200), f * 2 ** (cents / 1200)
    m = (freqs >= lo) & (freqs <= hi)
    return float(spec[m].max()) if m.any() else 0.0


def main() -> int:
    freqs_map = note_freqs_17c()
    gt = collect_gt()
    # 1-2. isolated single-note events -> partial ratios per tine
    partials: dict[str, list[list[float]]] = {}   # note -> list of ratio lists
    contamination: dict[tuple[str, str], list[float]] = {}  # (striker, victim) -> ratios
    for tx, onsets in gt.items():
        ap = audio_for(tx)
        if ap is None:
            continue
        audio, sr = load_audio(ap)
        times = [t for t, _ in onsets]
        for i, (t, notes) in enumerate(onsets):
            if len(notes) != 1 or notes[0] not in freqs_map:
                continue
            prev_gap = t - times[i - 1] if i > 0 else 99
            next_gap = times[i + 1] - t if i + 1 < len(times) else 99
            if prev_gap < ISOLATION_SEC or next_gap < ISOLATION_SEC:
                continue
            fs = body_spectrum(audio, sr, t)
            if fs is None:
                continue
            fr, spec = fs
            f0 = freqs_map[notes[0]]
            e0 = peak_near(fr, spec, f0)
            if e0 <= 0:
                continue
            # partial peaks: local maxima in the band above MIN_PEAK_REL
            lo, hi = f0 * PARTIAL_BAND[0], f0 * PARTIAL_BAND[1]
            m = (fr >= lo) & (fr <= hi)
            band_f, band_s = fr[m], spec[m]
            is_pk = np.zeros(len(band_s), bool)
            is_pk[1:-1] = (band_s[1:-1] > band_s[:-2]) & (band_s[1:-1] >= band_s[2:])
            ratios = [float(bf / f0) for bf, bs in zip(band_f[is_pk], band_s[is_pk]) if bs >= e0 * MIN_PEAK_REL]
            # keep the strongest few distinct ratios
            strong = sorted(
                [(float(bs), float(bf / f0)) for bf, bs in zip(band_f[is_pk], band_s[is_pk]) if bs >= e0 * MIN_PEAK_REL],
                reverse=True,
            )[:4]
            partials.setdefault(notes[0], []).append([r for _, r in strong])
            # 5. contamination: striker's energy at every other tine fundamental
            for other, fo in freqs_map.items():
                if other == notes[0]:
                    continue
                eo = peak_near(fr, spec, fo)
                contamination.setdefault((notes[0], other), []).append(eo / e0 if e0 else 0.0)

    # 3. aggregate per tine
    print("=== measured per-tine partials (median of strongest ratios, n>=%d) ===" % MIN_EVENTS_PER_TINE)
    table: dict[str, list[float]] = {}
    for note in sorted(partials, key=lambda n: freqs_map[n]):
        samples = partials[note]
        if len(samples) < MIN_EVENTS_PER_TINE:
            continue
        flat = [r for lst in samples for r in lst]
        if not flat:
            continue
        # cluster ratios coarsely (0.05 bins) and take medians of populated clusters
        flat.sort()
        clusters: list[list[float]] = [[flat[0]]]
        for r in flat[1:]:
            if r - clusters[-1][-1] <= 0.05:
                clusters[-1].append(r)
            else:
                clusters.append([r])
        meds = [round(float(np.median(c)), 3) for c in clusters if len(c) >= max(2, len(samples) // 3)]
        table[note] = meds
        print(f"  {note:4s} (n={len(samples):2d}): ratios={meds}")

    # 4. collision map
    print("\n=== collision map: measured partial within +-%.0f cents of another tine fundamental ===" % COLLISION_CENTS)
    collisions = []
    for note, meds in table.items():
        f0 = freqs_map[note]
        for r in meds:
            pf = f0 * r
            for other, fo in freqs_map.items():
                if other == note:
                    continue
                cents = 1200 * np.log2(pf / fo)
                if abs(cents) <= COLLISION_CENTS:
                    cont = contamination.get((note, other), [])
                    med_cont = float(np.median(cont)) if cont else float("nan")
                    collisions.append((note, r, other, float(cents), med_cont))
                    print(f"  {note:4s} partial x{r:.3f} -> {other:4s} ({cents:+.0f}c)  contamination(median E_victim/E_striker)={med_cont:.3f}")
    if not collisions:
        print("  (none)")

    # summary judgement inputs
    print("\n=== contamination extremes (top 8 pairs by median ratio, regardless of collision) ===")
    rows = [(k, float(np.median(v)), len(v)) for k, v in contamination.items() if len(v) >= MIN_EVENTS_PER_TINE]
    rows.sort(key=lambda x: -x[1])
    for (a, b), med, n in rows[:8]:
        print(f"  {a:4s} -> {b:4s}: median {med:.3f} (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
