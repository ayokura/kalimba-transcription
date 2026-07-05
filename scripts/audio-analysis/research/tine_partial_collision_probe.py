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


def request_for(tx: str) -> dict | None:
    for base, _ in GT_SOURCES:
        q = REPO / base / tx / "request.json"
        if q.is_file():
            return json.loads(q.read_text())
    q = DATA_DIR / tx / "request.json"
    return json.loads(q.read_text()) if q.is_file() else None


# Tester-performed recordings confirmed by the user (2026-07-05). Recordings
# predating f617f8c carry no client metadata, so UA inference alone cannot
# separate performers — this explicit list is authoritative.
KNOWN_TESTER = {
    "17ea7626-3c5d-450d-ae74-0116dea6e881",
    "47902d34-95f4-4761-a634-6c1ef154531f",
    "70cc6637-ca99-4848-a0d4-944b53e5c742",
    "d7a82772-f77f-4820-9798-00133ae45f4e",
    "a9e30986-5300-4401-8b69-152cba821042",
    "1955b5bd-ee2e-41b6-a81e-6a1ab1ac22ec",
    "98019f67-690d-4deb-a882-98ed9878c519",
}


def resolve_source(tx: str) -> tuple[str, str, dict[str, float]] | None:
    """(tuning_id, group_label, note->freq) for a tx, or None if unresolvable.

    Group = tuning x performer. KNOWN_TESTER is authoritative; for newer
    recordings the client metadata (micLabel/userAgent, f617f8c) is used:
    "VBMatrix" mic or Windows UA => author, iPhone/iPad UA => tester.
    """
    req = request_for(tx)
    if req is None:
        return None
    tuning = req["tuning"]["id"]
    freqs = {n["noteName"]: float(n["frequency"]) for n in req["tuning"]["notes"]}
    client = req.get("client") or {}
    mic = (client.get("device") or {}).get("micLabel", "")
    ua = client.get("userAgent", "")
    if tx in KNOWN_TESTER:
        who = "tester"
    elif "VBMatrix" in mic or "Windows" in ua:
        who = "author"
    elif "iPhone" in ua or "iPad" in ua:
        who = "tester"
    elif not client:
        who = "author"  # pre-f617f8c captures not in KNOWN_TESTER are the author's
    else:
        who = "unknown"
    return tuning, f"{tuning}|{who}", freqs


def window_spectrum(audio: np.ndarray, sr: int, start_sec: float) -> tuple[np.ndarray, np.ndarray] | None:
    a = int(start_sec * sr)
    b = a + int(BODY_WINDOW * sr)
    if a < 0 or b > len(audio):
        return None
    seg = audio[a:b] * np.hanning(b - a)
    spec = np.abs(np.fft.rfft(seg, n=1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1.0 / sr)
    return freqs, spec


def body_spectrum(audio: np.ndarray, sr: int, t: float) -> tuple[np.ndarray, np.ndarray] | None:
    return window_spectrum(audio, sr, t + BODY_OFFSET)


def pre_spectrum(audio: np.ndarray, sr: int, t: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Same-length window ending just before the onset — captures the ring-out
    of earlier notes so their decay can be subtracted from contamination.
    (Kalimba tines ring for seconds; ISOLATION_SEC alone cannot exclude them.)"""
    return window_spectrum(audio, sr, t - 0.01 - BODY_WINDOW)


def peak_near(freqs: np.ndarray, spec: np.ndarray, f: float, cents: float = 60.0) -> float:
    lo, hi = f * 2 ** (-cents / 1200), f * 2 ** (cents / 1200)
    m = (freqs >= lo) & (freqs <= hi)
    return float(spec[m].max()) if m.any() else 0.0


def analyze_group(group: str, freqs_map: dict[str, float],
                  events: list[tuple[str, np.ndarray, int, float, str]]) -> None:
    """events: (tx, audio, sr, t, note) isolated single-note strikes."""
    partials: dict[str, list[list[float]]] = {}   # note -> list of ratio lists
    contamination: dict[tuple[str, str], list[float]] = {}  # (striker, victim) -> ratios
    fresh: dict[tuple[str, str], list[float]] = {}  # ring-out-subtracted contamination
    for tx, audio, sr, t, note in events:
        fs = body_spectrum(audio, sr, t)
        if fs is None:
            continue
        fr, spec = fs
        ps = pre_spectrum(audio, sr, t)
        pre = ps[1] if ps is not None else None
        f0 = freqs_map[note]
        e0 = peak_near(fr, spec, f0)
        if e0 <= 0:
            continue
        # partial peaks: local maxima in the band above MIN_PEAK_REL
        lo, hi = f0 * PARTIAL_BAND[0], f0 * PARTIAL_BAND[1]
        m = (fr >= lo) & (fr <= hi)
        band_f, band_s = fr[m], spec[m]
        is_pk = np.zeros(len(band_s), bool)
        is_pk[1:-1] = (band_s[1:-1] > band_s[:-2]) & (band_s[1:-1] >= band_s[2:])
        strong = sorted(
            [(float(bs), float(bf / f0)) for bf, bs in zip(band_f[is_pk], band_s[is_pk]) if bs >= e0 * MIN_PEAK_REL],
            reverse=True,
        )[:4]
        partials.setdefault(note, []).append([r for _, r in strong])
        # contamination: striker's energy at every other tine fundamental.
        # raw = body-window energy (includes earlier notes' ring-out);
        # fresh = raw minus the pre-onset level (attributable to this strike).
        for other, fo in freqs_map.items():
            if other == note:
                continue
            eo = peak_near(fr, spec, fo)
            contamination.setdefault((note, other), []).append(eo / e0 if e0 else 0.0)
            if pre is not None:
                ep = peak_near(fr, pre, fo)
                fresh.setdefault((note, other), []).append(max(eo - ep, 0.0) / e0 if e0 else 0.0)

    n_events = len(events)
    print(f"\n######## group {group}: isolated events={n_events} ########")
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
                    fr_ = fresh.get((note, other), [])
                    med_fresh = float(np.median(fr_)) if fr_ else float("nan")
                    collisions.append((note, r, other, float(cents), med_cont))
                    print(f"  {note:4s} partial x{r:.3f} -> {other:4s} ({cents:+.0f}c)  contamination raw={med_cont:.3f} fresh={med_fresh:.3f}")
    if not collisions:
        print("  (none)")

    print("\n=== contamination extremes (top 10 pairs by median raw ratio; fresh = ring-out subtracted) ===")
    rows = [(k, float(np.median(v)), len(v)) for k, v in contamination.items() if len(v) >= MIN_EVENTS_PER_TINE]
    rows.sort(key=lambda x: -x[1])
    for (a, b), med, n in rows[:10]:
        fr_ = fresh.get((a, b), [])
        med_fresh = float(np.median(fr_)) if fr_ else float("nan")
        print(f"  {a:4s} -> {b:4s}: raw {med:.3f} fresh {med_fresh:.3f} (n={n})")


def main() -> int:
    gt = collect_gt()
    # bucket isolated single-note events per instrument group (tuning x performer)
    groups: dict[str, tuple[dict[str, float], list]] = {}
    skipped: list[str] = []
    for tx, onsets in gt.items():
        src = resolve_source(tx)
        ap = audio_for(tx)
        if src is None or ap is None:
            skipped.append(tx)
            continue
        _tuning, group, freqs_map = src
        audio, sr = load_audio(ap)
        times = [t for t, _ in onsets]
        bucket = groups.setdefault(group, (freqs_map, []))[1]
        for i, (t, notes) in enumerate(onsets):
            if len(notes) != 1 or notes[0] not in freqs_map:
                continue
            prev_gap = t - times[i - 1] if i > 0 else 99
            next_gap = times[i + 1] - t if i + 1 < len(times) else 99
            if prev_gap < ISOLATION_SEC or next_gap < ISOLATION_SEC:
                continue
            bucket.append((tx, audio, sr, t, notes[0]))
    if skipped:
        print(f"(skipped, no request.json/audio: {', '.join(s[:8] for s in skipped)})")
    for group in sorted(groups, key=lambda g: -len(groups[g][1])):
        freqs_map, events = groups[group]
        txs = sorted({e[0][:8] for e in events})
        print(f"\n[group {group}] recordings: {', '.join(txs)}")
        analyze_group(group, freqs_map, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
