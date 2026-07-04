"""Phase-tracking onset detector probe (per-tine tracker go/no-go, #141).

Continuously tracks each tine's narrow-band complex envelope (heterodyne +
low-pass). Detection quantity per tine per hop:
  - phase_err: RMS deviation of the unwrapped phase from a linear fit of the
    preceding 100 ms (a re-struck tine breaks its own phase trend)
  - jerk: positive envelope derivative normalized by recent envelope level

An onset is declared where phase_err and jerk jointly exceed the bars
calibrated on 2026-07-05 labelled instants, with non-max suppression and a
near-frequency dominance guard (bleed rejection, ratio 5).

Evaluation (no GT yet for the Twinkle takes):
  - standard take 70cc6637: broadband found 147 onsets ~ 1:1 with strokes →
    sanity reference (agreement + spurious rate)
  - fast take 47902d34: broadband collapsed to 39; same piece → expect ~140+
    real strokes. How many does phase tracking recover?
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

ROOT = Path("/home/ayokura/kalimba-transcription")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "apps" / "api"))

HOP_SEC = 0.005          # 5 ms tracking grid
PAST_SEC = 0.10          # phase-fit window
FUT_SEC = 0.06           # extrapolation window
PHASE_BAR = 0.7          # rad, from labelled-instant calibration
JERK_BAR = 50.0          # /s, same
MIN_SEP_SEC = 0.06       # per-tine non-max suppression
DOM_RATIO = 5.0          # near-frequency bleed guard
DOM_CENTS = 350.0        # "near" = within ~3 semitones
ENV_GATE_FRAC = 0.02     # tine must reach 2% of its own peak env to count


def load(tx: str) -> tuple[np.ndarray, int]:
    with wave.open(str(ROOT / "data/transactions" / tx / "audio.wav"), "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
        ch = w.getnchannels(); width = w.getsampwidth()
    dt = {2: np.int16, 4: np.int32}[width]
    d = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d / (np.max(np.abs(d)) or 1.0), sr


def tine_freqs(tx: str) -> list[tuple[str, float]]:
    req = json.loads((ROOT / "data/transactions" / tx / "request.json").read_text())
    return [(n["noteName"], float(n["frequency"])) for n in req["tuning"]["notes"]]


def demodulate(audio: np.ndarray, sr: int, freq: float, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Heterodyne to baseband, causal low-pass, decimate to hop grid.
    Returns (envelope, unwrapped_phase) per hop."""
    t = np.arange(len(audio)) / sr
    z = audio * np.exp(-2j * np.pi * freq * t)
    # causal low-pass ~ half-bandwidth 2.5% of tine freq, floor 12 Hz
    bw = max(freq * 0.025, 12.0)
    sos = butter(3, bw, btype="low", fs=sr, output="sos")
    zf = sosfilt(sos, z.real) + 1j * sosfilt(sos, z.imag)
    zh = zf[::hop]
    return np.abs(zh), np.unwrap(np.angle(zh))


SUSTAIN_FROM_SEC = 0.10
SUSTAIN_TO_SEC = 0.25
SUSTAIN_MIN_FRAC = 0.10


def detect_track(env: np.ndarray, phase: np.ndarray, hop_sec: float, abs_gate: float) -> list[tuple[int, float, float]]:
    """Scan one tine's track. Returns [(hop_idx, phase_err, jerk)] raw hits.

    abs_gate: absolute envelope floor shared across tines — a per-tine
    relative gate lets never-struck tines fire on broadband attack
    transients (their own peak IS the transient).
    A genuine kalimba stroke also RINGS: require the envelope 100-250 ms
    after the hit to hold >=10% of the attack peak; filter transients
    (impulse response of the narrow band-pass) decay much faster.
    """
    n = len(env)
    past = int(PAST_SEC / hop_sec)
    fut = int(FUT_SEC / hop_sec)
    s_from = int(SUSTAIN_FROM_SEC / hop_sec)
    s_to = int(SUSTAIN_TO_SEC / hop_sec)
    hits: list[tuple[int, float, float]] = []
    tt = np.arange(n) * hop_sec
    for i in range(past + 2, n - s_to - 1):
        seg = env[i - 2 : i + fut]
        denv = np.diff(seg) / hop_sec
        ref = np.median(env[i - past : i - 2]) + 1e-9
        jerk = float(denv.max() / ref) if len(denv) else 0.0
        if jerk < JERK_BAR:
            continue
        attack_peak = env[i : i + fut].max()
        if attack_peak < abs_gate:
            continue
        sustain = np.median(env[i + s_from : i + s_to])
        if sustain < attack_peak * SUSTAIN_MIN_FRAC:
            continue  # rings like a filter transient, not a tine
        p_t = tt[i - past : i - 2]
        p_y = phase[i - past : i - 2]
        coef = np.polyfit(p_t, p_y, 1)
        pred = np.polyval(coef, tt[i + 1 : i + fut])
        err = float(np.sqrt(np.mean((phase[i + 1 : i + fut] - pred) ** 2)))
        if err < PHASE_BAR:
            continue
        hits.append((i, err, jerk))
    return hits


def nms(hits: list[tuple[int, float, float]], hop_sec: float) -> list[tuple[int, float, float]]:
    """Per-tine non-max suppression by jerk within MIN_SEP."""
    if not hits:
        return []
    min_sep = int(MIN_SEP_SEC / hop_sec)
    hits = sorted(hits, key=lambda h: -h[2])
    kept: list[tuple[int, float, float]] = []
    taken: list[int] = []
    for h in hits:
        if all(abs(h[0] - t) >= min_sep for t in taken):
            kept.append(h)
            taken.append(h[0])
    return sorted(kept)


def run(tx: str, label: str, reference_onsets: list[float] | None):
    audio, sr = load(tx)
    hop = int(sr * HOP_SEC)
    hop_sec = hop / sr
    notes = tine_freqs(tx)
    tracks = []
    for name, freq in notes:
        env, phase = demodulate(audio, sr, freq, hop)
        tracks.append((name, freq, env, phase))
    global_peak = max(t[2].max() for t in tracks) or 1.0
    abs_gate = global_peak * ENV_GATE_FRAC
    events: list[tuple[float, str, float, float, float]] = []  # (time, note, err, jerk, freq)
    for name, freq, env, phase in tracks:
        for i, err, jerk in nms(detect_track(env, phase, hop_sec, abs_gate), hop_sec):
            events.append((i * hop_sec, name, err, jerk, freq))
    events.sort()
    # near-frequency dominance guard (bleed): drop event if a >=5x-jerk event
    # exists within ±30 ms on a tine within ±350 cents
    kept = []
    for e in events:
        t, name, err, jerk, freq = e
        # adjacent-frequency exclusivity: within ±250 cents a stroke's
        # spectral skirt leaks into the neighbour band (3rd-order LPF skirt
        # keeps ringing as long as the true note rings, so the sustain test
        # cannot reject it).  Winner-takes-all by jerk among neighbours.
        dominated = any(
            abs(o[0] - t) <= 0.03
            and o[1] != name
            and (
                (abs(1200 * np.log2(o[4] / freq)) <= 250 and o[3] > jerk)
                or (
                    abs(1200 * np.log2(o[4] / freq)) <= DOM_CENTS
                    and o[3] >= jerk * DOM_RATIO
                )
            )
            for o in events
        )
        if dominated:
            continue
        # harmonic-parent suppression: a fresh stroke floods its own
        # 2nd/3rd/4th partial bands, resetting THEIR phase too.  If a
        # candidate fundamental exists at ~1/2, 1/3 or 1/4 of this tine's
        # frequency within ±30 ms, treat this event as partial bleed of
        # that stroke (octave-dyad simultaneities are the accepted loss).
        harmonic_parent = any(
            abs(o[0] - t) <= 0.03
            and o[1] != name
            and any(
                abs(1200 * np.log2(freq / (o[4] * m))) <= 50
                for m in (2.0, 3.0, 4.0)
            )
            for o in events
        )
        if not harmonic_parent:
            kept.append(e)
    print(f"=== {label} ({tx[:8]}) ===")
    print(f" phase-tracking onsets: {len(kept)}  (raw before bleed-guard: {len(events)})")
    if reference_onsets:
        ref = np.asarray(reference_onsets)
        matched_ref = sum(1 for r in ref if any(abs(k[0] - r) <= 0.06 for k in kept))
        novel = sum(1 for k in kept if not any(abs(k[0] - r) <= 0.06 for r in ref))
        print(f" vs broadband({len(ref)}): agree={matched_ref}  phase-only(new)={novel}  broadband-only={len(ref)-matched_ref}")
    per_note: dict[str, int] = {}
    for k in kept:
        per_note[k[1]] = per_note.get(k[1], 0) + 1
    top = sorted(per_note.items(), key=lambda kv: -kv[1])[:8]
    print(f" per-note top: {top}")
    return kept


def broadband_onsets(tx: str) -> list[float]:
    d = json.loads((ROOT / "data/transactions" / tx / "debug.json").read_text())
    return sorted(d.get("gapValidatedOnsetTimes") or [])


std = run("70cc6637-ca99-4848-a0d4-944b53e5c742", "standard take", broadband_onsets("70cc6637-ca99-4848-a0d4-944b53e5c742"))
fast = run("47902d34-95f4-4761-a634-6c1ef154531f", "fast take", broadband_onsets("47902d34-95f4-4761-a634-6c1ef154531f"))
out = Path(__file__).parent / "phase_tracking_results.json"
out.write_text(json.dumps({
    "standard": [[round(t, 3), n, round(e, 2), round(j, 0)] for t, n, e, j, _ in std],
    "fast": [[round(t, 3), n, round(e, 2), round(j, 0)] for t, n, e, j, _ in fast],
}, ensure_ascii=False))
print("saved:", out)
