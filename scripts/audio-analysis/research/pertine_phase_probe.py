"""Per-tine tracker phase probe: does a re-stroke show phase discontinuity
in the note's narrow band while resonance stays phase-continuous?

Method: narrow bandpass (+-2.5%) -> Hilbert analytic signal -> unwrap phase.
Fit instantaneous frequency on [onset-0.12, onset-0.02], extrapolate to
[onset+0.02, onset+0.08], measure RMS phase error (radians).  Also measure
amplitude-envelope jerk (max positive d(env)/dt near onset, normalized).
"""
import sys, wave
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
ROOT = Path("/home/ayokura/kalimba-transcription")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"apps"/"api"))
from app.tunings import get_default_tunings

def load(rel):
    with wave.open(str(ROOT/rel/"audio.wav"), "rb") as w:
        sr = w.getframerate(); raw = w.readframes(w.getnframes())
        ch = w.getnchannels(); width = w.getsampwidth()
    dt = {1:np.int8,2:np.int16,4:np.int32}[width]
    d = np.frombuffer(raw, dtype=dt).astype(np.float64)
    if ch>1: d = d.reshape(-1,ch).mean(axis=1)
    return d/(np.max(np.abs(d)) or 1.0), sr

FREQS = {n.note_name: n.frequency for n in get_default_tunings()[0].notes}

def phase_probe(audio, sr, onset, freq):
    lo, hi = freq*0.975, freq*1.025
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    a, b = int((onset-0.30)*sr), int((onset+0.15)*sr)
    if a < 0 or b > len(audio): return None
    seg = sosfiltfilt(sos, audio[a:b])
    analytic = hilbert(seg)
    phase = np.unwrap(np.angle(analytic))
    env = np.abs(analytic)
    t = np.arange(len(seg))/sr + (onset-0.30)
    # fit phase slope pre-onset
    pre = (t >= onset-0.12) & (t <= onset-0.02)
    post = (t >= onset+0.02) & (t <= onset+0.08)
    if pre.sum() < 10 or post.sum() < 10: return None
    coef = np.polyfit(t[pre], phase[pre], 1)
    pred = np.polyval(coef, t[post])
    err = phase[post] - pred
    rms_err = float(np.sqrt(np.mean(err**2)))
    # amplitude jerk: max positive derivative of env in [onset-0.01, onset+0.05]
    w = (t >= onset-0.01) & (t <= onset+0.05)
    denv = np.diff(env[w]) * sr
    pre_env = np.median(env[pre]) + 1e-9
    jerk = float(np.max(denv) / pre_env) if len(denv) else 0.0
    return rms_err, jerk

CASES = [
 ("REAL", "data/transactions/4e1ae5c6-df9a-4876-917d-b7e47699c8e5", 8.197, "C5"),
 ("REAL", "data/transactions/4e1ae5c6-df9a-4876-917d-b7e47699c8e5", 16.923, "C5"),
 ("REAL", "data/transactions/9ce7df83-33a0-455d-bf86-c9392ce6f777", 12.459, "C5"),
 ("REAL", "data/transactions/9ce7df83-33a0-455d-bf86-c9392ce6f777", 16.811, "C5"),
 ("REAL", "apps/api/tests/fixtures/free-performance-corpus/17ea7626-3c5d-450d-ae74-0116dea6e881", 11.547, "C5"),
 ("FP",   "apps/api/tests/fixtures/manual-captures/kalimba-17-c-bwv147-sequence-163-01", 80.819, "F5"),
 ("FP",   "apps/api/tests/fixtures/manual-captures/kalimba-17-c-bwv147-sequence-163-01", 191.653, "C6"),
 ("FP",   "apps/api/tests/fixtures/manual-captures/kalimba-17-c-c4-to-g4-sequence-17-01", 12.667, "E5"),
 ("FP",   "apps/api/tests/fixtures/manual-captures/kalimba-17-c-c4-to-g4-sequence-17-01", 13.448, "C5"),
 ("RESO", "data/transactions/4e1ae5c6-df9a-4876-917d-b7e47699c8e5", 18.475, "C5"),
 ("RESO", "data/transactions/4e1ae5c6-df9a-4876-917d-b7e47699c8e5", 18.800, "C5"),
 ("RESO", "data/transactions/4e1ae5c6-df9a-4876-917d-b7e47699c8e5", 19.232, "C5"),
 ("RESO", "apps/api/tests/fixtures/free-performance-corpus/d7a82772-f77f-4820-9798-00133ae45f4e", 4.176, "F5"),
 ("RESO", "apps/api/tests/fixtures/free-performance-corpus/d7a82772-f77f-4820-9798-00133ae45f4e", 7.589, "A4"),
]
cache = {}
print(f"{'label':6s} {'time':>8s} {'note':4s} {'phase_rms(rad)':>14s} {'amp_jerk(/s)':>12s}")
for label, rel, t, note in CASES:
    if rel not in cache: cache[rel] = load(rel)
    audio, sr = cache[rel]
    r = phase_probe(audio, sr, t, FREQS[note])
    if r is None:
        print(f"{label:6s} {t:8.3f} {note:4s} {'None':>14s}")
        continue
    print(f"{label:6s} {t:8.3f} {note:4s} {r[0]:14.2f} {r[1]:12.1f}")
