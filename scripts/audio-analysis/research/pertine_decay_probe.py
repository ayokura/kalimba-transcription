"""Per-tine tracker spike probe (S5 / bets #4 / #141).

Hypothesis: an exponential-decay fit of the note band's energy history
predicts the energy at onset time; a re-stroke shows a large positive
deviation from the prediction, while resonance/bleed does not — because
bleed contaminates the history too, so the fit already "expects" it.

Labelled cases from today's S5 agenda 4 audit:
- REAL: 5 GT-verified un-muted re-strokes (bg rescue targets)
- FP:   4 fixture false rescues (bg rescue casualties)
- RESO: 5 true-resonance rejections (recent-note bg <= 0.93)
"""
import sys, wave
from pathlib import Path
import numpy as np
ROOT = Path("/home/ayokura/kalimba-transcription")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"apps"/"api"))
from app.transcription.peaks import _note_band_energy
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

def deviation(audio, sr, onset, freq,
              hist_start=0.42, hist_end=0.06, hop=0.02, post=0.03):
    """Fit log-energy decay over [onset-hist_start, onset-hist_end],
    predict at attack time, return measured/predicted ratio."""
    ts = np.arange(onset - hist_start, onset - hist_end + 1e-9, hop)
    es = np.array([_note_band_energy(audio, sr, t, freq, window_seconds=0.04) for t in ts])
    mask = es > 1e-3
    if mask.sum() < 4:
        return None, None
    t_fit, e_fit = ts[mask], np.log(es[mask])
    slope, intercept = np.polyfit(t_fit, e_fit, 1)
    t_meas = onset + post
    predicted = np.exp(slope * t_meas + intercept)
    measured = _note_band_energy(audio, sr, t_meas, freq, window_seconds=0.04)
    return measured / (predicted + 1e-9), slope

CASES = [
 # label, dir, time, note
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
audio_cache = {}
print(f"{'label':6s} {'time':>8s} {'note':4s} {'dev_ratio':>10s} {'decay_slope':>12s}")
for label, rel, t, note in CASES:
    if rel not in audio_cache:
        audio_cache[rel] = load(rel)
    audio, sr = audio_cache[rel]
    dev, slope = deviation(audio, sr, t, FREQS[note])
    dev_s = f"{dev:10.2f}" if dev is not None else "      None"
    sl_s = f"{slope:12.2f}" if slope is not None else "        None"
    print(f"{label:6s} {t:8.3f} {note:4s} {dev_s} {sl_s}")
