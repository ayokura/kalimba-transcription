"""Run PESTO (self-supervised monophonic pitch tracker) on a wav and emit
note events as JSON (stdout).

Runs in an ISOLATED Python 3.11 environment (torch dependency; external AMT
is a dev instrument only — guardrail 3, never prod/distribution).  Invoke:

    uv run --python 3.11 --no-project --with pesto-pitch --with torchaudio \
        python scripts/audio-analysis/research/pesto_infer.py <audio.wav>

Part of the S6 bets #3 blind-spot overlap measurement (#203 rules fixed
2026-07-06).  PESTO is frame-level and MONOPHONIC: on polyphonic kalimba it
tracks the locally dominant pitch, so its note events are a second opinion
on the melody stream, not a full transcription.  Frame->note segmentation:
consecutive frames (10 ms step) whose pitch rounds to the same MIDI note,
confidence-gated, minimum duration, small gap bridging.

Output: [{"start": s, "end": s, "midi": int, "confidence": f}, ...]
"""
from __future__ import annotations

import json
import sys

STEP_MS = 10.0
CONF_MIN = 0.5      # frame admission; events also report their mean conf
MIN_DUR_SEC = 0.04  # >= 4 frames
GAP_BRIDGE_SEC = 0.03


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: pesto_infer.py <audio.wav>", file=sys.stderr)
        return 2
    import numpy as np
    import torch
    import torchaudio

    x, sr = torchaudio.load(sys.argv[1])
    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    x = x.squeeze(0)
    with torch.inference_mode():
        timesteps, pitch_hz, confidence, _act = __import__("pesto").predict(
            x, int(sr), step_size=STEP_MS)
    t = timesteps.cpu().numpy() / 1000.0  # ms -> s
    f0 = pitch_hz.cpu().numpy()
    conf = confidence.cpu().numpy()

    midi = np.full(len(f0), -1, dtype=int)
    ok = (conf >= CONF_MIN) & (f0 > 0) & np.isfinite(f0)
    midi[ok] = np.round(69 + 12 * np.log2(f0[ok] / 440.0)).astype(int)

    rows = []
    i = 0
    gap_frames = int(GAP_BRIDGE_SEC * 1000 / STEP_MS)
    while i < len(midi):
        if midi[i] < 0:
            i += 1
            continue
        j = i + 1
        last_valid = i
        while j < len(midi):
            if midi[j] == midi[i]:
                last_valid = j
            elif midi[j] >= 0 or (j - last_valid) > gap_frames:
                break
            j += 1
        start, end = float(t[i]), float(t[last_valid])
        if end - start >= MIN_DUR_SEC:
            seg_conf = conf[i:last_valid + 1][midi[i:last_valid + 1] == midi[i]]
            rows.append({
                "start": round(start, 4), "end": round(end, 4),
                "midi": int(midi[i]),
                "confidence": round(float(seg_conf.mean()), 4),
            })
        i = last_valid + 1
    json.dump(rows, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
