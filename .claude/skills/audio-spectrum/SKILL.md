---
name: audio-spectrum
description: Analyze spectral characteristics using librosa (BW90, centroid, spread)
user_invocable: true
arguments:
  - name: file
    description: Audio file path (fixture name or full path)
    required: true
  - name: time
    description: Onset time to analyze in seconds
    required: true
  - name: window
    description: Analysis window in ms (default 15)
    required: false
---

<command-name>audio-spectrum</command-name>

Analyze spectral characteristics at a specific onset time using the spectrum_stats.py script.

## Arguments
- `file`: Audio file path or fixture name
- `time`: Onset time to analyze (seconds)
- `window`: Analysis window duration in ms (default: 15)

## Instructions

1. Resolve the file path (fixture name to full path if needed)

2. Run the analysis script:
```bash
PYTHONPATH=apps/api uv run python3 scripts/audio-analysis/spectrum_stats.py <file> <time> [<window>]
```

3. Report results with interpretation (see guide below)

## Computed Features
- **BW90**: 90% energy bandwidth (Hz) - how spread out the frequency content is
- **Centroid**: Spectral center of mass (Hz) - "brightness" of the sound
- **Spread**: Standard deviation from centroid (Hz)
- **HF Ratio**: Energy above 2kHz / total energy
- **VHF Ratio**: Energy above 8kHz / total energy
- **Peak Frequency**: Dominant frequency
- **Classification**: KALIMBA / NOISE / UNCLEAR (auto-determined by script)

## Interpretation Guide

| Feature | Noise | Real Kalimba Note |
|---------|-------|-------------------|
| BW90 | >6000 Hz (broadband) | <2000 Hz (narrow) |
| Centroid | >3000 Hz | <1500 Hz |
| HF Ratio | >30% | <10% |
| VHF Ratio | >2% | <1% |
| Peak Freq | Often high or erratic | Near kalimba fundamental |

These thresholds are validated against real fixture data (2026-03-30 audit):
- c4-c5-dyad @ 1.44s: BW90=515, centroid=390 → KALIMBA (genuine, score=0.1)
- c4-e4-g4-triad @ 0.06s: BW90=11531, centroid=3858 → NOISE
- e6-to-c4-sequence @ 1.10s: BW90=10359, centroid=3974 → NOISE

## Example Usage
```
/audio-spectrum d5-repeat-01 0.059
/audio-spectrum c4-e4-g4-triad-repeat-01 0.056 20
/audio-spectrum /path/to/audio.wav 0.1 30
```

## Notes
- Combine with `/audio-onset` to get onset times first
- Use `/audio-visualize` for visual confirmation
- Short windows (10-20ms) capture attack characteristics
- Longer windows (50-100ms) show sustained tone characteristics
