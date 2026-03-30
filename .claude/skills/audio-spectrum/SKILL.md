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

Analyze spectral characteristics at a specific onset time using librosa.

## Arguments
- `file`: Audio file path or fixture name
- `time`: Onset time to analyze (seconds)
- `window`: Analysis window duration (default: 15ms)

## Computed Features
- **BW90**: 90% energy bandwidth (Hz) - how spread out the frequency content is
- **Centroid**: Spectral center of mass (Hz) - "brightness" of the sound
- **Spread**: Standard deviation from centroid (Hz)
- **HF Ratio**: Energy above 2kHz / total energy
- **VHF Ratio**: Energy above 8kHz / total energy
- **Peak Frequency**: Dominant frequency

## Interpretation Guide
| Feature | Noise | Real Kalimba Note |
|---------|-------|-------------------|
| BW90 | >6000 Hz (broadband) | <2000 Hz (narrow) |
| Centroid | >3000 Hz | <1000 Hz |
| VHF Ratio | >2% | <1% |

## Instructions

1. Resolve the file path

2. Run Python analysis script using uv:
```python
import librosa
import numpy as np

# Load audio, extract window at onset time
# Compute FFT
# Calculate: BW90, centroid, spread, HF/VHF ratios, peak freq
```

3. Report results with interpretation:
   - Raw values for each feature
   - Classification: likely NOISE vs likely KALIMBA NOTE
   - Confidence level based on feature agreement

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
