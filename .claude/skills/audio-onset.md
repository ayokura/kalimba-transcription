---
name: audio-onset
description: Detect onsets (note beginnings) using aubio
user_invocable: true
arguments:
  - name: file
    description: Audio file path (fixture name or full path)
    required: true
  - name: method
    description: Detection method (default, energy, hfc, complex, phase, specdiff, kl, mkl, specflux)
    required: false
  - name: threshold
    description: Onset detection threshold (default 0.3)
    required: false
  - name: silence
    description: Silence threshold in dB (default -90)
    required: false
---

<command-name>audio-onset</command-name>

Detect onset times (beginning of sound events) using aubio.

## Arguments
- `file`: Audio file path or fixture name
- `method`: Detection algorithm (default: `default`)
  - `energy`: Energy-based
  - `hfc`: High-Frequency Content
  - `complex`: Complex domain
  - `phase`: Phase-based
  - `specdiff`: Spectral difference
  - `kl`: Kullback-Leibler
  - `mkl`: Modified KL
  - `specflux`: Spectral flux
- `threshold`: Detection sensitivity (0.0-1.0, lower = more sensitive)
- `silence`: Ignore sounds below this dB level

## Instructions

1. Resolve the file path (fixture name to full path if needed)

2. Run aubio onset detection:
```bash
aubio onset "<file>" [-m <method>] [-t <threshold>] [-s <silence>]
```

3. Parse and display results:
   - List of onset times
   - Count of detected onsets
   - Group by time regions (e.g., first 0.1s, 0.1-1s, 1s+)

4. If multiple methods requested, compare results

## Example Usage
```
/audio-onset d5-repeat-01
/audio-onset d5-repeat-01 hfc 0.5
/audio-onset mixed-sequence-01 specflux 0.3 -70
```

## Notes
- aubio is fast and suitable for real-time applications
- Different methods may detect different onset types
- Lower threshold = more onsets detected (more false positives)
- Compare with `/audio-visualize` for visual verification
