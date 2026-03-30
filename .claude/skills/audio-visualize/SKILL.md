---
name: audio-visualize
description: Generate spectrograms using sox for visual audio analysis
user_invocable: true
arguments:
  - name: file
    description: Audio file path (fixture name or full path)
    required: true
  - name: start
    description: Start time in seconds (default 0)
    required: false
  - name: duration
    description: Duration in seconds (default 0.5)
    required: false
  - name: output
    description: Output PNG path (default /tmp/spectrogram.png)
    required: false
---

<command-name>audio-visualize</command-name>

Generate a spectrogram image from an audio file using sox.

## Arguments
- `file`: Audio file path. Can be:
  - Full path: `/path/to/audio.wav`
  - Fixture name: `d5-repeat-01` (resolves to `apps/api/tests/fixtures/manual-captures/kalimba-17-c-{name}/audio.wav`)
- `start`: Start time in seconds (default: 0)
- `duration`: Duration in seconds (default: 0.5)
- `output`: Output PNG path (default: `/tmp/spectrogram.png`)

## Instructions

1. Resolve the file path:
   - If it's a fixture name (no `/`), expand to full fixture path
   - Verify the file exists

2. Generate spectrogram using sox:
```bash
sox "<input>" -n spectrogram -x 1200 -Y 500 -S <start> -d <duration> -t "<title>" -o "<output>"
```

Note: Use `-Y` (capital, total height) not `-y` (per-channel). Use `-S` for start position and `-d` for duration.

3. Read and display the generated PNG image using the Read tool

4. Report:
   - File analyzed
   - Time range
   - Visual observations: frequency bands, harmonic structure vs broadband noise, attack transients, resonance tails

## Interpretation Guide

| Pattern | Meaning |
|---------|---------|
| Bright vertical line with horizontal harmonics | Genuine kalimba attack |
| Faint diffuse energy across all frequencies | Broadband noise (click, breath, ambient) |
| Gradually fading horizontal lines | Sustain/resonance from prior note |
| Concentrated energy below 2kHz | Typical kalimba fundamental + low harmonics |
| Energy spread above 4kHz | Noise, transient click, or very high register note |

## Example Usage
```
/audio-visualize d5-repeat-01
/audio-visualize d5-repeat-01 0 2
/audio-visualize a4-d4-f4-triad-repeat-01 0 4 /tmp/triad_leading.png
```
