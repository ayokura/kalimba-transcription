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
sox "<input>" -n trim <start> <duration> spectrogram -x 800 -y 500 -z 70 -o "<output>" -t "<title>"
```

3. Read and display the generated PNG image using the Read tool

4. Report:
   - File analyzed
   - Time range
   - Output path
   - Visual observations (frequency bands, patterns, noise vs harmonic structure)

## Example Usage
```
/audio-visualize d5-repeat-01
/audio-visualize d5-repeat-01 0.05 0.1
/audio-visualize /path/to/audio.wav 0 1 /tmp/my_spectrogram.png
```
