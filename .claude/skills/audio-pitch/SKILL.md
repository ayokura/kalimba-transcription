---
name: audio-pitch
description: High-precision pitch detection using praat
user_invocable: true
arguments:
  - name: file
    description: Audio file path (fixture name or full path)
    required: true
  - name: start
    description: Start time in seconds (default 0)
    required: false
  - name: duration
    description: Duration to analyze in seconds (default 3)
    required: false
  - name: step
    description: Time step in seconds (default 0.01)
    required: false
  - name: min_pitch
    description: Minimum pitch in Hz (default 75)
    required: false
  - name: max_pitch
    description: Maximum pitch in Hz (default 1500)
    required: false
---

<command-name>audio-pitch</command-name>

Perform high-precision pitch detection using Praat.

## Arguments
- `file`: Audio file path or fixture name
- `start`: Analysis start time (default: 0)
- `duration`: Duration to analyze (default: 3 seconds)
- `step`: Time resolution (default: 0.01 = 10ms)
- `min_pitch`: Lower frequency bound (default: 75 Hz)
- `max_pitch`: Upper frequency bound (default: 1500 Hz, covers kalimba range)

## Kalimba Reference Frequencies (17-key C-tuned)
- C4: 261.6 Hz, D4: 293.7 Hz, E4: 329.6 Hz, F4: 349.2 Hz
- G4: 392.0 Hz, A4: 440.0 Hz, B4: 493.9 Hz
- C5: 523.3 Hz, D5: 587.3 Hz, E5: 659.3 Hz, F5: 698.5 Hz
- G5: 784.0 Hz, A5: 880.0 Hz, B5: 987.8 Hz
- C6: 1046.5 Hz, D6: 1174.7 Hz, E6: 1318.5 Hz

## Instructions

1. Resolve the file path

2. Create and run a Praat script:
```bash
cat > /tmp/pitch_analysis.praat <<'PRAAT_SCRIPT'
Read from file: "<file>"
sound = selected("Sound")
To Pitch: <step>, <min_pitch>, <max_pitch>
pitch = selected("Pitch")
writeInfoLine: "time", tab$, "pitch_Hz", tab$, "note"
# ... iterate and output pitch values
PRAAT_SCRIPT
praat --run /tmp/pitch_analysis.praat
```

3. Parse results and identify notes:
   - Map detected frequencies to nearest kalimba note
   - Calculate deviation percentage
   - Flag non-kalimba frequencies

4. Report:
   - Pitch timeline with note names
   - Stable pitch regions (sustained notes)
   - Anomalies (frequencies not matching kalimba notes)

## Example Usage
```
/audio-pitch d5-repeat-01
/audio-pitch d5-repeat-01 2.0 1.0 0.005
/audio-pitch mixed-sequence-01 0 5 0.01 200 1000
```

## Notes
- Praat provides research-grade pitch detection
- More accurate than aubio for detailed analysis
- Use with `/audio-onset` to analyze specific onset times
