---
name: audio-diagnose
description: Comprehensive onset diagnosis - determine if onset is noise or real kalimba note
user_invocable: true
arguments:
  - name: file
    description: Audio file path (fixture name or full path)
    required: true
  - name: time
    description: Onset time to diagnose (seconds). If omitted, analyzes all early onsets (<0.2s)
    required: false
---

<command-name>audio-diagnose</command-name>

Comprehensive diagnosis of an onset to determine if it's noise or a real kalimba note.

## Arguments
- `file`: Audio file path or fixture name
- `time`: Specific onset time to diagnose. If omitted, finds and analyzes all onsets in first 0.2 seconds.

## Diagnostic Tests Performed

### 1. Spectral Classification (spectrum_stats.py)
```bash
PYTHONPATH=apps/api uv run python3 scripts/audio-analysis/spectrum_stats.py <file> <time>
```
Returns KALIMBA / NOISE / UNCLEAR with BW90, centroid, HF ratio.

### 2. Pitch Detection (praat)
```bash
praat --run scripts/audio-analysis/pitch_detect.praat <file> <start> <duration>
```
Detected frequency, nearest kalimba note, deviation percentage.

### 3. Visual Confirmation (sox spectrogram)
```bash
sox <file> -n spectrogram -x 1200 -Y 500 -S <start-0.1> -d 0.5 -t "<title>" -o /tmp/diagnose.png
```
Then read the PNG with the Read tool to visually confirm.

### 4. Temporal Evolution (optional, for ambiguous cases)
Use `/audio-energy-trace` to track note-band energy over time around the onset.

## Classification Criteria

| Test | Noise | Real Note |
|------|-------|-----------|
| BW90 | >6000 Hz | <2000 Hz |
| Centroid | >3000 Hz | <1500 Hz |
| HF Ratio | >30% | <10% |
| VHF% | >2% | <1% |
| Pitch match | >10% deviation | <5% deviation |
| Spectrogram | Diffuse broadband | Clear harmonic lines |

## Decision Logic

1. Run spectrum_stats.py — if KALIMBA or NOISE classification is clear, that's the primary answer
2. If UNCLEAR, run pitch detection for additional evidence
3. Generate spectrogram for visual confirmation
4. Aggregate: majority vote across tests

## Output
- **NOISE**: Recording artifact, environmental sound, ambient noise
- **KALIMBA**: Real kalimba note with identified pitch
- **UNCLEAR**: Ambiguous, requires manual review — show all evidence

## Example Usage
```
/audio-diagnose d5-repeat-01
/audio-diagnose d5-repeat-01 0.059
/audio-diagnose c4-e4-g4-triad-repeat-01 0.056
```
