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

### 1. Spectral Analysis (librosa)
- BW90 (90% bandwidth)
- Spectral centroid
- VHF ratio (>8kHz)

### 2. Pitch Detection (praat)
- Detected frequency
- Nearest kalimba note
- Deviation percentage

### 3. Temporal Evolution
- Fund growth: fundamental energy at 40-60ms / 0-20ms
- HF decay: high-frequency energy decay pattern

### 4. Visual Confirmation (sox)
- Generate spectrogram of onset region

## Classification Criteria

| Test | Noise | Real Note |
|------|-------|-----------|
| BW90 | >6000 Hz | <2000 Hz |
| Centroid | >3000 Hz | <1000 Hz |
| VHF% | >2% | <1% |
| Pitch match | >10% deviation | <5% deviation |
| Fund growth | <1.0 (decreasing) | >1.0 (increasing) |

## Output
- **NOISE**: Recording artifact, environmental sound, etc.
- **KALIMBA**: Real kalimba note with identified pitch
- **UNCLEAR**: Ambiguous, requires manual review

## Instructions

1. If no specific time given:
   - Use aubio to detect onsets in first 0.2 seconds
   - Analyze each detected onset

2. For each onset:
   - Run `/audio-spectrum` analysis
   - Run `/audio-pitch` detection
   - Generate spectrogram with `/audio-visualize`
   - Compute temporal evolution metrics

3. Aggregate results and classify:
   - Count how many tests indicate NOISE vs NOTE
   - Provide confidence score
   - Show detailed breakdown

4. Provide recommendation:
   - Whether this onset should be included in transcription
   - Suggested threshold adjustments if needed

## Example Usage
```
/audio-diagnose d5-repeat-01
/audio-diagnose d5-repeat-01 0.059
/audio-diagnose c4-e4-g4-triad-repeat-01 0.056
```

## Related Skills
- `/audio-visualize` - Visual spectrogram
- `/audio-onset` - Onset detection
- `/audio-pitch` - Pitch detection
- `/audio-spectrum` - Spectral features
