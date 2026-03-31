---
name: audio-peak-track
description: Track per-note peak frequency and cents deviation over time
user_invocable: true
---

Track the actual peak frequency in a note's frequency band over time, showing tuning deviation in cents.

## Arguments
- `file`: Audio file path or fixture name (e.g., `bwv147-sequence-163-01`)
- `start`: Start time in seconds
- `duration`: Duration in seconds
- `notes`: Comma-separated note names (default: `D4`)
- `step`: Time step in seconds (default: `0.05`)
- `n-fft`: Minimum FFT size (default: `8192`)
- `band-cents`: Search band in ±cents (default: `100`)
- `min-energy`: Minimum energy to display (default: `1.0`)

## Instructions

Run the peak tracking script:
```bash
uv run python scripts/audio-analysis/note_peak_track.py <file> <start> <duration> --notes <notes> --step <step> [--n-fft <n_fft>] [--band-cents <cents>] [--min-energy <min>]
```

## Output Columns

| Column | Meaning |
|--------|---------|
| `peak_Hz` | Actual peak frequency found in the note's band |
| `cents` | Deviation from tuning reference in cents |
| `✓` / `✗` | Whether peak is within ±40 cents of tuning |
| `energy` | FFT magnitude at the peak |

## Interpretation Guide

| Pattern | Meaning |
|---------|---------|
| Stable peak_Hz with high energy | Note is genuinely present |
| Energy spike at one time, then decay | Note attack followed by sustain |
| Peak always `---` | Note not present (or below min-energy) |
| Peak present but `✗` (outside ±40¢) | Tuning drift or wrong note detection |
| Consistent cents offset across time | Systematic tuning deviation of the tine |

## Use Cases

- **Verify note presence**: Check if a note expected by score_structure is actually in the audio
- **Diagnose FFT resolution**: Default n_fft=8192 avoids the low-frequency bin gap issue (#101)
- **Tuning calibration**: Track cents deviation to check if a tine is consistently sharp/flat
- **Attack detection**: Energy jump at a specific time indicates genuine onset

## Example Usage
```
/audio-peak-track bwv147-sequence-163-01 40.5 1.5 --notes D4,B4,G4
/audio-peak-track c4-to-g4-sequence-17-01 13.0 1.0 --notes C5,E5 --step 0.02
/audio-peak-track d5-repeat-01 0 3 --notes D5 --band-cents 50
```
