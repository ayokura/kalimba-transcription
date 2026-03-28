---
name: audio-separate
description: Compare onset groups to find distinguishing waveform/spectral features (separation analysis)
user_invocable: true
arguments:
  - name: args
    description: "Flexible: fixture + onset times, or JSON config path"
    required: true
---

<command-name>audio-separate</command-name>

Compare groups of onset samples to find which waveform/spectral features best separate them.
Uses Cohen's d (standardized mean difference) and overlap analysis.

## Usage Patterns

### Quick comparison (single audio file)

```
/audio-separate bwv147-restart-prefix-01 --real 1.87,3.15,5.06 --compare 4.16
```

### Multi-file JSON config

```
/audio-separate --config /path/to/samples.json
```

## Instructions

1. Parse the user's request to determine the mode:
   - **Quick mode**: user provides fixture + onset times for two groups
   - **Config mode**: user provides a JSON file path

2. Run the analysis script:
```bash
# Quick mode
uv run python scripts/audio-analysis/onset_separation_analysis.py \
  --audio <fixture-or-path> --real <times> --compare <times>

# Config mode
uv run python scripts/audio-analysis/onset_separation_analysis.py \
  --config <path>
```

3. Interpret the results:
   - **Cohen's d > 2.0 (***\*)**: Strong separation — reliable filter candidate
   - **Cohen's d > 1.5 (**\*)**: Moderate separation — usable with care
   - **CLEAN**: No overlap between groups — threshold can cleanly separate
   - **overlap**: Groups overlap — threshold will have false positives/negatives

4. Report:
   - Top 5 features by separation score
   - For each: what it measures, why it separates, and suggested threshold
   - Whether the feature is robust (amplitude-independent, dynamics-independent)

## Computed Features (40+)

| Category | Features |
|----------|----------|
| Energy | rms_ratio, broadband_gain (20ms, 80ms windows) |
| Spectral | diff_flatness, diff_crest, diff_centroid, gain_positive_frac |
| Band gains | sub_bass through brilliance (7 bands, 5ms window) |
| Harmonicity | autocorrelation (20ms, 50ms), autocorr_change |
| Waveform | crest_factor (3/5/10ms), kurtosis_20ms, post_crest_20ms |
| Transient | max_deriv_5ms, mean_deriv_5ms, crest_change |

## JSON Config Format

```json
{
  "groups": {
    "real": [
      {"audio": "bwv147-restart-prefix-01", "onset": 1.87, "label": "E5"},
      {"audio": "/full/path/to/audio.wav", "onset": 3.15, "label": "C5"}
    ],
    "noise_trailing": [
      {"audio": "a4-d4-f4-triad-repeat-01", "onset": 16.97, "label": "triad_noise"}
    ]
  },
  "reference_group": "real"
}
```

## Key Insight

Different fake types (noise, residual) have fundamentally different characteristics.
**Always analyze per fake type** — mixing types hides clean separations.

## Related Skills

- `/audio-diagnose` - Single onset classification
- `/audio-spectrum` - Spectral features at one onset
- `/audio-visualize` - Visual spectrogram
