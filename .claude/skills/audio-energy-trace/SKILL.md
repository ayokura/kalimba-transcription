---
name: audio-energy-trace
description: Trace per-note frequency band energy over time for detailed onset/resonance analysis
user_invocable: true
arguments:
  - name: file
    description: Audio file path (fixture name or full path)
    required: true
  - name: start
    description: Start time in seconds
    required: true
  - name: duration
    description: Duration in seconds
    required: true
  - name: notes
    description: Comma-separated note names to track (default G4,G5,E4)
    required: false
  - name: step
    description: Time step in seconds (default 0.05)
    required: false
---

<command-name>audio-energy-trace</command-name>

Trace energy at specific note frequencies over time to distinguish genuine attacks from sympathetic resonance.

## Arguments
- `file`: Audio file path or fixture name (e.g., `bwv147-restart-tail-01`)
- `start`: Start time in seconds
- `duration`: Duration in seconds
- `notes`: Comma-separated note names (default: `G4,G5,E4`)
- `step`: Time step in seconds (default: `0.05`)

## Instructions

Run the energy trace script:
```bash
uv run python scripts/audio-analysis/energy_trace.py <file> <start> <duration> --notes <notes> --step <step>
```

## Interpretation Guide

| Pattern | Meaning |
|---------|---------|
| Energy spike at onset, sustained decay | Genuine note attack |
| Energy appears with another note's onset | Sympathetic resonance or harmonic artifact |
| Energy ratio (note/reference) stable ~0.001 | Minimal coupling (noise floor) |
| Energy ratio jumps from ~0.001 to >0.1 at onset | Likely genuine attack |
| Energy ratio >0.1 sustained after onset | Note is genuinely present |

## Example Usage
```
/audio-energy-trace bwv147-restart-tail-01 0.2 2.5 --notes G4,G5,E4
/audio-energy-trace bwv147-sequence-163-01 185 3 --notes G4,G5,E4 --step 0.03
```
