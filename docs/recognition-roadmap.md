# Recognition Roadmap

## Current State

### Stable / completed
- single notes
- octave dyads
- repeated triads
- repeated four-note `slide_chord`
- triple glissando / three-note `slide_chord`
- four-note ascending `slide_chord`

### Not yet complete
- clean four-note `strict_chord` reference

### Explicitly not a current acoustic regression target
- legacy four-note fixture with broken metadata
- smartphone app reference video/audio

## Current Bottleneck

The main remaining blocker is not recognizer coverage for `slide_chord`.
It is the lack of one clean four-note `strict_chord` ground-truth fixture for:

- `E4 + G4 + B4 + D5`

Until that exists, four-note simultaneous-chord work has a weak reference baseline.

## Active Fixture Policy

### Primary active rerecord target
- `kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-02`
- status: `rerecord`
- issue: [#1](https://github.com/ayokura/kalimba-transcription/issues/1)

### Legacy reference only
- `kalimba-17-c-e4-g4-b4-d5-four-note-repeat-01`
- status: `reference_only`
- reason: missing reliable `captureIntent`, broken scenario metadata, superseded by explicit fixtures

## User-Facing Gesture Families

Current canonical user-facing families are:
- `strict_chord`
- `slide_chord`
- `separated_notes`
- `ambiguous`

For kalimba semantics, `rolled_chord` and `gliss` are unified as `slide_chord`.

## Future Boundary: Arpeggio

`arpeggio` should **not** be folded into `slide_chord`.

Why:
- `slide_chord` still resolves to one chord gesture family
- `arpeggio` is an ordered broken-chord pattern with distinct time structure
- later notation and editing will need note order and direction, not just the harmonic set

Planned direction:
- keep `arpeggio` separate from `slide_chord`
- attach it to a main harmonic event or explicitly link it to one
- avoid double-counting one musical idea as several unrelated top-level chord events

Tracking issue:
- [#6](https://github.com/ayokura/kalimba-transcription/issues/6)

## Immediate Next Engineering Tasks

1. wait for clean four-note `strict_chord` rerecord
2. promote that rerecord to `completed` when it satisfies regression conditions
3. keep `slide_chord` recognition stable while strict baseline is rebuilt
4. do not expand speculative four-note recognizer heuristics without a better strict reference
5. start arpeggio sample collection only after the data model boundary is fixed

## Suggested Future Sample Matrix

When collection resumes, prefer paired samples for the same pitch set:

### For the same note set
- `strict_chord`
- `slide_chord`
- `arpeggio`

### Suggested note sets
- `C4 + E4 + G4`
- `A4 + C5 + E5`
- `E4 + G4 + B4 + D5`

This will make family boundaries testable without changing pitch content.

## Smartphone App Reference Video

Local path:
- `C:\src\calimba-score\.codex-media\source-videos\ScreenRecording_03-23-2026_13-09-56_1.mov`

Use it only as:
- visual vocabulary reference
- possible future `reference_only` UI material

Do **not** use it as:
- acoustic regression input
- real-device performance ground truth

Reason:
- it reflects app rendering and phone capture behavior, not real kalimba acoustics or hand technique

## Recording Request Template

Use this structure whenever asking for new manual data:

- `goal`
- `gesture`
- `notes`
- `repetitions`
- `spacing`
- `success criteria`

For the current strict four-note rerecord:
- goal: rebuild clean simultaneous four-note reference
- gesture: `strict_chord`
- notes: `E4 + G4 + B4 + D5`
- repetitions: `5`
- spacing: about `1s` silence between takes
- success criteria: `5 events`, each `E4+G4+B4+D5`
