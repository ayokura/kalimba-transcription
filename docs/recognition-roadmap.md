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
- fixture explainability reason codes
- `arpeggio` data-model introduction
- source-profile aware evaluation policy

### Explicitly not a current acoustic regression target
- legacy four-note fixture with broken metadata
- smartphone app reference video/audio

## Current Bottleneck

The strict four-note reference is now stable. The main remaining bottlenecks are:

- fixture explainability coverage and reason codes ([#5](https://github.com/ayokura/kalimba-transcription/issues/5))
- `arpeggio` modeling separate from `slide_chord` ([#6](https://github.com/ayokura/kalimba-transcription/issues/6))
- future source-profile differentiation for app/synth input ([#7](https://github.com/ayokura/kalimba-transcription/issues/7))
- tempo-estimation optimization ([#10](https://github.com/ayokura/kalimba-transcription/issues/10))

## Active Fixture Policy

### Completed strict four-note reference
- `kalimba-17-c-e4-g4-b4-d5-four-note-strict-repeat-03`
- status: `completed`
- issue: [#1](https://github.com/ayokura/kalimba-transcription/issues/1)
- note: recognizer now restores `E4+G4+B4+D5 x 6`

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

## Local Rule Debt

For fixture-specific or very local recognizer rules that may become debt for future free-performance transcription, see [recognizer-local-rules.md](recognizer-local-rules.md).
For Strategy B gap-candidate design and the current candidate/promotion prototype, see [strategy-b-gap-candidates.md](strategy-b-gap-candidates.md).

## Immediate Next Engineering Tasks

1. keep the completed strict four-note baseline stable
2. improve fixture explainability and coverage reporting
3. prepare `arpeggio` semantics and future source-profile support
4. keep editor gesture handling safe before adding `arpeggio`
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

Longer term, intent should move from recording-level metadata to event-level metadata. The current one-intent-per-capture model is acceptable only because fixtures are deliberately collected as one-gesture-per-take.
## Next Recording Priorities

The current acoustic recognizer no longer has a strong repeated-pattern redesign blocker. The next useful data is directional and register-sensitive practical material.

### Priority 1: Descending separated-note runs

Purpose:
- validate whether ascending-only local carryover cleanup generalizes
- measure whether descending runs need their own logic or only more data

Suggested captures:
- E6 -> D6 -> C6 -> B5 -> A5 -> G5 -> F5 -> E5 -> D5 -> C5 -> B4 -> A4 -> G4 -> F4 -> E4 -> D4 -> C4
- single pass
- same phrase repeated 3 times without intentional silence between notes

### Priority 2: High-register short-tine coverage

Purpose:
- measure whether D6/E6 need note-specific handling or only better fixture support
- capture the shorter sustain / different timbre the user identified near the top of the instrument

Suggested captures:
- alternating D6 / E6 single notes, 5 cycles
- C6 / D6 / E6 ascending and descending, each 5 cycles
- short phrase endings such as A5 / B5 / C6 / D6 / E6

### Priority 3: High-register mixed phrase endings

Purpose:
- test whether phrase tails near D6/E6 are still robust when they follow denser mid-register material

Suggested captures:
- C4 -> ... -> B5 -> C6 -> D6 -> E6
- E6 -> C4 restart after a brief gap
- one or two realistic phrase endings rather than isolated unit patterns only

Collection rule:
- prefer real-device fixtures
- keep articulation natural unless the goal specifically requires silence between notes
- when a phrase mixes techniques, add intent notes in the memo even if the current schema is still recording-level
## Future Input Source Profiles

For a detailed comparison between the current acoustic recognizer and the local app/synth reference audio, see [app-synth-audio-gap-analysis.md](C:/src/calimba-score/docs/app-synth-audio-gap-analysis.md).


Long term, the recognizer should not assume one acoustic environment.
Different input sources should be represented explicitly.

Initial profile split:
- `acoustic_real`: real kalimba recorded by microphone
- `app_synth`: smartphone or software kalimba app audio/video

Likely future dimensions inside `acoustic_real`:
- close mic vs room mic
- quiet room vs noisy room
- different phone / laptop microphone responses
- different kalimba models and resonance behavior

Policy:
- primary regression stays on `acoustic_real`
- app-derived material stays `reference_only` until a separate profile exists
- source profile should affect fixture status, evaluation policy, and future feature normalization

Why this matters:
- phone app audio can still be useful for pattern discovery and symbolic references
- but mixing it directly into the real-device regression pool will blur recognizer tuning decisions

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

## App-Video Arpeggio Candidate

The later half of the local smartphone-app reference is not best understood as `slide_chord`.
It is better treated as an `arpeggio candidate` / broken-chord reference.

Current evidence from `.codex-media/derived-analysis/kira-kira-expected-performance.json`:
- strongest candidate block: about `15.41s-22.83s`
- clearest arpeggio-like sub-block: about `19.25s-20.89s`
- projected sequence in that sub-block: `F4+F5 / A4 / C5 / A5 / A4 / F4`

Interpretation:
- this is useful as a reference for future `arpeggio` semantics and sample planning
- it is not a current acoustic regression target
- it should not be folded into `slide_chord` semantics

Near-term use:
- keep the source media local under `.codex-media/`
- use the derived event sequence only as `reference_only` design evidence
- collect equivalent real-device samples before recognizer work starts

## Future Real-Device Sample Families

The app-video analysis suggests these real-device families are worth collecting later:
- wide-register dyads such as `G5 + C4`, `C5 + G4`, `F4 + F5`
- wide-register triads such as `G5 + E4 + G4`
- chord-to-single continuations such as `B4 + D5 -> D5`
- broken-chord / `arpeggio` patterns such as `F4+F5 / A4 / C5 / A5 / A4 / F4`

These should be recorded as real-device fixtures, not derived directly from the app video.

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

