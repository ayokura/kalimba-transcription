# Arpeggio Minimal Design

## Goal

Introduce `arpeggio` without breaking the current note-set based transcription flow.

## Current Constraint

Current `ScoreEvent` is a flat event with:
- startBeat
- durationBeat
- notes[]
- isGlissLike
- gesture

Current notation views are derived from note sets only. They do not encode performance style.

## Minimal Phase

Phase 1 should add `arpeggio` only to the gesture / capture-intent vocabulary.

Scope:
- backend gesture vocabulary accepts `arpeggio`
- debug capture accepts `arpeggio` intent
- explainability / fixture tooling can label a sample as `arpeggio`
- notation/event structure stays unchanged

Non-goals:
- no event-link graph yet
- no ornament attachment model yet
- no dedicated arpeggio recognizer yet
- no notation-specific arpeggio rendering yet

## Why This Phase Is Small Enough

This keeps three concerns separate:
- harmonic content
- performance gesture family
- future notation-specific ornament semantics

That allows sample collection and review to start before recognizer work lands.

## Editor Policy

Current editor operations (`mergeWithNext`, `splitEvent`, note add/remove) preserve the existing `gesture` field implicitly by copying the event object.
This is unsafe once `arpeggio` exists.

Required rule for Phase 1:
- after merge / split / note edits, reset `gesture` to `ambiguous`
- do not silently preserve stale gesture labels

Reason:
- edited note sets may no longer support the previous gesture claim
- recomputing correctly in the editor is harder than clearing the label

Future option:
- add explicit recompute later if a reliable local classifier exists

## Data Model Boundary

Recommended vocabulary after Phase 1:
- `strict_chord`
- `slide_chord`
- `arpeggio`
- `separated_notes`
- `ambiguous`

Recommended interpretation:
- `strict_chord`: one harmonic event with near-simultaneous attack
- `slide_chord`: one harmonic event formed by a slide/rolled sweep
- `arpeggio`: ordered broken-chord motion that should remain distinct from `slide_chord`

## Fixture / Capture Policy

For Phase 1, `expectedPerformance` remains note-only.
Distinguish `slide_chord` and `arpeggio` by `captureIntent`, not by changing event payload shape.

Implication:
- the same note set may appear in both fixture families
- review tooling must compare expected note sets and intended gesture separately

Current interpretation:
- `captureIntent` is recording-level metadata
- this is sufficient while we collect one-gesture-per-take fixtures

Future direction:
- move intent down to the event level
- keep an optional recording-level default intent for convenience
- allow mixed recordings where one capture contains `strict_chord`, `slide_chord`, `arpeggio`, and `separated_notes` events

Recommended future shape:
- recording-level: `defaultCaptureIntent`
- event-level: `expectedPerformance.events[].intent`

That structure matches the eventual product better than a single recording-wide intent field.

## Event-Level Intent Schema Draft

Recommended capture shape once mixed recordings become normal:

```json
{
  "expectedPerformance": {
    "defaultCaptureIntent": "separated_notes",
    "events": [
      {
        "index": 1,
        "display": "C4",
        "intent": "separated_notes"
      },
      {
        "index": 2,
        "display": "E4 + G4 + C5",
        "intent": "slide_chord"
      },
      {
        "index": 3,
        "display": "F4 / A4 / C5 / A4",
        "intent": "arpeggio"
      }
    ]
  }
}
```

Rules:
- `defaultCaptureIntent` remains optional convenience metadata
- `events[].intent` overrides the recording default
- old fixtures with only recording-level intent remain valid until migration is complete
- explainability should compare both pitch content and intended gesture family

## Mixed Capture Collection Policy

Once event-level intent exists, mixed recordings should be accepted when:
- the phrase is short enough to annotate reliably
- `slide_chord` and `arpeggio` sections are explicitly marked
- the expected event list is stable enough to act as a `pending` target

Recommended operator guidance:
- keep one phrase per recording when possible
- if mixing gesture families, annotate boundaries explicitly
- prefer `reference_only` over forcing low-confidence mixed phrases into `completed`

## Migration Order

1. keep current recording-level `captureIntent` support
2. add optional `events[].intent` in exported/request metadata
3. update explainability CLI to surface event-level intent mismatches
4. update debug capture UI to let the operator set intent per event
5. only then start collecting mixed `slide_chord` + `arpeggio` fixtures at scale

## Future Phase

Later, move beyond gesture labels and model arpeggio as one of:
1. a top-level event with ordered-note metadata
2. an ornament attached to a harmonic anchor event

Do not decide that now. The current need is only to avoid semantic collapse with `slide_chord`.
