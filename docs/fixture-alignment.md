# Fixture Alignment and Recording-Specific Overrides

This document describes how fixture score intent, recording-specific performance
differences, and regression assertions relate to each other.

## Alignment Overrides

`alignment_overrides.json` is used when `score_structure.json` is musically
correct as sheet-music intent, but the actual recording differs from that score.
It patches specific events into the "truth for this recording".

Important rules:

- `score_structure.json` represents the intended score and should not be changed
  to match one imperfect recording.
- `alignment_overrides.json` records recording-specific facts: the transformation
  from score intent to the performed event sequence.
- Like `ignoredRanges`, agents may add or change alignment overrides **only when
  the user explicitly permits or requests it**. Do not add them on agent judgment
  alone.
- Each override must include a `reason` describing the evidence, such as ear
  confirmation or spectrum analysis.

## Schema

Both v1-compatible replace entries and v2 operation entries are supported. If
`op` is omitted, the entry behaves like `replace`.

```json
{
  "version": 2,
  "overrides": [
    {"op": "replace", "eventIndex": 64, "expectedNotes": ["C5", "E4"], "reason": "..."},
    {
      "op": "insert",
      "afterEventIndex": 115,
      "expectedNotes": ["E5"],
      "reason": "R3 playing error: E116 D5 missed, E5 played instead; restarted from D5"
    },
    {"op": "skip", "eventIndex": 170, "reason": "performer skipped this note"}
  ]
}
```

### Operations

- **replace**: overwrite `expectedNotes` for an existing `eventIndex`. This is
  v1-compatible and is also the default when `op` is omitted.
- **insert**: insert an extra performed event not present in the score, directly
  after `afterEventIndex`. The label is `E{afterEventIndex}{suffix}`, for example
  `E115a`. Multiple inserts after the same event automatically receive `a`, `b`,
  `c`, ... unless a `suffix` field is explicitly provided.
- **skip**: mark a score event as not played in the recording.

## Relationship Between Fixture Files

- `score_structure.json`: score truth / musical intent. Treat as stable.
- `alignment_overrides.json`: recording-specific diff from score to recording.
- `expected.json:expectedEventNoteSetsOrdered`: final performed order used by
  test assertions. It should stay consistent with the score after applying
  alignment overrides.
- `score_alignment_diagnosis.py`: applies alignment overrides to score structure
  and compares that result with recognizer output.
