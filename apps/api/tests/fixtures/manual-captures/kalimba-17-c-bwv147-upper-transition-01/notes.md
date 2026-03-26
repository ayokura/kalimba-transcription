# Manual Notes

- tester: manual
- verdict: reference_only
- scenario: 2026-03-25-bwv147-upper-transition-01-kalimba-17-c
- expected note: F5 / [F5,A4] / A5 / G5 / [G5,E4]
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-restart-high-register-01
- extracted range: 5.550s - 13.200s
- audio clip range: 5.550s - 13.200s
- extracted events: 6-10

## Expected Performance

- summary: F5 / [F5,A4] / A5 / G5 / [G5,E4]

### Events

- 1. F5 :: F5 (#14)
- 2. [F5,A4] [intent: strict_chord] :: A4 (#6), F5 (#14)
- 3. A5 :: A5 (#15)
- 4. G5 :: G5 (#3)
- 5. [G5,E4] [intent: strict_chord] :: G5 (#3), E4 (#10)

## Review

- summary: current full-clip output is `G5 / [F5,A4] / A5 / G5`, so this clip is retained only as provenance reference rather than an active recognizer target
- reason: keep this fixture as provenance reference until the clip is reinterpreted or recut around the actual upper-transition phrase; the present clip does not support `F5 / [F5,A4] / A5 / G5 / [G5,E4]` as a clean active target

- explain output on the current clip is `G5 / [F5,A4] / A5 / G5`
- the opening expected `F5` is not present as the first detected event, and the terminal expected `[G5,E4]` does not appear at the end of the clip
- this looks like a provenance/scoping mismatch, not a narrow recognizer miss
