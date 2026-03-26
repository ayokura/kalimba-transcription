# Manual Notes

- tester: manual
- verdict: review_needed
- scenario: 2026-03-25-bwv147-f5-a5-transition-01-kalimba-17-c
- expected note: F5 / [F5,A4] / A5
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-restart-high-register-01
- extracted range: 5.550s - 11.200s
- audio clip range: 5.550s - 11.200s
- extracted events: 6-8

## Expected Performance

- summary: F5 / [F5,A4] / A5

### Events

- 1. F5 :: F5 (#14)
- 2. [F5,A4] [intent: strict_chord] :: A4 (#6), F5 (#14)
- 3. A5 :: A5 (#15)

## Review

- summary: expected F5/A5 ordering does not match the extracted audio; the scoped clip begins on G5 and does not provide a clean active recognizer target
- reason: expected F5/A5 ordering does not match the extracted audio; the scoped clip begins on G5 and does not provide a clean active recognizer target

- scoped debug starts on `G5` and ends on `A4+F4`; the expected `F5 / [F5,A4] / A5` ordering is not aligned closely enough to use as an active recognizer target
