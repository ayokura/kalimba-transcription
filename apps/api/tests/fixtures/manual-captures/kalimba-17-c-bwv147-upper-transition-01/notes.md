# Manual Notes

- tester: manual
- verdict: review_needed
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

- summary: expected event ordering around the upper transition does not match the extracted audio; provenance must be reviewed before using this child as an active recognizer target
- reason: expected event ordering around the upper transition does not match the extracted audio; provenance must be reviewed before using this child as an active recognizer target

- scoped debug shows the expected `A5` is shifted later in the clip; the extracted audio around event 3 ranks `A4`/`F4` instead of `A5`
