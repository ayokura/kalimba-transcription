# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-25-bwv147-late-upper-tail-01-kalimba-17-c
- expected note: G5 / E5 / <C5,A4> / D5 / E5
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-sequence-163-01
- extracted range: 260.700s - 267.500s
- audio clip range: 259.700s - 268.500s
- extracted events: 149-153

## Expected Performance

- summary: G5 / E5 / <C5,A4> / D5 / E5

### Events

- 1. G5 :: G5 (#3)
- 2. E5 :: E5 (#4)
- 3. <C5,A4> [intent: slide_chord] :: C5 (#5), A4 (#6)
- 4. D5 :: D5 (#13)
- 5. E5 :: E5 (#4)

## Review

- summary: human listening confirms the expected late-tail D5 / E5 region, so this remains a practical pending recognizer target
- reason: ear review confirms D5 / E5 without a boundary problem, with the terminal E5 played softly but intentionally; recognizer under-recovery here is a real practical miss

- evaluation window trimmed to exclude preceding phrase tail from the parent recording


- user audio review: D5 / E5 is audible, boundary issue is absent, and the terminal E5 is intentionally soft (confidence: high)
