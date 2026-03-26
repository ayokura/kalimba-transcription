# Manual Notes

- tester: manual
- verdict: review_needed
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

- summary: expected late-tail ordering does not match the scoped audio; D5/E5 tail events are not evident enough to keep this child as an active recognizer target
- reason: expected late-tail ordering does not match the scoped audio; D5/E5 tail events are not evident enough to keep this child as an active recognizer target

- evaluation window trimmed to exclude preceding phrase tail from the parent recording


- scoped debug shows only `G5`, `E5`, and the following mixed clusters; the expected `D5 / E5` tail is not isolated in the extracted audio
