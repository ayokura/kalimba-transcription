# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-26-bwv147-lower-mixed-roll-01-kalimba-17-c
- expected note: [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4>
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-sequence-163-01
- extracted range: 232.990s - 243.309s
- audio clip range: 231.990s - 244.309s
- extracted events: 127-133

## Expected Performance

- summary: [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4>

### Events

- 1. [<B4,G4>,D4] :: D4 (#8), G4 (#11), B4 (#12)
- 2. C5 :: C5 (#5)
- 3. D5 :: D5 (#13)
- 4. [G4,D4] [intent: strict_chord] :: D4 (#8), G4 (#11)
- 5. B4 :: B4 (#12)
- 6. D5 :: D5 (#13)
- 7. <F5,D5,B4,G4> [intent: slide_chord] :: G4 (#11), B4 (#12), D5 (#13), F5 (#14)

## Review

- summary: human listening confirms the scoped lower mixed BWV147 phrase with high confidence, and the current recognizer still under-segments the middle and terminal lower-roll region
- reason: ear review confirms the phrase at 3:52.990-4:03.309 with high confidence; the leading edge has only minor resonance contamination, while the current recognizer still collapses much of the middle and terminal lower-roll content, so this should stay pending as a practical target

- user audio review: all expected events are audible in the scoped region, the leading edge has only minor resonance contamination, and confidence is high
