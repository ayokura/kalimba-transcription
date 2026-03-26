# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-27-bwv147-lower-context-roll-01-kalimba-17-c
- expected note: C5 / [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4> / E5
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-sequence-163-01
- extracted range: 231.990s - 244.309s
- audio clip range: 231.990s - 244.309s
- extracted events: 126-134

## Expected Performance

- summary: C5 / [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4> / E5

### Events

- 1. C5 :: C5 (#5)
- 2. [<B4,G4>,D4] :: D4 (#8), G4 (#11), B4 (#12)
- 3. C5 :: C5 (#5)
- 4. D5 :: D5 (#13)
- 5. [G4,D4] [intent: strict_chord] :: D4 (#8), G4 (#11)
- 6. B4 :: B4 (#12)
- 7. D5 :: D5 (#13)
- 8. <F5,D5,B4,G4> [intent: slide_chord] :: G4 (#11), B4 (#12), D5 (#13), F5 (#14)
- 9. E5 :: E5 (#4)

## Review

- summary: human listening confirms the wider lower context phrase is cleanly scoped, and the current recognizer now recovers the exact 9-event sequence on the full clip
- reason: this companion child keeps the leading C5 and trailing E5 inside the intended phrase instead of trying to crop them away; that preserves the long-gap structure the recognizer needs, so it is a stable practical completed regression target

- user audio review: C5 resonance / [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4> / E5 is audible; the cut is correct and boundary handling is OK (confidence: high)
