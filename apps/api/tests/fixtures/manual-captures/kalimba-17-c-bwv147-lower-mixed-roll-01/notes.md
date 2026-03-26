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

- summary: human listening confirms the scoped lower mixed BWV147 phrase is correctly cut; the full clip now recovers the target phrase, but the current scoped evaluation window still compresses the long-gap structure
- reason: ear review confirms the clip is scoped correctly: the audible content is C5 resonance / [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4> / E5, with the leading C5 resonance and trailing E5 outside the intended evaluation phrase; after the lower-mixed opening extension, long-gap segmentation fix, and lower-roll tail extension, the full clip recognizer output now recovers the intended phrase, but the current evaluationWindow crop removes too much surrounding context and collapses the scoped regression to five events, so the fixture should remain pending until its scoped evaluation strategy is revised

- user audio review: C5 resonance / [<B4,G4>,D4] / C5 / D5 / [G4,D4] / B4 / D5 / <F5,D5,B4,G4> / E5 is audible; the cut is correct and boundary handling is OK (confidence: high)
