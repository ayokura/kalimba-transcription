# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-25-bwv147-mid-gesture-cluster-01
- expected note: A4 / [G5,G4] / [F5,A4] / B4 / <E5,C5> / <D5,B4,G4>
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-sequence-163-01
- extracted range: 135.740s - 146.850s
- audio clip range: 134.240s - 148.350s
- extracted events: 77-82

## Expected Performance

- summary: A4 / [G5,G4] / [F5,A4] / B4 / <E5,C5> / <D5,B4,G4>

### Events

- 1. A4 :: A4 (#6)
- 2. [G5,G4] [intent: strict_chord] :: G5 (#3), G4 (#11)
- 3. [F5,A4] [intent: strict_chord] :: A4 (#6), F5 (#14)
- 4. B4 :: B4 (#12)
- 5. <E5,C5> [intent: slide_chord] :: E5 (#4), C5 (#5)
- 6. <D5,B4,G4> [intent: slide_chord] :: G4 (#11), B4 (#12), D5 (#13)

## Review

- summary: practical BWV147 mid-phrase gesture-cluster child fixture extracted from the parent corpus for orthogonal recognizer evaluation
- reason: practical BWV147 mid-phrase gesture-cluster child fixture extracted from the parent corpus for orthogonal recognizer evaluation
