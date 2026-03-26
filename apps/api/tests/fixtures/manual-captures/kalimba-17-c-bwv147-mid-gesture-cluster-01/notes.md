# Manual Notes

- tester: manual
- verdict: review_needed
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

- summary: extracted practical BWV147 child; opening events 1-3 currently appear mismatched to the audio and need provenance review
- reason: short attack windows still rank A4+F5 rather than the expected A4 / [G5,G4] / [F5,A4]; verify parent mapping or user-entered expected before treating this as a recognizer target
- analysis note: 60-300ms opening windows continue to rank F5/A4 as the dominant content, which points to metadata/provenance uncertainty rather than a clean recognizer miss.
