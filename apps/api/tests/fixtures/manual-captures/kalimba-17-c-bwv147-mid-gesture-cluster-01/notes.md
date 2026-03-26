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

- summary: human listening confirms the intended opening A4 / [G5,G4] / [F5,A4], but the current extraction still has leading-boundary contamination
- reason: the expected opening is correct by ear, but a preceding resonance bleeds into the start of the clip and the initial A4 is intentionally soft; suffix-only re-windowing is not enough because it collapses the later `B4`, so re-scope this child from the parent before treating it as a clean active recognizer target
- user audio review: A4 / [G5,G4] / [F5,A4] is audible, with the A4 played softly and a leading boundary issue from the previous note residue (confidence: high)
- scoped re-windowing test: trimming this clip down to the later `B4 / <E5,C5> / <D5,B4,G4>` suffix drops `B4` and yields only `C5+E5 / B4+D5+G4`, so a new child must preserve more parent context instead of relying on `evaluationWindows`
