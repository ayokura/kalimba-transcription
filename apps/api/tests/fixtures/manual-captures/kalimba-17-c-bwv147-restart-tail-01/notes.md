# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-25-bwv147-restart-tail-01-kalimba-17-c
- expected note: G5 / [G5,<G4,E4>] / C6 / B5 / [C6,<C5,A4>]
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-restart-high-register-01
- extracted range: 12.550s - 19.620s
- audio clip range: 12.350s - 19.920s
- extracted events: 9-13
- corresponds to: bwv147-163 R2 events 108-112 (around 184.9s-191.9s)

## Expected Performance

- summary: G5 / [G5,<G4,E4>] / C6 / B5 / [C6,<C5,A4>]

### Events

- 1. G5 :: G5 (#3)
- 2. [G5,<G4,E4>] [intent: slide_chord] :: G5 (#3), G4 (#11), E4 (#10)
- 3. C6 :: C6 (#16)
- 4. B5 :: B5 (#2)
- 5. [C6,<C5,A4>] [intent: slide_chord] :: C6 (#16), C5 (#5), A4 (#6)

## Review

- summary: practical BWV147 restart-tail child; event 2 and 5 corrected via energy analysis
- reason: event 2 corrected from [G5,E4] to [G5,<G4,E4>] — energy analysis confirms G4 genuinely struck (G4/G5=0.173, matching bwv147-163 true positive pattern). event 5 corrected from [C6,A4] to [C6,<C5,A4>] — C5 energy 168K confirms slide_chord. Corresponds to R2 events 108-112 in bwv147-163, matching R5 event 148 pattern [C6,<C5,A4>].
