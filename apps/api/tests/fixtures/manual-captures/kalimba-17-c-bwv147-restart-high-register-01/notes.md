# Manual Notes

- tester: manual
- verdict: review_needed
- scenario: 2026-03-25-bwv147-restart-high-register-01-kalimba-17-c
- expected note: [E5,C4] / C5 / D5 / <E5,C5> / G5 / F5 / [F5,A4] / A5 / G5 / [G5,E4] / C6 / B5 / [C6,A4]
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-sequence-163-01
- extracted range: 173.800s - 191.920s
- audio clip range: 172.300s - 193.420s
- extracted events: 37-49

## Expected Performance

- summary: [E5,C4] / C5 / D5 / <E5,C5> / G5 / F5 / [F5,A4] / A5 / G5 / [G5,E4] / C6 / B5 / [C6,A4]

### Events

- 1. [E5,C4] [intent: strict_chord] :: E5 (#4), C4 (#9)
- 2. C5 :: C5 (#5)
- 3. D5 :: D5 (#13)
- 4. <E5,C5> [intent: slide_chord] :: E5 (#4), C5 (#5)
- 5. G5 :: G5 (#3)
- 6. F5 :: F5 (#14)
- 7. [F5,A4] [intent: strict_chord] :: A4 (#6), F5 (#14)
- 8. A5 :: A5 (#15)
- 9. G5 :: G5 (#3)
- 10. [G5,E4] [intent: strict_chord] :: G5 (#3), E4 (#10)
- 11. C6 :: C6 (#16)
- 12. B5 :: B5 (#2)
- 13. [C6,A4] [intent: strict_chord] :: A4 (#6), C6 (#16)

## Review

- summary: human listening confirms the F5 / [F5,A4] and final [C6,A4] subregions, but the current extraction still mixes in a mistaken middle retake and clips the final onset at the window edge
- reason: 6s-10s ear review confirms F5 -> [F5,A4] with a clear gap, and the final region audibly contains [C6,A4], but the current child still includes a mistaken middle retake noted in request metadata and trims the final onset at the evaluation-window edge; re-scope before promoting it back to pending

- user audio review: F5 -> [F5,A4] is audible with a gap in the 6s-10s region, and the final region audibly contains [C6,A4] (confidence: high, boundary issue: no)
