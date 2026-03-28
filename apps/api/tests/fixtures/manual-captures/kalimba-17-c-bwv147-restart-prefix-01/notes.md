# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-26-bwv147-restart-prefix-01-kalimba-17-c
- expected note: [E5,C4] / C5 / D5 / <E5,C5> / G5
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-25T13:49:01.459Z
- source fixture: kalimba-17-c-bwv147-sequence-163-01
- extracted range: 173.800s - 179.100s
- audio clip range: 172.300s - 179.900s
- extracted events: 37-41

## Expected Performance

- summary: [E5,C4] / C5 / D5 / <E5,C5> / G5

### Events

- 1. [E5,C4] [intent: strict_chord] :: E5 (#4), C4 (#9)
- 2. C5 :: C5 (#5)
- 3. D5 :: D5 (#13)
- 4. <E5,C5> [intent: slide_chord] :: E5 (#4), C5 (#5)
- 5. G5 :: G5 (#3)

## Review

- summary: practical BWV147 restart-prefix child now matches the scoped opening phrase cleanly and avoids the provenance-ambiguous tail region
- reason: practical BWV147 restart-prefix child now matches the scoped opening phrase cleanly and avoids the provenance-ambiguous tail region
