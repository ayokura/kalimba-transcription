# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-25-c6-to-e6-sequence-13-separated-notes-kalimba-17-c
- expected note: C6 x 3 / D6 x 5 / E6 x 4 / C6 + E6
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:56:19.004Z

## Expected Performance

- summary: C6 x 3 / D6 x 5 / E6 x 4 / C6 + E6

### Events

- 1. C6 [intent: separated_notes] :: C6 (#16)
- 2. D6 [intent: separated_notes] :: D6 (#1)
- 3. E6 [intent: separated_notes] :: E6 (#17)
- 4. C6 [intent: separated_notes] :: C6 (#16)
- 5. D6 [intent: separated_notes] :: D6 (#1)
- 6. E6 [intent: separated_notes] :: E6 (#17)
- 7. E6 [intent: separated_notes] :: E6 (#17)
- 8. D6 [intent: separated_notes] :: D6 (#1)
- 9. C6 [intent: separated_notes] :: C6 (#16)
- 10. D6 [intent: separated_notes] :: D6 (#1)
- 11. C6 + E6 [intent: strict_chord] :: C6 (#16), E6 (#17)
- 12. D6 [intent: separated_notes] :: D6 (#1)
- 13. E6 [intent: separated_notes] :: E6 (#17)

## Review

- summary: high-register phrase and intended chord now match expected output.
- reason: recognizer already returns the expected 13 events, including the intended strict C6+E6 chord, so this fixture is now a stable regression target.

## Memo

- high-register reference fixture for C6/D6/E6 behavior; no note-specific heuristic required for this case.
