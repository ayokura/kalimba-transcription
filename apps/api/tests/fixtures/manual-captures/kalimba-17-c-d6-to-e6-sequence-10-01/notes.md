# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-25-d6-to-e6-sequence-10-separated-notes-kalimba-17-c
- expected note: D6 x 5 / E6 x 5
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:46:42.276Z

## Expected Performance

- summary: D6 x 5 / E6 x 5

### Events

- 1. D6 [intent: separated_notes] :: D6 (#1)
- 2. E6 [intent: separated_notes] :: E6 (#17)
- 3. D6 [intent: separated_notes] :: D6 (#1)
- 4. E6 [intent: separated_notes] :: E6 (#17)
- 5. D6 [intent: separated_notes] :: D6 (#1)
- 6. E6 [intent: separated_notes] :: E6 (#17)
- 7. D6 [intent: separated_notes] :: D6 (#1)
- 8. E6 [intent: separated_notes] :: E6 (#17)
- 9. D6 [intent: separated_notes] :: D6 (#1)
- 10. E6 [intent: separated_notes] :: E6 (#17)

## Review

- summary: ordered high-register separated-notes sequence now matches expected note-set order.
- reason: Recognizer now preserves the alternating D6/E6 pattern without the former bridge dyad collapse.


## Memo

(empty)
