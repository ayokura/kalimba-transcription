# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-25-e6-to-g4-sequence-06-separated-notes-kalimba-17-c
- expected note: E6 / C4 / D4 / E4 / F4 / G4
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-25T11:57:23.176Z

## Expected Performance

- summary: E6 / C4 / D4 / E4 / F4 / G4

### Events

- 1. E6 [intent: separated_notes] :: E6 (#17)
- 2. C4 [intent: separated_notes] :: C4 (#9)
- 3. D4 [intent: separated_notes] :: D4 (#8)
- 4. E4 [intent: separated_notes] :: E4 (#10)
- 5. F4 [intent: separated_notes] :: F4 (#7)
- 6. G4 [intent: separated_notes] :: G4 (#11)

## Review

- summary: restart separated-notes phrase が current recognizer で一致しました。
- reason: stale upper E6 carryover と mirrored D4/E4 overlap を local に除去して practical restart sample を completed に昇格しました。

## Memo

(empty)
