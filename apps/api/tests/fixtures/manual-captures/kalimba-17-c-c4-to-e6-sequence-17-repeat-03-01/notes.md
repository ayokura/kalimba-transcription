# Manual Notes

- tester: manual
- verdict: pending
- scenario: 2026-03-24-c4-to-e6-sequence-17-separated-notes-kalimba-17-c
- expected note: (C4 / D4 / E4 / F4 / G4 / A4 / B4 / C5 / D5 / E5 / F5 / G5 / A5 / B5 / C6 / D6 / E6) x3
- capture intent: separated_notes
- default capture intent: separated_notes
- source profile: acoustic_real
- captured at: 2026-03-24T02:48:12.667Z

## Expected Performance

- summary: (C4 / D4 / E4 / F4 / G4 / A4 / B4 / C5 / D5 / E5 / F5 / G5 / A5 / B5 / C6 / D6 / E6) x3

### Events

- 1. C4 [intent: separated_notes] :: C4 (#9)
- 2. D4 [intent: separated_notes] :: D4 (#8)
- 3. E4 [intent: separated_notes] :: E4 (#10)
- 4. F4 [intent: separated_notes] :: F4 (#7)
- 5. G4 [intent: separated_notes] :: G4 (#11)
- 6. A4 [intent: separated_notes] :: A4 (#6)
- 7. B4 [intent: separated_notes] :: B4 (#12)
- 8. C5 [intent: separated_notes] :: C5 (#5)
- 9. D5 [intent: separated_notes] :: D5 (#13)
- 10. E5 [intent: separated_notes] :: E5 (#4)
- 11. F5 [intent: separated_notes] :: F5 (#14)
- 12. G5 [intent: separated_notes] :: G5 (#3)
- 13. A5 [intent: separated_notes] :: A5 (#15)
- 14. B5 [intent: separated_notes] :: B5 (#2)
- 15. C6 [intent: separated_notes] :: C6 (#16)
- 16. D6 [intent: separated_notes] :: D6 (#1)
- 17. E6 [intent: separated_notes] :: E6 (#17)

## Review

- summary: continuous articulation のまま 51 event を回復。repeat-03 は completed へ昇格。
- reason: gap-head segmentation と sparse-gap-tail の高音オクターブ簡約で、3 cycle 全体の separated-notes を practical fixture として保持できる状態になりました。

## Memo

- continuous articulation: notes were played back-to-back without intentional silence between them
