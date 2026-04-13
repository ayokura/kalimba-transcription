# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-23-c4-to-g4-sequence-17-kalimba-17-c
- expected note: C4 x 4 / G4 x 5 / E5 + C5 x 3 / C5 x 2 / D5 / E5 + C4 / G5 + E4
- capture intent: unknown
- default capture intent: unknown
- source profile: acoustic_real
- captured at: 2026-03-23T16:25:51.289Z

## Expected Performance

- summary: C4 x 4 / G4 x 5 / E5 + C5 x 3 / C5 x 2 / D5 / E5 + C4 / G5 + E4

### Events

- 1. C4 :: C4 (#9)
- 2. G4 :: G4 (#11)
- 3. E5 + C5 :: E5 (#4), C5 (#5)
- 4. C4 :: C4 (#9)
- 5. G4 :: G4 (#11)
- 6. E5 + C5 :: E5 (#4), C5 (#5)
- 7. C4 :: C4 (#9)
- 8. G4 :: G4 (#11)
- 9. E5 + C5 :: E5 (#4), C5 (#5)
- 10. C4 :: C4 (#9)
- 11. G4 :: G4 (#11)
- 12. C5 :: C5 (#5)
- 13. D5 :: D5 (#13)
- 14. E5 + C4 :: E5 (#4), C4 (#9)
- 15. C5 :: C5 (#5)
- 16. G5 + E4 :: G5 (#3), E4 (#10)
- 17. G4 :: G4 (#11)

## Review

- summary: expected note-set と event 数が揃いました。
- reason: `12.6667-12.7520` の two-onset gap を局所 segment 化することで、`C5` のミュート/再打鍵付近の `D5` を独立 event として復元できました。

## Recapture Guidance

(not specified)

## Memo

- Event 15 (C5 at ~13.47s) は非常に短い弾き直し。C5 のミュート dip (11→0.5→13) が確認済み。ただし onset 検出器がミュート時 (13.448s) に発火し、E5 残響が primary を奪うため現在未検出。
- Event 16 は G5+E4 (E5 ではなく E4、energy trace で確認済み 2026-03-31)。
