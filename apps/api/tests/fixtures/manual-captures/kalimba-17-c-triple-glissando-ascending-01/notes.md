# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-22-triple-glissando-kalimba-17-c
- expected note: C4 + E4 + G4 / A4 + F4 + D4 / G4 + B4 / C5 + A4 + F4 / G4 + B4 + D5 / E5 + C5 + A4 / B4 + D5 + F5 / G5 + E5 + C5
- captured at: 2026-03-21T16:05:17.078Z

## Expected Performance

- summary: C4 + E4 + G4 / A4 + F4 + D4 / G4 + B4 / C5 + A4 + F4 / G4 + B4 + D5 / E5 + C5 + A4 / B4 + D5 + F5 / G5 + E5 + C5

### Events

- 1. C4 + E4 + G4 :: C4 (#9), E4 (#10), G4 (#11)
- 2. A4 + F4 + D4 :: A4 (#6), F4 (#7), D4 (#8)
- 3. G4 + B4 :: G4 (#11), B4 (#12) — E4 (#10) は打鍵意図ありだが発音していない（耳確認済み、energy trace で E4=365 vs G4=994/B4=1411 で onset transient のみ）
- 4. C5 + A4 + F4 :: C5 (#5), A4 (#6), F4 (#7)
- 5. G4 + B4 + D5 :: G4 (#11), B4 (#12), D5 (#13)
- 6. E5 + C5 + A4 :: E5 (#4), C5 (#5), A4 (#6)
- 7. B4 + D5 + F5 :: B4 (#12), D5 (#13), F5 (#14)
- 8. G5 + E5 + C5 :: G5 (#3), E5 (#4), C5 (#5)

## Memo

C4~G5までの3音 slide_chord で上がっていくパターンのテスト

### Event 3: E4 は false positive (2026-04-05 耳確認)

本来は E4+G4+B4 の3音コードを意図した打鍵だが、E4 は指が触れた程度のかすりで発音していない。
耳確認 + energy trace で確認: G4=994, B4=1411 に対し E4=365 で onset transient のみ、直後に 25 に急落。
正しい期待値は G4+B4 だが、current recognizer (×1 adaptive FFT) は contiguous-tertiary-extension で
E4 を rescue (og=55.37) するため、expected.json は現状の出力 E4+G4+B4 を維持。
×2 adaptive FFT では E4 の og が 15.45 に低下し正しく棄却される（#105 sweep で確認）。
