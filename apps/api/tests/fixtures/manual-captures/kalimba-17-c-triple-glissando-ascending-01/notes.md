# Manual Notes

- tester: manual
- verdict: completed
- scenario: 2026-03-22-triple-glissando-kalimba-17-c
- expected note: C4 + E4 + G4 / A4 + F4 + D4 / E4 + G4 + B4 / C5 + A4 + F4 / G4 + B4 + D5 / E5 + C5 + A4 / B4 + D5 + F5 / G5 + E5 + C5
- captured at: 2026-03-21T16:05:17.078Z

## Expected Performance

- summary: C4 + E4 + G4 / A4 + F4 + D4 / E4 + G4 + B4 / C5 + A4 + F4 / G4 + B4 + D5 / E5 + C5 + A4 / B4 + D5 + F5 / G5 + E5 + C5

### Events

- 1. C4 + E4 + G4 :: C4 (#9), E4 (#10), G4 (#11)
- 2. A4 + F4 + D4 :: A4 (#6), F4 (#7), D4 (#8)
- 3. E4 + G4 + B4 :: E4 (#10), G4 (#11), B4 (#12) — E4 は弱い打鍵（energy trace: E4=365 vs G4=994/B4=1411）だが意図した演奏ノート
- 4. C5 + A4 + F4 :: C5 (#5), A4 (#6), F4 (#7)
- 5. G4 + B4 + D5 :: G4 (#11), B4 (#12), D5 (#13)
- 6. E5 + C5 + A4 :: E5 (#4), C5 (#5), A4 (#6)
- 7. B4 + D5 + F5 :: B4 (#12), D5 (#13), F5 (#14)
- 8. G5 + E5 + C5 :: G5 (#3), E5 (#4), C5 (#5)

## Memo

C4~G5までの3音 slide_chord で上がっていくパターンのテスト

### Event 3: E4 は弱い打鍵 (2026-04-05 耳確認・分析)

E4 は意図した演奏ノートだが弱い打鍵。energy trace: G4=994, B4=1411 に対し E4=365。
og=55.37 で backward_attack_gain gate onset override (>=20.0) により rescue される。
backward_gain=15.66 < 20.0 だが onset_gain が十分高いため genuine attack と判定。
×2 adaptive FFT では E4 の og が 15.45 に低下し正しく棄却される（#105 sweep で確認）。
