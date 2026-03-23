# Manual Notes

- tester: manual
- verdict: reference_only
- scenario: 2026-03-22-a4-plus-f4-plus-d4-repeat-05-kalimba-17-c
- expected note: E4 + G4 + B4 + D5 x 5
- captured at: 2026-03-22T06:09:57.001Z

## Expected Performance

- summary: E4 + G4 + B4 + D5 x 5

### Events

- 1. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 2. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 3. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 4. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)
- 5. E4 + G4 + B4 + D5 :: E4 (#10), G4 (#11), B4 (#12), D5 (#13)

## Memo

normal speed

## Review Update (2026-03-23)

- fixture status moved to `reference_only`
- original scenario metadata is incorrect and capture intent was never recorded
- this case is now superseded by explicit strict / slide_chord captures for the same note set
- treat this fixture as legacy reference evidence only; do not use it as a primary regression target

## Independent Audit (2026-03-23)

- Independent audit confirms the note set can be plausible, but legacy scenario metadata and missing captureIntent still make this fixture unsuitable as a primary regression target.



