// AUTO-GENERATED from apps/api/app/tunings.py (kalimba-17-c). Do not hand-edit.
// Regenerate per docs/wasm-pitch-id-port-plan.md section 6. The browser pitch-ID
// demo passes these note frequencies to rank_tuning_candidates (integer-comb path,
// which ignores per-tine partials, so only key/noteName/frequency are needed).
export type TuningNote = { key: number; noteName: string; frequency: number };

// 17 Key C Major (17 keys)
export const KALIMBA_17C_TUNING: TuningNote[] = [
  { key: 1, noteName: "D6", frequency: 1174.659072 },
  { key: 2, noteName: "B5", frequency: 987.766603 },
  { key: 3, noteName: "G5", frequency: 783.990872 },
  { key: 4, noteName: "E5", frequency: 659.255114 },
  { key: 5, noteName: "C5", frequency: 523.251131 },
  { key: 6, noteName: "A4", frequency: 440.0 },
  { key: 7, noteName: "F4", frequency: 349.228231 },
  { key: 8, noteName: "D4", frequency: 293.664768 },
  { key: 9, noteName: "C4", frequency: 261.625565 },
  { key: 10, noteName: "E4", frequency: 329.627557 },
  { key: 11, noteName: "G4", frequency: 391.995436 },
  { key: 12, noteName: "B4", frequency: 493.883301 },
  { key: 13, noteName: "D5", frequency: 587.329536 },
  { key: 14, noteName: "F5", frequency: 698.456463 },
  { key: 15, noteName: "A5", frequency: 880.0 },
  { key: 16, noteName: "C6", frequency: 1046.502261 },
  { key: 17, noteName: "E6", frequency: 1318.510228 },
];
