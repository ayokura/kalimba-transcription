// Browser-side per-note band-energy trace via the kalimba-dsp WASM module (#205).
//
// Same vendored-artifact pattern as onset.ts / pitch.ts: the Rust core
// (band_energy_trace_core in crates/kalimba-dsp/src/lib.rs) samples
// note_band_energy on a uniform time grid. Semantics are the recognizer's
// (±HARMONIC_BAND_CENTS band, peak FFT magnitude), NOT energy_trace.py's
// ±Hz band-power sum — the review UI should look at the quantity the
// recognizer actually scores.
import init, { band_energy_trace } from "./kalimba_dsp";

// Recognizer defaults (must match apps/api/app/transcription/constants.py).
const HARMONIC_BAND_CENTS = 40.0;
// note_band_energy の評価窓。recognizer の segment 系と同オーダーの短窓。
const WINDOW_SECONDS = 0.05;

let wasmReady: Promise<void> | null = null;

function ensureWasm(): Promise<void> {
  if (!wasmReady) {
    wasmReady = init().then(() => undefined);
  }
  return wasmReady;
}

export type NoteEnergyTrace = {
  startSec: number;
  stepSec: number;
  steps: number;
  /** frequency-major: values[noteIndex * steps + stepIndex] */
  values: Float32Array;
  elapsedMs: number;
};

/**
 * Compute per-note band-energy traces over [startSec, startSec + durationSec).
 * `frequencies` follows the caller's note order; the result is frequency-major.
 * Synchronous WASM under the hood (v1: main thread — a Worker is the v2 path
 * if long windows on low-end phones become an issue; see #205).
 */
export async function traceNoteBandEnergies(
  samples: Float32Array,
  sampleRate: number,
  frequencies: number[],
  startSec: number,
  durationSec: number,
  stepSec: number,
): Promise<NoteEnergyTrace> {
  await ensureWasm();
  const t0 = performance.now();
  const values = band_energy_trace(
    samples,
    BigInt(Math.round(sampleRate)),
    new Float64Array(frequencies),
    startSec,
    durationSec,
    stepSec,
    WINDOW_SECONDS,
    HARMONIC_BAND_CENTS,
  );
  const steps = frequencies.length > 0 ? values.length / frequencies.length : 0;
  return {
    startSec,
    stepSec,
    steps,
    values,
    elapsedMs: performance.now() - t0,
  };
}
