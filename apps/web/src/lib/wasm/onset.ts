// Browser-side onset detection via the kalimba-dsp WebAssembly module.
//
// This is the zero-server-round-trip proof of the WASM pipeline: the same Rust
// shared core that the API server delegates to (crates/kalimba-dsp/src/onset.rs)
// runs in the browser on a WebAudio-decoded Float32Array.
//
// The .wasm + glue (./kalimba_dsp.{js,d.ts}, ./kalimba_dsp_bg.wasm) are vendored
// build artifacts, co-located so Turbopack resolves the glue's
// `new URL('kalimba_dsp_bg.wasm', import.meta.url)` as a bundled asset.
// Regenerate with:
//   wasm-pack build crates/kalimba-dsp --target web -- --no-default-features --features wasm
// then copy pkg/kalimba_dsp.{js,d.ts} + pkg/kalimba_dsp_bg.wasm -> this directory.
import init, { onset_strength, onset_detect } from "./kalimba_dsp";

// Recognizer defaults (must match apps/api/app/transcription/constants.py).
const HOP_LENGTH = 256;
const N_FFT = 2048;
const N_MELS = 128;

let wasmReady: Promise<void> | null = null;

function ensureWasm(): Promise<void> {
  if (!wasmReady) {
    // No arg -> the glue resolves the co-located .wasm via
    // `new URL('kalimba_dsp_bg.wasm', import.meta.url)`, which Turbopack emits
    // as a bundled asset.
    wasmReady = init().then(() => undefined);
  }
  return wasmReady;
}

export type OnsetResult = {
  onsetTimesSec: number[];
  frameCount: number;
  elapsedMs: number;
};

/**
 * Run mel spectral-flux onset detection entirely in the browser.
 * `samples` is a mono Float32Array (e.g. AudioBuffer.getChannelData(0)).
 *
 * Note: WebAudio's `decodeAudioData` resamples to the AudioContext's sample
 * rate (commonly 44.1 kHz), so a 96 kHz source is processed at the context
 * rate. The recognizer is sample-rate-robust, so onsets still land within a
 * few ms of the server's; for exact frame parity with a native-rate server
 * run, decode through an `OfflineAudioContext` at the file's sample rate.
 */
export async function detectOnsetsInBrowser(
  samples: Float32Array,
  sampleRate: number,
): Promise<OnsetResult> {
  await ensureWasm();
  const sr = BigInt(Math.round(sampleRate));
  const t0 = performance.now();
  const env = onset_strength(samples, sr, HOP_LENGTH, N_FFT, N_MELS);
  const frames = onset_detect(env, sr, HOP_LENGTH, true);
  const elapsedMs = performance.now() - t0;
  const onsetTimesSec = Array.from(frames, (frame) => (Number(frame) * HOP_LENGTH) / sampleRate);
  return { onsetTimesSec, frameCount: env.length, elapsedMs };
}
