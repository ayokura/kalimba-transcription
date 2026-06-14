// Browser-side pitch identification via the kalimba-dsp WebAssembly module.
//
// The step after onset detection (see ./onset.ts): for each detected onset, slice
// an analysis window, compute the f64 magnitude spectrum (chunk_spectrum), build
// the bin-frequency ramp, and rank the tuning's notes (rank_tuning_candidates —
// the recognizer's integer-harmonic-comb default path). The top-scoring note is the
// identified pitch. Same Rust core the API server uses, zero server round-trip.
//
// Parity with the server recognizer is guarded by
// crates/kalimba-dsp/check_wasm.sh and apps/api/tests/test_chunk_spectrum_rust.py.
//
// Regenerate the vendored .wasm/glue per ./onset.ts (the build + copy is shared).
import init, { chunk_spectrum, rank_tuning_candidates, adaptive_n_fft } from "./kalimba_dsp";
import { KALIMBA_17C_TUNING, type TuningNote } from "./tuning-17c";

// Must match apps/api/app/transcription/constants.py (HARMONIC_BAND_CENTS).
const HARMONIC_BAND_CENTS = 40.0;
// adaptive_n_fft min-bins inside the harmonic band — matches the recognizer's
// segment-peak path (_adaptive_n_fft min_bins=2).
const MIN_BINS = 2;
// Per-onset analysis window cap (seconds). Demo heuristic: the chunk runs from the
// onset to the next onset, capped here. The server uses a segment-aware window —
// if note ID looks off, this is the knob to refine (see plan section 7), not the port.
const MAX_WINDOW_SEC = 0.25;
// chunk_spectrum / the recognizer skip windows shorter than this.
const MIN_CHUNK_SAMPLES = 256;

// init() is idempotent (the vendored glue guards on `wasm !== undefined`), so a
// separate memoized promise here cannot double-load the module.
let wasmReady: Promise<void> | null = null;
function ensureWasm(): Promise<void> {
  if (!wasmReady) {
    wasmReady = init().then(() => undefined);
  }
  return wasmReady;
}

export type IdentifiedNote = {
  onsetTimeSec: number;
  noteName: string;
  key: number;
  score: number;
};

export type PitchResult = {
  notes: IdentifiedNote[];
  elapsedMs: number;
};

/**
 * Identify the top-1 note at each onset, entirely in the browser.
 *
 * `samples` is a mono Float32Array (e.g. AudioBuffer.getChannelData(0)),
 * `onsetTimesSec` the output of {@link detectOnsetsInBrowser}. Returns one
 * identified note per usable onset window (windows shorter than ~256 samples,
 * e.g. two onsets closer than the chunk floor, are skipped).
 *
 * This is monophonic top-1 identification — `rank_tuning_candidates` returns the
 * single strongest pitch per window, not a polyphonic chord.
 */
export async function identifyNotesInBrowser(
  samples: Float32Array,
  sampleRate: number,
  onsetTimesSec: number[],
  tuning: TuningNote[] = KALIMBA_17C_TUNING,
): Promise<PitchResult> {
  await ensureWasm();
  const sr = Math.round(sampleRate);
  const srBig = BigInt(sr);
  const noteFreqs = new Float64Array(tuning.map((n) => n.frequency));
  const minFreq = Math.min(...tuning.map((n) => n.frequency));
  const maxWindowSamples = Math.round(MAX_WINDOW_SEC * sr);

  const t0 = performance.now();
  const notes: IdentifiedNote[] = [];
  for (let i = 0; i < onsetTimesSec.length; i++) {
    const startSample = Math.max(0, Math.round(onsetTimesSec[i] * sr));
    const nextOnsetSample =
      i + 1 < onsetTimesSec.length
        ? Math.round(onsetTimesSec[i + 1] * sr)
        : samples.length;
    const endSample = Math.min(startSample + maxWindowSamples, nextOnsetSample, samples.length);
    const chunkLen = endSample - startSample;
    if (chunkLen < MIN_CHUNK_SAMPLES) continue;

    const chunk = samples.subarray(startSample, endSample);
    const nFft = adaptive_n_fft(srBig, minFreq, chunkLen, MIN_BINS, HARMONIC_BAND_CENTS);
    const spectrum = chunk_spectrum(chunk, srBig, nFft);
    // rfft bin frequencies for bins 0..nFft/2 == k*sr/nFft (== np.fft.rfftfreq).
    const frequencies = new Float64Array(spectrum.length);
    for (let k = 0; k < frequencies.length; k++) {
      frequencies[k] = (k * sr) / nFft;
    }
    const scores = rank_tuning_candidates(frequencies, spectrum, noteFreqs, HARMONIC_BAND_CENTS);
    let best = 0;
    for (let j = 1; j < scores.length; j++) {
      if (scores[j] > scores[best]) best = j;
    }
    notes.push({
      onsetTimeSec: onsetTimesSec[i],
      noteName: tuning[best].noteName,
      key: tuning[best].key,
      score: scores[best],
    });
  }
  const elapsedMs = performance.now() - t0;
  return { notes, elapsedMs };
}
