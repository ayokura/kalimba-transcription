/* tslint:disable */
/* eslint-disable */

/**
 * FFT size giving >= `min_bins` bins inside the ±`harmonic_band_cents` band.
 * `u32` I/O for natural JS `number` interop (n_fft fits well within u32).
 */
export function adaptive_n_fft(sample_rate: bigint, frequency: number, chunk_len: number, min_bins: number, harmonic_band_cents: number): number;

/**
 * Batched `peak_energy_near` over many center frequencies.
 */
export function batch_peak_energies(frequencies: Float64Array, spectrum: Float64Array, center_freqs: Float64Array, band_cents: number): Float64Array;

export function detect_gap_rise_attack(audio: Float32Array, sample_rate: bigint, gap_start: number, gap_end: number, frequency: number, window_seconds: number, pre_offset: number, post_offset: number, rise_ratio: number, min_post_energy: number, min_pre_energy: number, harmonic_band_cents: number): number | undefined;

/**
 * Slaney mel filterbank, row-major `n_mels * (n_fft/2+1)` Float32Array.
 */
export function mel_filterbank(sample_rate: bigint, n_fft: number, n_mels: number): Float32Array;

/**
 * Peak FFT magnitude in `frequency`'s ±`harmonic_band_cents` band within a
 * window centered on `center_time`. Shared core with the pyo3 binding; the
 * browser pipeline calls this on a JS `Float32Array` of decoded audio.
 */
export function note_band_energy(audio: Float32Array, sample_rate: bigint, center_time: number, frequency: number, window_seconds: number, harmonic_band_cents: number): number;

/**
 * Snap onset events back to preceding local energy minima.
 */
export function onset_backtrack(events: Uint32Array, energy: Float32Array): Uint32Array;

/**
 * Full onset-frame detection (normalise -> peak-pick -> backtrack).
 */
export function onset_detect(onset_env: Float32Array, sample_rate: bigint, hop_length: number, backtrack: boolean): Uint32Array;

/**
 * Mel spectral-flux onset strength envelope.
 */
export function onset_strength(audio: Float32Array, sample_rate: bigint, hop_length: number, n_fft: number, n_mels: number): Float32Array;

/**
 * Peak magnitude within `±band_cents` of `center_freq` over a precomputed spectrum.
 */
export function peak_energy_near(frequencies: Float64Array, spectrum: Float64Array, center_freq: number, band_cents: number): number;

/**
 * Greedy peak picker; returns peak frame indices (Uint32Array).
 */
export function peak_pick(x: Float32Array, pre_max: number, post_max: number, pre_avg: number, post_avg: number, delta: number, wait: number): Uint32Array;

/**
 * Frame-wise RMS energy (center=True, constant pad).
 */
export function rms(audio: Float32Array, frame_length: number, hop_length: number): Float32Array;

export function scan_gap_for_mute_dip_with_window(audio: Float32Array, sample_rate: bigint, gap_start: number, gap_end: number, frequency: number, window_seconds: number, mute_dip_energy_window: number, max_dip_window: number, max_recovery_window: number, coarse_step: number, fine_step: number, min_pre_energy: number, max_dip_ratio: number, min_post_energy: number, min_recovery_ratio: number, harmonic_band_cents: number): number | undefined;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly adaptive_n_fft: (a: bigint, b: number, c: number, d: number, e: number) => number;
    readonly batch_peak_energies: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly detect_gap_rise_attack: (a: number, b: number, c: bigint, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => [number, number];
    readonly mel_filterbank: (a: bigint, b: number, c: number) => [number, number];
    readonly note_band_energy: (a: number, b: number, c: bigint, d: number, e: number, f: number, g: number) => number;
    readonly onset_backtrack: (a: number, b: number, c: number, d: number) => [number, number];
    readonly onset_detect: (a: number, b: number, c: bigint, d: number, e: number) => [number, number];
    readonly onset_strength: (a: number, b: number, c: bigint, d: number, e: number, f: number) => [number, number];
    readonly peak_energy_near: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly peak_pick: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number];
    readonly rms: (a: number, b: number, c: number, d: number) => [number, number];
    readonly scan_gap_for_mute_dip_with_window: (a: number, b: number, c: bigint, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
