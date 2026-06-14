/* @ts-self-types="./kalimba_dsp.d.ts" */

/**
 * FFT size giving >= `min_bins` bins inside the ±`harmonic_band_cents` band.
 * `u32` I/O for natural JS `number` interop (n_fft fits well within u32).
 * @param {bigint} sample_rate
 * @param {number} frequency
 * @param {number} chunk_len
 * @param {number} min_bins
 * @param {number} harmonic_band_cents
 * @returns {number}
 */
export function adaptive_n_fft(sample_rate, frequency, chunk_len, min_bins, harmonic_band_cents) {
    const ret = wasm.adaptive_n_fft(sample_rate, frequency, chunk_len, min_bins, harmonic_band_cents);
    return ret >>> 0;
}

/**
 * Batched `peak_energy_near` over many center frequencies.
 * @param {Float64Array} frequencies
 * @param {Float64Array} spectrum
 * @param {Float64Array} center_freqs
 * @param {number} band_cents
 * @returns {Float64Array}
 */
export function batch_peak_energies(frequencies, spectrum, center_freqs, band_cents) {
    const ptr0 = passArrayF64ToWasm0(frequencies, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(spectrum, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(center_freqs, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ret = wasm.batch_peak_energies(ptr0, len0, ptr1, len1, ptr2, len2, band_cents);
    var v4 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v4;
}

/**
 * @param {Float32Array} audio
 * @param {bigint} sample_rate
 * @param {number} gap_start
 * @param {number} gap_end
 * @param {number} frequency
 * @param {number} window_seconds
 * @param {number} pre_offset
 * @param {number} post_offset
 * @param {number} rise_ratio
 * @param {number} min_post_energy
 * @param {number} min_pre_energy
 * @param {number} harmonic_band_cents
 * @returns {number | undefined}
 */
export function detect_gap_rise_attack(audio, sample_rate, gap_start, gap_end, frequency, window_seconds, pre_offset, post_offset, rise_ratio, min_post_energy, min_pre_energy, harmonic_band_cents) {
    const ptr0 = passArrayF32ToWasm0(audio, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.detect_gap_rise_attack(ptr0, len0, sample_rate, gap_start, gap_end, frequency, window_seconds, pre_offset, post_offset, rise_ratio, min_post_energy, min_pre_energy, harmonic_band_cents);
    return ret[0] === 0 ? undefined : ret[1];
}

/**
 * Slaney mel filterbank, row-major `n_mels * (n_fft/2+1)` Float32Array.
 * @param {bigint} sample_rate
 * @param {number} n_fft
 * @param {number} n_mels
 * @returns {Float32Array}
 */
export function mel_filterbank(sample_rate, n_fft, n_mels) {
    const ret = wasm.mel_filterbank(sample_rate, n_fft, n_mels);
    var v1 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v1;
}

/**
 * Peak FFT magnitude in `frequency`'s ±`harmonic_band_cents` band within a
 * window centered on `center_time`. Shared core with the pyo3 binding; the
 * browser pipeline calls this on a JS `Float32Array` of decoded audio.
 * @param {Float32Array} audio
 * @param {bigint} sample_rate
 * @param {number} center_time
 * @param {number} frequency
 * @param {number} window_seconds
 * @param {number} harmonic_band_cents
 * @returns {number}
 */
export function note_band_energy(audio, sample_rate, center_time, frequency, window_seconds, harmonic_band_cents) {
    const ptr0 = passArrayF32ToWasm0(audio, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.note_band_energy(ptr0, len0, sample_rate, center_time, frequency, window_seconds, harmonic_band_cents);
    return ret;
}

/**
 * Snap onset events back to preceding local energy minima.
 * @param {Uint32Array} events
 * @param {Float32Array} energy
 * @returns {Uint32Array}
 */
export function onset_backtrack(events, energy) {
    const ptr0 = passArray32ToWasm0(events, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF32ToWasm0(energy, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.onset_backtrack(ptr0, len0, ptr1, len1);
    var v3 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v3;
}

/**
 * Full onset-frame detection (normalise -> peak-pick -> backtrack).
 * @param {Float32Array} onset_env
 * @param {bigint} sample_rate
 * @param {number} hop_length
 * @param {boolean} backtrack
 * @returns {Uint32Array}
 */
export function onset_detect(onset_env, sample_rate, hop_length, backtrack) {
    const ptr0 = passArrayF32ToWasm0(onset_env, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.onset_detect(ptr0, len0, sample_rate, hop_length, backtrack);
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Mel spectral-flux onset strength envelope.
 * @param {Float32Array} audio
 * @param {bigint} sample_rate
 * @param {number} hop_length
 * @param {number} n_fft
 * @param {number} n_mels
 * @returns {Float32Array}
 */
export function onset_strength(audio, sample_rate, hop_length, n_fft, n_mels) {
    const ptr0 = passArrayF32ToWasm0(audio, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.onset_strength(ptr0, len0, sample_rate, hop_length, n_fft, n_mels);
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Peak magnitude within `±band_cents` of `center_freq` over a precomputed spectrum.
 * @param {Float64Array} frequencies
 * @param {Float64Array} spectrum
 * @param {number} center_freq
 * @param {number} band_cents
 * @returns {number}
 */
export function peak_energy_near(frequencies, spectrum, center_freq, band_cents) {
    const ptr0 = passArrayF64ToWasm0(frequencies, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(spectrum, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ret = wasm.peak_energy_near(ptr0, len0, ptr1, len1, center_freq, band_cents);
    return ret;
}

/**
 * Greedy peak picker; returns peak frame indices (Uint32Array).
 * @param {Float32Array} x
 * @param {number} pre_max
 * @param {number} post_max
 * @param {number} pre_avg
 * @param {number} post_avg
 * @param {number} delta
 * @param {number} wait
 * @returns {Uint32Array}
 */
export function peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait) {
    const ptr0 = passArrayF32ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.peak_pick(ptr0, len0, pre_max, post_max, pre_avg, post_avg, delta, wait);
    var v2 = getArrayU32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Frame-wise RMS energy (center=True, constant pad).
 * @param {Float32Array} audio
 * @param {number} frame_length
 * @param {number} hop_length
 * @returns {Float32Array}
 */
export function rms(audio, frame_length, hop_length) {
    const ptr0 = passArrayF32ToWasm0(audio, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rms(ptr0, len0, frame_length, hop_length);
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * @param {Float32Array} audio
 * @param {bigint} sample_rate
 * @param {number} gap_start
 * @param {number} gap_end
 * @param {number} frequency
 * @param {number} window_seconds
 * @param {number} mute_dip_energy_window
 * @param {number} max_dip_window
 * @param {number} max_recovery_window
 * @param {number} coarse_step
 * @param {number} fine_step
 * @param {number} min_pre_energy
 * @param {number} max_dip_ratio
 * @param {number} min_post_energy
 * @param {number} min_recovery_ratio
 * @param {number} harmonic_band_cents
 * @returns {number | undefined}
 */
export function scan_gap_for_mute_dip_with_window(audio, sample_rate, gap_start, gap_end, frequency, window_seconds, mute_dip_energy_window, max_dip_window, max_recovery_window, coarse_step, fine_step, min_pre_energy, max_dip_ratio, min_post_energy, min_recovery_ratio, harmonic_band_cents) {
    const ptr0 = passArrayF32ToWasm0(audio, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.scan_gap_for_mute_dip_with_window(ptr0, len0, sample_rate, gap_start, gap_end, frequency, window_seconds, mute_dip_energy_window, max_dip_window, max_recovery_window, coarse_step, fine_step, min_pre_energy, max_dip_ratio, min_post_energy, min_recovery_ratio, harmonic_band_cents);
    return ret[0] === 0 ? undefined : ret[1];
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./kalimba_dsp_bg.js": import0,
    };
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

function getArrayU32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('kalimba_dsp_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
