"""Generate native-Rust reference values for the wasm equivalence check.

Runs the *native* pyo3 `kalimba_dsp` extension over a battery of inputs and
writes them — plus raw float32/uint32 array buffers — to an output directory.
The node-side `check_wasm.cjs` replays the same inputs through the *wasm* build
and asserts the outputs match. Both bindings share the same pure-Rust core, so a
mismatch means the binding glue (Float32Array / Uint32Array marshalling, i64/u32
ABI) drifted, not the algorithm.

Usage (driven by ../check_wasm.sh):
    PYTHONPATH=apps/api uv run python tools/wasm_reference.py <out_dir>

Case schema (reference.json):
    {"name", "fn", "args": [<arg>...], <output-spec>}
  arg ::= {"f32arr": file} | {"u32arr": file} | {"i64": v} | {"f64": v}
        | {"u32": v} | {"bool": v}
  output-spec ::= {"expected": scalar, "exact"|"rtol"/"atol"}
               |  {"expectedArray": file, "rtol", "atol"}
               |  {"expectedIndices": file}        # exact uint32 match

Extend `build_cases()` whenever a new shared-core primitive is exposed to wasm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import kalimba_dsp as K

try:
    from app.transcription.peaks import HARMONIC_BAND_CENTS as _H
    H = float(_H)
except Exception:  # pragma: no cover - app not importable (raw crate checkout)
    H = 40.0

FRAME_LENGTH = 2048
HOP_LENGTH = 256
N_MELS = 128


def _sine(frequency: float, sample_rate: int, duration: float, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def _multi_onset(sample_rate: int) -> np.ndarray:
    """Compact decaying-harmonic multi-note signal for onset cases."""
    rng = np.random.default_rng(7)
    n = int(sample_rate * 1.6)
    a = np.zeros(n, dtype=np.float32)
    for freq, ts in [(261.63, 0.1), (392.0, 0.6), (523.25, 1.1)]:
        d = int(sample_rate * 0.4)
        t = np.arange(d) / sample_rate
        note = (np.exp(-4.5 * t) * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        i = int(ts * sample_rate)
        seg = note[: max(0, min(d, n - i))]
        a[i : i + len(seg)] += seg
    a += (0.002 * rng.standard_normal(n)).astype(np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


def build_cases(out_dir: Path) -> list[dict]:
    arrays: dict[str, np.ndarray] = {}

    def save_f32(name: str, arr: np.ndarray) -> dict:
        a = np.ascontiguousarray(arr, dtype=np.float32)
        a.tofile(out_dir / f"{name}.f32")
        arrays[name] = a
        return {"f32arr": f"{name}.f32"}

    def save_u32(name: str, arr: np.ndarray) -> dict:
        a = np.ascontiguousarray(arr, dtype=np.uint32)
        a.tofile(out_dir / f"{name}.u32")
        return {"u32arr": f"{name}.u32"}

    def save_f64(name: str, arr: np.ndarray) -> dict:
        a = np.ascontiguousarray(arr, dtype=np.float64)
        a.tofile(out_dir / f"{name}.f64")
        arrays[name] = a
        return {"f64arr": f"{name}.f64"}

    def f64_expected(name: str, arr: np.ndarray) -> str:
        np.ascontiguousarray(arr, dtype=np.float64).tofile(out_dir / f"{name}.f64")
        return f"{name}.f64"

    def f32_expected(name: str, arr: np.ndarray) -> str:
        np.ascontiguousarray(arr, dtype=np.float32).tofile(out_dir / f"{name}.f32")
        return f"{name}.f32"

    def u32_expected(name: str, arr: np.ndarray) -> str:
        np.ascontiguousarray(arr, dtype=np.uint32).tofile(out_dir / f"{name}.u32")
        return f"{name}.u32"

    cases: list[dict] = []

    # --- note_band_energy (audio -> scalar) ---
    for sr in (44100, 96000):
        for freq in (261.63, 440.0):
            ain = save_f32(f"tone_{sr}_{int(freq)}", _sine(freq, sr, 0.5))
            for ws in (0.02, 0.08):
                exp = K.note_band_energy(arrays[f"tone_{sr}_{int(freq)}"], sr, 0.25, freq, ws, H)
                cases.append({
                    "name": f"note_band_energy/{sr}/{int(freq)}/ws{ws}",
                    "fn": "note_band_energy",
                    "args": [ain, {"i64": sr}, {"f64": 0.25}, {"f64": freq}, {"f64": ws}, {"f64": H}],
                    "expected": exp, "rtol": 1e-3, "atol": 1e-3,
                })

    # --- adaptive_n_fft (-> exact integer) ---
    for sr in (44100, 96000):
        for freq in (40.0, 1046.5):
            for chunk_len in (256, 3528, 4096):
                for min_bins in (1, 2):
                    exp = K.adaptive_n_fft(sr, freq, chunk_len, min_bins, H)
                    cases.append({
                        "name": f"adaptive_n_fft/{sr}/{int(freq)}/{chunk_len}/{min_bins}",
                        "fn": "adaptive_n_fft",
                        "args": [{"i64": sr}, {"f64": freq}, {"u32": chunk_len}, {"u32": min_bins}, {"f64": H}],
                        "expected": exp, "exact": True,
                    })

    # --- broadband spectral energy (f64 in -> f64 scalar / array) ---
    for sr, n_fft in [(44100, 4096), (96000, 8192)]:
        rng = np.random.default_rng(sr)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        spec = np.abs(rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs)))
        fin = save_f64(f"bb_freqs_{sr}", freqs)
        sin = save_f64(f"bb_spec_{sr}", spec)
        centers = np.array([261.63, 392.0, 523.25, 1046.5, 40.0, 0.0], dtype=np.float64)
        cin = save_f64(f"bb_centers_{sr}", centers)
        farr = arrays[f"bb_freqs_{sr}"]
        sarr = arrays[f"bb_spec_{sr}"]
        carr = arrays[f"bb_centers_{sr}"]
        bexp = np.asarray(K.batch_peak_energies(farr, sarr, carr, H), dtype=np.float64)
        cases.append({
            "name": f"batch_peak_energies/{sr}", "fn": "batch_peak_energies",
            "args": [fin, sin, cin, {"f64": H}],
            "expectedArrayF64": f64_expected(f"bb_batch_{sr}", bexp), "rtol": 1e-9, "atol": 1e-12,
        })
        for c in (261.63, 1046.5, 40.0):
            pexp = K.peak_energy_near(farr, sarr, float(c), H)
            cases.append({
                "name": f"peak_energy_near/{sr}/{int(c)}", "fn": "peak_energy_near",
                "args": [fin, sin, {"f64": c}, {"f64": H}],
                "expected": pexp, "rtol": 1e-9, "atol": 1e-12,
            })

    # --- pitch: chunk_spectrum (f32 audio -> f64 magnitude spectrum) ---
    for sr, n_fft in [(48000, 4096), (96000, 8192)]:
        rng = np.random.default_rng(sr ^ n_fft)
        for chunk_len in (2048, 3528):
            nm = f"cs_chunk_{sr}_{n_fft}_{chunk_len}"
            cin = save_f32(nm, (0.5 * rng.standard_normal(chunk_len)).astype(np.float32))
            cexp = np.asarray(K.chunk_spectrum(arrays[nm], sr, n_fft), dtype=np.float64)
            cases.append({
                "name": f"chunk_spectrum/{sr}/{n_fft}/{chunk_len}", "fn": "chunk_spectrum",
                "args": [cin, {"i64": sr}, {"u32": n_fft}],
                "expectedArrayF64": f64_expected(f"cs_spec_{sr}_{n_fft}_{chunk_len}", cexp),
                "rtol": 1e-9, "atol": 1e-12,
            })

    # --- pitch: rank_tuning_candidates (integer-comb per-note scores) ---
    # 17-key C-major-ish frequency span. Values only need to be identical between
    # the wasm and native call (this harness checks binding-glue parity, not tuning
    # correctness — that is covered by apps/api/tests/test_chunk_spectrum_rust.py).
    note_freqs = np.array([
        261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25, 587.33,
        659.25, 698.46, 783.99, 880.0, 987.77, 1046.5, 1174.66, 1318.51,
    ], dtype=np.float64)
    for sr, n_fft, seed in [(48000, 8192, 3), (96000, 8192, 9)]:
        rng = np.random.default_rng(seed)
        rfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float64)
        rspec = np.zeros_like(rfreqs)
        for nf in note_freqs[::4]:
            for m in (1, 2, 3):
                b = int(round(nf * m / (sr / n_fft)))
                if 0 <= b < len(rspec):
                    rspec[b] += 1.0 / m
        rspec += 0.01 * np.abs(rng.standard_normal(len(rfreqs)))
        fin = save_f64(f"rk_freqs_{sr}", rfreqs)
        sin = save_f64(f"rk_spec_{sr}", rspec)
        nin = save_f64(f"rk_notes_{sr}", note_freqs)
        rexp = np.asarray(
            K.rank_tuning_candidates(arrays[f"rk_freqs_{sr}"], arrays[f"rk_spec_{sr}"], arrays[f"rk_notes_{sr}"], H),
            dtype=np.float64,
        )
        cases.append({
            "name": f"rank_tuning_candidates/{sr}", "fn": "rank_tuning_candidates",
            "args": [fin, sin, nin, {"f64": H}],
            "expectedArrayF64": f64_expected(f"rk_scores_{sr}", rexp), "rtol": 1e-9, "atol": 1e-12,
        })

    # --- mel_filterbank (-> flat f32 matrix) ---
    for sr, n_fft, n_mels in [(44100, 2048, 128), (96000, 2048, 128), (48000, 1024, 64)]:
        exp = np.asarray(K.mel_filterbank(sr, n_fft, n_mels), dtype=np.float32)
        cases.append({
            "name": f"mel_filterbank/{sr}/{n_fft}/{n_mels}",
            "fn": "mel_filterbank",
            "args": [{"i64": sr}, {"u32": n_fft}, {"u32": n_mels}],
            "expectedArray": f32_expected(f"mel_{sr}_{n_fft}_{n_mels}", exp),
            "rtol": 1e-5, "atol": 1e-6,
        })

    # --- onset DSP on a shared multi-onset signal ---
    for sr in (44100, 96000):
        audio = _multi_onset(sr)
        ain = save_f32(f"onset_audio_{sr}", audio)

        rms_exp = np.asarray(K.rms(audio, FRAME_LENGTH, HOP_LENGTH), dtype=np.float32)
        cases.append({
            "name": f"rms/{sr}", "fn": "rms",
            "args": [ain, {"u32": FRAME_LENGTH}, {"u32": HOP_LENGTH}],
            "expectedArray": f32_expected(f"rms_{sr}", rms_exp), "rtol": 1e-4, "atol": 1e-5,
        })

        env = np.asarray(K.onset_strength(audio, sr, HOP_LENGTH, FRAME_LENGTH, N_MELS), dtype=np.float32)
        cases.append({
            "name": f"onset_strength/{sr}", "fn": "onset_strength",
            "args": [ain, {"i64": sr}, {"u32": HOP_LENGTH}, {"u32": FRAME_LENGTH}, {"u32": N_MELS}],
            "expectedArray": f32_expected(f"onset_env_{sr}", env), "rtol": 1e-3, "atol": 1e-3,
        })

        env_in = save_f32(f"env_{sr}", env)
        det = np.asarray(K.onset_detect(env, sr, HOP_LENGTH, True), dtype=np.uint32)
        cases.append({
            "name": f"onset_detect/{sr}", "fn": "onset_detect",
            "args": [env_in, {"i64": sr}, {"u32": HOP_LENGTH}, {"bool": True}],
            "expectedIndices": u32_expected(f"det_{sr}", det),
        })

    # --- peak_pick / onset_backtrack on small constructed arrays ---
    x = np.array([0.0, 0.1, 0.9, 0.2, 0.05, 0.3, 0.95, 0.4, 0.1, 0.0, 0.8, 0.2], dtype=np.float32)
    pp = np.asarray(K.peak_pick(x, 2, 2, 3, 3, 0.05, 2), dtype=np.uint32)
    cases.append({
        "name": "peak_pick/constructed", "fn": "peak_pick",
        "args": [save_f32("pp_x", x), {"u32": 2}, {"u32": 2}, {"u32": 3}, {"u32": 3}, {"f64": 0.05}, {"u32": 2}],
        "expectedIndices": u32_expected("pp_out", pp),
    })
    energy = np.array([0.5, 0.2, 0.4, 0.9, 0.3, 0.1, 0.6, 0.95, 0.2, 0.7], dtype=np.float32)
    events = np.array([3, 7, 9], dtype=np.uint32)
    bt = np.asarray(K.onset_backtrack([int(e) for e in events], energy), dtype=np.uint32)
    cases.append({
        "name": "onset_backtrack/constructed", "fn": "onset_backtrack",
        "args": [save_u32("bt_events", events), save_f32("bt_energy", energy)],
        "expectedIndices": u32_expected("bt_out", bt),
    })

    return cases


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: wasm_reference.py <out_dir>", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(out_dir)
    (out_dir / "reference.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
