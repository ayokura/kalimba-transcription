"""Generate native-Rust reference values for the wasm equivalence check.

Runs the *native* pyo3 `kalimba_dsp` extension over a battery of inputs and
writes them — plus the raw float32 audio buffers — to an output directory. The
node-side `check_wasm.cjs` then replays the same inputs through the *wasm* build
and asserts the outputs match. Both bindings share the same pure-Rust core, so a
mismatch means the binding glue (Float32Array marshalling, i64/u32 ABI) drifted.

Usage (driven by ../check_wasm.sh):
    PYTHONPATH=apps/api uv run python tools/wasm_reference.py <out_dir>

Extend `build_cases()` whenever a new shared-core primitive is exposed to wasm
(e.g. the R3 segments STFT/onset ports).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import kalimba_dsp

try:
    # Keep the band constant in lock-step with the recognizer's value.
    from app.transcription.peaks import HARMONIC_BAND_CENTS as _H
    H = float(_H)
except Exception:  # pragma: no cover - app not importable (raw crate checkout)
    H = 40.0


def _sine(frequency: float, sample_rate: int, duration: float = 0.5, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.ascontiguousarray((amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32))


def build_cases(out_dir: Path) -> dict:
    audio_files: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}

    def audio(name: str, arr: np.ndarray) -> str:
        if name not in audio_files:
            arr.tofile(out_dir / f"{name}.f32")
            audio_files[name] = {"file": f"{name}.f32", "n": int(arr.size)}
            arrays[name] = arr
        return name

    cases: list[dict] = []

    # note_band_energy: peak band magnitude on pure tones (audio -> scalar).
    for sr in (44100, 96000):
        for freq in (261.63, 440.0, 523.25):
            name = audio(f"tone_{sr}_{int(freq)}", _sine(freq, sr))
            for ws in (0.02, 0.08):
                for ct in (0.1, 0.25):
                    expected = kalimba_dsp.note_band_energy(arrays[name], sr, ct, freq, ws, H)
                    cases.append({
                        "name": f"note_band_energy/{name}/ws{ws}/ct{ct}",
                        "fn": "note_band_energy",
                        "audio": name,
                        "sampleRate": sr,
                        "scalars": [ct, freq, ws, H],
                        "expected": expected,
                        "rtol": 1e-3,
                        "atol": 1e-3,
                    })

    # adaptive_n_fft: integer FFT sizing (no audio -> exact integer).
    for sr in (44100, 96000):
        for freq in (40.0, 261.63, 1046.5):
            for chunk_len in (256, 3528, 4096, 9000):
                for min_bins in (1, 2):
                    expected = kalimba_dsp.adaptive_n_fft(sr, freq, chunk_len, min_bins, H)
                    cases.append({
                        "name": f"adaptive_n_fft/{sr}/{int(freq)}/{chunk_len}/{min_bins}",
                        "fn": "adaptive_n_fft",
                        "audio": None,
                        "sampleRate": sr,
                        "scalars": [freq, chunk_len, min_bins, H],
                        "expected": expected,
                        "exact": True,
                    })

    return {"audioFiles": audio_files, "cases": cases}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: wasm_reference.py <out_dir>", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_cases(out_dir)
    (out_dir / "reference.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases, {len(payload['audioFiles'])} audio files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
