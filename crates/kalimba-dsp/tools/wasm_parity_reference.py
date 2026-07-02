"""Generate real-fixture parity reference data for check_wasm_parity.cjs.

A0 (browser offline parity harness, sprint-plan 2026-07 S5). The synthetic
harness (wasm_reference.py) feeds a *native-generated* envelope into
onset_detect, so a drift in the wasm FFT stage cannot surface there
(2026-07 audit). This script instead dumps committed fixture WAV audio plus
the Python-side reference values so the node checker can run the wasm
through-path audio -> onset_strength -> onset_detect end-to-end and compare:

- native reference: the production path (Rust pyo3, same shared core as wasm)
  -> expected to match the wasm outputs frame-exactly; any mismatch means the
  two compilations of the shared core diverged (rustfft SIMD paths etc.)
- numpy reference: the independent pure-numpy oracle (_*_numpy in segments.py)
  -> guards the shared core itself against silent drift on real audio

Additionally dumps segment-level intermediates (detect_segments debug subset:
active ranges / segment boundaries / discard results). These have no wasm
counterpart yet (B1 port pending); the harness records them as the pinned
reference that the future TS/Rust segment port must reproduce, and the node
checker verifies the plumbing by self-comparison until B1 lands.

Usage: PYTHONPATH=apps/api uv run python \
    crates/kalimba-dsp/tools/wasm_parity_reference.py <out_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import kalimba_dsp as K
from app.transcription import segments
from app.transcription.constants import FRAME_LENGTH, HOP_LENGTH

N_MELS = 128

REPO_ROOT = Path(__file__).resolve().parents[3]

# Committed fixture WAVs covering the three sample rates in play:
# 44.1k (fixture native), 48k (browser-live rate, free-performance corpus),
# 96k (bulk of manual captures). Short 96k fixtures keep CI time bounded.
CASES = [
    (
        "bwv147-34l-44k1",
        "apps/api/tests/fixtures/manual-captures/kalimba-34l-c-bwv147-sequence-163-01/audio.wav",
    ),
    (
        "free-perf-17ea7626-48k",
        "apps/api/tests/fixtures/free-performance-corpus/17ea7626-3c5d-450d-ae74-0116dea6e881/audio.wav",
    ),
    (
        "mixed-sequence-96k",
        "apps/api/tests/fixtures/manual-captures/kalimba-17-c-mixed-sequence-01/audio.wav",
    ),
    (
        "c4-to-e6-sequence-96k",
        "apps/api/tests/fixtures/manual-captures/kalimba-17-c-c4-to-e6-sequence-17-single-01/audio.wav",
    ),
]

# detect_segments debug keys pinned for the (future) B1 segment port.
SEGMENT_DEBUG_KEYS = [
    "onsetTimes",
    "gapValidatedOnsetTimes",
    "rawActiveRanges",
    "activeRanges",
    "shortBridgeActiveRanges",
    "rmsThreshold",
    "activeRangeSegments",
    "segments",
]


def load_mono_f32(path: Path) -> tuple[np.ndarray, int]:
    """Mirror app.transcription.audio.read_audio decode semantics."""
    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio, dtype=np.float32), int(sample_rate)


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_f32(name: str, arr: np.ndarray) -> str:
        fname = f"{name}.f32"
        np.asarray(arr, dtype=np.float32).tofile(out_dir / fname)
        return fname

    def save_u32(name: str, arr: np.ndarray) -> str:
        fname = f"{name}.u32"
        np.asarray(arr, dtype=np.uint32).tofile(out_dir / fname)
        return fname

    manifest: dict = {
        "constants": {"hopLength": HOP_LENGTH, "nFft": FRAME_LENGTH, "nMels": N_MELS},
        "cases": [],
    }

    for case_id, rel_wav in CASES:
        wav_path = REPO_ROOT / rel_wav
        audio, sr = load_mono_f32(wav_path)

        # Native (Rust pyo3) through-path — same shared core the wasm build uses.
        env_native = np.asarray(
            K.onset_strength(audio, sr, HOP_LENGTH, FRAME_LENGTH, N_MELS), dtype=np.float32
        )
        frames_native = np.asarray(
            K.onset_detect(env_native, sr, HOP_LENGTH, True), dtype=np.uint32
        )

        # Independent numpy oracle through-path.
        env_numpy = segments._onset_strength_numpy(audio, sr, HOP_LENGTH, FRAME_LENGTH, N_MELS)
        frames_numpy = np.asarray(
            segments._onset_detect_numpy(env_numpy, sr, HOP_LENGTH, backtrack=True),
            dtype=np.uint32,
        )

        # Segment-level reference (Python-only until the B1 port exists).
        seg_result = segments.detect_segments(audio, sr)
        seg_debug = {key: seg_result.debug.get(key) for key in SEGMENT_DEBUG_KEYS}

        manifest["cases"].append(
            {
                "id": case_id,
                "wav": rel_wav,
                "sampleRate": sr,
                "samples": int(audio.shape[0]),
                "audio": save_f32(f"audio_{case_id}", audio),
                "native": {
                    "env": save_f32(f"env_native_{case_id}", env_native),
                    "frames": save_u32(f"frames_native_{case_id}", frames_native),
                },
                "numpy": {
                    "env": save_f32(f"env_numpy_{case_id}", env_numpy),
                    "frames": save_u32(f"frames_numpy_{case_id}", frames_numpy),
                },
                "segment": seg_debug,
            }
        )

    (out_dir / "parity_reference.json").write_text(json.dumps(manifest))
    print(f"wrote {len(manifest['cases'])} parity cases to {out_dir}")


if __name__ == "__main__":
    main()
