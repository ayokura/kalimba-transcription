# kalimba-dsp

Rust DSP primitives for kalimba transcription. This crate is **dual-binding**:

- **`python`** (default feature) — a [pyo3](https://pyo3.rs) extension module
  named `kalimba_dsp`, built by [maturin](https://www.maturin.rs) and consumed
  by the API server (`apps/api`).
- **`wasm`** — [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) wrappers
  for future browser-side (WebAudio + WebAssembly) transcription, with no server
  round-trip.

## Architecture: shared core + thin bindings

The numeric core is **binding-agnostic pure Rust** and operates on plain
`&[f32]` slices:

| Core function | Role |
|---|---|
| `cached_hanning` | cached Hanning window |
| `adaptive_n_fft` | FFT size selection for a target band |
| `note_band_energy_inner` / `note_band_energy` | peak FFT magnitude in a note's band |
| `peak_energy_near` / `batch_peak_energies` | peak magnitude near center freq(s) over a precomputed f64 spectrum |
| `scan_gap_for_mute_dip_with_window_inner` | mute-dip-then-recovery scan |
| `detect_gap_rise_attack_inner` | two-point energy-rise check inside a gap |

The `onset` module (`src/onset.rs`) holds the browser-pipeline front end — ports
of the librosa-derived numpy reference in `apps/api/app/transcription/segments.py`:

| Onset function | Role |
|---|---|
| `mel_filterbank` | Slaney mel filterbank (bit-exact to numpy) |
| `rms` | frame-wise RMS energy |
| `onset_strength` | mel spectral-flux onset envelope (periodic-Hann STFT) |
| `peak_pick` | greedy peak picker (librosa `util.peak_pick`) |
| `onset_backtrack` | snap onsets to preceding energy minima |
| `onset_detect` | full chain: normalise -> peak-pick -> backtrack |

The server recognizer's primary onset path delegates to these (frame-exact to
the numpy reference across the full fixture suite, 10-17x faster); the same crate
runs onset detection in the browser. Pinned to the numpy reference by
`apps/api/tests/test_onset_dsp_rust.py`, the fixture regression suite, and the
wasm equivalence harness.

Each binding is a thin wrapper that only adapts the input array to a `&[f32]`
and delegates to the `*_inner` core:

- **pyo3** (`mod python_binding`, `#[cfg(feature = "python")]`): converts a
  `numpy::PyReadonlyArray1<f32>` to a slice.
- **wasm-bindgen** (`mod wasm_binding`, `#[cfg(feature = "wasm")]`): takes a
  `&[f32]`, which maps to a JS `Float32Array`.

This keeps the algorithm portable and avoids coupling the DSP logic to either
Python (numpy/pyo3) or the browser (wasm-bindgen).

## Features

```toml
default = ["python"]
python  = ["dep:pyo3", "dep:numpy", "pyo3/extension-module"]
wasm    = ["dep:wasm-bindgen"]
```

`pyo3` and `numpy` are optional and pulled in only by `python`; `wasm-bindgen`
only by `wasm`. The two features are mutually exclusive in practice (each
defines its own top-level functions), so always disable defaults when building
for wasm.

## Building

### Python extension (default)

Normal workflow — built automatically by `uv` / maturin from the repo root:

```sh
uv sync                              # builds & installs the kalimba_dsp ext
uv sync --reinstall-package kalimba-dsp   # force-rebuild after Rust changes
```

Direct maturin (uses `[tool.maturin] features = ["python"]` in `pyproject.toml`):

```sh
maturin develop     # build + install into the active venv
maturin build       # produce a wheel
```

Plain cargo (default feature is `python`):

```sh
cargo build
cargo build --features python   # explicit, equivalent
```

### WASM (browser) — requires one-time setup by the user

The wasm target is **not** installed in this environment. Install it once:

```sh
rustup target add wasm32-unknown-unknown
```

Then build the `.wasm` artifact:

```sh
cargo build --no-default-features --features wasm --target wasm32-unknown-unknown
```

For a full browser-ready package (JS glue + `.wasm` + TypeScript types),
install [`wasm-pack`](https://rustwasm.github.io/wasm-pack/) and run. Cargo flags
go **after `--`**; wasm-pack's own `--target` selects the JS output kind
(`web` / `bundler` / `nodejs`), not a rustc target — passing them before `--`
makes wasm-pack try to `rustup target add web` and fail:

```sh
cargo install wasm-pack
wasm-pack build --target web -- --no-default-features --features wasm
```

Output lands in `pkg/` (git-ignored): `kalimba_dsp_bg.wasm` (~216 KB,
wasm-opt'd), `kalimba_dsp.js` glue, and `.d.ts` types exporting every shared-core
function with `Float32Array` audio I/O.

### Validating the wasm build against the native extension

Both bindings share one pure-Rust core, so the wasm output must be numerically
identical to the pyo3 extension. `check_wasm.sh` builds the nodejs-target package
and replays a battery of inputs through both, asserting equality:

```sh
crates/kalimba-dsp/check_wasm.sh
```

It exits non-zero on any wasm-vs-native mismatch (binding-glue regression:
`Float32Array` marshalling, `i64`->BigInt / `u32` ABI). Requires `wasm-pack`,
`node`, and the uv-managed Python env. Extend `tools/wasm_reference.py` +
`tools/check_wasm.cjs` when a new shared-core primitive is exposed to wasm.

The same script also runs the **fixture parity harness** (A0, sprint-plan
2026-07 S5): `tools/wasm_parity_reference.py` dumps committed fixture WAVs
(44.1k / 48k / 96k) plus Python references — the native pyo3 through-path,
the independent pure-numpy oracle (`segments._*_numpy`), and the
`detect_segments` debug subset (active ranges / segment boundaries /
discards) — and `tools/check_wasm_parity.cjs` replays the wasm build
end-to-end (`audio -> onset_strength -> onset_detect`, no native-generated
intermediates injected) against them. Envelope tolerances: 1e-4/1e-5 vs
native, 1e-3/1e-3 vs numpy; onset frames must be frame-exact vs both.
The segment-level reference has no wasm counterpart yet — it is the pinned
contract for the B1 segment port, and the checker currently validates the
manifest plumbing only.

### Host type-check of the wasm wrappers (no wasm target needed)

`wasm-bindgen` compiles on the host, so you can type-check the wasm binding
without installing the wasm32 target:

```sh
cargo check --no-default-features --features wasm
```

This is the CI-friendly way to confirm the wasm wrappers stay syntactically and
type-correct without producing a wasm artifact.
