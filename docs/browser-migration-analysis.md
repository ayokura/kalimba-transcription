# Browser Migration Analysis

Analysis of migrating the server-side transcription pipeline to WebAudio API / browser-side implementation, covering both batch and real-time scenarios.

## Current Pipeline Overview

```
[Browser]                          [Server (Python)]
MediaRecorder → WebM → WAV encode → normalize → detect_segments → segment_peaks → post-processing → score
                                     │              │                  │                │
                                     │              │                  │                └─ 18+ pure Python functions
                                     │              │                  └─ numpy FFT + rank_tuning_candidates
                                     │              └─ librosa RMS / onset_strength / HPSS + pure-numpy onset_detect / tempo
                                     └─ numpy array ops
```

## Part 1: Batch Migration (Recording-then-analyze)

### Browser-portable (no obstacles)

| Block | Current implementation | Browser equivalent |
|-------|----------------------|-------------------|
| Normalization | `np.mean` / `np.max` for DC removal + peak normalization | `Float32Array` arithmetic |
| FFT | `np.fft.rfft` / `np.fft.rfftfreq` | JS FFT library (fft.js) or WebAssembly (KissFFT wasm build) |
| Post-processing (18+ functions) | Pure Python list operations | Direct TypeScript port. No external dependencies |
| Score generation | `quantize_beat`, `build_notation_views` | Pure logic. `notation.ts` already has similar `rebuildNotationViews` |

### Obstacles

#### A. RMS energy — `librosa.feature.rms`

librosa computes RMS via STFT-based framing (`frame_length=2048`, `hop_length=256`). WebAudio's `AnalyserNode` is designed for real-time streams, not batch processing of a complete buffer.

**Mitigation**: Get raw `Float32Array` from `AudioBuffer.getChannelData()` and implement frame-by-frame RMS manually. Computation is trivial in JS.

#### B. Onset strength — `librosa.onset.onset_strength`

Requires: STFT → mel filterbank → power-to-dB → frame-to-frame spectral flux. WebAudio's `AnalyserNode` does not provide mel-scale conversion.

**Mitigation**: Meyda.js supports mel spectrogram and spectral flux. However, exact numerical compatibility with librosa is not guaranteed — segment boundaries may shift slightly.

#### C. Onset detection — ~~`librosa.onset.onset_detect`~~ (2026-04-16 更新: pure-numpy 化済み)

~~Peak detection itself is straightforward (local maxima + adaptive threshold), but librosa's `backtrack` option walks backward to energy minima — a custom algorithm that must be ported precisely.~~

**(2026-04-16 更新)** PR #187 で `librosa.onset.onset_detect` を pure-numpy の `_onset_detect_numpy` + `_peak_pick_numpy` に置換済み。`_peak_pick_numpy` は librosa の `__peak_pick` gufunc kernel の 1:1 port (ISC License)。backtrack は `librosa.onset.onset_backtrack` を引き続き使用するが、こちらは numba 非依存の単純な配列操作。**Browser 移植時は JS への直接移植が可能。**

#### D. Tempo estimation — ~~`librosa.beat.beat_track` (largest obstacle)~~ (2026-04-16 更新: pure-numpy 化済み)

~~Internally computes: onset_envelope → tempogram (autocorrelation via 635 FFTs) → tempo estimation → beat tracking. This is 90% of current processing time. In-browser on the main thread, this would freeze the UI. Web Workers can help, but JS FFT is 3-10x slower than native scipy.~~

**(2026-04-16 更新)** PR #187 で `librosa.beat.beat_track` を pure-numpy autocorrelation (`_estimate_tempo_autocorr`) に置換済み。tempogram FFT を排除し、onset envelope の autocorrelation で直接テンポ推定する。**Browser 移植のブロッカーではなくなった** — JS の配列演算で直接実装可能。既知の制限として sub-harmonic ambiguity があり (bwv147: 128 BPM → 41 BPM)、streaming 再設計時に対応予定。

#### E. FFT performance for segment analysis

numpy's C-backed FFT handles `n_fft=4096+` per segment. JS FFT libraries are 3-5x slower. Current profile shows ~22ms/segment; at 3x slowdown that's ~66ms/segment — acceptable.

**Mitigation**: WebAssembly FFT for near-native performance if needed.

#### F. Vectorized batch operations — `batch_peak_energies`

Uses numpy broadcasting (`log_positive[np.newaxis, :] - log_centers[:, np.newaxis]`) across all notes × all frequency bins. JS has no equivalent.

**Mitigation**: With only 17-21 tuning notes, a naive double loop in JS is fast enough. numpy batching was specifically to avoid Python loop overhead, which JS does not have.

### Batch migration summary

(2026-04-16 更新: onset_detect, beat_track の pure-numpy 化を反映)

| Category | Target | Difficulty |
|----------|--------|-----------|
| Immediately portable | Normalization, post-processing (18 functions), score generation, beat quantization | Low |
| Already pure-numpy (JS port straightforward) | onset_detect (`_peak_pick_numpy`), tempo estimation (`_estimate_tempo_autocorr`) | Low-Medium |
| JS implementation needed | RMS, FFT, peak energy, harmonic suppression | Medium (library selection or self-implementation) |
| Algorithm porting needed | onset_strength (mel spectrogram) | Medium-High (compatibility verification is critical) |

---

## Part 2: Real-time Migration

Adding real-time analysis changes the obstacle landscape significantly.

### What gets easier

| Component | Batch obstacle | Real-time situation |
|-----------|---------------|-------------------|
| RMS / segment detection | Manual frame-by-frame implementation needed | `AnalyserNode` provides this natively. "Currently active" is directly observable |
| Onset strength | Mel spectrogram porting needed | `AnalyserNode.getFloatFrequencyData()` provides per-frame FFT; spectral flux = diff with previous frame |
| FFT performance | JS FFT 3-5x slower | `AnalyserNode` runs native FFT internally |
| Tempo estimation | 90% of processing time, largest obstacle | **Problem disappears or changes nature**. Not needed during playback; can use lightweight IOI-based estimation after recording |

### What gets harder — post-processing real-time compatibility

Each post-processing function classified by temporal dependency:

#### Causal (past-only reference) — real-time compatible

- `suppress_low_confidence_dyad_transients` — examines each event's own properties only
- `suppress_subset_decay_events` — references cleaned[-1]
- `merge_adjacent_events` — references merged[-1]

#### Forward-looking (next 1 event) — compatible with 1-event latency

These access `raw_events[index + 1]` and can run once the next event arrives (~0.1-0.3s delay):

- `simplify_short_secondary_bleed`
- `merge_short_gliss_clusters`
- `simplify_short_gliss_prefix_to_contiguous_singleton`
- `merge_four_note_gliss_clusters`
- `suppress_leading_gliss_subset_transients`
- `suppress_leading_gliss_neighbor_noise`
- `suppress_leading_single_transient`
- `split_ambiguous_upper_octave_pairs`
- `suppress_bridging_octave_pairs`
- `suppress_short_residual_tails`
- `merge_short_chord_clusters`

#### Full-batch (all events required) — **incompatible with real-time**

These scan all events to identify the dominant pattern (most frequent 3/4-note set), then correct outliers against it. The dominant pattern is unknowable until the performance ends.

- `normalize_repeated_four_note_family`
- `normalize_repeated_four_note_gliss_patterns`
- `normalize_repeated_explicit_four_note_patterns`
- `normalize_repeated_triad_patterns`

### Obstacle priority under real-time

(2026-04-16 更新: onset_detect は pure-numpy 化済みのため obstacle 3 を下方修正、obstacle 4 は解消)

| Priority | Obstacle | Impact |
|----------|---------|--------|
| **1** | Full-batch post-processing (6 functions) | Pattern normalization is impossible in real-time. Requires two-phase architecture: provisional display → post-recording correction |
| **2** | Forward-looking function latency design | Current event display is blocked until the next event is confirmed. 1-event latency (~0.1-0.3s) |
| **3** | onset_detect backtrack reproduction | Segment start accuracy. `_peak_pick_numpy` is already pure-numpy; backtrack is simple array logic. Solvable with a few frames of buffering |
| ~~**4**~~ | ~~Tempo estimation redesign~~ | **(2026-04-16 解消)** `_estimate_tempo_autocorr` (pure-numpy) が `beat_track` を置換済み。Real-time では IOI-based lightweight estimation で十分 |

### Recommended architecture

```
[Real-time layer — Browser, during performance]
  AnalyserNode → onset detection → FFT → rank_tuning_candidates → causal post-processing
  → Display provisional score immediately (with 1-event latency)

[Finalization layer — Browser or Server, after recording]
  Full-batch post-processing (pattern normalization, 6 functions)
  Tempo estimation (IOI-based lightweight version)
  → Update to finalized score
```

This enables notes to appear on screen in real-time during performance, with pattern correction running in a few hundred milliseconds after recording ends.

### Comparison: batch vs real-time migration

(2026-04-16 更新: tempo estimation が pure-numpy 化により batch 側のブロッカーではなくなった)

| Aspect | Batch migration | Real-time migration |
|--------|----------------|-------------------|
| Largest obstacle | onset_strength mel spectrogram porting | Full-batch post-processing (architecturally incompatible) |
| onset_detect | Pure-numpy (`_peak_pick_numpy`), JS port straightforward | `AnalyserNode` + incremental peak picking |
| onset_strength | Mel spectrogram porting needed | `AnalyserNode` provides natively |
| FFT performance | JS speed is a concern | `AnalyserNode` provides native implementation |
| Tempo estimation | Pure-numpy autocorrelation, JS port straightforward | Problem disappears or becomes lightweight |
| Post-processing | Directly portable | Two-phase architecture required |

---

## Related

- #8 — Pipeline optimization (implemented)
- #10 — Tempo estimation optimization (proposed, prerequisite for batch migration)
- `architecture.md` — Current system architecture
