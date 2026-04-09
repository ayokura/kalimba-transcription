# Browser Migration Analysis

Analysis of migrating the server-side transcription pipeline to WebAudio API / browser-side implementation, covering both batch and real-time scenarios.

## Current Pipeline Overview

```
[Browser]                          [Server (Python)]
MediaRecorder → WebM → WAV encode → normalize → detect_segments → segment_peaks → post-processing → score
                                     │              │                  │                │
                                     │              │                  │                └─ 18+ pure Python functions
                                     │              │                  └─ numpy FFT + rank_tuning_candidates
                                     │              └─ librosa RMS / onset / beat_track
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

#### C. Onset detection — `librosa.onset.onset_detect`

Peak detection itself is straightforward (local maxima + adaptive threshold), but librosa's `backtrack` option walks backward to energy minima — a custom algorithm that must be ported precisely.

**Mitigation**: Port the algorithm from librosa source. It is pure array logic, but compatibility testing against current fixtures is essential.

#### D. Tempo estimation — `librosa.beat.beat_track` (largest obstacle)

Internally computes: onset_envelope → tempogram (autocorrelation via 635 FFTs) → tempo estimation → beat tracking. This is 90% of current processing time. In-browser on the main thread, this would freeze the UI. Web Workers can help, but JS FFT is 3-10x slower than native scipy.

**Mitigation**: Resolve #10 (tempo estimation optimization) first. Reducing FFT count by 10x via coarser hop_length + audio truncation makes Web Worker execution realistic.

#### E. FFT performance for segment analysis

numpy's C-backed FFT handles `n_fft=4096+` per segment. JS FFT libraries are 3-5x slower. Current profile shows ~22ms/segment; at 3x slowdown that's ~66ms/segment — acceptable.

**Mitigation**: WebAssembly FFT for near-native performance if needed.

#### F. Vectorized batch operations — `batch_peak_energies`

Uses numpy broadcasting (`log_positive[np.newaxis, :] - log_centers[:, np.newaxis]`) across all notes × all frequency bins. JS has no equivalent.

**Mitigation**: With only 17-21 tuning notes, a naive double loop in JS is fast enough. numpy batching was specifically to avoid Python loop overhead, which JS does not have.

### Batch migration summary

| Category | Target | Difficulty |
|----------|--------|-----------|
| Immediately portable | Normalization, post-processing (18 functions), score generation, beat quantization | Low |
| JS implementation needed | RMS, FFT, peak energy, harmonic suppression | Medium (library selection or self-implementation) |
| Algorithm porting needed | onset_strength (mel spectrogram), onset_detect (backtrack) | Medium-High (compatibility verification is critical) |
| Largest obstacle | beat_track (tempo estimation) | High (#10 optimization is prerequisite) |

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

| Priority | Obstacle | Impact |
|----------|---------|--------|
| **1** | Full-batch post-processing (6 functions) | Pattern normalization is impossible in real-time. Requires two-phase architecture: provisional display → post-recording correction |
| **2** | Forward-looking function latency design | Current event display is blocked until the next event is confirmed. 1-event latency (~0.1-0.3s) |
| **3** | onset_detect backtrack reproduction | Segment start accuracy. Solvable with a few frames of buffering, but adds latency |
| **4** | Tempo estimation redesign | Not needed during real-time. IOI-based lightweight estimation after recording. beat_track becomes unnecessary |

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

| Aspect | Batch migration | Real-time migration |
|--------|----------------|-------------------|
| Largest obstacle | Tempo estimation compute cost | Full-batch post-processing (architecturally incompatible) |
| onset_strength | Mel spectrogram porting needed | `AnalyserNode` provides natively |
| FFT performance | JS speed is a concern | `AnalyserNode` provides native implementation |
| Tempo estimation | Biggest bottleneck | Problem disappears or becomes lightweight |
| Post-processing | Directly portable | Two-phase architecture required |

---

## Related

- #8 — Pipeline optimization (implemented)
- #10 — Tempo estimation optimization (proposed, prerequisite for batch migration)
- `architecture.md` — Current system architecture
