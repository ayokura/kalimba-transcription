# Browser / Realtime 実装サーベイ補遺 (2026-06-26)

## 目的

将来の browser-side-only / near-real-time transcription に向け、AMT 論文とは別に、
実装技術・既存 repo・WebAudio/WASM/NN inference の観点を整理する。

この補遺は「今すぐ実装する設計書」ではなく、今後の spike で迷わないための調査メモである。

---

## 1. 見るべき実装 / 一次ソース

### 1.1 Spotify Basic Pitch

見る対象:

- `spotify/basic-pitch`
- `spotify/basic-pitch-ts`
- Basic Pitch paper: arXiv `2203.09893`

見るべき箇所:

- onset / note / contour output から note event を作る後段処理
- thresholding / note creation / contour integration
- TypeScript/browser 実装での model loading / audio preprocessing
- pitch bend をどう event へ畳むか

このプロジェクトへの使い道:

- free-performance baseline
- candidate generator
- browser-side note event decoding の参考
- Review UI の "model suggested candidate" source

注意:

- Basic Pitch は instrument-agnostic であり、kalimba 固有の inharmonic partial / sympathetic resonance を
  理解しているわけではない。
- 置換ではなく teacher / baseline として扱う。

### 1.2 PESTO / streamable VQT

見る対象:

- PESTO ISMIR 2023 / arXiv `2309.02265`
- streamable VQT 版 arXiv `2508.01488`

見るべき箇所:

- VQT frame の生成方法
- cached convolution による streaming
- latency の定義
- low-frequency window が latency を支配する点
- ONNX / lightweight model の deployability

このプロジェクトへの使い道:

- dominant pitch evidence
- log-frequency front-end 比較
- causal feature extraction の参考
- browser/worker/WASM での ring-buffer 設計参考

注意:

- 基本は monophonic / dominant-pitch 寄り。polyphonic kalimba の本体置換には向かない。

### 1.3 ONNX Runtime Web / WebGPU / WebNN

見る対象:

- ONNX Runtime Web
- WebGPU backend
- WebNN backend
- TensorFlow.js audio model examples

このプロジェクトへの使い道:

- Basic Pitch / PESTO 等を teacher / candidate generator として browser に載せる場合の選択肢。

注意:

- AudioWorklet 内で NN inference を直接回すより、AudioWorklet は audio capture / ring buffer に限定し、
  worker 側へ渡す構成の方が安全。
- WebGPU / WebNN は環境差が大きいため、MVP は WASM + worker を基本線にする。

### 1.4 WebAudio AudioWorklet / WASM DSP

見る対象:

- WebAudio AudioWorklet
- WASM FFT / SIMD
- Rust/WASM memory copy cost
- Worker / SharedArrayBuffer / ring buffer patterns

このプロジェクトへの使い道:

- 現行 `kalimba-dsp` crate の browser-side port の実行基盤。
- onset / chunk_spectrum / rank_tuning_candidates の incremental 化。

---

## 2. Note event decoding / candidate 処理への示唆

Basic Pitch 系の重要な示唆:

- NN 出力はそのまま楽譜ではなく、**frame/onset/contour → note event** の decoding が別問題。
- このプロジェクトも同様に、`rankedCandidates` / onset evidence / segment candidates から
  `ScoreEvent` を作る後段が本質。
- したがって、browser 化でも「DSP を移植する」だけでなく、**event decoding と candidate preservation** を
  同型に保つ必要がある。

実装方針:

- Python batch と browser realtime で、最終 `CandidateSlot` / `ScoreEvent` schema を共有する。
- debug 限定情報を、本番 response / review session が使える structured candidate に寄せる。
- latency のために 1-best を早く出しても、後段で候補を更新できるようにする。

---

## 3. Realtime / streaming のボトルネック

### 3.1 非因果 / lookahead 処理

現状で注意すべきもの:

- `peak_pick + backtrack`
- active range 終端判断
- full-audio noise floor
- repeated pattern normalization
- final merge / quantization の一部

方針:

- まず fixed-latency batch simulation を作る。
- 例: 50ms / 100ms / 200ms の lookahead で Python pipeline を制限し、F1 / Candidate Recall / Correction Burden を測る。
- その後 Rust/WASM に移す。

### 3.2 AudioWorklet / Worker 分担

推奨分担:

- AudioWorklet:
  - input capture
  - ring buffer write
  - minimal metering
- Worker / WASM:
  - FFT / VQT / onset feature
  - candidate ranking
  - incremental state update
- Main thread:
  - UI
  - review / correction state
  - notation rendering

### 3.3 Memory copy

注意:

- WebAudio → JS → WASM のコピーコストは、小さい chunk で過剰に呼ぶと効く。
- 10ms frame を逐次処理する場合でも、feature extraction は複数 frame batch でまとめる方がよい可能性がある。

---

## 4. このプロジェクトへの取り込み順

### Phase A: browser parity を拡張

- 既存 `/wasm-demo` の `chunk_spectrum` / `rank_tuning_candidates` を基準にする。
- Python と WASM で candidate rank parity を増やす。
- debug JSON に browser output を保存できるようにする。

### Phase B: fixed-latency simulation

- Python 上で lookahead を制限した streaming simulation を作る。
- 50ms / 100ms / 200ms で比較。
- 指標:
  - one-best F1
  - Candidate Recall
  - Correction Burden
  - latency

### Phase C: AudioWorklet + Worker prototype

- AudioWorklet で録音し、Worker/WASM へ chunk を渡す。
- まず onset / candidate preview まで。
- 完全な notation は後回し。

### Phase D: NN teacher integration

- Basic Pitch / PESTO を offline / worker teacher として追加。
- 完了 fixture の正解とは混ぜない。
- `source: "dsp" | "basic_pitch" | "pesto" | "user_verified"` のように provenance を保持。

---

## 5. やらない方がよいこと

- AudioWorklet 内で重い NN inference を直接回す。
- WebGPU / WebNN を MVP の必須条件にする。
- Python batch と browser realtime で別 schema を作る。
- まず CQT/VQT 全面置換から始める。
- 低レイテンシ化だけを追い、Candidate Recall / Correction Burden を測らない。

## 履歴

- 2026-06-26: 新規作成。Basic Pitch / PESTO / ONNX Runtime Web / WebAudio/WASM 実装観点を整理。
