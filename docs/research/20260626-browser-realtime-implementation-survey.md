# Browser / Realtime 実装サーベイ補遺 (2026-06-26)

## 目的

将来の browser-side-only / near-real-time transcription に向け、AMT 論文とは別に、
実装技術・既存 repo・WebAudio/WASM/NN inference の観点を整理する。

この補遺は「今すぐ実装する設計書」ではなく、今後の spike で迷わないための調査メモである。

---

## 0. 検証済みの要点 (2026-06-26, 一次ソース確認)

browser/realtime 担当サーベイで一次ソース確認できた要点と、私のドラフトに対する修正。

- **Basic Pitch は 22,050 Hz に resample・mono 化して処理する。** ブラウザ移植で kalimba audio を
  そのまま 48 kHz で渡す前提にしない。前処理 (mono downmix + resample) を Python/TS/WASM で揃える。
- **Basic Pitch TS は AudioBuffer / ファイル処理寄り**であり、AudioWorklet live streaming の参照実装では
  ない。**「browser offline parity baseline + note decoder 参照」**として扱う (live spine は自前)。
- Basic Pitch の note decoding は `output_to_notes_polyphonic()`: onset local maxima を threshold →
  対応 pitch bin の frame energy が閾値を切るまで note を伸長 → 採用 energy と隣接 pitch bin を
  `remaining_energy` から消し込み。さらに `melodia_trick` が onset 無しの残余 energy を note 化する
  rescue path。**この隣接 semitone 消し込みは kalimba の partial collision には荒い** → tine/partial
  table ベースに置換する。`melodia_trick` 相当は final note ではなく alternate candidate へ降格。
- **PESTO は single-pitch / dominant-pitch estimator** で、出力は `time, frequency, confidence`。
  pitch 復元は既定で Argmax-Local Weighted Averaging。`streaming=True` の circular buffer mode と
  ONNX export (stateless model + 呼び出し側 cache 更新) がある。chord/共鳴分離を単独で解くものではなく、
  **onset 周辺の dominant pitch confidence / teacher / ORT Web POC** に使う。
- **ONNX Runtime Web の WASM multi-threading は WebAssembly threads + `crossOriginIsolated` が前提。**
  → dev/prod サーバの **COOP/COEP 方針を早期に決める**必要がある。`env.wasm.proxy=true` は WebGPU EP と
  併用不可。自前 Worker を持つなら、その Worker 内で ORT を import する方が制御しやすい。
- **ORT Web 配布は JS bundle と WASM binary の version 一致が必須**(不一致は初期化失敗)。大きい model は
  IndexedDB cache を検討。軽量モデルは WASM EP 優先、重い時だけ WebGPU。
- **AudioWorklet `process()` の audio quantum は現状 128 frames だが将来可変になり得る**ため、固定 128 を
  前提にせず毎回 buffer length を見る。44.1 kHz では安定 stream の処理予算が約 3ms と非常に小さい
  → AudioWorklet には重い処理 (FFT/VQT/ONNX/allocation/JSON) を置かない。
- full CQT は低域 kernel が巨大化する (PESTO 例: A0/48kHz で約 131,072 samples ≈ 2.7s)。kalimba は
  A0 ほど低くないが、realtime では **full CQT より VQT / bounded-window FFT / 17-tine resonator bank** を優先。
- live **Web MIDI output は secure context + permission 依存**で対応も限定的 → core 要件にせず optional UX。
  MIDI file export は note event 安定後に `@tonejs/midi` 等で生成。
- Magenta.js Onsets and Frames は solo piano 向け・推論が音声長の半分程度で、**kalimba 本体移植候補では
  ない**(browser 推論/chunk 処理/UI 連携の参考に留める)。

### このサーベイで残った gap / 懸念 (要追検証)

- **COOP/COEP を有効化すると既存 web app の埋め込み/3rd-party リソースが壊れ得る。** Next.js app
  (`apps/web`) で `crossOriginIsolated` を入れる影響範囲は未調査。multi-thread を諦めて single-thread WASM で
  始める退避策も併記して判断する。
- **SharedArrayBuffer も COOP/COEP 必須。** ring buffer 設計が前提条件に縛られる点は spine 着手前に確定する。
- **既存 `kalimba-dsp` (Rust/WASM) と ORT Web (別 WASM) の二重ロード**コスト・初期化順序は未評価。
- Basic Pitch / PESTO のライセンスと model asset 配布条件は、teacher として組み込む前に要確認。
- 数値 (22,050 Hz, 128 frames, ~3ms 予算, kernel sample 数) は出典時点の値。実装着手時に再確認する。

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

> **検証サーベイの推奨順 (より細粒度):**
> browser offline parity harness → candidate schema/metrics 固定 → AudioWorklet+Worker+SharedArrayBuffer
> spine (精度を上げず latency/jitter/underflow だけ測る) → 既存 broadband onset/FFT scorer を Worker/WASM へ移植
> → Basic Pitch decoder を candidate 処理の参照として比較 → PESTO+ORT Web POC (Worker 内) → VQT/17-tine
> resonator を research line で dual-run → MIDI/MusicXML/Web MIDI export は最後。
> 下記 Phase A–D はこれを粗くまとめたもの。

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
- 2026-06-26: §0 追加。browser/realtime 担当サーベイの一次ソース検証 (Basic Pitch 22.05kHz/AudioBuffer 寄り、
  PESTO single-pitch + streaming/ONNX、ORT Web の COOP/COEP・version 一致・proxy 注意、AudioWorklet 128frame/
  ~3ms 予算、CQT 低域 kernel) と gap/懸念、検証推奨順を反映。
