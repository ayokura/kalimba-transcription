# Browser / Realtime 実装サーベイ補遺 (2026-06-26)

## 目的

将来の browser-side-only / near-real-time transcription に向け、AMT 論文とは別に、
実装リポジトリ・WebAudio/WASM・ONNX Runtime Web・browser audio-to-MIDI の実装知見を整理する。

この補遺は、browser/realtime 担当サーベイの一次ソース確認結果を主ソースとして再稿したもの。
先行ドラフトに追記する形ではなく、調査担当 agent の結果を中心に、プロジェクト方針へ再構成している。

---

## 1. 結論

このプロジェクトでの位置づけは次の通り。

- **Basic Pitch**: browser 実行可能な AMT baseline / note decoder 参照。live AudioWorklet 実装の参照ではない。
- **PESTO**: streamable VQT と低レイテンシ pitch-confidence の参照。polyphonic kalimba transcription の置換ではない。
- **ONNX Runtime Web**: 軽量 teacher / calibration model の browser 実行基盤。
- **AudioWorklet + Worker + WASM**: 本 recognizer を realtime 化する spine。

最短ロードマップ:

> **Browser offline parity harness → candidate schema/metrics → AudioWorklet+Worker spine → existing FFT/onset port → Basic Pitch decoder comparison → PESTO/ONNX POC → VQT/resonator research → MIDI/MusicXML/Web MIDI export**

この順なら、既存 recognizer の精度資産を壊さずに、browser-side-only / streaming へ段階的に移行できる。

---

## 2. 見るべき repo / docs

### 2.1 Basic Pitch Python / TypeScript

#### Spotify `basic-pitch` Python repo

見る理由:

- 公式 Python 実装。
- 軽量・instrument-agnostic・polyphonic な audio-to-MIDI を目的とする。
- TensorFlow / CoreML / TensorFlowLite / ONNX 形式のモデルを同梱。
- CLI は MIDI、raw model outputs、note events を保存できる。
- programmatic API は `model_output`, `midi_data`, `note_events` を返す。
- 入力は **mono 化され、22,050 Hz に resample** される。

このプロジェクトへの含意:

- Basic Pitch を baseline / teacher として使う場合、kalimba audio を 48 kHz のまま比較してはいけない。
- Python / browser / WASM の前処理差分が評価差分に混ざらないよう、mono downmix と resample 条件を明示する。

#### `basic_pitch/inference.py`

見る理由:

- 音声を固定長 window に切る。
- overlap を扱う。
- model output を unwrap する。

このプロジェクトへの含意:

- browser streaming では、この「window/chunk + overlap + unwrap」を bounded-latency chunk 処理へ置き換える参考になる。
- ただし、そのまま live AudioWorklet spine にはならない。

#### `basic_pitch/note_creation.py`

見る理由:

- note event decoding の最重要参照。
- `note` frame activation、`onset` activation、`contour` を note events / MIDI / pitch bend へ変換する。

`output_to_notes_polyphonic()` の構造:

1. onset の local maxima を threshold で拾う。
2. 対応 pitch bin の frame energy が閾値を下回るまで note を伸ばす。
3. 採用済み energy と隣接 pitch bin を `remaining_energy` から消し込む。
4. `melodia_trick` で onset がなかった残余 frame energy を note 化する rescue path がある。

kalimba への取り込み方:

- `onset peak` → note-on gate の参照。
- `frame/body energy` → sustain / offset evidence の参照。
- `remaining_energy` → occupancy model の参照。
- `contour / pitch bend` → calibration signal として保存価値あり。

注意:

- Basic Pitch の隣接 semitone 消し込みは、kalimba の実音配置・partial collision には荒い。
- kalimba では tine table / partial table ベースの occupancy に置換する。
- `melodia_trick` 相当は final note ではなく、低 confidence alternate candidate に降格する。

#### `spotify/basic-pitch-ts` / `@spotify/basic-pitch`

見る理由:

- Python sibling との feature parity を意図した公式 TypeScript 実装。
- `AudioBuffer` を `evaluateModel()` に渡す。
- `frames`, `onsets`, `contours` を集める。
- `outputToNotesPoly()` → `addPitchBendsToNoteEvents()` → `noteFramesToTime()` へ流す例がある。

このプロジェクトへの含意:

- **browser offline parity baseline** として有用。
- Basic Pitch TS は「ブラウザで動く」が、主導線は AudioBuffer / ファイル処理寄り。
- **AudioWorklet live streaming の参照実装として扱わない**。

---

### 2.2 PESTO / streamable VQT

#### SonyCSLParis `pesto` repo

見る理由:

- 公式 inference repo。
- 出力は `time, frequency, confidence`。
- pitch bin distribution から Argmax-Local Weighted Averaging 等で pitch を復元する。
- `streaming=True` で内部 circular buffer を使う streaming mode がある。
- ONNX export では stateless model + 呼び出し側 cache 更新の例がある。

このプロジェクトへの含意:

- PESTO は polyphonic note decoder ではなく、**onset region ごとの dominant pitch / confidence feature** として使う。
- Worker 内 ONNX Runtime Web POC の題材に向く。

具体的な使い道:

- broadband onset で見つけた attack 周辺の dominant pitch confidence。
- low-confidence segment の candidate ranking 補助。
- free-performance baseline。
- browser ONNX Runtime Web POC。
- VQT / log-frequency frontend の teacher。

注意:

- PESTO は基本的に **single-pitch / dominant-pitch estimator**。
- kalimba の chord、sympathetic resonance、overlapping decay を単独で解くものではない。

#### PESTO v2 / streamable VQT paper

見る理由:

- VQT frame を入力にする single-pitch estimator。
- streamable VQT を cached convolution で実装する設計が browser/realtime に直接効く。
- 論文は「raw compute が速いだけでは不十分で、causal / stream 処理できる必要がある」と明示。
- CQT 低域 kernel の巨大化を VQT で緩和する議論がある。

重要な数値例:

- full CQT は低域 kernel が長くなりやすい。
- PESTO paper は A0 / 48 kHz の CQT kernel が 131,072 samples、約 2.7 秒になり得る例を示す。

このプロジェクトへの含意:

- kalimba は A0 ほど低くないが、browser realtime では full CQT より、
  **VQT / bounded-window FFT / 17-tine resonator bank** を優先するのが安全。

---

### 2.3 ONNX Runtime Web

見る対象:

- Get started / Web docs
- Web tutorial
- env flags / session options
- deploy docs
- WebGPU EP docs

重要な確認事項:

- `onnxruntime-web` は WASM / WebGPU / WebGL / WebNN など複数 execution provider を持つ。
- WASM は ONNX operator coverage が広い。
- GPU 系 EP は subset 対応になり得る。
- 軽量モデルはまず WASM EP、重いモデルだけ WebGPU を検討する。

実装上の注意:

- `env.wasm.numThreads` は browser 環境に依存する。
- WASM multi-threading は WebAssembly threads と `crossOriginIsolated` が有効な場合のみ。
- つまり dev/prod server で **COOP/COEP 方針を早めに決める**必要がある。
- `env.wasm.proxy=true` は UI 応答性を保つのに便利だが、WebGPU EP と併用できない。
- 自前 Worker を持つなら、その Worker 内で ORT を import する方が制御しやすい。
- production では JS bundle、ORT WASM binaries、model files が必要。
- **WASM binary と JS bundle は同一 build/version に揃える**。不一致は初期化失敗の原因になる。
- 大きい model file は IndexedDB cache を検討する。

このプロジェクトへの含意:

- PESTO-like model や calibration model は、まず Worker 内で `onnxruntime-web/wasm` を試す。
- WebGPU は MVP 必須条件にしない。
- `kalimba-dsp` の WASM と ORT Web の WASM が二重ロードになるため、初期化順序と asset size を測る。

---

### 2.4 WebAudio / AudioWorklet / WASM

#### MDN `AudioWorkletProcessor.process()`

見る理由:

- `process()` は audio rendering thread から同期呼び出しされる。
- 現状の audio block は 128 frames だが、将来可変になり得る。
- 固定 128 前提ではなく、毎 callback で buffer length を見る。

#### Chrome Audio Worklet Design Pattern

見る理由:

- AudioWorklet + WebAssembly + ring buffer + SharedArrayBuffer + Worker の設計メモとして重要。
- 44.1 kHz では安定 audio stream の処理予算が約 3 ms とされる。
- WASM は JS JIT / GC を避ける手段として説明される。

このプロジェクトへの含意:

- AudioWorklet に FFT/VQT/ONNX/allocation/JSON serialize を置かない。
- AudioWorklet は capture + ring buffer write に限定する。
- 重い解析は Worker + WASM に置く。

#### GoogleChromeLabs Web Audio Samples

見る理由:

- AudioWorklet + SharedArrayBuffer + Worker の実装例がある。
- realtime spine の雛形として使える。

#### Essentia.js

見る理由:

- Essentia C++ backend を WebAssembly 化し、JS / TypeScript API で browser / Node の real-time / offline audio analysis をサポート。
- CQT/VQT をそのまま採用するというより、WASM audio analysis API 設計・feature extractor 実装の参考。

---

### 2.5 Browser audio-to-MIDI 周辺

#### Magenta.js Onsets and Frames

見る理由:

- browser で raw audio を MIDI に変換する代表実装。
- ただし solo piano 向け。
- 公式 docs でも solo piano 録音向き・推論時間が音声長の半分程度という制約がある。

このプロジェクトへの扱い:

- kalimba 本体への直接移植候補ではない。
- browser 推論・chunk 処理・UI 連携の参考に留める。

#### Web MIDI API

見る理由:

- live MIDI device I/O の公式 API。
- secure context、Permission Policy、ユーザー許可が必要。

このプロジェクトへの扱い:

- core transcription の必須要件にしない。
- optional live output として後回し。

#### `@tonejs/midi`

見る理由:

- MIDI file read/write 用。
- 内部 note event が安定した後に browser で `.mid` を生成する用途。

---

## 3. Candidate object / provenance への示唆

streaming では、即時に 1-best を確定して不可逆 drop するより、短い固定遅延内で candidate を保持する方がよい。

```ts
type StreamingCandidate = {
  onsetTimeSec: number;
  emittedAtSec: number;
  noteName: string;
  midi: number;
  confidence: number;
  rank: number;

  evidence: {
    broadbandOnset: number;
    perNoteAttack: number;
    frameBody: number;
    backwardAttackGain?: number;
    lateDecayRisk: number;
    partialCollisionRisk?: number;

    pestoPitchHz?: number;
    pestoConfidence?: number;

    basicPitchOnset?: number;
    basicPitchFrame?: number;
    basicPitchContourHz?: number;
  };

  state: "ATTACK" | "BODY" | "LATE_DECAY" | "RESONANCE_ONLY";
  decision: "accepted" | "alternate" | "deferred" | "rejected";
  reason: string;

  alternatives: Array<{
    noteName: string;
    midi: number;
    confidence: number;
    reason: string;
  }>;
};
```

この object は最終 schema ではなく、research / prototype 用の検討形。
重要なのは、次の evidence を一つの score に潰しすぎないこと。

- broadband onset
- per-note attack
- frame/body energy
- backward attack gain
- late decay risk
- partial collision risk
- PESTO pitch confidence
- Basic Pitch onset/frame/contour

評価指標としては、1-best F1 だけでなく:

- Candidate Recall@K
- Correction Burden
- drop-to-candidate rate

を入れる。

---

## 4. Thread / latency 設計

### 4.1 推奨 thread 構成

#### Main/UI thread

担当:

- UI
- waveform / piano-roll / score preview
- settings
- results display

置かないもの:

- FFT
- resampling
- ONNX inference
- heavy JSON processing

#### AudioWorkletProcessor

担当:

- microphone / playback audio 受け取り
- mono downmix
- ring buffer write
- minimal metering

置かないもの:

- FFT / VQT
- ONNX inference
- allocation-heavy processing
- JSON serialize
- candidate decoding

#### Dedicated Worker

担当:

- SharedArrayBuffer ring buffer から hop 単位で読み出し
- resampling
- windowing
- FFT / VQT / resonator bank
- onset detection
- candidate decoding
- 必要なら ONNX Runtime Web inference

#### WASM DSP module

担当:

- FFT / filterbank / VQT kernel / feature extraction
- 既存 `kalimba-dsp` の browser-side port

注意:

- AudioWorklet 内 WASM は可能だが、重い解析は Worker 側 WASM に置く方が安定。
- WASM heap と JS typed array の copy cost を測る。
- preallocated buffer と shared memory 設計が重要。

### 4.2 レイテンシ設計

- 48 kHz の 128 frames は約 2.67 ms。
- 44.1 kHz の 128 frames は約 2.90 ms。
- AudioWorklet の処理予算は非常に小さい。
- onset preview は 5–10 ms hop 程度から始める。
- note 確定は 30–80 ms の bounded delay を許容する二段階 UX が現実的。

---

## 5. このプロジェクトへの取り込み順

### 5.1 Browser offline parity harness

最初に realtime ではなく、固定 WAV / AudioBuffer を browser/TS で処理し、Python fixture と比較できる harness を作る。

含めるもの:

- 既存 WASM `chunk_spectrum` / `rank_tuning_candidates`
- Basic Pitch TS baseline
- Python output との candidate rank 比較
- false positive / missed note / Candidate Recall 比較

目的:

- browser 実行差分を realtime の複雑さなしで測る。

### 5.2 Candidate schema と評価指標

既存 recognizer の 1-best だけでなく、以下を本出力に近い形で保持する。

- Top-K candidates
- alternatives
- rejected-but-explainable candidates
- provenance
- confidence
- reason codes

Basic Pitch の onset/frame/contour 分離と `remaining_energy` 的 occupancy を参考にするが、kalimba では final prediction ではなく **candidate provenance** として扱う。

### 5.3 AudioWorklet + Worker + SharedArrayBuffer spine

ここでは認識精度を上げない。

測るもの:

- latency trace
- buffer underflow / overflow
- jitter
- GC 影響
- main thread blocking

### 5.4 既存 broadband onset / FFT scorer の Worker/WASM 移植

当面の本線は、既存 recognizer の broadband onset + per-note evidence + candidate grouping を browser に運ぶこと。
Basic Pitch / PESTO に置換しない。

### 5.5 Basic Pitch decoder 比較

`output_to_notes_polyphonic()` の構造を、kalimba candidate pipeline と比較する。

特に見るもの:

- onset peak → frame sustain
- remaining-energy occupancy
- residual rescue (`melodia_trick`)

ただし `melodia_trick` 相当は final note ではなく alternate candidate に限定する。

### 5.6 PESTO + ONNX Runtime Web POC

目的:

- Worker 内 ONNX inference
- cache state 更新
- WASM/WebGPU EP 比較
- model asset 配布
- latency 計測

最終 transcription 目的ではない。

### 5.7 VQT / 17-tine resonator bank research

PESTO の streamable VQT は設計参照として強い。
一方、kalimba は音域・tine 数が限られるため、full VQT より **17-tine resonator / bounded log-frequency filterbank** の方が browser-side-only に合う可能性がある。

main に直投入せず、STFT scorer と dual-run し、以下で比較する。

- Candidate Recall@K
- Correction Burden
- resonance FP
- latency

### 5.8 MIDI / MusicXML / Web MIDI export

最後に接続する。

- MIDI file / MusicXML export は内部 note event が安定してから。
- live Web MIDI output は optional。

---

## 6. やらない方がよいこと

- AudioWorklet 内で重い NN inference を直接回す。
- AudioWorklet 内で FFT/VQT/candidate decoding を抱え込む。
- WebGPU / WebNN を MVP の必須条件にする。
- Python batch と browser realtime で別 schema を作る。
- まず CQT/VQT 全面置換から始める。
- Basic Pitch / PESTO を本体置換候補として扱う。
- 低レイテンシ化だけを追い、Candidate Recall / Correction Burden を測らない。
- MIDI / MusicXML / Web MIDI export を先に作る。

---

## 7. 未解決の懸念 / 追検証項目

- COOP/COEP を有効化すると、既存 Next.js app の埋め込み / 3rd-party resource / dev server に影響する可能性。
- SharedArrayBuffer を使うなら COOP/COEP が必要。
- multi-thread WASM を諦め、single-thread WASM から始める退避策も持つ。
- `kalimba-dsp` WASM と ORT Web WASM の二重ロードによる初期化時間・bundle size。
- Basic Pitch / PESTO の model asset 配布条件・ライセンス。
- AudioWorklet quantum や browser support は実装時に再確認する。

## 履歴

- 2026-06-26: 新規作成。browser/realtime 担当サーベイ結果を主ソースとして再稿。
