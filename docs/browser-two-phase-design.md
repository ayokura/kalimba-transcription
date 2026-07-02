# Browser two-phase architecture — 設計決定 (2026-07-03)

sprint-plan-2026-07 Sprint 5 の設計決定記録。browser/WASM トラックの
アーキテクチャ前提を確定させる。分析の背景は
[`browser-migration-analysis.md`](browser-migration-analysis.md) (2026-04)、
実装状態の検証基盤は A0 parity harness
(`crates/kalimba-dsp/check_wasm.sh` 4-5 段目、2026-07-03 導入) を参照。

## 決定事項サマリ

| # | 決定 | 一言で |
|---|---|---|
| 1 | **two-phase 構成を採用** (Phase P: causal preview / Phase F: batch finalize) | full-batch normalizer は構造的に streaming 不能なので分離必須 |
| 2 | **FFT は wasm 共有コア (onset_strength) に一本化。AnalyserNode は使わない** | parity 検証可能な実装を唯一の真実にする |
| 3 | **browser live の target SR = 48kHz 固定。SR 依存パラメータ (#140) は現状維持** | SR 正規化はチューニング再調整とセットでないと投入不可 (#140 の結論) |
| 4 | **COOP/COEP は当面有効化しない = single-thread wasm が本線** | 現行 wasm は単一スレッドで実時間要件を満たす。再評価トリガーを明記 |

---

## 1. Two-phase 構成

### なぜ必須か

`patterns.py` の repeated-pattern normalizer 群
(`normalize_repeated_four_note_family` / `normalize_repeated_four_note_gliss_patterns` /
`normalize_repeated_explicit_four_note_patterns` が `repeated_pattern_passes()` に配線済み)
は「全 event を走査して dominant pattern を特定し、外れ値を補正する」full-batch
処理で、演奏が終わるまで dominant pattern が確定しない。causal 化は不可能
(browser-migration-analysis.md obstacle 1)。

### 構成

```
[Phase P — preview (browser, 演奏中, causal)]
  mic → AudioWorklet (A3) → wasm onset_strength → causal onset_detect (A1)
      → chunk_spectrum + rank_tuning_candidates (top-1) → 暫定譜面表示
  遅延許容: onset 確定まで ≤50ms (A1 の有界遅延) + 1-event 表示遅延

[Phase F — finalize (録音完了後, batch)]
  全量 audio → 現行 recognizer フルパイプライン
  (segments 全体 / repeated-pattern normalizers / tempo / candidateSlots)
  → 確定譜面で Phase P の暫定表示を置換
```

- **Phase F の実行場所は当面 server (現行 API)**。既存の
  `/api/transcriptions` がそのまま Phase F を務めるので、two-phase の初期形は
  「browser preview + 既存 API」の組み合わせで成立する。B スライス
  (segments 移植) が進んだ時点で in-browser batch に置換可能になるが、
  それはこの設計の後段であり前提ではない。
- **Phase P → F の契約**: Phase P の出力は provisional events
  (onset 時刻 + top-1 note、モノフォニック)。Phase F の確定出力は現行
  response schema (events + alternateGroupings + candidateSlots)。
  UI は provisional → final の置換を前提に作る (#16 の event-first
  correction workflow と同じ event 単位)。
- **1-event latency**: 現行 pipeline の forward-looking 判定
  (次 onset を見て duration/gesture を確定) は Phase P では「暫定値で即表示、
  次 onset 到着で更新」に置き換える (obstacle 2 への回答)。

### FFT の一本化 (AnalyserNode を使わない)

2026-04 分析は real-time 経路に WebAudio `AnalyserNode` を挙げていたが、
**採用しない**。理由: AnalyserNode は窓関数・hop・mel 投影を制御できず、
Python/native と数値突合できない別実装になる。A0 parity harness で
`audio → onset_strength → onset_detect` の wasm 通し経路が numpy オラクルと
frame-exact 一致することを確認済み (44.1k/48k/96k、4 fixture、20/20)。
この検証可能性を捨てる理由がない。A2 (incremental onset_strength) も
同じ wasm STFT の増分化として実装し、batch 版との parity を A0 基盤で pin する。

## 2. Target SR 方針 (#140 との関係)

- **browser live 経路は 48kHz 固定** (`new AudioContext({ sampleRate: 48000 })`、
  wasm-demo が既にこの形)。根拠: wasm-demo の SR sweep 実測で pitch 一致率は
  32k まで 100%、22k で ~97%、16k 以下で高音部分音が Nyquist 割れ。48k は
  マージン込みの安全域で、getUserMedia のネイティブレートでもある。
- **ファイルアップロード経路も 48k にリサンプルされて処理される**
  (WebAudio decodeAudioData の仕様)。96k ソースの frame-exact parity が
  必要な検証には `OfflineAudioContext` を使う (onset.ts の JSDoc 参照)。
- **`FRAME_LENGTH=2048` / `HOP_LENGTH=256` (固定サンプル数) は当面維持**。
  #140 の結論の通り、SR 正規化 (案A: 入力リサンプル統一 / 案B: パラメータ
  時間ベース化) は onset timing 全体をシフトさせ、下流閾値の再チューニングと
  セットでないと投入できない。過適合ゲート (非飽和 GT ≥2) が満たされるまで
  着手しない。
- **緩和要因**: A0 parity harness が 44.1k / 48k / 96k の 3 点で onset 経路を
  pin しているため、将来 #140 を動かす時に SR 別の回帰が自動測定できる。
  96k で STFT 窓が 21.3ms になる事実 (44.1k の 46.4ms に対し) と、それが
  E162 型の分離問題に効いている可能性は #140 に記録済み。

## 3. COOP/COEP 方針

**決定: 当面 COOP/COEP ヘッダを有効化しない。** SharedArrayBuffer /
multi-thread wasm / ワーカ間共有メモリを前提にしない。

- 根拠:
  - 現行 wasm は single-thread で 17.8s 録音の onset+pitch を ms オーダーで
    処理 (wasm-demo 実測表示)。Phase P の実時間要件 (hop=256 ≒ 5.3ms/frame
    @48k) に対し計算余裕が大きい。
  - COOP/COEP を Next.js アプリ全体に適用すると、外部リソース埋め込み・
    dev server・将来の第三者統合に波及する (browser survey §7 の警告)。
    single-thread 退避策を「退避」ではなく本線にすれば、この複雑さを
    最初から回避できる。
- **再評価トリガー** (いずれかが実測で発生した時のみ再検討):
  1. A2/A3 の streaming 処理が frame budget を超える実測が出た
  2. ONNX Runtime Web (PESTO spike、S7 以降) が multi-thread を要求した
  3. 再評価する場合も、アプリ全体ではなく **専用 route への route-level
     headers 適用** (Next.js の headers() で path 限定) を第一候補とする。

## 4. スライスへの帰着

| スライス | 内容 | 状態 |
|---|---|---|
| A0 | offline parity harness (onset 通し + segment 参照/comparator) | **完了 (2026-07-03)** — check_wasm.sh 4-5 段目、CI 配線済み |
| A1 | causal peak_pick + backtrack (有界遅延 ≤50ms)、batch 版と F1±2% pin | S6 分岐の fallback 本命 (中間レビュー時点で条件未成立見込み) |
| A2 | incremental onset_strength (wasm STFT の増分化) | A1 完了後 |
| A3 | AudioWorklet spine (mic → ring buffer → wasm) | parity gate 後 (精度に寄与しないため最後) |
| B1 | segments.py の active range 計算移植 → A0 の compareSegments で pin | 着手条件 = A0 完了 (満了)。TS 移植 vs Rust 共有コア拡張の方針決定込み |

B1 の実装言語判断 (TS vs Rust 共有コア拡張) はまだ確定しない。判断材料:
active range 計算は秒ベース閾値の軽量ロジックで TS でも書けるが、
`compareSegments` での frame-exact pin と「同一コアを server/browser で共有する」
現方針 (kalimba_dsp の成功パターン) は Rust 拡張側に分がある。B1 着手時に
active range 計算の依存範囲 (attack profile 系を引き込むか) を見て決める。
