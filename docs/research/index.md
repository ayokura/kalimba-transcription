# Research Notes Index

このディレクトリは、カリンバ自動採譜 (AMT) の研究サーベイ、現実装への適用判断、将来の検証メモを置く。

## 2026-06-26/27 再サーベイの結論

2026-06-26/27 の再サーベイで、`20260406-*` の LLM 調査レポート由来のバイアスを明示し、現実装の事実と最新の隣接研究を突き合わせ直した。

現時点の設計判断では、以下を優先する。

1. **認識器の大改造より先に、free-performance 評価ループを作る。**
2. **1-best 精度だけでなく、Candidate Recall@K / Correction Burden / ConfidenceCalibration を測る。**
3. **multi-candidate / correction UX を free-performance readiness の中心に置く。**
4. **per-tine partial 既定化や note-state machine 化は、本線ではなく ablation / research spike で検証する。**
5. **Basic Pitch / PESTO / MT3 系は本体置換ではなく、baseline / teacher / candidate generator として扱う。**

## まず読むべき文書

### 1. バイアス除去版の総合再評価

- [`20260626-unbiased-amt-reassessment.md`](./20260626-unbiased-amt-reassessment.md)

旧サーベイのバイアス、現実装の事実、最新研究の読み直し、now/next/later の大局方針をまとめた入口。
新規の設計判断では、まずこの文書を確認する。

### 2. 評価指標・benchmark

- [`20260626-amt-evaluation-survey.md`](./20260626-amt-evaluation-survey.md)

mir_eval / PEAMT / Onsets and Frames / MV2H / AMT Challenge / calibration をもとに、Candidate Recall@K、Correction Burden、ConfidenceCalibration の位置づけを整理。

次の実装候補である `note_f1_benchmark.py` 拡張時に参照する。

### 3. Product UX / correction workflow

- [`20260626-transcription-product-ux-survey.md`](./20260626-transcription-product-ux-survey.md)

Basic Pitch demo、AnthemScore、ScoreCloud、Klangio、Melody Scanner などの公式情報・公開レビューから、Review UI と Correction Burden の操作単位を整理。

#16 review / repair workflow、#178 multi-candidate output の設計時に参照する。

### 4. Browser / realtime 実装

- [`20260626-browser-realtime-implementation-survey.md`](./20260626-browser-realtime-implementation-survey.md)

Basic Pitch Python/TS、PESTO、ONNX Runtime Web、AudioWorklet + Worker + WASM の実装知見を整理。

browser-side-only / streaming spike 時に参照する。

### 5. Kalimba / mbira 音響

- [`20260626-kalimba-acoustics-survey.md`](./20260626-kalimba-acoustics-survey.md)

Chapman / McNeil & Mitran などの一次ソースを確認し、引用メタデータの誤り、未検証の倍音比レンジ、attack 中の pitch 確定リスク、corpus 収集項目を整理。

per-tine partial / note-state / resonance handling の spike 前に参照する。

## 2026-04-06 旧サーベイ群の扱い

以下は履歴として残すが、新規の設計判断ではそのまま根拠にしない。

- [`20260406-deep-research-report.md`](./20260406-deep-research-report.md)
- [`20260406-kalimba_amt_survey.md`](./20260406-kalimba_amt_survey.md)
- [`20260406-research-to-implementation-mapping.md`](./20260406-research-to-implementation-mapping.md)

理由:

- LLM 調査レポートを一次ソースにした二次資料である。
- `citeturn..search` 形式の LLM 検索トークンが未編集で残っている。
- 「インパクト: 大/中」ランクが ablation 等で実測されていない。
- 実装現状とズレている箇所がある (例: onset gate は現在実装済み、per-tine partial scoring は実装済みだが既定無効)。

詳細は [`20260406-research-to-implementation-mapping.md`](./20260406-research-to-implementation-mapping.md) 冒頭のバイアス警告を参照。

## 関連 issue への索引

今回の再サーベイ結果は、以下の issue に短い索引コメントとしても記録している。

- #18 corpus diversification
- #178 multi-candidate output
- #16 review / repair workflow
- #194 quality indicators / confidence calibration

## 次にやるなら

**(2026-07-02 更新)** 当初推奨していた `note_f1_benchmark.py` への skeleton 追加 (Candidate Recall@K / Correction Burden / HardMissRate / ConfidenceCalibration) は **ab2cca3 で実装済み** (`test_note_f1_benchmark_metrics.py` が検証)。

**(2026-07-05 更新)** 次の作業は現行計画 [`docs/sprint-plan-2026-07c.md`](../sprint-plan-2026-07c.md) (第 3 期) と [#203](https://github.com/ayokura/kalimba-transcription/issues/203) の最新コメントに従う。旧 `sprint-plan-2026-07.md` (第 1 期) は superseded であり、この索引から作業を拾わないこと。

## 履歴

- 2026-06-27: 新規作成。2026-06-26/27 再サーベイが一段落したため、読み順と設計判断上の優先文書を整理。
