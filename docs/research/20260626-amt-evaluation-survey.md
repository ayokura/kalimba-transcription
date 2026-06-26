# AMT 評価指標・Benchmark サーベイ補遺 (2026-06-26)

## 目的

free-performance 化に向けて、既存の fixture exact-match / note F1 だけでは不足する。
本補遺は、AMT 評価指標・benchmark 担当サーベイの結果を主ソースとして、
`Candidate Recall` と `Correction Burden` を標準指標・一次ソースへ接続するための設計メモとして再稿したもの。

特にこのプロジェクトの性質:

- カリンバは減衰楽器で offset が曖昧。
- onset gate / weak attack / sympathetic resonance の扱いが認識品質を左右する。
- 最終出力は「編集可能なイベント列 → 楽譜」であり、ユーザー修正コストが重要。

に直結する指標だけを扱う。

---

## 1. 見るべき一次ソース / 標準実装

### 1.1 `mir_eval`: A Transparent Implementation of Common MIR Metrics

- Raffel et al., ISMIR 2014
- URL:
  - https://craffel.github.io/mir_eval/
  - https://github.com/mir-evaluation/mir_eval

見るべきもの:

- `mir_eval.transcription.precision_recall_f1_overlap`
- default `onset_tolerance=0.05` (50ms)
- default pitch tolerance: 50 cents
- default `offset_ratio=0.2` (20%)
- default `offset_min_tolerance=0.05` (50ms)
- Average Overlap Ratio (AOR / IoU 的な overlap 指標)

このプロジェクトへの含意:

- project-specific 指標を作る前に、まず mir_eval 系の標準 note transcription 指標を基準線として置く。
- `Candidate Recall` / `Correction Burden` は mir_eval の代替ではなく、**上乗せ指標**。
- `offset_ratio=None` の onset-only 評価と、offset-aware 評価を分けて出す。

### 1.2 Investigating the Perceptual Validity of Evaluation Metrics for Automatic Music Transcription

- Ycart et al., TISMIR 2020
- URL: https://transactions.ismir.net/articles/10.5334/tismir.57

見るべきもの:

- 既存 AMT 指標と人間の知覚的な転写品質評価の相関。
- PEAMT。
- ピアノのように offset が不明瞭な減衰楽器では、onset-only F尺度が人間判断と強く相関する、という結論。

このプロジェクトへの含意:

- カリンバもダンパーのない減衰楽器であり、offset の物理的定義は曖昧。
- 主評価は onset / note-on 側へ寄せる。
- offset-aware overlap は sustain 表現や notation layer の補助指標に留める。

### 1.3 Onsets and Frames: Dual-Objective Piano Transcription

- Hawthorne et al., ISMIR 2018 / arXiv 2017
- URL: https://arxiv.org/abs/1710.11153

見るべきもの:

- onset detector と frame detector を分ける設計。
- 推論時に onset detector が認めない限り新規 note-on を許可しない制約。
- onset tolerance 50ms が弱い打鍵の評価では緩すぎる場合がある、という議論。

このプロジェクトへの含意:

- `use_onset_gate=True` の方向性を補強する。
- ただし現実装は broadband / per-note / backward-attack の3証拠が全滅した時だけ棄却する弱 AND gate。
- 今後は「強く捨てる」より、低 confidence candidate へ降格して review 可能にする方向が安全。

### 1.4 Evaluating Automatic Polyphonic Music Transcription / MV2H

- McLeod & Steedman, ISMIR 2018
- Paper: https://ismir2018.ircam.fr/doc/pdfs/148_Paper.pdf
- Implementation: https://github.com/apmcleod/MV2H

見るべきもの:

- Multi-pitch / Voice / Meter / Value / Harmony を統合評価する MV2H。
- 最終的な sheet music quality を見据える評価。
- WER / 編集距離的な考え方。

このプロジェクトへの含意:

- `Correction Burden` は完全に独自発明ではなく、audio-to-score / music transcription の編集距離思想へ接続できる。
- ただし、いきなり MV2H 全体を採用しない。
- まず acoustic event correction と notation layer correction を分ける。

### 1.5 Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge

- NeurIPS 2025 / arXiv 2026
- URL: https://arxiv.org/abs/2603.27528

見るべきもの:

- Multi-instrument Note Onset F1 などの challenge 指標。
- 制約付き合成データ (FluidSynth / MIDI) であっても、polyphony が大きな壁として残る点。
- MT3 baseline を上回るチームが限定的である点。

このプロジェクトへの含意:

- 大規模 NN を本体置換候補にしない判断を補強する。
- 一方で、polyphony / timbre variation / recording condition を free-performance corpus に入れる必要がある。
- 合成データ benchmark の成功を実録音 kalimba へそのまま一般化しない。

### 1.6 On Calibration of Modern Neural Networks

- Guo et al., ICML 2017
- URL: https://arxiv.org/abs/1706.04599

見るべきもの:

- Expected Calibration Error (ECE)
- Reliability diagram
- confidence と正解率のズレ

このプロジェクトへの含意:

- `quality_indicators.py` / `needsReview` / low-confidence candidate の評価に使う。
- 目標は「全部見てください」ではなく、「怪しいと言った箇所が本当に怪しい」状態にすること。

---

## 2. このプロジェクトに効く評価指標

### 2.1 Onset-only metrics を主軸に置く

#### Onset-only Precision / Recall / F1

標準量への接続:

- `mir_eval.transcription` を `offset_ratio=None` 相当で使う。
- onset tolerance はまず 50ms を基準にする。
- pitch tolerance はまず 50 cents を基準にする。

このプロジェクトでの意味:

- onset-only Recall は「弾かれた物理 event を見逃していないか」を見る。
- onset-only F1 は「FP を増やしすぎず note-on を拾えているか」を見る。
- offset は主評価から外す。

注意:

- `Candidate Recall` そのものとは同一ではない。
- Onset-only Recall は標準指標、Candidate Recall は候補保持を含む project-specific 指標。
- ただし、Candidate Recall の外部説明時には onset-only Recall が最も近い標準概念になる。

### 2.2 Candidate Recall@K

定義:

> ground truth note / note-set が、1-best event だけでなく、`alternateGroupings`, dropped candidate,
> `rankedCandidates` Top-K などの候補集合に含まれる率。

目的:

- 正解を完全に捨てたのか、候補には残したが ranking / UI が未熟なのかを分ける。
- #178 multi-candidate の進捗を測る。
- hard drop の危険性を定量化する。

標準指標との接続:

- onset-only Recall に近いが、candidate set を対象にする点が異なる。
- 外部文脈では「onset-only candidate recall」「top-K note event recall」などへ翻訳する。

注意:

- 候補乱発で簡単に上がる。
- 単独 KPI にしない。
- `Correction Burden`, `candidateSlotsPerEvent`, `ReviewBurden` と対で見る。

### 2.3 Correction Burden

定義:

> predicted event sequence から ground truth event sequence へ到達するための semantic edit cost。

標準指標との接続:

- MV2H / audio-to-score 評価の WER / 編集距離思想。
- 非対称 F-beta: FN を重く見る場合、Recall 側を重くする `β > 1` と接続できる。

このプロジェクトでの意味:

- 1-best が外れていても、候補に残っていれば修正コストは低い。
- missing event が候補にもなければ修正コストは高い。
- UX 上の価値を F1 より直接測る。

注意:

- UI 操作設計に依存する。
- 初期は粗い重みでよい。
- 後で Review UI の実操作ログから調整する。

### 2.4 Onset-only と Offset-aware を分ける

主軸:

- Onset-only Precision / Recall / F1
- Candidate Recall@K
- Correction Burden

補助:

- Offset-aware overlap ratio
- AOR / IoU
- sustain / notation duration quality

理由:

- カリンバは長く減衰し、late decay と sympathetic resonance の境界が曖昧。
- offset を厳密評価に入れると、知覚的妥当性が下がりやすい。
- sheet music の duration は acoustic offset ではなく notation layer の問題として扱う。

### 2.5 Confidence calibration

指標:

- Expected Calibration Error (ECE)
- reliability diagram
- flagged event precision
- missed-error rate
- high-confidence wrong rate
- low-confidence correct rate

目的:

- `needsReview` が本当に誤りやすい箇所を指しているかを測る。
- low-confidence candidate を出しすぎてユーザーを疲れさせていないかを見る。

---

## 3. 実装メモ

### 3.1 `note_f1_benchmark.py` の拡張

出力 JSON 例:

```json
{
  "fixtureId": "...",
  "oneBest": {
    "onsetPrecision": 0.91,
    "onsetRecall": 0.94,
    "onsetF1": 0.925,
    "offsetAwareF1": 0.81,
    "averageOverlapRatio": 0.72
  },
  "candidates": {
    "recallAt1": 0.94,
    "recallAt3": 0.98,
    "recallAt5": 0.99,
    "candidateSlotsPerEvent": 2.1,
    "hardMissRate": 0.03
  },
  "correction": {
    "estimatedCost": 12,
    "candidateAssistedFixRate": 0.73,
    "insertions": 2,
    "deletions": 1,
    "noteAdds": 4,
    "noteRemoves": 2,
    "mergeSplits": 1,
    "gestureRelabels": 1
  },
  "confidence": {
    "flaggedEventPrecision": 0.64,
    "missedErrorRate": 0.12,
    "highConfidenceWrongRate": 0.04
  }
}
```

最初から完璧な UI 操作モデルにしない。
まずは以下を測る:

1. 正解が 1-best にあるか。
2. 正解が候補にはあるか。
3. 候補を使えば低コストで直せるか。
4. low-confidence flag が実際に役立つか。

### 3.2 mir_eval 依存の扱い

選択肢:

1. `mir_eval` を dev dependency / analysis script dependency として使う。
2. 依存を増やしたくない場合、TP/FP/FN matching ロジックを NumPy / pure Python で小さく移植する。

注意:

- core API runtime の依存に入れる必要はない。
- benchmark / analysis script 層で使う。

### 3.3 onset tolerance

初期値:

- 50ms を標準値として採用。

ただし:

- ground_truth 側の `toleranceSec` を優先する。
- weak attack / soft attack / human annotation uncertainty では個別 tolerance を使う。
- BPM 適応 tolerance は research option として検討する。

BPM 適応案:

- 固定 50ms だけでなく、周辺 tempo に基づく 1/32 音符長などを上限/下限付きで試す。
- ただし評価値が変わりやすいため、CI の regression target に入れる前に frozen policy を決める。

### 3.4 Candidate matching

Candidate Recall を測るには、benchmark 時に以下を集める必要がある。

- 1-best selected notes
- `alternateGroupings`
- soft candidate alternates
- dropped segment candidates
- `rankedCandidates` Top-K
- reason codes
- candidate confidence

現状 `rankedCandidates` / `secondaryDecisionTrail` は debug 限定。
本番 response にすぐ出さなくても、benchmark JSON には出せるようにする。

### 3.5 CI への入れ方

completed fixture exact-match は維持する。
追加するなら別レイヤ:

- free-performance benchmark job
- candidate recall report
- correction burden report
- confidence calibration report

合否条件は最初は厳しくしすぎない。
まず baseline を保存し、差分を可視化する。

---

## 4. 危険な誤用

### 4.1 F1至上主義

FP を減らすために閾値を上げると、F1 が一見維持されても Candidate Recall が落ちることがある。
ユーザーにとっては「弾いた音が抜けている」方が重い場合が多い。

対策:

- Onset Recall
- Candidate Recall
- HardMissRate
- Correction Burden

を同時に見る。

### 4.2 offset 評価の過度な厳密化

ダンパーのないカリンバに対し、note length の20%以内などを合否判定の主軸にしない。
Offset-aware overlap は補助指標にする。

### 4.3 合成データへの過剰適合

2025 AMT Challenge も制約付き合成データであり、実録音 kalimba の acoustic regression とは違う。
合成データでの recall 向上を、物理状態遷移や practical fixture を無視した local hack で達成しない。

### 4.4 Candidate Recall / Correction Burden を外部標準指標と混同する

これらは project-specific 指標。
外部説明時は、次のように翻訳する。

- Candidate Recall → onset-only / top-K event recall
- Correction Burden → weighted edit distance / WER-like correction cost
- Confidence calibration → ECE / reliability / needs-review precision

### 4.5 exact-match を弱める

completed fixture exact-match はリファクタの安全網。
新指標は追加レイヤであり、既存 regression target の代替ではない。

---

## 5. 現行方針への反映

- 次の実装着手は `Candidate Recall` / `Correction Burden` の benchmark skeleton が最優先。
- #178 multi-candidate は UI だけでなく評価可能性の基盤。
- #18 corpus 収集では、free-performance audio だけでなく `ground_truth.json` の onset time / tolerance / method を同時に集める。
- #194 quality indicators は、ECE / needs-review precision / high-confidence wrong rate で再較正する。

## 履歴

- 2026-06-26: 新規作成。mir_eval / MIREX / AMT Challenge / onset annotation の観点を整理。
- 2026-06-27: 評価指標担当サーベイ結果を主ソースとして全文再稿。mir_eval, PEAMT, Onsets and Frames, MV2H, AMT Challenge, calibration を反映。
