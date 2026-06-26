# AMT 評価指標・Benchmark サーベイ補遺 (2026-06-26)

## 目的

free-performance 化に向けて、既存の fixture exact-match / note F1 だけでは不足する。
本補遺は、MIREX / mir_eval / AMT Challenge / onset annotation 系の一次ソースを整理し、
`Candidate Recall` と `Correction Burden` を追加する際の設計指針を残す。

---

## 1. 参照すべき一次ソース / 標準実装

### 1.1 mir_eval.transcription

- Source: `mir_eval.transcription` documentation
- 重要関数:
  - `precision_recall_f1_overlap(ref_intervals, ref_pitches, est_intervals, est_pitches, ...)`
  - onset tolerance default: `0.05s`
  - pitch tolerance default: `50 cents`
  - offset tolerance: ratio / min tolerance
  - Average Overlap Ratio (AOR) も返す

**このプロジェクトへの含意:**

まず project-specific 指標を作る前に、標準 note transcription 指標を明示的に置く。
`Candidate Recall` や `Correction Burden` は mir_eval の代替ではなく、**上乗せ指標**として扱う。

### 1.2 MIREX Multiple Fundamental Frequency Estimation & Tracking

- Source: MIREX Multiple-F0 task pages
- frame-level active F0 estimation と note tracking を分けて扱う。
- multi-F0 では frame-by-frame の active pitch、note tracking では onset / duration / pitch event が評価対象。

**このプロジェクトへの含意:**

kalimba recognizer でも、以下を分けるべき:

1. **Frame / spectral evidence**: ある時刻周辺にどの tine 成分が見えるか
2. **Event hypothesis**: いつ、何の note-on として採用するか
3. **Correction target**: UI 上でユーザーが修正する event / note set / merge-split

`rankedCandidates` は frame/evidence 層、`ScoreEvent` は event 層、Review UI は correction 層として分離する。

### 1.3 2025 Automatic Music Transcription Challenge

- Source: *Advancing Multi-Instrument Music Transcription: Results from the 2025 AMT Challenge*
- 重要点:
  - Multi-instrument Note Onset F1 などが主要指標。
  - 近年のモデルでも polyphony / timbre variation は難しい。
  - MT3 baseline を上回るチームは限定的で、challenge は「SOTA でも残る失敗」を見るのに有用。

**このプロジェクトへの含意:**

大規模 NN を本体置換候補にしない判断を補強する。
同時に、free-performance benchmark では polyphony / timbre variation / recording condition を意識して
corpus を組む必要がある。

### 1.4 Onset annotation / refinement

- Source: *Snapping Matters: Context-Aware Onset Refinement for Automatic Music Transcription* など
- 重要点:
  - onset label の精度は AMT 学習・評価に強く効く。
  - note onset は「検出」だけでなく「annotation snapping / refinement」の問題でもある。

**このプロジェクトへの含意:**

`ground_truth.json` の `timeSec` / `toleranceSec` は単なる補助情報ではなく、今後の評価基盤である。
`ear_verified`, `spectrogram_verified`, `aubio_cross_checked`, `user_corrected` の `method` を必ず残す。

---

## 2. 指標設計: 既存 + 追加

### 2.1 既存を維持する指標

#### Fixture exact-match

- completed fixture の regression target。
- リファクタ・spike の安全網。
- 弱めてはいけない。

#### Note-level F1

- free-performance の粗い品質指標。
- ただし既知 corpus では飽和しやすい。
- event grouping / review burden / confidence を測れない。

### 2.2 追加すべき指標

#### Candidate Recall@K

定義案:

> ground truth note / note-set が、1-best event だけでなく、
> `alternateGroupings`, dropped segment candidate, `rankedCandidates` の Top-K に含まれる率。

用途:

- recognizer が「正解を完全に捨てた」のか、「候補には残したが ranking / UI が未熟」なのかを分ける。
- #178 multi-candidate の進捗を測る。

注意:

- 候補乱発で簡単に上がるため、単独では使わない。
- `Correction Burden` とセットで見る。

#### Correction Burden

定義案:

> predicted event sequence から ground truth event sequence へ到達するための編集操作コスト。

操作単位の例:

- event delete
- event insert
- event enable dropped candidate
- note add/remove within event
- alternateGrouping select
- split event
- merge adjacent events
- timing nudge
- gesture relabel (`strict_chord` / `slide_chord` / `arpeggio` 等)

用途:

- ユーザーがどれだけ直しやすいかを測る。
- `Candidate Recall` の候補乱発を抑制する。

注意:

- 操作コストは UI 設計に依存する。
- 最初は粗い重みでよいが、Review UI の実装と同期して更新する。

#### Confidence calibration

定義案:

> recognizer が low confidence / needs review とした箇所が、実際に誤りやすいか。

用途:

- `quality_indicators.py` の再較正。
- ユーザーへ「どこを見ればよいか」を示す。

#### Drop-to-candidate rate

定義案:

> 従来 hard drop していた segment / candidate のうち、review 可能な candidate slot に残せた率。

用途:

- onset gate / secondary rejection / residual suppression の hard decision を multi-candidate 化する進捗指標。

---

## 3. 実装メモ

### 3.1 `note_f1_benchmark.py` の拡張方針

まず次の JSON を出す:

```json
{
  "fixtureId": "...",
  "oneBest": {
    "noteF1": 0.93,
    "onsetF1": 0.91
  },
  "candidates": {
    "recallAt1": 0.93,
    "recallAt3": 0.97,
    "recallAt5": 0.99,
    "candidateSlotsPerEvent": 2.1
  },
  "correction": {
    "estimatedCost": 12,
    "insertions": 2,
    "deletions": 1,
    "noteAdds": 4,
    "noteRemoves": 2,
    "mergeSplits": 1
  },
  "confidence": {
    "lowConfidenceErrorRate": 0.42,
    "highConfidenceErrorRate": 0.06
  }
}
```

最初から完璧な UI 操作モデルにしない。まずは「候補に残っているか」「編集距離が増えていないか」を測る。

### 3.2 `ground_truth.json` との接続

- `timeSec` / `notes` / `toleranceSec` を candidate matching に使う。
- `method` を保持する。
- model_suggested / teacher_suggested 由来は completed 昇格に使わない。

### 3.3 `rankedCandidates` の扱い

現状は debug 限定。
Candidate Recall を測るには、少なくとも benchmark 時に:

- 1-best selected notes
- alternateGroupings
- soft candidate alternates
- dropped segment candidates
- rankedCandidates top-K

を集める必要がある。

本番 response に即出さなくても、benchmark JSON には出す。

---

## 4. 危険な誤用

- **F1 と Candidate Recall を混同しない。**
  - Candidate Recall は「候補に残ったか」であって「ユーザーにとって正しい出力か」ではない。
- **Candidate Recall を単独 KPI にしない。**
  - 候補乱発で上がるため Correction Burden と対で見る。
- **exact-match を弱めない。**
  - completed fixture は recognizer リファクタの安全網。
- **teacher model の出力を ground truth と同一視しない。**
  - `method` で区別し、人間 verification 前に regression target にしない。
- **onset tolerance を後から無自覚に変えない。**
  - `toleranceSec` の変更は評価値を大きく変えるため、理由を記録する。

---

## 5. 現行方針への反映

- 次の実装着手は `Candidate Recall` / `Correction Burden` の benchmark skeleton が最優先。
- #178 multi-candidate は「UI便利機能」ではなく、評価可能性を上げる基盤。
- #18 corpus 収集では、free-performance audio だけでなく `ground_truth.json` の onset quality も同時に集める。

## 履歴

- 2026-06-26: 新規作成。mir_eval / MIREX / AMT Challenge / onset annotation の観点を整理。
