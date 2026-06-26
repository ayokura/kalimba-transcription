# 自動採譜 Product / UX サーベイ補遺 (2026-06-26)

## 目的

free-performance 転写では、recognizer が一発で完全正解を出すことよりも、
**ユーザーが短時間で正しい譜面へ直せること**が重要になる。

本補遺は、既存 audio-to-MIDI / automatic transcription 製品の UX パターンをもとに、
`Correction Burden` と Review UI の設計観点を整理する。

---

## 1. 見るべき製品 / UI パターン

### 1.1 Basic Pitch demo

特徴:

- audio upload / recording
- MIDI / note event output
- pitch bend / contour 由来の note event
- Web 上で試せる

このプロジェクトへの示唆:

- browser-side candidate preview の参考。
- 「完璧な楽譜」ではなく「MIDI 的な編集可能イベント」を返す考え方が近い。

### 1.2 AnthemScore

特徴:

- spectrogram / piano-roll / notation 的表示
- automatic transcription 後の手動修正前提
- MusicXML / MIDI export

示唆:

- 音響 evidence と譜面 event を並べて見せることが重要。
- correction workflow は「認識結果の後処理」ではなく製品の中心機能。

### 1.3 ScoreCloud / Klangio / Melody Scanner / Songscription 系

共通パターン:

- audio-to-score / audio-to-MIDI
- upload / record
- score viewer
- export (MIDI, MusicXML, PDF)
- edit mode / manual correction

示唆:

- 出力形式よりも、**修正しやすさ**と**export までの到達コスト**が UX の価値。
- perfect transcription を謳っても、実際は user correction が前提。

---

## 2. よくある修正作業

自動採譜 product でユーザーが直す作業は、おおむね以下に分解できる。

### 2.1 音高 / note set 修正

- note add
- note remove
- note replace
- chord 内の extra note 削除
- missing note 追加

kalimba での対応:

- secondary / tertiary の誤採用・取りこぼし。
- `rankedCandidates` / `secondaryDecisionTrail` を UI に出せると修正コストが下がる。

### 2.2 event 存在修正

- false positive event delete
- missing event insert
- dropped candidate enable

kalimba での対応:

- onset gate / residual suppression で捨てたものを `CandidateSlot` として残す価値が高い。

### 2.3 merge / split 修正

- 隣接 event を1つの chord / slide に統合
- 1つの event を複数 onset に分割
- arpeggio / slide / separated_notes の境界修正

kalimba での対応:

- #151 `alternateGroupings` はこの修正コストを下げる基盤。
- #6 arpeggio は UI 操作単位としても別カテゴリにすべき。

### 2.4 timing / rhythm 修正

- onset nudge
- duration adjust
- quantization grid change
- tempo / beat alignment correction

kalimba での対応:

- `ground_truth.json` の onset time / tolerance を評価に使う。
- notation quantization と acoustic event detection を分離する。

### 2.5 gesture / notation 修正

- strict chord vs slide chord
- arpeggio direction
- notation spelling / grouping
- tie / sustain handling

kalimba での対応:

- 音響 recognizer の責務と、記譜 UI の責務を分ける。
- gesture classification は confidence 付きで出す方がよい。

---

## 3. Correction Burden への落とし込み

### 3.1 操作コスト案

最初は粗い重みでよい。

| 操作 | コスト案 | 備考 |
|---|---:|---|
| alternateGrouping select | 1 | 候補に残っていれば低コスト |
| dropped candidate enable | 1 | candidate slot がある場合 |
| note add/remove within event | 1 | chord 修正 |
| event delete | 1 | FP 削除 |
| event insert from scratch | 3 | candidate がない missing event |
| split event | 2 | UI が支援するなら低下 |
| merge adjacent events | 2 | alternateGrouping があれば 1 |
| timing nudge | 1 | 小修正 |
| gesture relabel | 1 | strict/slide/arpeggio |

### 3.2 Candidate Recall との関係

同じ誤りでも、候補に残っているかで修正コストは大きく違う。

例:

- missing D5 が dropped candidate slot にある → enable で cost 1
- missing D5 がどこにもない → event insert + note select で cost 3+

したがって:

- Candidate Recall は「正解が候補に残っているか」
- Correction Burden は「候補を使ってどれだけ楽に直せるか」

を測る。

---

## 4. Review UI に入れるべき項目

### 4.1 イベント単位

- selected note set
- alternate note sets
- dropped candidate slots nearby
- confidence / needs review
- reason codes
- source fragments
- audition button

### 4.2 音響 evidence

- event audio playback
- onset marker
- narrow-band energy / pitch evidence preview
- candidate rank list

最初から spectrogram full UI を作る必要はない。
ただし「なぜこの候補があるか」が見える最低限の evidence は必要。

### 4.3 correction 操作

- enable candidate
- disable event
- add/remove note from candidate list
- select alternate grouping
- split / merge
- timing nudge
- reset to recognizer output

### 4.4 provenance

候補や修正の出自を保持する:

- `recognizer_1best`
- `recognizer_alternate`
- `dropped_candidate`
- `user_corrected`
- `teacher_basic_pitch`
- `teacher_pesto`

これは後で `ground_truth.json` / `corrections.json` を昇格する時に重要。

---

## 5. やらない方がよいこと

- 完璧な notation editor を先に作る。
- spectrogram UI を先に作り込みすぎる。
- correction 操作をログ化しない。
- teacher model と user correction を provenance なしで混ぜる。
- gesture classification をユーザーに見せないまま内部だけで確定する。

---

## 6. 現行方針への反映

- #16 review / repair workflow は、cosmetic ではなく free-performance readiness の中心。
- #178 multi-candidate は UX と評価指標の両方に効く。
- `Correction Burden` は UI 操作ログと接続して進化させる。
- 最初の MVP は「完璧な楽譜」ではなく「候補つきで直しやすいイベント列」を目標にする。

## 履歴

- 2026-06-26: 新規作成。audio-to-MIDI / 自動採譜 product の correction workflow 観点を整理。
