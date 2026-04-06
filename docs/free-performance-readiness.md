# Free Performance Readiness Assessment

## Purpose

各 recognizer コンポーネントについて、Free Performance（楽譜知識なし・Expected Performance なしの自由演奏転写）への適合度を評価する。チケット処理のたびに関連コンポーネントを再評価し、このドキュメントを更新する。

**最終更新: 2026-04-06 (コミット 00d43dc, #126 gate調整)**

## 評価基準

| レベル | 意味 |
|--------|------|
| **Ready** | Free Performance でそのまま使える設計 |
| **Mostly Ready** | 概ね汎用だが、一部に fixture/楽譜依存の前提あり |
| **Needs Work** | 動作するが設計上の制約が Free Performance を阻害する |
| **TBD** | 今回未評価 |

---

## Stage 1: Audio Input (`audio.py`)

**評価: TBD**

- `read_audio()`: モノラル変換、無音チェック
- Free Performance 固有の懸念は少ないと思われるが未評価

---

## Stage 2: Onset & Segment Detection (`segments.py`)

**評価: TBD**

- `detect_segments()`: librosa RMS + onset detection + attack profile validation
- 34-key BWV147 で NO MATCH が 4件 (E46, E62, E83, E162) 残っており onset detection 層の問題
- streaming/causal 化の際に最も大きな再設計が必要になる可能性

### 未評価項目
- active range 検出の楽譜非依存性
- multi-onset gap collector の汎用性
- attack profile validation の閾値がBWV147固有かどうか

---

## Stage 3: Per-Segment Peak Detection & Candidate Selection (`peaks.py`)

**評価: Mostly Ready（candidate ranking） / Needs Work（secondary/tertiary selection + rescue path）**

### 良い点

- **Candidate ranking (`rank_tuning_candidates`)** は楽譜非依存。FFT + harmonic scoring でチューニングノートをスコアリングする純粋にスペクトルベースの処理
- **`is_physically_playable_chord`** はチューニング定義のみに依存し、楽譜知識を使わない
- **Octave dyad判定 (`allow_octave_secondary`)** は fundamental_ratio ベースで汎用的。今回の閾値緩和 (0.85→0.75) で octave-4 ノートの過度な棄却が解消。non-octave-4 の閾値 (0.32) と比較して 0.75 はまだ保守的
- **Evidence gate (`onset_gain`, `backward_attack_gain`)** はセグメント内の物理的な attack 特性に基づく判定であり、楽譜に依存しない

### 懸念点

- **Rescue path の複雑性**: `_evidence_rescue_gate` が 5 層の条件分岐に成長。Phase B の gate 設計意図との関係が不透明。gate 調整のたびに分岐が増える構造的問題 (#111 に記録済み)
- **Sequential accept loop**: primary → secondary → tertiary の逐次選択は、候補の評価順序に結果が依存する。E136 の問題（E4 が先に棄却されたために C4 が playability チェックに失敗）はこの構造的制約の典型例。chord selector (#111) で根本解決の可能性あり
- **Octave-4 fr 閾値 (0.75)**: 今回の緩和は BWV147 の2件で検証済みだが、Free Performance での広範な文脈で偽 octave dyad を生むリスクは broader fixture coverage がないと評価困難
- **Tertiary rescue bypass (og >= 2.0)**: carryover rescue と同じ閾値の流用であり、tertiary rescue に最適な閾値かは理論的根拠が弱い
- **Ranked candidate 不在問題**: E148 C6 のように、segment 全体の FFT では primary の倍音に吸収されて ranked に入らない genuine note がある。onset-focused FFT window 等の spectral acquisition 改善が必要 (#125)

---

## Stage 4: Raw Event Aggregation (`pipeline.py`)

**評価: TBD**

- `recent_note_names`, `ascending_run_ceiling` 等の文脈状態を構築
- sparse gap tail filtering にレジスタ・下降パターンのヒューリスティクスあり

### 未評価項目
- 文脈状態の構築が Expected Performance に依存しないか
- sparse gap tail filtering の汎用性

---

## Stage 5: Event Post-Processing (`events.py`)

**評価: Needs Work**

- 27 個の suppression/collapse/simplify 関数が逐次適用される
- `recognizer-local-rules.md` に記録されている fixture-specific ルールが複数含まれる
- 適用順序に結果が依存する

### 既知の fixture-specific debt
- `collapse_restart_tail_subset_into_following_chord` (events.py)
- `suppress_recent_upper_echo_mixed_clusters` (patterns.py)
- `lower-mixed-roll-extension` (peaks.py)
- ~~`collect_two_onset_terminal_tail_segments`~~ — #122 で削除済み（AVC trailing collector で代替）
- `recent-upper-octave-alias-primary` promotion — Stage 3 (peaks.py) のスコープ。詳細は Stage 3 の懸念点を参照

詳細は [recognizer-local-rules.md](recognizer-local-rules.md) ��照。

---

## Stage 6: Pattern Recognition (`patterns.py`)

**評価: Needs Work**

- `apply_repeated_pattern_passes()`: repeated four-note, triad, gliss パターンの正規化
- 「繰り返しパターン」の検出は corpus-wide な dominant pattern 書き換えに依存しており、Free Performance ではパターンの事前知識がない
- AGENTS.md に「Treat repeated-pattern normalizers as suspicious until proven necessary」と記載あり

### 未評価項目
- ablation で各パスの影響度がどの程度か
- Free Performance で repeated pattern pass を全て無効化した場合の影響

---

## Stage 7: Final Merging & Adjacency (`events.py`)

**評価: TBD**

- `merge_adjacent_events()`, `merge_short_chord_clusters()`
- ascending run での dyad split

### 未評価項目
- merge 判定がチューニング依存のみか、文脈依存があるか

---

## Stage 8: Quantization & Notation (`notation.py`)

**評価: Ready**

- beat-quantized representation へのマッピング
- 入力はイベントの timing と note set のみ
- 楽譜知識不要、Free Performance でそのまま使用可能

---

## Stage 9: Output Assembly (`pipeline.py`)

**評価: Ready**

- `ScoreEvent` 構築と `TranscriptionResult` パッケージング
- 楽譜知識不要

---

## Cross-cutting concerns

### Streaming/Causal 化

- 現在のパイプラインはバッチ処理。segment detection が全音声を必要とする
- `detect_segments` の active range 検出と onset detection が最大のボトルネック
- peaks.py の candidate selection は segment 単位で独立しており、streaming 親和性が高い
- event post-processing は前後の event 文脈に依存するため causal 化に工夫が必要

### Browser-side 実装

- librosa (onset detection), numpy (FFT) への依存が core algorithm にある
- peaks.py の scoring logic は数値演算のみで WebAssembly/WebAudio に移植可能
- segments.py の RMS/onset detection は WebAudio API の AnalyserNode で代替可能だが精度差の検証が必要

---

## 更新履歴

| 日付 | チケット | 更新内容 |
|------|----------|----------|
| 2026-04-06 | #126 | 初版作成。Stage 3 (peaks.py) を詳細評価。octave dyad 閾値緩和 + rescue bypass の Free Performance 影響を記載 |
