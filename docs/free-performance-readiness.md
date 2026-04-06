# Free Performance Readiness Assessment

## Purpose

各 recognizer コンポーネントについて、Free Performance（楽譜知識なし・Expected Performance なしの自由演奏転写）への適合度を評価する。チケット処理のたびに関連コンポーネントを再評価し、このドキュメントを更新する。

**最終更新: 2026-04-06 (コミット 7e2b6f8, Stage 2/7 評価追加)**

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

**評価: Mostly Ready（onset/active range） / Needs Work（SR依存性）**

**最終更新: 2026-04-06 (コミット 7e2b6f8, #140 SR正規化調査)**

### 良い点

- **Active range 検出**: RMS ベースで楽譜非依存。threshold = max(0.18*max_rms, 2.2*median_rms) は adaptive で楽器特性に追従
- **Attack profile validation**: broadband_gain + high_band_flux の組み合わせ判定。moderate gain-flux gate (gain≥3.0, flux≥0.8) の追加で 34-key の genuine attack 2件を救済済み。閾値は物理量ベースで fixture-specific でない
- **Gap collector (AVC)**: onset の attack profile で判定しており因果的・楽譜非依存
- **librosa onset_strength + onset_detect**: 標準的な broadband spectral flux。大半の onset を正しく検出（17-key 160/163, 34-key 156/163）

### 懸念点

- **SR 依存性 (#140)**: FRAME_LENGTH=2048, HOP_LENGTH=256 が固定サンプル数。sr=44100 で STFT窓=46.4ms、sr=96000 で 21.3ms。onset 検出の時間分解能が楽器/録音環境で異なる。リサンプル実験で -5 exact 回帰、n_fft 変更で -4 exact — チューニング再調整なしの単独投入は不可
- **Polyphonic onset の限界**: 単音 attack が他の音の残響に埋もれると onset_strength に現れない（E162 B4: onset_strength=0.6 vs background 0.3-0.5）。per-note onset detection で補完する設計を策定済み（docs/per-note-onset-detection-design.md）
- **Streaming 再設計**: librosa.onset.onset_strength は batch 処理。streaming 化には incremental spectral flux + peak picking の再実装が必要。per-note onset の部品（`_note_band_energy()` 等）は因果的で streaming 互換

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

**評価: Mostly Ready（suppress/simplify 大半） / Needs Work（パターン依存 3関数）**

**最終更新: 2026-04-06 (suppress 系全関数棚卸し)**

27 個の suppression/collapse/simplify/merge 関数が line 191-229 で逐次適用される。個別評価の結果、大半は汎用的だが一部にパターン依存あり。

### 依存度別分類

**なし（時間 + ノート構造のみ）— 8関数:**
- `suppress_onset_decaying_carryover` — onset_gain < 1.0 の減衰二次ノート削除
- `suppress_leading_single_transient` — 先頭の短い transient 削除 (≤0.1s)
- `suppress_leading_gliss_subset_transients` — gliss 前の低信頼 transient 削除
- `suppress_low_confidence_dyad_transients` — 低信頼 dyad 削除 (score ≤ 0.5)
- `simplify_short_gliss_prefix_to_contiguous_singleton` — gliss 前の 2-note → 1-note
- `suppress_leading_gliss_neighbor_noise` — gliss 前の noise 削除
- `suppress_subset_decay_events` — subset decay 削除 (gap ≤ 0.02s)
- `suppress_short_residual_tails` — 短い residual tail 削除 (≤0.14s)

**低（tuning step / cents 距離チェック）— 6関数:**
- `suppress_leading_descending_overlap` — 下降 dyad の上位ノート絞り込み
- `simplify_descending_adjacent_dyad_residue` — 下降隣接 dyad の単純化
- `suppress_descending_upper_singleton_spikes` — 下降ランのスパイク削除
- `suppress_short_descending_return_singletons` — 上位 octave キャリーオーバー抑制
- `suppress_descending_upper_return_overlap` — 下降リターンの dyad 単純化
- `suppress_post_tail_gap_bridge_dyads` — post-tail gap bridge 削除

**中（周波数パターン認識）— 2関数:**
- `simplify_short_secondary_bleed` — dyad bleed の単純化（cents 範囲 + score ratio）
- `suppress_bridging_octave_pairs` — octave pair bridge 削除（harmonic relation 判定）

**高（fixture-specific パターン依存）— 3関数:**
- `suppress_resonant_carryover` — 繰り返し周波数 + harmonic relation + phrase reset 検出。高域 restart-tail 特化ロジックあり
- `suppress_descending_terminal_residual_cluster` — tuning rank ベースの terminal suffix 構築
- `suppress_descending_restart_residual_cluster` — tuning rank ベースの restart 2-note cluster 検出

### 既知の fixture-specific debt
- `collapse_restart_tail_subset_into_following_chord` (events.py) — Stage 7 collapse 系
- `suppress_recent_upper_echo_mixed_clusters` (patterns.py) — Stage 6
- `lower-mixed-roll-extension` (peaks.py) — Stage 3
- ~~`collect_two_onset_terminal_tail_segments`~~ — #122 で削除済み
- `recent-upper-octave-alias-primary` promotion — Stage 3

### 潜在的 debt 候補（今回新規特定）
- `suppress_resonant_carryover` の `keep_high_register_repeated_lower_restart` ケース
- `suppress_descending_terminal_residual_cluster` の tuning rank suffix ロジック
- `suppress_descending_restart_residual_cluster` の restart-specific gap limit (0.8-1.5s)

詳細は [recognizer-local-rules.md](recognizer-local-rules.md) を参照。

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

**評価: Ready（merge 系） / Needs Work（collapse/suppress 系）**

**最終更新: 2026-04-06 (Stage 7 棚卸し)**

27 個の suppress/collapse/merge/split 関数が line 191-229 で逐次適用される。`merge_adjacent_events()` が3回挟まれ、各 collapse の結果を吸収する構造。

### Merge 系（4関数）— Ready

| 関数 | 判定条件 | Free Performance 依存 |
|------|---------|----------------------|
| `merge_adjacent_events()` | 同一 notes + gap ≤ 120ms | なし |
| `merge_short_chord_clusters()` | singleton+dyad → triad, gap ≤ 80ms, 連続キー | なし |
| `merge_short_gliss_clusters()` | 2-3音 gliss, gap ≤ 60ms, 連続キー | なし |
| `merge_four_note_gliss_clusters()` | 4音 gliss, gap ≤ 60ms, 連続キー | なし |

時間閾値 + ノート構造（連続キー、ノート数）のみで判定。楽譜知識を使わない。

### Collapse 系（6関数）— Mostly Ready ～ Needs Work

| 関数 | 依存度 | 懸念 |
|------|--------|------|
| `collapse_same_start_primary_singletons()` | 中 | phrase_reset_lower が周波数パターンに依存 |
| `collapse_high_register_adjacent_bridge_dyads()` | 中 | octave ≥ 6 の楽器レジスター制限 |
| `collapse_restart_tail_subset_into_following_chord()` | 中 | `is_adjacent_tuning_step()` による tuning チェック |
| `collapse_late_descending_step_handoffs()` | 高 | cents_distance による音程パターン認識 |
| `collapse_ascending_restart_lower_residue_singletons()` | 高 | 再開パターン + tuning step 認識 |
| `split_adjacent_step_dyads_in_ascending_runs()` | 高 | 上昇ラン構造の認識 |

高依存の3関数は recognizer-local-rules.md で fixture-specific debt として既に記録されている。

### Suppress 系（17関数）— 個別評価未実施

line 191-215 の suppress/simplify 系関数群。大半は時間 + 周波数比較ベースだが、一部に fixture-specific なパターン認識が含まれる可能性がある。棚卸しは Stage 5 評価と合わせて実施が効率的。

### per-note onset Pass 3 (post-merge) との関係

`merge_adjacent_events()` の条件「同一 notes + gap ≤ 120ms」は、per-note onset splitting で生じた誤分割（異なる notes の sub-segments）を吸収しない。`merge_short_chord_clusters()` は部分的にカバーするが連続キー条件がある。既存 merge だけでは Pass 2 の誤分割吸収は不十分な可能性が高く、Pass 3 の設計検討が必要。

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
