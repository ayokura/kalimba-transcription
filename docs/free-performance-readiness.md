# Free Performance Readiness Assessment

## Purpose

各 recognizer コンポーネントについて、Free Performance（楽譜知識なし・Expected Performance なしの自由演奏転写）への適合度を評価する。チケット処理のたびに関連コンポーネントを再評価し、このドキュメントを更新する。

**最終更新: 2026-04-09 (#153 Phase B (`36cb3de`) + #154 noise floor calibration + 17-key BWV147 sequence-163 promoted to completed (`400852a`) + #162 heuristic constants audit 起票 + 34-key BWV147 sequence-163 E83 fix (`35bca12`) → 34-key も completed promote (`7535464`))**

## 評価軸と凡例

各 stage を **3 軸独立**で評価する。1 つの stage が「Free Performance では問題ないが Streaming では再設計必要」のような nuance を表現できるようにするため。

### 3 軸の意味

- **Free Performance**: 楽譜知識 / Expected Performance に依存せず動作するか
- **Streaming / Causal**: 全音声を入力に持たなくても (リアルタイム入力で先頭から逐次処理して) 動作するか
- **Browser-side**: librosa / numpy 等のサーバ依存を取り除いて WebAudio / WebAssembly で動作するか

### レベル

| 記号 | レベル | 意味 |
|---|---|---|
| ✅ | **Ready** | そのまま使える |
| 🟡 | **Adaptable** | 軽微な調整で対応可能 |
| ⚠️ | **Needs Work** | 設計的な工夫が必要 |
| ❌ | **Blocking** | 根本的な再設計が必要 |
| — | **TBD** | 未評価 |

## サマリー (2026-04-09)

| Stage | コンポーネント | Free Perf | Streaming | Browser |
|---|---|---|---|---|
| 1 | Audio Input (`audio.py`) | — | — | — |
| 2 | Onset & Segment Detection (`segments.py`) | 🟡 | ⚠️ | ⚠️ |
| **2.5** | **Per-Recording Calibration (`noise_floor.py`)** ← **新設 #154** | ✅ | 🟡 | ✅ |
| 3 | Per-Segment Peak Detection (`peaks.py`) | 🟡 | ✅ | 🟡 |
| 4 | Raw Event Aggregation (`pipeline.py`) | — | — | — |
| 5 | Event Post-Processing (`events.py` suppress/simplify) | 🟡 | ⚠️ | ✅ |
| 6 | Pattern Recognition (`patterns.py`) | ⚠️ | ⚠️ | ✅ |
| 7 | Final Merging & Adjacency (`events.py` merge/collapse) | 🟡 | ⚠️ | ✅ |
| 8 | Quantization & Notation (`notation.py`) | ✅ | ✅ | ✅ |
| 9 | Output Assembly (`pipeline.py`) | ✅ | ✅ | ✅ |

各 stage の詳細評価は以下のセクション参照。

---

## Stage 1: Audio Input (`audio.py`)

**評価: Free Perf — / Streaming — / Browser —**

- `read_audio()`: モノラル変換、無音チェック
- Free Performance 固有の懸念は少ないと思われるが未評価
- Streaming / Browser についても実装を確認する必要あり (UploadFile 経由で全音声受領前提なので、streaming では別エントリポイントが必要だろう)

---

## Stage 2: Onset & Segment Detection (`segments.py`)

**評価: Free Perf 🟡 / Streaming ⚠️ / Browser ⚠️**

**最終更新: 2026-04-09 (#153 Phase B で broadband onset → segment 化 structural gap を downstream で補完する設計に到達)**

### 3 軸サマリー
- **Free Perf 🟡**: active range / onset 検出は楽譜非依存。ただし polyphonic onset 限界・masked re-attack・broadband onset → segment 化の structural gap が残る (Stage 3 / 5 で補完)
- **Streaming ⚠️**: librosa.onset.onset_strength は batch 処理。incremental spectral flux + peak picking の再実装が必要
- **Browser ⚠️**: librosa 依存。WebAudio AnalyserNode で代替可能だが精度差の検証が必要

### 良い点

- **Active range 検出**: RMS ベースで楽譜非依存。threshold = max(0.18*max_rms, 2.2*median_rms) は adaptive で楽器特性に追従
- **Attack profile validation**: broadband_gain + high_band_flux の組み合わせ判定。moderate gain-flux gate (gain≥3.0, flux≥0.8) の追加で 34-key の genuine attack 2件を救済済み。閾値は物理量ベースで fixture-specific でない
- **Gap collector (AVC)**: onset の attack profile で判定しており因果的・楽譜非依存
- **librosa onset_strength + onset_detect**: 標準的な broadband spectral flux。大半の onset を正しく検出（17-key 161/163, 34-key 159/163）
- **Per-note onset detection (Pass 1)**: gap mute-dip rescue 実装済み (#144)。broadband onset が見逃した same-note re-attack を per-note の mute-dip パターンで検出し、`confirmed_primary` 付き segment を生成。compact-window アルゴリズムで自然減衰の false positive を排除。楽譜非依存・因果的・WebAssembly 互換
- **Sub-onset aware per-note attack window** (#152, commit e449df8): segment 内に複数の broadband sub-onset がある場合、対象 note の actual attack を `pick_matching_sub_onset` で picked し、`onset_energy_gain` の窓をその時刻に anchor する。slide chord での staggered attack や、segment 開始よりやや遅い primary attack を救済。
- **Short-segment secondary guard** (commit 1f3bda4): segment duration < 30ms (典型は gap-mute-dip rescue の 6.7ms) で secondary 全部を strip。FFT 窓が segment 幅未満になり secondary が信頼できないため。`from_short_segment_guard` フラグで下流から識別可能、`shortSegmentGuardActive` debug field 経由で trace 可能。`suppress_short_residual_tails` は guarded primary を carryover と誤認しないよう exempt。

### 懸念点

- **SR 依存性 (#140)**: FRAME_LENGTH=2048, HOP_LENGTH=256 が固定サンプル数。sr=44100 で STFT窓=46.4ms、sr=96000 で 21.3ms。onset 検出の時間分解能が楽器/録音環境で異なる。リサンプル実験で -5 exact 回帰、n_fft 変更で -4 exact — チューニング再調整なしの単独投入は不可
- **Polyphonic onset の限界**: 単音 attack が他の音の残響に埋もれると onset_strength に現れない（E162 B4: onset_strength=0.6 vs background 0.3-0.5）。per-note onset detection の Pass 2 (onset splitting) で補完予定 (#145)
- ~~**Masked re-attack**~~ → **#153 Phase A.4 で解決済** (`ed729bb`)。`recover_masked_reattack_via_narrow_fft` が `pick_matching_sub_onset` + 4 disambiguators (energy/fr/dominance/sub_onsets≥3) で per-attack window narrow FFT して救出。E97/E133 の D5 が rescue 済
- **Broadband onset → segment 化の structural gap**: broadband onset detector (librosa) は実際には早期 attack を検出している (例: 17-key R1 E97 で broadband が 168.0827s に onset を出している) が、segmenter (active range / collector logic) がその onset を「segment の始点」として消費せず捨てるケースがある。結果として attack window 全体がどの segment にも含まれない → segment_peaks がそもそも fresh attack を見られない → carryover 扱いで棄却。**#153 Phase B (`recover_pre_segment_attack_via_narrow_fft` `36cb3de`) が downstream で補完**: event 開始前の lookback 範囲 (200ms) で「未消費 broadband onset」を見つけて narrow FFT する。これは workaround pass であり、本質的には Stage 2 の onset → segment 化ロジックを強化する余地がある (将来の課題)
- **Streaming 再設計**: librosa.onset.onset_strength は batch 処理。streaming 化には incremental spectral flux + peak picking の再実装が必要。per-note onset の部品（`_note_band_energy()` 等）は因果的で streaming 互換
- **HPSS onset 分離**: 試験の結果、percussive 単独置換は回帰 (76%→53%)。カリンバの撥弦 attack が harmonic/percussive に分散し、percussive のノイズフロア上昇で偽 onset 増加。パイプライン全体チューンが必要 (#148)

---

## Stage 2.5: Per-Recording Calibration (`noise_floor.py`)

**評価: Free Perf ✅ / Streaming 🟡 / Browser ✅**

**最終更新: 2026-04-09 (#154 で新設、#153 Phase B narrow-FFT pass の閾値基盤として使用)**

### 位置付け

Stage 2 (segment 検出) が完了した直後、Stage 3 (segment 内の peak detection) が始まる前に **per-recording・per-band の calibration を 1 回だけ行う**新しいミニ stage。pipeline.py 上は `rescue_gap_mute_dips` (per-note Pass 1) の直後、`segment_peaks` ループの直前に `measure_noise_floor` を呼ぶ。

```python
# pipeline.py
segments = rescue_gap_mute_dips(segments, audio, sample_rate, tuning)
noise_floor = measure_noise_floor(audio, sample_rate, tuning, segments)  # ← Stage 2.5
for segment in segments:
    candidates, ... = segment_peaks(...)  # Stage 3
```

将来的に「per-band 平均 attack profile」「tuning verification」等の他の per-recording calibration が必要になればここに追加する。

### 良い点

- **完全に楽譜非依存**: 入力は audio + tuning + segments のみ。score / Expected Performance に一切触れない
- **Silent region detection** が segment-gap 駆動 (`segment[i].end → segment[i+1].start` の隙間 ≥ 100ms) なので、楽譜的なヒントを必要としない
- **同じ narrow FFT 設定 (`NARROW_FFT_WINDOW_SECONDS = 30ms`)** で計測するので、Stage 3 / Stage 5 の merge passes が比較する `fundamental_energy` と単位が完全に一致する。calibration ↔ 消費側の単位ずれによる凡ミスがない
- **Median 集約** で transient artifact (silent region に紛れ込んだクリック等) に強い
- **Hard floor 機構**: `NoiseFloorMeasurement.threshold_for(note, *, factor, fallback, hard_floor)` ヘルパーで「pathologically low な calibration 値が出ても下限を下回らない」safety net を持つ

### 3 軸の根拠

- **Free Perf ✅**: 楽譜非依存。recording のみで自己完結
- **Streaming 🟡**: 現状の実装は **全 segment が出揃ってから silent region を集める**ので strict な streaming にはならない。ただし設計上は **leading silent region (recording 冒頭の最初の attack より前)** だけで calibration 可能 (median は十分な sample 数があれば安定するため)。streaming 化は「最初の N ms を leading silence と仮定して calibrate → その後固定値を使う」or「running median を維持」の選択。設計の方向性は明確で大規模再設計は不要
- **Browser ✅**: librosa 依存なし。`peaks._adaptive_n_fft` (numpy.fft.rfft + 単純な算術) と `peaks.batch_peak_energies` (numpy ベース) のみを再利用。WebAssembly 移植時に numpy → JS 配列演算 + WebAudio AnalyserNode の組み合わせで等価実装可能

### 設計知見 (Phase B 経由で発見)

1. **eval_scope vs full audio で noise_floor 値が ~1.5x 異なる**: 17-key BWV147 で `noise_floor[G4]` が full audio 0.116 / eval scope 0.181。テスト infra の splice/zerofill が silent region の取り方を変えるため。**実用上の含意**: noise_floor multiplier に強く依存する threshold 設計は脆い → 本 calibration は「**下限ガード**」として使い、真の discriminator は時間ローカルな測定 (`backward_attack_gain` 等) に置く設計が安定する
2. **`measure_narrow_fft_note_scores` (peaks.py 既存 API) は silent region では `None` を返す**: `rank_tuning_candidates` の rank-1 score が 1e-6 未満で弾かれる仕様。calibration では使えないため `_narrow_fft_band_energies` で `batch_peak_energies` を直接呼ぶ別ルートで実装

### 懸念点

- **計測値の安定性が silent region の総量に依存**: 録音が密に演奏で埋まっていて silent gap が短い fixture では `_select_samples` が `min_silent_window_seconds` 以上の slice を確保できず `NoiseFloorMeasurement.is_empty == True` になる。この場合 fallback (Phase A 固定閾値) に戻る。実用上問題が出るのは「ほぼ無音区間がない極端な録音」のみだが、streaming 化時に「leading silence 不足」の handling が必要
- **Tuning 全 note を均等に sample しているとは限らない**: silent region の周波数特性は録音環境のノイズ特性 (HVAC、マイク self-noise 等) に依存する。音響的に高音域が空くマイクで録音すると `noise_floor[C6]` が極小になり threshold が hard floor まで貼り付く。当面は hard floor で対処、将来的には「一定値以下は集約から除外」等の robustness 強化が候補

### 関連

- **#154** Per-recording per-band noise floor measurement (2026-04-09 完了)
- **#162** narrow-FFT-pass heuristic constants audit — `noise_floor` を使った threshold 置換が「Class B (環境依存だが正規化済)」の代表例

---

## Stage 3: Per-Segment Peak Detection & Candidate Selection (`peaks.py`)

**評価: Free Perf 🟡 / Streaming ✅ / Browser 🟡**

**最終更新: 2026-04-09 (#153 Phase A + B 完了 + 34-key E83 fix `35bca12` — narrow FFT 系 pass 経由で octave-coincident chord aliasing と masked re-attack の主要 case を解決、`residual-forward-scan` を segment classifier として再利用して 演奏者/テンポ違いの carryover にも対応)**

### 3 軸サマリー
- **Free Perf 🟡**: ranking は楽譜非依存で堅牢。Phase A.2/A.4 の merge / rescue passes + E83 fix で残存課題が大幅に解消。残るのは secondary/tertiary 選択経路の複雑性と #111 chord selector、および「演奏者/テンポ違いに対する robustness」の汎化検証 (現状は `residual-forward-scan` 発火 segment に限定)
- **Streaming ✅**: per-segment 独立評価で cross-segment 依存なし。`_resolve_primary` が segment 内で完結。E83 fix の `residual-forward-scan-replaced-primary` / `forced_evidence_gates` も segment 内で完結 (cross-segment 依存なし)
- **Browser 🟡**: numpy.fft + tuning ベースの scoring は portable だが、`rank_tuning_candidates` / `segment_peaks` の Python 実装が大規模 (2700 行+)。WebAssembly 移植は大仕事だが、algorithmic logic は数値演算のみで原理的に portable

### 良い点

- **Candidate ranking (`rank_tuning_candidates`)** は楽譜非依存。FFT + harmonic scoring でチューニングノートをスコアリングする純粋にスペクトルベースの処理
- **`is_physically_playable_chord`** はチューニング定義のみに依存し、楽譜知識を使わない
- **Octave dyad判定 (`allow_octave_secondary`)** は fundamental_ratio ベースで汎用的。今回の閾値緩和 (0.85→0.75) で octave-4 ノートの過度な棄却が解消。non-octave-4 の閾値 (0.32) と比較して 0.75 はまだ保守的
- **Evidence gate (`onset_gain`, `backward_attack_gain`)** はセグメント内の物理的な attack 特性に基づく判定であり、楽譜に依存しない
- **Fast mute-dip 検出**: 50ms + 30ms の2段階 FFT 窓で ~20ms の高速 mute-dip も検出 (コミット e599d4b)。E83 G4 型（segment 内で residual-decay 棄却されていた genuine re-attack）を rescue
- **Segment provenance**: `confirmed_primary` / `hint_primary` フィールド (#143) で per-note パスから peaks 層への情報伝達基盤を確立。`confirmed_primary` 付き segment は `_resolve_primary` をスキップし residual-decay 免除
- **Sub-onset aware per-note attack** (#152, commit e449df8): `pick_matching_sub_onset` + `onset_energy_gain` 改修。早い attack を持つ candidate に対し正しい sub-onset を picked し、early window のみ anchor。`recent_note_names` 制限で false positive 防止。Lower-octave harmonic-related-to-selected bypass で octave alias 誤棄却を回避。tertiary-weak-onset の対称化で carryover 経路の不整合解消。これにより E148 C5 (delayed within-segment attack) と C4 (octave alias 誤棄却) を救済
- **Short-segment secondary guard** (commit 1f3bda4): 30ms 未満 segment では secondary 全 strip + `RawEvent.from_short_segment_guard` フラグ + `shortSegmentGuardActive` debug field。`GATE_CATEGORIES` に新カテゴリ `"guard"` 追加。`_CandidateDecision.reasons` に `short-segment-secondary-guarded` で trail 経由 traceable。下流 `suppress_short_residual_tails` は guarded を exempt。これにより E148 で gap-mute-dip 由来の 6.7ms segment が A5+G5 noise を拾わず C6 のみ保持
- **`residual-forward-scan` を segment classifier として再利用** (`35bca12`): `residual-forward-scan` は元々 primary promotion 経路の 1 つだったが、これが発火する = 「初期 primary が recent note で residual decay を示し mute-dip reattack なし、かつ別の recent note が genuine reattack を持つ」という強い物理的判定。この情報を以下 2 箇所で再利用:
  1. **`residual-forward-scan-replaced-primary` (Phase A structural gate)**: 置換された元 primary note を secondary candidates からも除外。内部 sustain 判定との整合性を保つ
  2. **`forced_evidence_gates` (Phase B 上の secondary slot に tertiary 並み gate を強制適用)**: 1 だけだと「強い carryover を消すと弱い carryover が tertiary slot から secondary slot に昇格して同じ tertiary gate を回避する」モグラ叩き構造になる。`residual-forward-scan` 発火 segment は carryover-prone なので secondary slot にも `tertiary-weak-onset` / `tertiary-weak-backward-attack` を強制適用
  
  これにより 34-key BWV147 sequence-163 R1 E83 が解決 (162/163 → 163/163)。同じ score でも 17-key (E82→E83 1.93s) と 34-key (0.48s, 約 4 倍速) で carryover 量が異なる「演奏者/テンポ違い」に対する robustness を獲得。設計哲学は Phase B `recover_pre_segment_attack_via_narrow_fft` の bg-ordered iteration と一致 — 強い fresh-attack signature が見つかった時点で、それより前の sustain candidate は再評価して落とす。`primary_promotion_debug` field (名前に反して debug request flag から独立した制御 field) を gate 条件に使うパターンは line 2027-2034 の `recent-upper-octave-alias-secondary-blocked` と同じ既存パターンの拡張

### 懸念点

- **Rescue path の複雑性**: `_evidence_rescue_gate` が 5 層の条件分岐に成長。Phase B の gate 設計意図との関係が不透明。gate 調整のたびに分岐が増える構造的問題 (#111 に記録済み)
- **Sequential accept loop**: primary → secondary → tertiary の逐次選択は、候補の評価順序に結果が依存する。E136 の問題（E4 が先に棄却されたために C4 が playability チェックに失敗）はこの構造的制約の典型例。chord selector (#111) で根本解決の可能性あり
- **Octave-4 fr 閾値 (0.75)**: 今回の緩和は BWV147 の2件で検証済みだが、Free Performance での広範な文脈で偽 octave dyad を生むリスクは broader fixture coverage がないと評価困難
- **Tertiary rescue bypass (og >= 2.0)**: carryover rescue と同じ閾値の流用であり、tertiary rescue に最適な閾値かは理論的根拠が弱い
- ~~**Ranked candidate 不在問題 / Octave-coincident chord aliasing**~~ → **#153 Phase A.2 で解決済** (`933d088`)。`merge_short_segment_guard_via_narrow_fft` が短い guarded singleton (例: E148 の C6 6.7ms gap-mute-dip 由来 segment) を後続 segment に narrow FFT cross-validation で rejoin する。E148 の他 E121/E127 prefix splitting / E97/E133 D5+F5 / E100 C4 / E97 G4 も Phase A.3/A.4 + Phase B で解決済 (詳細は recognition-roadmap.md の "解決済" 節)
- ~~**Masked re-attack threshold**~~ → **#153 Phase A.4 で解決済** (`ed729bb`)。詳細は Stage 2 の同項目参照
- **Pre-segment attack の structural gap** → **#153 Phase B で workaround 解決** (`36cb3de`): events.py 側に新 pass `recover_pre_segment_attack_via_narrow_fft` を追加。本質的には Stage 2 で broadband onset を全部 segment 化すべきだが、その再設計は大規模なので Stage 5/7 layer での補完を採用 (詳細は Stage 2 の structural gap 節 + Stage 5 参照)
- **Heuristic constants の環境依存性** → **#162 で audit 中**。Phase A + B で 27 個の新定数を追加した経験から、Class C (環境依存で未正規化) に該当する定数の data-driven 化候補を抽出する作業を起票済
- **演奏者/テンポ違いに対する robustness — 汎化検証が必要**: 34-key E83 fix (`35bca12`) で、同じ score でも演奏速度が 4 倍違うと carryover の量が大きく変わり、recognizer の secondary slot の挙動が変わることが確認された (17-key の 1.93s 間隔は通り、34-key の 0.48s 間隔は通らなかった)。今回の fix は **`residual-forward-scan` が発火した segment** に対象を絞った安全な対応だが、より広い carryover-prone segment に対しては未対応。次に新しい failure case (異なる promotion path / 通常 segment での carryover) が出てきた段階で、より一般的な segment classifier (e.g. `broadbandOnsetGain < threshold`) ベースの forced evidence gates 適用を検討する。**Free Performance では演奏者/楽器/テンポの組み合わせが事前に分からない**ため、この robustness 汎化は本質的な課題
- **tertiary / secondary 区別の本質的な是非** (将来検討): E83 fix の経験から、carryover-prone segment では「secondary slot か tertiary slot か」という区別自体が弱く、「全 candidate に物理的 attack evidence を要求すべき」という方向の設計が示唆される。現状は `forced_evidence_gates` という限定的トリガーで slot 区別を維持しているが、将来的には tertiary 固有の gate を secondary normal path にも常時適用する大規模リファクタが選択肢として残る (E83 解決時にユーザーから提示された方向)。今回は限定対応で 162/163 → 163/163 達成したため見送り

---

## Stage 4: Raw Event Aggregation (`pipeline.py`)

**評価: Free Perf — / Streaming — / Browser —**

- `recent_note_names`, `ascending_run_ceiling` 等の文脈状態を構築
- sparse gap tail filtering にレジスタ・下降パターンのヒューリスティクスあり

### 未評価項目
- 文脈状態の構築が Expected Performance に依存しないか
- sparse gap tail filtering の汎用性
- 文脈状態を「過去 N events のみ」に制限すれば streaming OK か

---

## Stage 5: Event Post-Processing (`events.py`)

**評価: Free Perf 🟡 / Streaming ⚠️ / Browser ✅**

**最終更新: 2026-04-09 (#153 Phase A + B で narrow-FFT 系 4 passes 追加。`recover_pre_segment_attack_via_narrow_fft` は本 stage に属する)**

### 3 軸サマリー
- **Free Perf 🟡**: 大半の suppress/simplify は時間 + ノート構造のみで汎用的。fixture-specific debt 3 関数 + Phase A/B で追加した 4 narrow-FFT 系 passes はいずれも楽譜非依存だが、heuristic constants の環境依存性 (#162) が懸念
- **Streaming ⚠️**: event post-processing は前後の event 文脈に依存するため strict streaming にはならない。「lookahead/lookbehind window を有限にする」工夫でほぼ causal 化可能だが本格的な再設計が必要
- **Browser ✅**: pure list operations のみ (audio 処理は narrow-FFT 系 4 passes が唯一だが、これも numpy/portable)

### #153 Phase A + B で本 stage に追加した passes (新規 4 passes)

| Pass | Phase | 役割 |
|---|---|---|
| `merge_short_segment_guard_via_narrow_fft` | A.2 (`933d088`) | 短い (gap-mute-dip 由来 6.7ms 等) guarded primary singleton を後続 segment に narrow FFT cross-validation で rejoin (E148 C6 救出) |
| `merge_gliss_split_segments` | A.3 (`2fd433e`) | gliss prefix splitting / late-note splitting を union + semitone dedup で統合 (E121/E127 prefix, E97/E133 F5 trailing 救出) |
| `recover_masked_reattack_via_narrow_fft` | A.4 (`ed729bb`) | 同一 note の carryover decay 上の masked re-attack を `pick_matching_sub_onset` + 4 disambiguators で救出 (E97/E133 D5 救出) |
| `recover_pre_segment_attack_via_narrow_fft` | B (`36cb3de`) | broadband onset → segment 化の structural gap を埋める。event 直前 lookback の未消費 onset で narrow FFT、bg-ordered iteration + bg dominance ratio で sympathetic resonance を排除 (E97 G4 / E100 C4 救出) |

これら 4 passes はすべて Stage 2.5 (`noise_floor`) を消費して per-band 閾値を持つ。設計上の依存関係は **Stage 2 → Stage 2.5 → Stage 3 → Stage 5 (本 passes)** の順で計算される。

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
- `lower-mixed-roll-extension` (peaks.py) — Stage 3
- ~~`collect_two_onset_terminal_tail_segments`~~ — #122 で削除済み
- ~~`suppress_recent_upper_echo_mixed_clusters` (patterns.py) — Stage 6~~ — 2026-04-09 削除済み (G ablation)
- `recent-upper-octave-alias-primary` promotion — Stage 3

### 潜在的 debt 候補（今回新規特定）
- `suppress_resonant_carryover` の `keep_high_register_repeated_lower_restart` ケース
- `suppress_descending_terminal_residual_cluster` の tuning rank suffix ロジック
- `suppress_descending_restart_residual_cluster` の restart-specific gap limit (0.8-1.5s)

詳細は [recognizer-local-rules.md](recognizer-local-rules.md) を参照。

---

## Stage 6: Pattern Recognition (`patterns.py`)

**評価: Free Perf ⚠️ / Streaming ⚠️ / Browser ✅**

### 3 軸サマリー
- **Free Perf ⚠️**: corpus-wide な dominant pattern 書き換えに依存。Free Performance ではパターンの事前知識がない。AGENTS.md に「Treat repeated-pattern normalizers as suspicious until proven necessary」と記載
- **Streaming ⚠️**: multi-event pattern detection は lookahead が必要。strict streaming には再設計が必要
- **Browser ✅**: pure logic のみ

- `apply_repeated_pattern_passes()`: repeated four-note, triad, gliss パターンの正規化

### 未評価項目
- ablation で各パスの影響度がどの程度か
- Free Performance で repeated pattern pass を全て無効化した場合の影響

---

## Stage 7: Final Merging & Adjacency (`events.py`)

**評価: Free Perf 🟡 / Streaming ⚠️ / Browser ✅**

**最終更新: 2026-04-06 (Stage 7 棚卸し)**

### 3 軸サマリー
- **Free Perf 🟡**: merge 系 4 関数は時間 + ノート構造ベースで Ready。collapse 系 6 関数のうち 3 関数が高依存 (cents/tuning step pattern)
- **Streaming ⚠️**: cross-event merging は lookahead が必要。Stage 5 と同じ課題
- **Browser ✅**: pure list operations のみ

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

**評価: Free Perf ✅ / Streaming ✅ / Browser ✅**

- beat-quantized representation へのマッピング
- 入力はイベントの timing と note set のみ
- 楽譜知識不要、Free Performance でそのまま使用可能
- pure 数値処理 → streaming / browser ともに自然に対応

---

## Stage 9: Output Assembly (`pipeline.py`)

**評価: Free Perf ✅ / Streaming ✅ / Browser ✅**

- `ScoreEvent` 構築と `TranscriptionResult` パッケージング
- 楽譜知識不要、pure データ構造変換

---

## Cross-cutting concerns

### Streaming / Causal 化

ボトルネック順:

1. **Stage 2 onset detection**: `librosa.onset.onset_strength` が batch 処理。incremental spectral flux + peak picking の再実装が必須 (#140 参照)
2. **Stage 5/7 event post-processing**: 27 個の suppress/merge 関数が前後 event の文脈に依存。lookahead/lookbehind window を有限化する設計改修が必要 (cf. patterns.py の `apply_repeated_pattern_passes` も同類)
3. **Stage 2.5 noise_floor**: 現状は全 segment 出揃い後に silent gap を集めるが、設計上は **leading silent region (録音冒頭の最初の attack より前)** だけで calibration 可能。streaming 版の方針は明確 (大規模再設計不要)
4. **Stage 3 peaks.py / Stage 8 / Stage 9**: それぞれ自然に streaming 互換 (per-segment 独立 / pure 数値変換)

### Browser-side 実装

- **librosa への依存** が segments.py (Stage 2 onset detection) に集中。それ以外の stage は numpy + 標準 Python のみ
- **noise_floor.py (Stage 2.5)** は意図的に librosa を使わず `peaks._adaptive_n_fft` (numpy.fft) + `peaks.batch_peak_energies` (numpy 算術) のみで実装 → WebAssembly 移植は直接的
- **peaks.py の scoring logic (Stage 3)** は数値演算のみで原理的に portable だが、Python 実装が大規模 (2700 行+) なので移植コストは大きい
- **events.py の suppress/merge (Stage 5/7)** は pure list operations で、WebAssembly 不要 (素の JS で動く)
- **patterns.py (Stage 6)** も同様に pure logic
- segments.py の RMS/onset detection は WebAudio API の `AnalyserNode` で代替可能だが、librosa との数値一致は別途検証が必要

---

## 更新履歴

| 日付 | 契機 | 更新内容 |
|------|------|----------|
| 2026-04-06 | #126 gate調整 | 初版作成。Stage 3 (peaks.py) を詳細評価。octave dyad 閾値緩和 + rescue bypass の Free Performance 影響を記載 |
| 2026-04-06 | 34-key NO MATCH調査 | Stage 2 (segments.py) 評価追加。SR依存性 (#140)、polyphonic onset 限界、per-note onset 設計 (#141) を記載 |
| 2026-04-06 | Stage 5/7 棚卸し | Stage 5 suppress/simplify 系 19関数の個別評価（依存度別4段階分類）。Stage 7 merge 4関数 + collapse 6関数の評価追加。潜在 debt 3件特定 |
| 2026-04-07 | #142-144, E83分析 | Stage 2: per-note onset Pass 1 (gap mute-dip rescue) + HPSS 試験結果追加。Stage 3: fast mute-dip 30ms 窓フォールバック + Segment provenance 追加 |
| 2026-04-08 | #152 完了 + commit 1f3bda4 + #153 起票 | Stage 2: #152 sub-onset aware per-note attack window + commit 1f3bda4 short-segment secondary guard 追加。Masked re-attack 課題 (#153 で扱う) 追記。Stage 3: #152 の3拡張 + commit 1f3bda4 のフラグ/marking 機構を「良い点」に追加。**Octave-coincident chord aliasing** を独立課題として「懸念点」に明示 (C5/C6 周波数衝突の物理的制約、#125/#153)。Masked re-attack threshold 課題追記。残り failure 6/7 が #153 一本で解決可能と確定 |
| 2026-04-09 | #153 Phase B + #154 + 17-key promote + #162 起票 | **3 軸評価 (Free Perf / Streaming / Browser-side) に再設計** + サマリー表追加。**Stage 2.5 (Per-Recording Calibration `noise_floor.py`) を新設** (#154)。Stage 2: masked re-attack を Phase A.4 で解決済としてマーク、broadband onset → segment 化 structural gap を新規課題として明示 (Phase B `recover_pre_segment_attack_via_narrow_fft` で workaround 解決)。Stage 3: octave-coincident aliasing / masked re-attack を解決済としてマーク。Stage 5: Phase A + B で追加した narrow-FFT 系 4 passes (`merge_short_segment_guard`, `merge_gliss_split`, `recover_masked_reattack`, `recover_pre_segment_attack`) のテーブル追加。Stage 6/7/8/9 を 3 軸表記に統一。Cross-cutting concerns を Streaming/Browser それぞれに対する stage 別ボトルネック分析に書き直し |
| 2026-04-09 | 34-key E83 fix + 34-key promote (`35bca12`, `7535464`) | Stage 3: `residual-forward-scan` を segment classifier として再利用する 2 段構成 fix (`residual-forward-scan-replaced-primary` + `forced_evidence_gates`) を「良い点」に追加。これにより 34-key BWV147 sequence-163 が 162/163 → 163/163 完全達成し、両 BWV147 sequence-163 fixture (17-key と 34-key) が completed regression target に。「懸念点」に新規課題 2 件追加: (1) 演奏者/テンポ違いに対する robustness の汎化検証 (現状は `residual-forward-scan` 発火 segment 限定)、(2) tertiary/secondary 区別の本質的是非 (将来の大規模リファクタ候補として記録) |
