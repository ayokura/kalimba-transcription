# Deferred Ideas

検討・実装したが不採用になったアイデア、または将来再検討の余地があるアプローチをリストアップする。
各アイデアに不採用理由と再検討条件を記録し、同じ問題に再遭遇したときの参考にする。

## Carryover rejection: backward_attack_gain guard on mute-dip exemption

- **Issue**: #107
- **日付**: 2026-04-01
- **概要**: `_has_mute_dip_reattack()` が True を返す場合に `onset_backward_attack_gain()` で二重検証し、bwg < 2.0 なら reject
- **動機**: mute-dip の false positive (sympathetic excitation) が E17 A4, E129 C5 で発生。backward_attack_gain は genuine attack (>7.9) と carryover (<0.8) を数桁の差で分離可能
- **不採用理由**: mute-dip energy window を 50ms に拡大したところ false positive が解消。根本原因は 15ms 窓の測定不安定であり、窓の修正で backward_attack_gain guard は不要になった
- **再検討条件**: mute-dip window 拡大後も false positive が発生するケースが見つかった場合。特に F3 など低音域で mute-dip と窓サイズが拮抗する場合に有用な可能性がある
- **コミット**: `0587562` (追加) → `0f79e7d` (削除)

## Upper secondary carryover: backward_attack_gain-based elif gate

- **Issue**: #107
- **日付**: 2026-04-01
- **概要**: upper secondary で `primary_onset_gain < 20.0` の場合に、`onset_gain < 2.5 AND mute_dip=False AND backward_attack_gain < 2.0` で reject する代替パス
- **動機**: E28 <B4,G4> に extra C5。upper secondary gate の `primary_onset_gain >= 20.0` 条件が不成立 (B4=16.0) で gate が発火しない
- **不採用理由**: 降順シーケンス (E6→C4) で同パターンの genuine carryover note (post-processing で除去される前提) が elif で reject → 別 candidate が accept → 回帰。segment_peaks レベルでの区別が困難
- **再検討条件**: E28 の extra C5 を解決する別アプローチが見つからない場合。post-processing 側での対応、または `primary_onset_gain` 閾値の引き下げ (20.0→12.0) も候補
- **関連データ**: E28 C5: onset_gain=0.87, backward_attack_gain=0.76, mute_dip=False

## HARMONIC_BAND_CENTS 拡大 (±40→±60-80 cents)

- **Issue**: #101 handoff で言及
- **日付**: 2026-03-31
- **概要**: FFT 帯域幅を広げて n_fft を変えずに低音域の bin 不足を解消
- **動機**: 96kHz + n_fft=4096 で D4 帯域に bin がゼロになる問題
- **不採用理由**: 未検証。隣接半音 (C4-D4 間 ~200 cents) との分離が悪化するトレードオフ。dynamic n_fft (#101) で代替解決
- **再検討条件**: dynamic n_fft のパフォーマンスオーバーヘッドが問題になった場合、または帯域幅拡大が他の品質指標を改善する場合

## Goertzel algorithm for note-band energy

- **Issue**: #101 handoff で言及
- **日付**: 2026-03-31
- **概要**: FFT の代わりに Goertzel アルゴリズムで特定周波数のエネルギーを直接計算
- **動機**: FFT の n_fft / zero-padding / spectral leakage 問題を根本的に回避。ブラウザ移植 (WebAudio) に最適
- **不採用理由**: 変更規模が大きく、現状の dynamic n_fft + 50ms window で十分な安定性が得られている
- **再検討条件**: ブラウザサイド実装フェーズ、またはリアルタイム処理でのパフォーマンス最適化が必要になった場合

## Subband onset detection for narrowband attacks

- **Issue**: #117 (closed — 前提の誤診)
- **日付**: 2026-04-02
- **概要**: mel spectrogram を 8 帯域に分割し、各帯域で独立に onset peak-picking → consensus voting (min_votes=6) で union
- **動機**: E143 A5 / E163 chord が broadband onset detection で検出されないと考えた
- **不採用理由**: 調査の結果、broadband onset は両方とも検出済み。E163 は waveform_stats kurtosis filter (#118)、E143 は octave alias + residual-decay (#119) が原因。subband detection は E163 を偶然的に改善（5ms 近接 onset 追加で AVC collector trigger）したが根本解決ではない
- **実験結果**: 8-band 全バンド → 2949 onset (14x増)、consensus min_votes=6 → 85 novel onset。gap-only filter で回帰なし (305 passed) だが E143 は改善せず
- **再検討条件**: genuine な narrowband onset 未検出が確認された場合（今回は broadband で検出済みだった）

## Short segment extension for octave-alias resolution

- **Issue**: #119
- **日付**: 2026-04-02
- **概要**: 0.5s 未満の短い segment を 0.5s まで延長（次 segment とぶつかる場合を除く）して FFT データ量を増やす
- **動機**: E143 A5 の segment が 141ms で短く、spectral resolution 不足で A4 octave alias が発生
- **不採用理由**: 0.32s / 0.5s いずれに延長しても primary は A4 のまま (residual-decay-no-reattack で棄却)。A4 の倍音構造 (440+880+1320...) が A5 (880+1760...) を必然的に上回るため、segment 長に関わらず octave alias は解消しない。0.5s では 15 regressions。
- **再検討条件**: `trimmed_from` と統合した形で、segment 管理の一般的改善として再設計する場合。octave alias 自体は別のアプローチ（residual-decay での octave 候補チェック等）が必要

## Terminal collector 5種 (closeTerminalOrphan, delayedTerminalOrphan, terminalMultiOnset, twoOnsetTerminalTail, trailingTerminalOrphan)

- **Issue**: #122
- **日付**: 2026-04-02
- **概要**: active range 末尾に向かって onset を拾う 5種の terminal collector。各自固有の定数セット（計21定数）を持つ
- **動機**: active range 末尾の残存 onset を segment 化する
- **不採用理由**: 42 fixture 中 6 でしか使用されず、使用されていた 90% のケースが AVC trailing で既にカバー。全 ablate で回帰ゼロ。非因果的（末尾の確定を待つ必要がある）で streaming 不適
- **削除**: `3809993`, `6077be8` (-228行, 21定数, 5関数, 5 ablation フラグ)
- **再検討条件**: AVC trailing がカバーしない terminal onset パターンが実用 fixture で確認された場合

## Gap collector 3種 (gapInjected, singleOnsetGapHead, postTailGapHead)

- **Issue**: #129
- **日付**: 2026-04-03
- **概要**: gap 内の onset パターンを特定条件で segment 化する 3種の collector（計13定数）
- **動機**: active range 間の gap に存在する音を拾う
- **不採用理由**: completed fixture で使用ゼロ。全 ablate で回帰ゼロ。AVC inter-range が同等のカバーを提供
- **削除**: `bc529cd` (-150行, 13定数, 2関数+1 inline, 3 ablation フラグ)
- **再検討条件**: なし（使用実績ゼロのため、再検討より新設計が妥当）

## leadingOrphan collector

- **Issue**: #130
- **日付**: 2026-04-03
- **概要**: 最初の active range 直前の孤立 onset を segment 化する collector（3定数）
- **動機**: 演奏開始前の 1 ノートを拾う
- **不採用理由**: 全 4 fixture の使用が AVC leading で完全カバーされていることを ablation で確認（全テスト pass）。AVC 自体が因果的（onset の attack profile で判定）であり、leadingOrphan を残す streaming 上の理由もない
- **削除**: `feeb054` (-36行, 3定数, 1関数, 1 ablation フラグ)
- **再検討条件**: AVC leading が mid_performance_start 等の条件で無効化される場面で leading onset が失われる場合

## onset_strength n_fft の SR 適応 (segments.py)

- **日付**: 2026-04-06
- **概要**: `librosa.onset.onset_strength` の n_fft をサンプルレート依存にして STFT 窓の実時間を ~23ms で統一する
- **動機**: 34-key fixture (sr=44100) で STFT 窓 = 46.4ms、17-key fixture (sr=96000) で 21.3ms。34-key E162 B4→E163 chord の ~40ms gap が窓より短く onset 分離不能
- **実験**: n_fft を `min(2048, 2**round(log2(sr * 0.023)))` で計算（sr=44100 → 1024, sr=96000 → 2048）。E162 は改善せず（B4 attack の onset_strength=0.6 がバックグラウンド 0.3-0.5 に埋もれたまま）。一方、全イベントの onset 時刻がシフトし score alignment で -4 exact (156→152) の回帰。テストスイートは pass（34-key fixture が pending のため）
- **不採用理由**: (1) ターゲット問題 (E162) が解決しない — B4 単音 attack の broadband spectral flux が C5 残響 (127K) に対して小さすぎ、窓サイズではなく polyphonic onset detection の限界。(2) onset_strength envelope の変化が全 segment 境界に波及し、downstream の閾値群（peak_pick, active range, gap collector 等）が n_fft=2048 前提でチューニングされているため広範な回帰が発生
- **関連知見**: peaks.py は `_adaptive_n_fft()` で SR 適応済み。profiles.py は時間ベースパラメータ (ONSET_ENERGY_WINDOW_SECONDS) で SR 非依存。segments.py のみ固定サンプル数 (FRAME_LENGTH=2048, HOP_LENGTH=256) を使用しており SR 依存性が残っている。ただし onset_strength の n_fft 単独変更は downstream 連鎖が大きすぎるため、やるなら FRAME_LENGTH / HOP_LENGTH / 閾値群を含む包括的な SR 正規化（または入力リサンプル）が必要
- **再検討条件**: (1) segments.py 全体の SR 正規化設計時。(2) 入力リサンプル（統一 SR）の導入時。(3) per-note onset detection（note-band energy tracking ベースの onset 検出）が実装された場合、polyphonic onset の根本問題が解消されて本変更も有効になる可能性

## Pre-segment rescue: alternative discriminators (decay max ratio / rank-1 only)

- **Issue**: #154 / #153 Phase B (audit umbrella: #162)
- **日付**: 2026-04-09
- **背景**: `recover_pre_segment_attack_via_narrow_fft`（unconsumed broadband onset で lookback narrow FFT して chord 不足ノートを救出する pass）の false-positive 抑制で複数のアプローチを試した。最終的に「bg-ordered iteration + bg-dominance ratio gate (in-event の最大 bg と比較)」に落ち着いた。以下は採用に至らなかった代替案
- **試行コミット**: なし（同一作業セッション内で iterate）

### 案 A: rank-1 (score-by-score) only iteration

- **概要**: narrow FFT score 順で並べ、最初の non-event 候補が全 gate を通過しなければ rescue を諦める（lower-ranked 候補は試さない）
- **動機**: 17-key d4-d5-octave-dyad fixture で sympathetic resonance ノート (B4, G5 等) が複数 rank に並び、iteration で必ず何かが gate を通り抜ける問題を抑える
- **不採用理由**: 34-key BWV147 R2 E100 C4 を rescue 不能にする。E100 では narrow FFT score の rank-1 not-in-event は D#4 (fr=0.702 で fail)、しかし C4 は score rank 5 にある。Score 順ではなく bg 順で並べると C4 が rank-1 (bg=267) になる
- **再検討条件**: bg-ordering でも同様の sympathetic-resonance fixture で false positive が出る場合。または bg ordering の per-note bg 計算コストが性能課題になった場合

### 案 B: decay max ratio bound (`fund_e_onset / fund_e_segment_start ≤ 2.0`)

- **概要**: decay min ratio (rising-into-segment 排除) に加えて、上限も設ける。Real kalimba decay は ~30ms で 50% 以上下がらないという物理前提から、急激な energy drop は sympathetic transient spike と判定
- **動機**: d4-d5 18.1173s の B4 (5.947 → 2.363, ratio 2.52) を catch する
- **不採用理由**: G5 (1.431 → 0.911, ratio 1.57) や他の resonance 候補が 2.0 以下に収まり catch できない。閾値を 1.4 まで絞ると 34-key C4 (1.31) との margin が 7% しかなくなり不安定。bg-dominance ratio が同じ問題を **物理的により直接的な指標** (in-event note の bg と比較) で解けるため、decay 上限は不要になった
- **再検討条件**: bg-dominance ratio が future fixture で発火しないが decay 上限が discriminator として機能する場合
- **注**: decay min ratio (`≥ 0.8`) は採用済み。rising-into-segment 排除（34-key R5 E154 D4 ratio 0.18）は依然必要

## Residual-decay veto (per-tine autopsy) の event 昇格

- **Issue**: #206 / #141 (round 3)
- **日付**: 2026-07-06
- **概要**: `residual-decay-no-reattack` で丸ごと棄却された segment 窓を per-tine tracker の検出核で裁定し、優勢な fire を event 昇格する post-stage veto (`use_pertine_residual_autopsy`、コードは research branch に保存)
- **動機**: #206 の 4 段連鎖 (偽 onset → 再分割 → 一括棄却 → カスケード) の 3-4 段目置換 + forward-scan (recent-note memory 依存 rescue) の退役 = merge 条件 (3) の実証
- **不採用理由**: 2×2 ablation で統合後の正味 GT 効果が 70cc6637 の FP +1 のみ (probe の FN 回収見込みは較正 guard に抑止されるか round-2 rescue/fscan が回収済み)。metamorphic WARN も解消せず (F5 は救うが +0.1s ずれ、D5 系は純減)。docs/research/pertine-round3-ablation.json、設計 doc §6.5
- **再検討条件**: (1) lowpass 級の帯域劣化録音が実運用で増え、時刻ずれ許容の緩い評価 (±0.15s 等) が正当化される場合、(2) segment 形成段の頑健化 (下記) が入って偽 onset 由来の再分割が減り、autopsy の裁定対象が「真に全落ちした窓」だけになった場合
- **コミット**: 755274d (実装) → 933f1d2 (既定 OFF 撤退)

## Lowpass 頑健な segment 形成 (#206 の真因対処)

- **Issue**: #206
- **日付**: 2026-07-06
- **概要**: lowpass 8kHz 級の変換で broadband 偽 onset が segment を再分割し event 時刻が動く。#206 WARN の真因は形成段にあり、post-stage 裁定では時刻ずれとして残ることが round 3 で実証された
- **動機**: metamorphic alarm の lowpass WARN 恒久解消。tracker 特徴 (heterodyne 復調) は lowpass 完全不変 (probe C) なので、per-tine 情報を**形成段の偽 onset 判別**に使う路線は物理的には生きている
- **不採用理由**: 未着手 (アイデア段階)。events.py 新規 pass 禁止の制約下で形成段 (segments.py) への介入形を要設計。research line の去就判断 (S6 exit) 待ち
- **再検討条件**: research line 継続 (選択肢 a) が選ばれた場合の第 3 カウント巡ターゲット候補。または metamorphic 警報がトリガー 1 (patch 衝突) を再検出した場合

## 統合 per-tine complex front-end (振幅+位相の同時利用に evidence 系を一本化)

- **Issue**: #141 (umbrella) / #96 (broadband と per-note 判定の分離) 関連
- **日付**: 2026-07-06 (round-4 merge 条件 3 裁定時のユーザー発案)
- **概要**: 全 tine の複素復調トレース z_t (heterodyne demod: env=|z|, phase=angle(z)) を常設の共有表現とし、現在バラバラの evidence 計算 — mute-dip / onset gain / residual-decay 判定 (windowed FFT 振幅、Rust note_band_energy) と oracle (demod 位相+env) と fscan (recent-note 走査 + FFT 振幅照会) — を全部この単一表現の読み出しに置換する。統合統計量の本命は複素 AR(1) 予測残差 |z_t − ẑ_t|: 振幅崩壊 (mute)・エネルギー再注入 (attack)・位相リセット (fresh pluck) が単一の残差時系列に統合される。第 3 期 bets で「吸収」判定済みの per-band 予測残差 onset (AR(1)) / 位相コヒーレンス声部分離と同じ物理量
- **動機**: round-4 の限界寄与測定で「mute-dip 振幅と oracle 位相は相補的な物理証拠なのに別機構・別サイトで評価している」構造が明確化。tracker (pertine.py) は demod で env+phase を既に同時取得しており、統合の下半分は実装済み。fscan は全 tine トレース常設なら「走査」から「参照」に退化する。demod は causal + Rust 化前提 (streaming/WASM 適合) で、windowed FFT primitive の置換は browser 移植の primitive 統一も兼ねる
- **現時点で不着手の理由**: 実質 per-note onset detection への全面移行 (broadband-vs-per-note-policy の 4 トリガー制で gate)。mute-dip の速い dip (20-50ms、2 窓 fallback が現に必要) を demod の LPF/hop で解像する再設計 + PHASE_BAR/JERK_BAR 級の ROC 再較正が必要で、counted round 相当の投資になる。round 4 時点は evidence サイトを減らすフェーズ (mute-dip OR バックアップ退役) が先
- **再検討条件**: per-note 移行トリガーのいずれかが発火した時 (その時の evidence 層設計の第一候補)。または fscan/mute-dip の残存サイトに新たな較正問題が出て、個別修正よりも表現統一の方が安い状況になった時
