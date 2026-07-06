# 第 3 カウント巡設計: residual-decay 棄却判定の in-stage 置換 (#206 系、merge 条件 3 再挑戦)

- 作成: 2026-07-06 (S6、ユーザー去就判断 (a) を受けて。#203 記録)
- 目的: #141 merge 3 条件のうち未実証の **(3) 既存 suppression pass / gate reason の削減** を、**棄却判定そのものの証拠源置換** (in-stage) で実証する。第 3 巡 (post-stage autopsy、negative で撤退 — `2026-07-pertine-round3-residual-decay-replacement.md` §6.5) の後継
- 巡カウント: 本実装の統合 dual-run が **kill 判定の第 3 カウント巡**。K6 期限 2026-08-04。現在 2 巡連続 PASS (hard-kill は 2 巡連続失敗で発動)
- 相乗り: fscan 退役の再評価 (実装なし — held-out 込みの寄与実測が誤差内なら除去の C2 材料。round-3 実測: +0.002 R)

## 1. 置換対象と置換形

現行 (peaks.py `_resolve_primary`、multi-primary branching 側にも同型):

```
primary が recent_note_names に含まれる
AND is_residual_decay(primary)          # 包絡が滑らかな減衰形
AND NOT has_mute_dip_reattack(primary)  # 指ミュートの窪みが無い
→ segment 丸ごと棄却 (residual-decay-no-reattack)
```

置換後:

```
primary が recent_note_names に含まれる
AND is_residual_decay(primary)
AND NOT tracker_fresh_attack(segment 窓内のいずれかの tine)   # 位相リセット + reinject
→ 棄却 (理由は residual-decay-no-fresh-attack に改名)
```

- **mute-dip 判定の置換**: 位相リセット証拠は mute-dip の上位互換仮説 (probe B: mute-dip が救えなかった FN 21 中 16 に発火)。mute-dip は「指を触れてから弾く」再打鍵しか捕まえられないが、位相リセットは触れずに弾いた再打鍵も、**segment 内の別 tine の fresh attack** (第 3 巡 #206 の F5 型) も捕まえる
- **(2026-07-06 改訂、ユーザー指摘)**: 初版の「oracle が mute-dip を置換」は誤り — 全数 dump が測ったのは**現在落ちている** 33 slot のみで、**mute-dip が現在救っている segment** (oracle 陰性なら純置換で新規回帰になる集団、repeat 系 fixture が大量に該当) は未測定だった。配線は **oracle OR mute-dip** (棄却には両陰性が必要 = 現行救済の上位集合) とし、mute-dip のみが救ったケースに provenance マーカー `residual-fresh-mute-dip-only` を付けて限界寄与を実測する。寄与 ≈ 0 が実測されたらそのデータを添えて mute-dip 項の退役 (純置換) を判断
- **gate reason の削減**: `residual-decay-no-reattack` の mute-dip 依存判定が退役。fscan 相乗りが成立すれば recent-note 走査 + octave-up rescue も退役 (merge 条件 3 の実証は前者だけでも成立し得るが、判断はユーザー)

## 2. post-stage (第 3 巡) との構造差 — なぜ今回は時刻ずれ問題が出ないか

第 3 巡の敗因は「棄却後に検死しても、event 時刻は tracker hit 時刻になり、segment 再分割による時刻ずれが残る」だった。in-stage は**棄却しない**という判定なので、生き残った segment は通常の選択フロー (primary/secondary 選択・時刻は segment 由来) で event 化される。lowpass WARN の F5 について期待される動作: 再分割された segment が棄却されなくなり F5 は segment 時刻で出る。**ただし再分割自体が segment 境界を動かす可能性は残る**ため、WARN 完全解消は成功条件にしない (期待値は「dropped の純減」まで。§5)

## 3. 実装アーキテクチャ (C2 相談対象)

- tracker の復調は現 pipeline hook で audio 全体に対して 1 回/リクエスト。segment 評価 (peaks) で使うには**評価器を前段で構築して segment_peaks に渡す**形になる: `pertine.build_fresh_attack_oracle(audio, sr, tuning) -> callable(t0, t1) -> bool` を pipeline が構築し、`segment_peaks(..., fresh_attack_oracle=...)` で注入
- peaks.py の変更は分岐条件 1 行 + oracle 引数の配線。**新規定数は追加しない** (検出核 bars・reinject は pertine.py の既存較正値。窓は segment [start, end] の strict 意味論 — 第 3 巡較正 1 をそのまま採用)
- oracle の判定 = 第 3 巡 veto の受理条件から「既存 event 系 guard」を除いたもの: 検出核 (phase ≥ 0.7 / jerk ≥ 150) + reinject ≥ 1.0 + bleed explaining-away。tier/優勢判定は不要 (event を作るのではなく「棄却しない」だけなので、その後の選択フローが精査する)
- `use_pertine_tracker_rescue` とは独立の flag `use_pertine_residual_oracle` (dual-run の toggle)。**C2 相談パッケージ**: この設計 + 置き換える既存機構 (mute-dip 依存判定、成立時は fscan も) + probe B 実測を添えて実装前に提出

## 4. リスクと主戦場

1. **clean suppression 18 件の保全が主戦場**: 棄却判定の緩和なので、守っていた precision を壊すと K4 (非飽和 FP ≤ 61) に直撃する。probe B の実測では clean suppression の dropped primary への誤発火は 1/18 (strict 窓で 0/18) だが、これは「dropped primary の tine」だけの測定。oracle は**全 tine** を見るため、第 3 巡と同じ「probe が測っていない regime」がある — 実装後にまず 33 slot 全数で oracle 判定を dump して fixture/GT と突き合わせる (第 3 巡の反省: fixture 側 regime を先に測る)
2. **棄却しなかった segment の下流挙動**: 棄却回避 = そのままでは residual 音が primary として event 化されうる。oracle が発火した segment では「fresh attack した tine」が ranked に居るはずで、通常の選択フローが拾う想定だが、**residual primary が勝ち残るケース**が fixture で出る可能性が高い (第 3 巡の教訓から事前に想定)。対策候補は「oracle 発火 tine を confirmed_primary 相当で優遇」だが、これは追加機構なので**まず素の置換で測る**
3. **過適合規律**: held-out 2 本 (1955b5bd / 98019f67) は較正に不使用、測定 1 回のみ (#203 固定)。fixture/GT 較正で新しい定数を足す事態になったら、それ自体を「置換が成立しない徴候」として扱い設計を見直す

## 5. 検証プロトコル (事前固定)

1. **oracle 全数 dump**: 現行認識の residual-decay 棄却 33 slot (probe A) に対し oracle 判定を照合 — FN 重なり slot の回収可能数 / clean suppression の誤発火数を実装前ベースラインとして記録
2. fixture suite (609) 非劣化
3. **統合 dual-run (第 3 カウント巡)**: base = tracker OFF / 対象 = rescue + oracle ON。K2 ≥ 0.875 / K3 ΔR ≥ +0.015 (CI が 0 を跨がない) / K4 FP ≤ 61 (S4 分母)
4. metamorphic alarm (ebecf0c6 lowpass): 成功条件は **dropped の純減** (5→4 以下)。完全解消は要求しない (§2)
5. held-out 到着後: 全数値を同一集合で 1 回再測 (rescue 寄与・oracle 寄与・fscan 寄与を arm 分解)
6. merge 条件 (3) の判定材料: `residual-decay` 判定の mute-dip 依存の退役 + (成立すれば) fscan 退役を C2 相談で提案

## 6. 撤退基準 (事前固定。**発火時は即撤退ではなく再相談** — 2026-07-06 C2 相談でのユーザー条件)

- clean suppression 誤発火が oracle 全数 dump の時点で **3/18 超** → 実装に進まず相談 (probe B の 1/18 から大きく悪化 = 前提崩れ)
- fixture 較正で **pertine.py の既存定数以外の新規閾値**が必要になった → 「素の置換では成立しない」徴候として相談
- dual-run で K2-K4 のいずれか失敗 → 連続失敗 1/2 として記録し、第 4 カウント巡に進むかはユーザー相談

C2 相談は 2026-07-06 に実施済み・承認 (条件: 撤退基準発火時は撤退実行前に再相談)。#203 記録。

## 8. 実装巡の経過記録 (2026-07-06、実装セッションの実測ログ)

素の条件差し替え (OR 配線) は full suite 8 fail (ebecf0c6 で TP 17→14 — 救った残響 segment が後段 merge/suppression と相互作用し正解を壊す)。**再相談 → ユーザー承認で option-i に転換**: oracle は fresh tine の note を返し、既存 forward-scan promotion 経路 (alternative_primary) で segment を乗っ取らせる。その後の較正 3 段 (いずれも既存定数・実測テーブルのみ):

1. **親リダイレクト**: 勝者 tine が他 tine の実測 partial 上にあり親 envelope も同瞬間に re-inject → 親へ差し替え (#149 型衝突。ebecf0c6 F5→C6)
2. **promotion 受理ゲート**: octave-up rescue と同じ score bar (primary_rejection_max_score) + per-note onset gain (RESIDUAL_DECAY_MIN_ONSET_GAIN) を oracle promotion にも適用 (segment 自身が支持しない音を位相だけで立てない)
3. **fscan fallback 復活**: oracle が棄権/ゲート落ちした時は fscan が走る (直列)。**fscan 退役はまだ earned でない**実測 (c4-to-g4 の軟再打鍵 C5 は oracle 圏外 = round 3 §3 の mute-dip 領分の再確認)。両者の限界寄与は dual-run の arm で定量化する

同 note 救済は「oracle が primary を fresh と判定 OR mute-dip」(マーカー `residual-fresh-mute-dip-only` で mute-dip 限界寄与を実測 — GT 15 録音の途中経過では 0 が続く)。branching 第 2 サイトは同 note + mute-dip のみの保守配線。

## 9. 検証プロトコル実測結果 (2026-07-06、§5 手順 2-4 + merge 条件 3 材料)

全機械出力: `pertine-round4-dualrun.json` / `pertine-round4-mutedip-margin.json` (同 dir)。held-out 2 本 (1955b5bd / 98019f67) は不使用 (裁定待ち、測定 1 回のみの規律)。

### 9.1 第 3 カウント巡判定 — K2/K3/K4 全 PASS (3 巡連続)

dual-run 5 arm (base = rescue OFF + oracle OFF / rescue_only = 第 2 巡状態 / full = branch 既定 / oracle_only = full + fscan 切り / no_mutedip = full + mute-dip OR バックアップ切り)、full vs base、S4 分母:

- K2: carryover recall **14/16 = 0.875** (閾値ちょうど) [PASS]
- K3: 非飽和 ΔR **+0.021**、paired bootstrap CI95 **[0.0048, 0.0502]** (0 を跨がない) [PASS]
- K4: 非飽和 FP **53 → 53** (上限 61、増加ゼロ) [PASS]
- 非飽和 headline: base P=0.852 R=0.724 F1=0.782 → full P=0.855 R=0.745 F1=0.796 CI=[0.6695, 0.8663]

### 9.2 限界寄与の arm 分解 (merge 条件 3 材料)

| 機構 | 測定 | 結果 |
|---|---|---|
| oracle (in-stage) | full vs rescue_only | GT 15 録音で**変化なし** (F1 中立) |
| fscan | full vs oracle_only | **2 録音 +2 TP** (70cc6637 0.848→0.852 / 8039a34c 0.998→1.000) |
| mute-dip OR バックアップ | full vs no_mutedip + marker 全数 | **変化なし + marker 0/15 録音** (二重測定で寄与ゼロ) |

fixture 側依存地図 (KALIMBA_SETTINGS_OVERRIDES で実 pytest suite):

- **mute-dip バックアップ切り: 609 全 green** → GT・fixture 両側で依存ゼロ = **退役可の証拠成立**
- **fscan 切り: 3 failed / 606 passed** — fscan 自体の mechanism test (自明) / c4-to-g4-sequence-17-01 / bwv147-sequence-163-01 (163→162 events)

### 9.3 fscan 固有獲物の正体 — 物理 3 event、全て同 note 軟再打鍵

1. 70cc6637 (きらきら星): **C4@25.13s** (直前 C4=23.73s、1.4s 間隔の再打鍵)
2. 8039a34c: **G4@47.01s** (直前 G4=46.53s、0.48s 間隔)。**8039a34c は bwv147-sequence-163-01 fixture と同一録音** (PCM 全 frame 数一致・max|diff|=1 LSB の再エンコード差のみ) — GT 変化と fixture fail は同一 event
3. c4-to-g4 E15: **C5@~13.47s** (ミュート後の軟弾き直し。notes.md memo の mute dip 11→0.5→13 に加え、energy trace 再測で 13.425→13.475s に 519→23→198 の急落 → 13.500s に 1987 へ再励起を確認。耳確認記録はなし = #52 の任意残 2 件の 1 つだが、複数観測により実在判断 B+)

3 件とも oracle の promotion 受理ゲート (score bar + per-note onset gain) が構造的に通さない軟再打鍵 regime = round 3 §3 で特定した mute-dip/recent-note 領分。**fscan 退役は not earned** — oracle の直列 fallback として存置が妥当。

### 9.4 metamorphic alarm (ebecf0c6 lowpass、#206 発端)

dropped **5 → 3 (純減、成功条件 ≤4 充足)**。WARN 自体は残存 (diff=6 > threshold 2) — §2 の予見どおり境界移動由来の added/dropped 組が残るが完全解消は非要求。

### 9.5 測定上の注意 (再現用)

- 直 POST の in-process 診断は dedup で 2 回目の応答が汚染される (本巡でも bwv147 probe が一度 on=off=163 の偽結果を返した)。**dryRun+force か nfb.transcribe_payload 経由必須** (memory 済みの既知罠の再演)
- default 設定 suite は 609 green (新 flag `ablate_residual_mute_dip_backup` は既定 no-op)
