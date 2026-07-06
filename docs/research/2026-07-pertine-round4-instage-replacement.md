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

## 6. 撤退基準 (事前固定)

- clean suppression 誤発火が oracle 全数 dump の時点で **3/18 超** → 実装に進まず設計見直し (probe B の 1/18 から大きく悪化 = 前提崩れ)
- fixture 較正で **pertine.py の既存定数以外の新規閾値**が必要になった → 「素の置換では成立しない」として撤退記録 (第 3 巡と同形式の clean negative)
- dual-run で K2-K4 のいずれか失敗 → 連続失敗 1/2 として記録し、第 4 カウント巡に進むかはユーザー相談
