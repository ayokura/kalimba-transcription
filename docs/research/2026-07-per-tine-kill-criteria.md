# per-tine research line — kill 条件・GO 条件 (実装前固定版)

- 作成: 2026-07-05 (第 3 期 S0、Fable)。**この数値は per-tine tracker 実装着手前に固定されたものである** (第 3 期ガードレール 12)
- 変更手続き: 本文書の数値を実装開始後に変更する場合、(1) `docs/decision-log.md` に変更理由をエントリ追記、(2) **別セッションの敵対的レビュー役** (Codex 等の別系モデル推奨) による監査所見を第 3 期 tracking issue (#203) に記録、の両方を必須とする
- 対象: per-tine 確率トラッカー + A1 causal onset を束ねた research line (bets #4 格上げ、#141)

## 基準値 (2026-07-05 実測、recognizer fingerprint f52bebb6f91763da)

- **非飽和 headline (n=7)**: tp=158 / truth=233 / pred=192 → **micro P=0.823 / R=0.678 / F1=0.744**
- 非飽和 7 件: 17ea7626 (0.866) / 4e1ae5c6 (0.769) / 9ce7df83 (0.769) / a9e30986 (0.532) / d7a82772 (0.828) / ea7edd71 (0.865) / ebecf0c6 (0.872)
- fixture suite: 574 tests green / corpus gate 13 録音 baseline
- 注: S1 の corpus 昇格・S2 の GT 除染・きらきら星 GT 追加で基準値は移動する。**その場合は S4 (実装ゲート) 時点の再測定値を「基準値」として本文書に追記する (数値の置換ではなく追記)**。kill/consult 判定は常に「実装開始時点の基準値」との比較で行う

## S4 時点の基準値追記 (2026-07-05、recognizer 29eaaf1daa45ca68 — 本文書の規定による追記、S0 値の置換ではない)

S1 corpus 昇格 + S2 GT 除染 + きらきら星 2 本 GT 化を経た再測定。**実装後の kill/consult 判定はこの S4 基準値との比較で行う** (gpt-5.5 監査 2026-07-05 の指摘 5 対応):

- **非飽和 headline (n=9)**: tp=304 / FP=53 / FN=115 → **micro P=0.852 / R=0.726 / F1=0.784** (bootstrap CI95 [0.666, 0.849])
- 非飽和 9 件: 17ea7626 (0.866) / 47902d34 (0.818) / 4e1ae5c6 (0.769) / 70cc6637 (0.835) / 9ce7df83 (0.769) / a9e30986 (0.526) / d7a82772 (0.837) / ea7edd71 (0.865) / ebecf0c6 (0.872)
- fixture suite: 596 tests green
- **判定式の分母の明確化**:
  - K2: carryover-mask 2 録音 (4e1ae5c6 + 9ce7df83) の合計 recall。現状 10/16 = 0.625 → dual-run で **≥ 14/16 = 0.875**
  - K3: 非飽和 (n=9) micro R = 0.726 に対し dual-run 側が **+0.015 以上** (= R ≥ 0.741)、かつ bootstrap 95% CI が 0 改善を含まない
  - K4: 非飽和 pooled FP = 53 に対し **+15% ≈ +8 FP 以内** (= FP ≤ 61) に 2 巡で収められること
- 注: 評価録音集合が n=9 から変わる場合 (新 GT 追加) は、dual-run の両側を同一集合で測り、本節の値と直接比較しない (集合を明記して再計算)
- **historical note**: 下方の S0 節・kill 条件表の根拠列にある数値 (tp=158→162 / +0.017 / FP=34 / +5 FP 等) は S0 基準 (n=7, f52bebb6f91763da) 当時のものであり、**判定分母としては本節が正** (fugu-ultra 監査 2026-07-05 の残渣指摘対応)

**K6 の改訂 (2026-07-05 ユーザー承認済み)**: スプリント再定義 (人間 GO 区切り) により時間 box 性が弱まったとの監査指摘 (gpt-5.5 指摘 5) を受け、K6 を **「S5-S7 の 3 スプリント、または S5 実装開始から 30 暦日のいずれか早い方」** に改訂した。変更手続き (ガードレール 12): 別系モデル監査 = gpt-5.5 指摘 5 (変更の動機) + fugu-ultra 残渣所見 (手続き確認)、decision-log 2026-07-05 エントリに記録、ユーザー承認 = 2026-07-05 (#203 のセッションで明示)。

## 実測済みの参照点 (期待値の根拠)

- carryover-mask 2 録音 (4e1ae5c6 / 9ce7df83): S5 spike の bg rescue / phase-reset rescue はいずれも **0.769→0.933** を達成した (fixture 回帰と引き換えのため未マージ)。これが per-tine の**最低獲物**であり、tracker がこれを fixture 無回帰で回収できないなら single-instant 手法に対する優位がない
- 最低獲物の headline 換算: carryover-mask の C5 ×4 回収で tp 158→162、**非飽和 R 0.678→0.695 (+0.017)**
- spike 2 巡の教訓: bg rescue = fixture 2 本回帰 / phase-reset (絶対閾値+単発 dominance) = fixture 5 本回帰。**単発時点判定の fixture 回帰は 2〜5 本** — tracker はこれを 0 にできて初めて意味がある
- 位相追跡検出プローブ: broadband 包含 92-96%、高速テイクで 6 倍検出、**過剰検出 2-5 倍** (cross-tine 状態推定なしの上限)

## GO 条件 (S4 実装ゲート — すべて満たしたら実装開始可)

| # | 条件 | 判定材料 |
| --- | --- | --- |
| G1 | #149 衝突プローブが「per-tine partial table が隣接 tine 汚染で自壊する」ことを**示さない** | S0-S3 のプローブ結果 (`2026-07-149-collision-probe.md`) |
| G2 | 用途検証が「精度が律速」を支持する (「粗い転写で足りる」が支持された場合はガードレール 14 により自動再審査) | S2 用途検証の定量結果 (事前定義の判定様式) |
| G3 | GT 除染 (bp-only 統合) 後も、under-detection (FN) が非飽和 headline の主要因子である (FN > FP) | S2 除染後の headline 内訳 |
| G4 | spectral pin 済み録音 ≥2 (きらきら星 2 本想定) — timing-sensitive 評価の土台 (ガードレール 13) | S3 pin 済みリスト |

## kill 条件 (実装後)

dual-run (main の recognizer vs per-tine 統合版) を評価単位とする。「巡」= dual-run 1 回とその改修サイクル。

条件は 2 種類ある (2026-07-05 ユーザーレビューで K1/K5 を consult 化):

- **hard-kill (K2/K3/K4/K6)**: いずれか成立で research line を kill し、NMF 対抗馬の起動判断へ
- **consult (C1/C2)**: 成立しても自動 kill しない。**ユーザーとの相談トリガー**であり、trade-off の内容を添えて判断を仰ぐ

| # | 種別 | 条件 | 数値/定義 | 根拠 |
| --- | --- | --- | --- | --- |
| C1 | consult | fixture 回帰が解消できない | full fixture suite の回帰 ≥1 件が **2 巡連続**で残る → 相談 | 得る価値の方が大きい場合がある: TOP-1 で見ている fixture の回帰でも、候補に残る等で編集対応が容易なら受容を検討しうる (2026-07-05 ユーザー判断)。相談時は「回帰 fixture の一覧 + 各回帰の候補残存状況 (cR@3) + 編集コスト見積り」を添える |
| K2 | hard-kill | 最低獲物の未回収 | carryover-mask 2 録音の合計 recall が dual-run で 0.875 (= 14/16 GT) に届かない状態が **2 巡連続** | spike 実測 0.933 が到達可能性の証明。tracker がこれ未満なら single-instant 比の優位なし |
| K3 | hard-kill | headline 改善なし | 非飽和 micro R の改善が **+0.015 未満** (かつ bootstrap 95% CI が 0 改善を含む) が **2 巡連続** ※判定分母は S4 節 (R=0.726) | 最低獲物換算 +0.017 を下回る = 獲物を取れていない (根拠数値は S0 当時) |
| K4 | hard-kill | precision の悪化 | 非飽和 pooled FP が main 比 **+15% 超**の悪化を 2 巡連続で解消できない ※判定分母は S4 節 (FP=53 → 上限 61) | 根拠列の pred=192 / tp=158 / FP=34 / +5 FP は **S0 (n=7) 当時の値** — 判定には使わない (S4 節が正)。位相追跡の過剰検出を tracker が制御できない兆候 |
| C2 | consult | 本線への漏出 | recognizer 本線 (`constants.py` / events.py) への新規定数・pass の追加が必要になる → 相談 | 既存本線の構造は 3 回見直しているとはいえ推測混じりであり、成果次第では構造改革もあり得る。そうでなくても次回の構造変更時の考慮事項になる (2026-07-05 ユーザー判断)。相談時は「必要になった変更の内容 + それが置き換える/追加する既存機構」を添える |
| K6 | hard-kill | 期間 | S5-S7 の 3 スプリント、**または S5 実装開始から 30 暦日のいずれか早い方**で merge 判断材料 (K2-K4 の判定) が揃わない (2026-07-05 改訂、ユーザー承認済み — 上の S4 節参照)。**起算日 pin: S5 実装開始 = 2026-07-05 (tracker v0 5e007c5) → 期限 2026-08-04** (#203 記録と一致。2026-07-06 注記追記 — 数値変更ではない) | 無期限 research 化の防止。スプリントが GO 区切りになったため壁時計上限を併設 |

## merge 条件 (kill を生き延びた後、本線投入の判断)

reassessment §3.3 NEXT exit (= #141 の 3 条件) を継承する: **(1) fixture exact-match 非劣化 + (2) 自由演奏指標 (非飽和 headline) の改善 + (3) 既存 suppression pass / gate reason の削減**を同時に満たした時だけ merge を検討する。(3) は「tracker が既存 patch を置き換える」ことの実証であり、単なる追加レイヤーなら merge しない。

## 判定の運用

- **巡カウントの起点**: offline/spike 形の dual-run (S5 round 1 の後段判定器測定など) は**カウント外**。kill 判定の「巡」は**統合 dual-run (pipeline 統合後の settings-flag toggle 形) の第 1 回を第 1 巡**として数える (2026-07-06 明文化 — S5 実績の #203 記録「統合 dual-run 第 2 巡 (カウント第 1 巡)」と同義)
- 判定は dual-run の機械出力 (fixture 結果 + 非飽和 headline + CI) を第 3 期 tracking issue に貼った上で行う。agent の散文解釈のみでの判定は不可
- consult (C1/C2) の成立は tracking issue に相談事項として記録し、ユーザー判断を待つ (agent が受容/kill を仮判断しない)。
- kill 発動も「失敗」ではなく exit 成立として記録する (第 3 期 S5 exit criteria)。kill 時の資産 (partial 実測テーブル・位相特徴・プローブ群) は NMF 対抗馬と較正系 #172-174 に引き継ぐ
