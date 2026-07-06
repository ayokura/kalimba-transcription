# 中期作業計画 第 4 期 (2026-07-07 起点、4 方向 × 8 スプリントの計画ストック)

- 作成: 2026-07-06 (第 3 期 S7)。**状態: draft — 別系モデル監査 (docs/research/2026-07-s7-plan-audit-request.md、ユーザー指摘で GO の前提に昇格) → 所見反映 → ユーザー GO で active 化** (S7 exit の硬ゲート)
- 前計画 [`sprint-plan-2026-07c.md`](sprint-plan-2026-07c.md) (第 3 期) は S7 完了で superseded (実績記録として凍結)
- tracking: 第 4 期 tracking issue (GO 時に新設。#203 は第 3 期記録として凍結)
- **実行体制の前提: Fable 非依存** (ユーザー指示 2026-07-06)。主担当 = Opus 4.8、定型実装 = Sonnet、敵対的レビュー = Codex ([[model-roles-third-term]] 継承)。Fable はサブスク復帰時に「再評価ポイント」で合流する

## 本計画の構造 (第 3 期からの最大の変更)

第 4 期は**単一のスプリント列ではなく、独立した 4 本の計画 (方向性 A-D、各 8 スプリント) の直列ストック**である。ユーザーの構想: 複数の計画を順番に完走し、Fable のサブスク復帰を待つ。

- 各計画は単独で完結する価値を持ち、8 スプリント (= 人間 GO 区切りの作業チャンク、第 3 期定義を継承) で構成する
- **一度に active なのは 1 計画のみ**。計画間の切替もユーザー GO (計画レベルの硬ゲート)
- 各計画の S4 相当に**中間ゲート**を置き、「継続 / 次計画へ切替 / Fable 復帰待ち凍結」を判断できる構造にする
- 推奨実行順序: **B (出力・UX) → A (較正・汎化) → D (残 FN 攻略) → C (streaming/browser)**。根拠は「実行順序の設計」節

## 第 3 期からの引き継ぎ (前提となる到達点)

- per-tine research line 本線投入済み (PR #210: tracker + in-stage oracle、K 3 巡 PASS)。kill-criteria doc は役目終了
- headline: 非飽和 (n=11) micro F1 **0.744 CI95 [0.627, 0.846]** (再固定済み baseline: ffb5af9)
- M1 進入条件充足 (他者録音 7 本、うち 34L-C で楽器個体差の実データ取得)
- bets 決着: PESTO 退場 / NMF 対抗馬は起動条件消滅 / partial 実測・メタモルフィック警報は常設化 (#203 2026-07-06 記録)
- 残 open: #209 (Phase 2-3 PR)・#202 (方式判定待ち)・#211 (e2e flaky)・#208 (menu)。98019f67 は S7 中に権利確定 (サリーガーデン、PD) → repo corpus 昇格済み (非飽和ゲート n=6 復帰)

## 確度評価の定義 (継承)

| 確度 | 意味 |
|---|---|
| A | 要否確定・着手可 |
| B | 実施ほぼ確実・調整余地あり |
| C | 外部条件依存 (録音到着・先行結果・ユーザー判断) |
| D | 方向性の仮置き |

---

## 計画 A: 較正・汎化 — M1 正面 (較正系 #172-174 + 楽器適応)

**目的**: 「作者の 17-C でしか動かない認識器」から「楽器・録音環境・奏者に適応する認識器」へ。M1 の実装本体。
**根拠データ**: 1955b5bd (34L-C) で実証された楽器個体差 — A4 tine +20c offset (7 測定で一貫)、フルクロマチック配列で梁 partial が実在 tine に着地する構造 (E6←A4x3 等)。gain 絶対閾値のマイク距離脆弱性 (#33)。
**kill/consult 規律**: 認識器コアの閾値変更を含むため、着手前に kill 条件数値固定 (ガードレール 12 継承) を A3 開始前に行う。

| S | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| A1 | 34L-C partial table 実測 | gt_agent_observations の手法を汎用化し、34L-C の per-tine partial table を高 confidence 単独発音から自己教師あり構築 (17-C 版 S3 と同手法)。1955b5bd + 手持ち 34L-C 録音 | A |
| A2 | tine tuning offset 推定 | per-recording の tine 実チューニング offset (A4+20c 型) を認識時に自動推定する設計 + 実装 (#172 の実体)。cents 同定は gt_agent 系の実測済み手法 | B |
| A3 | offset 較正の dual-run | A2 を 34L-C/G-low 録音で dual-run 評価。kill 条件は A3 開始前に数値固定 | B |
| A4 | **中間ゲート** | 較正 ON/OFF の統合判定 + 実行順序の再確認 (ユーザー GO)。NO-GO 分岐 = #173 を先行 | A |
| A5 | backward_attack_gain 正規化 | #173: gain 絶対閾値の per-recording 分布正規化 (マイク距離不変化)。magnetic pickup 系 (G-low/34L-C) vs mic 系 (17-C) の実データで | B |
| A6 | 録音プロファイル | デバイスメタデータ (client.userAgent 推定, f617f8c) → 較正プリセットの永続化。corpus metadata の recording.device 活用 | C |
| A7 | held-out 検証 | 新着録音 (テスター/ユーザー) での 1 回測定。録音到着に依存 | C |
| A8 | 統合 + 去就 | merge 判断・BPM 適応 noise floor (#174) の要否再評価・次計画への引き継ぎ | B |

**Exit criteria**: 較正系 ≥1 件が実データで効果測定され merge/kill が判定済み。34L-C partial table が実測資産として存在。(硬ゲート) ユーザー GO。
**再計画トリガー**: 較正で headline が悪化 → 即中間ゲート前倒し / 新楽器録音が到着 → A6-A7 を前倒し。

---

## 計画 B: 出力・UX — 転写を「使える」ものに (M4 正面)

**目的**: 認識結果を演奏者が実際に使える形 (譜面・修正・候補提示) にする。dogfooding G2「精度が律速」の反証実験も兼ねる — 出力側の改善で「使える度」がどこまで動くかを実測する。
**根拠**: correction cost 0.451 (実測)、candidateFixRate 0.347、#204 Phase 1 が本線入り済みで受け皿が整った。認識器コアに触れないため新体制 (Opus+Sonnet) の立ち上げリスクが最小。

| S | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| B1 | #209 の完成 | PR #209 (run 切替 UI・stale バッジ・baseRunId・一括再認識) のレビュー対応・マージ・prod 反映。未決事項 (dspFingerprint 併用等) の裁定 | A |
| B2 | 記譜法 v1 第 1 次 | #202 方式判定が出ていれば実装流し込み。未判定なら拍グリッド (案 A') の本実装 + 判定材料の再提示 | C |
| B3 | #178 Phase 1 | candidate tier (per-tine 由来の低 confidence 候補) の UI 可視化 — soft-reject を「見える化」し、修正 1 タップ化 | B |
| B4 | **中間ゲート** | 弾き戻しループ第 2 回 + correction cost 再測。出力改善の「使える度」寄与を定量判定 (ユーザー) | A |
| B5 | 修正 UX 深化 | correction UI の改善 (baseRunId 活用の provenance 表示、修正→GT 化の動線短縮) | B |
| B6 | エクスポート | 印刷 CSS / PDF / MusicXML のいずれか 1 形式 (ユーザー選好で決定) | C |
| B7 | dogfooding 拡大 | 判定様式は既存のまま追加 1-2 本 (長め・別難度)。曖昧性カタログの記譜法への反映 | C |
| B8 | 統合 + M4 評価 | M4 実測 (修正時間・弾き戻し成功率) の before/after 総括 | B |

**Exit criteria**: correction cost / 弾き戻し成功率の before/after が記録され、#209/#178-P1 が本線入り。(硬ゲート) ユーザー GO。
**再計画トリガー**: #202 裁定が出る → B2 を即時再計画 / dogfooding が「出力ではなく精度が律速」を再確認 → B5-B7 を縮約して次計画へ早期切替。

---

## 計画 C: streaming/causal + browser 単独 (M2/M3 正面)

**目的**: batch 前提の認識器を causal 化し、ブラウザ単独 (WASM) の転写に到達する。「M1 の前に M2/M3 へ深入りしない」原則があるため**実行順序は最後尾を推奨**。ただし per-tine demod が causal 設計済みで、部品は揃い始めている。
**根拠**: /wasm-demo で onset+pitch-ID が in-browser 稼働済み。kalimba_dsp (Rust) が note_band_energy 等を共有コア化済み。demod tracker は streaming 適合設計 (heterodyne + LPF + hop)。

| S | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| C1 | causal 化の棚卸し | segments.py の batch 依存の全列挙 (全体 noise floor・backtrack・2-pass 構造) + causal 代替の設計メモ | B |
| C2 | 1-pass spike | research branch で 2-pass→1-pass 化 (ablation 比較、メタモルフィック警報を回帰網に) | C |
| C3 | segment 化の WASM 移植 | kalimba-dsp に active-range/collector/gap-rescue を移植 (native/wasm parity test 拡張、check_wasm.sh の CI 配線判断込み) | C |
| C4 | **中間ゲート** | causal 版の品質差 (vs batch) を非飽和 headline で測定 → M2 進入可否 (ユーザー) | A |
| C5 | browser full pipeline | 認識器全段の in-browser 実行 (peaks/events 相当の移植 or 簡約版) | D |
| C6 | streaming UI | 漸進表示プロトタイプ (録音しながら音名が出る) | D |
| C7 | レイテンシ較正 | 窓長/確定遅延のトレードオフ実測 (per-tine demod の確定遅延を含む) | D |
| C8 | 統合 + M2/M3 判断 | 製品形態判断の材料化 (server/browser ハイブリッド構成案) | D |

**Exit criteria**: causal 版の品質差が数値化され、M2/M3 の進入判断材料が揃う。(硬ゲート) ユーザー GO。
**再計画トリガー**: C2 で品質差が大きい (headline -0.02 超) → C5 以降を凍結し「部分 causal (onset のみ streaming)」に再設計。

---

## 計画 D: 残 FN 攻略 — 多声・弱打・hard miss (認識器の次の一手)

**目的**: 非飽和 F1 0.744 の残り部分の系統的攻略。1955b5bd (0.509、hard miss 34) / a9e30986 (0.526、hard 33) が最難関。per-tine の次の research line。
**規律**: 認識器コアの research line のため、**Codex 敵対的レビュー体制の慣熟後**が安全 (実行順序 3 番目の根拠)。kill 条件事前固定 (ガードレール 12)・カウント巡制・C2 相談制を per-tine line と同運用で。

| S | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| D1 | 残 FN の系統分類 | 非飽和 11 録音の FN 全数 (~170) を機構別に自動分類 (broadband 不検出 / segment 未形成 / 棄却 / 候補落ち) + 人手サンプル検証。gt_agent 系計器を流用 | A |
| D2 | 最大系統の標的設計 | probe 先行 (物理前提の検証、投機実装しない)。34L-C partial 混同が大きければ計画 A の成果を前提に組む | B |
| D3 | 実装第 1 次 | kill 条件事前固定 + research branch + dual-run (カウント巡 1) | C |
| D4 | **中間ゲート** | K 判定 + broadband-vs-per-note の 4 トリガー再評価 (#208 統合 front-end の起動判断を含む) | A |
| D5 | 分岐 | 継続 (カウント巡 2) or #208 complex front-end の設計着手 | C |
| D6 | 実装第 2 次 | 巡継続 or front-end probe (複素 AR(1) 残差の判別力測定) | C |
| D7 | held-out 検証 | 新着録音で 1 回測定 (録音到着依存) | C |
| D8 | 統合 + 去就 | merge/kill 判定 + 次期 (第 5 期) への材料化 | B |

**Exit criteria**: FN 系統分類が資産化され、標的 1 系統について merge/kill が判定済み。(硬ゲート) ユーザー GO。
**再計画トリガー**: D1 で「最大系統が較正起因」と判明 → 計画 A 未消化なら A へ切替提案 / 4 トリガー発火 → #208 を D5 で正面化。

---

## 実行順序の設計 (推奨: B → A → D → C)

| 順 | 計画 | 根拠 |
|---|---|---|
| 1 | **B (出力・UX)** | #209 が既に open で即着手可。認識器コア不触 = 新モデル体制の立ち上げリスク最小。#202 裁定が出れば即流し込める。ユーザーの「使う」ループが回り始め、以降の計画の反証データ (用途検証) が増える |
| 2 | **A (較正・汎化)** | M1 正面。34L-C の実データが手元にあり録音非依存分 (A1-A3) から始められる。B の期間中に新録音が届けば A7 の材料になる |
| 3 | **D (残 FN)** | 認識器 research line は敵対的レビュー体制 (Codex) の慣熟後が安全。A の較正成果 (partial table・offset) が D1 分類の精度を上げる依存関係もある |
| 4 | **C (streaming/browser)** | 「M1 の前に M2/M3 へ深入りしない」原則の帰結。A/D で認識器が固まってからの移植が手戻り最小 |

- 順序変更はユーザー判断でいつでも可 (計画間切替はユーザー GO)
- **Fable 復帰時の合流点**: 各計画の中間ゲート (S4 相当) と計画境界。復帰時は「現計画の残りを Fable で加速」or「次計画の設計を Fable で前倒し」をユーザーが裁定。深い音響分析・設計文書・統合判定資料が Fable 向きの仕事 (第 3 期実績より)

## ガードレール (第 4 期 — 第 3 期 1-14 を継承 + 追加)

第 3 期ガードレール 1-14 は全て継承 (events.py 新規 pass 禁止 / 実測 partial のみ / NN 本体置換禁止・外部 AMT 計器限定 / headline 語彙 / R@K 単独 KPI 禁止 / 閾値調整は非飽和 n≥3 [充足済 n=11] / augmentation 非 gate / pseudo-GT 昇格手続き / prod cadence / 記録一本化 / スプリント境界硬ゲート / kill 事前固定 / spectral pin / 反証優先)。追加・改訂:

15. **【新】計画レベルの硬ゲート**: 計画 (A-D) の開始・切替・凍結はユーザー GO。スプリント境界ゲート (11) の上位版
16. **【新】agent 主導 GT 裁定の運用化**: 人間裁定不能な録音 (多声・非 17 鍵) は複数観測裏付けの agent 裁定 (spectrogram_verified) を正規手続きとする。ただし (a) 事後の per-recording 較正情報 (tuning offset 等) を必ず資産化、(b) ユーザー初期判定を覆す場合は明示報告、(c) 権利未確認録音の観測 JSON は local-only (2026-07-06 の 1955b5bd 運用を規範化)
17. **【改訂】飽和ゲートの運用**: 非飽和録音が飽和で「卒業」した場合の gate 再固定は fixture-policy 手続き (--allow-baseline-regression + ユーザー承認 + 経緯記録) — 2026-07-06 の 4e1ae5c6 運用を規範化

## 再計画トリガー (計画横断)

- **他者録音の新規到着** → 現計画を問わず受入を最優先割り込み (横断 1 継承)。A7/D7 の held-out 材料
- **#202 方式判定が出る** → 計画 B の B2 を即時再計画 (B 実行中でなければ B の優先度を再提案)
- **Fable サブスク復帰** → 合流点ルール (実行順序の節) で裁定
- **非飽和 headline が 2 計画連続で変化なし** → 計画ストック全体の再設計 (第 5 期前倒し)
- **モデル体制の受入チェック失敗** (新モデルが現在地を自力提示できない) → 引き継ぎ資産の補修を最優先

## 人間アクション予算 (継承: 1 スプリント 1-2h 上限)

| 優先 | アクション | 計画/時期 |
|---|---|---|
| 1 | 本計画への GO (+ 実行順序の裁定) | S7 exit |
| 2 | #202 方式判定 (モック 4 案) | B2 前が効率最大 |
| 4 | 弾き戻し検証第 2 回 + dogfooding 追加 | B4/B7 |
| 5 | 各計画の中間ゲート判定 | 各 S4 |
| 6 | controlled isolated repeats 録音 (S4 監査由来、C4/A4/D5/F5 + 同時ペア) | A1-A2 の較正データ拡充 |
| 7 | 2cc06261 / 01fc3b8b の gt-review 裁定 (残 2 本) | 随時 (非飽和データ拡充) |

## 中長期展望 (M1-M5、更新)

- **M1 (汎化)**: **進入済み**。計画 A が実装本体。exit 条件 (次期に定義): 未較正の新環境録音で headline 劣化が CI 幅以内
- **M2 (streaming)**: 計画 C の中間ゲートが進入判定。per-tine demod の causal 設計済みが追い風
- **M3 (browser 単独)**: 計画 C 後半。自前 AMT + WASM 方針は不変
- **M4 (出力)**: 計画 B が実測を前倒し。correction cost 0.451 が baseline
- **M5 (公開)**: 変更なし (M1 の供給源)。build-in-public devlog は menu 維持

## 関連

- 第 3 期実績: #203 コメント列 (2026-07-05〜06)。第 3 期計画: [`sprint-plan-2026-07c.md`](sprint-plan-2026-07c.md) (凍結)
- per-tine 本線化: PR #210 / decision-log 2026-07-06 / 統合 front-end: #208
- 較正系: #172-174 / 出力: #202・#178・#204 (#209)・#72 / 一般化: #33
