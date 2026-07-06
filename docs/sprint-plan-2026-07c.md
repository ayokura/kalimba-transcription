# 中期作業計画 第 3 期 (2026-07-05 起点、S0 + 7 スプリント + 中長期展望)

- 作成: 2026-07-05。**状態: active**
- 権威戦略 doc: [`research/20260626-unbiased-amt-reassessment.md`](research/20260626-unbiased-amt-reassessment.md)
- 前計画 [`sprint-plan-2026-07b.md`](sprint-plan-2026-07b.md) (第 2 期) は superseded (実績記録として凍結)。第 2 期の S1-S8 相当は 2026-07-04〜05 の 2 日で消化された — この事実自体が本計画の運用変更 (スプリント再定義) の根拠である
- tracking: 第 3 期 tracking issue (S0 で新設。#199 は第 2 期記録として凍結)
- 策定入力: 第 2 期実績 (#199) + 敵対的レビュー 1 本 + 新規挑戦提案 14 件 (AI 実行可能 7 + ユーザー主体 7) + ユーザー決定 (背骨・Fable 残時間・bets 判定・他者録音の実態)
- 8 チャンク (S0-S7) は最低ライン。見直しタイミング: S4 (実装ゲート) または再計画トリガー発火時

## スプリントの再定義 (第 3 期の最重要な運用変更)

第 2 期の反省: 2 日で 8 スプリントを消化した事実は、スプリントが時間 box として機能していないことを示す。第 3 期では **「スプリント = 人間の明示 GO で区切られる作業チャンク」** と再定義する。

- 各スプリントの exit criteria は **outcome ゲート** (測定可能な結果) と **硬ゲート** (人間の明示 GO) の 2 層
- **agent の仮判断でスプリント境界を跨ぐことを禁止** (ガードレール 11)。GO 待ちの間は「次スプリントに属さない継続作業」(非同期 #202、bets 棚、レビュー対応、到着録音の受入) のみ可
- 確度評価 A-D の定義と運用規則は第 2 期から無変更で継承

## 前期からの戦略転換 (5 点)

第 2 期は認識器の前進 (非飽和録音での実改善 + per-tine への収束実証) を達成したが、敵対的レビューが 4 つの構造欠陥を指摘した: (1) headline 指標 (13 録音 micro F1 0.926) は 69% が飽和録音の希釈で、非飽和実力は recall ~0.76、(2) CI が守るのは repo corpus 3 録音のみで残り 10 はローカル依存、(3) per-tine への「3 方向収束」は同一 agent・同一日の解釈で確証バイアスの疑い、(4) GT が recognizer 出力を種にした user_corrected で盲点を継承 (bp-only 23 / both-miss 56 が傍証)。

1. **headline の正直化**: 進捗報告の headline を「非飽和限定 micro F1 + bootstrap 95% CI 併記」に置換する。全録音 pooled micro F1 の headline 使用 (「13 録音 micro F1」) は禁止語彙とする。飽和録音は回帰網 (pass/fail) 専用
2. **評価の再現性を CI に**: 敵対的セルフテイク 4 件 (作者演奏 = 権利明白) を repo corpus に昇格し、CI に非飽和ゲートを追加する (CI 網 3→7 録音)。これが第 3 期の最優先配管
3. **背骨 = 並行 + 実装ゲート**: 反証系 (録音多様化・用途検証・GT 除染) と per-tine research line の録音非依存分 (kill 条件・#149 検証・ROC 較正・設計) を並行で走らせ、**per-tine 実装本体の GO は反証結果の通過を条件**とする。「最も安い反証を最も高い投資より先に」の運用形
4. **確証バイアスの構造的防御**: research line の kill 条件は実装前に数値固定 (S0)。実装後の変更は decision-log 記録 + 別セッションの敵対的レビュー役 (Codex 等の別系モデル推奨) の監査を必須にする
5. **モデル引き継ぎの計画化**: Fable (現行モデル) 残り ~48h を S0 として明示的に組み、引き継ぎ資産の整備と「Fable がやると得な深掘り」の前倒しを完了させてから Opus+Sonnet / Codex 体制へ移行する

## 確度評価の定義 (第 1-2 期から継承)

| 確度 | 意味 |
|---|---|
| A | 要否確定・着手可 |
| B | 実施ほぼ確実・調整余地あり |
| C | 外部条件依存 (録音到着・先行結果・ユーザー判断) |
| D | 方向性の仮置き |

運用規則: スプリント確度 = exit criteria コア項目の確度。exit 外 B 項目のスリップは自動繰越、2 連続繰越で要否再判定。

## 全体構図 (レーン再編)

- **レーン 1: 反証系** (最上位) — (a) 録音多様化 (テスター 3 曲受入・きらきら星 verdict・敵対的テイク昇格・ユーザー自身の新録音)、(b) 用途検証 (dogfooding・弾き戻しループ・「粗い転写で足りる」仮説の検証)、(c) GT 除染 (bp-only 23 件の人手 verify → GT 統合)
- **レーン 2: research line (統合)** — per-tine 確率トラッカー + A1 causal onset + 位相リセット/位相追跡 onset を単一 research line に束ねる (bets #4 格上げの受け皿)。前半は録音非依存分のみ、実装本体は S4 硬ゲート通過後
- **レーン 3: 計器・CI 配管** — benchmark 正直化 (S0)、CI 非飽和ゲート (S1)、メタモルフィック警報 (S2)、較正系 #172-174 (他者録音到着で重要化)
- **レーン 4: web・出力** — #202 記譜法 (非同期、ブロッカーにしない)、譜面/レビュー UI 保守。browser/wasm は独立レーンを廃止しこのレーンの従属スライスに (A1 が research line に移管されたため)。上限運用 (1 スプリント最大 1 スライス、コミット比率 25%) は継承。(2026-07-05 追記) 新規 2 件を編入: [#205](https://github.com/ayokura/kalimba-transcription/issues/205) gt-review energy trace (wasm) → S2 スライス (GO 待ち非同期着手可)、[#204](https://github.com/ayokura/kalimba-transcription/issues/204) 認識結果 run 分離 → S5 (Phase 1) / S6 (Phase 2-3)
- **横断 1: 他者録音到着分岐** (維持) — 到着次第、最優先割り込み。テスター 3 曲の到着が既知の発火予定
- **横断 2: research bets 棚** (維持、合計 ≤1sp/sprint 目安)
- **横断 3: モデル引き継ぎ運用** (新設) — S0 で重く、S1 で受入確認、以降は運用規約として常駐

## 人間アクション予算 (1 スプリント 1-2h 上限、優先順位付き)

| 優先 | ユーザーアクション | Sprint | 準備物 (エージェント) |
|---|---|---|---|
| 1 | きらきら星 2 本の verdict 完了 | S0-S1 | gt-review 残行一覧 + 差分ハイライト |
| 2 | テスター 3 曲の権利確認 + GT 化 (到着次第) | S1 | corpus scaffold + 事前 alignment 診断 + gt-review タブ |
| 2' | (2026-07-06 追記) 上記の実体対応: テスター録音は計 7 本到着し 5 本処理済み。**残 2 本 = 1955b5bd (34L-C、memo「ゲゲゲ」— 権利確認に曲名確定も必要) と 98019f67 (17-C)**。この 2 本の gt-review 裁定 + 権利確認が S6 held-out 検証 (第 2/3 巡の再測) の unblock 条件 (#203) | S6 | gt-review ドラフト生成済み |
| 3 | bp-only 23 件の人手 verify (GT 除染) | S2 | 音声スニペット + 両判定併記 + ワンタップ verify UI。20-30 分 ×2 想定 |
| 4 | 用途検証: dogfooding 1 回 + 弾き戻し検証 1 回 (15-30 分/回) | S2-S3 | プロトコル票 (曖昧性カタログ記入欄付き) + 弾き戻し用譜面 export |
| 5 | per-tine 実装 GO/NO-GO 判断 (S4 硬ゲート) | S4 | 反証 3 系統の統合判定資料 + 別系モデル監査所見 |
| 6 | ユーザー自身の新録音 1-2 本 (BWV147 以来の不足解消) | S2-S4 任意 | 敵対的メニュー第 2 版 (tulip 型低 F1 帯狙い) |
| 7 | #202 方式判定 (モック 4 案を「弾き戻せるか」で) | 非同期 | 案 B 実データモック + 拍グリッドプロトタイプ (S0 成果物) |
| 8 | issue triage 承認 (S0 の一括提案) | S0-S1 | 根拠付き提案一覧 |
| - | (随時) 到着録音の権利確認 | 随時 | scaffold 一式 |

## スプリント表 (要約)

| S | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| 0 | **Fable 最終 48h** | 引き継ぎ資産整備 / kill 条件数値設計 / #149 プローブ / benchmark 正直化 / #202 下ごしらえ | A |
| 1 | 計器の正直化の運用 + 録音受入 | テイク 4 件 corpus 昇格 + CI 非飽和ゲート / テスター 3 曲処理 / verdict 完了 / headline 切替全面適用 | A (受入 C) |
| 2 | GT 除染 + 用途検証開始 | bp-only 23 verify / dogfooding + 弾き戻しループ / 曖昧性カタログ / メタモルフィック警報 v0 | B |
| 3 | research line 録音非依存分 | #149 プローブ結果統合 / per-tine partial 実測 / ROC 較正 / spectral pin / 統合設計文書 | B |
| 4 | **実装ゲート + 中間レビュー** | 反証 3 系統統合判定 / 別系モデル監査 / per-tine GO/NO-GO (ユーザー) / 後半再確定 | A |
| 5 | 分岐: per-tine 実装 or 代替 | GO: tracker 実装 (dual-run、kill 条件 = S0 固定値) / NO-GO: 較正系 #172-174 or 出力再配分 | C |
| 6 | 分岐継続 + 較正・品質信号 | per-tine 続き or 較正系 / PESTO 盲点重複率 / #202 実装流し込み / (gated) 再合成距離 spike | C |
| 7 | 統合 + 次期計画 | readiness 再評価 / 全 bets 去就 / M1 進入判定 / 第 4 期計画 | D |

## 各スプリント詳細

### Sprint 0: Fable 最終 48h 【確度 A】

Fable の残り時間で「Fable がやると得な仕事」を使い切る。3 領域 (a)(b)(c) は全部やる (ユーザー決定)。時間が尽きた場合の切り捨て順は c → b の一部。a は不可侵。

**(a) 引き継ぎ資産整備** (不可侵): 本計画 doc の確定 + 第 2 期 superseded 処理 + 新 tracking issue 新設 / decision-log 追記 (第 3 期転換・bets 判定・PESTO/外部 AMT 方針・モデル移行) / memory 整理 (禁止語彙・live ポインタ張替え・evergreen 化) / issue 一斉整理の提案 (#14・#145 前提無効化・#140・#136/#137・#96/#97・#194 実質完了。承認制) / #33 刷新 / readiness headline 注記

**(b) 深い分析の前倒し**: per-tine kill 条件の数値設計 (`docs/research/2026-07-per-tine-kill-criteria.md`、変更手続き込み) / #149 衝突の事前検証プローブ設計 + 実行 / benchmark 正直化 (非飽和 headline + `bootstrap_micro_f1_ci` 必須併記、全録音 micro を secondary 降格)

**(c) #202 下ごしらえ**: 案 B 実データモック (verdict 進行待ち) / 拍グリッド (案 A') プロトタイプ (/score dev フラグ配下) / 層 2 音価再量子化の設計メモ / モック類の repo 退避 (セッション Artifact は消滅前提)

**Exit criteria**: (outcome) 引き継ぎ資産一式が repo/issue 上に存在、kill 条件文書が数値付きで存在、benchmark 正直化 merge 済み、#202 に材料追記済み。(硬ゲート) **ユーザーの明示 GO で新モデル体制 (S1) を開始**。GO 時にモデル引き継ぎ受入チェックを実施

### Sprint 1: 計器の正直化の運用 + 録音受入 【確度 A / 受入は C】

| 作業 | 確度 | 補足 |
|---|---|---|
| 敵対的セルフテイク 4 件 (4e1ae5c6 / 9ce7df83 / ebecf0c6 / ea7edd71) の repo corpus 昇格 + **CI 非飽和ゲート** | A | 作者演奏で権利明白。CI 網 3→7 録音、非飽和が過半に。最優先配管 |
| headline 切替の全面適用 (報告・memory・README・readiness) | A | 以降の全報告は非飽和 micro F1 + CI 併記 |
| テスター 3 曲の到着処理 (登録→試聴→review 支援→権利確認→GT 化→昇格判断) | C | 到着待ち。初の「他者の自由演奏」= M1 進入条件の実質クリア。録音環境メタデータの聞き取り (較正系の設計入力) を併せて依頼 |
| きらきら星 2 本の verdict 完了支援 → GT 化 → 位相追跡 ROC の基盤データ化 | B | 人間予算 優先 1 |
| S0 issue triage 提案の承認処理と実行 | A | |
| ベンチ履歴の再固定 | B | GT 除染 (S2) 前の基準点を benchmark_history に明示記録 |

**Exit criteria**: (outcome) CI 非飽和ゲート green / repo corpus ≥7 / 新語彙 headline が tracking issue に記録 / テスター 3 曲の状態が記録。(硬ゲート) ユーザー GO。冒頭に受入チェック (新モデルが現在地 3 行要約・硬ゲート状態・禁止語彙・kill 条件所在を自力提示)

### Sprint 2: GT 除染 + 用途検証開始 【確度 B】

| 作業 | 確度 | 補足 |
|---|---|---|
| bp-only 23 件の人手 verify 支援 → verify 済み分の GT 統合 → headline 再測定 | B | GT の盲点継承除染。統合後変動は S1 再固定基準と比較報告 |
| 用途検証の開始: dogfooding プロトコル + 弾き戻し検証ループ第 1 回 | B | 判定様式を事前定義 (修正回数・修正時間・「直すより諦めた」数・弾き戻し成功率)。**S4 実装ゲートの入力** |
| 音楽的曖昧性カタログの運用開始 | B | dogfooding 票の記入欄として副産物方式 |
| メタモルフィック警報 v0 | B | augmentation 資産流用。既知不変変換での出力一貫性チェックを non-blocking 警報として実装。トリガー 1 (patch 衝突) 自動検出が狙い。回帰 gate 化はしない |
| テスター 3 曲の GT 化継続 (到着済なら) | C | |
| (2026-07-05 追記) gt-review energy trace v1 ([#205](https://github.com/ayokura/kalimba-transcription/issues/205)、レーン 4 の S2 スライス) | B | GT 化支援。70cc6637 (147 行) の verdict セッションと bp-only 23 件 verify の**前**に届く順序で最優先。S1 GO 待ち中の非同期着手可 (レビュー対応相当)。scope は v1 限定 (Rust band_energy_trace + 再生位置 ±1s のスパークライン) |

**Exit criteria**: (outcome) bp-only 23 件の verify 判定が全件付与 (統合は S3 持ち越し可) / 用途検証第 1 回の定量結果が記録 / 警報 1 回以上実走。(硬ゲート) ユーザー GO

### Sprint 3: research line 録音非依存分 【確度 B】

per-tine 実装の手前まで全部終わらせる。実装本体には入らない。

| 作業 | 確度 | 補足 |
|---|---|---|
| #149 衝突プローブの実行 (S0 設計に基づく、未了分) | A | GO/NO-GO 資料の中核 |
| 自己教師あり per-tine partial 実測 (新 bets) | B | 高 confidence 単独発音イベントから per-tine partial table を実測構築。tracker/NMF/再合成の共通前提部品。実測値なのでガードレール 2 (改訂) と両立 |
| 位相追跡 onset の ROC 較正 (きらきら星 GT で) | B | precision 2-5 倍過剰を閾値 sweep で ROC 曲線化。非飽和 n≥5 到達済みで閾値調整は解禁 |
| spectral onset pin (per-tine 対象録音) | A | GT timing は approximate のため timing-sensitive 実装前に必須 (ガードレール 13)。#201 の意味論に従う/なければ確定 |
| per-tine tracker + A1 causal onset の統合設計文書 | B | 観測モデル (位相 RMS / envelope jerk / narrow FFT / 予測残差の比較)、状態空間、causal 化共通基盤、dual-run 計画 |

**Exit criteria**: (outcome) プローブ結果 + partial 実測 + ROC + pin 済みリスト + 設計文書が揃い「S4 で判定可能」。(硬ゲート) ユーザー GO

### Sprint 4: 実装ゲート + 中間レビュー 【確度 A (ゲート自体)】

第 3 期の要。作業量は小さいが判断の質が全て。

| 作業 | 補足 |
|---|---|
| 反証 3 系統の統合判定資料 | (a) 録音多様化: テスター 3 曲 + 除染後 GT での非飽和 headline と弱点分布、(b) 用途検証の定量結果、(c) GT 除染後の recall 盲点実態。1 枚に統合 |
| 別セッション敵対的レビュー役の監査 | kill 条件文書と判定資料への確証バイアス監査。Codex 等別系モデルで実施する運用をここで確立 |
| per-tine 実装 GO/NO-GO (硬ゲート = ユーザー判断) | NO-GO 時の分岐先も事前定義: 較正系 #172-174 前倒し or 出力/UX 再配分 |
| 後半 (S5-S7) の再確定 | 第 2 期 S4 中間レビューと同形式 |

**Exit criteria**: (outcome) 判定資料 + 監査所見 + GO/NO-GO が tracking issue に記録、S5-S7 再確定。(硬ゲート) GO/NO-GO そのものがユーザー判断

### Sprint 5: 分岐 — per-tine 実装本体 or 代替 【確度 C】

- **GO 側**: per-tine 確率トラッカーの実装 (research branch + dual-run)。観測モデルは S3 の partial 実測テーブル + 位相特徴。kill 条件は S0 固定値をそのまま適用 (変更はガードレール 12 手続き)。評価は非飽和 headline + spectral pin 済み録音での timing 検証
- **NO-GO 側**: 較正系 #172-174 をテスター録音・新環境データで駆動、または #202 決定済なら記譜法実装へ再配分
- 共通: #202 の決定が出ていれば実装流し込み (拍グリッド本実装・層 2 再量子化)
- (2026-07-05 追記) 共通: [#204](https://github.com/ayokura/kalimba-transcription/issues/204) Phase 1 (runs/ storage + 再認識 endpoint + 最新 run 解決)。GO 側では dual-run 比較の可視化受け皿、NO-GO 側では出力/UX 再配分の筆頭。発火条件: 認識器の次の実改善 merge までに Phase 1 が入っていること (重複 tx 問題の再発防止)

**Exit criteria**: (outcome, GO 側) dual-run 結果が kill 条件に対して判定済み (改善記録 or kill 発動のどちらでも exit 成立)。(NO-GO 側) 較正系 1 件以上が実データで効果測定済み。(硬ゲート) ユーザー GO

### Sprint 6: 分岐継続 + 較正・品質信号 【確度 C】

| 作業 | 確度 | 補足 |
|---|---|---|
| per-tine 継続 (kill 未発動なら): merge 判断 or 追加検証 | C | merge 条件は #141 の 3 条件を継承 |
| 較正系 #172-174 の残り | C | 他者録音の環境メタデータが設計入力 |
| PESTO 盲点重複率測定 (bets #3 次段) | B | 隔離実行厳守。重複率が事前固定閾値以上なら PESTO 追加打ち切り |
| #202 実装流し込み (未消化分) | C | |
| (gated) 再合成距離 spike | D | partial 実測の成功が起動条件。原理検証のみ |
| (2026-07-05 追記) [#204](https://github.com/ayokura/kalimba-transcription/issues/204) Phase 2-3 (run 切替 UI・queue stale バッジ・corrections baseRunId・corpus 一括再認識) | C | Phase 1 (S5) の完了が前提。一括再認識の before/after は research line の dual-run 評価と共用 |

**Exit criteria**: (outcome) research line の去就 (merge/継続/kill) が記録 / PESTO 判定が出ている。(硬ゲート) ユーザー GO

### Sprint 7: 統合 + 次期計画 【確度 D】

- free-performance-readiness.md の stage 再評価 (S0-S6 反映)
- 全 bets の去就判定 (新規採用分含む)
- **M1 進入判定**: 非飽和 held-out ≥5 (到達済み) + 他者録音 ≥1 (きらきら星 2 本で実質クリア、テスター 3 曲で確定見込み) — 満たせば第 4 期は M1 (汎化) を正面に
- 第 4 期計画の策定 (モデル体制の実績評価を含む)

**Exit criteria**: 次期計画が確定し、全 bets に判定が付き、M1 進入判定が記録されている。(硬ゲート) 次期計画へのユーザー GO

## research bets 棚

### 継続分 (第 2 期からの引き継ぎ、ユーザー判定済み 2026-07-05)

| # | bet | 判定 | 第 3 期での扱い |
|---|---|---|---|
| 旧 1 | DSP augmentation | 完了 | 資産はメタモルフィック警報に転用。新録音到着時の kill 条件検証のみ |
| 旧 2 | ablation observatory | 継続 | 常設計器化 (on-demand)。第 2 巡はデータ拡充後、dead 26 トグルの除去判断とセット |
| 旧 3 | 合議 teacher | 継続 | S6 で PESTO 盲点重複率。外部 AMT は開発計器限定・prod 不混入 (ユーザー方針) |
| 旧 4 | per-tine トラッカー | 格上げ | レーン 2 の本体に昇格 (棚から卒業)。A1 causal onset と同一 research line |
| 旧 5 | playability 拘束 | 凍結 | 変更なし (最終出力 violation 0/491 の実測により FP 信号としては dead) |
| 旧 6 | 物理合成 | 破棄 | 目的 (非飽和データ確保) が実録音で達成済み |

### 新規候補の採否 (2026-07-05 の発散提案 14 件)

| 提案 | 由来 | 判定 |
|---|---|---|
| 自己教師あり per-tine partial 実測 | AI | **採用 (本線)** → S3。research line の共通前提部品 (tracker/NMF/再合成を解錠) |
| メタモルフィックテスト | AI | **採用** → S2 警報 v0。GT フリー回帰網・トリガー 1 自動検出。non-blocking 限定 |
| 再合成距離の GT フリー評価 | AI | **gated 採用** → S6。partial 実測の成功が起動条件 |
| NMF 生成モデル | AI | **gated 採用 (対抗馬)**。起動条件 = per-tine kill 発動 + partial 実測テーブルの存在 |
| per-band 予測残差 onset (AR(1)) | AI | **吸収** — research line 観測モデル候補として設計文書内で比較 |
| 位相コヒーレンス声部分離 | AI | **吸収** — 同上 (位相リセット判別と同じ物理量) |
| tine 遷移言語モデル | AI | **凍結 (menu)** — 単一奏者過適合リスクが M1 方針と衝突。M1 進入後に再評価 |
| 弾き戻し検証ループ | user | **採用 (本線)** → S2 開始。用途検証 + M4 実測 + 記譜法閉ループを兼ねる最重要のユーザー主体挑戦 |
| 音楽的曖昧性カタログ | user | **採用 (軽量)** → S2 から dogfooding 票の記入欄として (副産物方式) |
| 録音環境多様化 | user | **採用 (予算表組込み)** — ユーザー新録音 + テスター録音で実質進行 |
| 正典譜→複数演奏ペア | user | menu — 録音依頼キットの様式に組み込み |
| 即興語彙の自己分析 | user | menu — tine 遷移 LM の前提資料、LM 再評価時に同時起動 |
| build-in-public devlog | user | menu — M5 接続。草稿支援は要請あり次第 |
| 製作者との対話 | user | menu — M1/M5 接続 |

## ガードレール (第 3 期改訂)

1. events.py への新規 suppression pass 追加禁止 (継承。新規 pass は research line 経由のみ、候補保持/降格形は可)
2. per-tine partial の**信念による**既定化はしない (改訂)。**実測 + 検証済み** partial table (S3) の使用は可 — 「実測+検証」と「信念」の区別を明文化
3. NN 本体置換禁止 (継承) + **外部 AMT (Basic Pitch/PESTO) は開発計器限定、配布物・prod コード不混入** (2026-07-05 ユーザー方針を昇格)
4. **headline は非飽和限定 micro F1 + bootstrap 95% CI 併記を必須**とし、全録音 pooled micro F1 の headline 使用を禁止語彙とする (F1=1.000 成功指標化禁止の強化)
5. Candidate Recall@K の単独 KPI 化をしない (継承)
6. 閾値調整を伴う recognizer 改修は非飽和 n≥3 まで禁止 (継承 — n≥5 到達済みで現在は解禁状態。新環境/新奏者の較正は #172-174 の設計に従う)
7. augmentation/合成データは回帰 gate に使わない (継承)。メタモルフィック違反は non-blocking 警報に限定
8. pseudo-GT は人手 verify 前に completed 昇格しない (継承)。bp-only 由来の GT 統合も同一手続き (全件人手 verify)
9. prod デプロイ cadence (継承)
10. 実績記録は第 3 期 tracking issue に一本化 (継承。#199 は凍結)
11. **【新】スプリント境界の硬ゲート**: 人間の明示 GO なしに次スプリントを開始しない。agent の仮判断で境界を跨がない。GO 待ち中は非同期継続作業のみ可
12. **【新】kill 条件の事前固定**: research line の実装着手前に kill 条件の数値を文書固定。実装後の変更は decision-log 記録 + 別セッション敵対的レビュー役の監査を必須とする
13. **【新】timing-sensitive 実装は spectral pin 前提**: GT timing は approximate のため、timing に感度を持つ実装・評価は対象録音の spectral pin 済み onset を前提とする
14. **【新】反証優先の実装ゲート**: 用途検証が「粗い転写で足りる」を支持した場合、per-tine 実装 GO は自動的に再審査 (decision-log 2026-07-04 の留保条項の運用化)

## 再計画トリガー

- テスター 3 曲が S2 末までに未着 → レーン 1(a) をユーザー新録音 + 敵対的メニュー第 2 版に切替。M1 の他者録音条件はきらきら星 2 本で仮充足とする
- 用途検証が「粗い転写で足りる」を支持 → ガードレール 14 発火。per-tine 実装計画を停止し再配分の再計画
- #149 プローブが per-tine の前提破綻を示す → research line 再設計 (NMF 対抗馬の起動判断を含む)
- GT 除染で headline が CI 幅を超えて変動 → baseline 再固定 + 過去比較の全 doc 脚注化
- #202 の方式判定が出る → S5/S6 への実装流し込みを即時計画
- メタモルフィック警報 or observatory がトリガー 1 (patch 衝突) を検出 → research line の優先度前倒し判断
- 非飽和 headline が 2 スプリント連続で変化なし (research line 進行中に) → kill 条件の中間判定を前倒し
- モデル引き継ぎ受入チェックが失敗 → S1 を止めて S0 資産の補修を最優先

## モデル引き継ぎ運用 (Fable → Opus+Sonnet / Codex)

### 引き継ぎ資産一覧 (S0 で整備)

| 資産 | 場所 | 状態要件 |
|---|---|---|
| 第 3 期計画 doc | 本ファイル | AGENTS.md からリンク |
| 新 tracking issue | GitHub | #199 のクローズ記録 + 引き継ぎポインタ |
| AGENTS.md | repo ルート | 禁止語彙・ガードレール 11-14 が最新 |
| decision-log | docs/ | 第 3 期転換 + モデル移行エントリ |
| kill 条件文書 | docs/research/2026-07-per-tine-kill-criteria.md | 数値固定 + 変更手続き |
| #149 プローブ結果 | #149 コメント + docs/research/ | 実行済み or 実行可能な手順 |
| memory | Claude Code auto-memory | 禁止語彙反映・live ポインタ・evergreen のみ |
| 分析 skills | .claude/skills/ | bash ベースでモデル非依存。smoke 確認済み |
| readiness doc | docs/ | headline 注記 (非飽和実力 ~0.76) |
| #33 / #202 | GitHub | 刷新 / 材料追記済み |

### 次モデルが最初に読むべきもの (順序固定)

1. `AGENTS.md`
2. 本計画 doc
3. 第 3 期 tracking issue の最新コメント (live 状態)
4. `docs/decision-log.md` の直近 3 エントリ
5. kill 条件文書 + `docs/research/20260626-unbiased-amt-reassessment.md`
6. 必要時: readiness doc、`.claude/skills/` の各 SKILL.md

### 役割分担 (第 3 期運用、ユーザー承認は S0 末に確認)

> (2026-07-06 追記) 主担当の読み替え: **Fable 期限 (2026-07-07) までは Fable が主担当**であり、下表の「Opus」行は期限後に有効になる。期限後の Opus は **Opus 4.8** (`claude-opus-4-8`) を指す (4.7 は同価格で fast mode deprecated のため不採用 — memory model-roles 2026-07-06 評価)。

- **Opus**: スプリント計画・統合判定資料 (S4)・設計文書・深い音響分析
- **Sonnet**: 定型実装 (CI 配管・UI・GT 化支援ツール)・スプリント内実装消化
- **Codex (別系)**: 敵対的レビュー役 (ガードレール 12 の監査) に固定起用 — 別系モデルであることが確証バイアス分離に構造的に有利。加えて dual-run の独立再現
- S1 冒頭の受入チェック: 次モデルが (a) 現在地 3 行要約、(b) 直近の硬ゲート状態、(c) 禁止語彙、(d) kill 条件の所在を自力で言えること。失敗なら再計画トリガー発火

## ユーザーの判断・追加調査が必要な部分

| # | 項目 | 必要な判断/調査 | 期限感 |
|---|---|---|---|
| 1 | テスター 3 曲の到着時期・権利確認 | 到着見込み確認 (S2 末トリガー直結)。権利確認の形式 (口頭同意で足りるか、記録様式) | S1 |
| 2 | テスター 3 曲の録音環境メタデータ | **調査**: デバイス/マイク/環境の聞き取り — 較正系 #172-174 の設計入力 | S1-S2 |
| 3 | 用途検証の判定基準 | 「粗い転写で足りる」の操作的定義 (修正時間 X 分・弾き戻し成功率 Y% 等) の事前合意 — 事後解釈の恣意性防止 | S2 前 |
| 4 | per-tine GO/NO-GO 判定材料 | S0 の kill 条件文書 + GO 条件リストへの合意 (数値含む) | S0 末 |
| 5 | #202 方式判定 | モック 4 案の裁定 + 奏法記号 v1 範囲 | 非同期 (S5 前なら効率最大) |
| 6 | ユーザー自身の新録音 | 意思と時期。敵対的メニュー第 2 版で録るか | S2-S4 |
| 7 | 第 3 期 tracking issue 新設 | #199 継続か新設か (本計画は新設を推奨) | S0 |
| 8 | Codex 敵対的レビュー役の固定 | 別系モデル起用の是非 + 監査様式 | S0-S1 |
| 9 | bp-only 23 件 verify セッション設計 | **調査**: 23 件の難易度分布 (即断可能/要精聴) の事前仕分け → セッション分割の妥当性確認 | S2 前 |
| 10 | PESTO 打ち切り閾値 | 盲点重複率いくつ以上で打ち切るかの事前固定 | S6 前 |
| 11 | メタモルフィック警報 cadence | nightly 実行の許容 (ローカル資源) or on-demand のみ | S2 |
| 12 | M1 進入時の公開準備 | M5 data card 草案等を第 4 期に前倒すか | S7 |

## 中長期展望 (M1-M5、第 2 期からの更新)

第 2 期 doc の M1-M5 構造を継承。更新点のみ:

- **M1 (汎化)**: 進入条件「非飽和 held-out ≥5 + 他者録音 ≥1」のうち前者は**到達済み**、後者はきらきら星 2 本 (他者演奏) で実質クリア、テスター 3 曲で確定見込み。**第 4 期は M1 を正面に据える前提**で S7 の次期計画を書く。較正系 #172-174 が M1 の実装本体
- **M2 (streaming)**: A1 が research line に統合されたため、進入条件を「research line の causal 化判断が済んでいること」に更新
- **M3 (browser 単独)**: 変更なし (製品判断待ち)。外部 AMT 不混入方針 (自前 AMT + WASM) が M3 の根拠を強化
- **M4 (出力)**: 弾き戻し検証ループ (S2 開始) が M4 の実測を前倒しで開始
- **M5 (公開)**: 変更なし

マイルストーン間の関係 (継承): M1 が本丸、M4 が製品の顔。M1 の前に M2/M3 へ深入りしない。M5 は M1 の供給源としても機能する。

## 関連

- 第 2 期実績: #199 コメント列 (2026-07-02〜05)
- 敵対的レビュー・新規挑戦提案: 2026-07-05 セッション (要点は decision-log 2026-07-05 エントリに記録)
- 記譜法: #202 / per-note: #141 / 較正系: #172-174 / 認識結果 run 分離: #204 / gt-review energy trace: #205
