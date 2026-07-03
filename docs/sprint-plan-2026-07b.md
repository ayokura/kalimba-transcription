# 中期作業計画 第 2 期 (2026-07-04 起点、8 スプリント + 中長期展望)

- 作成: 2026-07-04 / **状態: active** / 権威戦略 doc: [`research/20260626-unbiased-amt-reassessment.md`](research/20260626-unbiased-amt-reassessment.md)
- 前計画 [`sprint-plan-2026-07.md`](sprint-plan-2026-07.md) (superseded) の S1–S5 完了を受けた全面再設計。旧 S6–S8 は本計画に吸収。
- tracking は引き続き [#199](https://github.com/ayokura/kalimba-transcription/issues/199)。**実績記録は #199 コメントに一本化し、本 doc へのインライン実績追記はしない** (ガードレール 10)。
- 策定入力: 7 本の現状監査 + 発散提案 2 本 + 敵対的批判 1 本 + 4 視点の計画検証 (2026-07-04)。
- 8 スプリントは**最低ライン**。その先は末尾「中長期展望 (M1–M5)」をマイルストーン + 進入条件で保持する。
- 見直しタイミング: Sprint 4 終了時、または再計画トリガー発火時。

## 前計画からの戦略転換 (5 点)

第 1 期の敵対的レビューが突いた核心は「**攻めるべき律速 (非飽和 corpus) だけが空欄のまま、攻められる仕事を精緻に並べていた**」こと。第 1 期 25 コミット中、認識ロジックの前進は #197 の 1 件で、非飽和 GT は 17ea7626 の 1 件から増えなかった。これを正面から直す。

1. **録音は待つものではなく作るもの**: corpus 収集を「横断ストリーム (受動)」から**第 1 レーン (能動)** に昇格。実態: data/transactions 39 dir は sha 重複を除くと **unique 23 録音、うち GT 済み 7 件 (30%)**、repo corpus (CI 網) は 1 件、未判断 unique ≈10 件。この未判断分の消化、敵対的セルフ録音の設計、埋もれ候補の昇格を最優先に置く。**テスター録音の到着分岐は引き続き維持** — 到着すれば常に割り込み最優先。
2. **exit criteria を機構達成から outcome へ**: 各スプリントの exit に「非飽和 held-out で結果が測れたか」を含める。非飽和 held-out の段階目標: **n≥2 (S2) → n≥3 (S3) → n≥5 (S6)** (旧「GT 20 件」固定目標を置換。reassessment §3.3 NOW exit の数値は本計画が supersede し、S1 で履歴追記する)。
3. **patch 追加の凍結を運用化**: events.py の pass 数 (32) と gate reason 数 (~40) が fixture 数 (**35**) に対し、reason は超過・pass は接近 — per-note 移行トリガー 4 の限界域。**新規 suppression pass は #141 research spike 経由のみ** (非 pass 形 — 候補保持/降格 — の改修は従来どおり可)。AGENTS.md の「patch で対処できるケースは patch で」既定方針は S1 で同期改訂する。
4. **browser レーンに運用可能な上限**: 「browser スライスの着手は 1 スプリントに最大 1 本、かつ corpus/recognizer レーンの確度 A 項目が全て着手済みの場合のみ」。#199 のスプリント末報告に browser 関連コミット比率を必須記載し、25% 超過なら次スプリントは browser 停止。
5. **記録の一本化**: 実績は #199 コメントのみ。memory の時点スナップショットは**今日から作らない** (live 状態は gh/#199、memory は evergreen のみ)。S4 の整理バッチまでの暫定運用: handoff への追記は「live 状態は #199 参照」のポインタ 1 行に留める。

## 確度評価の定義 (第 1 期から継承)

| 記号 | 意味 |
|---|---|
| **A** | 要否確定。今の情報だけで着手できる |
| **B** | 実施ほぼ確実。スコープ・実装形・順序に調整余地 |
| **C** | 要否・内容が外部条件依存 (録音到着、先行結果、ユーザー判断) |
| **D** | 方向性の仮置き。再計画で入替・削除の可能性が高い |

運用規則は第 1 期と同じ (スプリント確度 = exit criteria コア項目の確度 / exit 外 B 項目のスリップは自動繰越、2 連続繰越で要否再判定)。

## 全体構図 (レーン再編)

- **レーン 1: corpus / 計器** (最上位・能動化) — 手持ち消化 → 敵対的録音 → GT/baseline 拡充 → 計器の識別力回復
- **レーン 2: recognizer** — 録音非依存分 (#178 構造拡張、ablation 自己監査、dead pass 除去) と、非飽和 n で解禁される gated 分
- **レーン 3: web / UX** — corpus KPI 直撃分 (triage-first review、queue 優先度、intake 安全網) を先行し、出力ターゲット (記譜法 → arrange) を後半に
- **レーン 4: browser / wasm** (上限運用は戦略転換 4) — 小スライス列。attack profile 移植 (大) は gated
- **横断 1: テスター録音到着分岐** (維持) — 到着次第、登録 → review 支援 → 権利確認 (ユーザー) → --to-corpus → --write-baseline → S5 分岐再評価を最優先割り込みで実施
- **横断 2: research bets** — 発散提案からの採用実験 (bets 棚)。見切り条件付き、research branch + dual-run、合計 ≤1sp/sprint 目安

### 人間アクション予算

**1 スプリントあたりユーザー作業 1–2 時間を上限の目安**とし、エージェントが全準備を先回りで整える。依頼はスプリント冒頭に優先順位付きで 1 回提示し、**予算超過時は優先度の低い側から次スプリントに繰り越す**。

| 優先 | ユーザーアクション | スプリント | 準備物 (エージェント) |
|---|---|---|---|
| 1 | 敵対的セルフ録音セッション (~15 分 ×1–2 回) | S1–S2 | 破壊メニュー票を**機械可読な期待シーケンス (演奏予定譜)** として作成 — 録音後は自動整列し、ユーザーは差分確認のみで GT 化できる形にする |
| 2 | 敵対的録音の GT 差分確認 | S2 | 自動整列済み corrections + 差分ハイライト |
| 3 | 手持ち録音の試聴判断 (dedupe 済み優先上位 ~15 件、10 件バッチ ×20 分形式) | S1–S2 | /debug/triage ページ (下記) + 非飽和尤度ランキング |
| 4 | bbd6797f の rights review | S1 | 内容サマリ + 昇格コマンド一式 |
| 5 | issue triage 承認 (close/凍結の一括提案) | S1 | 提案一覧 (根拠付き) |
| 6 | c9e6f699 の GT 作成 (**2–3 セッションに分割可、S3 末まで**) | S2–S3 | 事前 alignment 診断 + 修正候補提示。**local-only** (copyright unknown / consent 未取得のため repo・CI 昇格対象外。権利確認は将来の条件付き行) |
| 7 | 記譜法 v1 の定義 | S6–S7 | 既存記譜規約の調査サマリ + モックレンダリング 3 案 |
| - | (到着時) テスター録音の権利確認 | 随時 | corpus scaffold 一式 |

## スプリント表 (要約)

| Sprint | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| 1 | **計器の修理 1: 手持ち消化** | /debug/triage・非飽和プリスクリーニング・埋もれ候補昇格・敵対的録音依頼・台帳/文書整合 | **A** |
| 2 | **計器の修理 2: 非飽和 n≥2** | 敵対的録音 GT 化・崩壊録音 GT 化・DSP augmentation・FN 再分類 (Mech3 帰着込み) | **B** |
| 3 | **triage-first review + intake 安全網** | needsReview 導出/前出し・#178 構造拡張・queue 暫定ソート・funnel e2e・n≥3 追い込み | **A** |
| 4 | **patch 自己監査 + memory 整理 + 中間レビュー** | ablation observatory (既存トグル範囲)・dead pass 除去・#131 バンドル・memory バッチ・後半再確定 | **B** |
| 5 | **分岐: Mech2/3 (非 patch 形) or A1 causal onset** | 条件二段化 (物理検証+候補保持は n≥2 / 閾値調整は n≥3)。未成立なら A1 (設計判断 + GT 非依存 pin) | **C** |
| 6 | **品質信号: #194 較正 or 代替** | confidence 較正 + queue 本配線、per-event 信号設計、合議 teacher spike (Basic Pitch) | **C** |
| 7 | **出力ターゲット: 記譜法 → 譜面強化** | 記譜法 v1 (ユーザー)・休符/音価・export (PNG/SVG)・stepper/arrange scaffold | **B/C** |
| 8 | **統合 + 次期計画** | browser 残余帰着 (A1 去就込み)・readiness 再評価・bets 去就判定・次期計画 | **D** |

---

## 各スプリント詳細

### Sprint 1: 計器の修理 1 — 手持ち消化 【確度 A】

録音を「待つ」前に、既にある素材と埋もれ候補で非飽和 held-out を増やしにいく。

| 作業 | 確度 | 補足 |
|---|---|---|
| **開発用 試聴トリアージページ (/debug/triage、temporary)** | A | ユーザー発案 (2026-07-04): CLI では音の再生ができず試聴確認の摩擦が高い。既存部品 (review-queue API / audio 配信 / review-status PUT) を束ね、**sha256 dedupe 済み unique 録音**をインライン audio プレイヤー + 自動サマリ (peak dB・イベント数・警告・スペクトル所見) + ワンタップ判定ボタンで 1 ページ化。**ユーザー自身の録音機能も同ページに載せる** (RecorderPanel + createTranscriptionWithCapture 流用) — 破壊メニュー票を表示しながら「その場で録音 → 自動採譜 → 即時判定」を 1 画面で閉じる。**dev 限定・main nav 非リンク・開発が落ち着いたら/本番運用時に撤去** (撤去条件をページ内に明記) |
| **非飽和尤度プリスクリーニング** (エージェント単独) | A | 未判断 unique 全件を現行 recognizer で再採譜し、崩壊シグナル (event 密度異常・旧 response との乖離・warnings) でランキング。既知の有力候補: **2cc06261 (80.5s で events=1、ユーザー memo が破綻を示唆) / 01fc3b8b (32.1s で events=1) / 2772aa11 (103.5s/163ev) / 9a51c1a3 (44.5s/84ev)**。c9e6f699 は「未測定 (飽和の可能性あり)」としてここで優先度判定 |
| bbd6797f の corpus 昇格 (rights review 依頼 → --to-corpus → baseline) | A | 飽和録音 (minF1=1.0) のため**非飽和目標には寄与しない配管作業** — CI 保護網が 1→2 録音になる価値のみ。04fe81e5 は 16b37356 と同一 sha の重複音声のため promote せず、台帳に duplicate 記録のみ |
| 敵対的セルフ録音の「破壊メニュー票」作成 + セッション日程の合意 | A | 既知の弱点 (Mech2 carryover / Mech3 密集 / 弱 attack / 半音隣接 #138 / グリッサンド / ミュート) から逆算。**非飽和録音を意図的に作る = 過適合ゲートを外す唯一の合法手段**。メニューは機械可読な期待シーケンス形式 (GT 化を自動整列 + 差分確認に圧縮) |
| triage 台帳の是正 | A | triage_verdicts.json の stale コメント (#192 録音の F1 記述が #197 修正前のまま) を実測と同期し、台帳に検証時 recognizer fingerprint 列を追加して陳腐化を検出可能にする |
| intake funnel 簡易計測 | B | 期間内の 提出数 / review_completed 数 / promote 数 を data/transactions + review_status 集計で算出し #199 スプリント報告に含める (専用実装なし) |
| issue triage 提案 (close 候補 #127 #52 + stale 11 件 + 凍結漏れ #21 #138 #188) | A | ユーザー承認制。第 1 期 triage と同形式で #199 に提案 |
| **文書整合 (supersession 処理)** | A | (a) AGENTS.md の計画リンクを本 doc へ差し替え、(b) 第 1 期 doc 冒頭に superseded ヘッダ (S1–S5 実績は歴史記録として凍結、S6–S8 は本計画に吸収)、(c) AGENTS.md「Broadband patch vs per-note」節を戦略転換 3 に同期改訂 (トリガー 4 限界域の判断記録込み)、(d) reassessment §3.3 NOW exit 数値の置換を履歴追記、(e) handoff memory 冒頭に「live 状態は #199 参照」ポインタ行を先行挿入 |

**Exit criteria** (2 群に分けて判定):
- **配管**: corpus CI 網 2 録音 / 台帳是正済み / triage 提案が #199 に出ている / 文書整合 (supersession) 完了
- **非飽和進捗の先行指標 (主 outcome)**: プリスクリーニングランキング完成 + GT 候補 ≥2 件の特定 / 敵対的セッションの**実施日が合意済み (または明示的辞退が記録済み)** / 試聴判断が優先上位 ~15 件に付与 (残りは S2 繰越可)

### Sprint 2: 計器の修理 2 — 非飽和 held-out n≥2 【確度 B】

n≥3 の経路を**敵対的録音と backlog 崩壊録音の二重化**で確保する (単一経路依存にしない)。

| 作業 | 確度 | 補足 |
|---|---|---|
| 敵対的録音の登録 → 自動整列 → 差分確認 → GT 化 → baseline 追加 | B | 1 セッションで非飽和 2–3 本が期待値。GT 化コストは S1 の機械可読メニューで圧縮済み |
| 崩壊録音 (2cc06261 / 01fc3b8b 等、プリスクリーニング上位) の GT 化 | B | events=1 の検出崩壊録音は「ほぼ確実な非飽和」かつ GT 作成が軽い (演奏内容の記憶/推定 + 試聴確認) |
| c9e6f699 の GT 作成支援 (該当する場合) | C | プリスクリーニングで不一致が出た場合のみ。2–3 セッション分割、S3 末まで。**local-only** (権利未クリア) |
| DSP augmentation 頑健性マップ (bets #1) | B | 実録音への既知変換 (時間伸縮/残響 IR/ノイズ/gain) で GT を機械導出し「どの変換強度で F1 が崩れるか」の surface を出す。**report-only — 回帰 gate にも過適合ゲート/S5 分岐条件の n にも算入しない** (n は実録音 + 人手 GT レビュー済みのみ) |
| FN taxonomy の再測定 (#197 後・複数録音) | A | 資料間の FN 数不整合 (7/8/12) を解消し、Mech2/Mech3 の対象を機構別に再確定。**Mech3 の帰着先確定 (S5 分岐 or #111 送り) を出力に含める** |
| quality_indicators 較正アンカー修正 + -35dB 分離性検証 | A | -15→-35dB drift の是正に加え、S1 試聴サマリの peak dBFS 分布で -35 閾値の分離性 (unusable と usable の分布) を検証、必要なら閾値提案をユーザー判断へ |
| F1 × fingerprint 時系列 store | B | `--write-baseline` 実行時に append する軽量 hook として最小実装 (非飽和 n が立ってから価値が出る計器の受け皿) |

**Exit criteria (outcome ゲート)**: **非飽和 held-out ≥2** (敵対的録音 or 崩壊録音で達成)。n≥3 は S3 への追い込み分として持ち越し可。**未達の場合は記録ではなく計画変異を自動実行**: (a) 物理合成 bet (bets #6) を即起動、(b) S5 分岐の既定を A1 に確定、(c) #199 に「非飽和 +0 で 2 スプリント経過」を再計画トリガー発火として記録。

### Sprint 3: triage-first review + intake 安全網 【確度 A】

録音非依存で確実に進む recognizer×web の本線。corpus KPI (レビュー完了率・提出率) に直撃する順で。

| 作業 | 確度 | 補足 |
|---|---|---|
| **per-event triage 信号 (needsReview) の導出 + 前出し UI** | A | 現状は候補表示自体は実装済み (SlotCard/次候補チップ)。**実差分**は: needsReview を既存 proxy (alternateGroupings 有無・confidence、隣接 candidateSlot、warnings、gesture==ambiguous) から導出し、EventCard に triage 状態バッジ + 「要確認のみ」フィルタ/前出しビューを追加。DROP_REASON_LABELS を 6 種の drop-reason 全てに拡充 (現 3 種) |
| #178 hard-miss coverage の構造拡張 | B | 候補化経路の追加は録音非依存で進める。**confidence テーブルの値付け・閾値再調整は非飽和 n≥2 成立後に限る** (未成立時は placeholder 値 + TODO で出荷) |
| AlternateGrouping の merge/split サジェスト表示 | B | サーバーが既に出している combinesWith/splitInto を UI で活かす (#16 §4.2 残) |
| /review/queue 暫定優先度ソート | A | candidateSlotCount/warningCount/hasCorrections ベース。GT 不要。**priority_report の拡張はせず queue 側に一本化** (第 1 期 S4 の「母集団不一致」判断を継承し、report は corpus 限定ツールのまま) |
| SimpleHome intake funnel e2e | A | **最小形**: 録音済み blob 注入で採譜 → /score 遷移 + dedup/pending 復元を検証 (マイク許可ダイアログ層はスコープ外)。fake-media 基盤込みで 0.5–1.5 日 |
| イベント前後の loop 再生 (acoustic evidence) | B | #16 §4.3 の最小形。曖昧イベントの判断根拠 |

**Exit criteria**: **要確認イベントが前出しされ、テスターが全走査せずに済む導線が prod にある** / queue が優先度順 / funnel e2e green / **非飽和 held-out ≥3** (S2 からの追い込み完了、未達なら S2 の計画変異が発火済みであること)。

### Sprint 4: patch 自己監査 + memory 整理 + 中間レビュー 【確度 B】

| 作業 | 確度 | 補足 |
|---|---|---|
| ablation observatory 第 1 巡 (bets #2) | B | **既存トグル (feature flag + ablate + disabled_gates 38 種) の範囲で**単独 ablation レポート + dead 判定 — 追加実装ほぼゼロ、全 fixture 一巡 ~40 分。未トグル pass の計装は第 1 巡で「疑わしい pass」に絞って S5 以降。相互作用は単独 ablation で delta が出たペアのみ。実行は on-demand ローカル / nightly (CI per-run に載せない) |
| dead pass の除去 (fixture 非回帰のもの) | B | #171 原則の自動化。pass:fixture 比の改善 |
| pass:fixture 比の CI 可視化 | C | 静的カウントなので軽量。0.8 超で見直しトリガー |
| #131 バンドル移行 | B | observatory が特定した load-bearing 定数を RecognizerSettings へ (単独では価値が薄いためセットで) |
| **memory 整理バッチ** (独立の低リスク作業、1 日) | A | 監査済みチェックリストに**スコープ凍結** (リスト外の stale 発見は #199 にメモして持ち越し): handoff 圧縮 (31KB→<10KB、live 状態は #199 ポインタ化) / project_open_issues.md 削除 + wikilink 張替え / stale 行是正 (旧 trailer 規約・-15dB 記述等) / 統合 3 組 (commit 系 3→1、revert+WIP→1、ablation 許容 2→1) / 34lc・web_rebuild・browser_wasm の陳腐化解消 / MEMORY.md 再構成 (揮発値排除・診断メソドロジーのクラスタ化) |
| **中間レビュー**: 録音状況・observatory 結果・非飽和 n の 3 入力で S5–8 を再確定 | A | |

**Exit criteria**: observatory 第 1 巡レポートが存在し dead pass ≥1 件が除去されている / memory 整理チェックリスト完了 / 後半 4 スプリントが再確定 / (outcome) 非飽和 held-out の現在値と推移が #199 に記録されている。

### Sprint 5: 分岐 — Mech2/3 (非 patch 形) or A1 causal onset 【確度 C】

**分岐条件 (二段化)**: テスター録音の到着は常にこの分岐を再評価させる。
- **物理前提検証 + 候補保持/降格形の改修 (閾値 sweep を伴わない)**: 非飽和 **n≥2** で着手可 (AGENTS.md 過適合ゲートの下限)
- **閾値調整を含む変更**: 非飽和 **n≥3** 成立まで禁止 (批判レポートの n≥3 化は「一律 3」ではなくこの二段化で採用)

分岐内容:
- **条件成立時 — Mech2/Mech3 を「patch」ではない形で**: (a) まず物理前提検証 (energy trace + narrow FFT + gapValidatedOnsetTimes)、(b) 解法は候補保持/降格の拡張、または **per-tine 確率トラッカー spike** (bets #4、#141 research line、research branch + dual-run)。**Mech3 は #111 chord-selector 側の解法候補として同じ分岐で扱う** (S2 の帰着先確定に従う)。新規 suppression pass は不可
- **条件未成立時 — A1 causal onset**: スライスの本体は**設計判断**: post_avg lookahead ≈101ms と目標 ≤50ms の構造的衝突を「post_avg 短縮 (要 retune → n≥3 まで不可)」か「有界遅延を実測値 (~100ms) に緩める」かで確定。**pin は GT 非依存**: causal 版 vs batch 版の onset/event 直接差分 (同一録音・同一 eval_scope) で一致率を pin し、GT-F1 は参考値 (17ea7626 の GT timing は approximate のため)。timing 検証は spectral 確認済みの fixture 層を使う。**A1 の設計判断 + pin テストは browser 枠の計上対象外 (streaming/recognizer 共通基盤)。実装本体と A2 以降は枠内**

**Exit criteria (分岐別・outcome 付き)**: Mech2/3 側 = 非飽和録音での F1/R 改善が baseline 更新で記録 or 不成立の記録 + spike 判定。A1 側 = 遅延設計判断が文書化され、causal 版が batch 版と直接差分で pin されている。

### Sprint 6: 品質信号 — #194 較正 or 代替 【確度 C】

| 作業 | 確度 | 補足 |
|---|---|---|
| #194: quality indicators 較正 (ECE / reliability / flaggedPrecision) + 通過時は /review/queue 本配線 | C | 前提 = 非飽和録音で F1 分散が生じていること (S2–S3 の成果次第)。**通らなければ drop 判断も正当** |
| per-event confidence 信号の設計 (under-detection を捉える slot/timeline-level 信号) | D | S3 の needsReview proxy の裏付けとなる本信号。#16 の triage 表示と接続 |
| 合議 teacher spike (bets #3): Basic Pitch offline 実行 + 現行出力との不一致マップ | C | pseudo-label ではなく**盲点マップ**として開始 (人手 verify 前に completed 昇格しない運用を厳守)。**PESTO は凍結** — Basic Pitch 単独で盲点マップの有効性を先に判定し、有効なら次期計画でライセンス判断 (LGPL-3.0) をユーザーに依頼する |

**Exit criteria**: #194 が「較正完了 / 持ち越し / drop」のいずれかで記録され、不一致マップの有効性 (recognizer の盲点を 1 つ以上指せたか) が判定されている。(outcome) 非飽和 held-out **n≥5** の到達判定と、未達なら阻害要因の記録。

### Sprint 7: 出力ターゲット — 記譜法 → 譜面強化 【確度 B/C】

「認識精度は上がったが何を出すかは曖昧」を解消するスプリント。**記譜法定義 (ユーザー主体) が上流**で、エージェントは調査・モック・実装で従属する。

| 作業 | 確度 | 補足 |
|---|---|---|
| 記譜法 v1 の決定支援 | B | 既存カリンバ記譜規約の調査サマリ + 代表フレーズのモックレンダリング 3 案 (五線/数字タブ/ハイブリッド)。ユーザーが「弾き戻せるか」で判定。奏法記号 (グリッサンド/トレモロ/ミュート) は v1 の範囲をユーザーが決める |
| DoReMiScore への休符・音価 (durationBeat) 導入 | B | schema には既にある。リズム欠如で「直すより諦める」を減らす |
| 譜面 export (PNG/SVG、@media print 整形) | B | テスターに「持ち帰れる成果物」を与える録音インセンティブ |
| capture→transcribe→review→arrange の stepper UI + arrange scaffold | C | #72 判断3。arrange の入力仕様は記譜法 v1 に従属 |
| playability 拘束 spike (bets #5) | D | tine 物理配置で「同時打鍵不可能な和音」を検出。記譜法/arrange と接続する optional。不可能構成 = FP 信号の副産物が本命 |

**Exit criteria**: 記譜法 v1 が決定・文書化 / 譜面に休符・音価が表示される / export が prod で使える。(ユーザーの記譜法判断が出ない場合は譜面強化 + export のみに縮退 — 再計画トリガー参照)

### Sprint 8: 統合 + 次期計画 【確度 D】

- browser レーン残余の帰着確定: B1 attack-profile 移植 (大) の要否 + **S5 が Mech2/3 側だった場合の A1 の去就** (次期送り or 枠内実施)
- free-performance-readiness.md の stage 再評価 (S1–S7 反映) + 次期計画策定
- 全 bets の去就判定 (継続/凍結/破棄) — Mech3/#111 の帰着、物理合成、トラッカー、playability を含む

**Exit criteria**: 次期計画が確定し、全 bets に判定が付いている。

---

## レーン 4: browser / wasm の小スライス列 (S1–S8 に分散)

運用規則 (戦略転換 4): **1 スプリント最大 1 スライス、corpus/recognizer レーンの A 項目が全て着手済みのときのみ着手。** #199 実績にコミット比率を記載し、25% 超過なら次スプリント停止。依存順:

1. kalimba-dsp 衛生 (Cargo.toml メタデータ + **pkg vendoring のスクリプト化** + peak_pick dtype 整合判断) — 0.5–1sp、以降の全スライスの前提
2. wasm-demo への active range フィルタ配線 (要 re-vendoring → 1 の後) — Phase P の表示品質向上、acquisition フックとしての価値も
3. B1 純ロジック群の移植 (short-bridge / dedupe 三種 / sparse_gap_tail body — attack profile 非依存分)
4. (gated) B1 attack profile stack 移植 — slice 1 の数倍の規模。着手判断は S8
5. (gated) A2 incremental onset_strength — A1 完了後、正規化戦略を A1 と共通化

## 横断 1: テスター録音到着分岐 (維持)

到着した瞬間に最優先割り込み: 登録 → 試聴サマリ (/debug/triage) → review 支援 → review_completed → ユーザー権利確認 → `promote --to-corpus` → `--write-baseline` → S5 分岐の再評価。フローは実装済み (第 1 期 S2)。

## 横断 2: research bets 棚

各 bet は research branch + 見切り条件付き。bets が本線工数を侵食しない (合計 ≤1sp/sprint 目安)。

| # | bet | 配置 | 見切り条件 |
|---|---|---|---|
| 1 | DSP augmentation 頑健性マップ | S2 | augment の弱点予測が実新録音の弱点と一致しなければ不変性テスト用途に格下げ |
| 2 | ablation observatory | S4 | 「削除安全」判定が fixture 回帰を起こしたらペア ablation に格上げ or 撤退 |
| 3 | 合議 teacher / 不一致マップ (Basic Pitch) | S6 | 既知 GT での抜き取りで recognizer 単独を上回らなければ破棄 |
| 4 | per-tine 確率トラッカー spike | S5 (Mech2/3 分岐内) | Mech2 を patch 同等以上に拾えず定数も減らなければ凍結 (#141 merge 3 条件) |
| 5 | playability 拘束 | S7 (optional) | 真の和音の不可能誤判定率が拾える FP 数を上回れば error 検出は断念 |
| 6 | モーダル物理合成コーパス | **gated: S2 末時点で非飽和 n<3 なら即起動** | 初期見切り = 17ea7626 の既知 FN taxonomy の機構別再現、#192 型グルーピングバグの再現に成功するか。実録音ランキングとの Spearman ≥0.5 は**非飽和 n≥5 到達後の較正用途昇格条件**として適用 |

## 発散提案 14 件の採否表

| 提案 | 由来 | 判定 |
|---|---|---|
| モーダル物理合成コーパス | Fable 1 | **gated 採用** → bets #6 (S2 末 n<3 で即起動) |
| DSP augmentation | Fable 2 | **採用** → S2 (bets #1) |
| per-tine matched filter 辞書 | Fable 3 | **見送り** — bets #4 (トラッカー) と目的重複。#141 spike の解法候補として吸収 |
| ablation observatory | Fable 4 | **採用** → S4 (bets #2) |
| 合議 teacher / 不一致マップ | Fable 5 | **採用 (縮小形)** → S6 (bets #3、Basic Pitch のみ。PESTO は凍結 — 有効性判定後に次期でライセンス判断) |
| per-tine 確率ベイズトラッカー | Fable 6 | **採用 (spike)** → S5 Mech2/3 分岐内 (bets #4) |
| playability 拘束アレンジャ | Fable 7 | **採用 (optional spike)** → S7 (bets #5) |
| 敵対的セルフ録音セッション | user 1 | **採用 (本線)** → S1–S2 レーン 1 の中核 |
| カリンバ記譜法の定義 | user 2 | **採用 (本線)** → S7 の上流 |
| 他奏者を巻き込む | user 3 | **menu** — M1 (汎化) の進入条件。エージェントは録音依頼キットを要請あり次第準備 |
| コンタクトマイク/ピエゾ実験 | user 4 | **menu** — M1 接続。並行録音プロトコル + A/B ハーネスを要請あり次第準備 |
| テスター構造化インタビュー | user 5 | **menu** — M4 接続。インタビューガイドを要請あり次第準備 |
| 日次 dogfooding 習慣 | user 6 | **menu** — いつでも開始可能な最安の corpus 供給源。diagnosis ダイジェストを要請あり次第準備 |
| データセット公開 / 研究発表 | user 7 | **menu** — M5 接続。data card 草案 + Go/No-go 資料を要請あり次第準備 |

批判レポート是正案の採否: 「録音は作るもの」「outcome ゲート」「patch 凍結」「browser 上限」「記録一本化」= 採用 (戦略転換 1–5)。「S5 分岐 n≥3 一律化」= **二段化で部分採用** (S5 参照)。「/wasm-demo を acquisition フックに」= レーン 4 スライス 2 に反映。「録音ファネル計測」= S1 に簡易計測として採用。

## ガードレール (第 2 期改訂)

1. events.py への**新規 suppression pass の追加禁止**。新規 pass は #141 research spike 経由のみ (非 pass 形 — 候補保持/降格 — は可)。**AGENTS.md 側も S1 で同期改訂**
2. per-tine partial の信念による既定化はしない (#149、継承)
3. NN (MT3 系) の本体置換はしない。teacher / baseline / candidate generator に限定 (継承)
4. F1=1.000 の成功指標化をしない。**進捗は非飽和 held-out での改善のみ** (exit criteria に組込み)
5. Candidate Recall@K の単独 KPI 化をしない (継承)
6. 閾値調整を伴う recognizer 改修は非飽和録音 n≥3 まで行わない (S5 の二段条件として運用)
7. augmentation / 合成データを**回帰 gate に使わない** (report-only)。**過適合ゲート・S5 分岐条件の n にも算入しない** — n は実録音 + 人手 GT レビュー済みのみ
8. pseudo-GT (model_suggested / augment 由来) は人手 verify 前に completed 昇格しない (継承)
9. prod デプロイ cadence: スプリント末に成果を prod 反映 (継承)。レビュー進行中の recognizer 変更デプロイは「レビュー中 tx の corrections 有無を確認してから」の軽量チェックに緩和
10. **実績記録は #199 コメントに一本化** (browser コミット比率の記載を含む)。計画 doc へのインライン実績追記・memory の時点スナップショット作成はしない

## 再計画トリガー

- テスター録音バッチの到着 (→ S5 分岐と S2/S6 の gated 項目を即再評価)
- S2 末で非飽和 n<2 (→ S2 exit 記載の 3 アクションを自動実行: 合成 bet 起動 / S5 既定 A1 / #199 に発火記録)
- 敵対的録音でも非飽和が作れない (認識器が予想より強い) → 計器の定義を見直す
- ablation observatory が patch 衝突 (トリガー 1) を検出 → #141 spike の起動判断を前倒し
- ユーザーの記譜法判断が S7 までに出ない → S7 は譜面強化 (休符/音価/export) のみに縮退

---

## 中長期展望 (Sprint 9 以降 / 3–6 ヶ月のマイルストーン地図)

8 スプリントは最低ラインであり、その先はスプリントに刻まず**マイルストーン + 進入条件**で持つ。確度 D 相当の方向地図で、S8 の次期計画策定時に具体化する。ユーザー主体の挑戦 (採否表の menu 群) はこの地平に置き、固定タスクにはしない。

### M1: 汎化 — 「作者専用チューナー」からの脱皮

- **内容**: 他奏者・他楽器・他録音環境の録音が corpus に入り、per-recording / per-tine 較正 (#172–#174) が実データで駆動できる状態。#33 汎化ロードマップの実体化。
- **進入条件**: 非飽和 held-out ≥5 + 他者録音 ≥1 (menu「他奏者を巻き込む」が鍵)。
- **接続**: ピエゾ/コンタクトマイク実験 (入力チャネル多様化)。録音方式が partial 構造を変える事実 (17-C=mic / G-low・34L-C=pickup) は較正系の設計入力。

### M2: streaming / near-real-time — two-phase の Phase P 完成

- **内容**: A1 causal onset → A2 incremental onset_strength → A3 AudioWorklet spine で「演奏中に暫定譜面」体験を成立させる。確率トラッカー (bets #4) が当たれば Phase P の認識品質自体が上がる。
- **進入条件**: A1 の遅延設計判断が済んでいること。**S5 が Mech2/3 側に倒れた場合の A1 の去就は S8 で確定する** (このマイルストーンが宙に浮かないように)。COOP/COEP 不採用 (single-thread) は再評価トリガー発火まで維持。

### M3: browser 単独 — Phase F の in-browser 化判断

- **内容**: peaks/events/patterns ≈7,400 行の orchestration 移植は multi-sprint 規模。「Phase F = server 維持」が既定で、in-browser 化は**製品判断** (オフライン需要・ホスティングコスト・プライバシー) が立ってから。
- **進入条件**: M2 完了 + 製品判断。B1 attack-profile 移植はこの判断の先行投資 — S8 で要否確定。

### M4: 出力とプロダクトの同一性 — 譜面が「成果物」になる

- **内容**: 記譜法 v1 (S7) → arrange 段の完成 (注記・改行・記譜法バリエーション・エクスポート) → 共有リンク/印刷品質。テスターインタビュー (menu) で「本当の用途」(練習の鏡 / 耳コピ支援 / 演奏アーカイブ) を確かめ、機能優先度を再配線する。
- **進入条件**: 記譜法 v1 決定 (S7)。

### M5: コミュニティ / データセットという出口

- **内容**: rights review 済み corpus の公開ベンチマーク化 (data card + ライセンス)、カリンバコミュニティへの露出、外部貢献者の受け入れ。個人プロジェクトを「作者が飽きたら終わる」状態から脱させる最大の地平シフト。
- **進入条件**: corpus ≥10 録音 + ユーザーの公開判断。corpus が伸び悩む場合は**公開準備 (data card 草案・rights review 基準の文書化・録音依頼キット) の前倒し**を検討する (公開そのものは進入条件を満たしてから)。dogfooding 習慣 (menu) は M5 を待たず始められる最安の corpus 供給源。

### マイルストーン間の関係

M1 (汎化) が最終目標への本丸で、M4 (出力) が製品の顔。M2/M3 (streaming/browser) は体験の質を変えるが、**M1 の前に M2/M3 へ深入りしない** (レーン 4 の上限運用はこの原則の運用形)。M5 は M1 の供給源としても機能する (公開 → 他奏者録音の流入)。
