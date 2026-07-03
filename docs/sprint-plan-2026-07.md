# 中期作業計画: 8スプリント (2026-07 起点)

**作成: 2026-07-02 / 状態: active / tracking issue: [#199](https://github.com/ayokura/kalimba-transcription/issues/199) / 権威戦略doc: [research/20260626-unbiased-amt-reassessment.md](research/20260626-unbiased-amt-reassessment.md)**

進捗・スプリント境界の記録は #199 のコメントで行い、計画本体の改訂はこのファイルを更新する。

## 目的と位置づけ

- 研究再サーベイ (2026-06-26/27) で確定した NOW / NEXT / LATER 方針を、実行可能な 8 スプリントの作業計画に落としたもの。
- スプリント = 約 1 週間相当の作業まとまり。**日付ではなく exit criteria 駆動**で運用する (reassessment §3.3 と同じ流儀)。
- 計画は縛りではなく指針。スプリント間での細分化・追加・入替は前提 (特に確度 C / D の項目)。
- 見直しタイミング: Sprint 4 終了時、または後述の再計画トリガー発火時。

## 確度評価の定義

各作業に「要否の確実性と内容の修正可能性」を A〜D で付す。

| 記号 | 意味 |
|---|---|
| **A** | 要否確定。内容の大幅修正はまず起きない。今の情報だけで着手できる |
| **B** | 実施はほぼ確実。スコープ・実装形・順序に調整余地がある |
| **C** | 要否・内容が外部条件に依存する (録音到着、先行スプリントの結果、ユーザー判断)。大きく変わりうる |
| **D** | 方向性の仮置き。スプリント到達時の再計画で入替・削除される可能性が高い |

運用規則:

- **スプリント確度 = exit criteria を構成するコア項目の確度** (要否・内容の 2 軸のうち弱い方)。表内の個別項目確度はこれと独立に付す。
- **exit criteria 外の B 項目がスリップした場合は次スプリントへ自動繰越**。2 スプリント連続で繰り越したら中間レビューで要否を再判定する。

## 全体構図

3 レーン + 1 横断ストリームで構成する。

- **レーン 1: recognizer / 評価ループ** (本線) — 評価基盤の CI 接続 → #197 → 候補保持 (candidate 降格) → 機構 2/3 → #178 本出力 / #194
- **レーン 2: web / UX** — correction burden 削減、corrections 脆性修正、IA 整理 (#72 / #16)
- **レーン 3: browser / WASM** — parity harness → two-phase 設計 → causal onset / segment 化移植 (研究方針上は LATER。後半に配置)
- **横断ストリーム: corpus 収集** — ユーザー録音 (#18) + review + rights review。**全スプリントを貫く律速要因**であり、後半スプリント (機構 2/3、#194、candidate recall 検証) の成立可否を決める

### 計画全体の前提 (現状ベースライン、2026-07-02 実測)

- main = 04b99ad、pytest 529 passed、prod 3 サービス稼働 (prod=4a9932a、差分は serving code 外 — scripts / docs / tests / fixtures / AGENTS.md / .gitignore のみ)
- F1 corpus 7 録音 micro F1=0.988 (recognizer fingerprint 61da1b4dc730dc32)。非飽和は 17ea7626 のみ (F1=0.851, R=0.769, 12FN)
- repo 管理 corpus は 17ea7626 の 1 件。confidence 指標は無情報 (flaggedPrecision=0.000)
- candidate 系 schema は実装済みが先行: `CandidateSlot` / `candidateSlots` / `AlternateGrouping` は本 response schema に存在 (#151 Layer A/B + #178 soft-reject 実装済み)。未着手なのは **onset-gate 棄却分の候補への載せ込みと review UI 表示**
- レビュー待ち: recorded_only 2 件 (低信号)、c9e6f699 検証待ち

## スプリント表 (要約)

| Sprint | テーマ | 主要作業 | 確度 |
|---|---|---|---|
| 1 | 評価基盤の CI 接続 + 衛生 | corpus regression gate、AGENTS.md/CI 修正、#18 録音依頼 + #72 IA 判断依頼の提出 | **A** |
| 2 | #197 + 残 FN 再測定 | 末尾和音 segmentation gap 修正、baseline 更新、promotion スクリプト化、issue triage 提案 | **A** |
| 3 | 候補保持 (NOW 機構面の完遂) | onset-gate 棄却→候補降格、benchmark への candidate 出力、corrections 脆性修正、e2e CI 配線 | **B** |
| 4 | correction burden UX + 中間レビュー | note picker 等の編集 UX (e2e 操作数で計測)、#72 決定分 cleanup、**中間レビュー** | **B** |
| 5 | browser track 起動 | offline parity harness (segment 比較込み)、two-phase 設計決定、COOP/COEP 方針、B1 着手 | **B** |
| 6 | 機構 2 (carryover) or causal onset | 条件: GT 済み非飽和録音 ≥2 → Mech2 patch 試行 / 未達なら A1 causal peak_pick | **C** |
| 7 | #178 本出力昇格 + #194 較正 | candidate の本出力載せ込み + review UI 表示 (録音非依存)、quality indicators 再較正 (録音条件付き) | **B/C** |
| 8 | 統合・次期計画 | #16 実装第 1 弾 or browser 続行 (A1→A2 依存順)、readiness 再評価、次期 8 スプリント立案 | **D** |

---

## 各スプリント詳細

### Sprint 1: 評価基盤の CI 接続 + 衛生 【確度 A】

以降のすべての recognizer 作業の安全網と測定基盤を作る。研究方針 NOW の「benchmark CI 配線」に相当。

| 作業 | 確度 | 補足 |
|---|---|---|
| free-performance corpus の pytest regression gate 新設 (per-recording baseline 比較、audio 実走) | A | git-tracked corpus (現在 1 件) を対象。gate は F1==1.0 ではなく **per-item baseline (17ea7626: F1≥0.851 等) + hardMiss 上限**。gate に使う F1 は **tolerance-aware な既存メトリクス** (GT timeSec は近似のため note identity/order 中心、timing assert は緩く)。ローカル専有 6 録音は CI 対象外のため、**ローカル実行用 baseline 突合コマンドを同時提供**し recognizer 変更時の手動実行を運用に組み込む |
| baseline artifact 形式 + 更新規律の docs 化 | A | 「baseline 更新は改善方向のみ。低下時は regression として扱い、意図的 tradeoff は fixture policy 準拠で記録」を明文化 |
| corpus metadata governance test (rightsReview=approved_for_repository、必須ファイル存在) | A | corpus-management.md の要件を機械検査化。~0.5 日 |
| docs/testing.md + AGENTS.md Test Architecture 節に第 4 層「Corpus benchmark regression」を追記 | A | 3 層→4 層。第 4 層の assertion 権威は per-recording baseline ファイルである旨を **AGENTS.md 側にも** 1 文追加 (testing.md だけだと shared source of truth と矛盾する) |
| AGENTS.md「Research References」節の全面書き換え | A | リンクを 20260626-unbiased-amt-reassessment.md へ差し替えるだけでなく、**節内の設計前提 bullet (onset gate / 状態遷移モデル等) も reassessment §1 の実装事実ベースに置換** (リンクだけ直しても stale な前提が残る) |
| CI (test.yml) の `uv python install 3.12` を 3.14 に修正 | A | pyproject `>=3.14,<3.15` との矛盾解消 |
| **#18 録音依頼バッチ + #72 IA 判断依頼をユーザーへ提出** | A | **計画全体の律速**。録音依頼には mixed/strict/slide/arpeggio-like + 環境多様化 + calibration サンプル (acoustics survey 参照) に加え、**#194 用の「意図的に難しい free-performance」数本 (速い / 弱 attack / 多声)** を明記。rerecord fixture (four-note-repeat-02) の扱いも同梱で判断を仰ぐ。#72 IA 判断依頼も S4 を待たずここで提出しリードタイムを確保 |
| 過適合ゲートの明文化 (AGENTS.md Recognizer Strategy Notes) | A | 「閾値調整を伴う recognizer 改修は GT レビュー済みの非飽和録音 2 件以上を条件とする。構造的欠陥修正 (#197 型) はこの限りでない」 |
| #161 (pending fixture assertion) の方針決定 | B | **判断済み (2026-07-02): 必要になるまで pending** (自由演奏拡充で pending fixture が再発しうるため issue は open 維持、enforce 実装は保留) |
| 【監査反映】benchmark provenance に `kalimba_dsp_fingerprint()` を併記 | A | 2026-07-02 の 53 コミット監査で判明。Rust だけ変えた 2 つのベンチ結果が同一 fingerprint になる。baseline artifact 設計と同時に修正 |
| 【監査反映】promote スクリプトの SHA dedup を free-performance-corpus 層まで拡張 | A | 同監査。同一音声の再アップロードでベンチが二重カウントする穴 |
| 【監査反映】review_completed × 未保存 corrections のガード (web) | A | 同監査の最重要 workflow 欠陥: dirty 状態で review_completed を確定でき、古い corrections が GT 昇格対象になる |
| 【監査反映】doc drift 一括修正 | A | roadmap E83 stale bullet / #162 closed 未反映 (roadmap+readiness) / readiness の「6録音 F1=1.000」stale / research index「次にやるなら」実装済み / testing.md の実在しない test_gap_filter 引用 |

**Exit criteria**: corpus gate が CI で green / baseline artifact が commit 済み / 録音依頼 + IA 判断依頼がユーザーに届いている。

### Sprint 2: #197 実装 + 残 FN 再測定 【確度 A】

唯一の即着手可能な recognizer スライスを、Sprint 1 で作った安全網の上で実装する。

| 作業 | 確度 | 補足 |
|---|---|---|
| #197: trailing final chord segmentation gap 修正 | A | broadband onset 4 本 gapValidated 済みなのに segment 未形成。#153 Phase B lookback は pre-segment 専用で末尾未カバー。12FN 中 4FN の回収を期待 (期待値であって exit 条件ではない) |
| 全 fixture + F1 bench で検証 → baseline 更新 (改善の lock-in) | A | 既存 completed fixture の exact-match 非劣化が gate |
| **#197 の prod 反映は Sprint 3 の corrections 脆性修正後まで保留** | A | 進行中レビュー (c9e6f699 等) の corrections が再採譜の timeSec ずれで壊れるのを防ぐ (ガードレール 8 参照)。CI/baseline 更新は S2 内で完了してよい |
| 17ea7626 残 FN の機構別再評価 | A | #196 の Mech2 (carryover) / Mech3 (密集誤選択) の難度・分布を #197 後の状態で再測定し、Sprint 6 以降の入力にする |
| promote_corrections への repo-corpus promotion 追加 (`--to-corpus` 相当: metadata scaffold + rights review 記録欄) | B | 人手を権利判断のみに減らす。corpus 拡充の反復コスト削減 |
| issue triage: 全 open issue の sweep → **close-rescope 提案リストを作成しユーザー確認後に実行** | B | 「現戦略との整合」でラベリング: alive / gated-by-trigger (#145 #146 は #141 トリガー配下、#140 は browser track 前提) / deferred / close 候補 (2026-03 停滞 design 系、実装済み確認済みの #69 等)。一括 close は方向性判断のためユーザー承認を挟む |

**Exit criteria**: #197 の segment 形成 gap が解消され 17ea7626 末尾 onset cluster (14.66–14.86s) を覆う segment が形成・候補評価されている / 全 fixture exact-match 非劣化 / baseline が改善方向に更新済み / 残 FN の機構分類が更新済み。

### Sprint 3: 候補保持 — NOW 機構面の完遂 【確度 B】

研究方針 NOW (b)「棄却を drop から候補降格に統一」と、#178 の計測先行を行う。NOW exit そのもの (自由演奏 GT≥20 でのベースライン確立) は corpus ストリーム側で継続であり、本スプリントで完了するのは機構面のみ。

| 作業 | 確度 | 補足 |
|---|---|---|
| onset-gate 棄却の「低 confidence candidate slot への降格」統一 | A | **検証済み・実装済みと確定 (2026-07-02)**: primary 棄却 (onset-gate-no-evidence 含む) → dropped-segment candidate slot (drop reason 別 confidence)、secondary 棄却 → soft alternates、が #178 Phase 1/2 で既に本 response schema に載っている。17ea7626 で実証 (slot 5 件 + alternates 4 件、FN の C5@11.547 / C5@6.635 / E4@14.784 が候補として観測可能)。**残スコープ = hard miss 7 件の coverage 拡大 + review UI 表示 → S7 (#178)** |
| rankedCandidates / droppedSegments の benchmark JSON 出力 | A | **実装済みと確定 (ab2cca3)**: collect_surface_candidates (candidateSlots + alternateGroupings) + collect_debug_ranked_candidates が benchmark に既存。candidateSources 集計も出力済み |
| benchmark の mir_eval 互換 onset-only F1 (±50ms / 50cents) + bootstrap CI (B=1000) 追加 | B | evaluation survey §2 準拠。**外部比較用の報告値のみで gate には使わない** (GT timing が近似のため)。小 corpus での改善判定が noise chasing になるのを防ぐ |
| corrections↔再採譜の timeSec ±0.005s 突合脆性の修正 | B | recognizer が改善するほど review 途中の corrections が壊れる隠れ依存 (reviewCorrections.ts の TIME_MATCH_TOLERANCE_SEC=0.005)。sourceEventId ベース突合 or tolerance 拡大。**完了後に #197 を prod 反映** |
| Playwright e2e の CI 配線 + corrections round-trip テスト追加 | B | UI 改修スプリント (4, 7, 8) の安全網。webServer 設定済みで配線コストは小 |

**Exit criteria**: 棄却イベントが候補として benchmark で観測できる / corrections round-trip e2e が timeSec 摂動 (±50ms 相当) を模した再採譜に対して green / #197 が prod 反映済み。

### Sprint 4: correction burden UX + 中間レビュー 【確度 B】

recognizer 非依存で今すぐ効く UX 改善。**スプリント末に計画全体の中間レビュー**を行う。

注意: benchmark の correction cost (costPerTruthNote) は recognizer 出力 vs GT の意味的編集コストであり、**UI 操作性の改善では動かない**。UX 改善の効果は e2e 上の操作数で測る (recognizer 側の効果測定と混同しない)。

| 作業 | 確度 | 補足 |
|---|---|---|
| カリンバ鍵盤レイアウト型 note picker (ドロップダウン置換) | B | 編集 1 操作あたりの手数削減 |
| 音のワンタップ置換 / 既存 event の timeSec 微調整 / redo | B | 現状: 置換 = 削除+追加の 2 操作、timeSec 編集不可、redo なし |
| 代表修正シナリオの操作数計測 (Playwright e2e で before/after) | B | FP 削除 / FN 挿入 / 音高置換の 3 シナリオ。S3 で配線した e2e 基盤を利用 |
| /review/queue への review_priority_report スコア統合 | C | 「次にレビューすべき録音」の提示。corpus 拡充の人的効率化。優先度はレビュー待ち件数次第 |
| #72 IA 判断の決定分 cleanup 着手 | B | **判断済み (2026-07-02)**: (1) orphan /review stack は削除 (useAudioPlayback 等の再利用部品は保持)、(2) Capture Pack ZIP は keep (将来のアップロード/デバッグ用途)、(3) 段階分割 UI (capture→transcribe→review→arrange) を目指す — arrange は入力を決めれば独立検討可能。TranscriptionStudio dead code 除去と合わせて S4 で実施 |
| 新録音の登録 / review 支援 / GT 昇格 (到着次第、以降毎スプリント継続) | C | 録音到着に依存。recorded_only 2 件と c9e6f699 の処理も含む |
| **中間レビュー: Sprint 5-8 の再計画** | A | 録音到着状況・#197 後の残 FN・IA 判断 (あれば) の 3 入力で後半を確定させる |

**Exit criteria**: 代表修正シナリオの操作数削減が e2e 計測で確認できる / UX 改修が prod 反映済み / 後半 4 スプリントが再確定している。

**実績 (2026-07-03 完了)**:

- **#72 cleanup 完了**: TranscriptionStudio user-mode dead code 除去 + ScoreViewer warnings 表示 (8af0c72)、orphan /review stack 削除 1,182 行 (1181c76)。useAudioPlayback / reviewSession / reviewAudioStore はユーザー指示どおり部品として温存
- **UX 改修完了** (e07901f): KalimbaNotePicker (Range Map の review 移管)、ワンタップ置換 (replaceNote reducer)、timeSec ±10/50ms 微調整、redo。操作数計測: **音高置換 3→2 操作** (e2e review-edit-ops、before=2aaa13c)、FP 削除 2 / FN 挿入 1 (据え置き)
- **割り込み対応**: 低音量警告閾値 -15→-35 dB 緩和 (01bf136、ユーザー指示。適正値の再較正は後続)
- **/review/queue への priority スコア統合は defer**: 調査の結果、review_priority_report のスコアは ground_truth 必須で、live queue の母集団 (data/transactions、GT 0 件) とほぼ交差しない。載せても大半が「スコアなし」になり、算出も per-recording フル再採譜でレイテンシ非現実的。**将来形は GT 不要の quality_indicators.py (実装済み・未配線・PROVISIONAL) を list_review_queue 側に配線する案** — 指標検証 (#194) が先行条件なので S7 の #194 判断と束ねる。priority_report 自体は corpus 限定のベンチマークツールとして現行運用を維持
- prod 反映済み (prod=e07901f、api/web/tunnel active、health 200)。中間レビューは下記「中間レビュー (2026-07-03)」参照

### 中間レビュー (2026-07-03, Sprint 4 末)

3 入力の確認結果と後半 4 スプリントの確定:

1. **録音到着状況**: 新着なし。recorded_only 2 件 (1955b5bd / 98019f67、低信号) と c9e6f699 検証待ちは継続。bbd6797f (review_completed) は fixture 層に GT 済みで新規性なし。**GT レビュー済み非飽和録音は依然 17ea7626 の 1 件のみ**
2. **#197 後の残 FN**: 変化なし (S4 は recognizer 非接触)。#196 の分類どおり Mech2 = 8 FN、Mech3 = 13.323s D5/F5 FN + E6 FP
3. **IA 判断**: #72 判断 1-3 とも S4 で反映完了。追加の IA 判断は未発生

**判定: S5-8 は計画どおり継続 (構成変更なし)**。ただし:

- **S6 分岐は現状のままなら条件未成立 (非飽和 GT 1 件 < 2 件) → A1 causal onset 側が本命見込み**。S6 着手時に再判定し、それまでに録音 + レビューが進めば Mech2 patch 側に切替
- S7 の #194 較正判断に「quality_indicators の queue 配線 (S4 defer 分) を較正通過時のセット項目とする」を追加
- S5 の wasm ビルドチェーン (rustup target 追加 + CI ジョブ追加可否) はユーザーアクション待ちのまま — S5 着手時に依頼を再提示する

### Sprint 5: browser track 起動 【確度 B】

研究方針では LATER だが、録音待ちで recognizer 本線が細る時期の並行レーンとして起動する。**精度に関わる移植より先に、検証基盤と architecture 決定を置く** (browser survey の推奨順)。

| 作業 | 確度 | 補足 |
|---|---|---|
| A0: browser offline parity harness (固定 WAV を browser/TS 処理し Python fixture と突合) | B | B 系全スライスの検証基盤として最初に作る。**onset 比較に加え segment-level 比較モード (segment 境界 / active range / discard 判定の Python 中間データとの突合) を含める** — B1 の検証に必須 |
| two-phase architecture (browser preview + batch finalize) の設計決定 | B | patterns.py の repeated_pattern_passes() に配線された full-batch normalizer 3 pass (+未配線 1、no-op 1) は streaming と構造的非互換 → two-phase 化は必須。**target SR の決定と #140 (SR 依存パラメータ) の扱いも設計入力に含める** (browser ライブ経路 48k vs fixture 44.1k/96k) |
| COOP/COEP 方針決定 (SharedArrayBuffer / multi-thread WASM の前提) | B | browser survey §7 の明示警告。Next.js app への影響検証。single-thread 退避策も確認 |
| B1 着手スライス: segments.py (1,354 行) の segment 化ロジック移植のうち **active range 計算を parity harness で pin するまで** | C | 着手条件 = A0 の segment 比較モード完成。TS 移植 vs Rust 共有コア拡張の方針決定込み。**B1 残余の帰着先は中間レビュー / S8 / 次期計画で明示する** (本計画内で B1 完遂は保証しない) |
| wasm ビルドチェーン整備 (check_wasm.sh の CI 配線、vendoring 手順) | C | rustup target 追加はユーザー実行 (installer 運用ルール)。CI ジョブ追加コストの承認もユーザー判断。**監査反映: parity ハーネスに実 fixture 音声 + 「audio→onset_strength→onset_detect」通し比較を追加** (現行は native 生成 envelope を両者に食わせるため FFT 段の wasm/native 乖離を検出できない)。Rust peak_pick の窓平均 dtype 整合 (numpy f32 累積 vs Rust f64 累積、frame-exact 主張の破れ) もここか S2 で判断 |

**Exit criteria**: parity harness で browser/Python の onset + segment 出力差分が自動測定できる / two-phase 設計 (SR 方針込み) が文書化されている。

**実績 (2026-07-03, exit criteria 達成)**:

- **A0 完了** (5c86860, b75f1e6): 実 fixture WAV 4 本 (44.1k / 48k / 96k) で
  wasm 通し経路 (audio→onset_strength→onset_detect) を native + 独立 numpy オラクルと突合。
  onset frame は 3 SR とも frame-exact 一致 (監査指摘の FFT 段 gap を閉塞)。check_wasm.sh 5 段構成
- **wasm CI ジョブ追加** (ed3f2d3、ユーザー承認): test/web と独立の別ジョブ。rustup target はユーザー実行済み
- **two-phase 設計文書化** (e8cb355 → docs/browser-two-phase-design.md): Phase P/F 分離、FFT の wasm 一本化
  (AnalyserNode 不採用)、browser SR=48k 固定 + #140 現状維持 (過適合ゲート充足まで)、COOP/COEP 不採用 (再評価トリガー付き)
- **B1 スライス 1 完了** (e9f90c3): active range 計算 (rms 閾値 + 走査 + merge) を Rust 共有コアに移植、
  実装言語判断は Rust 拡張で確定。wasm 出力を実 fixture の detect_segments debug と実差分比較 (24/24)、
  pyo3 側は test_segment_dsp_rust.py で差分検証。**B1 残余** (short-bridge 抑制 / 境界生成 / collector /
  discard — attack profile 系への依存が深い) **の帰着先は S8 or 次期計画で確定** (計画どおり本計画内での完遂は非保証)
- **/wasm-demo 試聴検証 合格** (2026-07-03 ユーザー実施): scale 17 音 + mixed 6 イベント (和音 top-1 込み) が
  音名・順序とも一致。デモ側の余分 onset はすべてサーバー生 onset 列にも存在する実音 (active range フィルタで
  除去される類) で、wasm 偽検出ゼロ — two-phase 設計 (Phase P = raw preview) の前提を実証。
  次スライス候補: B1 の raw_active_ranges を wasm-demo に配線して Phase P でも範囲外 onset を落とす

### Sprint 6: 機構 2 (carryover) or causal onset 【確度 C】

**分岐スプリント**。実施自体は確定 (どちらかの分岐は必ず実施)、内容が外部条件依存のため C。

分岐条件 (単一判定): **S6 着手時点で GT レビュー済みの非飽和録音が 2 件以上あるか** (rights review は repo 収載時のみ必要 — benchmark 評価は data/transactions のローカル GT で可)。

- **条件成立時**: Mech2 = carryover vs re-attack 判別の patch 試行。C5@11.55s 型 (fresh attack 49→1278 が直前 B4 carryover ≈4000–9400 に飲まれる) を対象。
- **条件未成立時**: A1 = causal peak_pick+backtrack (有界遅延 ≤50ms) の Rust 実装 + batch 版と F1±2% pin。LATER 項目の前倒し。

| 作業 | 確度 | 補足 |
|---|---|---|
| Mech2 patch 試行 (条件付き本命) | C | 単一録音相手の閾値調整は過適合ゲートに抵触するため、録音 2 件以上が前提 |
| #141 research spike (patch 試行 2-3 回失敗時) | C | **spike 起動 (research branch, dual-run) は #141 consensus の枠内で実施可**。ユーザー判断が必要なのは **main への merge 判断 / umbrella 再オープン時** (merge 3 条件 = exact-match 非劣化 + 自由演奏指標改善 + suppression pass 削減)。branch prefix は `claude/` |
| A1 causal onset (fallback) | C | backtrack の lookahead 依存で onset 時刻が系統的にずれる可能性 → ground_truth timing assertion との接触に注意 |

**Exit criteria (分岐別)**: Mech2 側 = 非飽和録音での F1/R 改善が baseline 更新で記録されている、または patch 失敗の記録 + #141 spike の結果整理 (merge 判断が必要ならユーザーに提出済み)。A1 側 = causal 版が batch 版と F1±2% で pin され parity harness に記録済み。

### Sprint 7: #178 本出力昇格 + #194 較正 【確度 B/C】

**#178 の実装は録音非依存** (schema は実装済みで、レビュー待ち録音という表示先も既にある) — 前提条件なしで着手できる。録音が必要なのは Candidate Recall による効果検証と #194 のみ。

| 作業 | 確度 | 補足 |
|---|---|---|
| #178: onset-gate 降格分・dropped segments の本出力への載せ込み + review UI 表示 | B | NEXT の中心 (issue 本文 Phase 1 の完遂)。schema (candidateSlots / alternateGroupings) は実装済みなので、スコープは**候補充填カバレッジの拡大と UI 表示**。#16 の event-first correction workflow と接続 |
| Candidate Recall による #178 効果検証 | C | corpus 拡充が前提。未達なら検証のみ録音到着後へ持ち越し (実装は先行してよい) |
| #194: quality indicators 再較正 (ECE / reliability / flagged precision) | C | 前提 = harder 録音が corpus に入り F1 分散が生じていること。較正自体は小さい (~0.5 sp)。**検証が通らなければ drop の判断も正当** (issue 本文に明記済み)。**較正通過時は quality_indicators の /review/queue 配線 (S4 defer 分) をセットで実施** |
| per-event confidence 信号の設計 (under-detection を捉える slot/timeline-level 信号) | D | event-level flag では under-detection を構造的に捉えられない。carryover-rejection 密度 / segment 形成 gap 等の新信号 emit が必要。別スライスとして扱う |
| PESTO 検証 spike (research line) | D | streaming VQT + ONNX + 自己教師あり (130k params) で唯一の有望外部コンポーネント。monophonic なので pitch id の第 2 意見と位置づけ。**LGPL-3.0 のライセンス採否判断 (ユーザー) が先**。1 スプリント検証で捨てる覚悟 |

**Exit criteria**: candidate が本出力/review UI に現れている (録音非依存で判定可能) / #194 は「較正完了」「録音待ち持ち越し」「drop」のいずれかの判断が記録されている。

### Sprint 8: 統合・次期計画 【確度 D】

ここは方向性のみ。Sprint 4 の中間レビューと Sprint 6/7 の結果で内容が決まる。

| 作業 | 確度 | 補足 |
|---|---|---|
| #16 review/repair 再設計の実装第 1 弾 (event-first correction workflow) | D | UX サーベイで方向性確定済みだが、スコープ膨張リスク大 (設計史が長い)。スライス厳守 |
| browser track 続行: 未実施スライスを依存順に (B1 残余 → A1 → A2 incremental onset_strength → A3 AudioWorklet spine) | D | **A2 は A1 完了が前提、A3 は精度に寄与しないので parity gate 後**。S6 で A1 済みなら A2 から |
| NN teacher 配線の判断 (Basic Pitch / PESTO を offline teacher に、method: "model_suggested") | D | corpus 拡充が回っていることが前提。人手 verify 前に completed 昇格しない運用ルールを先に決める |
| readiness 再評価 + 次期 8 スプリント計画 | B | 本計画の後継。free-performance-readiness.md の stage 評価更新と B1 残余の帰着先確定を含む |

---

## 横断ストリーム: corpus 収集とユーザーアクション

計画期間を通じて回し続ける。**律速はすべて人間作業**なので、エージェント側は依頼・支援・即時処理に徹する。

**デプロイ cadence**: 各スプリント末に、そのスプリントの成果 (recognizer 改修 / UX 改修) を `.runtime-local/deploy.md` の手順で prod (score.ayokura.net) に反映し smoke 確認する。corpus 収集がテスターの prod 利用に依存するため、prod 未反映の改善は corpus ストリームに効かない。順序制約はガードレール 8 を参照。

| ユーザーアクション | 対応スプリント | 影響 |
|---|---|---|
| #18 新録音 (mixed / strict / slide / arpeggio-like + 環境多様化 + **#194 用の意図的に難しい free-performance**) | S1 で依頼 → 随時 | **単一最大 unblock**。S6/S7 検証の成立条件。NOW exit (GT 20 件) のボトルネック |
| recorded_only 2 件 (1955b5bd / 98019f67) の試聴・レビュー判断 | 随時 | 低信号 (peak −17.9 / −21.5 dB) のため usable 判断から |
| c9e6f699 (G-low 162 events) の 7 mismatch 検証 | 随時 | corpus 8 件目 |
| corpus 昇格の rights review (録音ごと) | 随時 | repo 収載の必須人間 gate (corpus-management.md) |
| #72 IA 判断 (orphan /review、Capture Pack ZIP、workflow 分割) | **S1 で依頼** → S4 で反映 | web 系 cleanup の unblock |
| issue triage の close-rescope 承認 | S2 | 一括 close は方向性判断のためユーザー確認を挟む |
| pending fixture reclassify の許可 (#161) | S1-S2 | enforce 方針決定の前提 |
| #141 の merge 判断 / umbrella 再オープン (spike が成果を出した場合のみ) | S6 以降 | spike 起動自体は consensus 枠内でエージェント実施可 |
| wasm ビルドチェーン導入 (rustup target 等、installer はユーザー実行) + wasm CI ジョブ追加の可否 | S5 | browser track の前提 |
| /wasm-demo 音名の試聴検証 (browser track の人手 oracle) | S5 前が望ましい | 移植の正しさの最終確認 |
| PESTO の LGPL-3.0 ライセンス採否判断 | S7 | 検証 spike の前提 |

## ガードレール (やらないこと)

reassessment §3.4 の「やらない方がよいこと」+ 本計画で追加した運用ゲート。

1. **events.py への新規 suppression pass の追加継続** — 限界収益低下。候補保持 / 降格 / provenance 方向に振る
2. **per-tine partial の信念による既定化** — research spike / ablation 限定 (#149)
3. **NN (MT3 系) の本体置換** — teacher / baseline / candidate generator に限定
4. **F1=1.000 の成功指標化** — tuning-set 飽和のサイン。非飽和録音での改善のみを進捗と数える
5. **Candidate Recall@K の単独 KPI 化** — 候補乱発で水増し可能。Correction Burden / slots/event / HardMissRate と対で監視
6. **閾値調整を伴う recognizer 改修を非飽和録音 1 件で行う** — 過適合ゲート (Sprint 1 で AGENTS.md に明文化)
7. **baseline の下方更新で CI を通す** — baseline 更新は改善方向のみ。意図的 tradeoff は fixture policy の手続きで
8. **進行中レビューがある状態で recognizer 変更を prod 反映する** — corrections の timeSec 突合が壊れる (S3 の脆性修正が入るまでは特に厳守)
9. **UI 操作性の改善を benchmark の correction cost で主張する** — costPerTruthNote は recognizer 出力の指標。UX 効果は e2e 操作数で測る

## 再計画トリガー

以下のいずれかが発生したら、スプリント境界を待たずに計画を見直す。

- **録音バッチ到着** — S6/S7 の条件が変わる (前倒し方向)
- **#141 移行トリガー発火** (patch 衝突 / 物理的検出不能 / streaming 要求 / patch 数 ≈ fixture 数) — research line の比重を上げる
- **#194 の drop 判断** — confidence 系の後続作業 (S7 の信号設計) を削除
- **corpus gate の想定外 regression** — 原因調査を最優先に差し込む
- **#72 IA 判断の大幅な方向転換** — web レーンの S4/S8 を組み替え

## 関連

- 権威戦略: [research/20260626-unbiased-amt-reassessment.md](research/20260626-unbiased-amt-reassessment.md) / [research/index.md](research/index.md)
- 評価設計: [research/20260626-amt-evaluation-survey.md](research/20260626-amt-evaluation-survey.md)
- browser 実装順: [research/20260626-browser-realtime-implementation-survey.md](research/20260626-browser-realtime-implementation-survey.md)
- fixture 現況: [recognition-roadmap.md](recognition-roadmap.md) / readiness: [free-performance-readiness.md](free-performance-readiness.md)
- corpus 統治: [corpus-management.md](corpus-management.md) / テスト構造: [testing.md](testing.md)
- 主要 issue: #18 (corpus) / #196・#197 (17ea7626) / #178 (multi-candidate) / #194 (品質指標) / #141 (per-note umbrella) / #72・#16 (web IA/review)
