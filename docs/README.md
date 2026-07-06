# docs/ 索引

新しい agent / セッションはまずここを読む。**現役文書から作業を拾い、凍結・参照文書を設計判断の根拠にしない。** 研究文書の読み順は [`research/index.md`](research/index.md) が正。

## 現役 (現行の計画・運用・設計の権威)

| 文書 | 役割 |
|---|---|
| [`sprint-plan-2026-07d.md`](sprint-plan-2026-07d.md) | **現行計画 (第 4 期、2026-07-07 GO)**。5 計画 × 8 スプリントの直列ストック、active = B0-B4 |
| [`sprint-plan-2026-07d-plain.md`](sprint-plan-2026-07d-plain.md) | 第 4 期計画のやさしい版 (専門用語辞書付き、非技術者向け。正本は技術版) |
| [`decision-log.md`](decision-log.md) | 期をまたぐ戦略判断の追記専用記録 |
| [`research/index.md`](research/index.md) | 研究文書の読み順・deprecated 指定 |
| [`research/20260626-unbiased-amt-reassessment.md`](research/20260626-unbiased-amt-reassessment.md) | 設計判断の権威 (実コード確認済み実装事実 + NOW/NEXT/LATER) |
| [`research/2026-07-per-tine-kill-criteria.md`](research/2026-07-per-tine-kill-criteria.md) | #141/#149 research line の kill 条件 (S0 固定) |
| [`research/2026-07-per-tine-partial-table.md`](research/2026-07-per-tine-partial-table.md) | 実測 per-tine partial table (S3、機械可読 JSON 併設) |
| [`research/2026-07-phase-tracking-roc.md`](research/2026-07-phase-tracking-roc.md) | 位相追跡 onset の ROC 較正結果 (S3、S4 判定材料) |
| [`research/2026-07-per-tine-tracker-design.md`](research/2026-07-per-tine-tracker-design.md) | per-tine tracker + causal onset 統合設計 (S3 集大成、S4 判定材料) |
| [`research/2026-07-s4-gate-materials.md`](research/2026-07-s4-gate-materials.md) | S4 実装ゲート判定資料 (反証 3 系統統合、GO/counter 併記) |
| [`research/2026-07-s4-adversarial-audit-request.md`](research/2026-07-s4-adversarial-audit-request.md) | S4 敵対的監査の依頼パッケージ (別系モデル用) |
| [`research/2026-07-s7-plan-audit-request.md`](research/2026-07-s7-plan-audit-request.md) | 第 4 期計画への敵対的監査依頼 (S7 exit の前提、Codex 用) |
| [`research/2026-07-pertine-round3-residual-decay-replacement.md`](research/2026-07-pertine-round3-residual-decay-replacement.md) | 第 3 巡 (post-stage autopsy): 設計 + §6.5 測定結果 = clean negative で撤退 (#206) |
| [`research/2026-07-pertine-round4-instage-replacement.md`](research/2026-07-pertine-round4-instage-replacement.md) | 第 3 カウント巡設計: 棄却判定の in-stage 置換 (mute-dip → 位相リセット証拠。撤退基準・検証プロトコル事前固定) |
| [`testing.md`](testing.md) | テスト 4 層モデルの詳細・手動テスト手順 |
| [`task-management.md`](task-management.md) | issue ラベル体系・spike タグ書式 |
| [`corpus-management.md`](corpus-management.md) | free-performance corpus の管理・権利レビュー |
| [`fixture-alignment.md`](fixture-alignment.md) | score_structure / alignment_overrides の schema と規則 |
| [`recognition-roadmap.md`](recognition-roadmap.md) | fixture 状態の現況 |
| [`free-performance-readiness.md`](free-performance-readiness.md) | free-performance 評価の現況 (headline 注記付き) |
| [`architecture.md`](architecture.md) | パイプライン概説 |
| [`instrument-layouts.md`](instrument-layouts.md) | tuning 物理配置リファレンス |
| [`broadband-vs-per-note-policy.md`](broadband-vs-per-note-policy.md) | broadband patch vs per-note 方針の詳細 (規範は AGENTS.md) |
| [`browser-two-phase-design.md`](browser-two-phase-design.md) | ブラウザ/WASM の現行設計 |
| [`deploy-cloudflare-tunnel.md`](deploy-cloudflare-tunnel.md) | デプロイ手順テンプレ (host 実値は `.runtime-local/deploy.md`) |
| [`adversarial-recording-menu.md`](adversarial-recording-menu.md) | 敵対的録音メニュー (機械可読版 adversarialMenu.ts が正) |
| [`recognizer-local-rules.md`](recognizer-local-rules.md) / [`deferred-ideas.md`](deferred-ideas.md) | 継続更新の living ledger |

## 参照 (歴史・経緯。凍結ヘッダ付き、設計根拠に使わない)

- 旧計画: [`sprint-plan-2026-07.md`](sprint-plan-2026-07.md) (第 1 期) / [`sprint-plan-2026-07b.md`](sprint-plan-2026-07b.md) (第 2 期) / [`sprint-plan-2026-07c.md`](sprint-plan-2026-07c.md) (第 3 期) — superseded 明記済み
- 旧研究 (LLM バイアスあり deprecated): `research/20260406-*` 3 件 — バイアス警告ヘッダ付き
- 時点記録 (凍結): [`strategy-b-gap-candidates.md`](strategy-b-gap-candidates.md) / [`issue-43-leading-gap-noise-analysis.md`](issue-43-leading-gap-noise-analysis.md) / [`arpeggio-design.md`](arpeggio-design.md) / [`api-contract-design.md`](api-contract-design.md) / [`browser-migration-analysis.md`](browser-migration-analysis.md) / [`per-note-onset-detection-design.md`](per-note-onset-detection-design.md) (#141 入力としては現役)
- 計測履歴: [`performance/`](performance/) (2026-04-15 プロファイル基準)
- 記譜法: [`research/20260705-kalimba-notation-survey.md`](research/20260705-kalimba-notation-survey.md) + [`notation/`](notation/) (S7 決定支援)

## アーカイブ ([`archive/`](archive/))

参照アセット消失・完了済み・環境前提が失われた文書。読む必要が生じるのは考古学的調査のみ:
`app-synth-audio-gap-analysis.md` / `video-expected-performance-analysis.md` (Windows ローカルパス依存) / `wasm-pitch-id-port-plan.md` (完了済み overnight goal 指示書)

---
運用ルール: 新しい文書を追加したらこの索引に 1 行足す。文書を凍結するときは冒頭に 📌 凍結ヘッダを付けてこの索引の分類を移す。
