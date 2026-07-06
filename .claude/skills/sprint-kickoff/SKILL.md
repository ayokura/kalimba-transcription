---
name: sprint-kickoff
description: セッション/スプリント開始時の定型キックオフ。固定読み順で文脈を復元し、受入チェックに答え、live issue を動的取得して着手候補と blocker を提示する。「キックオフして」「セッション開始」「現状把握から始めて」で起動。
---

# Sprint Kickoff

新セッション・新スプリントの開始手順を定型化する。目的は 2 つ: (1) 引き継ぎ資産だけで文脈を復元できることの検証、(2) ユーザーが毎回キックオフ指示文を書かなくて済むこと。

## 手順

### 1. 固定読み順で文脈復元

memory の handoff-next-session (次セッション再開ポインタ) が指す現行ポインタに従う。2026-07 時点の読み順:

1. `AGENTS.md` (自動ロード済みのはず — 未読なら読む)
2. `docs/README.md` (docs 索引 — 現役文書の把握)
3. 現行 sprint plan (handoff memory が指すもの。例: 第 4 期 `docs/sprint-plan-2026-07d.md`)
4. live issue の最新コメント数件 (handoff memory が指す記録一本化 issue)
5. `docs/decision-log.md` の直近 3 エントリ
6. research line が動いている場合: その kill 条件文書 + `docs/research/20260626-unbiased-amt-reassessment.md` (per-tine の kill-criteria doc は 2026-07-06 本線化で役目終了 — 歴史資料)

### 2. 受入チェックに答える

読み順を終えたら、以下を**自分の言葉で**簡潔に報告する (コピペではなく理解の証明):

- **現在地 3 行**: どのスプリントの何が終わっていて、次は何か
- **硬ゲート状態**: スプリント境界の GO 待ちか、スプリント内継続か (ガードレール 11)
- **報告語彙の確認**: headline は非飽和限定 micro F1 + CI95 のみ (pooled micro 禁止) を認識しているか
- **kill 条件の所在**: active な research line があればその kill 条件文書の場所と判定状態。なければ「現在 research line なし (per-tine は 2026-07-06 本線化済み)」と言えること
- **(第 4 期) 計画ストック構造の理解**: 4 計画 (A-D) のどれが active か、計画レベル硬ゲート (ガードレール 15) の認識

### 3. live issue の動的取得

```bash
gh issue list --state open --limit 30
```

対応表を memory に持たない (stale 化するため)。毎回この一覧を取り、handoff / sprint plan の記述と突き合わせる。

### 4. 着手提案

- 着手候補を優先度付きで列挙し、各候補の blocker (人間の裁定・録音・GO 待ち等) を明示する
- blocker が人間側にある項目は memory の human-action-items に集約されているか確認する
- 停止 6 類型 (feedback-batch-queue-mode) を再確認し、それ以外では止まらず進める前提を宣言する

## 注意

- スプリント境界は人間の明示 GO でのみ跨げる。キックオフ時に「前スプリント完了 → 次を開始してよいか」を自己判断しない。GO 待ちなら非同期継続作業のみ列挙する。第 4 期はさらに**計画 (A-D) の開始・切替・凍結もユーザー GO** (ガードレール 15)。
- このスキルの読み順・受入チェック項目が handoff memory と食い違ったら、handoff memory 側が正 (このファイルの更新も提案する)。
