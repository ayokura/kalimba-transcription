# 敵対的セルフ録音 — 破壊メニュー票

第 2 期 S1 ([sprint-plan-2026-07b.md](sprint-plan-2026-07b.md))。認識器を**意図的に壊す**録音セッションの設計図。非飽和 held-out 録音を作ることが目的なので、「上手く弾く」必要はない — 予想どおり壊れたら成功、壊れなかったらそれも収穫 (認識器が予想より強い証拠)。

**機械可読版が正**: [`apps/web/src/lib/adversarialMenu.ts`](../apps/web/src/lib/adversarialMenu.ts)。/debug/triage ページに演奏指示と期待列が表示され、その場で録音 → 採譜 → 登録まで完結する。期待イベント列は `expectedPerformance` として録音に添付されるため、GT 化は自動整列 + 差分確認だけで済む。

## セッションの回し方 (~15 分)

1. score.ayokura.net/debug/triage を開く
2. メニューから項目を選ぶ → 演奏指示に従って録音 → 「採譜して登録」
3. 全 8 項目 (またはやりたい分だけ)。1 項目 1–2 分想定
4. 終わったら知らせてもらえれば、エージェント側で diagnosis → GT 化支援 → baseline 追加まで進める

## メニュー概要 (詳細は TS / ページ参照)

| # | 項目 | 狙う機構 | 予想される失敗 |
|---|---|---|---|
| 1 | 残響マスキング | Mech2 carryover vs re-attack | 残響中の弱い C5 の見逃し (17ea7626 C5@11.55s 型) |
| 2 | 密集連打 | Mech3 密集誤選択 | D5/F5 見逃し + E6 捏造 (13.323s 型) |
| 3 | 消え入る弱打 | 弱 attack の broadband flux 閾値割れ | pp 単音の見逃し |
| 4 | 物理隣接 tine 同時打 | #138 隣接 leak | 和音片方の欠落 or leak 捏造 |
| 5 | フルレンジ・グリッサンド | gliss の segment 分割 | イベント過分割/欠落 (期待列なし・耳確認) |
| 6 | 即ミュート → 再打鍵 | mute-dip rescue の限界 | 再打鍵見逃し or 接触音捏造 |
| 7 | 極端なダイナミクス差 | noise floor 較正・gain 絶対量依存 | pp 側全滅 or ff 残響 FP |
| 8 | 最速トレモロ | onset 時間分解能 (wait/pre_max 窓) | 8 連打の数え落とし |
