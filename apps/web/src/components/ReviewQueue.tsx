"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchReviewQueue } from "@/lib/api";
import { TxIdBadge } from "@/components/TxIdBadge";
import { REVIEW_STATUS_OPTIONS, reviewStatusLabel } from "@/lib/reviewStatus";
import { ReviewQueueEntry, ReviewStatusValue } from "@/lib/types";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; entries: ReviewQueueEntry[] }
  | { kind: "error"; message: string };

type StatusFilter = ReviewStatusValue | "all";

type SortMode = "priority" | "newest";

// 暫定優先度 (第 2 期 S3、GT 不要 proxy のみ):
// 未トリアージ (recorded_only / status なし) を最優先層、確認中を次層に置き、
// 層内は「認識器の不確かさ」= 棄却候補 slot 数 + 警告数で降順。
// 修正が既に付いている録音はレビューが進んでいる証拠なので同点時に僅かに繰り上げる。
const STATUS_PRIORITY_RANK: Record<string, number> = {
  recorded_only: 0,
  review_started: 1,
  uncertain: 2,
  review_completed: 3,
  unusable: 4,
};

export function queuePriorityScore(entry: ReviewQueueEntry): number {
  // #194 (S6): recognizer の内部 difficulty 自己評価 (0-1) を優先度に加算。
  // 13 録音の較正 (2026-07-05) で difficulty は (1-F1) と相関 0.73、red flag
  // は F1 最低 2 録音を正しく指した。×20 で red (≈0.7) ≈ slot 14 個相当 —
  // slot が少なくても「録音品質そのものが悪い」録音が沈まないようにする。
  // 旧 payload (qualityDifficulty なし) は 0 扱いで従来順位のまま。
  const difficultyBoost = (entry.qualityDifficulty ?? 0) * 20;
  return (
    entry.candidateSlotCount +
    entry.warningCount * 3 +
    (entry.hasCorrections ? 1 : 0) +
    difficultyBoost
  );
}

export function sortQueueEntries(
  entries: ReviewQueueEntry[],
  mode: SortMode,
): ReviewQueueEntry[] {
  if (mode === "newest") return entries;
  return [...entries].sort((a, b) => {
    const rankA = STATUS_PRIORITY_RANK[a.reviewStatus ?? "recorded_only"] ?? 0;
    const rankB = STATUS_PRIORITY_RANK[b.reviewStatus ?? "recorded_only"] ?? 0;
    if (rankA !== rankB) return rankA - rankB;
    const scoreDiff = queuePriorityScore(b) - queuePriorityScore(a);
    if (scoreDiff !== 0) return scoreDiff;
    return b.createdAt - a.createdAt;
  });
}

function formatCreatedAt(createdAt: number): string {
  // API returns seconds (st_mtime). Render as a locale date-time.
  const ms = createdAt > 1e12 ? createdAt : createdAt * 1000;
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return "-";
  }
}

export function ReviewQueue() {
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [sortMode, setSortMode] = useState<SortMode>("priority");
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  const load = useCallback(async (status: StatusFilter) => {
    setState({ kind: "loading" });
    try {
      const entries = await fetchReviewQueue({
        limit: 200,
        status: status === "all" ? null : status,
      });
      setState({ kind: "ready", entries });
    } catch (err) {
      setState({
        kind: "error",
        message: err instanceof Error ? err.message : "読み込みに失敗しました。",
      });
    }
  }, []);

  useEffect(() => {
    void load(filter);
  }, [filter, load]);

  const counts = useMemo(() => {
    if (state.kind !== "ready") return null;
    return state.entries.length;
  }, [state]);

  const sortedEntries = useMemo(() => {
    if (state.kind !== "ready") return [];
    return sortQueueEntries(state.entries, sortMode);
  }, [state, sortMode]);

  return (
    <main className="review-queue-shell">
      <header className="review-queue-header">
        <div className="review-queue-header-row">
          <Link href="/" className="review-queue-home-link">
            ← トップへ
          </Link>
          <h1 className="review-queue-title">確認キュー</h1>
        </div>
        <p className="muted">
          どの録音を次に確認すべきかを一覧します。状態でしぼり込めます。
        </p>
      </header>

      <div className="review-queue-filters" role="group" aria-label="状態でしぼり込む">
        <button
          type="button"
          className={`review-queue-filter${filter === "all" ? " active" : ""}`}
          aria-pressed={filter === "all"}
          onClick={() => setFilter("all")}
        >
          すべて
        </button>
        {REVIEW_STATUS_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`review-queue-filter${filter === option.value ? " active" : ""}`}
            aria-pressed={filter === option.value}
            onClick={() => setFilter(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>

      <div className="review-queue-filters" role="group" aria-label="並び順">
        <button
          type="button"
          className={`review-queue-filter${sortMode === "priority" ? " active" : ""}`}
          aria-pressed={sortMode === "priority"}
          onClick={() => setSortMode("priority")}
        >
          優先度順
        </button>
        <button
          type="button"
          className={`review-queue-filter${sortMode === "newest" ? " active" : ""}`}
          aria-pressed={sortMode === "newest"}
          onClick={() => setSortMode("newest")}
        >
          新着順
        </button>
      </div>

      {state.kind === "loading" ? (
        <p className="muted">読み込み中…</p>
      ) : state.kind === "error" ? (
        <p className="empty">読み込めませんでした: {state.message}</p>
      ) : state.entries.length === 0 ? (
        <p className="empty">該当する録音はありません。</p>
      ) : (
        <>
          <p className="muted review-queue-count">
            {counts} 件
            {sortMode === "priority" ? " (未確認 → 不確かさの高い順)" : ""}
          </p>
          <ul className="review-queue-list">
            {sortedEntries.map((entry) => (
              <li key={entry.transactionId} className="review-queue-item">
                <Link
                  href={`/score/${entry.transactionId}/review`}
                  className="review-queue-link"
                >
                  <span className="review-queue-item-main">
                    <span className="review-queue-item-title">
                      {entry.tuningName ?? entry.tuningId ?? "(調律不明)"} ·{" "}
                      {entry.eventCount} events
                    </span>
                    <span className="review-queue-item-time muted">
                      {formatCreatedAt(entry.createdAt)}
                      {" · "}
                      <TxIdBadge id={entry.transactionId} />
                    </span>
                  </span>
                  <span className="review-queue-item-badges">
                    <span
                      className={`review-queue-status status-${entry.reviewStatus ?? "recorded_only"}`}
                    >
                      {reviewStatusLabel(entry.reviewStatus)}
                    </span>
                    {entry.hasCorrections ? (
                      <span className="review-queue-flag">修正あり</span>
                    ) : null}
                    {entry.hasMemo ? <span className="review-queue-flag">メモ</span> : null}
                    {entry.warningCount > 0 ? (
                      <span className="review-queue-flag warn">
                        警告 {entry.warningCount}
                      </span>
                    ) : null}
                    {entry.candidateSlotCount > 0 ? (
                      <span className="review-queue-flag">
                        候補 {entry.candidateSlotCount}
                      </span>
                    ) : null}
                    {entry.isStale ? (
                      <span
                        className="review-queue-flag stale"
                        title="保存されている認識結果は現在の認識器と異なるバージョンです。再認識で最新化できます。"
                      >
                        更新あり
                      </span>
                    ) : null}
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        </>
      )}
    </main>
  );
}
