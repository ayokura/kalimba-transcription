"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

import { fetchReviewQueue } from "@/lib/api";
import { REVIEW_STATUS_OPTIONS, reviewStatusLabel } from "@/lib/reviewStatus";
import { ReviewQueueEntry, ReviewStatusValue } from "@/lib/types";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; entries: ReviewQueueEntry[] }
  | { kind: "error"; message: string };

type StatusFilter = ReviewStatusValue | "all";

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

      {state.kind === "loading" ? (
        <p className="muted">読み込み中…</p>
      ) : state.kind === "error" ? (
        <p className="empty">読み込めませんでした: {state.message}</p>
      ) : state.entries.length === 0 ? (
        <p className="empty">該当する録音はありません。</p>
      ) : (
        <>
          <p className="muted review-queue-count">{counts} 件</p>
          <ul className="review-queue-list">
            {state.entries.map((entry) => (
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
