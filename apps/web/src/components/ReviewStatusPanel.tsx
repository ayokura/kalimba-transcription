"use client";

import { useCallback, useState } from "react";

import { saveReviewStatus } from "@/lib/api";
import { REVIEW_STATUS_OPTIONS } from "@/lib/reviewStatus";
import { ReviewStatusPayload, ReviewStatusValue } from "@/lib/types";

type SaveState = "idle" | "saving" | "saved" | "error";

export function ReviewStatusPanel({
  transactionId,
  initialStatus,
}: {
  transactionId: string;
  initialStatus: ReviewStatusPayload | null;
}) {
  const [current, setCurrent] = useState<ReviewStatusValue | null>(
    initialStatus?.status ?? null,
  );
  const [saveState, setSaveState] = useState<SaveState>("idle");

  const choose = useCallback(
    async (status: ReviewStatusValue) => {
      setSaveState("saving");
      try {
        const saved = await saveReviewStatus(transactionId, status);
        setCurrent(saved.status);
        setSaveState("saved");
      } catch {
        setSaveState("error");
      }
    },
    [transactionId],
  );

  return (
    <section className="review-status-panel" aria-label="この録音の状態">
      <div className="review-status-head">
        <p className="eyebrow">この録音の状態</p>
        <p className="muted">
          全部直さなくて大丈夫です。「録音だけ提出」でも貢献になります。
        </p>
      </div>
      <div className="review-status-options" role="group" aria-label="状態を選ぶ">
        {REVIEW_STATUS_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`review-status-btn${current === option.value ? " active" : ""}`}
            aria-pressed={current === option.value}
            onClick={() => choose(option.value)}
            disabled={saveState === "saving"}
            title={option.description}
          >
            <strong>{option.label}</strong>
            <span className="review-status-desc">{option.description}</span>
          </button>
        ))}
      </div>
      <p className="review-status-feedback muted" role="status">
        {saveState === "saving"
          ? "保存中…"
          : saveState === "error"
          ? "状態を保存できませんでした"
          : saveState === "saved"
          ? "状態を保存しました"
          : current
          ? "状態は保存済みです"
          : "\u00a0"}
      </p>
    </section>
  );
}
