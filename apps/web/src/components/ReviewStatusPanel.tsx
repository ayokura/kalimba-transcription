"use client";

import { useCallback, useState } from "react";

import { saveReviewStatus } from "@/lib/api";
import { REVIEW_STATUS_OPTIONS } from "@/lib/reviewStatus";
import { ReviewStatusPayload, ReviewStatusValue } from "@/lib/types";

type SaveState = "idle" | "saving" | "saved" | "error";

export function ReviewStatusPanel({
  transactionId,
  initialStatus,
  hasUnsavedCorrections = false,
}: {
  transactionId: string;
  initialStatus: ReviewStatusPayload | null;
  hasUnsavedCorrections?: boolean;
}) {
  const [current, setCurrent] = useState<ReviewStatusValue | null>(
    initialStatus?.status ?? null,
  );
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [blockedMessage, setBlockedMessage] = useState<string | null>(null);

  const choose = useCallback(
    async (status: ReviewStatusValue) => {
      // review_completed is the only status that gates corrections -> GT
      // promotion; confirming it while corrections are unsaved would promote
      // a stale corrections.json if the tab closes without saving.
      if (status === "review_completed" && hasUnsavedCorrections) {
        setBlockedMessage(
          "未保存の修正があります。先に「保存」を押してから「確認・修正完了」にしてください。",
        );
        return;
      }
      setBlockedMessage(null);
      setSaveState("saving");
      try {
        const saved = await saveReviewStatus(transactionId, status);
        setCurrent(saved.status);
        setSaveState("saved");
      } catch {
        setSaveState("error");
      }
    },
    [transactionId, hasUnsavedCorrections],
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
        {blockedMessage
          ? blockedMessage
          : saveState === "saving"
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
