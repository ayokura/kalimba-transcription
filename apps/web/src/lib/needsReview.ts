import { CandidateSlot, ScoreEvent } from "@/lib/types";

// per-event triage 信号 (第 2 期 S3)。recognizer 出力の既存 proxy から
// 「要確認イベント」を導出する。閾値は 2026-07-04 に data/transactions の
// 46 tx / 1536 events で較正: strong-alternate 8% + ambiguous-grouping 4% +
// adjacent-slot 13% → 複合 23.4% (テスターが全走査せず約 1/4 に絞れる水準)。
// gesture=="ambiguous" 単独は 68% に付くため複音イベント限定にしている。

export const ADJACENT_SLOT_WINDOW_SEC = 0.6;
export const ALTERNATE_CONFIDENCE_MIN = 0.5;

export type NeedsReviewReasonKey =
  | "strong-alternate"
  | "ambiguous-grouping"
  | "adjacent-slot";

export type NeedsReviewReason = {
  key: NeedsReviewReasonKey;
  label: string;
};

export function needsReviewReasons(
  event: ScoreEvent,
  slots: CandidateSlot[],
): NeedsReviewReason[] {
  const reasons: NeedsReviewReason[] = [];
  const strongAlternate = (event.alternateGroupings ?? []).some(
    (alt) => alt.confidence >= ALTERNATE_CONFIDENCE_MIN,
  );
  if (strongAlternate) {
    reasons.push({ key: "strong-alternate", label: "有力な別候補" });
  }
  if (event.notes.length > 1 && event.gesture === "ambiguous") {
    reasons.push({ key: "ambiguous-grouping", label: "同時/連打あいまい" });
  }
  if (
    slots.some(
      (slot) => Math.abs(slot.startTime - event.startTimeSec) <= ADJACENT_SLOT_WINDOW_SEC,
    )
  ) {
    reasons.push({ key: "adjacent-slot", label: "近くに棄却候補" });
  }
  return reasons;
}
