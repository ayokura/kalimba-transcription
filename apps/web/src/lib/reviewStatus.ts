import { ReviewStatusValue } from "@/lib/types";

export type ReviewStatusMeta = {
  value: ReviewStatusValue;
  label: string;
  description: string;
};

// Ordered for the tester collection workflow. recorded_only is a valid terminal
// contribution: a tester can submit a recording without doing a full manual
// transcription. review_completed is the only status the GT-promotion script
// treats as ready for ground-truth candidacy.
export const REVIEW_STATUS_OPTIONS: ReviewStatusMeta[] = [
  {
    value: "recorded_only",
    label: "録音だけ提出",
    description: "録音を提出済み。修正はしていません（これだけでも貢献になります）。",
  },
  {
    value: "review_started",
    label: "確認中",
    description: "確認・修正を始めました。まだ途中です。",
  },
  {
    value: "review_completed",
    label: "確認・修正完了",
    description: "音の過不足を直し終えました。教師データ候補になります。",
  },
  {
    value: "uncertain",
    label: "判断保留",
    description: "合っているか分からない箇所があります。",
  },
  {
    value: "rerecord_needed",
    label: "録り直しが必要",
    description: "この録音は直すより録り直した方が早いです。",
  },
  {
    value: "unusable",
    label: "使えない録音",
    description: "ノイズ・音量などで使えません。",
  },
];

const META_BY_VALUE = new Map(REVIEW_STATUS_OPTIONS.map((o) => [o.value, o]));

export function reviewStatusLabel(value: ReviewStatusValue | null | undefined): string {
  if (!value) return "未設定";
  return META_BY_VALUE.get(value)?.label ?? value;
}
