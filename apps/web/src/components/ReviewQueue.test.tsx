import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewQueue, sortQueueEntries } from "@/components/ReviewQueue";
import { ReviewQueueEntry } from "@/lib/types";

const fetchReviewQueue = vi.fn();

vi.mock("@/lib/api", () => ({
  fetchReviewQueue: (...args: unknown[]) => fetchReviewQueue(...args),
}));

function entry(overrides: Partial<ReviewQueueEntry> = {}): ReviewQueueEntry {
  return {
    transactionId: "tx-1",
    createdAt: 1_700_000_000,
    tuningId: "kalimba-17-c",
    tuningName: "17 Key C Major",
    eventCount: 12,
    audioSha256: "abc",
    reviewStatus: "review_started",
    reviewStatusUpdatedAt: null,
    hasCorrections: true,
    hasMemo: false,
    warningCount: 1,
    candidateSlotCount: 2,
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  fetchReviewQueue.mockReset();
});

describe("ReviewQueue", () => {
  it("lists entries with status and metadata badges and links to review", async () => {
    fetchReviewQueue.mockResolvedValue([entry()]);
    render(<ReviewQueue />);

    await waitFor(() => {
      expect(screen.getByText(/17 Key C Major/)).toBeTruthy();
    });
    expect(screen.getByText("修正あり")).toBeTruthy();
    expect(screen.getByText("警告 1")).toBeTruthy();
    expect(screen.getByText("候補 2")).toBeTruthy();
    const link = screen.getByRole("link", { name: /17 Key C Major/ });
    expect(link.getAttribute("href")).toBe("/score/tx-1/review");
  });

  it("re-queries the API with a status filter when a chip is clicked", async () => {
    fetchReviewQueue.mockResolvedValue([]);
    render(<ReviewQueue />);

    await waitFor(() => {
      expect(fetchReviewQueue).toHaveBeenCalledWith({ limit: 200, status: null });
    });

    await userEvent.click(screen.getByRole("button", { name: "録り直しが必要" }));

    await waitFor(() => {
      expect(fetchReviewQueue).toHaveBeenCalledWith({ limit: 200, status: "rerecord_needed" });
    });
  });

  it("shows an empty message when no entries match", async () => {
    fetchReviewQueue.mockResolvedValue([]);
    render(<ReviewQueue />);
    await waitFor(() => {
      expect(screen.getByText("該当する録音はありません。")).toBeTruthy();
    });
  });

  it("renders in priority order by default and newest order on toggle", async () => {
    const untriagedUncertain = entry({
      transactionId: "tx-untriaged",
      reviewStatus: null,
      createdAt: 100,
      warningCount: 0,
      candidateSlotCount: 3,
      hasCorrections: false,
    });
    const completedNewest = entry({
      transactionId: "tx-completed",
      reviewStatus: "review_completed",
      createdAt: 300,
      warningCount: 5,
      candidateSlotCount: 5,
    });
    const startedNoise = entry({
      transactionId: "tx-started",
      reviewStatus: "review_started",
      createdAt: 200,
      warningCount: 2,
      candidateSlotCount: 0,
      hasCorrections: false,
    });
    fetchReviewQueue.mockResolvedValue([completedNewest, startedNoise, untriagedUncertain]);
    render(<ReviewQueue />);

    const reviewHrefs = () =>
      screen
        .getAllByRole("link")
        .map((l) => l.getAttribute("href"))
        .filter((href) => href?.endsWith("/review"));
    await waitFor(() => {
      expect(reviewHrefs().length).toBe(3);
    });
    // 優先度順 (既定): 未トリアージ → 確認中 → 完了 (新しさより状態層が優先)
    let hrefs = reviewHrefs();
    expect(hrefs).toEqual([
      "/score/tx-untriaged/review",
      "/score/tx-started/review",
      "/score/tx-completed/review",
    ]);

    await userEvent.click(screen.getByRole("button", { name: "新着順" }));
    hrefs = reviewHrefs();
    // 新着順 = API の返却順 (newest first) をそのまま維持
    expect(hrefs).toEqual([
      "/score/tx-completed/review",
      "/score/tx-started/review",
      "/score/tx-untriaged/review",
    ]);
  });
});

describe("sortQueueEntries", () => {
  it("同一状態層では candidateSlotCount + warningCount*3 の降順、同点は新しい順", () => {
    const low = entry({ transactionId: "low", reviewStatus: null, warningCount: 0, candidateSlotCount: 1, hasCorrections: false, createdAt: 50 });
    const high = entry({ transactionId: "high", reviewStatus: null, warningCount: 2, candidateSlotCount: 1, hasCorrections: false, createdAt: 10 });
    const tieNewer = entry({ transactionId: "tie-newer", reviewStatus: null, warningCount: 0, candidateSlotCount: 1, hasCorrections: false, createdAt: 99 });
    const sorted = sortQueueEntries([low, high, tieNewer], "priority");
    expect(sorted.map((e) => e.transactionId)).toEqual(["high", "tie-newer", "low"]);
  });

  it("qualityDifficulty が高い録音は slot が少なくても優先される (#194)", () => {
    // red (difficulty 0.7) ×20 = +14 — slot 10 個の録音より上に来る。
    const manySlots = entry({ transactionId: "many-slots", reviewStatus: null, warningCount: 0, candidateSlotCount: 10, hasCorrections: false, createdAt: 10 });
    const hardRecording = entry({ transactionId: "hard", reviewStatus: null, warningCount: 0, candidateSlotCount: 2, hasCorrections: false, createdAt: 5, qualityDifficulty: 0.7 });
    const legacyPayload = entry({ transactionId: "legacy", reviewStatus: null, warningCount: 0, candidateSlotCount: 2, hasCorrections: false, createdAt: 20 });
    const sorted = sortQueueEntries([manySlots, hardRecording, legacyPayload], "priority");
    expect(sorted.map((e) => e.transactionId)).toEqual(["hard", "many-slots", "legacy"]);
  });
});
