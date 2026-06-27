import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewQueue } from "@/components/ReviewQueue";
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
});
