import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewStatusPanel } from "@/components/ReviewStatusPanel";

const saveReviewStatus = vi.fn();

vi.mock("@/lib/api", () => ({
  saveReviewStatus: (...args: unknown[]) => saveReviewStatus(...args),
}));

afterEach(() => {
  cleanup();
  saveReviewStatus.mockReset();
});

describe("ReviewStatusPanel", () => {
  it("renders all status options and highlights the initial status", () => {
    render(
      <ReviewStatusPanel
        transactionId="tx-1"
        initialStatus={{ version: 1, status: "review_started" }}
      />,
    );
    expect(screen.getByRole("button", { name: /録音だけ提出/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /確認・修正完了/ })).toBeTruthy();
    const started = screen.getByRole("button", { name: /確認中/ });
    expect(started.getAttribute("aria-pressed")).toBe("true");
  });

  it("saves the chosen status and updates the active selection", async () => {
    saveReviewStatus.mockResolvedValue({ version: 1, status: "review_completed" });
    render(<ReviewStatusPanel transactionId="tx-1" initialStatus={null} />);

    await userEvent.click(screen.getByRole("button", { name: /確認・修正完了/ }));

    await waitFor(() => {
      expect(saveReviewStatus).toHaveBeenCalledWith("tx-1", "review_completed");
    });
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /確認・修正完了/ }).getAttribute("aria-pressed"),
      ).toBe("true");
    });
    expect(screen.getByRole("status").textContent).toContain("状態を保存しました");
  });

  it("shows an error message when saving fails", async () => {
    saveReviewStatus.mockRejectedValue(new Error("nope"));
    render(<ReviewStatusPanel transactionId="tx-1" initialStatus={null} />);

    await userEvent.click(screen.getByRole("button", { name: /録音だけ提出/ }));

    await waitFor(() => {
      expect(screen.getByRole("status").textContent).toContain("保存できませんでした");
    });
  });
});
