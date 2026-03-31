import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ReviewWorkspace } from "@/components/ReviewWorkspace";
import { TranscriptionReviewSession } from "@/lib/types";

const mocks = vi.hoisted(() => ({
  isReviewSessionStorageAvailable: vi.fn(),
  loadReviewSession: vi.fn(),
  loadReviewAudio: vi.fn(),
  updateReviewSessionUiState: vi.fn(),
  useSearchParams: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: mocks.useSearchParams,
}));

vi.mock("@/lib/reviewSession", () => ({
  isReviewSessionStorageAvailable: mocks.isReviewSessionStorageAvailable,
  loadReviewSession: mocks.loadReviewSession,
  updateReviewSessionUiState: mocks.updateReviewSessionUiState,
}));

vi.mock("@/lib/reviewAudioStore", () => ({
  loadReviewAudio: mocks.loadReviewAudio,
}));

function buildSession(): TranscriptionReviewSession {
  return {
    sessionVersion: 1,
    sessionId: "review-session-1",
    createdAt: "2026-03-31T00:00:00.000Z",
    acquisitionMode: "live_mic",
    tuning: {
      id: "kalimba-17-c",
      name: "17 Key C Major",
      keyCount: 17,
      notes: [
        { key: 1, noteName: "C4", frequency: 261.63 },
        { key: 2, noteName: "E4", frequency: 329.63 },
        { key: 3, noteName: "G4", frequency: 392 },
      ],
    },
    instrumentProfile: null,
    recordingProfile: null,
    requestSnapshot: {
      sourceProfile: "acoustic_real",
    },
    responseSnapshot: {
      instrumentTuning: {
        id: "kalimba-17-c",
        name: "17 Key C Major",
        keyCount: 17,
        notes: [
          { key: 1, noteName: "C4", frequency: 261.63 },
          { key: 2, noteName: "E4", frequency: 329.63 },
          { key: 3, noteName: "G4", frequency: 392 },
        ],
      },
      tempo: 90,
      warnings: [],
      notationViews: {
        western: ["C4", "E4", "G4"],
        numbered: ["1", "3", "5"],
        verticalDoReMi: [["ド"], ["ミ"], ["ソ"]],
      },
      events: [
        {
          id: "evt-1",
          startBeat: 1,
          durationBeat: 1,
          notes: [
            {
              key: 1,
              pitchClass: "C",
              octave: 4,
              labelDoReMi: "ド",
              labelNumber: "1",
              frequency: 261.63,
            },
          ],
          isGlissLike: false,
          gesture: "strict_chord",
        },
        {
          id: "evt-2",
          startBeat: 2,
          durationBeat: 1,
          notes: [
            {
              key: 2,
              pitchClass: "E",
              octave: 4,
              labelDoReMi: "ミ",
              labelNumber: "3",
              frequency: 329.63,
            },
          ],
          isGlissLike: false,
          gesture: "strict_chord",
        },
        {
          id: "evt-3",
          startBeat: 3,
          durationBeat: 1,
          notes: [
            {
              key: 3,
              pitchClass: "G",
              octave: 4,
              labelDoReMi: "ソ",
              labelNumber: "5",
              frequency: 392,
            },
          ],
          isGlissLike: false,
          gesture: "strict_chord",
        },
      ],
      debug: null,
    },
    notationMode: "vertical",
    activeEventId: "evt-1",
    editedDraft: null,
  };
}

describe("ReviewWorkspace", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      value: vi.fn(),
      configurable: true,
      writable: true,
    });
    vi.stubGlobal("URL", {
      createObjectURL: vi.fn(() => "blob:review-audio"),
      revokeObjectURL: vi.fn(),
    });
    mocks.useSearchParams.mockReturnValue(new URLSearchParams("session=review-session-1"));
    mocks.isReviewSessionStorageAvailable.mockReturnValue(true);
    mocks.loadReviewSession.mockReturnValue(buildSession());
    mocks.loadReviewAudio.mockReturnValue(null);
    mocks.updateReviewSessionUiState.mockReset();
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("shows a fallback message when storage is unavailable", () => {
    mocks.isReviewSessionStorageAvailable.mockReturnValue(false);

    render(<ReviewWorkspace />);

    expect(screen.getByText("この環境では review session を開けません。")).toBeTruthy();
  });

  it("shows a fallback message when the session is missing", () => {
    mocks.loadReviewSession.mockReturnValue(null);

    render(<ReviewWorkspace />);

    expect(screen.getByText("review session が見つかりません。")).toBeTruthy();
  });

  it("updates the active event when focus navigation moves to the next event", async () => {
    const user = userEvent.setup();

    render(<ReviewWorkspace />);

    expect(screen.getByText("1 / 3")).toBeTruthy();
    expect(screen.getAllByText("evt-1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ド").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: "次の event へ移動" }));

    expect(screen.getByText("2 / 3")).toBeTruthy();
    expect(screen.getAllByText("evt-2").length).toBeGreaterThan(0);
    expect(screen.getAllByText("ミ").length).toBeGreaterThan(0);
    expect(screen.getAllByText("同時和音").length).toBeGreaterThan(0);
    expect(screen.getByText("この review session には audio が残っていません。same-tab の解析直後に `/review` を開き直してください。")).toBeTruthy();
    expect(mocks.updateReviewSessionUiState).toHaveBeenCalledWith("review-session-1", {
      notationMode: "vertical",
      activeEventId: "evt-2",
    });
  });

  it("persists notation mode changes back into the review session", async () => {
    const user = userEvent.setup();

    render(<ReviewWorkspace />);

    await user.click(screen.getByRole("button", { name: "通常表記" }));

    expect(mocks.updateReviewSessionUiState).toHaveBeenCalledWith("review-session-1", {
      notationMode: "western",
      activeEventId: "evt-1",
    });
  });

  it("marks the selected event in the sidebar and keeps event metadata visible", () => {
    render(<ReviewWorkspace />);

    const selectedEvent = screen.getByRole("button", { name: /evt-1/i });
    expect(selectedEvent.getAttribute("aria-current")).toBe("true");
    expect(screen.getAllByText("#1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1拍").length).toBeGreaterThan(0);
  });

  it("supports keyboard navigation in the event list", async () => {
    const user = userEvent.setup();

    render(<ReviewWorkspace />);

    const firstEvent = screen.getByRole("button", { name: /evt-1/i });
    firstEvent.focus();

    await user.keyboard("{ArrowDown}");

    expect(screen.getByText("2 / 3")).toBeTruthy();
    expect(mocks.updateReviewSessionUiState).toHaveBeenLastCalledWith("review-session-1", {
      notationMode: "vertical",
      activeEventId: "evt-2",
    });

    await user.keyboard("{End}");

    expect(screen.getByText("3 / 3")).toBeTruthy();
    expect(mocks.updateReviewSessionUiState).toHaveBeenLastCalledWith("review-session-1", {
      notationMode: "vertical",
      activeEventId: "evt-3",
    });
  });

  it("renders whole-track playback when audio is available and revokes the object URL on unmount", () => {
    mocks.loadReviewAudio.mockReturnValue(new Blob(["audio"], { type: "audio/wav" }));

    const { container, unmount } = render(<ReviewWorkspace />);

    expect(container.querySelector("audio")).toBeTruthy();

    unmount();

    expect(URL.createObjectURL).toHaveBeenCalled();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:review-audio");
  });

  it("handles empty events without crashing", () => {
    const emptySession = buildSession();
    emptySession.responseSnapshot.events = [];
    emptySession.responseSnapshot.notationViews = {
      western: [],
      numbered: [],
      verticalDoReMi: [],
    };
    emptySession.activeEventId = null;
    mocks.loadReviewSession.mockReturnValue(emptySession);

    render(<ReviewWorkspace />);

    expect(screen.getAllByText("0 events").length).toBeGreaterThan(0);
    expect(screen.getByText("event がありません。")).toBeTruthy();
  });
});
