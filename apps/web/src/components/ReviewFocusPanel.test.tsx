import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ReviewFocusPanel } from "@/components/ReviewFocusPanel";
import { ScoreEvent } from "@/lib/types";

function buildEvents(): ScoreEvent[] {
  return [
    {
      id: "evt-1",
      startBeat: 1,
      durationBeat: 1,
      startTimeSec: 0.5,
      notes: [
        { key: 1, pitchClass: "C", octave: 4, labelDoReMi: "ド", labelNumber: "1", frequency: 261.63 },
      ],
      isGlissLike: false,
      gesture: "strict_chord",
    },
    {
      id: "evt-2",
      startBeat: 2,
      durationBeat: 2,
      startTimeSec: 1.0,
      notes: [
        { key: 2, pitchClass: "E", octave: 4, labelDoReMi: "ミ", labelNumber: "3", frequency: 329.63 },
        { key: 3, pitchClass: "G", octave: 4, labelDoReMi: "ソ", labelNumber: "5", frequency: 392 },
      ],
      isGlissLike: false,
      gesture: "arpeggio",
    },
  ];
}

describe("ReviewFocusPanel", () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("shows focused event details in one place", () => {
    render(
      <ReviewFocusPanel
        events={buildEvents()}
        activeEventId="evt-2"
        onActiveEventIdChange={() => {}}
      />,
    );

    expect(screen.getAllByText("evt-2").length).toBeGreaterThan(0);
    expect(screen.getByText("2拍目から 2拍")).toBeTruthy();
    expect(screen.getAllByText("アルペジオ").length).toBeGreaterThan(0);
    expect(screen.getByText("Key 2 · E4 · ミ")).toBeTruthy();
    expect(screen.getByText("Key 3 · G4 · ソ")).toBeTruthy();
  });

  it("navigates to neighboring events", async () => {
    const user = userEvent.setup();
    const onActiveEventIdChange = vi.fn();

    render(
      <ReviewFocusPanel
        events={buildEvents()}
        activeEventId="evt-2"
        onActiveEventIdChange={onActiveEventIdChange}
      />,
    );

    await user.click(screen.getByRole("button", { name: "前の event へ移動" }));

    expect(onActiveEventIdChange).toHaveBeenCalledWith("evt-1");
  });
});
