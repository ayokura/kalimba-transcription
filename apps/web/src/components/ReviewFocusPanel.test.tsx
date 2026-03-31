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
      durationBeat: 2,
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
      gesture: "arpeggio",
    },
  ];
}

describe("ReviewFocusPanel", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows previous/current/next context and lets navigation buttons change focus", async () => {
    const onActiveEventIdChange = vi.fn();
    const user = userEvent.setup();

    render(
      <ReviewFocusPanel
        events={buildEvents()}
        activeEventId="evt-2"
        onActiveEventIdChange={onActiveEventIdChange}
      />,
    );

    expect(screen.getByText("2 / 3")).toBeTruthy();
    expect(screen.getAllByText("evt-1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("evt-2").length).toBeGreaterThan(0);
    expect(screen.getAllByText("evt-3").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: "前の event へ移動" }));
    expect(onActiveEventIdChange).toHaveBeenCalledWith("evt-1");

    await user.click(screen.getByRole("button", { name: "次の event へ移動" }));
    expect(onActiveEventIdChange).toHaveBeenCalledWith("evt-3");
  });

  it("falls back to the first event when the active id is missing and disables prev navigation", () => {
    render(
      <ReviewFocusPanel
        events={buildEvents()}
        activeEventId={null}
        onActiveEventIdChange={() => {}}
      />,
    );

    expect(screen.getByText("1 / 3")).toBeTruthy();
    expect((screen.getByRole("button", { name: "前の event へ移動" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "次の event へ移動" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it("handles a single-event review without neighboring context", () => {
    render(
      <ReviewFocusPanel
        events={[buildEvents()[0]!]}
        activeEventId="evt-1"
        onActiveEventIdChange={() => {}}
      />,
    );

    expect(screen.getByText("1 / 1")).toBeTruthy();
    expect((screen.getByRole("button", { name: "前の event へ移動" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "次の event へ移動" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getAllByText("該当なし")).toHaveLength(2);
  });
});
