import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ReviewEventListPanel } from "@/components/ReviewEventListPanel";
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
      durationBeat: 1,
      startTimeSec: 1.0,
      notes: [
        { key: 2, pitchClass: "E", octave: 4, labelDoReMi: "ミ", labelNumber: "3", frequency: 329.63 },
      ],
      isGlissLike: false,
      gesture: "slide_chord",
    },
    {
      id: "evt-3",
      startBeat: 3,
      durationBeat: 1,
      startTimeSec: 1.5,
      notes: [
        { key: 3, pitchClass: "G", octave: 4, labelDoReMi: "ソ", labelNumber: "5", frequency: 392 },
      ],
      isGlissLike: false,
      gesture: "arpeggio",
    },
  ];
}

describe("ReviewEventListPanel", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      value: vi.fn(),
      configurable: true,
      writable: true,
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("shows event metadata and current selection", () => {
    render(
      <ReviewEventListPanel
        events={buildEvents()}
        activeEventId="evt-1"
        onActiveEventIdChange={() => {}}
      />,
    );

    const selectedEvent = screen.getByRole("button", { name: /evt-1/i });
    expect(selectedEvent.getAttribute("aria-current")).toBe("true");
    expect(screen.getAllByText("#1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1拍").length).toBeGreaterThan(0);
    expect(screen.getAllByText("スライド和音").length).toBeGreaterThan(0);
  });

  it("supports keyboard navigation between events", async () => {
    const user = userEvent.setup();
    const onActiveEventIdChange = vi.fn();

    render(
      <ReviewEventListPanel
        events={buildEvents()}
        activeEventId="evt-1"
        onActiveEventIdChange={onActiveEventIdChange}
      />,
    );

    const firstEvent = screen.getByRole("button", { name: /evt-1/i });
    firstEvent.focus();

    await user.keyboard("{ArrowDown}{End}");

    expect(onActiveEventIdChange).toHaveBeenNthCalledWith(1, "evt-2");
    expect(onActiveEventIdChange).toHaveBeenNthCalledWith(2, "evt-3");
  });

  it("does not render audition controls when audition is unavailable", () => {
    render(
      <ReviewEventListPanel
        events={buildEvents()}
        activeEventId="evt-1"
        onActiveEventIdChange={() => {}}
      />,
    );

    expect(screen.queryByLabelText("evt-1 をここから再生")).toBeNull();
  });

  it("selecting an event seeks/plays when audition is available", async () => {
    const user = userEvent.setup();
    const onActiveEventIdChange = vi.fn();
    const onAuditionEvent = vi.fn();

    const { container } = render(
      <ReviewEventListPanel
        events={buildEvents()}
        activeEventId="evt-1"
        onActiveEventIdChange={onActiveEventIdChange}
        onAuditionEvent={onAuditionEvent}
      />,
    );

    // event 本体 (.event-list-item) を選ぶ。per-event の再生 span も role=button を
    // 持つため、role 名だけでは曖昧になるのでクラスで絞り込む。
    const items = Array.from(container.querySelectorAll<HTMLButtonElement>(".event-list-item"));
    const evt2 = items.find((el) => el.textContent?.includes("evt-2"));
    expect(evt2).toBeTruthy();
    await user.click(evt2 as HTMLButtonElement);

    expect(onActiveEventIdChange).toHaveBeenCalledWith("evt-2");
    expect(onAuditionEvent).toHaveBeenCalledWith("evt-2");
  });

  it("the per-event play control auditions without re-triggering selection twice", async () => {
    const user = userEvent.setup();
    const onActiveEventIdChange = vi.fn();
    const onAuditionEvent = vi.fn();

    render(
      <ReviewEventListPanel
        events={buildEvents()}
        activeEventId="evt-1"
        onActiveEventIdChange={onActiveEventIdChange}
        onAuditionEvent={onAuditionEvent}
      />,
    );

    await user.click(screen.getByLabelText("evt-3 をここから再生"));

    // stopPropagation で親 button の選択ハンドラは発火しない。
    expect(onAuditionEvent).toHaveBeenCalledTimes(1);
    expect(onAuditionEvent).toHaveBeenCalledWith("evt-3");
    expect(onActiveEventIdChange).not.toHaveBeenCalled();
  });
});
