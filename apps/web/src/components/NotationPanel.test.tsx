import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { NotationPanel } from "@/components/NotationPanel";
import { TranscriptionResult } from "@/lib/types";

function buildResult(): TranscriptionResult {
  return {
    instrumentTuning: {
      id: "kalimba-17-c",
      name: "17 Key C Major",
      keyCount: 17,
      notes: [
        { key: 1, noteName: "C4", frequency: 261.63 },
        { key: 2, noteName: "E4", frequency: 329.63 },
      ],
    },
    tempo: 90,
    warnings: [],
    notationViews: {
      western: ["C4", "E4"],
      numbered: ["1", "3"],
      verticalDoReMi: [["ド"], ["ミ"]],
    },
    events: [
      {
        id: "evt-1",
        startBeat: 1,
        durationBeat: 1,
        startTimeSec: 0.5,
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
        startTimeSec: 1.0,
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
    ],
  };
}

describe("NotationPanel", () => {
  it("highlights the selected event and lets notation clicks change selection", async () => {
    const onActiveEventIdChange = vi.fn();
    const user = userEvent.setup();
    const { container } = render(
      <NotationPanel
        result={buildResult()}
        mode="vertical"
        onModeChange={() => {}}
        activeEventId="evt-2"
        onActiveEventIdChange={onActiveEventIdChange}
      />,
    );

    const eventButtons = container.querySelectorAll(".event-card-button");
    expect(eventButtons).toHaveLength(2);
    expect(eventButtons[1]?.className).toContain("selected");

    await user.click(eventButtons[0] as HTMLElement);

    expect(onActiveEventIdChange).toHaveBeenCalledWith("evt-1");
  });
});
