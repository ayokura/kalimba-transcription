import { describe, expect, it } from "vitest";

import { findEventById, findEventIdAtSec } from "@/lib/eventTiming";
import { ScoreEvent } from "@/lib/types";

function buildEvent(id: string, startTimeSec: number): ScoreEvent {
  return {
    id,
    startBeat: 0,
    durationBeat: 1,
    startTimeSec,
    notes: [],
    isGlissLike: false,
    gesture: "ambiguous",
  };
}

describe("findEventIdAtSec", () => {
  it("returns null for empty event list", () => {
    expect(findEventIdAtSec([], 1.0)).toBeNull();
  });

  it("returns null when currentSec is before first event", () => {
    const events = [buildEvent("evt-1", 1.0), buildEvent("evt-2", 2.0)];
    expect(findEventIdAtSec(events, 0.5)).toBeNull();
  });

  it("returns the event whose startTimeSec <= sec < next.startTimeSec", () => {
    const events = [
      buildEvent("evt-1", 0.0),
      buildEvent("evt-2", 1.0),
      buildEvent("evt-3", 2.5),
    ];
    expect(findEventIdAtSec(events, 0.0)).toBe("evt-1");
    expect(findEventIdAtSec(events, 0.99)).toBe("evt-1");
    expect(findEventIdAtSec(events, 1.0)).toBe("evt-2");
    expect(findEventIdAtSec(events, 2.49)).toBe("evt-2");
    expect(findEventIdAtSec(events, 2.5)).toBe("evt-3");
  });

  it("returns the last event for any time past its start", () => {
    const events = [buildEvent("evt-1", 0.0), buildEvent("evt-2", 1.0)];
    expect(findEventIdAtSec(events, 1000.0)).toBe("evt-2");
  });

  it("handles a single event", () => {
    const events = [buildEvent("evt-only", 0.5)];
    expect(findEventIdAtSec(events, 0.0)).toBeNull();
    expect(findEventIdAtSec(events, 0.5)).toBe("evt-only");
    expect(findEventIdAtSec(events, 100)).toBe("evt-only");
  });
});

describe("findEventById", () => {
  it("returns the matching event", () => {
    const events = [buildEvent("evt-1", 0.0), buildEvent("evt-2", 1.0)];
    expect(findEventById(events, "evt-2")?.startTimeSec).toBe(1.0);
  });

  it("returns null when id is not found", () => {
    const events = [buildEvent("evt-1", 0.0)];
    expect(findEventById(events, "missing")).toBeNull();
  });
});
