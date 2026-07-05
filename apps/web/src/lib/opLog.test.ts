import { beforeEach, describe, expect, it } from "vitest";

import { clearOpLog, loadOpLog, logOp, summarizeOpLog } from "@/lib/opLog";

const TX = "tx-opLog-test";

beforeEach(() => {
  clearOpLog(TX);
});

describe("logOp / loadOpLog", () => {
  it("appends entries and preserves order", () => {
    logOp(TX, "chord-note-remove", { timeSec: 1.2, notes: ["C4"] });
    logOp(TX, "chord-note-add", { timeSec: 1.2, notes: ["D4"] });
    const entries = loadOpLog(TX);
    expect(entries).toHaveLength(2);
    expect(entries[0].cls).toBe("chord-note-remove");
    expect(entries[1].cls).toBe("chord-note-add");
  });

  it("caps stored entries at the capacity guard, dropping the oldest first", () => {
    for (let i = 0; i < 2005; i += 1) {
      logOp(TX, "other");
    }
    const entries = loadOpLog(TX);
    expect(entries.length).toBe(2000);
  });

  it("ignores corrupted localStorage content instead of throwing", () => {
    window.localStorage.setItem("kalimba.opLog.v1.corrupt-tx", "{not json");
    expect(() => loadOpLog("corrupt-tx")).not.toThrow();
    expect(loadOpLog("corrupt-tx")).toEqual([]);
  });

  it("filters out entries with an unknown cls or malformed meta", () => {
    window.localStorage.setItem(
      "kalimba.opLog.v1.mixed-tx",
      JSON.stringify([
        { ts: 1, cls: "chord-note-add" },
        { ts: 2, cls: "not-a-real-class" },
        { ts: 3, cls: "undo", meta: { notes: "not-an-array" } },
        { ts: 4 },
      ]),
    );
    const entries = loadOpLog("mixed-tx");
    expect(entries).toEqual([{ ts: 1, cls: "chord-note-add" }]);
  });

  it("clearOpLog removes all entries for the transaction", () => {
    logOp(TX, "undo");
    clearOpLog(TX);
    expect(loadOpLog(TX)).toEqual([]);
  });
});

describe("summarizeOpLog", () => {
  it("counts by class and totals, zero-filling unused classes", () => {
    const summary = summarizeOpLog([
      { ts: 1000, cls: "chord-note-remove" },
      { ts: 2000, cls: "chord-note-remove" },
      { ts: 3000, cls: "undo" },
    ]);
    expect(summary.countsByClass["chord-note-remove"]).toBe(2);
    expect(summary.countsByClass.undo).toBe(1);
    expect(summary.countsByClass["event-remove"]).toBe(0);
    expect(summary.totalCount).toBe(3);
  });

  it("sums active time from gaps, capping any single gap at 120s", () => {
    const summary = summarizeOpLog([
      { ts: 0, cls: "chord-note-add" },
      { ts: 5_000, cls: "chord-note-add" }, // +5s
      { ts: 5_000 + 300_000, cls: "chord-note-add" }, // +300s, capped to 120s
    ]);
    expect(summary.activeTimeSec).toBeCloseTo(5 + 120, 5);
  });

  it("computes wall time as first-to-last elapsed, uncapped", () => {
    const summary = summarizeOpLog([
      { ts: 0, cls: "other" },
      { ts: 500_000, cls: "other" },
    ]);
    expect(summary.wallTimeSec).toBeCloseTo(500, 5);
  });

  it("is order-independent (sorts by ts before aggregating)", () => {
    const a = summarizeOpLog([
      { ts: 10_000, cls: "chord-note-add" },
      { ts: 0, cls: "chord-note-remove" },
    ]);
    const b = summarizeOpLog([
      { ts: 0, cls: "chord-note-remove" },
      { ts: 10_000, cls: "chord-note-add" },
    ]);
    expect(a).toEqual(b);
  });

  it("computes average gap per class from the immediately preceding op", () => {
    const summary = summarizeOpLog([
      { ts: 0, cls: "other" },
      { ts: 2_000, cls: "chord-note-add" }, // gap 2s attributed to chord-note-add
      { ts: 6_000, cls: "chord-note-add" }, // gap 4s attributed to chord-note-add
    ]);
    expect(summary.avgGapSecByClass["chord-note-add"]).toBeCloseTo((2 + 4) / 2, 5);
    expect(summary.avgGapSecByClass.other).toBeUndefined();
  });

  it("counts distinct touched (rounded timeSec, note) pairs", () => {
    const summary = summarizeOpLog([
      { ts: 0, cls: "chord-note-remove", meta: { timeSec: 1.201, notes: ["C4"] } },
      { ts: 1000, cls: "chord-note-add", meta: { timeSec: 1.203, notes: ["C4"] } }, // same bucket+note
      { ts: 2000, cls: "chord-note-add", meta: { timeSec: 1.203, notes: ["D4"] } }, // same bucket, diff note
      { ts: 3000, cls: "event-remove", meta: { timeSec: 9.0, notes: ["C4", "E4"] } },
      { ts: 4000, cls: "undo" }, // no meta -> not counted
    ]);
    // (1.20,C4) + (1.20,D4) + (9.00,C4) + (9.00,E4) = 4 distinct
    expect(summary.touchedNoteCount).toBe(4);
  });

  it("returns zeroed summary for an empty log", () => {
    const summary = summarizeOpLog([]);
    expect(summary.totalCount).toBe(0);
    expect(summary.activeTimeSec).toBe(0);
    expect(summary.wallTimeSec).toBe(0);
    expect(summary.touchedNoteCount).toBe(0);
    expect(summary.firstTs).toBeNull();
    expect(summary.lastTs).toBeNull();
  });
});
