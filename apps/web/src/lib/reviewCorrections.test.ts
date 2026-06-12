import { describe, expect, it } from "vitest";

import {
  activeEvents,
  addNote,
  buildInitialState,
  hasActiveEventAt,
  insertEvent,
  removeNote,
  resolveScoreNote,
  restoreStateFromCorrections,
  buildKnownNoteIndex,
  toCorrectionsPayload,
  toggleRemoved,
} from "@/lib/reviewCorrections";
import { CorrectionsPayload, ScoreNote, TranscriptionResult } from "@/lib/types";

function note(pitchClass: string, octave: number, frequency: number, key = 1): ScoreNote {
  return {
    key,
    pitchClass,
    octave,
    labelDoReMi: pitchClass,
    labelNumber: "1",
    frequency,
  };
}

const D4 = note("D", 4, 293.665, 8);
const D5 = note("D", 5, 587.33, 13);
const F4 = note("F", 4, 349.228, 7);

function makeResult(): TranscriptionResult {
  return {
    transactionId: "tx-1",
    instrumentTuning: {
      id: "kalimba-17-c",
      name: "17 Key C Major",
      keyCount: 17,
      notes: [
        { key: 7, noteName: "F4", frequency: 349.228 },
        { key: 8, noteName: "D4", frequency: 293.665 },
        { key: 9, noteName: "C4", frequency: 261.626 },
        { key: 13, noteName: "D5", frequency: 587.33 },
      ],
    },
    tempo: 120,
    events: [
      {
        id: "evt-1",
        startBeat: 0,
        durationBeat: 1,
        startTimeSec: 4.445,
        notes: [D4, D5],
        isGlissLike: false,
        gesture: "ambiguous",
        alternateGroupings: [
          {
            combinesWith: null,
            combinedNotes: null,
            splitInto: null,
            alternateNote: F4,
            reason: "soft_rejected:score-below-threshold",
            confidence: 0.7,
          },
        ],
      },
      {
        id: "evt-2",
        startBeat: 1,
        durationBeat: 1,
        startTimeSec: 7.069,
        notes: [D4, D5],
        isGlissLike: false,
        gesture: "ambiguous",
      },
    ],
    candidateSlots: [],
    notationViews: { western: [], numbered: [], verticalDoReMi: [] },
    warnings: [],
  };
}

describe("buildInitialState", () => {
  it("maps recognizer events with origin recognizer", () => {
    const state = buildInitialState(makeResult());
    expect(state.events).toHaveLength(2);
    expect(state.events[0].origin).toBe("recognizer");
    expect(state.events[0].sourceEventId).toBe("evt-1");
  });
});

describe("note editing", () => {
  it("addNote sorts by frequency and marks edited", () => {
    const state = addNote(buildInitialState(makeResult()), "evt-1", F4);
    const event = state.events[0];
    expect(event.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D4", "F4", "D5"]);
    expect(event.origin).toBe("edited");
  });

  it("addNote is a no-op for duplicate notes", () => {
    const state = addNote(buildInitialState(makeResult()), "evt-1", D4);
    expect(state.events[0].notes).toHaveLength(2);
    expect(state.events[0].origin).toBe("recognizer");
  });

  it("removeNote keeps at least one note", () => {
    let state = removeNote(buildInitialState(makeResult()), "evt-1", "D5");
    expect(state.events[0].notes.map((n) => n.pitchClass)).toEqual(["D"]);
    expect(state.events[0].origin).toBe("edited");
    state = removeNote(state, "evt-1", "D4");
    expect(state.events[0].notes).toHaveLength(1);
  });
});

describe("event-level operations", () => {
  it("toggleRemoved flips removal and excludes from payload", () => {
    const state = toggleRemoved(buildInitialState(makeResult()), "evt-1");
    expect(activeEvents(state)).toHaveLength(1);
    const payload = toCorrectionsPayload(state);
    expect(payload.events).toHaveLength(1);
    expect(payload.events[0].timeSec).toBeCloseTo(7.069);
  });

  it("insertEvent keeps time order and assigns ins- ids", () => {
    const state = insertEvent(buildInitialState(makeResult()), 5.5, [F4], "inserted-manual");
    const ids = state.events.map((e) => e.id);
    expect(ids).toEqual(["evt-1", "ins-1", "evt-2"]);
  });
});

describe("payload roundtrip", () => {
  it("restoreStateFromCorrections reconciles edits, removals, insertions", () => {
    const result = makeResult();
    const corrections: CorrectionsPayload = {
      version: 1,
      events: [
        { timeSec: 4.445, notes: ["D4", "F4", "D5"], origin: "edited" },
        { timeSec: 5.5, notes: ["F4"], origin: "inserted-manual" },
      ],
    };
    const state = restoreStateFromCorrections(result, corrections);
    const active = activeEvents(state);
    expect(active).toHaveLength(2);
    expect(active[0].notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D4", "F4", "D5"]);
    expect(active[0].origin).toBe("edited");
    expect(active[1].id).toBe("ins-1");
    const removed = state.events.find((e) => e.id === "evt-2");
    expect(removed?.removed).toBe(true);

    expect(toCorrectionsPayload(state)).toEqual(corrections);
  });
});

describe("hasActiveEventAt (candidateSlot 表示抑制)", () => {
  it("detects events inserted from a slot in the current session", () => {
    const state = insertEvent(buildInitialState(makeResult()), 5.5, [F4], "inserted-slot");
    expect(hasActiveEventAt(state, 5.5)).toBe(true);
    expect(hasActiveEventAt(state, 6.5)).toBe(false);
  });

  it("detects inserted-slot events restored from saved corrections", () => {
    const result = makeResult();
    const state = restoreStateFromCorrections(result, {
      version: 1,
      events: [
        { timeSec: 4.445, notes: ["D4", "D5"], origin: "recognizer" },
        { timeSec: 7.069, notes: ["D4", "D5"], origin: "recognizer" },
        { timeSec: 5.5, notes: ["F4"], origin: "inserted-slot" },
      ],
    });
    // 復元された inserted-slot イベントの時刻では slot を再表示しない
    expect(hasActiveEventAt(state, 5.5)).toBe(true);
  });

  it("ignores removed events", () => {
    const state = toggleRemoved(buildInitialState(makeResult()), "evt-1");
    expect(hasActiveEventAt(state, 4.445)).toBe(false);
  });
});

describe("resolveScoreNote", () => {
  it("prefers known notes from the result", () => {
    const result = makeResult();
    const index = buildKnownNoteIndex(result);
    expect(resolveScoreNote("F4", index, result.instrumentTuning)).toBe(F4);
  });

  it("falls back to tuning definition for unseen notes", () => {
    const result = makeResult();
    const index = buildKnownNoteIndex(result);
    const c4 = resolveScoreNote("C4", index, result.instrumentTuning);
    expect(c4?.pitchClass).toBe("C");
    expect(c4?.labelDoReMi).toBe("ド");
    expect(c4?.frequency).toBeCloseTo(261.626);
  });

  it("returns null for notes outside the tuning", () => {
    const result = makeResult();
    const index = buildKnownNoteIndex(result);
    expect(resolveScoreNote("G9", index, result.instrumentTuning)).toBeNull();
  });
});
