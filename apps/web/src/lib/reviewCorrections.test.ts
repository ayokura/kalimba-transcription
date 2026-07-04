import { describe, expect, it } from "vitest";

import {
  activeEvents,
  addNote,
  buildInitialState,
  hasActiveEventAt,
  insertEvent,
  removeNote,
  replaceNote,
  resolveScoreNote,
  restoreStateFromCorrections,
  buildKnownNoteIndex,
  setEventTime,
  toCorrectionsPayload,
  toggleAccompanimentOnly,
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
        { key: 6, noteName: "Bb4", frequency: 466.164 },
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

describe("replaceNote (ワンタップ置換)", () => {
  it("単音イベントでも 1 操作で置換できる (removeNote の最後の 1 音ガードに縛られない)", () => {
    const result = makeResult();
    let state = buildInitialState(result);
    state = removeNote(state, "evt-1", "D5"); // D4 単音にする
    const replaced = replaceNote(state, "evt-1", "D4", F4);
    const evt1 = replaced.events.find((e) => e.id === "evt-1");
    expect(evt1?.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["F4"]);
    expect(evt1?.origin).toBe("edited");
  });

  it("和音では対象の 1 音だけを入れ替え、frequency 順を保つ", () => {
    const state = buildInitialState(makeResult());
    const replaced = replaceNote(state, "evt-1", "D5", F4); // [D4, D5] → [D4, F4]
    const evt1 = replaced.events.find((e) => e.id === "evt-1");
    expect(evt1?.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D4", "F4"]);
  });

  it("置換先が既に含まれる場合は旧音の除去に縮退する", () => {
    const state = buildInitialState(makeResult());
    const replaced = replaceNote(state, "evt-1", "D4", D5); // [D4, D5] + (D4→D5) → [D5]
    const evt1 = replaced.events.find((e) => e.id === "evt-1");
    expect(evt1?.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D5"]);
  });

  it("旧音が存在しない・同一音への置換は no-op (state 参照を保つ)", () => {
    const state = buildInitialState(makeResult());
    expect(replaceNote(state, "evt-1", "G4", F4)).toBe(state);
    expect(replaceNote(state, "evt-1", "D4", D4)).toBe(state);
  });
});

describe("setEventTime (タイミング微調整)", () => {
  it("timeSec を書き換えて時刻順に再ソートし、edited へ昇格する", () => {
    const state = buildInitialState(makeResult());
    const updated = setEventTime(state, "evt-1", 8.0); // evt-2 (7.069) を追い越す
    expect(updated.events.map((e) => e.id)).toEqual(["evt-2", "evt-1"]);
    const evt1 = updated.events.find((e) => e.id === "evt-1");
    expect(evt1?.timeSec).toBe(8.0);
    expect(evt1?.origin).toBe("edited");
  });

  it("負の時刻は 0 にクランプする", () => {
    const state = buildInitialState(makeResult());
    const updated = setEventTime(state, "evt-1", -0.5);
    expect(updated.events.find((e) => e.id === "evt-1")?.timeSec).toBe(0);
  });

  it("同一時刻への設定は no-op (state 参照を保つ)", () => {
    const state = buildInitialState(makeResult());
    expect(setEventTime(state, "evt-1", 4.445)).toBe(state);
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

  it("toggleAccompanimentOnly round-trips through payload and restore", () => {
    const result = makeResult();
    const state = toggleAccompanimentOnly(buildInitialState(result), "evt-1");
    expect(state.events.find((e) => e.id === "evt-1")?.accompanimentOnly).toBe(true);

    const payload = toCorrectionsPayload(state);
    expect(payload.events[0].accompanimentOnly).toBe(true);
    // フラグなしイベントには key 自体を含めない (旧スキーマ互換)
    expect("accompanimentOnly" in payload.events[1]).toBe(false);

    const restored = restoreStateFromCorrections(result, payload);
    expect(restored.events.find((e) => e.id === "evt-1")?.accompanimentOnly).toBe(true);
    expect(restored.events.find((e) => e.id === "evt-2")?.accompanimentOnly).toBe(false);
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

  it("resolves flat note names from the tuning (e.g. Bb major)", () => {
    const result = makeResult();
    const index = buildKnownNoteIndex(result);
    const bb4 = resolveScoreNote("Bb4", index, result.instrumentTuning);
    expect(bb4?.pitchClass).toBe("Bb");
    expect(bb4?.labelDoReMi).toBe("シb");
    expect(bb4?.frequency).toBeCloseTo(466.164);
  });

  it("returns null for notes outside the tuning", () => {
    const result = makeResult();
    const index = buildKnownNoteIndex(result);
    expect(resolveScoreNote("G9", index, result.instrumentTuning)).toBeNull();
  });
});

describe("restoreStateFromCorrections — 再採譜への頑健性 (2026-07-02 監査)", () => {
  it("recognizer の onset が数十 ms ずれても corrections が全滅しない", () => {
    const result = makeResult();
    // 保存時点 (旧 recognizer): 4.445 / 7.069。再採譜で +30ms / -20ms ずれた想定。
    result.events[0].startTimeSec = 4.475;
    result.events[1].startTimeSec = 7.049;
    const corrections: CorrectionsPayload = {
      version: 1,
      events: [
        { timeSec: 4.445, notes: ["D4", "F4", "D5"], origin: "edited" },
        { timeSec: 7.069, notes: ["D4", "D5"], origin: "recognizer" },
      ],
    };
    const state = restoreStateFromCorrections(result, corrections);
    const active = activeEvents(state);
    expect(active).toHaveLength(2);
    // 挿入イベント化していない (recognizer 枠に束ねられている)
    expect(active.every((e) => e.sourceEventId !== null)).toBe(true);
    expect(active[0].notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D4", "F4", "D5"]);
    expect(active[0].origin).toBe("edited");
    // timeline は現在の transcription の時刻に追従する
    expect(active[0].timeSec).toBe(4.475);
    expect(state.events.some((e) => e.removed)).toBe(false);
  });

  it("50ms を超えるずれは従来どおり removed + 挿入に分解される", () => {
    const result = makeResult();
    result.events[0].startTimeSec = 4.6; // 155ms ずれ
    const corrections: CorrectionsPayload = {
      version: 1,
      events: [{ timeSec: 4.445, notes: ["D4", "D5"], origin: "recognizer" }],
    };
    const state = restoreStateFromCorrections(result, corrections);
    expect(state.events.find((e) => e.id === "evt-1")?.removed).toBe(true);
    expect(state.events.some((e) => e.id === "ins-1")).toBe(true);
  });

  it("密集区間では最近傍優先で誤束ねしない", () => {
    const result = makeResult();
    // 60ms 間隔の隣接イベントが双方 +20ms ずれた想定
    result.events[0].startTimeSec = 4.465; // 旧 4.445
    result.events[1].startTimeSec = 4.525; // 旧 4.505
    const corrections: CorrectionsPayload = {
      version: 1,
      events: [
        { timeSec: 4.445, notes: ["F4"], origin: "edited" },
        { timeSec: 4.505, notes: ["D4"], origin: "edited" },
      ],
    };
    const state = restoreStateFromCorrections(result, corrections);
    const byId = new Map(state.events.map((e) => [e.id, e]));
    expect(byId.get("evt-1")?.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["F4"]);
    expect(byId.get("evt-2")?.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D4"]);
    expect(state.events.some((e) => e.removed)).toBe(false);
  });

  it("tight 一致が relaxed 束ねより常に優先される", () => {
    const result = makeResult();
    // evt-1 (4.445) に tight 一致する correction と、40ms 先の correction が併存
    const corrections: CorrectionsPayload = {
      version: 1,
      events: [
        { timeSec: 4.485, notes: ["F4"], origin: "inserted-manual" },
        { timeSec: 4.445, notes: ["D4", "D5"], origin: "recognizer" },
      ],
    };
    const state = restoreStateFromCorrections(result, corrections);
    const evt1 = state.events.find((e) => e.id === "evt-1");
    expect(evt1?.notes.map((n) => `${n.pitchClass}${n.octave}`)).toEqual(["D4", "D5"]);
    expect(evt1?.origin).toBe("recognizer");
    // 4.485 の correction は挿入イベントとして残る
    expect(state.events.some((e) => e.id.startsWith("ins-") && e.timeSec === 4.485)).toBe(true);
  });
});
