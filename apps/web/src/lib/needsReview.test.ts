import { describe, expect, it } from "vitest";

import { needsReviewReasons } from "@/lib/needsReview";
import { CandidateSlot, ScoreEvent, ScoreNote } from "@/lib/types";

const NOTE_C4: ScoreNote = {
  key: 1,
  pitchClass: "C",
  octave: 4,
  labelDoReMi: "ド",
  labelNumber: "1",
  frequency: 261.63,
};

const NOTE_E4: ScoreNote = { ...NOTE_C4, key: 2, pitchClass: "E", frequency: 329.63 };

function makeEvent(overrides: Partial<ScoreEvent> = {}): ScoreEvent {
  return {
    id: "evt-1",
    startBeat: 0,
    durationBeat: 1,
    startTimeSec: 2.0,
    durationSec: 0.5,
    notes: [NOTE_C4],
    isGlissLike: false,
    gesture: "ambiguous",
    alternateGroupings: null,
    ...overrides,
  };
}

function makeSlot(startTime: number): CandidateSlot {
  return {
    startTime,
    endTime: startTime + 0.3,
    primaryNote: NOTE_C4,
    candidates: [NOTE_C4],
    dropReason: "residual-decay-no-reattack",
    confidence: 0.15,
  };
}

describe("needsReviewReasons", () => {
  it("単音 + gesture=ambiguous だけでは要確認にしない (68% に付く既定値のため)", () => {
    expect(needsReviewReasons(makeEvent(), [])).toEqual([]);
  });

  it("複音 + gesture=ambiguous はグルーピング曖昧として要確認", () => {
    const reasons = needsReviewReasons(makeEvent({ notes: [NOTE_C4, NOTE_E4] }), []);
    expect(reasons.map((r) => r.key)).toEqual(["ambiguous-grouping"]);
  });

  it("複音でも gesture が確定していれば flag しない", () => {
    const reasons = needsReviewReasons(
      makeEvent({ notes: [NOTE_C4, NOTE_E4], gesture: "strict_chord" }),
      [],
    );
    expect(reasons).toEqual([]);
  });

  it("confidence >= 0.5 の alternateGrouping で要確認", () => {
    const grouping = {
      combinesWith: null,
      combinedNotes: null,
      splitInto: null,
      alternateNote: NOTE_E4,
      reason: "octave-neighbor",
      confidence: 0.5,
    };
    const reasons = needsReviewReasons(makeEvent({ alternateGroupings: [grouping] }), []);
    expect(reasons.map((r) => r.key)).toEqual(["strong-alternate"]);
  });

  it("confidence < 0.5 の alternateGrouping は flag しない (median=0.35 の弱候補)", () => {
    const grouping = {
      combinesWith: null,
      combinedNotes: null,
      splitInto: null,
      alternateNote: NOTE_E4,
      reason: "octave-neighbor",
      confidence: 0.35,
    };
    expect(needsReviewReasons(makeEvent({ alternateGroupings: [grouping] }), [])).toEqual([]);
  });

  it("±0.6s 以内の candidateSlot で要確認、それより遠いと flag しない", () => {
    expect(
      needsReviewReasons(makeEvent(), [makeSlot(2.55)]).map((r) => r.key),
    ).toEqual(["adjacent-slot"]);
    expect(needsReviewReasons(makeEvent(), [makeSlot(2.7)])).toEqual([]);
  });

  it("lowConfidenceReason 付き event は打鍵証拠弱として要確認 (S5 agenda 2)", () => {
    const reasons = needsReviewReasons(
      makeEvent({ lowConfidenceReason: "onset-gate-no-evidence" }),
      [],
    );
    expect(reasons.map((r) => r.key)).toEqual(["low-confidence-gate"]);
  });

  it("複数条件は全て列挙する", () => {
    const grouping = {
      combinesWith: null,
      combinedNotes: null,
      splitInto: null,
      alternateNote: NOTE_E4,
      reason: "octave-neighbor",
      confidence: 0.7,
    };
    const reasons = needsReviewReasons(
      makeEvent({ notes: [NOTE_C4, NOTE_E4], alternateGroupings: [grouping] }),
      [makeSlot(1.8)],
    );
    expect(reasons.map((r) => r.key)).toEqual([
      "strong-alternate",
      "ambiguous-grouping",
      "adjacent-slot",
    ]);
  });
});
