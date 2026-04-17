import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  createReviewSession,
  isReviewSessionStorageAvailable,
  loadReviewSession,
  removeReviewSession,
  saveReviewSession,
  updateReviewSessionUiState,
} from "@/lib/reviewSession";
import { TranscriptionCapture } from "@/lib/api";
import { InstrumentTuning, TranscriptionResult } from "@/lib/types";

class FakeStorage {
  private readonly store = new Map<string, string>();

  get length() {
    return this.store.size;
  }

  clear() {
    this.store.clear();
  }

  getItem(key: string) {
    return this.store.get(key) ?? null;
  }

  key(index: number) {
    return [...this.store.keys()][index] ?? null;
  }

  removeItem(key: string) {
    this.store.delete(key);
  }

  setItem(key: string, value: string) {
    this.store.set(key, value);
  }
}

function buildTuning(): InstrumentTuning {
  return {
    id: "kalimba-17-c",
    name: "17 Key C Major",
    keyCount: 17,
    notes: [
      { key: 1, noteName: "C4", frequency: 261.63 },
      { key: 2, noteName: "E4", frequency: 329.63 },
    ],
  };
}

function buildResult(transactionId: string | null = "test-txn-00000000-0000-0000-0000-000000000001"): TranscriptionResult {
  return {
    transactionId,
    instrumentTuning: buildTuning(),
    tempo: 96,
    warnings: [],
    notationViews: {
      western: ["C4 E4"],
      numbered: ["1 3"],
      verticalDoReMi: [["ド", "ミ"]],
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
    ],
  };
}

function buildCapture(): TranscriptionCapture {
  const tuning = buildTuning();
  return {
    generatedAt: "2026-03-31T00:00:00.000Z",
    audioWav: new Blob(["wav"], { type: "audio/wav" }),
    requestPayload: {
      capturedAt: "2026-03-31T00:00:00.000Z",
      scenario: "manual-test",
      expectedNote: null,
      expectedPerformance: null,
      memo: null,
      captureIntent: null,
      sourceProfile: "acoustic_real",
      tuning,
      audio: {
        sampleRate: 44100,
        channels: 1,
        durationSec: 1.25,
        mimeType: "audio/wav",
        sizeBytes: 1234,
      },
    },
    responsePayload: buildResult(),
  };
}

describe("reviewSession", () => {
  const fakeStorage = new FakeStorage();

  beforeEach(() => {
    Object.defineProperty(window, "localStorage", {
      value: fakeStorage,
      configurable: true,
    });
    fakeStorage.clear();
  });

  afterEach(() => {
    fakeStorage.clear();
  });

  it("creates a review session from capture snapshots", () => {
    const session = createReviewSession({
      capture: buildCapture(),
      acquisitionMode: "live_mic",
      notationMode: "vertical",
      activeEventId: "evt-1",
    });

    expect(session.sessionVersion).toBe(1);
    expect(session.acquisitionMode).toBe("live_mic");
    expect(session.tuning.name).toBe("17 Key C Major");
    expect(session.responseSnapshot.events[0]?.id).toBe("evt-1");
    expect(session.activeEventId).toBe("evt-1");
  });

  it("round-trips a saved review session through localStorage", () => {
    const session = createReviewSession({
      capture: buildCapture(),
      acquisitionMode: "live_mic",
      notationMode: "vertical",
      activeEventId: "evt-1",
    });

    saveReviewSession(session);

    expect(loadReviewSession(session.sessionId)).toEqual(session);
  });

  it("returns null for malformed or invalid stored payloads", () => {
    fakeStorage.setItem("kalimba:review-session:broken-json", "{");
    fakeStorage.setItem(
      "kalimba:review-session:invalid-shape",
      JSON.stringify({ sessionVersion: 99, sessionId: "bad" }),
    );

    expect(loadReviewSession("broken-json")).toBeNull();
    expect(loadReviewSession("invalid-shape")).toBeNull();
  });

  it("reports localStorage availability and supports removal", () => {
    const session = createReviewSession({
      capture: buildCapture(),
      acquisitionMode: "live_mic",
      notationMode: "vertical",
      activeEventId: "evt-1",
    });

    expect(isReviewSessionStorageAvailable()).toBe(true);

    saveReviewSession(session);
    removeReviewSession(session.sessionId);

    expect(loadReviewSession(session.sessionId)).toBeNull();
  });

  it("updates persisted notation mode and active event without replacing the full snapshot", () => {
    const session = createReviewSession({
      capture: buildCapture(),
      acquisitionMode: "live_mic",
      notationMode: "vertical",
      activeEventId: "evt-1",
    });

    saveReviewSession(session);
    updateReviewSessionUiState(session.sessionId, {
      notationMode: "western",
      activeEventId: null,
    });

    expect(loadReviewSession(session.sessionId)).toEqual({
      ...session,
      notationMode: "western",
      activeEventId: null,
    });
  });

  it("preserves transactionId through round-trip", () => {
    const capture = buildCapture();
    const session = createReviewSession({
      capture,
      acquisitionMode: "live_mic",
      notationMode: "vertical",
      activeEventId: "evt-1",
    });

    expect(session.transactionId).toBe("test-txn-00000000-0000-0000-0000-000000000001");

    saveReviewSession(session);
    const loaded = loadReviewSession(session.sessionId);
    expect(loaded?.transactionId).toBe("test-txn-00000000-0000-0000-0000-000000000001");
  });

  it("accepts null transactionId", () => {
    const capture = buildCapture();
    capture.responsePayload = buildResult(null);
    const session = createReviewSession({
      capture,
      acquisitionMode: "live_mic",
      notationMode: "vertical",
      activeEventId: "evt-1",
    });

    expect(session.transactionId).toBeNull();

    saveReviewSession(session);
    const loaded = loadReviewSession(session.sessionId);
    expect(loaded?.transactionId).toBeNull();
  });
});
