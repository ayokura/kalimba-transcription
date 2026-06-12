import "fake-indexeddb/auto";

import { beforeEach, describe, expect, it } from "vitest";

import {
  clearPendingRecordings,
  deletePendingRecording,
  loadLatestPendingRecording,
  savePendingRecording,
  type PendingRecording,
} from "./pendingRecordingStore";

function entry(id: string, createdAt: number): PendingRecording {
  return {
    id,
    blob: new Blob(["audio-bytes"], { type: "audio/wav" }),
    source: "mic",
    tuningId: "kalimba-17-c",
    createdAt,
  };
}

describe("pendingRecordingStore", () => {
  beforeEach(async () => {
    await clearPendingRecordings();
  });

  it("saves and loads the latest pending recording", async () => {
    const now = Date.now();
    await savePendingRecording(entry("a", now - 5000));
    await savePendingRecording(entry("b", now - 1000));

    const latest = await loadLatestPendingRecording(now);
    expect(latest?.id).toBe("b");
    expect(latest?.tuningId).toBe("kalimba-17-c");
    expect(latest?.source).toBe("mic");
  });

  it("returns null when empty", async () => {
    expect(await loadLatestPendingRecording()).toBeNull();
  });

  it("delete removes the entry", async () => {
    const now = Date.now();
    await savePendingRecording(entry("a", now));
    await deletePendingRecording("a");
    expect(await loadLatestPendingRecording(now)).toBeNull();
  });

  it("prunes entries older than 24h", async () => {
    const now = Date.now();
    await savePendingRecording(entry("old", now - 25 * 60 * 60 * 1000));
    await savePendingRecording(entry("fresh", now - 60 * 1000));

    const latest = await loadLatestPendingRecording(now);
    expect(latest?.id).toBe("fresh");

    // 期限切れ entry は load 時に削除済み
    await deletePendingRecording("fresh");
    expect(await loadLatestPendingRecording(now)).toBeNull();
  });
});
