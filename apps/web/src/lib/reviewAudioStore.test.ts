import { afterEach, describe, expect, it } from "vitest";

import { loadReviewAudio, removeReviewAudio, saveReviewAudio } from "@/lib/reviewAudioStore";

describe("reviewAudioStore", () => {
  const sessionId = "review-session-1";

  afterEach(() => {
    removeReviewAudio(sessionId);
  });

  it("stores and loads the same blob instance", () => {
    const blob = new Blob(["hello"], { type: "audio/wav" });

    saveReviewAudio(sessionId, blob);

    expect(loadReviewAudio(sessionId)).toBe(blob);
  });

  it("returns null after removal", () => {
    saveReviewAudio(sessionId, new Blob(["hello"], { type: "audio/wav" }));

    removeReviewAudio(sessionId);

    expect(loadReviewAudio(sessionId)).toBeNull();
  });
});
