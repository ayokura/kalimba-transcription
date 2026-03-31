"use client";

// Audio blobs stay in-memory on purpose. The review session JSON goes to
// sessionStorage, but the raw audio is kept out of that payload to avoid size
// pressure and serialization issues. This means whole-track playback is
// same-tab only until a later persistence strategy is introduced.
const reviewAudioStore = new Map<string, Blob>();

export function saveReviewAudio(sessionId: string, audioBlob: Blob) {
  reviewAudioStore.set(sessionId, audioBlob);
}

export function loadReviewAudio(sessionId: string) {
  return reviewAudioStore.get(sessionId) ?? null;
}

export function removeReviewAudio(sessionId: string) {
  reviewAudioStore.delete(sessionId);
}
