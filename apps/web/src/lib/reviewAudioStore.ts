"use client";

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
