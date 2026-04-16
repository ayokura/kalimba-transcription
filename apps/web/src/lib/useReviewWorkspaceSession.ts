"use client";

import { useEffect, useMemo, useState } from "react";

import { fetchTranscriptionAudioBlob } from "@/lib/api";
import { loadReviewAudio } from "@/lib/reviewAudioStore";
import { isReviewSessionStorageAvailable, loadReviewSession, updateReviewSessionUiState } from "@/lib/reviewSession";
import { NotationMode } from "@/lib/types";

export function useReviewWorkspaceSession(sessionId: string) {
  const storageAvailable = isReviewSessionStorageAvailable();
  const session = useMemo(
    () => (storageAvailable && sessionId ? loadReviewSession(sessionId) : null),
    [sessionId, storageAvailable],
  );
  const audioBlob = useMemo(
    () => (storageAvailable && sessionId ? loadReviewAudio(sessionId) : null),
    [sessionId, storageAvailable],
  );
  const sourceProfile = typeof session?.requestSnapshot.sourceProfile === "string"
    ? session.requestSnapshot.sourceProfile
    : "unknown-profile";
  const [notationMode, setNotationMode] = useState<NotationMode>(session?.notationMode ?? "vertical");
  const [activeEventId, setActiveEventId] = useState<string | null>(session?.activeEventId ?? null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const activeEvent = session?.responseSnapshot.events.find((event) => event.id === activeEventId)
    ?? session?.responseSnapshot.events[0]
    ?? null;

  useEffect(() => {
    if (audioBlob) {
      const nextUrl = URL.createObjectURL(audioBlob);
      setAudioUrl(nextUrl);
      return () => {
        URL.revokeObjectURL(nextUrl);
      };
    }

    if (!session?.transactionId) {
      setAudioUrl(null);
      return;
    }

    let cancelled = false;
    fetchTranscriptionAudioBlob(session.transactionId)
      .then((blob) => {
        if (cancelled) return;
        const nextUrl = URL.createObjectURL(blob);
        setAudioUrl(nextUrl);
      })
      .catch(() => {
        if (!cancelled) setAudioUrl(null);
      });
    return () => {
      cancelled = true;
    };
  }, [audioBlob, session?.transactionId]);

  useEffect(() => {
    setNotationMode(session?.notationMode ?? "vertical");
    setActiveEventId(session?.activeEventId ?? null);
  }, [sessionId, session?.notationMode, session?.activeEventId]);

  useEffect(() => {
    if (!session || !storageAvailable || !sessionId) {
      return;
    }

    if (notationMode === session.notationMode && activeEventId === session.activeEventId) {
      return;
    }

    updateReviewSessionUiState(sessionId, {
      notationMode,
      activeEventId,
    });
  }, [activeEventId, notationMode, session, sessionId, storageAvailable]);

  return {
    activeEvent,
    activeEventId,
    audioBlob,
    audioUrl,
    notationMode,
    session,
    sourceProfile,
    storageAvailable,
    setActiveEventId,
    setNotationMode,
  };
}
