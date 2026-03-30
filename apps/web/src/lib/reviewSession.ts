import { TranscriptionCapture } from "@/lib/api";
import {
  AcquisitionMode,
  InstrumentProfileRef,
  NotationMode,
  RecordingProfileRef,
  TranscriptionReviewSession,
} from "@/lib/types";

const REVIEW_SESSION_PREFIX = "kalimba:review-session:";

function buildStorageKey(sessionId: string) {
  return `${REVIEW_SESSION_PREFIX}${sessionId}`;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isStringOrNull(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function isReviewSession(value: unknown): value is TranscriptionReviewSession {
  if (!isObject(value)) {
    return false;
  }

  if (value.sessionVersion !== 1) {
    return false;
  }

  if (
    typeof value.sessionId !== "string"
    || typeof value.createdAt !== "string"
    || (value.acquisitionMode !== "live_mic" && value.acquisitionMode !== "uploaded_file")
    || !isObject(value.tuning)
    || !isObject(value.requestSnapshot)
    || !isObject(value.responseSnapshot)
    || (value.notationMode !== "vertical" && value.notationMode !== "numbered" && value.notationMode !== "western")
    || !isStringOrNull(value.activeEventId)
  ) {
    return false;
  }

  if (!(value.instrumentProfile === null || isObject(value.instrumentProfile))) {
    return false;
  }

  if (!(value.recordingProfile === null || isObject(value.recordingProfile))) {
    return false;
  }

  return value.editedDraft === null || isObject(value.editedDraft);
}

export function createReviewSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `review-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function isReviewSessionStorageAvailable() {
  if (typeof window === "undefined") {
    return false;
  }

  try {
    const probeKey = `${REVIEW_SESSION_PREFIX}probe`;
    window.sessionStorage.setItem(probeKey, "1");
    window.sessionStorage.removeItem(probeKey);
    return true;
  } catch {
    return false;
  }
}

type CreateReviewSessionOptions = {
  capture: TranscriptionCapture;
  acquisitionMode: AcquisitionMode;
  notationMode: NotationMode;
  activeEventId: string | null;
  instrumentProfile?: InstrumentProfileRef | null;
  recordingProfile?: RecordingProfileRef | null;
};

export function createReviewSession({
  capture,
  acquisitionMode,
  notationMode,
  activeEventId,
  instrumentProfile = null,
  recordingProfile = null,
}: CreateReviewSessionOptions): TranscriptionReviewSession {
  return {
    sessionVersion: 1,
    sessionId: createReviewSessionId(),
    createdAt: capture.generatedAt,
    acquisitionMode,
    tuning: capture.requestPayload.tuning,
    instrumentProfile,
    recordingProfile,
    requestSnapshot: capture.requestPayload,
    responseSnapshot: capture.responsePayload,
    notationMode,
    activeEventId,
    editedDraft: null,
  };
}

export function saveReviewSession(session: TranscriptionReviewSession) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.setItem(buildStorageKey(session.sessionId), JSON.stringify(session));
}

type UpdateReviewSessionUiState = {
  notationMode?: NotationMode;
  activeEventId?: string | null;
};

export function loadReviewSession(sessionId: string) {
  if (typeof window === "undefined") {
    return null;
  }

  const raw = window.sessionStorage.getItem(buildStorageKey(sessionId));
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    return isReviewSession(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function updateReviewSessionUiState(sessionId: string, nextState: UpdateReviewSessionUiState) {
  const current = loadReviewSession(sessionId);
  if (!current) {
    return;
  }

  saveReviewSession({
    ...current,
    ...(nextState.notationMode !== undefined ? { notationMode: nextState.notationMode } : {}),
    ...(nextState.activeEventId !== undefined ? { activeEventId: nextState.activeEventId } : {}),
  });
}

export function removeReviewSession(sessionId: string) {
  if (typeof window === "undefined") {
    return;
  }
  window.sessionStorage.removeItem(buildStorageKey(sessionId));
}
