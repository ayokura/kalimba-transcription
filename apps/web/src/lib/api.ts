import {
  CorrectionsPayload,
  InstrumentTuning,
  ReviewQueueEntry,
  ReviewStatusPayload,
  ReviewStatusValue,
  TranscriptionResult,
} from "@/lib/types";
import { WavMetadata, toWavWithMetadata } from "@/lib/audio";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export type CaptureAssessmentStatus = "completed" | "pending" | "rerecord" | "review_needed" | "reference_only";

export type CaptureIntent = "strict_chord" | "slide_chord" | "arpeggio" | "separated_notes" | "unknown";

export type SourceProfile = "acoustic_real" | "app_synth";

export type CaptureAssessment = {
  status: CaptureAssessmentStatus;
  label: string;
  summary: string;
  reason: string;
  mismatchCount: number;
  expectedEventCount: number;
  detectedEventCount: number;
};

export type CaptureAssessmentEvent = {
  index: number;
  expected: string | null;
  detected: string | null;
  matches: boolean;
};

export type CaptureAssessmentDetails = CaptureAssessment & {
  events: CaptureAssessmentEvent[];
  extraEventCount: number;
  missingEventCount: number;
};

export type ManualCaptureExpectedKey = {
  key: number;
  noteName: string;
};

export type ManualCaptureExpectedPart = {
  keys: ManualCaptureExpectedKey[];
  display?: string;
  intent?: CaptureIntent | null;
};

export type ManualCaptureExpectedEvent = {
  index: number;
  keys: ManualCaptureExpectedKey[];
  display: string;
  intent?: CaptureIntent | null;
  parts?: ManualCaptureExpectedPart[] | null;
};

export type ManualCaptureExpectedPerformance = {
  source: "clickable-kalimba-ui" | "adversarial-menu";
  version: 1;
  summary: string;
  defaultCaptureIntent?: CaptureIntent | null;
  events: ManualCaptureExpectedEvent[];
};

export type ManualCaptureRequestPayload = {
  capturedAt: string;
  scenario: string;
  expectedNote: string | null;
  expectedPerformance: ManualCaptureExpectedPerformance | null;
  memo: string | null;
  captureIntent: CaptureIntent | null;
  sourceProfile: SourceProfile;
  midPerformanceStart?: boolean;
  midPerformanceEnd?: boolean;
  tuning: InstrumentTuning;
  audio: WavMetadata & {
    mimeType: string;
    sizeBytes: number;
  };
};

export type TranscriptionCapture = {
  generatedAt: string;
  audioWav: Blob;
  requestPayload: ManualCaptureRequestPayload;
  responsePayload: TranscriptionResult;
};

export type CreateTranscriptionOptions = {
  scenario?: string;
  expectedNote?: string;
  expectedPerformance?: ManualCaptureExpectedPerformance | null;
  memo?: string;
  captureIntent?: CaptureIntent | null;
  sourceProfile?: SourceProfile;
  midPerformanceStart?: boolean;
  midPerformanceEnd?: boolean;
  force?: boolean;
  /** 録音デバイス推定用の生シグナル (mic label / platform 等) — request.json の client.device へ */
  clientInfo?: Record<string, unknown> | null;
};

export type RecentTranscriptionEntry = {
  transactionId: string;
  createdAt: number;
  tuningId: string | null;
  tuningName: string | null;
  eventCount: number;
  audioSha256: string | null;
};

export async function lookupTranscriptionByHash(
  audioSha256: string,
  tuningId: string,
): Promise<string | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/transcriptions/by-hash/${audioSha256}?tuning=${encodeURIComponent(tuningId)}`,
    { cache: "no-store" },
  );
  if (response.status === 404) return null;
  if (!response.ok) throw new Error("Failed to check audio hash.");
  const data = (await response.json()) as { transactionId: string };
  return data.transactionId;
}

export async function fetchRecentTranscriptions(limit = 10): Promise<RecentTranscriptionEntry[]> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/recent?limit=${limit}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("Failed to load recent transcriptions.");
  return response.json();
}

export async function fetchTunings(): Promise<InstrumentTuning[]> {
  const response = await fetch(`${API_BASE_URL}/api/tunings`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load tunings.");
  }
  return response.json();
}

export async function createTranscription(file: Blob, tuning: InstrumentTuning): Promise<TranscriptionResult> {
  const capture = await createTranscriptionWithCapture(file, tuning);
  return capture.responsePayload;
}

export async function createTranscriptionWithCapture(
  file: Blob,
  tuning: InstrumentTuning,
  options: CreateTranscriptionOptions = {},
): Promise<TranscriptionCapture> {
  const generatedAt = new Date().toISOString();
  const { wavBlob, metadata } = await toWavWithMetadata(file);
  const formData = new FormData();
  formData.append("file", wavBlob, "recording.wav");
  formData.append("tuning", JSON.stringify(tuning));
  if (options.midPerformanceStart) {
    formData.append("midPerformanceStart", "true");
  }
  if (options.midPerformanceEnd) {
    formData.append("midPerformanceEnd", "true");
  }
  if (options.force) {
    formData.append("force", "true");
  }
  // capture メタデータをサーバー側 request.json にも永続化する (2026-07-04 まで
  // これらはクライアントの requestPayload = Capture Pack ZIP にしか残らず、
  // data/transactions では失われていた)
  const scenarioText = options.scenario?.trim();
  if (scenarioText) {
    formData.append("scenario", scenarioText);
  }
  const expectedNoteText = cleanOptionalText(options.expectedNote);
  if (expectedNoteText) {
    formData.append("expectedNote", expectedNoteText);
  }
  if (options.expectedPerformance) {
    formData.append("expectedPerformance", JSON.stringify(options.expectedPerformance));
  }
  const memoText = cleanOptionalText(options.memo);
  if (memoText) {
    formData.append("memo", memoText);
  }
  if (options.captureIntent) {
    formData.append("captureIntent", options.captureIntent);
  }
  if (options.sourceProfile) {
    formData.append("sourceProfile", options.sourceProfile);
  }
  if (options.clientInfo) {
    formData.append("clientInfo", JSON.stringify(options.clientInfo));
  }

  const response = await fetch(`${API_BASE_URL}/api/transcriptions`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await parseError(response);
    throw new Error(detail);
  }

  const responsePayload = (await response.json()) as TranscriptionResult;
  return {
    generatedAt,
    audioWav: wavBlob,
    requestPayload: {
      capturedAt: generatedAt,
      scenario: options.scenario?.trim() || "manual-test",
      expectedNote: cleanOptionalText(options.expectedNote),
      expectedPerformance: options.expectedPerformance ?? null,
      memo: cleanOptionalText(options.memo),
      captureIntent: options.captureIntent ?? null,
      sourceProfile: options.sourceProfile ?? "acoustic_real",
      ...(options.midPerformanceStart ? { midPerformanceStart: true } : {}),
      ...(options.midPerformanceEnd ? { midPerformanceEnd: true } : {}),
      tuning,
      audio: {
        ...metadata,
        mimeType: wavBlob.type || "audio/wav",
        sizeBytes: wavBlob.size,
      },
    },
    responsePayload,
  };
}

export async function fetchTranscription(transactionId: string): Promise<TranscriptionResult> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/${transactionId}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load transcription.");
  }
  return response.json();
}

export async function fetchTranscriptionAudioBlob(transactionId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/${transactionId}/audio`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load audio.");
  }
  return response.blob();
}

export async function fetchMemo(transactionId: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/${transactionId}/memo`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error("Failed to load memo.");
  }
  const payload = (await response.json()) as { memo: string };
  return payload.memo;
}

export async function saveMemo(transactionId: string, memo: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/${transactionId}/memo`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memo }),
  });
  if (!response.ok) {
    throw new Error("Failed to save memo.");
  }
}

export async function fetchCorrections(transactionId: string): Promise<CorrectionsPayload | null> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/${transactionId}/corrections`, {
    cache: "no-store",
  });
  // 404 = 修正が存在しない (transaction なし / 旧 API)。それ以外の失敗は
  // 「保存済み修正があるのに見えていない」可能性があるため throw する
  // (黙って認識 baseline に戻すと、次の保存で既存修正を上書きしてしまう)。
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error("Failed to load corrections.");
  }
  const payload = (await response.json()) as { corrections: CorrectionsPayload | null };
  return payload.corrections;
}

export async function saveCorrections(
  transactionId: string,
  corrections: CorrectionsPayload,
): Promise<CorrectionsPayload> {
  const response = await fetch(`${API_BASE_URL}/api/transcriptions/${transactionId}/corrections`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(corrections),
  });
  if (!response.ok) {
    throw new Error("Failed to save corrections.");
  }
  const payload = (await response.json()) as { corrections: CorrectionsPayload };
  return payload.corrections;
}

export async function fetchReviewStatus(
  transactionId: string,
): Promise<ReviewStatusPayload | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/transcriptions/${transactionId}/review-status`,
    { cache: "no-store" },
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error("Failed to load review status.");
  }
  const payload = (await response.json()) as { reviewStatus: ReviewStatusPayload | null };
  return payload.reviewStatus;
}

export async function saveReviewStatus(
  transactionId: string,
  status: ReviewStatusValue,
  options: { note?: string | null; reviewer?: string | null } = {},
): Promise<ReviewStatusPayload> {
  const body: ReviewStatusPayload = {
    version: 1,
    status,
    note: options.note ?? null,
    reviewer: options.reviewer ?? null,
  };
  const response = await fetch(
    `${API_BASE_URL}/api/transcriptions/${transactionId}/review-status`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
  if (!response.ok) {
    throw new Error("Failed to save review status.");
  }
  const payload = (await response.json()) as { reviewStatus: ReviewStatusPayload };
  return payload.reviewStatus;
}

// --- Dev-only (temporary): /debug/triage が使う。ページと一緒に撤去する ---

export type DevTriageRecording = {
  sha16: string;
  primaryTx: string;
  duplicateTxs: string[];
  durationSec: number;
  sampleRate: number;
  peakDbfs: number | null;
  tuningId: string | null;
  storedEvents: number | null;
  freshEvents: number | null;
  recognizedEvents?: string[];
  expectedEvents?: string[] | null;
  warnings: string[];
  memo: string | null;
  reviewStatuses: Record<string, string | null>;
  hasCorrections: boolean;
  gtLayer: string | null;
  score: number;
  signals: string[];
};

export type DevTriageSummary = {
  generatedAt: string;
  recognizerFingerprint: string;
  totals: {
    transactionDirs: number;
    uniqueRecordings: number;
    withGt: number;
    statusCounts: Record<string, number>;
  };
  recordings: DevTriageRecording[];
};

export async function fetchDevTriage(): Promise<DevTriageSummary> {
  const response = await fetch(`${API_BASE_URL}/api/dev/triage`, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? "Failed to load triage summary.");
  }
  return (await response.json()) as DevTriageSummary;
}

// --- Dev-only (temporary): /debug/gt-review が使う。ページと一緒に撤去する ---

export type GtDraftTopCandidate = { note: string; score: number; share: number };

export type GtDraftRowFlag = "ok" | "pitch" | "chord" | "extra" | "ear";

export type GtDraftRow = {
  index: number;
  timeSec: number;
  top: GtDraftTopCandidate[];
  lowEvidence?: boolean;
  recognized: string;
  slot: string;
  expectedIndex: number | null;
  expectedNotes: string[] | null;
  flag: GtDraftRowFlag;
  draftNotes: string[];
  comment: string;
};

export type GtDraftRowVerdict = {
  decision?: "accept" | "fix" | "ignore";
  notes?: string[];
  comment?: string;
  // この onset に主旋律が含まれない (伴奏のみ) — decision と独立
  accompanimentOnly?: boolean;
};

export type GtDraftUnplacedVerdict = {
  decision: "discard" | "place";
  timeSec?: number;
};

export type GtDraftAddedNote = {
  timeSec: number;
  notes: string[];
  comment?: string;
  accompanimentOnly?: boolean;
};

export type GtDraftVerdict = {
  rows: Record<string, GtDraftRowVerdict>;
  unplaced: Record<string, GtDraftUnplacedVerdict>;
  // 認識器が検出しなかった音 (行にも MISS にも無い) のユーザー手動追加
  added?: GtDraftAddedNote[];
  comment?: string;
  done: boolean;
  savedAt?: string;
};

export type GtDraft = {
  txId: string;
  tx8: string;
  durationSec: number;
  sampleRate: number;
  memo: string | null;
  menuId: string | null;
  expectedSource: string;
  expectedCount: number | null;
  generatedAt: string;
  /** 元録音の peak dBFS (正規化前)。試聴ブースト量の根拠 */
  inputPeakDbfs?: number | null;
  tuningNotes: string[];
  rows: GtDraftRow[];
  unplacedExpected: { index: number; notes: string[] }[];
  verdict: GtDraftVerdict | null;
};

export async function fetchGtDrafts(): Promise<GtDraft[]> {
  const response = await fetch(`${API_BASE_URL}/api/dev/gt-drafts`, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? "Failed to load GT drafts.");
  }
  const data = (await response.json()) as { drafts: GtDraft[] };
  return data.drafts;
}

export async function saveGtDraftVerdict(tx8: string, verdict: GtDraftVerdict): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/dev/gt-drafts/${tx8}/verdict`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(verdict),
  });
  if (!response.ok) throw new Error("Failed to save GT draft verdict.");
}

// --- Dev-only (temporary): /debug/bp-verify が使う (#S2 GT 除染)。
// bp-only = GT にあるが recognizer が検出せず、Basic Pitch は検出した note。
// ページと一緒に撤去する ---

export type BpVerifyRow = {
  txId: string;
  tx8: string;
  timeSec: number;
  note: string;
  bandEnergy: number;
  noiseFloor: number;
  energyRatio: number;
  /** 事前仕分けのヒント (即断候補かどうか)。裁定そのものではない */
  likelyAudible: boolean;
};

export type BpVerifyDecision = "real" | "absent" | "unclear";

export type BpVerifyRowVerdict = {
  decision?: BpVerifyDecision;
  comment?: string;
};

export type BpVerifyVerdict = {
  rows: Record<string, BpVerifyRowVerdict>;
  done: boolean;
  savedAt?: string;
};

export type BpVerifyData = {
  generatedAt: string;
  rows: BpVerifyRow[];
  verdict: BpVerifyVerdict | null;
};

export function bpVerifyRowKey(row: Pick<BpVerifyRow, "txId" | "timeSec" | "note">): string {
  return `${row.txId}:${row.timeSec}:${row.note}`;
}

export async function fetchBpVerify(): Promise<BpVerifyData> {
  const response = await fetch(`${API_BASE_URL}/api/dev/bp-verify`, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? "Failed to load bp-verify rows.");
  }
  return (await response.json()) as BpVerifyData;
}

export async function saveBpVerifyVerdict(verdict: BpVerifyVerdict): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/dev/bp-verify/verdict`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(verdict),
  });
  if (!response.ok) throw new Error("Failed to save bp-verify verdict.");
}

export async function fetchReviewQueue(
  options: { limit?: number; status?: ReviewStatusValue | null } = {},
): Promise<ReviewQueueEntry[]> {
  const params = new URLSearchParams();
  if (options.limit) params.set("limit", String(options.limit));
  if (options.status) params.set("status", options.status);
  const query = params.toString();
  const response = await fetch(
    `${API_BASE_URL}/api/review-queue${query ? `?${query}` : ""}`,
    { cache: "no-store" },
  );
  if (!response.ok) {
    throw new Error("Failed to load review queue.");
  }
  return response.json();
}

function cleanOptionalText(value: string | undefined): string | null {
  const trimmed = value?.trim() ?? "";
  return trimmed.length > 0 ? trimmed : null;
}

async function parseError(response: Response) {
  try {
    const payload = await response.json();
    return payload.detail ?? "Transcription failed.";
  } catch {
    return "Transcription failed.";
  }
}
