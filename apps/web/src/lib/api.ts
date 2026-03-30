import { InstrumentTuning, TranscriptionResult } from "@/lib/types";
import { WavMetadata, toWavWithMetadata } from "@/lib/audio";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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
  source: "clickable-kalimba-ui";
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
};

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
